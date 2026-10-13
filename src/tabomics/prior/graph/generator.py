# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TabICL 2.2.0, tabicl/prior/_dataset.py, classes Prior and GraphPrior
# (https://github.com/soda-inria/tabicl), Copyright (c) 2025, Soda team @ Inria; see
# LICENSES/BSD-3-Clause-TabICL.txt. Changes: see CHANGES.md.
"""Batches of synthetic tables from the graph prior.

A batch is split into groups of tables that share their number of rows and train size; within a group, subgroups
share their number of columns; each table has its own number of classes and its own random graph. A table is
generated again until it is usable: some column left after dropping the constant ones, the same classes (at least
two) among the training and the test rows, and, if the configuration asks for it, a signal an ExtraTrees can
find. Random draws happen in the order of TabICL's GraphPrior (with n_jobs=1), so with the same seeds and
`raw=False` the batches are TabICL's, bit for bit.

Generation also depends on Python's hash seed (TabICL iterates over a set of strings), so reproducing a batch
needs the same PYTHONHASHSEED as well as the same seeds of numpy, torch and random.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import threadpoolctl
import torch
import torch.nn.functional as F
from scipy.stats import loguniform

from tabomics.prior.graph._config import PriorConfig
from tabomics.prior.graph._filter import should_filter
from tabomics.prior.graph.scm import GraphSCM


class PriorBatch(NamedTuple):
    """A batch of synthetic tables, all padded to the same width.

    Attributes:
        X: Features, shape (B, T, H); the columns past `num_features` are zero padding.
        y: Targets, shape (B, T): class indices (int64) or numbers.
        num_features: Columns of each table after dropping the constant ones, shape (B,).
        seq_lens: Rows of each table, shape (B,).
        train_sizes: Training rows of each table (the first ones), shape (B,).
        categorical: Which columns are categorical, shape (B, H) (False on the padding).
    """

    X: torch.Tensor
    y: torch.Tensor
    num_features: torch.Tensor
    seq_lens: torch.Tensor
    train_sizes: torch.Tensor
    categorical: torch.Tensor


def sample_seq_len(min_seq_len: int | None, max_seq_len: int, log: bool = False) -> int:
    """Number of rows: max_seq_len if min_seq_len is None, else uniform (or log-uniform) in [min, max)."""
    if min_seq_len is None:
        return max_seq_len
    if log:
        return int(loguniform.rvs(min_seq_len, max_seq_len))
    return np.random.randint(min_seq_len, max_seq_len)


def sample_train_size(min_train_size: int | float, max_train_size: int | float, seq_len: int) -> int:
    """Number of training rows: an int in [min, max), or a fraction in [min, max] of seq_len."""
    if isinstance(min_train_size, int) and isinstance(max_train_size, int):
        return np.random.randint(min_train_size, max_train_size)
    if isinstance(min_train_size, float) and isinstance(max_train_size, float):
        return int(seq_len * np.random.uniform(min_train_size, max_train_size))
    raise ValueError("Both training sizes must be of the same type (int or float)")


def check_train_size_range(min_train_size: int | float, max_train_size: int | float) -> None:
    """Raises if the range of training sizes is not valid."""
    if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
        raise TypeError("Training sizes must be int or float")
    if isinstance(min_train_size, int) and isinstance(max_train_size, int):
        if not 0 < min_train_size < max_train_size:
            raise ValueError("Integer training sizes need 0 < min_train_size < max_train_size")
    elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
        if not 0 < min_train_size <= max_train_size < 1:
            raise ValueError("Fractional training sizes need 0 < min_train_size <= max_train_size < 1")
    else:
        raise ValueError("Both training sizes must be of the same type (int or float)")


def drop_constant_columns(
    X: torch.Tensor, num_features: int, categorical: torch.Tensor
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Removes the columns with a single value, keeping the table's width with zero padding.

    Args:
        X: One table, shape (T, H).
        num_features: Its real columns (the rest is padding).
        categorical: Its categorical mask, shape (H,).

    Returns:
        The table, its number of columns left and its mask, with the constant columns removed.
    """
    keep = torch.tensor([len(torch.unique(X[:, j])) > 1 for j in range(num_features)], dtype=torch.bool)
    width, kept = X.shape[-1], int(keep.sum())
    X = F.pad(X[:, :num_features][:, keep], (0, width - kept), mode="constant", value=0)
    categorical = F.pad(categorical[:num_features][keep.to(categorical.device)], (0, width - kept), value=False)
    return X, kept, categorical


def valid_split(y: torch.Tensor, train_size: int, min_classes: int = 2) -> bool:
    """Whether the training and the test rows hold the same classes, at least min_classes of them."""
    if train_size <= 0 or train_size >= y.shape[0]:
        return False
    train_classes = set(torch.unique(y[:train_size]).tolist())
    return train_classes == set(torch.unique(y[train_size:]).tolist()) and len(train_classes) >= min_classes


def fix_split(X: torch.Tensor, y: torch.Tensor, train_size: int, attempts: int = 10) -> tuple | None:
    """The table as it is if its split is valid, else with its rows shuffled until it is (None if it never is)."""
    if valid_split(y, train_size):
        return X, y
    for _ in range(attempts):
        permutation = torch.randperm(y.shape[0])
        if valid_split(y[permutation], train_size):
            return X[permutation], y[permutation]
    return None


class GraphPrior:
    """Generates batches of tables from the graph prior.

    Args:
        regression: Numeric targets instead of classes.
        batch_size: Tables per batch.
        batch_size_per_gp: Tables per group (shared rows and train size; the whole batch shares them unless
            per-group lengths are added later).
        batch_size_per_subgp: Tables per subgroup (shared number of columns); batch_size_per_gp if None.
        min_features: Fewest columns of a table.
        max_features: Most columns of a table; every table is padded to this width.
        max_classes: Most classes of a table (each table draws 2..max_classes).
        min_seq_len: Fewest rows; if None every table has max_seq_len rows.
        max_seq_len: Most rows (exclusive when sampled).
        log_seq_len: Sample the rows log-uniformly.
        log_n_features: Sample the columns log-uniformly.
        min_train_size: Fewest training rows: an int, or a fraction of the rows.
        max_train_size: Most training rows, of the same type.
        config: Settings of the graph prior (PriorConfig()).
        device: Device of the returned tensors (tables are generated on it too).
        raw: Tables without TabICL's clamp and z-score (see GraphSCM).
    """

    def __init__(
        self,
        regression: bool = False,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: int | None = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: int | None = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        log_n_features: bool = False,
        min_train_size: int | float = 0.1,
        max_train_size: int | float = 0.9,
        config: PriorConfig | None = None,
        device: str = "cpu",
        raw: bool = True,
    ):
        if min_features > max_features:
            raise ValueError("min_features must not exceed max_features")
        check_train_size_range(min_train_size, max_train_size)
        self.regression = regression
        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = 0 if regression else max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.log_n_features = log_n_features
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.config = config or PriorConfig()
        self.device = device
        self.raw = raw

    def table_parameters(self, batch_size: int) -> list[dict]:
        """Draws the size of every table of a batch: rows, train size, columns and classes."""
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)
        seq_len = sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
        train_size = sample_train_size(self.min_train_size, self.max_train_size, seq_len)

        parameters = []
        for group in range(math.ceil(batch_size / size_per_gp)):
            group_size = min(size_per_gp, batch_size - group * size_per_gp)
            for subgroup in range(math.ceil(group_size / size_per_subgp)):
                subgroup_size = min(size_per_subgp, group_size - subgroup * size_per_subgp)
                if self.log_n_features:
                    num_features = int(loguniform.rvs(self.min_features, self.max_features))
                else:
                    num_features = round(np.random.uniform(self.min_features, self.max_features))
                for _ in range(subgroup_size):
                    num_classes = None if self.regression else np.random.randint(2, self.max_classes + 1)
                    parameters.append(
                        {
                            "regression": self.regression,
                            "device": self.device,
                            "config": self.config,
                            "seq_len": seq_len,
                            "train_size": train_size,
                            "max_features": self.max_features,
                            "num_features": num_features,
                            "num_classes": num_classes,
                        }
                    )
        return parameters

    @torch.no_grad()
    def generate_table(self, parameters: dict) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Generates tables with these parameters until one is usable; returns X, y, its columns and its mask."""
        while True:
            X, y, categorical = GraphSCM(**parameters, raw=self.raw)()
            X, num_features, categorical = drop_constant_columns(X, parameters["num_features"], categorical)
            if num_features == 0:
                continue
            if not self.regression:
                fixed = fix_split(X, y, parameters["train_size"])
                if fixed is None:
                    continue
                X, y = fixed
            if should_filter(X, y, self.config, is_classif=not self.regression):
                continue
            return X, y, num_features, categorical

    @torch.no_grad()
    def get_batch(self, batch_size: int | None = None) -> PriorBatch:
        """Generates a batch of tables (one CPU thread, as TabICL's generation with n_jobs=1)."""
        parameters = self.table_parameters(batch_size or self.batch_size)
        with threadpoolctl.threadpool_limits(limits=1):
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                tables = [self.generate_table(p) for p in parameters]
            finally:
                torch.set_num_threads(threads)
        X, y, num_features, categorical = zip(*tables, strict=True)
        return PriorBatch(
            X=torch.stack(X).to(self.device),
            y=torch.stack(y).to(self.device),
            num_features=torch.tensor(num_features, device=self.device, dtype=torch.long),
            seq_lens=torch.tensor([p["seq_len"] for p in parameters], device=self.device, dtype=torch.long),
            train_sizes=torch.tensor([p["train_size"] for p in parameters], device=self.device, dtype=torch.long),
            categorical=torch.stack(categorical).to(self.device),
        )

    def __iter__(self):
        return self

    def __next__(self) -> PriorBatch:
        return self.get_batch()
