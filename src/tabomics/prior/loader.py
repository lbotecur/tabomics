# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from TFM-Playground's TabICLPriorDataLoader (Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-TFM-Playground.txt). Modified: TabOmics' graph prior only,
# generation in parallel workers, fixed seeds per epoch and worker, categorical masks.
"""Batches of synthetic tables for training, generated on the CPU in parallel worker processes.

Generating the tables, not the GPU, limits training speed, so the loader runs the prior in `num_workers` processes.
With a seed, the tables of every epoch are fixed: each (epoch, worker) seeds the generators the prior draws from
(numpy's, torch's and random's), and the workers' batches arrive in a fixed order, so the same seed and number of
workers give the same tables, also after resuming a run. Generation also depends on Python's hash seed: start the
process with PYTHONHASHSEED set (the training script does it), or the tables change from run to run.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from tabomics.prior.graph import GraphPrior, PriorConfig


def seed_generation(seed: int, epoch: int, worker: int) -> None:
    """Seeds the generators the prior draws from for one worker in one epoch."""
    value = (seed * 1_000_003 + epoch * 10_007 + worker) % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def usable_batch(batch) -> dict | None:
    """The usable tables of a prior batch, cropped to the widest one, or None if there are none.

    Unusable tables (no column left after dropping the constant ones, or a failed generation, labelled -100) would
    crash or corrupt training. Tables in a batch can keep different numbers of columns; the batch keeps the widest
    (the others are zero padded).

    Args:
        batch: A PriorBatch.

    Returns:
        {"x": (B, T, H) float, "y": (B, T) float, "train_size": int, "categorical": (B, H) bool}, on the CPU.
    """
    keep = (batch.num_features > 0) & (batch.y.reshape(batch.y.shape[0], -1).min(dim=1).values >= 0)
    if not keep.any():
        return None
    width = int(batch.num_features[keep].max())
    return {
        "x": batch.X[keep][:, :, :width],
        "y": batch.y[keep].float(),  # the model reads class indices as numbers
        "train_size": int(batch.train_sizes[keep][0]),
        "categorical": batch.categorical[keep][:, :width],
    }


class _Batches(IterableDataset):
    """Usable batches for one epoch; in a DataLoader, each worker yields its share of the steps."""

    def __init__(self, loader: PriorLoader):
        self.loader = loader

    def __iter__(self):
        worker = get_worker_info()
        workers, index = (worker.num_workers, worker.id) if worker else (1, 0)
        steps = self.loader.num_steps // workers + (index < self.loader.num_steps % workers)
        if worker is not None:
            torch.set_num_threads(1)  # one core per worker: the workers already use the cores given
        if self.loader.seed is not None:
            seed_generation(self.loader.seed, self.loader.epoch, index)
        prior = self.loader.make_prior()
        for _ in range(steps):
            batch = None
            while batch is None:
                batch = usable_batch(prior.get_batch())
            yield batch


class PriorLoader:
    """Iterable over the batches of one epoch (iterate again for the next epoch).

    Args:
        num_steps: Batches per epoch.
        batch_size: Tables per batch.
        min_rows: Fewest rows of a table.
        max_rows: Most rows of a table (exclusive).
        min_features: Fewest columns.
        max_features: Most columns.
        max_classes: Most classes.
        raw: Tables in the form of inference tables (see tabomics.prior.graph.GraphSCM); False gives TabICL's.
        log_seq_len: Draw the rows log-uniformly.
        min_train_size: Smallest share of training rows.
        max_train_size: Largest share of training rows.
        batch_size_per_gp: Tables sharing their size and hyperparameter draws; the whole batch if None.
        config: Settings of the graph prior (PriorConfig()).
        num_workers: Generating processes (0 = in this process).
        seed: Makes the tables of every epoch fixed; None = random.
        epoch: Epochs already done (when resuming a run, so that it continues the sequence).
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        min_rows: int,
        max_rows: int,
        min_features: int,
        max_features: int,
        max_classes: int,
        *,
        raw: bool = True,
        log_seq_len: bool = False,
        min_train_size: float = 0.1,
        max_train_size: float = 0.9,
        batch_size_per_gp: int | None = None,
        config: PriorConfig | None = None,
        num_workers: int = 0,
        seed: int | None = None,
        epoch: int = 0,
    ):
        self.num_steps = num_steps
        self.prior_settings = dict(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp or batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_rows,
            max_seq_len=max_rows,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            config=config,
            raw=raw,
        )
        self.num_workers = num_workers
        self.seed = seed
        self.epoch = epoch  # incremented when each epoch starts
        if seed is not None and os.environ.get("PYTHONHASHSEED") in (None, "random"):
            warnings.warn(
                "PYTHONHASHSEED is not set: with a seed the tables are still not reproducible across runs "
                "(the prior iterates over a set of strings). Start Python with PYTHONHASHSEED set.",
                stacklevel=2,
            )
        self._dataloader = None
        if num_workers > 0:
            # with a seed the workers start again every epoch, so that each one gets that epoch's seed
            self._dataloader = DataLoader(
                _Batches(self), batch_size=None, num_workers=num_workers, persistent_workers=seed is None,
                prefetch_factor=4,
            )  # fmt: skip

    def make_prior(self) -> GraphPrior:
        """The graph prior that generates the tables (one per process)."""
        return GraphPrior(**self.prior_settings)

    def __iter__(self):
        if self.seed is not None:
            self.epoch += 1
        return iter(self._dataloader) if self._dataloader is not None else iter(_Batches(self))

    def __len__(self) -> int:
        return self.num_steps
