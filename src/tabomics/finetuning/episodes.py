# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Episodes from real tables: the batches of fine-tuning.

Pretraining draws a new synthetic table at every step; fine-tuning has a few real tables, so every step draws a new
random version of one of them (an episode): a subset of its rows split at random into context and query (every
class in the context), a subset of its columns in random order, and its class indices permuted. The rows of a table
thus take turns being context and query, and the model cannot learn one fixed split. A batch holds `batch_size`
episodes of one table, all of the same shape; consecutive batches take the tables in turn.

Each table is first split into a pool, from which the episodes are drawn, and a validation set, never part of an
episode, used to decide when to stop. The preprocessing (categorical codes, constant columns) is fitted on the pool
and applied to the validation rows, as inference fits it on the training rows and applies it to the test rows.
Standardization is the model's own, from the context rows of each episode, as at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabomics.inference.preprocessing import FeaturePreprocessor


@dataclass
class FinetuneTable:
    """One table ready for fine-tuning: model inputs of its pool and validation rows, labels 0..C-1.

    Attributes:
        X: Pool rows (episodes are drawn from them), shape (rows, columns), NaN for missing values.
        y: Their labels.
        X_val: Validation rows, never in an episode (may be empty).
        y_val: Their labels.
        categorical: Which model input columns are categorical, shape (columns,).
        num_classes: Classes of the table.
    """

    X: np.ndarray
    y: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    categorical: np.ndarray
    num_classes: int


def prepare_table(
    X,
    y,
    validation_fraction: float = 0.1,
    X_val=None,
    y_val=None,
    random_state: int = 0,
    **preprocessing,
) -> FinetuneTable:
    """Splits a table into pool and validation rows (stratified) and preprocesses it as inference does.

    Args:
        X: Training rows of the table (array or DataFrame); never its test rows.
        y: Their labels (any values).
        validation_fraction: Share of the rows held out for validation when X_val is not given; 0 = none.
        X_val: Validation rows given by the user (then nothing is held out from X).
        y_val: Their labels.
        random_state: Seed of the split.
        **preprocessing: Settings of FeaturePreprocessor (categorical_features, infer_categorical, ...).

    Returns:
        The prepared table.
    """
    encoder = LabelEncoder().fit(np.concatenate([np.asarray(y), np.asarray(y_val)]) if y_val is not None else y)
    y = encoder.transform(np.asarray(y))
    if X_val is None and validation_fraction > 0:
        counts = np.bincount(y)
        stratify = y if counts.min() >= 2 else None
        X, X_val, y, y_val = train_test_split(
            X, y, test_size=validation_fraction, stratify=stratify, random_state=random_state
        )
    elif y_val is not None:
        y_val = encoder.transform(np.asarray(y_val))
    preprocessor = FeaturePreprocessor(**preprocessing).fit(X)
    X_pool = preprocessor.transform(X).astype(np.float32)
    if X_val is not None and len(X_val):
        X_val = preprocessor.transform(X_val).astype(np.float32)
    else:
        X_val, y_val = np.empty((0, X_pool.shape[1]), np.float32), np.empty(0, np.int64)
    return FinetuneTable(
        X=X_pool,
        y=np.asarray(y, dtype=np.int64),
        X_val=X_val,
        y_val=np.asarray(y_val, dtype=np.int64),
        categorical=preprocessor.category_counts_ > 0,
        num_classes=len(encoder.classes_),
    )


def split_context_query(y: np.ndarray, num_query: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random context and query positions of an episode's rows, with every class in the context.

    Args:
        y: Labels of the episode's rows.
        num_query: Rows to put in the query.
        rng: Random generator.

    Returns:
        Positions of the context rows and of the query rows.
    """
    order = rng.permutation(len(y))
    _, first = np.unique(y[order], return_index=True)  # one row of each class, kept in the context
    reserved = np.zeros(len(y), dtype=bool)
    reserved[first] = True
    candidates = order[~reserved]
    query = candidates[:num_query]
    context = np.concatenate([order[reserved], candidates[num_query:]])
    return rng.permutation(context), query


class EpisodeLoader:
    """Batches of episodes of real tables, in the format of PriorLoader, for tabomics.training.train.

    Args:
        tables: The prepared tables.
        num_steps: Batches per epoch.
        batch_size: Episodes per batch (all of one table, of the same shape).
        query_fraction: Share of an episode's rows to predict (the rest is context).
        row_fraction: Range of the share of a table's pool rows in a batch's episodes.
        feature_fraction: Range of the share of a table's columns in a batch's episodes.
        max_features: Most columns in an episode (memory); None = no limit.
        seed: Seed of the episodes (the same seed gives the same batches).
    """

    def __init__(
        self,
        tables: list[FinetuneTable],
        num_steps: int,
        batch_size: int = 8,
        query_fraction: float = 0.2,
        row_fraction: tuple[float, float] = (0.8, 1.0),
        feature_fraction: tuple[float, float] = (0.5, 1.0),
        max_features: int | None = None,
        seed: int = 0,
    ):
        if not tables:
            raise ValueError("No tables to fine-tune on.")
        for table in tables:
            if len(table.y) < table.num_classes + 1:
                raise ValueError("A table needs more pool rows than classes (a context row per class and a query).")
        self.tables = tables
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.query_fraction = query_fraction
        self.row_fraction = row_fraction
        self.feature_fraction = feature_fraction
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.step = 0  # batches drawn so far: the table of a batch is step modulo the number of tables

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self):
        for _ in range(self.num_steps):
            yield self.next_batch()

    def next_batch(self) -> dict:
        """The next batch: episodes of the next table in turn."""
        table = self.tables[self.step % len(self.tables)]
        self.step += 1
        return self.batch(table)

    def batch(self, table: FinetuneTable) -> dict:
        """`batch_size` episodes of one table, of the same shape."""
        rng = self.rng
        n_pool, n_columns = table.X.shape
        rows = max(table.num_classes + 1, round(rng.uniform(*self.row_fraction) * n_pool))
        rows = min(rows, n_pool)
        columns = max(1, round(rng.uniform(*self.feature_fraction) * n_columns))
        if self.max_features is not None:
            columns = min(columns, self.max_features)
        num_query = min(max(1, round(self.query_fraction * rows)), rows - table.num_classes)
        xs, ys, masks = [], [], []
        for _ in range(self.batch_size):
            chosen = rng.choice(n_pool, rows, replace=False)
            context, query = split_context_query(table.y[chosen], num_query, rng)
            order = chosen[np.concatenate([context, query])]
            cols = rng.choice(n_columns, columns, replace=False)  # a subset, in random order
            relabel = rng.permutation(table.num_classes)  # class indices permuted
            xs.append(table.X[order][:, cols])
            ys.append(relabel[table.y[order]])
            masks.append(table.categorical[cols])
        return {
            "x": torch.from_numpy(np.stack(xs)),
            "y": torch.from_numpy(np.stack(ys)).float(),
            "train_size": rows - num_query,
            "categorical": torch.from_numpy(np.stack(masks)),
        }
