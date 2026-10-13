# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Embeddings of the training rows, comparable with those of the test rows."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from tabomics.inference.base import TabOmicsEstimator
from tabomics.inference.classifier import Classifier


def _rows(data, index: np.ndarray):
    """Selects rows keeping the container type (.iloc for pandas, so each column keeps its dtype)."""
    return data.iloc[index] if hasattr(data, "iloc") else np.asarray(data)[index]


def out_of_fold_embeddings(
    estimator: TabOmicsEstimator,
    X_train,
    y_train,
    n_folds: int = 10,
    shuffle: bool = False,
    random_state: int | None = None,
) -> np.ndarray:
    """Embeds every training row as a test row, out of fold (Ye et al., 2025).

    A training row embedded directly is seen with its own label, unlike a test row. Here the training rows are
    split into folds, and each fold is embedded as test rows with the other folds as context. Classifiers split
    with stratification. The estimator is fitted on all the training rows at the end, ready to use.

    Args:
        estimator: A Classifier or Regressor.
        X_train: Training rows (array or DataFrame).
        y_train: Their targets.
        n_folds: Number of folds, at least 2.
        shuffle: Shuffle the rows before splitting.
        random_state: Seed of the shuffle.

    Returns:
        Shape (training rows, embedding size), in the order of X_train.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    seed = random_state if shuffle else None
    if isinstance(estimator, Classifier):
        splits = StratifiedKFold(n_folds, shuffle=shuffle, random_state=seed).split(np.zeros(len(y_train)), y_train)
    else:
        splits = KFold(n_folds, shuffle=shuffle, random_state=seed).split(np.zeros(len(y_train)))
    pieces, positions = [], []
    for context, query in splits:
        estimator.fit(_rows(X_train, context), _rows(y_train, context))
        pieces.append(estimator.embeddings(_rows(X_train, query)))
        positions.append(query)
    embeddings = np.concatenate(pieces)[np.argsort(np.concatenate(positions))]
    estimator.fit(X_train, y_train)
    return embeddings
