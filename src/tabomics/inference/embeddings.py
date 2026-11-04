# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Embeddings and feature importance of the training rows, comparable with those of the test rows."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from tabomics.inference.base import TabOmicsEstimator
from tabomics.inference.classifier import Classifier


def _rows(data, index: np.ndarray):
    """Selects rows keeping the container type (.iloc for pandas, so each column keeps its dtype)."""
    return data.iloc[index] if hasattr(data, "iloc") else np.asarray(data)[index]


def _out_of_fold(estimator: TabOmicsEstimator, X_train, y_train, n_folds, shuffle, random_state, score) -> np.ndarray:
    """Runs `score(estimator, X_query)` on every fold of the training rows, fitted on the other folds.

    Classifiers split with stratification. The pieces are put back in the order of X_train, and the estimator is
    fitted on all the training rows at the end.
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
        pieces.append(score(estimator, _rows(X_train, query)))
        positions.append(query)
    result = np.concatenate(pieces)[np.argsort(np.concatenate(positions))]
    estimator.fit(X_train, y_train)
    return result


def out_of_fold_embeddings(
    estimator: TabOmicsEstimator,
    X_train,
    y_train,
    n_folds: int = 10,
    shuffle: bool = False,
    random_state: int | None = None,
    reduce: str = "mean",
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
        reduce: With several members, "mean" averages them and "concat" puts them side by side (see embeddings).

    Returns:
        Shape (training rows, embedding size), or (training rows, n_estimators x embedding size) with "concat", in
        the order of X_train.
    """
    if reduce not in ("mean", "concat"):
        raise ValueError(f"reduce must be 'mean' or 'concat', got {reduce!r}")
    return _out_of_fold(
        estimator, X_train, y_train, n_folds, shuffle, random_state, lambda est, X: est.embeddings(X, reduce=reduce)
    )


def out_of_fold_feature_importance(
    estimator: TabOmicsEstimator,
    X_train,
    y_train,
    n_folds: int = 10,
    shuffle: bool = False,
    random_state: int | None = None,
    per_row: bool = False,
) -> np.ndarray:
    """Feature importance of every training row as a test row, out of fold.

    As in out_of_fold_embeddings, the training rows are split into folds, stratified for classifiers, and each fold
    is scored by feature_importance as test rows with the other folds as context. The estimator is fitted on all the
    training rows at the end, ready to use.

    A column constant in the context of a fold is dropped by its preprocessing and scores NaN for the rows of that
    fold. The mean over the rows ignores those values, and a column stays NaN only if it is constant in every fold.

    Args:
        estimator: A Classifier or Regressor.
        X_train: Training rows (array or DataFrame).
        y_train: Their targets.
        n_folds: Number of folds, at least 2.
        shuffle: Shuffle the rows before splitting.
        random_state: Seed of the shuffle.
        per_row: One score per training row instead of their mean.

    Returns:
        One score per original column, shape (n_features_in_,), the mean over the training rows; with per_row,
        shape (training rows, n_features_in_), in the order of X_train.
    """
    scores = _out_of_fold(
        estimator, X_train, y_train, n_folds, shuffle, random_state,
        lambda est, X: est.feature_importance(X, per_row=True),
    )  # fmt: skip
    if per_row:
        return scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Mean of empty slice": a column constant in every fold
        return np.nanmean(scores, axis=0)
