# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Metrics, computed as TabPFN-Wide does (analysis/multiomics_feature_reduction.py)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """AUROC. Binary: of the probability of class 1. Multiclass: one-vs-rest, macro average.

    NaN when it is undefined (a test fold that lacks a class).
    """
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        labels = np.arange(proba.shape[1])
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        return float("nan")


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Accuracy of the most probable class."""
    return float(accuracy_score(y_true, proba.argmax(axis=1)))
