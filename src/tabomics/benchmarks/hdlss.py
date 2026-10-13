# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""HDLSS suite: 15 high-dimensional, low-sample-size datasets of scikit-feature, as in TabPFN-Wide.

Prepared exactly as TabPFN-Wide does (analysis/hdlss_benchmark.py), so the folds are theirs: rows shuffled with
seed 42, then stratified 3-fold repeated 10 times (3 times for datasets of 2500 rows or more).

Data: the .mat files of https://jundongl.github.io/scikit-feature/datasets.html, at <data_root>/HDLSS/<name>.mat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse
from scipy.io import loadmat
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

FOLDER = "HDLSS"
DATASETS = ("ALLAML", "BASEHOCK", "CLL_SUB_111", "GLI_85", "PCMAC", "Prostate_GE", "RELATHE", "SMK_CAN_187",
            "TOX_171", "arcene", "colon", "gisette", "leukemia", "lung", "madelon")  # fmt: skip
N_SPLITS, SEED, LARGE = 3, 42, 2500


def tasks(data_root: str | Path) -> list[str]:
    """The datasets of the suite."""
    return list(DATASETS)


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features (float32, rows shuffled with seed 42), labels 0..C-1, and the categorical columns (none)."""
    data = loadmat(Path(data_root) / FOLDER / f"{name}.mat")
    X, y = data["X"], np.asarray(data["Y"]).ravel()
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X, y = shuffle(X, y, random_state=SEED)
    return np.asarray(X, dtype=np.float32), LabelEncoder().fit_transform(y), []


def folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified 3-fold, 10 repeats (3 with 2500 rows or more), seed 42."""
    repeats = 10 if len(y) < LARGE else 3
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=repeats, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))
