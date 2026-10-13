# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""CC18 suite: the narrow tables of OpenML-CC18 (Bischl et al., 2021; OpenML suite 99).

At most 10,000 rows, 500 features and 10 classes; madelon is left out (a development dataset of TabOmics).
Stratified 10-fold, seed 42.

Data: written by scripts/download_cc18.py to <data_root>/OpenML-CC18/<name>.npz (features as float32, categorical
columns as codes, NaN for missing values; labels 0..C-1; indices of the categorical columns), listed in index.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

FOLDER = "OpenML-CC18"
SUITE_ID = 99
MAX_ROWS, MAX_FEATURES, MAX_CLASSES = 10_000, 500, 10
EXCLUDED = ("madelon",)
N_SPLITS, SEED = 10, 42


def keep(name: str, rows: int, features: int, classes: int) -> bool:
    """Whether a CC18 dataset belongs to the suite (features without the target)."""
    return name not in EXCLUDED and rows <= MAX_ROWS and features <= MAX_FEATURES and 2 <= classes <= MAX_CLASSES


def to_arrays(X: pd.DataFrame, y, categorical: list[bool]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """OpenML data to arrays.

    Returns:
        Features as float32 (categorical columns as codes, missing values as NaN), labels 0..C-1, and the indices
        of the categorical columns.
    """
    columns, categorical_columns = [], []
    for j, name in enumerate(X.columns):
        values = X[name]
        if categorical[j] or not pd.api.types.is_numeric_dtype(values):
            codes = pd.Categorical(values).codes.astype(np.float32)
            columns.append(np.where(codes < 0, np.nan, codes))  # code -1 means missing
            categorical_columns.append(j)
        else:
            columns.append(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32))
    X = np.stack(columns, axis=1).astype(np.float32)
    return X, LabelEncoder().fit_transform(np.asarray(y)), categorical_columns


def save_task(path: str | Path, X: np.ndarray, y: np.ndarray, categorical: list[int]) -> None:
    """Writes one dataset as .npz."""
    np.savez_compressed(path, X=X, y=y, categorical=np.asarray(categorical, dtype=np.int64))


def tasks(data_root: str | Path) -> list[str]:
    """The downloaded datasets, from index.csv."""
    return pd.read_csv(Path(data_root) / FOLDER / "index.csv").name.tolist()


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features, labels 0..C-1, and the categorical columns."""
    data = np.load(Path(data_root) / FOLDER / f"{name}.npz")
    return data["X"].astype(np.float32), data["y"].astype(np.int64), data["categorical"].tolist()


def folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified 10-fold, seed 42."""
    return list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
