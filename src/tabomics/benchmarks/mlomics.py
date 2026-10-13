# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""MLOmics suite: TCGA cancer-subtype classification (BRCA, COAD, GBM, LGG, OV), as in TabPFN-Wide.

Prepared exactly as TabPFN-Wide does (analysis/load_mm_data.py and analysis/data.py), so the folds are theirs:
all-zero and duplicated features dropped, missing values filled with the feature mean, stratified 5-fold, seed 42.
A task is a cancer and one omic or several concatenated: "BRCA/mrna", "BRCA/mrna+cnv+methylation+mirna".

Data: Main_Dataset of MLOmics (https://huggingface.co/datasets/AIBIC/MLOmics; scripts/download_mlomics.py), at
<data_root>/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets/GS-<CANCER>/Original/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

FOLDER = "Cancer-Multi-Omics-Benchmark"
CANCERS = ("BRCA", "COAD", "GBM", "LGG", "OV")
OMIC_FILES = {"mrna": "mRNA", "cnv": "CNV", "methylation": "Methy", "mirna": "miRNA"}
ALL_OMICS = "+".join(OMIC_FILES)
N_SPLITS, SEED = 5, 42


def tasks(data_root: str | Path) -> list[str]:
    """Each cancer with each omic alone and with all four together."""
    return [f"{cancer}/{omics}" for cancer in CANCERS for omics in (*OMIC_FILES, ALL_OMICS)]


def _load_omic(path: Path) -> pd.DataFrame:
    """One omic file (features x samples) -> samples x features, cleaned as TabPFN-Wide does."""
    data = pd.read_csv(path, index_col=0)
    data = data.loc[~(data == 0).all(axis=1)]  # all-zero features
    data = data[~data.index.duplicated(keep="first")]  # duplicated features
    data = data.T
    data = data.fillna(data.mean())
    data.columns = data.columns.astype(str)
    return data


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features (float32), labels 0..C-1, and the categorical columns (none: copy-number levels are ordinal)."""
    cancer, _, omics = name.partition("/")
    if cancer not in CANCERS:
        raise ValueError(f"{cancer} is not one of {CANCERS}")
    folder = Path(data_root) / FOLDER / "Main_Dataset" / "Classification_datasets" / f"GS-{cancer}" / "Original"
    parts = [_load_omic(folder / f"{cancer}_{OMIC_FILES[o]}.csv") for o in (omics or "mrna").split("+")]
    labels = pd.read_csv(folder / f"{cancer}_label_num.csv").values.flatten()  # same row order as the omics
    X = pd.concat(parts, axis=1)
    return X.to_numpy(dtype=np.float32), LabelEncoder().fit_transform(labels), []


def folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified 5-fold, one repeat, seed 42."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))
