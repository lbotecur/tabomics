# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Shamir suite: TCGA multi-omics benchmark of Rappoport & Shamir (2018), with subtype labels as in TabPFN-Wide.

Breast (PAM50), gbm (expression subtype) and sarcoma (histological type); colon is left out (its labels are not
in the public download) and kidney too (a single subtype). Prepared as TabPFN-Wide does: each omic standardized
over all samples, duplicated samples dropped, only patients with all three omics and a label. Two differences:
classes with fewer patients than folds (5) are left out, and samples are sorted (TabPFN-Wide intersects them
through a Python set, so its order and folds change from run to run). Stratified 5-fold, seed 42.
A task is a cancer and one omic or several: "breast/mrna", "breast/mrna+methylation+mirna".

Data: https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html, each zip unpacked in its own folder:
<data_root>/Shamir/<cancer>/{exp,methy,mirna} and <data_root>/Shamir/clinical/clinical/<cancer>.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, scale

FOLDER = "Shamir"
LABEL_COLUMNS = {"breast": "PAM50Call_RNAseq", "gbm": "GeneExp_Subtype", "sarcoma": "histological_type"}
OMIC_FILES = {"mrna": "exp", "methylation": "methy", "mirna": "mirna"}
ALL_OMICS = "+".join(OMIC_FILES)
N_SPLITS, SEED = 5, 42


def tasks(data_root: str | Path) -> list[str]:
    """Each cancer with each omic alone and with all three together."""
    return [f"{cancer}/{omics}" for cancer in LABEL_COLUMNS for omics in (*OMIC_FILES, ALL_OMICS)]


def _key(sample: str) -> str:
    """'TCGA.3C.AAAU.01' or 'TCGA-3C-AAAU-01' -> 'tcga.3c.aaau.01', to match omics and clinical samples."""
    return sample.replace("-", ".").lower()


def _load_omic(path: Path) -> pd.DataFrame:
    """One omic file (features x samples, whitespace separated) -> samples x features, standardized."""
    data = pd.read_table(path, header=0, index_col=0, sep=r"\s+").T
    data = data[~data.index.duplicated(keep="first")]
    if data.isna().any().any():
        raise ValueError(f"missing values in {path}")
    data[:] = scale(data)
    data.index = data.index.map(_key)
    return data


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features (float32, patients sorted by sample ID), labels 0..C-1, and the categorical columns (none)."""
    cancer, _, omics = name.partition("/")
    if cancer not in LABEL_COLUMNS:
        raise ValueError(f"{cancer} is not one of {tuple(LABEL_COLUMNS)}")
    folder = Path(data_root) / FOLDER
    data = {omic: _load_omic(folder / cancer / file) for omic, file in OMIC_FILES.items()}
    clinical = pd.read_table(folder / "clinical" / "clinical" / cancer)
    labels = clinical.set_index(clinical["sampleID"].map(_key))[LABEL_COLUMNS[cancer]].dropna()
    labels = labels[~labels.index.duplicated(keep="first")]
    samples = sorted(set(labels.index).intersection(*(set(d.index) for d in data.values())))
    counts = labels.loc[samples].value_counts()
    samples = [s for s in samples if counts[labels[s]] >= N_SPLITS]  # a class must reach every test fold
    X = pd.concat([data[omic].loc[samples] for omic in (omics or "mrna").split("+")], axis=1)
    return X.to_numpy(dtype=np.float32), LabelEncoder().fit_transform(labels.loc[samples].to_numpy()), []


def folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified 5-fold, one repeat, seed 42."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))
