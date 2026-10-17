# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""What the examples share: the data, the command-line options and the model settings."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

from tabomics.hub import DEFAULT_CLASSIFIER

# Golub et al. (1999): bone-marrow samples of acute lymphoblastic (ALL) and myeloid (AML) leukemia, expression of
# 7129 genes in 72 patients. OpenML dataset 1104.
LEUKEMIA_ID = 1104


def parser(description: str) -> argparse.ArgumentParser:
    """Options of every example: the data, the model and where it runs."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default="leukemia", help="'leukemia' (downloaded from OpenML), 'synthetic', or a CSV")
    p.add_argument("--target", default=None, help="label column of a CSV (default: the last one)")
    p.add_argument("--model", default=DEFAULT_CLASSIFIER, help="published name, release directory or checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true", help="tiny synthetic run, to check that the example works")
    return p


def load_data(args) -> tuple[pd.DataFrame, np.ndarray]:
    """The table and its labels (0..C-1)."""
    if args.quick or args.data == "synthetic":
        return synthetic(n_rows=60 if args.quick else 120, n_features=200 if args.quick else 2000)
    if args.data == "leukemia":
        import openml

        dataset = openml.datasets.get_dataset(LEUKEMIA_ID, download_qualities=False)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    else:
        table = pd.read_csv(args.data)
        target = args.target or table.columns[-1]
        X, y = table.drop(columns=target), table[target]
    return X, LabelEncoder().fit_transform(y)


def synthetic(n_rows: int, n_features: int, n_informative: int = 5, seed: int = 0):
    """A wide table whose label depends on its first `n_informative` columns only (the rest are noise)."""
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        shuffle=False,
        random_state=seed,
    )
    columns = [f"informative_{i}" for i in range(n_informative)] + [
        f"noise_{i}" for i in range(n_features - n_informative)
    ]
    return pd.DataFrame(X, columns=columns), y


def classifier_settings(args) -> dict:
    """Settings of the Classifier: an omics table is all numeric, so no column is taken as categorical."""
    amp = torch.bfloat16 if str(args.device).startswith("cuda") else None
    return dict(model=args.model, device=args.device, amp_dtype=amp, categorical_features=[], infer_categorical=False)
