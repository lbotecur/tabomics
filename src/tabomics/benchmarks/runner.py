# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Runs a model on the folds of benchmark suites and writes one row per fold to a CSV (resumable)."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from tabomics.benchmarks import cc18, hdlss, mlomics, shamir
from tabomics.benchmarks.metrics import accuracy, roc_auc

SUITES = {"hdlss": hdlss, "mlomics": mlomics, "shamir": shamir, "cc18": cc18}
COLUMNS = ["suite", "dataset", "fold", "model", "rows", "features", "classes", "roc_auc", "accuracy", "seconds"]

# Builds the estimator for one fold, given the indices of the categorical columns of the task.
EstimatorFactory = Callable[[list[int]], object]


def run_benchmark(
    make_estimator: EstimatorFactory,
    model_name: str,
    suites: list[str],
    data_root: str | Path,
    output: str | Path,
    datasets: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluates one model on every fold of the given suites.

    The estimator only needs fit(X, y) and predict_proba(X), as in scikit-learn. A fresh one is built for every
    fold, so nothing carries over between folds. Rows already in the output (same suite, dataset, fold and model
    name) are skipped, so an interrupted run can be resumed.

    Args:
        make_estimator: Called with the categorical columns of the task; returns the estimator.
        model_name: Name written in the CSV.
        suites: Names in SUITES.
        data_root: Folder with the benchmark data (see each suite's module).
        output: CSV to write (appended to if it exists).
        datasets: Only these tasks (e.g. "arcene", "BRCA/mrna"); default all the tasks of the suites.
        verbose: Print a line per fold.

    Returns:
        All the rows of the output.
    """
    unknown = set(suites) - set(SUITES)
    if unknown:
        raise ValueError(f"Unknown suites {sorted(unknown)}; available: {sorted(SUITES)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=COLUMNS)
    done = set(map(tuple, results[["suite", "dataset", "fold", "model"]].astype(str).values))
    for suite in suites:
        module = SUITES[suite]
        for task in module.tasks(data_root):
            if datasets is not None and task not in datasets:
                continue
            X, y, categorical = module.load_task(task, data_root)
            for fold, (train, test) in enumerate(module.folds(y)):
                if (suite, task, str(fold), model_name) in done:
                    continue
                start = time.time()
                estimator = make_estimator(categorical)
                estimator.fit(X[train], y[train])
                proba = np.asarray(estimator.predict_proba(X[test]))
                row = {
                    "suite": suite, "dataset": task, "fold": fold, "model": model_name, "rows": len(y),
                    "features": X.shape[1], "classes": int(y.max()) + 1, "roc_auc": roc_auc(y[test], proba),
                    "accuracy": accuracy(y[test], proba), "seconds": round(time.time() - start, 2),
                }  # fmt: skip
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                results.to_csv(output, index=False)
                if verbose:
                    print(f"{suite}/{task} fold {fold}: AUROC {row['roc_auc']:.4f} ({row['seconds']}s)", flush=True)
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean AUROC, accuracy and seconds per fold, per suite and model.

    Folds are averaged per dataset first, so every dataset weighs the same whatever its number of folds.
    """
    per_dataset = results.groupby(["suite", "model", "dataset"])[["roc_auc", "accuracy", "seconds"]].mean()
    summary = per_dataset.groupby(["suite", "model"]).mean()
    summary["datasets"] = per_dataset.groupby(["suite", "model"]).size()
    return summary
