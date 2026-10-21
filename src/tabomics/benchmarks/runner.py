# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Runs a model on the folds of benchmark suites and writes one row per fold to a CSV (resumable)."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from tabomics.benchmarks import cc18, hdlss, mlomics, shamir
from tabomics.benchmarks.metrics import accuracy, roc_auc

SUITES = {"hdlss": hdlss, "mlomics": mlomics, "shamir": shamir, "cc18": cc18}
COLUMNS = ["suite", "dataset", "fold", "model", "rows", "features", "classes", "roc_auc", "accuracy", "seconds",
           "status", "error"]  # fmt: skip
# Outcome of a fold: "ok", "oom" (out of memory) or "error" (any other exception, its message in "error").
STATUSES = ("ok", "oom", "error")

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
    retry_failed: bool = False,
) -> pd.DataFrame:
    """Evaluates one model on every fold of the given suites.

    The estimator only needs fit(X, y) and predict_proba(X), as in scikit-learn. A fresh one is built for every
    fold, so nothing carries over between folds. Rows already in the output (same suite, dataset, fold and model
    name) are skipped, so an interrupted run can be resumed.

    A fold that raises an exception does not stop the run: its row records the status "oom" (out of memory) or
    "error" (with the message), NaN metrics and the seconds spent, and the next fold follows. Failed rows count as
    done when resuming, unless retry_failed is set.

    Args:
        make_estimator: Called with the categorical columns of the task; returns the estimator.
        model_name: Name written in the CSV.
        suites: Names in SUITES.
        data_root: Folder with the benchmark data (see each suite's module).
        output: CSV to write (appended to if it exists).
        datasets: Only these tasks (e.g. "arcene", "BRCA/mrna"); default all the tasks of the suites.
        verbose: Print a line per fold.
        retry_failed: Evaluate again the folds whose row has a status other than "ok".

    Returns:
        All the rows of the output.
    """
    unknown = set(suites) - set(SUITES)
    if unknown:
        raise ValueError(f"Unknown suites {sorted(unknown)}; available: {sorted(SUITES)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=COLUMNS)
    if "status" not in results:  # written before the status column existed: every row was a success
        results["status"], results["error"] = "ok", ""
    if retry_failed:
        results = results[results.status == "ok"].reset_index(drop=True)
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
                row = {"suite": suite, "dataset": task, "fold": fold, "model": model_name, "rows": len(y),
                       "features": X.shape[1], "classes": int(y.max()) + 1}  # fmt: skip
                try:
                    estimator = make_estimator(categorical)
                    estimator.fit(X[train], y[train])
                    proba = np.asarray(estimator.predict_proba(X[test]))
                    row.update(roc_auc=roc_auc(y[test], proba), accuracy=accuracy(y[test], proba), status="ok",
                               error="")  # fmt: skip
                except Exception as exc:  # recorded, not raised: the other folds still run (Ctrl-C still stops)
                    row.update(roc_auc=np.nan, accuracy=np.nan, status=failure_status(exc),
                               error=f"{type(exc).__name__}: {exc}"[:500])  # fmt: skip
                    estimator = proba = None
                    release_memory()
                row["seconds"] = round(time.time() - start, 2)
                results = pd.concat([results, pd.DataFrame([row])[COLUMNS]], ignore_index=True)
                results.to_csv(output, index=False)
                if verbose:
                    outcome = f"AUROC {row['roc_auc']:.4f}" if row["status"] == "ok" else row["status"].upper()
                    print(f"{suite}/{task} fold {fold}: {outcome} ({row['seconds']}s)", flush=True)
    return results


def failure_status(exc: BaseException) -> str:
    """Status of a failed fold: "oom" for an out-of-memory error (CUDA or host), "error" for anything else."""
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc):
        return "oom"
    return "error"


def release_memory() -> None:
    """Frees what a failed fold left behind, so that the next fold starts with the memory it would otherwise have."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean AUROC, accuracy and seconds per fold, per suite and model, over the folds that succeeded.

    Folds are averaged per dataset first, so every dataset weighs the same whatever its number of folds. The column
    "failed" counts the folds that did not succeed (status other than "ok"), which the means leave out.
    """
    status = results["status"] if "status" in results else pd.Series("ok", index=results.index)
    ok = results[status == "ok"]
    per_dataset = ok.groupby(["suite", "model", "dataset"])[["roc_auc", "accuracy", "seconds"]].mean()
    summary = per_dataset.groupby(["suite", "model"]).mean()
    summary["datasets"] = per_dataset.groupby(["suite", "model"]).size()
    failed = results[status != "ok"].groupby(["suite", "model"]).size()
    summary = summary.reindex(summary.index.union(failed.index))  # a model whose folds all failed is still listed
    summary["datasets"] = summary["datasets"].fillna(0).astype(int)
    summary["failed"] = failed.reindex(summary.index, fill_value=0).astype(int)
    return summary
