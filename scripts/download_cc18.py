# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Downloads the CC18 suite (the narrow tables of OpenML-CC18) to <data_root>/OpenML-CC18 and writes index.csv.

Datasets already downloaded are skipped. Needs internet access.

    python scripts/download_cc18.py --data-root /path/to/data
"""

import argparse
from pathlib import Path

import openml
import pandas as pd

from tabomics.benchmarks import cc18

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--data-root", required=True)
args = parser.parse_args()

folder = Path(args.data_root) / cc18.FOLDER
folder.mkdir(parents=True, exist_ok=True)
rows = []
for task_id in openml.study.get_suite(cc18.SUITE_ID).tasks:
    task = openml.tasks.get_task(task_id, download_splits=False)
    dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False, download_qualities=True)
    q = dataset.qualities
    n, d, c = int(q["NumberOfInstances"]), int(q["NumberOfFeatures"]) - 1, int(q["NumberOfClasses"])
    if not cc18.keep(dataset.name, n, d, c):
        print(f"skip {dataset.name} ({n} rows, {d} features, {c} classes)", flush=True)
        continue
    path = folder / f"{dataset.name}.npz"
    if not path.exists():
        X, y, categorical, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
        cc18.save_task(path, *cc18.to_arrays(X, y, categorical))
    X, y, categorical = cc18.load_task(dataset.name, args.data_root)
    rows.append({"name": dataset.name, "task_id": task_id, "dataset_id": task.dataset_id, "rows": X.shape[0],
                 "features": X.shape[1], "classes": int(y.max()) + 1, "categorical": len(categorical)})  # fmt: skip
    print(f"{dataset.name}: {X.shape[0]} rows, {X.shape[1]} features, {rows[-1]['classes']} classes", flush=True)
pd.DataFrame(rows).to_csv(folder / "index.csv", index=False)
print(f"{len(rows)} datasets in {folder}")
