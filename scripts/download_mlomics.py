# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Downloads the classification datasets of MLOmics (Hugging Face dataset AIBIC/MLOmics).

They go to <data_root>/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets. Needs internet access.

    python scripts/download_mlomics.py --data-root /path/to/data
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from tabomics.benchmarks import mlomics

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--data-root", required=True)
args = parser.parse_args()

target = Path(args.data_root) / mlomics.FOLDER
snapshot_download(
    repo_id="AIBIC/MLOmics",
    repo_type="dataset",
    allow_patterns=[f"Main_Dataset/Classification_datasets/GS-{cancer}/Original/*" for cancer in mlomics.CANCERS],
    local_dir=target,
)
print(f"MLOmics classification datasets in {target / 'Main_Dataset' / 'Classification_datasets'}")
