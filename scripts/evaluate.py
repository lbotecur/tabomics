# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Evaluates a TabOmics model on benchmark suites: one row per fold in a CSV (resumable), and a summary.

    python scripts/evaluate.py --model v2.5 --suite hdlss mlomics --data-root /path/to/data
    python scripts/evaluate.py --model my_model.pth --suite cc18 --data-root /path/to/data --device cuda:1
    # fine-tuned on the training rows of each fold first (tabomics.finetuning.FineTunedClassifier)
    python scripts/evaluate.py --model v2.5 --suite hdlss --data-root /path/to/data --finetune

The data of each suite is described in its module (tabomics/benchmarks/) and in docs/benchmarks.md. Other models
(any scikit-learn-like estimator) are evaluated with tabomics.benchmarks.run_benchmark from Python.
"""

import argparse

import torch

from tabomics import Classifier
from tabomics.benchmarks import SUITES, run_benchmark, summarize
from tabomics.finetuning import FineTunedClassifier
from tabomics.hub import DEFAULT_CLASSIFIER
from tabomics.inference.base import default_device
from tabomics.model.io import load_model

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument(
    "--model",
    default=DEFAULT_CLASSIFIER,
    help="published name (default %(default)s), release directory, .safetensors or .pth",
)
parser.add_argument("--suite", nargs="+", required=True, choices=sorted(SUITES))
parser.add_argument("--data-root", required=True, help="folder with the benchmark data")
parser.add_argument("--datasets", nargs="+", default=None, help="only these tasks, e.g. arcene BRCA/mrna")
parser.add_argument("--output", default=None, help="CSV to write (default results/<model name>.csv)")
parser.add_argument("--name", default=None, help="model name in the CSV (default: --model)")
parser.add_argument("--device", default=None, help="default: CUDA if available, else CPU")
parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"], help="bf16 on GPU only (default)")
parser.add_argument("--n-estimators", type=int, default=1)
parser.add_argument("--finetune", action="store_true", help="fine-tune on the training rows of each fold first")
parser.add_argument("--retry-failed", action="store_true", help="evaluate again the folds that failed (oom, error)")
args = parser.parse_args()

device = args.device or str(default_device())
amp_dtype = torch.bfloat16 if args.precision == "bf16" and device.startswith("cuda") else None
name = args.name or args.model
if args.n_estimators > 1:
    name = f"{name} [{args.n_estimators} estimators]"
if args.finetune:
    name = f"{name} [fine-tuned]"
output = args.output or f"results/{name.replace('/', '_').replace(' ', '_')}.csv"
model = load_model(args.model)  # once, shared by the folds


def make_estimator(categorical: list[int]) -> Classifier:
    """The categorical columns of each task are declared, nothing else is inferred."""
    return (FineTunedClassifier if args.finetune else Classifier)(
        model=model, device=device, amp_dtype=amp_dtype, n_estimators=args.n_estimators,
        categorical_features=categorical, infer_categorical=False,
    )  # fmt: skip


results = run_benchmark(
    make_estimator, name, args.suite, args.data_root, output, datasets=args.datasets, retry_failed=args.retry_failed
)
print(summarize(results[results.model == name]).round(4).to_string())
print(f"results: {output}")
