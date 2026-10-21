# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Evaluates a baseline model on the benchmark suites, on the same folds as scripts/evaluate.py.

    python baselines/evaluate.py --model xgboost --suite hdlss mlomics --data-root /path/to/data
    python baselines/evaluate.py --model tabpfn-3.5:cover --suite hdlss --data-root /path/to/data --device cuda:1

Writes one row per fold to a CSV (results/<model>.csv by default; it resumes an interrupted run), with the same
columns as scripts/evaluate.py, so the results of all models can be put together and summarized.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import available, make_model  # noqa: E402

from tabomics.benchmarks import SUITES, run_benchmark, summarize  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--model", required=True, help=f"one of {', '.join(available())} (see models.py for suffixes)")
parser.add_argument("--suite", nargs="+", required=True, choices=sorted(SUITES))
parser.add_argument("--data-root", required=True, help="folder with the benchmark data (docs/benchmarks.md)")
parser.add_argument("--datasets", nargs="+", default=None, help="only these tasks, e.g. arcene BRCA/mrna")
parser.add_argument("--output", default=None, help="CSV to write (default results/<model>.csv)")
parser.add_argument("--device", default="cuda", help="for the models that use a GPU")
parser.add_argument("--retry-failed", action="store_true", help="evaluate again the folds that failed (oom, error)")
args = parser.parse_args()

output = args.output or f"results/{args.model.replace(':', '_')}.csv"
results = run_benchmark(
    lambda categorical: make_model(args.model, categorical, args.device),
    args.model,
    args.suite,
    args.data_root,
    output,
    datasets=args.datasets,
    retry_failed=args.retry_failed,
)
print(summarize(results[results.model == args.model]).round(4).to_string())
print(f"results: {output}")
