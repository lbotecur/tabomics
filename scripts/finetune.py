# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Fine-tunes a model on real tables: a model for a domain, to be evaluated on other tables.

    # v2 fine-tuned on the single-omic tables of MLOmics, then evaluated on the Shamir tables (never seen)
    python -u scripts/finetune.py --model v2 --suite mlomics --data-root /path/to/data --output runs/ft_mlomics \
        --datasets BRCA/mrna COAD/mrna GBM/mrna LGG/mrna OV/mrna --device cuda:0
    python scripts/evaluate.py --model runs/ft_mlomics/finetuned.pth --suite shamir --data-root /path/to/data
    # your own tables (CSV, label in --target or the last column)
    python -u scripts/finetune.py --model v2 --data cohort_a.csv cohort_b.csv --target subtype --output runs/ft_mine

Every row of the given tables is used (10% of each held out for early stopping), so tables used here must not be
the ones the model is evaluated on. To fine-tune on each dataset of a benchmark inside its folds, use
`scripts/evaluate.py --finetune` instead. Very wide tables may need --max-features (columns per episode) to fit in
memory. The output directory gets finetuned.pth (usable wherever a model is: Classifier, evaluate.py, export.py),
history.csv (losses per epoch) and steps.csv.
"""

import argparse

import pandas as pd
import torch

from tabomics.benchmarks import SUITES
from tabomics.finetuning import finetune, prepare_table

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--model", default="v2", help="published name (v2), release directory, .safetensors or .pth")
parser.add_argument("--output", required=True, help="directory for finetuned.pth and the logs")
parser.add_argument("--suite", nargs="+", default=[], choices=sorted(SUITES), help="tables of benchmark suites")
parser.add_argument("--data-root", default=None, help="folder with the benchmark data (with --suite)")
parser.add_argument("--datasets", nargs="+", default=None, help="only these tasks of the suites, e.g. BRCA/mrna")
parser.add_argument("--data", nargs="+", default=[], help="CSV tables")
parser.add_argument("--target", default=None, help="label column of the CSVs (default: the last one)")
parser.add_argument("--device", default=None, help="default: CUDA if available, else CPU")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--steps-per-epoch", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--accumulate", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--validation-fraction", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--max-features", type=int, default=None)
parser.add_argument("--freeze", nargs="+", default=None, help="name prefixes of parameters not to train")
parser.add_argument("--prior-fraction", type=float, default=0.0, help="share of batches of synthetic tables")
parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"], help="bf16 on GPU only (default)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if not args.suite and not args.data:
    parser.error("give tables with --suite and/or --data")
if args.suite and not args.data_root:
    parser.error("--suite needs --data-root")

tables, names, categorical = [], [], []
for suite in args.suite:
    module = SUITES[suite]
    for task in module.tasks(args.data_root):
        if args.datasets is None or task in args.datasets:
            X, y, cat = module.load_task(task, args.data_root)
            tables.append((X, y))
            names.append(f"{suite}:{task}")
            categorical.append(cat)
for path in args.data:
    table = pd.read_csv(path)
    target = args.target or table.columns[-1]
    tables.append((table.drop(columns=target), table[target].to_numpy()))
    names.append(path)
    categorical.append(None)
if not tables:
    parser.error("no table matched --datasets")
for name, (X, _) in zip(names, tables, strict=True):
    print(f"{name}: {X.shape[0]} rows, {X.shape[1]} columns")
# benchmark tables declare their categorical columns and nothing is inferred (as in evaluate.py); in CSV tables
# they are inferred, as Classifier does by default
prepared = [
    prepare_table(X, y, args.validation_fraction, random_state=args.seed + i, categorical_features=cat,
                  infer_categorical=cat is None)
    for i, ((X, y), cat) in enumerate(zip(tables, categorical, strict=True))
]  # fmt: skip

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = "bfloat16" if args.precision == "bf16" and device.startswith("cuda") else None
result = finetune(
    args.model, prepared, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
    accumulate=args.accumulate, lr=args.lr, patience=args.patience, max_features=args.max_features,
    freeze=args.freeze, prior_fraction=args.prior_fraction, device=device, amp_dtype=amp_dtype, seed=args.seed,
    output=args.output,
)  # fmt: skip
print(result.history.round(4).to_string(index=False))
print(f"Fine-tuned model: {args.output}/finetuned.pth")
