# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Fine-tunes a model on real tables: a model for a domain, to be used on other tables.

    # v2.5 fine-tuned on your own tables (CSV, label in --target or the last column)
    python -u scripts/finetune.py --model v2.5 --data cohort_a.csv cohort_b.csv --target subtype --output runs/ft_mine

Every row of the given tables is used (10% of each held out for early stopping), so tables used here must not be
the ones the model is evaluated on. Each table is preprocessed as inference does (Classifier): its categorical
columns are declared with --categorical, one entry per table in the order of --data, as 0-based positions among its
feature columns (the label column excluded), or inferred (--no-infer-categorical to infer nothing):

    python -u scripts/finetune.py --data a.csv b.csv --categorical 0,3 none --no-infer-categorical --output runs/ft

To fine-tune on the training rows of each fold of a cross-validation, use tabomics.finetuning.FineTunedClassifier
instead. Very wide tables may need --max-features (columns per episode) to fit in memory. The output directory gets
finetuned.pth (usable wherever a model is: Classifier, export.py), history.csv (losses per epoch) and steps.csv.
"""

import argparse

import pandas as pd
import torch

from tabomics.finetuning import finetune, prepare_table
from tabomics.hub import DEFAULT_CLASSIFIER

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument(
    "--model",
    default=DEFAULT_CLASSIFIER,
    help="published name (default %(default)s), release directory, .safetensors or .pth",
)
parser.add_argument("--output", required=True, help="directory for finetuned.pth and the logs")
parser.add_argument("--data", nargs="+", required=True, help="CSV tables")
parser.add_argument("--target", default=None, help="label column of the CSVs (default: the last one)")
parser.add_argument(
    "--categorical",
    nargs="+",
    default=None,
    metavar="POSITIONS",
    help="categorical columns of each table, in the order of --data: 0-based positions among its feature columns, "
    "comma separated, or 'none' (e.g. --categorical 0,3 none)",
)
parser.add_argument("--no-infer-categorical", action="store_true", help="only the declared columns are categorical")
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
if args.categorical is not None and len(args.categorical) != len(args.data):
    parser.error(f"--categorical needs one entry per table: {len(args.data)} tables, {len(args.categorical)} entries")


def positions(entry: str) -> list[int]:
    """'0,3' -> [0, 3]; 'none' -> []."""
    if entry.lower() == "none":
        return []
    try:
        return [int(p) for p in entry.split(",")]
    except ValueError:
        parser.error(f"--categorical takes comma-separated positions or 'none', not {entry!r}")


declared = [positions(e) for e in args.categorical] if args.categorical is not None else [None] * len(args.data)
prepared = []
for i, (path, categorical) in enumerate(zip(args.data, declared, strict=True)):
    table = pd.read_csv(path)
    target = args.target or table.columns[-1]
    X, y = table.drop(columns=target), table[target].to_numpy()
    # preprocessed as inference does: declared categorical columns, and the others inferred unless told not to
    prepared.append(
        prepare_table(X, y, args.validation_fraction, random_state=args.seed + i, categorical_features=categorical,
                      infer_categorical=not args.no_infer_categorical)
    )  # fmt: skip
    print(f"{path}: {X.shape[0]} rows, {X.shape[1]} columns, {int(prepared[-1].categorical.sum())} categorical")

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
