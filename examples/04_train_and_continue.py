# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Pretrain a small TabOmics of your own on synthetic tables, then continue training it with something changed.

    python examples/04_train_and_continue.py --device cuda      # ~15 minutes on a GPU

A model is any TabOmicsConfig: choose the layers, the embedding size, the heads... Stage 1 trains it from scratch
on tables of the graph prior; stage 2 starts from its weights and adds widening (thousands of extra, correlated
columns), as the published models were trained in stages. Each stage is the same `train` call; the same can be run
from the command line with a recipe:

    python scripts/train.py --config configs/v2/1_base.json --run-dir runs/mine --set model.num_layers=4

A small model trained for minutes is far from the published one (days on a GPU); the point is the workflow.
"""

import torch
from common import load_data, parser, synthetic
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabomics import Classifier
from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.prior import AugmentationConfig, PriorLoader
from tabomics.training import train

p = parser(__doc__.split("\n")[0])
p.add_argument("--run-dir", default="runs/example")
p.add_argument("--workers", type=int, default=4, help="processes generating tables")
args = p.parse_args()
quick = args.quick
amp = torch.bfloat16 if args.device.startswith("cuda") else None


def evaluate(model: TabOmicsModel, name: str) -> None:
    """AUROC on a held-out split of the table, and on a wide synthetic table (5 informative columns)."""
    for label, (X, y) in [("table", load_data(args)), ("wide synthetic", synthetic(100, 200 if quick else 2000))]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        clf = Classifier(
            model=model, device=args.device, amp_dtype=amp, categorical_features=[], infer_categorical=False
        ).fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        auroc = (
            roc_auc_score(y_test, proba[:, 1])
            if proba.shape[1] == 2
            else roc_auc_score(y_test, proba, multi_class="ovr")
        )
        print(f"{name}: AUROC on {label} {auroc:.3f}")


# Your architecture
config = TabOmicsConfig(
    embedding_size=16 if quick else 96,
    num_attention_heads=2 if quick else 4,
    mlp_hidden_size=32 if quick else 384,
    num_layers=2 if quick else 6,
)
torch.manual_seed(0)
model = TabOmicsModel(config)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")


def loader(max_rows: int) -> PriorLoader:
    """Batches of 8 tables of 40-max_rows rows and 2-50 columns from the graph prior."""
    return PriorLoader(
        num_steps=2 if quick else 500,
        batch_size=8,
        min_rows=40,
        max_rows=max_rows,
        min_features=2,
        max_features=50,
        max_classes=10,
        num_workers=0 if quick else args.workers,
    )


# Stage 1: from scratch, narrow tables, up to 10% missing values
train(
    model,
    loader(300),
    f"{args.run_dir}/stage1",
    epochs=1 if quick else 12,
    lr=3e-4,
    warmup_steps=100,
    augmentation=AugmentationConfig(missing_rate_max=0.1),
    device=args.device,
    amp_dtype=amp,
    log_every=100,
)
evaluate(model, "stage 1")

# Stage 2: the same weights, now with widening: 70% of the batches get 200-2000 extra columns, built from the
# existing ones (TabPFN-Wide), on tables shortened to 300 rows so that they fit in memory
widening = AugmentationConfig(
    add_features_min=200, add_features_max=2000, prob_no_widening=0.3, widen_max_rows=300, missing_rate_max=0.1
)
if quick:
    widening.add_features_min, widening.add_features_max = 5, 20
train(
    model,
    loader(300),
    f"{args.run_dir}/stage2",
    epochs=1 if quick else 6,
    lr=1e-4,
    warmup_steps=100,
    augmentation=widening,
    device=args.device,
    amp_dtype=amp,
    log_every=100,
)
evaluate(model, "stage 2")
print(
    f"checkpoints in {args.run_dir}/stage1 and stage2; use one with Classifier(model='{args.run_dir}/stage2/"
    "latest_checkpoint.pth')"
)
