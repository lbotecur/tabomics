# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Turns a training checkpoint (.pth) into a release directory, ready to upload to Hugging Face.

Writes model.safetensors, config.json and a model card (README.md), reloads the release and checks that it gives
exactly the outputs of the checkpoint, and prints the SHA-256 of the weights (to register the model in
tabomics/hub.py). For example:

    python scripts/export.py --checkpoint runs/v2.5/3_low_lr/epoch_30.pth --output release/tabomics-v2.5 --name v2.5 \
        --feature-clip 10
"""

import argparse
from pathlib import Path

import torch

from tabomics.hub import sha256_of
from tabomics.model.io import WEIGHTS_FILE, load_checkpoint, load_pretrained, save_pretrained

CARD = """---
license: gpl-3.0
library_name: tabomics
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
  - omics
---

# TabOmics {name}

TabOmics is a tabular foundation model for wide omics data. It is pretrained only on synthetic tables and uses
in-context learning: given labeled training rows and unlabeled test rows, it predicts the test rows in a single forward
pass, with no training or tuning on your data. It is designed for wide tables (thousands of features, few samples, as
in omics), processes every feature jointly, also works on ordinary tables, and provides sample embeddings and feature
importance from the same forward pass.

```python
from tabomics import Classifier
clf = Classifier(model="{name}").fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

## Model details

- {task_line}
- A Transformer of {num_layers} blocks, each with attention between features, attention between rows and an MLP;
  embedding size {embedding_size}, {num_attention_heads} attention heads, {parameters} parameters.
- Pretrained only on synthetic tables; the recipes of the published models are in `configs/` of the repository.
  Standardized features are clipped at {feature_clip:g} at inference.
- `n_estimators` averages ensemble members: forward passes on copies of the table with the columns reordered and the
  categories and classes renumbered at random (1 by default).

Code, documentation and license: https://github.com/lbotecur/tabomics (GNU General Public License v3.0 or later).
"""

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--checkpoint", required=True, help="training checkpoint (.pth)")
parser.add_argument("--output", required=True, help="release directory to write")
parser.add_argument("--name", required=True, help="model name, e.g. v2.5 (used in the model card)")
parser.add_argument("--task", default="classification", choices=["classification", "regression"])
parser.add_argument(
    "--permute-columns",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="store how the model is used: shuffle the columns once at inference (default: keep the checkpoint's)",
)
parser.add_argument(
    "--feature-clip",
    type=float,
    default=None,
    help="store the feature clip the model is meant to be used with (default: the checkpoint's own); v2 and v2.5 "
    "were trained with 100 and are used with 10",
)
args = parser.parse_args()

model = load_checkpoint(args.checkpoint)
if args.permute_columns is not None:
    model.config = type(model.config)(**{**model.config.to_dict(), "permute_columns": args.permute_columns})
if args.feature_clip is not None:
    model.config = type(model.config)(**{**model.config.to_dict(), "feature_clip": args.feature_clip})
    model.feature_clip = args.feature_clip
output = save_pretrained(model, args.output, task=args.task)

# the release must give exactly the checkpoint's outputs
release = load_pretrained(output)
generator = torch.Generator().manual_seed(0)
x = torch.randn(1, 60, 20, generator=generator)
y = torch.randint(0, 2, (1, 40), generator=generator).float()
with torch.no_grad():
    assert torch.equal(model(x, y, 40), release(x, y, 40)), "the release does not reproduce the checkpoint"
assert release.config == model.config

c = model.config
task_line = (
    f"Classifier of up to {c.num_outputs} classes."
    if args.task == "classification"
    else "Regressor: the label is decoded into a distribution over bins of its standardized value."
)
parameters = sum(p.numel() for p in model.parameters())
(Path(output) / "README.md").write_text(
    CARD.format(
        name=args.name,
        task_line=task_line,
        num_layers=c.num_layers,
        embedding_size=c.embedding_size,
        num_attention_heads=c.num_attention_heads,
        parameters=f"{parameters / 1e6:.1f}M",
        feature_clip=c.feature_clip,
    )
)
print(f"release written to {output}: {sorted(p.name for p in Path(output).iterdir())}")
print(f"config: {c}")
print(f"sha256 of {WEIGHTS_FILE}: {sha256_of(Path(output) / WEIGHTS_FILE)}")
