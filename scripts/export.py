# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Turns a training checkpoint (.pth) into a release directory, ready to upload to Hugging Face.

Writes model.safetensors, config.json and a model card (README.md), reloads the release and checks that it gives
exactly the outputs of the checkpoint, and prints the SHA-256 of the weights (to register the model in
tabomics/hub.py). For example:

    python scripts/export.py --checkpoint runs/v2/4_low_lr/epoch_30.pth --output release/tabomics-v2 --name v2 \
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

Tabular foundation model for wide omics data: classifies the rows of a table in context, in one forward pass,
and gives row embeddings and feature importance. {parameters:.1f}M parameters; feature clip {feature_clip:g};
column permutation {permute}.

```python
from tabomics import Classifier
clf = Classifier(model="{name}").fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

Code, documentation and licence: https://github.com/lbotecur/tabomics (GPL-3.0-or-later). The architecture
derives from nanoTabPFN (Pfefferle et al.); the synthetic training tables come from a generator derived from
TabICL's graph prior.
"""

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--checkpoint", required=True, help="training checkpoint (.pth)")
parser.add_argument("--output", required=True, help="release directory to write")
parser.add_argument("--name", required=True, help="model name, e.g. v2 (used in the model card)")
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
    help="store the feature clip the model is meant to be used with (default: the checkpoint's own); the v2 was "
    "trained with 100 and is used with 10",
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
(Path(output) / "README.md").write_text(
    CARD.format(
        name=args.name,
        parameters=sum(p.numel() for p in model.parameters()) / 1e6,
        feature_clip=c.feature_clip,
        permute="on" if c.permute_columns else "off",
    )
)
print(f"release written to {output}: {sorted(p.name for p in Path(output).iterdir())}")
print(f"config: {c}")
print(f"sha256 of {WEIGHTS_FILE}: {sha256_of(Path(output) / WEIGHTS_FILE)}")
