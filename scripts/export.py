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
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
  - omics
---

# TabOmics {name}

A tabular foundation model for wide omics data. Given labelled training rows and unlabelled test rows, it reads
them together and classifies the test rows in one forward pass, with no training on the user's data. It also gives
row embeddings and feature importance.

```python
from tabomics import Classifier  # pip install git+https://github.com/lbotecur/tabomics.git

clf = Classifier(model="{name}").fit(X_train, y_train)
proba = clf.predict_proba(X_test)
importance = clf.feature_importance(X_test)
```

Code, documentation, examples and licence: [github.com/lbotecur/tabomics](https://github.com/lbotecur/tabomics).

## Model

- Transformer over the cells of a table (attention between the columns of each row, then between rows; test rows
  attend to training rows only), with the label as one more column. {layers} layers, embedding {embedding},
  {heads} attention heads, {parameters:.1f}M parameters, up to {classes} classes.
- Each column is standardized with the training rows; standardized values are clipped to +/-{feature_clip:g}.
  Column permutation at inference: {permute}.
- Files: `model.safetensors` (weights) and `config.json` (architecture and how the model is used).

## Training

Pretrained on synthetic tables only: no real data was used. The tables come from random causal graphs (a generator
derived from TabICL's graph prior) and are widened with thousands of correlated columns (as TabPFN-Wide), with up to
10% missing values. The stages, with every setting, are in the repository's `configs/` (one folder per model), and
`scripts/train.py` reruns them.

## Intended use and limitations

- Classification of tables with few rows and many columns (omics cohorts: tens to a few thousand patients,
  thousands of features), and of ordinary tables; research use.
- Up to {classes} classes. Tables much longer or wider than the training tables are outside what the model saw
  and may be predicted less well.
- Feature importance is the model's attention, an indication of which columns it uses, not a causal or statistical
  test.
- Not a medical device: predictions on patient data must not be used for clinical decisions.

## Licence and credits

GPL-3.0-or-later. The architecture derives from nanoTabPFN (Pfefferle et al., 2025), the code from TFM-Playground,
the synthetic-table generator from TabICL's prior (Qu et al., 2025, 2026), and the widening follows TabPFN-Wide
(Kolberg et al., 2025); see the repository's README, NOTICE and LICENSES.
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
        layers=c.num_layers,
        embedding=c.embedding_size,
        heads=c.num_attention_heads,
        classes=c.num_outputs,
        feature_clip=c.feature_clip,
        permute="on" if c.permute_columns else "off",
    )
)
print(f"release written to {output}: {sorted(p.name for p in Path(output).iterdir())}")
print(f"config: {c}")
print(f"sha256 of {WEIGHTS_FILE}: {sha256_of(Path(output) / WEIGHTS_FILE)}")
