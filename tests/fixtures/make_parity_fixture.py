"""Writes parity_model.npz: a small model's weights, inputs, and the outputs of the reference implementation
(nanoTabPFN as modified in TFM-Playground, where TabOmics was developed). test_model_parity.py checks that
TabOmicsModel gives the same outputs. Needs TFM-Playground's tfmplayground package; run from the repository root:
    python tests/fixtures/make_parity_fixture.py
"""

from pathlib import Path

import numpy as np
import torch
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

torch.manual_seed(0)
model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
# Left in training mode (no .eval()), as the reference classifier ran it: v2 was trained and evaluated on that
# path of nn.MultiheadAttention (see tabomics.model.layers._no_fast_path).
rng = np.random.default_rng(0)
x = rng.normal(size=(2, 30, 7)).astype(np.float32)
x[0, 3, 2] = np.nan  # missing values, in training and test rows
x[1, 25, 4] = np.nan
x[:, :, 5] *= 50.0  # a column beyond the clip after normalization in some test rows
y = rng.integers(0, 3, size=(2, 20)).astype(np.float32)
train_size = 20

with torch.no_grad():
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    logits = model((xt, yt), train_test_split_index=train_size)
    model.save_embeddings = True
    for block in model.transformer_blocks:
        block.save_feature_attention = True
    model((xt, yt), train_test_split_index=train_size)
    embeddings = model.embeddings
    # averaged over all rows, training and test: compared with TabOmicsModel.run(label_attention_rows="all")
    attention = torch.stack([b.feature_attention for b in model.transformer_blocks]).mean(dim=0)

np.savez(
    Path(__file__).with_name("parity_model.npz"),
    x=x,
    y=y,
    train_size=train_size,
    logits=logits.numpy(),
    embeddings=embeddings.numpy(),
    label_attention=attention.numpy(),
    **{f"weight::{k}": v.numpy() for k, v in model.state_dict().items()},
)
print("written", {k: tuple(v.shape) for k, v in model.state_dict().items()}.__len__(), "tensors")
