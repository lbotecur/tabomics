"""TabOmicsModel gives the same outputs as the reference implementation it was cleaned from (nanoTabPFN as
modified in TFM-Playground), with the same weights: tests/fixtures/parity_model.npz holds a small model's
weights, inputs and the reference outputs (written by tests/fixtures/make_parity_fixture.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tabomics.model import TabOmicsConfig, TabOmicsModel

FIXTURE = Path(__file__).parent / "fixtures" / "parity_model.npz"


@pytest.fixture(scope="module")
def reference():
    data = np.load(FIXTURE)
    weights = {k.split("::", 1)[1]: torch.from_numpy(data[k]) for k in data.files if k.startswith("weight::")}
    model = TabOmicsModel(
        TabOmicsConfig(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
    )
    model.load_state_dict(weights)  # strict: the parameter names are the reference's
    return model.eval(), data


def test_same_parameter_names_as_the_reference(reference):
    model, data = reference
    names = {k.split("::", 1)[1] for k in data.files if k.startswith("weight::")}
    assert set(model.state_dict()) == names


def test_same_logits_embeddings_and_label_attention(reference):
    model, data = reference
    x, y, n = torch.from_numpy(data["x"]), torch.from_numpy(data["y"]), int(data["train_size"])
    with torch.no_grad():
        out = model.run(x, y, n, return_embeddings=True, return_label_attention=True)
    # same operations in the same order: identical on the same machine, rounding-level elsewhere
    np.testing.assert_allclose(out.logits.numpy(), data["logits"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(out.embeddings.numpy(), data["embeddings"], rtol=0, atol=1e-6)
    # the reference, patched to average over the test rows (tests/fixtures/reference_patch.py), of both tables
    mean = out.label_attention.mean(dim=(0, 1)).numpy()
    np.testing.assert_allclose(mean, data["test_label_attention"], rtol=0, atol=1e-6)
