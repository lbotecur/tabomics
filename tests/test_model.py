import warnings

import numpy as np
import pytest
import torch

from tabomics.model import TabOmicsConfig, TabOmicsModel, normalize_features, pad_targets
from tabomics.model.layers import apply_in_chunks, label_attention_weights

SMALL = TabOmicsConfig(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)


def small_model(**changes) -> TabOmicsModel:
    torch.manual_seed(0)
    config = TabOmicsConfig(**{**SMALL.to_dict(), **changes})
    return TabOmicsModel(config).eval()


def table(batch=2, rows=30, columns=7, train_size=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, rows, columns, generator=g)
    y = torch.randint(0, 3, (batch, train_size), generator=g).float()
    return x, y, train_size


# --- preprocessing ---------------------------------------------------------------------------------------------


def test_normalize_features_uses_training_rows_and_flags_missing():
    x = torch.tensor([[[1.0], [3.0], [float("nan")], [5.0]]])  # rows 0-1 train (mean 2, std sqrt 2), 2-3 test
    out = normalize_features(x, train_size=2, clip=100.0)
    assert out.shape == (1, 4, 1, 2)
    expected = (torch.tensor([1.0, 3.0, 2.0, 5.0]) - 2.0) / np.sqrt(2.0)  # the missing value becomes the mean
    assert torch.allclose(out[0, :, 0, 0], expected, atol=1e-5)
    assert out[0, :, 0, 1].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_normalize_features_clips_and_survives_a_single_training_row():
    x = torch.tensor([[[0.0], [1.0], [0.0], [-1.0], [1e6]]])
    assert normalize_features(x, 4)[0, -1, 0, 0].item() == 10.0  # default clip
    single = normalize_features(torch.tensor([[[2.0], [5.0]]]), 1)
    assert torch.isfinite(single).all()


def test_pad_targets_flags_test_rows():
    out = pad_targets(torch.tensor([[[0.0], [2.0]]]), num_rows=4)
    assert out.shape == (1, 4, 1, 2)
    assert out[0, :, 0, 0].tolist() == [0.0, 2.0, 1.0, 1.0]  # test rows get the training mean
    assert out[0, :, 0, 1].tolist() == [0.0, 0.0, 1.0, 1.0]


# --- building blocks -------------------------------------------------------------------------------------------


def test_apply_in_chunks_matches_one_piece_and_refuses_with_gradients():
    x = torch.randn(10, 3)
    with torch.no_grad():
        assert torch.equal(apply_in_chunks(lambda t: t * 2, x.clone(), 3), x * 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_in_chunks(lambda t: t * 2, x.clone().requires_grad_(), 3)
    assert any("disabled" in str(w.message) for w in caught)


def test_label_attention_weights_equal_the_full_attention_row():
    attention = torch.nn.MultiheadAttention(16, 2, batch_first=True)
    x = torch.randn(5, 7, 16)
    full = attention(x, x, x, need_weights=True, average_attn_weights=True)[1]  # (N, C, C)
    assert torch.allclose(label_attention_weights(attention, x), full[:, -1], atol=1e-6)


# --- the model -------------------------------------------------------------------------------------------------


def test_shapes_and_extras():
    model = small_model()
    x, y, n = table()
    with torch.no_grad():
        out = model.run(x, y, n, return_embeddings=True, return_label_attention=True)
    assert out.logits.shape == (2, 10, 10)
    assert out.embeddings.shape == (2, 30, 16)
    assert out.label_attention.shape == (8,)  # 7 columns + the label
    assert torch.allclose(out.label_attention.sum(), torch.tensor(1.0), atol=1e-5)
    with torch.no_grad():
        assert torch.equal(model(x, y, n), out.logits)


def test_test_rows_do_not_influence_each_other():
    model = small_model()
    x, y, n = table()
    changed = x.clone()
    changed[:, -1] += 5.0  # change only the last test row
    with torch.no_grad():
        a, b = model(x, y, n), model(changed, y, n)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_training_row_order_does_not_matter():
    model = small_model()
    x, y, n = table()
    order = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    shuffled = torch.cat([x[:, order], x[:, n:]], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(x, y, n), model(shuffled, y[:, order], n), atol=1e-5)


def test_chunking_and_low_memory_give_the_same_logits():
    model = small_model()
    x, y, n = table()
    with torch.no_grad():
        reference = model(x, y, n)
        chunked = model(x, y, n, num_mem_chunks=4)
        model.set_low_memory(True)
        low_memory = model(x, y, n, num_mem_chunks=4)
    assert torch.allclose(reference, chunked, atol=1e-6)
    assert torch.allclose(reference, low_memory, atol=1e-6)


def test_gradient_checkpointing_gives_the_same_gradients():
    x, y, n = table()
    target = torch.randint(0, 3, (2, 10))
    grads = []
    for checkpointing in (False, True):
        model = small_model().train()
        model.gradient_checkpointing = checkpointing
        loss = torch.nn.functional.cross_entropy(model(x, y, n).reshape(-1, 10), target.reshape(-1))
        loss.backward()
        grads.append(model.decoder.linear1.weight.grad.clone())
    assert torch.allclose(*grads, atol=1e-6)


def test_feature_clip_is_settable():
    model = small_model(feature_clip=100.0)
    assert model.feature_clip == 100.0
    model.feature_clip = 10
    assert model.feature_encoder.clip == 10.0


def test_config_round_trip_ignores_unknown_keys():
    config = TabOmicsConfig(num_layers=3, permute_columns=True)
    assert TabOmicsConfig.from_dict({**config.to_dict(), "something_else": 1}) == config


@pytest.mark.parametrize("batch", [1, 3])
def test_missing_values_are_handled(batch):
    model = small_model()
    x, y, n = table(batch=batch)
    x[0, 2, 1] = float("nan")
    x[0, 25, 3] = float("nan")
    with torch.no_grad():
        assert torch.isfinite(model(x, y, n)).all()
