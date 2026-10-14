"""Augmentation of prior batches: the same tables as TFM-Playground's training loop, where TabOmics was developed
(tests/fixtures/parity_widening.npz, written by tests/fixtures/make_widening_fixture.py), and the properties each
augmentation promises."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tabomics.prior import AugmentationConfig, augment_batch
from tabomics.prior.widening import (
    add_mixed_widening_features,
    detect_categorical_mask,
    inject_mcar_missingness,
    recode_categories,
    reduce_cardinality,
    shorten_table,
    sparsify_columns,
    split_widening_budget,
)

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
from widening_cases import CASES, case_input  # noqa: E402


@pytest.mark.parametrize("name", list(CASES))
def test_same_tables_as_the_training_loop_of_tfm_playground(name):
    case = CASES[name]
    expected = np.load(FIXTURES / "parity_widening.npz")
    x, y, masks = case_input(case)
    torch.manual_seed(case["seed"])
    x, y, train_size = augment_batch(x, y, case["train_size"], AugmentationConfig(**case["config"]), masks)
    assert train_size == int(expected[f"{name}/train_size"])
    assert x.shape == expected[f"{name}/x"].shape
    np.testing.assert_array_equal(np.isnan(x.numpy()), np.isnan(expected[f"{name}/x"]))  # same missing cells
    # widened columns are matrix products: rounding may differ between machines
    np.testing.assert_allclose(x.numpy(), expected[f"{name}/x"], rtol=1e-4, atol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(y.numpy(), expected[f"{name}/y"])


def test_nothing_is_done_by_default_and_a_seed_gives_the_same_tables():
    x, y, masks = case_input(CASES["widen_sparse"])
    same = augment_batch(x, y, 70, AugmentationConfig(), masks)
    assert torch.equal(same[0], x) and torch.equal(same[1], y) and same[2] == 70
    config = AugmentationConfig(**CASES["widen_sparse"]["config"])
    torch.manual_seed(5)
    first = augment_batch(x, y, 70, config, masks)
    torch.manual_seed(5)
    again = augment_batch(x, y, 70, config, masks)
    assert torch.equal(first[0].nan_to_num(nan=-1), again[0].nan_to_num(nan=-1))


def test_widening_keeps_the_originals_or_not_and_recodes_new_categoricals():
    x, _, masks = case_input(dict(CASES["widen_masks_short"], with_masks=True))
    torch.manual_seed(0)
    kept = add_mixed_widening_features(x, 40, 0.3, 0.5, include_original_prob=1.0, categorical_masks=masks,
                                       recode_categoricals=True)  # fmt: skip
    assert kept.shape == (2, 100, 52)
    for table, original in zip(kept, x, strict=True):  # every original column is still there, somewhere
        assert all(any(torch.equal(table[:, j], column) for j in range(52)) for column in original.T)
    dropped = add_mixed_widening_features(x, 40, 0.3, 0.5, include_original_prob=0.0, categorical_masks=masks,
                                          recode_categoricals=True)  # fmt: skip
    assert dropped.shape == (2, 100, 40)
    # new categorical columns: codes 0..K-1, K <= 20
    _, n_categorical = split_widening_budget(masks[0], 40)
    for column in dropped[0, :, -n_categorical:].T:
        values = torch.unique(column)
        assert torch.equal(values, torch.arange(len(values), dtype=values.dtype)) and len(values) <= 20


def test_shorten_table_keeps_the_share_of_training_rows():
    x, y = torch.arange(200.0).reshape(1, 200, 1), torch.arange(200.0)[None]
    xs, ys, n_train = shorten_table(x, y, 140, 50)
    assert xs.shape == (1, 50, 1) and n_train == 35
    assert torch.equal(ys[0, :35], torch.arange(35.0)) and torch.equal(ys[0, 35:], torch.arange(140.0, 155.0))


def test_mcar_sparse_and_category_helpers():
    torch.manual_seed(0)
    x = torch.randn(2, 2000, 30)
    rate = torch.isnan(inject_mcar_missingness(x, 0.2)).float().mean().item()
    assert abs(rate - 0.2) < 0.01 and torch.equal(inject_mcar_missingness(x, 0.0), x)
    sparse = sparsify_columns(x, 1.0)
    zeros = (sparse == 0).float().mean(dim=1)
    assert ((zeros >= 0.79) & (zeros <= 0.999)).all() and (sparse >= 0).all()  # 0.2% to 20% nonzero, from 0
    column = torch.tensor([5.0, 5.0, 7.0, 9.0, float("nan"), 7.0])
    recoded = recode_categories(column)
    assert torch.isnan(recoded[4]) and set(recoded[~torch.isnan(recoded)].tolist()) == {0.0, 1.0, 2.0}
    assert len(torch.unique(reduce_cardinality(torch.arange(50.0), 7))) == 7
    table = torch.stack([torch.randn(100), torch.randint(0, 3, (100,)).float()], dim=1)
    table[0, 1] = float("nan")  # a missing value is not a category
    assert detect_categorical_mask(table).tolist() == [False, True]
