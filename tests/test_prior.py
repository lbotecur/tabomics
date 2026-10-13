"""The graph prior: with raw=False it gives TabICL 2.2.0's tables (bit for bit on the same machine; against
tests/fixtures/parity_prior.npz, written by tests/fixtures/make_prior_fixture.py on another machine, the same tables
up to float rounding); with raw=True the same tables before TabICL's clamp and z-score, with their categorical
columns as codes 0..K-1 and marked."""

import importlib.util
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tabomics.prior import GraphPrior, GraphSCM, PriorConfig
from tabomics.prior.graph._scaling import outlier_removing, standard_scaling
from tabomics.prior.graph.generator import drop_constant_columns, fix_split

FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = dict(batch_size=6, batch_size_per_gp=3, min_features=2, max_features=15, max_classes=6, min_seq_len=60,
                max_seq_len=150)  # fmt: skip

# Generates the fixture's batches with TabOmics (or TabICL) in a fresh interpreter with PYTHONHASHSEED=0.
GENERATE = """
import sys
import numpy as np
sys.path.insert(0, {fixtures!r})
from prior_cases import generate
if {tabicl}:
    from tabicl.prior._dataset import GraphPrior
    from tabicl.prior.graph_lib._config import PriorConfig
    batches = generate(GraphPrior, PriorConfig, n_jobs=1)
else:
    from tabomics.prior import GraphPrior, PriorConfig
    batches = generate(GraphPrior, PriorConfig, raw=False)
np.savez(sys.argv[1], **batches)
"""


def seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def generate_in_subprocess(path: Path, tabicl: bool = False) -> dict:
    code = GENERATE.format(fixtures=str(FIXTURES), tabicl=tabicl)
    subprocess.run([sys.executable, "-c", code, str(path)], env={**os.environ, "PYTHONHASHSEED": "0"}, check=True)
    return dict(np.load(path))


def assert_same(expected: dict, actual: dict, exact: bool = True) -> None:
    """Same arrays, bit for bit; with exact=False (a fixture written on another machine), integers (labels, sizes)
    must still be equal but generated floats only close. CPU kernels round differently between machines (AVX-512,
    AVX2, plain): classification tables then differ by < 3e-3, while regression targets come out of chained random
    functions that amplify rounding (differences up to ~1 measured), so their floats are not compared across
    machines; the comparison with an installed tabicl, on the same machine, stays bit for bit."""
    assert sorted(expected) == sorted(actual)
    for key in expected:
        assert expected[key].dtype == actual[key].dtype and expected[key].shape == actual[key].shape, key
        if exact or not np.issubdtype(expected[key].dtype, np.floating):
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
        elif not key.startswith("regression/"):
            np.testing.assert_allclose(expected[key], actual[key], rtol=1e-3, atol=1e-2, err_msg=key)


# --- identity with TabICL ---------------------------------------------------------------------------------------


def test_scaled_tables_are_tabicls(tmp_path):
    fixture = dict(np.load(FIXTURES / "parity_prior.npz"))  # written on another machine
    assert_same(fixture, generate_in_subprocess(tmp_path / "ours.npz"), exact=False)


@pytest.mark.skipif(importlib.util.find_spec("tabicl") is None, reason="tabicl not installed")
def test_scaled_tables_are_installed_tabicls(tmp_path):
    tabicl = generate_in_subprocess(tmp_path / "tabicl.npz", tabicl=True)
    assert_same(tabicl, generate_in_subprocess(tmp_path / "ours.npz"))


# --- raw tables -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("regression", [False, True])
def test_raw_table_is_the_scaled_one_before_clamp_and_zscore(regression):
    settings = dict(seq_len=120, num_features=12, max_features=16, num_classes=None if regression else 4)
    seed(3)
    X_raw, y_raw, categorical_raw = GraphSCM(regression=regression, raw=True, **settings)()
    seed(3)
    X, y, categorical = GraphSCM(regression=regression, raw=False, **settings)()
    assert torch.equal(categorical_raw, categorical) and not categorical[12:].any()
    # both are per column, so they commute with the column permutation; the padding stays zero
    torch.testing.assert_close(standard_scaling(outlier_removing(X_raw[:, :12], threshold=4)), X[:, :12])
    assert not X_raw[:, 12:].any()
    if regression:
        torch.testing.assert_close(standard_scaling(outlier_removing(y_raw[:, None], threshold=4)).view(-1), y)
    else:
        assert torch.equal(y_raw, y)


def test_raw_batches_are_usable_and_reproducible():
    seed(0)
    batch = GraphPrior(**SETTINGS).get_batch()
    seed(0)
    again = GraphPrior(**SETTINGS).get_batch()
    assert all(torch.equal(a, b) for a, b in zip(batch, again, strict=True))

    assert batch.X.shape[:2] == batch.y.shape and batch.categorical.shape == (6, 15)
    assert batch.categorical.any()  # the prior does make categorical columns
    tables = zip(batch.X, batch.y, batch.num_features, batch.train_sizes, batch.categorical, strict=True)
    for X, y, d, n, categorical in tables:
        assert 0 < d <= 15 and not X[:, d:].any() and not categorical[d:].any()
        assert all(len(torch.unique(X[:, j])) > 1 for j in range(d))  # no constant column
        codes = X[:, categorical]
        assert torch.equal(codes, codes.round()) and (codes >= 0).all()  # categories as codes 0..K-1
        assert set(y[:n].tolist()) == set(y[n:].tolist()) and len(set(y.tolist())) >= 2


def test_filters_follow_the_config():
    seed(1)
    config = PriorConfig(filter_unpredictable_datasets=True, fct_types="default,tree")
    batch = GraphPrior(config=config, **SETTINGS).get_batch()
    assert batch.X.shape[0] == 6 and (batch.num_features > 0).all()


# --- helpers ----------------------------------------------------------------------------------------------------


def test_drop_constant_columns_keeps_the_mask_aligned():
    X = torch.tensor([[1.0, 5.0, 0.0, 2.0, 0.0], [2.0, 5.0, 1.0, 2.0, 0.0], [3.0, 5.0, 2.0, 2.0, 0.0]])
    categorical = torch.tensor([False, True, True, False, False])
    X_new, kept, categorical_new = drop_constant_columns(X, 4, categorical)  # the last column is padding
    assert kept == 2 and X_new.shape == X.shape
    assert torch.equal(X_new[:, :2], X[:, [0, 2]]) and not X_new[:, 2:].any()
    assert categorical_new.tolist() == [False, True, False, False, False]


def test_fix_split_shuffles_until_both_sides_hold_every_class():
    X, y = torch.arange(8.0)[:, None], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    torch.manual_seed(0)
    fixed = fix_split(X, y, train_size=4)
    assert fixed is not None and set(fixed[1][:4].tolist()) == {0, 1} == set(fixed[1][4:].tolist())
    assert torch.equal(fixed[1], (fixed[0][:, 0] >= 4).long())  # each row moved with its label
    assert fix_split(X, torch.zeros(8, dtype=torch.long), train_size=4) is None  # one class: never valid
