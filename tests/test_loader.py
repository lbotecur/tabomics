"""The prior loader: usable batches only, a fixed sequence of tables with a seed (with or without workers, and when
a run is resumed), each epoch different."""

import os

import pytest
import torch

from tabomics.prior import PriorBatch, PriorLoader
from tabomics.prior.loader import usable_batch

SETTINGS = dict(num_steps=3, batch_size=3, min_rows=40, max_rows=100, min_features=2, max_features=10, max_classes=4)


def epochs(loader: PriorLoader, n: int) -> list[list[dict]]:
    return [list(loader) for _ in range(n)]


def same(a: list[dict], b: list[dict]) -> bool:
    return len(a) == len(b) and all(
        torch.equal(p["x"], q["x"]) and torch.equal(p["y"], q["y"]) and p["train_size"] == q["train_size"]
        and torch.equal(p["categorical"], q["categorical"])
        for p, q in zip(a, b, strict=True)
    )  # fmt: skip


@pytest.fixture(autouse=True)
def hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # no warning (this process's hash seed is what it is)


def test_batches_are_usable_and_shaped():
    batches = list(PriorLoader(**SETTINGS, seed=0))
    assert len(batches) == 3
    for batch in batches:
        x, y, n, categorical = batch["x"], batch["y"], batch["train_size"], batch["categorical"]
        assert x.ndim == 3 and y.shape == x.shape[:2] and categorical.shape == (x.shape[0], x.shape[2])
        assert y.dtype == torch.float32 and (y >= 0).all() and 0 < n < x.shape[1]


def test_a_seed_fixes_every_epoch_and_resuming_continues_the_sequence():
    first = epochs(PriorLoader(**SETTINGS, seed=3), 2)
    again = epochs(PriorLoader(**SETTINGS, seed=3), 2)
    assert same(first[0], again[0]) and same(first[1], again[1])
    assert not same(first[0], first[1])  # each epoch has its own tables
    resumed = list(PriorLoader(**SETTINGS, seed=3, epoch=1))  # a run resumed after its first epoch
    assert same(resumed, first[1])
    assert not same(first[0], list(PriorLoader(**SETTINGS, seed=4)))


def test_workers_share_the_steps_and_are_reproducible():
    with_workers = epochs(PriorLoader(**SETTINGS, seed=3, num_workers=2), 2)
    assert [len(e) for e in with_workers] == [3, 3]
    assert same(with_workers[1], epochs(PriorLoader(**SETTINGS, seed=3, num_workers=2), 2)[1])


def test_unusable_tables_are_dropped_and_the_batch_cropped():
    X = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    batch = PriorBatch(
        X=torch.cat([X, X[:1]]), y=torch.tensor([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [-100] * 5]),
        num_features=torch.tensor([2, 3, 4]), seq_lens=torch.tensor([5, 5, 5]), train_sizes=torch.tensor([3, 3, 3]),
        categorical=torch.tensor([[True, False, False, False]] * 3),
    )  # fmt: skip
    usable = usable_batch(batch)
    assert usable["x"].shape == (2, 5, 3) and usable["train_size"] == 3  # the failed table is gone
    assert usable["categorical"].shape == (2, 3)
    empty = batch._replace(num_features=torch.tensor([0, 0, 4]))
    assert usable_batch(empty) is None  # only the failed table had columns


def test_warns_without_a_hash_seed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.warns(UserWarning, match="PYTHONHASHSEED"):
        PriorLoader(**SETTINGS, seed=0)
    assert "PYTHONHASHSEED" not in os.environ
