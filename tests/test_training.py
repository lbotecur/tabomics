"""Pretraining: the same losses and weights as TFM-Playground's train(), where TabOmics was developed
(tests/fixtures/parity_training.npz, written by tests/fixtures/make_training_fixture.py); resuming continues a run
exactly; recipes and the training script."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tabomics import Classifier
from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.model.io import load_model
from tabomics.prior import AugmentationConfig
from tabomics.training import Recipe, load_training_state, train

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
from training_cases import MISSING_RATE_MAX, SEED, SETTINGS, SMALL, WIDENING, Batches, batches  # noqa: E402

TINY = {
    "model": {"embedding_size": 16, "num_attention_heads": 2, "mlp_hidden_size": 32, "num_layers": 2},
    "prior": {"min_rows": 40, "max_rows": 100, "max_features": 8},
    "augmentation": {"add_features_max": 30, "missing_rate_max": 0.1},
    "training": {"epochs": 2, "steps_per_epoch": 4, "batch_size": 2, "accumulate": 2, "lr": 1e-3, "warmup_steps": 2,
                 "amp_dtype": None, "seed": 1, "snapshot_every": 1, "log_every": 2},
}  # fmt: skip


def test_same_losses_and_weights_as_tfm_playground(tmp_path):
    expected = np.load(FIXTURES / "parity_training.npz")
    torch.manual_seed(SEED)
    model = TabOmicsModel(TabOmicsConfig(**SMALL))
    model.gradient_checkpointing = True
    augmentation = AugmentationConfig(**WIDENING, missing_rate_max=MISSING_RATE_MAX)
    train(model, Batches(batches()), tmp_path, augmentation=augmentation, log_every=1, **SETTINGS)
    losses = pd.read_csv(tmp_path / "steps.csv")["loss"].to_numpy()
    np.testing.assert_allclose(losses, expected["losses"], atol=2e-5)  # written with 5 decimals
    state = torch.load(tmp_path / "latest_checkpoint.pth", weights_only=True)["model"]
    for key in [k for k in expected.files if k.startswith("weight::")]:
        # bit for bit on one machine with one thread setting; elsewhere rounding differs (~1e-5 after 4 steps)
        np.testing.assert_allclose(state[key.split("::", 1)[1]].numpy(), expected[key], rtol=1e-3, atol=1e-4)


def test_resuming_continues_the_run_exactly(tmp_path):
    recipe = Recipe.from_dict(TINY)

    def run(directory, epochs, resume=None):
        torch.manual_seed(1)
        model = TabOmicsModel(recipe.model)
        if resume is not None:
            model.load_state_dict(resume["model"])
        loader = recipe.make_loader(epoch=resume["epoch"] if resume else 0)
        train(model, loader, directory, epochs, lr=1e-3, accumulate=2, augmentation=recipe.augmentation,
              resume=resume)  # fmt: skip
        return model.state_dict()

    whole = run(tmp_path / "whole", 2)
    run(tmp_path / "cut", 1)
    resumed = run(tmp_path / "cut", 2, load_training_state(tmp_path / "cut" / "latest_checkpoint.pth"))
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)


def test_accumulation_must_divide_the_epoch(tmp_path):
    with pytest.raises(ValueError, match="accumulate"):
        train(TabOmicsModel(TabOmicsConfig(**SMALL)), Batches(batches()[:3]), tmp_path, 1, accumulate=2)


def test_a_group_cut_short_by_a_skipped_batch_still_steps(tmp_path, capsys):
    data = batches()
    data[1] = {**data[1], "y": data[1]["y"].clone()}
    data[1]["y"][0, 5] = float("nan")  # a missing training label: the batch is skipped
    train(TabOmicsModel(TabOmicsConfig(**SMALL)), Batches(data), tmp_path, 1, accumulate=2, lr=1e-3)
    # batches 0 and 2 make one step, batch 3 alone the other (its gradients would otherwise be dropped)
    optimizer = torch.load(tmp_path / "latest_checkpoint.pth", weights_only=False)["optimizer"]
    assert optimizer["param_groups"][0]["k"] == 2
    assert "Skipped 1 (missing labels)" in capsys.readouterr().out


def test_a_short_group_steps_as_a_mean_over_its_batches(tmp_path):
    # the short group [b] of accumulate=2 must give the step of a full group [b, b]
    b = batches()[0]
    skipped = {**b, "y": b["y"].clone()}
    skipped["y"][0, 0] = float("nan")

    def weights(data, directory):
        torch.manual_seed(0)
        model = TabOmicsModel(TabOmicsConfig(**SMALL))
        train(model, Batches(data), tmp_path / directory, 1, accumulate=2, lr=1e-3)
        return model.state_dict()

    short, full = weights([b, skipped], "short"), weights([b, b], "full")
    assert all(torch.allclose(short[k], full[k], atol=1e-6) for k in full)


def test_recipes_load_override_and_reject_typos(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(TINY))
    recipe = Recipe.load(path, ["training.lr=3e-4", "model.num_layers=3", "init_from=runs/a.pth"])
    assert recipe.training.lr == 3e-4 and recipe.model.num_layers == 3 and recipe.init_from == "runs/a.pth"
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    with pytest.raises(ValueError, match="Unknown keys"):
        Recipe.load(path, ["training.learning_rate=1e-3"])
    for config in sorted((REPO / "configs").rglob("*.json")):  # the shipped recipes are valid
        Recipe.load(config)


def test_train_script_end_to_end(tmp_path):
    config = tmp_path / "tiny.json"
    config.write_text(json.dumps(TINY))
    run_dir = tmp_path / "run"
    command = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(config), "--run-dir", str(run_dir),
               "--device", "cpu"]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert {"latest_checkpoint.pth", "epoch_1.pth", "epoch_2.pth", "steps.csv", "recipe.json"} <= {
        p.name for p in run_dir.iterdir()
    }
    first = load_model(run_dir / "latest_checkpoint.pth").state_dict()
    # the same command again from scratch: the same model (seeded prior, weights and augmentations)
    subprocess.run([*command[:5], str(tmp_path / "again"), *command[6:]], check=True, capture_output=True)
    again = load_model(tmp_path / "again" / "latest_checkpoint.pth").state_dict()
    assert all(torch.equal(first[k], again[k]) for k in first)
    # the checkpoint is usable as it is
    X, y = np.random.default_rng(0).normal(size=(30, 5)), np.arange(30) % 2
    proba = Classifier(model=str(run_dir / "latest_checkpoint.pth"), device="cpu").fit(X[:20], y[:20])
    assert proba.predict_proba(X[20:]).shape == (10, 2)
