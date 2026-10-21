"""Fine-tuning: episodes never contain validation rows, keep every class in the context and permute the classes;
early stopping keeps the best weights (the pretrained ones if no epoch improves); end to end."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import tabomics.finetuning.loop as loop
from tabomics import Classifier
from tabomics.finetuning import (
    EpisodeLoader,
    FineTunedClassifier,
    MixedLoader,
    finetune,
    prepare_table,
    split_context_query,
    validation_metrics,
)
from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.model.io import load_model
from tabomics.training import PriorSettings

REPO = Path(__file__).resolve().parents[1]

TINY = {"embedding_size": 16, "num_attention_heads": 2, "mlp_hidden_size": 32, "num_layers": 2}
FAST = {"epochs": 2, "steps_per_epoch": 4, "batch_size": 2, "accumulate": 2, "device": "cpu", "lr": 1e-3}


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return TabOmicsModel(TabOmicsConfig(**TINY)).eval()


def id_table(rows=60, columns=6, classes=3):
    """Every cell of row i holds i: any value of an episode tells which row it came from."""
    X = np.repeat(np.arange(rows, dtype=float)[:, None], columns, axis=1)
    return X, np.arange(rows) % classes


def random_table(seed=0, rows=60, columns=6, classes=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, rows)
    return rng.normal(size=(rows, columns)) + y[:, None], y


def same_weights(a, b):
    return all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def test_validation_rows_are_held_out_stratified():
    X, y = id_table(rows=100)
    table = prepare_table(X, y, validation_fraction=0.2, random_state=0)
    pool, val = set(table.X[:, 0]), set(table.X_val[:, 0])
    assert not pool & val and pool | val == set(range(100))
    assert len(val) == 20 and np.bincount(table.y_val).min() >= 6  # every class in validation
    given = prepare_table(X[:80], y[:80], X_val=X[80:], y_val=y[80:])
    assert len(given.y) == 80 and len(given.y_val) == 20
    assert len(prepare_table(X, y, validation_fraction=0).y_val) == 0


def test_labels_are_encoded_and_categories_detected():
    X = pd.DataFrame({"a": np.arange(60.0), "b": ["u", "v", "w"] * 20})
    table = prepare_table(X, np.array(["x", "y"] * 30), validation_fraction=0.1)
    assert set(table.y) == {0, 1} and table.num_classes == 2
    assert table.categorical.tolist() == [False, True]


def test_episodes_never_contain_validation_rows():
    X, y = id_table(rows=80)
    table = prepare_table(X, y, validation_fraction=0.25)
    loader = EpisodeLoader([table], num_steps=50, batch_size=4, seed=1)
    seen = np.unique(np.concatenate([batch["x"].numpy().ravel() for batch in loader]))
    assert set(seen) <= set(table.X[:, 0]) and not set(seen) & set(table.X_val[:, 0])


def test_episode_shapes_classes_in_context_and_class_permutation():
    X, y = id_table(rows=60, columns=6, classes=3)
    table = prepare_table(X, y, validation_fraction=0)
    loader = EpisodeLoader([table], num_steps=40, batch_size=4, row_fraction=(0.5, 1.0), feature_fraction=(0.5, 1.0))
    labels_of_row0 = set()
    for batch in loader:
        x, labels, train_size = batch["x"].numpy(), batch["y"].numpy().astype(int), batch["train_size"]
        assert x.shape[:2] == labels.shape and batch["categorical"].shape == (x.shape[0], x.shape[2])
        assert 3 <= x.shape[2] <= 6 and 30 <= x.shape[1] <= 60
        assert abs((x.shape[1] - train_size) - 0.2 * x.shape[1]) <= 1
        for episode, episode_labels in zip(x, labels, strict=True):
            rows = episode[:, 0].astype(int)
            assert set(episode_labels[:train_size]) == {0, 1, 2}  # every class in the context
            # a permutation of the classes: rows of one class share one label, different classes different labels
            mapping = {y[r]: label for r, label in zip(rows, episode_labels, strict=True)}
            assert sorted(mapping.values()) == [0, 1, 2]
            assert all(mapping[y[r]] == label for r, label in zip(rows, episode_labels, strict=True))
            labels_of_row0 |= {label for r, label in zip(rows, episode_labels, strict=True) if r == 0}
    assert len(labels_of_row0) > 1


def test_split_keeps_a_row_of_every_class_in_the_context():
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] + [2] * 5)
    for _ in range(50):
        context, query = split_context_query(y, num_query=23, rng=rng)
        assert set(y[context]) == {0, 1, 2} and len(query) == 23
        assert sorted(np.concatenate([context, query])) == list(range(len(y)))


def test_tables_take_turns_and_seed_fixes_the_batches():
    tables = [prepare_table(*random_table(0, columns=4), 0), prepare_table(*random_table(1, columns=7), 0)]

    def widths(seed):
        loader = EpisodeLoader(tables, num_steps=6, feature_fraction=(1.0, 1.0), seed=seed)
        return [(b["x"].shape[2], b["x"].sum().item()) for b in loader]

    first = widths(3)
    assert [w for w, _ in first] == [4, 7] * 3
    assert first == widths(3) and first != widths(4)


def test_a_table_needs_more_rows_than_classes():
    table = prepare_table(np.zeros((3, 2)) + np.arange(3)[:, None], np.arange(3), validation_fraction=0)
    with pytest.raises(ValueError, match="more pool rows"):
        EpisodeLoader([table], num_steps=1)


def test_mixed_loader_replaces_a_share_of_the_batches():
    table = prepare_table(*random_table(), validation_fraction=0)
    episodes = EpisodeLoader([table], num_steps=200, batch_size=1)
    prior = [{"x": torch.zeros(1, 5, 2), "y": torch.zeros(1, 5), "train_size": 3, "categorical": None}] * 7
    batches = list(MixedLoader(episodes, prior, prior_fraction=0.3))
    synthetic = sum(b["x"].shape[1] == 5 for b in batches)
    assert len(batches) == 200 and 40 < synthetic < 80 and episodes.step == 200 - synthetic


def test_validation_loss_is_the_classifier_s():
    model = tiny_model()
    X, y = random_table(rows=80)
    table = prepare_table(X[:60], y[:60], X_val=X[60:], y_val=y[60:])
    loss, accuracy = validation_metrics(model, [table])
    clf = Classifier(model=model, device="cpu", permute_columns=False).fit(X[:60], y[:60])
    proba = clf.predict_proba(X[60:])
    np.testing.assert_allclose(loss, -np.log(proba[np.arange(20), y[60:]]).mean(), rtol=1e-5)
    np.testing.assert_allclose(accuracy, (proba.argmax(1) == y[60:]).mean())


def scripted_validation(monkeypatch, losses):
    """Replaces the validation loss by a script; records the weights at each evaluation."""
    states = []

    def fake(model, tables, *args, **kwargs):
        states.append(loop._copy_state(model))
        return losses[len(states) - 1], 0.5

    monkeypatch.setattr(loop, "validation_metrics", fake)
    return states


def test_early_stopping_keeps_the_best_epoch(monkeypatch):
    states = scripted_validation(monkeypatch, [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.1, 0.1])
    result = finetune(tiny_model(), [random_table()], patience=3, **{**FAST, "epochs": 10})
    assert result.best_epoch == 2 and result.history["epoch"].tolist() == [0, 1, 2, 3, 4, 5]  # stopped at 5
    assert same_weights(result.model.state_dict(), states[2])
    assert not same_weights(states[2], states[0])


def test_the_pretrained_model_is_returned_if_no_epoch_improves(monkeypatch):
    scripted_validation(monkeypatch, [1.0, 1.0 - 1e-6, 1.1, 1.2])  # 1e-6 is below min_delta
    pretrained = tiny_model()
    original = {k: v.clone() for k, v in pretrained.state_dict().items()}
    result = finetune(pretrained, [random_table()], patience=3, **{**FAST, "epochs": 10})
    assert result.best_epoch == 0 and len(result.history) == 4
    assert same_weights(result.model.state_dict(), original)
    assert same_weights(pretrained.state_dict(), original)  # the model given is not modified


def test_frozen_parameters_do_not_change():
    pretrained = tiny_model()
    result = finetune(pretrained, [random_table()], validation_fraction=0, freeze=["feature_encoder"], **FAST)
    before, after = pretrained.state_dict(), result.model.state_dict()
    frozen = [k for k in before if k.startswith("feature_encoder")]
    assert frozen and all(torch.equal(before[k], after[k]) for k in frozen)
    assert not all(torch.equal(before[k], after[k]) for k in before if k not in frozen)
    assert result.best_epoch == FAST["epochs"]  # no validation: the last weights
    with pytest.raises(ValueError, match="No parameter"):
        finetune(pretrained, [random_table()], freeze=["encoder"], **FAST)


def test_finetune_end_to_end(tmp_path):
    tables = [random_table(0), random_table(1, columns=9, classes=2)]
    result = finetune(tiny_model(), tables, output=tmp_path, seed=5, **FAST)
    again = finetune(tiny_model(), tables, seed=5, **FAST)
    assert same_weights(result.model.state_dict(), again.model.state_dict())  # the seed fixes the run
    assert list(result.history.columns) == ["epoch", "train_loss", "val_loss", "val_accuracy"]
    assert {p.name for p in tmp_path.iterdir()} == {"finetuned.pth", "history.csv", "steps.csv"}
    assert same_weights(load_model(tmp_path / "finetuned.pth").state_dict(), result.model.state_dict())
    X, y = random_table(2)
    proba = Classifier(model=str(tmp_path / "finetuned.pth"), device="cpu").fit(X[:40], y[:40]).predict_proba(X[40:])
    assert proba.shape == (20, 3)


def test_finetune_with_synthetic_tables(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    prior = PriorSettings(min_rows=20, max_rows=40, max_features=5, max_classes=3)
    result = finetune(tiny_model(), [random_table()], prior_fraction=0.5, prior=prior, **FAST)
    assert len(result.history) == FAST["epochs"] + 1
    with pytest.raises(ValueError, match="max_classes"):
        finetune(tiny_model(), [random_table()], prior_fraction=0.5, prior=PriorSettings(max_classes=20), **FAST)


def test_finetuned_classifier_is_finetune_then_classifier():
    X, y = random_table(rows=80)
    settings = {k: v for k, v in FAST.items() if k != "device"}
    clf = FineTunedClassifier(model=tiny_model(), device="cpu", random_state=3, **settings).fit(X[:60], y[:60])
    result = finetune(tiny_model(), [(X[:60], y[:60])], seed=3, verbose=False, **FAST)
    reference = Classifier(model=result.model, device="cpu").fit(X[:60], y[:60])
    np.testing.assert_allclose(clf.predict_proba(X[60:]), reference.predict_proba(X[60:]), rtol=1e-5, atol=1e-6)
    assert clf.best_epoch_ == result.best_epoch and len(clf.finetune_history_) == len(result.history)


def test_finetuned_classifier_is_a_scikit_learn_estimator():
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    X, y = random_table(rows=60)
    clf = FineTunedClassifier(model=tiny_model(), device="cpu", epochs=1, steps_per_epoch=2, accumulate=2, lr=1e-3)
    assert clone(clf).get_params()["lr"] == 1e-3
    proba = cross_val_predict(clf, X, y, cv=3, method="predict_proba")
    assert proba.shape == (60, 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)


def test_finetune_script_end_to_end(tmp_path):
    for i in range(2):
        X, y = random_table(seed=i)
        table = pd.DataFrame(X, columns=[f"g{j}" for j in range(X.shape[1])]).assign(label=y)
        table.to_csv(tmp_path / f"t{i}.csv", index=False)
    release = tmp_path / "tiny"
    from tabomics.model.io import save_pretrained

    save_pretrained(tiny_model(), release)
    command = [sys.executable, str(REPO / "scripts" / "finetune.py"), "--model", str(release), "--data",
               str(tmp_path / "t0.csv"), str(tmp_path / "t1.csv"), "--output", str(tmp_path / "out"), "--device",
               "cpu", "--epochs", "2", "--steps-per-epoch", "4", "--accumulate", "2", "--lr", "1e-3"]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert {"finetuned.pth", "history.csv", "steps.csv"} <= {p.name for p in (tmp_path / "out").iterdir()}
    assert len(pd.read_csv(tmp_path / "out" / "history.csv")) >= 2


def test_finetune_script_declares_categorical_columns_per_table(tmp_path):
    for i in range(2):
        X, y = random_table(seed=i)
        table = pd.DataFrame(X, columns=[f"g{j}" for j in range(X.shape[1])]).assign(label=y)
        table.to_csv(tmp_path / f"t{i}.csv", index=False)
    release = tmp_path / "tiny"
    from tabomics.model.io import save_pretrained

    save_pretrained(tiny_model(), release)
    base = [sys.executable, str(REPO / "scripts" / "finetune.py"), "--model", str(release), "--data",
            str(tmp_path / "t0.csv"), str(tmp_path / "t1.csv"), "--output", str(tmp_path / "out"), "--device", "cpu",
            "--epochs", "1", "--steps-per-epoch", "2", "--accumulate", "1", "--lr", "1e-3"]  # fmt: skip
    result = subprocess.run([*base, "--categorical", "0,1", "none", "--no-infer-categorical"],
                            capture_output=True, text=True)  # fmt: skip
    assert result.returncode == 0, result.stderr[-3000:]
    lines = [line for line in result.stdout.splitlines() if line.endswith("categorical")]
    assert lines[0].endswith(" 2 categorical") and lines[1].endswith(" 0 categorical")
    wrong = subprocess.run([*base, "--categorical", "0,1"], capture_output=True, text=True)
    assert wrong.returncode != 0 and "one entry per table" in wrong.stderr
