import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tabomics import Classifier, hub
from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.model.io import CONFIG_FILE, WEIGHTS_FILE, load_model, load_pretrained, save_pretrained

SMALL = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
REPO = Path(__file__).resolve().parents[1]


def small_model(**config) -> TabOmicsModel:
    torch.manual_seed(0)
    return TabOmicsModel(TabOmicsConfig(**SMALL, **config)).eval()


def inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 30, 5, generator=g), torch.randint(0, 2, (1, 20), generator=g).float()


def test_release_round_trip_is_exact(tmp_path):
    model = small_model(feature_clip=10.0, permute_columns=True)
    save_pretrained(model, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [CONFIG_FILE, WEIGHTS_FILE]
    config = json.loads((tmp_path / CONFIG_FILE).read_text())
    assert config["task"] == "classification" and config["permute_columns"] is True
    for source in (tmp_path, tmp_path / WEIGHTS_FILE):
        loaded = load_pretrained(source)
        assert loaded.config == model.config
        x, y = inputs()
        with torch.no_grad():
            assert torch.equal(loaded(x, y, 20), model(x, y, 20))


def test_load_model_dispatches_on_the_source(tmp_path):
    model = small_model()
    assert load_model(model) is model
    save_pretrained(model, tmp_path / "release")
    torch.save({"architecture": dict(SMALL, feature_clip=10.0), "model": model.state_dict()}, tmp_path / "m.pth")
    x, y = inputs()
    with torch.no_grad():
        for source in (tmp_path / "release", str(tmp_path / "m.pth")):
            assert torch.equal(load_model(source)(x, y, 20), model(x, y, 20))
    with pytest.raises(FileNotFoundError, match="published model"):
        load_model("no_such_model")


@pytest.fixture
def published(tmp_path, monkeypatch):
    """Fake published models "v2" and the default: hf_hub_download serves the files of a local release."""
    model = small_model()
    release = save_pretrained(model, tmp_path / "hub")
    calls = []

    def fake_download(repo_id, filename, revision=None, cache_dir=None):
        calls.append((repo_id, filename, revision))
        return str(release / filename)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    spec = hub.PublishedModel("lbotecur/tabomics-v2", "abc123", hub.sha256_of(release / WEIGHTS_FILE), "c", "")
    monkeypatch.setitem(hub.PUBLISHED_MODELS, "v2", spec)
    monkeypatch.setitem(hub.PUBLISHED_MODELS, hub.DEFAULT_CLASSIFIER, spec)
    return model, calls


def test_download_pins_the_revision_and_checks_the_weights(published, monkeypatch):
    model, calls = published
    loaded = load_model("v2")
    assert {c[2] for c in calls} == {"abc123"} and {c[1] for c in calls} == {WEIGHTS_FILE, CONFIG_FILE}
    x, y = inputs()
    with torch.no_grad():
        assert torch.equal(loaded(x, y, 20), model(x, y, 20))
    wrong = hub.PublishedModel("lbotecur/tabomics-v2", "abc123", "0" * 64, "c", "")
    monkeypatch.setitem(hub.PUBLISHED_MODELS, "v2", wrong)
    with pytest.raises(RuntimeError, match="SHA-256"):
        hub.download("v2")


def test_classifier_without_a_model_uses_the_default(published):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.integers(0, 2, 40)
    proba = Classifier(device="cpu").fit(X[:30], y[:30]).predict_proba(X[30:])
    assert proba.shape == (10, 2)


def test_the_classifier_records_which_model_it_used(published, tmp_path):
    model, _ = published
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.integers(0, 2, 40)
    assert Classifier(device="cpu").fit(X, y).model_name_ == hub.DEFAULT_CLASSIFIER
    assert Classifier(model="v2", device="cpu").fit(X, y).model_name_ == "v2"
    release = save_pretrained(model, tmp_path / "mine")
    for source in (release, release / WEIGHTS_FILE):
        assert Classifier(model=source, device="cpu").fit(X, y).model_name_ == str(source.resolve())
    assert Classifier(model=model, device="cpu").fit(X, y).model_name_ == "TabOmicsModel"


def test_unknown_and_unpublished_names():
    with pytest.raises(KeyError, match="published models"):
        hub.download("v99")
    if hub.PUBLISHED_MODELS["v2"].revision is None:
        with pytest.raises(RuntimeError, match="not published"):
            hub.download("v2")


def test_published_models_are_pinned_and_the_default_is_one_of_them():
    assert hub.DEFAULT_CLASSIFIER in hub.PUBLISHED_MODELS
    for name, spec in hub.PUBLISHED_MODELS.items():
        assert spec.revision is not None and len(spec.revision) == 40, name  # a full commit, never a branch
        assert spec.sha256 is not None and len(spec.sha256) == 64, name
        assert spec.repo_id == f"lbotecur/tabomics-{name}", name


def test_export_script_writes_a_release_that_reproduces_the_checkpoint(tmp_path):
    model = small_model()
    torch.save({"architecture": dict(SMALL, feature_clip=10.0), "model": model.state_dict()}, tmp_path / "m.pth")
    out = tmp_path / "release"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export.py"), "--checkpoint", str(tmp_path / "m.pth"),
         "--output", str(out), "--name", "test", "--permute-columns"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert "sha256" in result.stdout
    assert sorted(p.name for p in out.iterdir()) == ["README.md", CONFIG_FILE, WEIGHTS_FILE]
    assert load_pretrained(out).config.permute_columns is True
    assert "license: gpl-3.0" in (out / "README.md").read_text()
