"""Classifier and Regressor give the same outputs as the reference implementation they were cleaned from
(TFM-Playground's NanoTabPFNClassifier/NanoTabPFNRegressor): tests/fixtures/parity_inference.npz holds a table
with missing values, categorical and constant columns, and the reference outputs (written by
tests/fixtures/make_inference_fixture.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tabomics import Classifier, Regressor
from tabomics.model import TabOmicsConfig, TabOmicsModel

FIXTURES = Path(__file__).parent / "fixtures"
SMALL = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2)
SETTINGS = {
    "inferred": dict(),
    "declared": dict(categorical_features=[1, 4], infer_categorical=False),
    "permuted": dict(categorical_features=[1, 4], infer_categorical=False, permute_columns=True, column_seed=3),
}


def model_from(data, prefix: str, num_outputs: int) -> TabOmicsModel:
    model = TabOmicsModel(TabOmicsConfig(**SMALL, num_outputs=num_outputs))
    weights = {k.split("::", 1)[1]: torch.from_numpy(data[k]) for k in data.files if k.startswith(prefix)}
    model.load_state_dict(weights)
    return model.eval()


@pytest.fixture(scope="module")
def data():
    return np.load(FIXTURES / "parity_inference.npz")


@pytest.fixture(scope="module")
def classifier_model():
    return model_from(np.load(FIXTURES / "parity_model.npz"), "weight::", 10)


@pytest.mark.parametrize("setting", SETTINGS)
def test_classifier_matches_the_reference(data, classifier_model, setting):
    X, y, n = data["X"], data["y_class"], int(data["n_train"])
    clf = Classifier(model=classifier_model, device="cpu", **SETTINGS[setting]).fit(X[:n], y[:n])
    np.testing.assert_allclose(clf.predict_proba(X[n:]), data[f"{setting}_proba"], rtol=0, atol=1e-6)
    assert (clf.predict(X[n:]) == data[f"{setting}_predict"]).all()
    np.testing.assert_allclose(clf.embeddings(X[n:]), data[f"{setting}_embeddings"], rtol=0, atol=1e-5)
    importance = clf.feature_importance(X[n:], rows="all")  # the reference averages over all rows
    np.testing.assert_allclose(importance, data[f"{setting}_importance"], rtol=0, atol=1e-6)


def test_regressor_matches_the_reference(data):
    X, y, n = data["X"], data["y_reg"], int(data["n_train"])
    model = model_from(data, "regression_weight::", 20)
    borders = torch.from_numpy(data["regression_borders"])
    reg = Regressor(model=model, bucket_borders=borders, device="cpu").fit(X[:n], y[:n])
    np.testing.assert_allclose(reg.predict(X[n:]), data["regression_predict"], rtol=0, atol=1e-5)
