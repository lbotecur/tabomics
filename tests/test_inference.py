import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from tabomics import Classifier, Regressor, out_of_fold_embeddings
from tabomics.inference import FeaturePreprocessor
from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.model.bar_distribution import bar_distribution_mean
from tabomics.model.io import load_checkpoint

SMALL = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2)


def small_model(num_outputs=10, **config) -> TabOmicsModel:
    torch.manual_seed(0)
    return TabOmicsModel(TabOmicsConfig(**SMALL, num_outputs=num_outputs, **config))


def data(n=80, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.where(X[:, 0] > 0, "yes", "no")
    return X, y


# --- preprocessing ---------------------------------------------------------------------------------------------


def test_preprocessor_types_columns_and_drops_constants():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=40),
            "few": rng.integers(0, 3, 40),  # numeric with 3 values: inferred categorical
            "text": rng.choice(["a", "b"], 40),  # not numeric: categorical
            "const": 1.0,  # dropped
            "num_str": [str(v) for v in rng.normal(size=40)],  # numbers stored as strings: numeric
        }
    )
    pre = FeaturePreprocessor().fit(X)
    assert pre.numeric_mask_.tolist() == [True, False, False, False, True]
    assert pre.categorical_mask_.tolist() == [False, True, True, False, False]
    out = pre.transform(X)
    assert out.shape == (40, 4)  # numeric first (num, num_str), then categorical codes (few, text)
    assert pre.original_columns_.tolist() == [0, 4, 1, 2]
    assert set(np.unique(out[:, 3])) == {0.0, 1.0}


def test_unknown_category_becomes_missing_and_small_tables_stay_numeric():
    X = pd.DataFrame({"c": ["a", "b", "a", "b"], "x": [1.0, 2.0, 1.0, 2.0]})
    pre = FeaturePreprocessor().fit(X)
    assert pre.numeric_mask_.tolist() == [False, True]  # 4 rows: too few to infer "x" as categorical
    assert np.isnan(pre.transform(pd.DataFrame({"c": ["z"], "x": [1.0]}))[0, 1])


def test_declared_categorical_and_strict_mode():
    X = np.column_stack([np.arange(40) % 3, np.linspace(0, 1, 40)])
    pre = FeaturePreprocessor(categorical_features=[1], infer_categorical=False).fit(X)
    assert pre.categorical_mask_.tolist() == [False, True]  # declared wins, inference off
    with pytest.raises(ValueError, match="not numeric"):
        FeaturePreprocessor(infer_categorical=False).fit(pd.DataFrame({"t": ["a", "b"] * 20}))
    with pytest.raises(ValueError, match="out of range"):
        FeaturePreprocessor(categorical_features=[5]).fit(X)


def test_permutation_maps_back_to_original_columns():
    X, _ = data()
    plain = FeaturePreprocessor().fit(X)
    shuffled = FeaturePreprocessor(permute_columns=True, column_seed=1).fit(X)
    assert sorted(shuffled.original_columns_) == list(range(6))
    assert np.array_equal(shuffled.transform(X), X[:, shuffled.original_columns_])
    assert np.array_equal(plain.transform(X), X)


# --- classifier and regressor ----------------------------------------------------------------------------------


def test_classifier_basics():
    X, y = data()
    clf = Classifier(model=small_model(), device="cpu").fit(X[:60], y[:60])
    proba = clf.predict_proba(X[60:])
    assert proba.shape == (20, 2) and np.allclose(proba.sum(axis=1), 1.0)
    assert set(clf.predict(X[60:])) <= {"no", "yes"} and list(clf.classes_) == ["no", "yes"]
    assert clf.embeddings(X[60:]).shape == (20, 16)
    assert 0.0 <= clf.score(X[60:], y[60:]) <= 1.0  # from ClassifierMixin


def test_feature_importance_has_one_score_per_column_and_nan_for_constant_ones():
    X, y = data()
    X[:, 2] = 3.0
    scores = Classifier(model=small_model(), device="cpu").fit(X[:60], y[:60]).feature_importance(X[60:])
    assert scores.shape == (6,) and np.isnan(scores[2]) and np.isfinite(np.delete(scores, 2)).all()


def test_settings_from_the_model_and_overrides():
    X, y = data()
    model = small_model(permute_columns=True, feature_clip=10.0)
    clf = Classifier(model=model, device="cpu").fit(X[:60], y[:60])
    assert not np.array_equal(clf.preprocessor_.original_columns_, np.arange(6))  # the model's own setting
    clf = Classifier(model=model, device="cpu", permute_columns=False, feature_clip=5.0).fit(X[:60], y[:60])
    assert np.array_equal(clf.preprocessor_.original_columns_, np.arange(6)) and model.feature_clip == 5.0


def test_low_memory_modes_agree_and_bad_values_are_refused():
    X, y = data()
    probas = [
        Classifier(model=small_model(), device="cpu", low_memory=mode).fit(X[:60], y[:60]).predict_proba(X[60:])
        for mode in (True, False, "auto")
    ]
    assert np.allclose(probas[0], probas[1], atol=1e-6) and np.allclose(probas[1], probas[2], atol=1e-6)
    with pytest.raises(ValueError, match="low_memory"):
        Classifier(model=small_model(), device="cpu", low_memory="sometimes").fit(X[:60], y[:60])


def test_errors_for_missing_model_and_too_many_classes():
    X, _ = data()
    with pytest.raises(ValueError, match="No model"):
        Regressor(device="cpu", bucket_borders=torch.linspace(-1, 1, 5)).fit(X, np.arange(80) % 2)
    with pytest.raises(ValueError, match="at most 10 classes"):
        Classifier(model=small_model(), device="cpu").fit(X, np.arange(80) % 11)


def test_scikit_learn_clone_and_pickle():
    X, y = data()
    clf = Classifier(model=small_model(), device="cpu", column_seed=4)
    assert clone(clf).get_params()["column_seed"] == 4
    fitted = clf.fit(X[:60], y[:60])
    restored = pickle.loads(pickle.dumps(fitted))
    assert np.allclose(restored.predict_proba(X[60:]), fitted.predict_proba(X[60:]))


def test_regressor_predicts_on_the_target_scale():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = 100.0 + 10.0 * X[:, 0]
    borders = torch.linspace(-3, 3, 21)
    reg = Regressor(model=small_model(num_outputs=20), bucket_borders=borders, device="cpu").fit(X[:40], y[:40])
    pred = reg.predict(X[40:])
    assert pred.shape == (20,) and np.all((pred > 50) & (pred < 150))  # around the training mean, original scale
    with pytest.raises(ValueError, match="bucket borders"):
        Regressor(model=small_model(num_outputs=20), device="cpu").fit(X, y)


def test_out_of_fold_embeddings_cover_every_training_row():
    X, y = data(n=50)
    clf = Classifier(model=small_model(), device="cpu")
    embeddings = out_of_fold_embeddings(clf, X, y, n_folds=5)
    assert embeddings.shape == (50, 16) and np.isfinite(embeddings).all()
    assert len(clf.X_train_) == 50  # refitted on all the rows


# --- model files and the bar distribution ----------------------------------------------------------------------


def test_load_checkpoint_reads_the_training_format(tmp_path):
    model = small_model()
    architecture = dict(SMALL, num_outputs=10)  # no feature_clip: written before it was stored
    torch.save({"architecture": architecture, "model": model.state_dict(), "epoch": 3}, tmp_path / "m.pth")
    loaded = load_checkpoint(tmp_path / "m.pth")
    assert loaded.feature_clip == 100.0 and not loaded.training
    x, y = torch.randn(1, 30, 5), torch.randint(0, 2, (1, 20)).float()
    with torch.no_grad():
        assert torch.equal(loaded(x, y, 20), model.eval()(x, y, 20))


def test_bar_distribution_mean():
    borders = torch.linspace(-2.0, 2.0, 5)  # buckets of width 1
    one_hot = torch.full((4,), -1e9)
    one_hot[1] = 0.0  # all the mass in the second bucket, [-1, 0]
    assert torch.isclose(bar_distribution_mean(one_hot, borders), torch.tensor(-0.5))
    assert torch.isclose(bar_distribution_mean(torch.zeros(4), borders), torch.tensor(0.0), atol=1e-6)  # symmetric
    left = torch.full((4,), -1e9)
    left[0] = 0.0  # the left tail: a half-normal below -1 with median at distance 1
    expected = -1.0 - (1.0 / 0.6744897) * np.sqrt(2 / np.pi)
    assert torch.isclose(bar_distribution_mean(left, borders), torch.tensor(expected, dtype=torch.float32))
