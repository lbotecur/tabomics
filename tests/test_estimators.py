import numpy as np
import pytest
import torch

from tabomics import Classifier, Regressor
from tabomics.inference.base import EnsembleMember
from tabomics.model import TabOmicsConfig, TabOmicsModel

SMALL = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2)


def small_model(num_outputs=10) -> TabOmicsModel:
    torch.manual_seed(0)
    return TabOmicsModel(TabOmicsConfig(**SMALL, num_outputs=num_outputs))


def table(n=90, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 1] = rng.integers(0, 4, n)  # categorical (inferred: 4 values)
    X[:, 5] = 2.0  # constant
    X[rng.random((n, 6)) < 0.05] = np.nan
    X[:, 5] = 2.0
    y = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    return X, y


def fit(**kwargs):
    X, y = table()
    return Classifier(model=small_model(), device="cpu", **kwargs).fit(X[:60], y[:60]), X[60:]


def test_member_relabels_codes_and_columns_and_keeps_missing_values():
    member = EnsembleMember(columns=np.array([2, 0, 1]), codes={1: np.array([2, 0, 1])})
    X = np.array([[5.0, 0.0, 7.0], [6.0, np.nan, 8.0], [9.0, 2.0, 1.0]])
    out = member.inputs(X)
    assert np.array_equal(out[:, 0], X[:, 2]) and np.array_equal(out[:, 1], X[:, 0])
    assert out[0, 2] == 2.0 and np.isnan(out[1, 2]) and out[2, 2] == 1.0


def test_one_estimator_is_the_plain_pass():
    clf, X_test = fit()
    assert len(clf.members_) == 1 and clf.members_[0].classes is None and not clf.members_[0].codes
    assert np.array_equal(clf.members_[0].columns, np.arange(len(clf.preprocessor_.original_columns_)))


def test_members_are_fixed_by_the_seed():
    a, X_test = fit(n_estimators=4, random_state=7)
    b, _ = fit(n_estimators=4, random_state=7)
    c, _ = fit(n_estimators=4, random_state=8)
    assert np.array_equal(a.predict_proba(X_test), b.predict_proba(X_test))
    assert not np.allclose(a.predict_proba(X_test), c.predict_proba(X_test))
    categorical = int(np.flatnonzero(a.preprocessor_.category_counts_ == 4)[0])
    assert all(sorted(m.codes[categorical]) == [0, 1, 2, 3] for m in a.members_[1:])


def test_probabilities_are_the_mean_of_the_passes_brought_back_to_the_classes():
    single, X_test = fit()
    pair, _ = fit(n_estimators=2, random_state=3)
    member = pair.members_[1]
    # the second pass, by hand: relabelled inputs and classes, then the classes put back in their order
    x = np.concatenate([member.inputs(pair.X_train_), member.inputs(pair.preprocessor_.transform(X_test))])
    y = member.classes[pair.y_train_]
    with torch.no_grad():
        logits = pair.model_(torch.tensor(x[None], dtype=torch.float), torch.tensor(y[None], dtype=torch.float), 60)
    second = torch.softmax(logits[0, :, :3], dim=1).numpy()[:, member.classes]
    expected = (single.predict_proba(X_test) + second) / 2
    assert np.allclose(pair.predict_proba(X_test), expected, atol=1e-6)
    assert np.allclose(pair.predict_proba(X_test).sum(axis=1), 1.0)


def test_feature_importance_averages_over_positions():
    clf, X_test = fit(n_estimators=5)
    scores = clf.feature_importance(X_test)
    assert scores.shape == (6,) and np.isnan(scores[5]) and np.isfinite(scores[:5]).all()
    single, _ = fit()
    assert not np.allclose(scores[:5], single.feature_importance(X_test)[:5])


def test_feature_importance_rows():
    clf, X_test = fit(n_estimators=2)
    test, train, every = (clf.feature_importance(X_test, rows=rows) for rows in ("test", "train", "all"))
    assert np.array_equal(test, clf.feature_importance(X_test), equal_nan=True)  # test rows by default
    n_train, n_test = len(clf.X_train_), len(X_test)
    np.testing.assert_allclose(every[:5], (n_train * train[:5] + n_test * test[:5]) / (n_train + n_test), atol=1e-6)
    assert np.isnan(train[5]) and np.isnan(every[5])
    with pytest.raises(ValueError, match="must be one of"):
        clf.feature_importance(X_test, rows="validation")


def test_embeddings_mean_and_concat():
    clf, X_test = fit(n_estimators=3)
    mean, concat = clf.embeddings(X_test), clf.embeddings(X_test, reduce="concat")
    assert mean.shape == (30, 16) and concat.shape == (30, 48)
    assert np.allclose(mean, concat.reshape(30, 3, 16).mean(axis=1), atol=1e-6)
    with pytest.raises(ValueError, match="reduce"):
        clf.embeddings(X_test, reduce="max")


def test_regressor_and_bad_values():
    X, _ = table()
    y = np.nan_to_num(X[:, 0]) * 3.0
    reg = Regressor(model=small_model(20), bucket_borders=torch.linspace(-3, 3, 21), device="cpu", n_estimators=3)
    assert reg.fit(X[:60], y[:60]).predict(X[60:]).shape == (30,)
    assert all(m.classes is None for m in reg.members_)
    with pytest.raises(ValueError, match="n_estimators"):
        fit(n_estimators=0)


@pytest.mark.parametrize("missing", [np.nan, None])
def test_training_labels_must_not_be_missing(missing):
    X, y = table()
    y = y[:60].astype(object)
    y[3] = missing
    with pytest.raises(ValueError, match="missing values"):
        Classifier(model=small_model(), device="cpu").fit(X[:60], y)


def test_training_targets_must_not_be_missing():
    X, _ = table()
    y = np.random.default_rng(0).normal(size=60)
    y[3] = np.nan
    reg = Regressor(model=small_model(20), bucket_borders=torch.linspace(-3, 3, 21), device="cpu")
    with pytest.raises(ValueError, match="missing values"):
        reg.fit(X[:60], y)
