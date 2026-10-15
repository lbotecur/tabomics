"""Writes parity_inference.npz: the outputs of the reference classifier and regressor (TFM-Playground's
NanoTabPFNClassifier/NanoTabPFNRegressor, where TabOmics was developed) on a small table with missing values and
categorical columns, for test_inference_parity.py. The classifier uses the model of parity_model.npz; the
regressor a small model whose weights are stored here. The reference's feature importance averages the label
attention over all rows, training and test: test_inference_parity.py compares it with feature_importance(rows="all").
Needs TFM-Playground's tfmplayground package; run from the repository root:
    python tests/fixtures/make_inference_fixture.py
"""

from pathlib import Path

import numpy as np
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel

HERE = Path(__file__).parent
SMALL = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2)


def classifier_model():
    data = np.load(HERE / "parity_model.npz")
    model = NanoTabPFNModel(**SMALL, num_outputs=10)
    model.load_state_dict({k.split("::", 1)[1]: torch.from_numpy(data[k]) for k in data.files if "::" in k})
    return model


rng = np.random.default_rng(1)
n, n_train = 90, 60
X = rng.normal(size=(n, 6))
X[:, 1] = rng.integers(0, 3, n)  # categorical codes, 3 levels
X[:, 4] = rng.integers(0, 4, n)  # categorical codes, 4 levels (inferred as categorical: <= 10 values)
X[:, 5] = 7.0  # constant: dropped by the preprocessing
X[rng.random((n, 6)) < 0.05] = np.nan  # missing values everywhere
X[:, 5] = 7.0
y_class = np.array(["a", "b", "c"])[(X[:, 0] > 0).astype(int) + (np.nan_to_num(X[:, 2]) > 0.5).astype(int)]
y_reg = 2.0 * np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 3])
train, test = slice(0, n_train), slice(n_train, n)

out = {"X": X, "y_class": y_class.astype("U1"), "y_reg": y_reg, "n_train": n_train}
settings = {
    "inferred": dict(),
    "declared": dict(categorical_features=[1, 4], infer_categorical=False),
    "permuted": dict(categorical_features=[1, 4], infer_categorical=False, permute_columns=True, column_seed=3),
}
for name, kwargs in settings.items():
    clf = NanoTabPFNClassifier(model=classifier_model(), device="cpu", **kwargs).fit(X[train], y_class[train])
    out[f"{name}_proba"] = clf.predict_proba(X[test])
    out[f"{name}_predict"] = clf.predict(X[test]).astype("U1")
    out[f"{name}_importance"] = clf.feature_attention_scores(X[test])
    out[f"{name}_embeddings"] = clf.get_embeddings(X[test])

torch.manual_seed(1)
regression_model = NanoTabPFNModel(**SMALL, num_outputs=20)
borders = torch.linspace(-3.0, 3.0, 21)
reg = NanoTabPFNRegressor(model=regression_model, dist=FullSupportBarDistribution(borders).float(), device="cpu")
out["regression_predict"] = reg.fit(X[train], y_reg[train]).predict(X[test])
out["regression_borders"] = borders.numpy()
out.update({f"regression_weight::{k}": v.numpy() for k, v in regression_model.state_dict().items()})

np.savez(HERE / "parity_inference.npz", **out)
print("written", sorted(k for k in out if "::" not in k))
