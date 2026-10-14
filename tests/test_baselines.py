"""The baselines: names, the classical models on a fake suite, and the evaluation script."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from tabomics.benchmarks import hdlss

BASELINES = Path(__file__).resolve().parents[1] / "baselines"
spec = importlib.util.spec_from_file_location("baseline_models", BASELINES / "models.py")
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


def fake_hdlss(root: Path, n=45, d=20):
    (root / hdlss.FOLDER).mkdir(parents=True, exist_ok=True)
    X = np.random.default_rng(0).normal(size=(n, d))
    savemat(root / hdlss.FOLDER / "arcene.mat", {"X": X, "Y": np.where(X[:, 0] > 0, 1, -1)[:, None]})


def test_names():
    assert models.parse_name("tabpfn-3.5:n8") == ("tabpfn-3.5", 8)
    assert models.parse_name("tabpfn-3.5:cover") == ("tabpfn-3.5", "cover")
    assert models.parse_name("tabicl-v2") == ("tabicl-v2", 1)
    with pytest.raises(ValueError, match="Unknown model"):
        models.make_model("svm", [])


@pytest.mark.parametrize("name", ["random_forest", "logreg", "logreg_en"])
def test_classical_models_fit_with_missing_and_categorical_values(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    X[:, 1] = rng.integers(0, 3, 60)
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int)
    proba = models.make_model(name, [1]).fit(X[:40], y[:40]).predict_proba(X[40:])
    assert proba.shape == (20, 2)


def test_evaluate_script(tmp_path):
    fake_hdlss(tmp_path)
    output = tmp_path / "rf.csv"
    subprocess.run(
        [sys.executable, str(BASELINES / "evaluate.py"), "--model", "random_forest", "--suite", "hdlss",
         "--datasets", "arcene", "--data-root", str(tmp_path), "--output", str(output)],
        check=True, capture_output=True,
    )  # fmt: skip
    results = pd.read_csv(output)
    assert len(results) == 30 and set(results.model) == {"random_forest"} and results.roc_auc.notna().all()
