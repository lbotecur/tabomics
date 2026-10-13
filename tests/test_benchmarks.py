import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import savemat
from sklearn.linear_model import LogisticRegression

from tabomics.benchmarks import SUITES, cc18, hdlss, mlomics, run_benchmark, shamir, summarize
from tabomics.benchmarks.metrics import accuracy, roc_auc
from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.model.io import save_pretrained

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def fake_hdlss(root: Path, name="arcene", n=60, d=30):
    (root / hdlss.FOLDER).mkdir(parents=True, exist_ok=True)
    X = RNG.normal(size=(n, d))
    y = np.where(X[:, 0] > 0, 1, -1)  # labels as in the .mat files: not 0..C-1
    savemat(root / hdlss.FOLDER / f"{name}.mat", {"X": X, "Y": y[:, None]})
    return X, y


def fake_mlomics(root: Path, cancer="BRCA", n=40):
    folder = root / mlomics.FOLDER / "Main_Dataset" / "Classification_datasets" / f"GS-{cancer}" / "Original"
    folder.mkdir(parents=True)
    samples = [f"s{i}" for i in range(n)]
    for omic, file in mlomics.OMIC_FILES.items():
        values = RNG.normal(size=(8, n))
        values[0] = 0.0  # an all-zero feature: dropped
        frame = pd.DataFrame(values, index=[f"{omic}{j}" for j in range(8)], columns=samples)
        frame.iloc[1, 3] = np.nan  # a missing value: filled with the feature mean
        frame.to_csv(folder / f"{cancer}_{file}.csv")
    pd.DataFrame({"label": RNG.integers(0, 3, n)}).to_csv(folder / f"{cancer}_label_num.csv", index=False)


def fake_shamir(root: Path, cancer="gbm", n=40):
    folder = root / shamir.FOLDER
    (folder / cancer).mkdir(parents=True)
    samples = [f"TCGA.AA.{i:04d}.01" for i in range(n)]
    for file in shamir.OMIC_FILES.values():
        frame = pd.DataFrame(RNG.normal(size=(6, n)), index=[f"f{j}" for j in range(6)], columns=samples)
        frame.to_csv(folder / cancer / file, sep=" ")
    (folder / "clinical" / "clinical").mkdir(parents=True)
    labels = np.array(["Classical", "Neural", "Proneural"])[np.arange(n) % 3]
    labels[0] = "Rare"  # a class with fewer patients than folds: left out
    clinical = pd.DataFrame({"sampleID": [s.replace(".", "-") for s in samples], "GeneExp_Subtype": labels})
    clinical.to_csv(folder / "clinical" / "clinical" / cancer, sep="\t", index=False)


def fake_cc18(root: Path):
    (root / cc18.FOLDER).mkdir(parents=True)
    X = pd.DataFrame({"a": RNG.normal(size=50), "b": RNG.choice(["x", "y", None], 50)})
    X_arr, y, categorical = cc18.to_arrays(X, RNG.choice(["no", "yes"], 50), [False, True])
    cc18.save_task(root / cc18.FOLDER / "toy.npz", X_arr, y, categorical)
    pd.DataFrame({"name": ["toy"]}).to_csv(root / cc18.FOLDER / "index.csv", index=False)


# --- the suites ------------------------------------------------------------------------------------------------


def test_hdlss_shuffles_encodes_and_folds_as_tabpfn_wide(tmp_path):
    fake_hdlss(tmp_path)
    X, y, categorical = hdlss.load_task("arcene", tmp_path)
    assert X.dtype == np.float32 and set(y) == {0, 1} and categorical == []
    folds = hdlss.folds(y)
    assert len(folds) == 30  # 3 folds x 10 repeats
    assert all(len(set(train) & set(test)) == 0 for train, test in folds)
    assert hdlss.tasks(tmp_path) == list(hdlss.DATASETS)


def test_mlomics_cleans_and_concatenates_omics(tmp_path):
    fake_mlomics(tmp_path)
    X, y, _ = mlomics.load_task("BRCA/mrna", tmp_path)
    assert X.shape == (40, 7) and not np.isnan(X).any()  # all-zero feature dropped, missing value filled
    X_all, _, _ = mlomics.load_task("BRCA/mrna+cnv+methylation+mirna", tmp_path)
    assert X_all.shape == (40, 28)
    assert len(mlomics.folds(y)) == 5 and "BRCA/mrna+cnv+methylation+mirna" in mlomics.tasks(tmp_path)


def test_shamir_matches_samples_and_drops_rare_classes(tmp_path):
    fake_shamir(tmp_path)
    X, y, _ = shamir.load_task("gbm/mrna+methylation+mirna", tmp_path)
    assert X.shape == (39, 18) and set(y) == {0, 1, 2}  # the patient of the rare class is left out


def test_cc18_codes_categoricals_and_missing_values(tmp_path):
    fake_cc18(tmp_path)
    X, y, categorical = cc18.load_task("toy", tmp_path)
    assert categorical == [1] and np.isnan(X[:, 1]).any() and set(np.unique(X[~np.isnan(X[:, 1]), 1])) <= {0, 1}
    assert len(cc18.folds(y)) == 10 and cc18.tasks(tmp_path) == ["toy"]
    assert cc18.keep("credit-g", 1000, 20, 2) and not cc18.keep("madelon", 2600, 500, 2)


def test_metrics():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    assert roc_auc(y, proba) == 1.0 and accuracy(y, proba) == 0.75
    assert np.isnan(roc_auc(np.array([1, 1]), np.array([[0.5, 0.5], [0.4, 0.6]])))  # one class only


# --- the runner and the script ---------------------------------------------------------------------------------


def test_runner_any_estimator_resumes_and_filters(tmp_path):
    fake_hdlss(tmp_path, "arcene")
    fake_hdlss(tmp_path, "colon")
    output = tmp_path / "out.csv"
    calls = []

    def factory(categorical):
        calls.append(categorical)
        return LogisticRegression(max_iter=1000)

    results = run_benchmark(factory, "logreg", ["hdlss"], tmp_path, output, datasets=["arcene", "colon"], verbose=False)
    assert len(results) == 60 and set(results.dataset) == {"arcene", "colon"} and len(calls) == 60
    again = run_benchmark(factory, "logreg", ["hdlss"], tmp_path, output, datasets=["arcene", "colon"], verbose=False)
    assert len(again) == 60 and len(calls) == 60  # nothing evaluated twice
    summary = summarize(again)
    assert summary.loc[("hdlss", "logreg"), "datasets"] == 2
    with pytest.raises(ValueError, match="Unknown suites"):
        run_benchmark(factory, "logreg", ["tabarena"], tmp_path, output)
    assert set(SUITES) == {"hdlss", "mlomics", "shamir", "cc18"}


def test_evaluate_script_end_to_end(tmp_path):
    fake_hdlss(tmp_path, "arcene", n=45)
    torch.manual_seed(0)
    small = TabOmicsConfig(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2)
    save_pretrained(TabOmicsModel(small), tmp_path / "release")
    output = tmp_path / "results.csv"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "evaluate.py"), "--model", str(tmp_path / "release"),
         "--suite", "hdlss", "--datasets", "arcene", "--data-root", str(tmp_path), "--device", "cpu",
         "--output", str(output), "--name", "small"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    results = pd.read_csv(output)
    assert len(results) == 30 and set(results.model) == {"small"} and results.roc_auc.notna().all()
    assert "results:" in result.stdout
