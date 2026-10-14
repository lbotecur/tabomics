# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""The baseline models, by name. Each is built for one fold and only needs fit(X, y) and predict_proba(X).

Every model gets the same data: a float matrix whose categorical columns hold integer codes (NaN for missing values)
and the list of those columns; each model is told about them in its own way. The packages of the foundation models
are not dependencies of TabOmics: install the requirements of the models to run (baselines/requirements/).
"""

from __future__ import annotations

import math

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Classical models; xgboost needs its package (requirements/classic.txt)
CLASSIC = ("random_forest", "logreg", "logreg_en", "xgboost")
# Other tabular foundation models (requirements/tabpfn.txt, requirements/tabicl.txt). A suffix ":n<k>" sets the
# ensemble members (default 1, a single forward pass, as TabOmics and as TabPFN-Wide evaluates its own model);
# ":auto" uses the package's default. TabPFN-3.5 sees at most 768 features per member, so one member only sees a
# random subset of a wide table; ":cover" uses enough members (at least 8) for every feature to be seen once.
FOUNDATION = ("tabpfn-wide-5k", "tabpfn-v2-gn2p4bpt", "tabpfn-3.5", "tabicl-v2", "tabicl-v1.1")
TABICL_CHECKPOINTS = {
    "tabicl-v2": "tabicl-classifier-v2-20260212.ckpt",
    "tabicl-v1.1": "tabicl-classifier-v1.1-20250506.ckpt",
}


def parse_name(name: str) -> tuple[str, int | str]:
    """'tabpfn-3.5:n8' -> ('tabpfn-3.5', 8); ':auto' and ':cover' are kept as strings; no suffix -> 1."""
    base, _, suffix = name.partition(":")
    if not suffix:
        return base, 1
    return base, suffix if suffix in ("auto", "cover") else int(suffix.removeprefix("n"))


def available() -> list[str]:
    """Names accepted by make_model (the foundation models also with a suffix)."""
    return [*CLASSIC, *FOUNDATION]


class FullFeatureCoverage:
    """Builds the classifier at fit time with max(minimum, ceil(features / features_per_member)) members."""

    def __init__(self, make, features_per_member: int, minimum: int = 8):
        self.make, self.features_per_member, self.minimum = make, features_per_member, minimum

    def fit(self, X, y):
        """Builds the classifier for this table's width and fits it."""
        self.n_estimators_ = max(self.minimum, math.ceil(X.shape[1] / self.features_per_member))
        self.model_ = self.make(self.n_estimators_).fit(X, y)
        return self

    def predict_proba(self, X):
        """Class probabilities of the fitted classifier."""
        return self.model_.predict_proba(X)


def elastic_net_cv() -> LogisticRegressionCV:
    """Elastic-net logistic regression (l1_ratio 0.5), C chosen by 3-fold CV on the training rows."""
    penalty = {} if tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 8) else {"penalty": "elasticnet"}
    return LogisticRegressionCV(
        solver="saga",
        l1_ratios=[0.5],
        Cs=5,
        cv=3,
        max_iter=500,
        tol=1e-3,
        scoring="neg_log_loss",
        n_jobs=-1,
        **penalty,
    )


def linear_inputs(categorical: list[int]):
    """For linear models: one-hot categorical columns; numeric columns imputed (median) and standardized."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    if not categorical:
        return numeric
    onehot = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("cat", onehot, list(categorical))], remainder=numeric)


def make_model(name: str, categorical: list[int], device: str = "cuda"):
    """A fresh estimator for one fold.

    Args:
        name: One of available(), the foundation models optionally with ":n<k>", ":auto" or ":cover".
        categorical: Indices of the categorical columns of the task.
        device: Where the models that can use a GPU run.
    """
    if name == "random_forest":  # as TabPFN-Wide: default settings, missing values imputed with the mode
        return make_pipeline(SimpleImputer(strategy="most_frequent"), RandomForestClassifier(n_jobs=-1))
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="most_frequent"), StandardScaler(), LogisticRegression(max_iter=5000)
        )
    if name == "logreg_en":
        return make_pipeline(linear_inputs(categorical), elastic_net_cv())
    if name == "xgboost":  # as TabPFN-Wide: 1000 trees, histogram method, on the GPU if given
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=1000, tree_method="hist", device=device, random_state=42, n_jobs=-1)

    base, n_estimators = parse_name(name)
    tabpfn_categorical = {"categorical_features_indices": list(categorical)}
    if base == "tabpfn-wide-5k":  # as TabPFN-Wide evaluates it: one feature per group
        from tabpfnwide.classifier import TabPFNWideClassifier

        return TabPFNWideClassifier(
            model_name="wide-v2-5k",
            device=device,
            n_estimators=n_estimators,
            features_per_group=1,
            **tabpfn_categorical,
        )
    if base == "tabpfn-v2-gn2p4bpt":  # TabPFN v2 checkpoint trained with one feature per group
        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNClassifier

        path = hf_hub_download(repo_id="Prior-Labs/TabPFN-v2-clf", filename="tabpfn-v2-classifier-gn2p4bpt.ckpt")
        return TabPFNClassifier(
            model_path=path,
            device=device,
            n_estimators=n_estimators,
            random_state=42,
            ignore_pretraining_limits=True,
            **tabpfn_categorical,
        )
    if base == "tabpfn-3.5":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        def tabpfn_35(n):
            return TabPFNClassifier.create_default_for_version(
                ModelVersion.V3_5,
                device=device,
                n_estimators=n,
                random_state=42,
                ignore_pretraining_limits=True,
                **tabpfn_categorical,
            )

        if n_estimators == "cover":
            return FullFeatureCoverage(tabpfn_35, features_per_member=768)
        return tabpfn_35(n_estimators)
    if base in TABICL_CHECKPOINTS:  # TabICL detects categorical columns itself
        from tabicl import TabICLClassifier

        if n_estimators == "cover":
            raise ValueError(":cover is only for tabpfn-3.5")
        return TabICLClassifier(
            checkpoint_version=TABICL_CHECKPOINTS[base],
            device=device,
            random_state=42,
            n_estimators=8 if n_estimators == "auto" else n_estimators,
        )
    raise ValueError(f"Unknown model {name!r}; available: {', '.join(available())}")
