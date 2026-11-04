# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Using a model on a table: classification, regression, embeddings and feature importance."""

from tabomics.inference.classifier import Classifier
from tabomics.inference.embeddings import out_of_fold_embeddings, out_of_fold_feature_importance
from tabomics.inference.preprocessing import FeaturePreprocessor
from tabomics.inference.regressor import Regressor

__all__ = [
    "Classifier",
    "FeaturePreprocessor",
    "Regressor",
    "out_of_fold_embeddings",
    "out_of_fold_feature_importance",
]
