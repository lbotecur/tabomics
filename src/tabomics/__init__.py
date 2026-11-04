# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""TabOmics: a tabular foundation model for wide omics data.

A small transformer, pretrained on synthetic tables, that classifies a new table in context (one forward pass on
its training and test rows, no gradient steps), and gives sample embeddings and feature importance on the way.
"""

__version__ = "0.4.0.dev0"

from tabomics.inference import (  # noqa: E402
    Classifier,
    Regressor,
    out_of_fold_embeddings,
    out_of_fold_feature_importance,
)

__all__ = ["Classifier", "Regressor", "__version__", "out_of_fold_embeddings", "out_of_fold_feature_importance"]
