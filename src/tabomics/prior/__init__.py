# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Synthetic tables to pretrain TabOmics on.

GraphPrior gives batches of tables from random causal graphs (TabICL's graph prior), by default in the form the
model gets at inference: numeric columns as generated, categorical columns as codes 0..K-1, and which columns are
categorical. PriorLoader generates them in parallel workers, reproducibly with a seed; augment_batch widens them
(TabPFN-Wide), makes columns sparse and sets missing values, as configured in an AugmentationConfig.
"""

from tabomics.prior.graph import GraphPrior, GraphSCM, PriorBatch, PriorConfig
from tabomics.prior.loader import PriorLoader
from tabomics.prior.widening import AugmentationConfig, augment_batch

__all__ = [
    "AugmentationConfig",
    "GraphPrior",
    "GraphSCM",
    "PriorBatch",
    "PriorConfig",
    "PriorLoader",
    "augment_batch",
]
