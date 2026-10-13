# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Synthetic tables to pretrain TabOmics on.

GraphPrior gives batches of tables from random causal graphs (TabICL's graph prior), by default in the form the
model gets at inference: numeric columns as generated, categorical columns as codes 0..K-1, and which columns are
categorical.
"""

from tabomics.prior.graph import GraphPrior, GraphSCM, PriorBatch, PriorConfig

__all__ = ["GraphPrior", "GraphSCM", "PriorBatch", "PriorConfig"]
