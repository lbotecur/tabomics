# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Standardization of regression targets, shared by training (torch, batched) and inference (NumPy).

Both paths must standardize identically, or the model would be trained and queried on different scales. Spec:
mean and unbiased standard deviation (ddof=1), plus the float32 machine epsilon; when the standard deviation is
undefined or zero (a single training row, or a constant target) it is 1.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

# Machine epsilon of float32, the model's compute dtype, added to the standard deviation.
TARGET_NORM_EPS = float(np.finfo(np.float32).eps)


def target_stats_torch(y_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of batched targets over the rows.

    Args:
        y_train: Shape (B, N, 1).

    Returns:
        Mean and standard deviation, each of shape (B, 1, 1). Keeps gradients.
    """
    mean = y_train.mean(dim=1, keepdim=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a single row makes the unbiased std warn; handled below
        std = y_train.std(dim=1, keepdim=True)
    degenerate = ~torch.isfinite(std) | (std == 0)
    std = torch.where(degenerate, torch.ones_like(std), std + TARGET_NORM_EPS)
    return mean, std


def target_stats_numpy(y_train: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of one target vector (see target_stats_torch)."""
    mean = np.mean(y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        std = np.std(y_train, ddof=1)
    std = np.float64(1.0) if not np.isfinite(std) or std == 0.0 else std + TARGET_NORM_EPS
    return mean, std


def standardize(y, mean, std):
    """(y - mean) / std, for torch tensors and NumPy arrays."""
    return (y - mean) / std


def unstandardize(y, mean, std):
    """Inverse of standardize: y * std + mean."""
    return y * std + mean
