# SPDX-License-Identifier: BSD-3-Clause
# Copied from TabICL 2.2.0, tabicl/prior/_reg2cls.py: torch_nanstd, outlier_removing and standard_scaling only
# (https://github.com/soda-inria/tabicl), Copyright (c) 2025, Soda team @ Inria; see LICENSES/BSD-3-Clause-TabICL.txt.

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import Tensor


def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> Tensor:
    """Calculates the standard deviation of a tensor, ignoring NaNs, using NumPy internally.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    dim : int or tuple[int], optional
        The dimension or dimensions to reduce. Defaults to None (reduce all dimensions).

    keepdim : bool, optional
        Whether the output tensor has `dim` retained or not. Defaults to False.

    ddof : int, optional
        Delta Degrees of Freedom.

    dtype : torch.dtype, optional
        The desired data type of returned tensor. Defaults to None.

    Returns
    -------
    Tensor
        The standard deviation.
    """
    device = input.device
    np_input = input.cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(np_input, axis=dim, dtype=dtype, keepdims=keepdim, ddof=ddof)

    return torch.from_numpy(std).to(dtype=torch.float, device=device)


def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    """Standardizes features by removing the mean and scaling to unit variance.

    NaNs are ignored in mean/std calculation.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H), where T is sequence length, H is features.

    clip_value : float, optional, default=100
        The value to clip the standardized input to, preventing extreme outliers.

    Returns
    -------
    Tensor
        The standardized input, clipped between -clip_value and clip_value.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std

    return torch.clip(scaled_input, min=-clip_value, max=clip_value)


def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    """Clamps outliers in the input tensor based on a specified number of standard deviations (threshold).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H).

    threshold : float, optional, default=4.0
        Number of standard deviations to use as the cutoff.

    Returns
    -------
    Tensor
        The tensor with outliers clamped.
    """
    # First stage: Identify outliers using initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    # Create mask for non-outlier, non-NaN values
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # Second pass using only non-outlier values for mean/std
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)

    # Handle cases where a column had <= 1 valid value after masking -> std is NaN or 0
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    # Recalculate cutoff with robust estimates
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off

    # Replace NaN bounds with +/- inf
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    return input.clamp(min=lower, max=upper)


