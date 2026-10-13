# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# The mean of the full-support bar distribution follows FullSupportBarDistribution.mean of the PFNs library
# (https://github.com/automl/PFNs, Apache License 2.0), reimplemented here to avoid that dependency.
"""Regression output: a distribution over buckets of the standardized target."""

from __future__ import annotations

import torch


def bar_distribution_mean(logits: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Mean of a full-support bar distribution.

    The model gives one logit per bucket between consecutive borders. Inner buckets are uniform; the two outer
    ones are half-normal tails beyond the second and the second-to-last border, with half of their mass within
    the bucket width, so the distribution covers the whole real line.

    Args:
        logits: Shape (..., num_buckets).
        borders: Increasing bucket borders, shape (num_buckets + 1,).

    Returns:
        Shape (...): the mean of each distribution.
    """
    borders = borders.to(logits.device, torch.float32)
    widths = borders[1:] - borders[:-1]
    means = borders[:-1] + widths / 2

    def half_normal_mean(width: torch.Tensor) -> torch.Tensor:
        # half-normal whose median is `width`: scale = width / icdf(0.5), mean = scale * sqrt(2 / pi)
        standard = torch.distributions.HalfNormal(torch.tensor(1.0))
        return torch.distributions.HalfNormal(width.cpu() / standard.icdf(torch.tensor(0.5))).mean.to(width.device)

    means[0] = -half_normal_mean(widths[0]) + borders[1]
    means[-1] = half_normal_mean(widths[-1]) + borders[-2]
    return torch.softmax(logits, -1) @ means
