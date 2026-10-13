# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from nanoTabPFN (TFM-Playground, Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-nanoTabPFN.txt). Modified (see inference/base.py).
"""The regressor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.base import RegressorMixin

from tabomics.inference.base import TabOmicsEstimator
from tabomics.model import TabOmicsModel
from tabomics.model.bar_distribution import bar_distribution_mean
from tabomics.model.targets import standardize, target_stats_numpy, unstandardize


class Regressor(RegressorMixin, TabOmicsEstimator):
    """Predicts a numeric target in context, with a scikit-learn interface.

    The model gives a distribution over buckets of the standardized target (one logit per bucket); the
    prediction is its mean, brought back to the original scale. A regression model needs its bucket borders.

    Args:
        model: A regression TabOmicsModel, or the path of its checkpoint (required: TabOmics has no regression
            model of its own yet).
        bucket_borders: Its bucket borders: a tensor of shape (num_outputs + 1,) or the path of a file saved
            with torch.save.
        **kwargs: The other parameters of TabOmicsEstimator.

    Example:
        >>> reg = Regressor(model="regressor.pth", bucket_borders="buckets.pth").fit(X_train, y_train)
        >>> y_pred = reg.predict(X_test)
    """

    def __init__(
        self,
        model: TabOmicsModel | str | Path | None = None,
        bucket_borders: torch.Tensor | str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        amp_dtype: torch.dtype | None = None,
        low_memory: bool | str = True,
        num_mem_chunks: int = 8,
        feature_clip: float | None = None,
        permute_columns: bool | None = None,
        column_seed: int = 0,
        categorical_features: list[int] | None = None,
        infer_categorical: bool = True,
        max_unique_for_categorical: int = 10,
        min_samples_for_categorical_inference: int = 30,
    ):
        super().__init__(
            model=model,
            cache_dir=cache_dir,
            device=device,
            amp_dtype=amp_dtype,
            low_memory=low_memory,
            num_mem_chunks=num_mem_chunks,
            feature_clip=feature_clip,
            permute_columns=permute_columns,
            column_seed=column_seed,
            categorical_features=categorical_features,
            infer_categorical=infer_categorical,
            max_unique_for_categorical=max_unique_for_categorical,
            min_samples_for_categorical_inference=min_samples_for_categorical_inference,
        )
        self.bucket_borders = bucket_borders

    def fit(self, X_train, y_train) -> Regressor:
        """Stores the training rows and the standardized targets."""
        if self.bucket_borders is None:
            raise ValueError("A regression model needs its bucket borders (bucket_borders).")
        borders = self.bucket_borders
        if not isinstance(borders, torch.Tensor):
            borders = torch.load(borders, map_location="cpu", weights_only=True)
        self.X_train_ = self._prepare(X_train)
        self.borders_ = borders.float()
        y_train = np.asarray(y_train)
        self.y_mean_, self.y_std_ = target_stats_numpy(y_train)
        self.y_train_ = standardize(y_train, self.y_mean_, self.y_std_)
        return self

    def _default_model(self) -> str:
        raise ValueError("No model given: TabOmics has no regression model of its own yet; pass one (model=...).")

    def _model_targets(self) -> np.ndarray:
        return self.y_train_

    def predict(self, X_test) -> np.ndarray:
        """Predicted target of each test row (the mean of its predicted distribution)."""
        logits = self._run(X_test, self.y_train_).logits.float().squeeze(0)
        mean = bar_distribution_mean(logits, self.borders_)
        return unstandardize(mean, self.y_mean_, self.y_std_).cpu().numpy()
