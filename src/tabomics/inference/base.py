# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from nanoTabPFN (TFM-Playground, Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-nanoTabPFN.txt). Modified: low-memory inference with an
# automatic mode, feature clip and column permutation from the checkpoint, mixed precision, embeddings and
# feature importance.
"""What the classifier and the regressor share: loading the model, preprocessing, and one forward pass."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator

from tabomics.inference.preprocessing import FeaturePreprocessor
from tabomics.model import TabOmicsModel, TabOmicsOutput
from tabomics.model.io import load_checkpoint

# With low_memory="auto", a forward pass whose (rows, columns + 1, embedding) table has at least this many
# elements runs in low-memory mode. Around 2**32 elements the normal path of PyTorch returned wrong results
# (AUROC 0.5 or NaN on gisette: 7000 rows x 5000 columns x 192) while low-memory was correct; 2**31 keeps a
# margin. It depends on the table only, not on the GPU, so results do not change across GPUs.
LOW_MEMORY_AUTO_ELEMENTS = 2**31


def default_device() -> torch.device:
    """CUDA if available, else Apple's MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model(model: TabOmicsModel | str | Path | None) -> TabOmicsModel:
    """The model to use: an instance as given, or loaded from a checkpoint file."""
    if model is None:
        raise ValueError("No model given: pass a TabOmicsModel or the path of a checkpoint.")
    if isinstance(model, TabOmicsModel):
        return model
    return load_checkpoint(model)


class TabOmicsEstimator(BaseEstimator):
    """Base of Classifier and Regressor (scikit-learn compatible: parameters are stored as given).

    Args:
        model: A TabOmicsModel, or the path of a checkpoint.
        device: Where to run; default CUDA, else MPS, else CPU.
        amp_dtype: Mixed precision, e.g. torch.bfloat16 (much faster on wide tables on GPU); None runs in
            float32.
        low_memory: Low-memory inference (True, the default: as accurate, and correct on very large tables),
            the normal path (False), or "auto" (low-memory for large tables only, and as a retry after running out
            of GPU memory).
        num_mem_chunks: Pieces the memory-heavy steps are split into.
        feature_clip: Bound of the standardized feature values; None keeps the model's own.
        permute_columns: Shuffle the columns once before the model; None keeps the model's own setting.
        column_seed: Seed of that permutation.
        categorical_features: Indices of the columns that are categorical (see FeaturePreprocessor).
        infer_categorical: Detect the other categorical columns.
        max_unique_for_categorical: Most distinct values of a numeric column inferred as categorical.
        min_samples_for_categorical_inference: Fewest observed rows for that inference.
    """

    def __init__(
        self,
        model: TabOmicsModel | str | Path | None = None,
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
        self.model = model
        self.device = device
        self.amp_dtype = amp_dtype
        self.low_memory = low_memory
        self.num_mem_chunks = num_mem_chunks
        self.feature_clip = feature_clip
        self.permute_columns = permute_columns
        self.column_seed = column_seed
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical
        self.max_unique_for_categorical = max_unique_for_categorical
        self.min_samples_for_categorical_inference = min_samples_for_categorical_inference

    def _prepare(self, X_train) -> np.ndarray:
        """Loads the model (once), applies the settings and fits the preprocessing; returns the training inputs."""
        if self.low_memory not in (True, False, "auto"):
            raise ValueError(f"low_memory must be True, False or 'auto', got {self.low_memory!r}")
        if getattr(self, "model_", None) is None or getattr(self, "_model_source", None) is not self.model:
            self.device_ = torch.device(self.device) if self.device is not None else default_device()
            self.model_ = resolve_model(self.model).to(self.device_).eval()
            self._model_source = self.model
        if self.feature_clip is not None:
            self.model_.feature_clip = self.feature_clip
        permute = self.model_.config.permute_columns if self.permute_columns is None else self.permute_columns
        self.preprocessor_ = FeaturePreprocessor(
            categorical_features=self.categorical_features,
            infer_categorical=self.infer_categorical,
            max_unique_for_categorical=self.max_unique_for_categorical,
            min_samples_for_categorical_inference=self.min_samples_for_categorical_inference,
            permute_columns=permute,
            column_seed=self.column_seed,
        )
        X = self.preprocessor_.fit_transform(X_train)
        self.n_features_in_ = self.preprocessor_.n_features_in_
        return X

    def _run(self, X_test, y_train: np.ndarray, **extras) -> TabOmicsOutput:
        """One forward pass over the training rows and the test rows, without gradients.

        Args:
            X_test: Test rows, in the original columns.
            y_train: Training targets as the model reads them (class indices, or standardized values).
            **extras: return_embeddings / return_label_attention, passed to TabOmicsModel.run.
        """
        x = np.concatenate([self.X_train_, self.preprocessor_.transform(X_test)])
        x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device_)
        y = torch.from_numpy(np.asarray(y_train)).unsqueeze(0).to(torch.float).to(self.device_)
        model, train_size = self.model_, len(self.X_train_)

        def forward(low_memory: bool) -> TabOmicsOutput:
            model.set_low_memory(low_memory)
            enabled = self.amp_dtype is not None
            with torch.no_grad(), torch.autocast(self.device_.type, dtype=self.amp_dtype, enabled=enabled):
                return model.run(x, y, train_size, num_mem_chunks=self.num_mem_chunks, **extras)

        if self.low_memory != "auto":
            return forward(bool(self.low_memory))
        elements = x.shape[1] * (x.shape[2] + 1) * model.config.embedding_size
        if elements >= LOW_MEMORY_AUTO_ELEMENTS:
            return forward(True)
        try:
            return forward(False)
        except torch.cuda.OutOfMemoryError:
            warnings.warn("Out of GPU memory in normal mode; retrying with low-memory inference.", stacklevel=3)
            torch.cuda.empty_cache()
            return forward(True)

    def embeddings(self, X_test) -> np.ndarray:
        """Embedding of each test row: its label-column representation before the decoder.

        For embeddings of the training rows comparable with these, use out_of_fold_embeddings: a training row
        seen with its own label is not in the same situation as a test row.

        Returns:
            Shape (test rows, embedding size).
        """
        out = self._run(X_test, self._model_targets(), return_embeddings=True)
        return out.embeddings[0, len(self.X_train_) :].float().cpu().numpy()

    def feature_importance(self, X_test) -> np.ndarray:
        """Attention of the label column to each feature, averaged over test rows, heads and layers.

        This is the importance measure of TabPFN-Wide. Constant columns, never seen by the model, get NaN.

        Returns:
            One score per original column, shape (n_features_in_,).
        """
        attention = self._run(X_test, self._model_targets(), return_label_attention=True).label_attention
        scores = np.full(self.n_features_in_, np.nan)
        scores[self.preprocessor_.original_columns_] = attention[:-1].cpu().numpy()  # the label is last
        return scores

    def _model_targets(self) -> np.ndarray:
        raise NotImplementedError
