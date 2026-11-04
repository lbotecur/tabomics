# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from nanoTabPFN (TFM-Playground, Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-TFM-Playground.txt). Modified: low-memory inference with an
# automatic mode, feature clip and column permutation from the checkpoint, mixed precision, embeddings and
# feature importance.
"""What the classifier and the regressor share: loading the model, preprocessing, and one forward pass."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from tabomics.inference.preprocessing import FeaturePreprocessor
from tabomics.model import TabOmicsModel, TabOmicsOutput
from tabomics.model.io import load_model

# With low_memory="auto", a forward pass whose (rows, columns + 1, embedding) table has at least this many
# elements runs in low-memory mode. Around 2**32 elements the normal path of PyTorch returned wrong results
# (AUROC 0.5 or NaN on gisette: 7000 rows x 5000 columns x 192) while low-memory was correct; 2**31 keeps a
# margin. The threshold depends on the table only, not on the GPU, so the same path is chosen on every GPU; the
# wrong results were observed on NVIDIA A100 GPUs.
LOW_MEMORY_AUTO_ELEMENTS = 2**31


def check_targets(y) -> np.ndarray:
    """The training labels or targets as an array, refused if any is missing.

    A training row without a label cannot be context, and the classifier would otherwise take the missing value for
    one more class.
    """
    y = np.asarray(y)
    if pd.isna(y).any():
        raise ValueError("y contains missing values; drop the training rows without a label")
    return y


@dataclass(frozen=True)
class EnsembleMember:
    """One member of an ensemble: a relabeling of the table that should not change the answer.

    The model should be indifferent to the order of the columns, to the codes given to the categories of a column,
    and to the indices given to the classes; it is only approximately so. Averaging over such relabelings removes
    that residual dependence.

    Attributes:
        columns: Order in which the (preprocessed) columns enter the model.
        codes: For each categorical column (by its preprocessed index), the new code of each category code.
        classes: New index of each class (classifier only), or None.
    """

    columns: np.ndarray
    codes: dict[int, np.ndarray]
    classes: np.ndarray | None = None

    def inputs(self, X: np.ndarray) -> np.ndarray:
        """Applies the relabeling to preprocessed rows (missing values stay missing)."""
        X = X.copy()
        for column, new_code in self.codes.items():
            observed = ~np.isnan(X[:, column])
            X[observed, column] = new_code[X[observed, column].astype(int)]
        return X[:, self.columns]


def model_name(source: TabOmicsModel | str | Path) -> str:
    """How a model source is recorded: a published name as is, a file or directory by its absolute path."""
    from tabomics import hub

    if isinstance(source, TabOmicsModel):
        return "TabOmicsModel"
    if isinstance(source, str) and source in hub.PUBLISHED_MODELS:
        return source
    return str(Path(source).resolve())


def resolve_amp_dtype(amp_dtype: torch.dtype | str | None, device: torch.device) -> torch.dtype | None:
    """The dtype of mixed precision on a device.

    "auto" is bfloat16 on CUDA and None (float32) elsewhere; a dtype name such as "bfloat16" is that torch dtype; a
    torch dtype or None is kept.
    """
    if amp_dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else None
    if isinstance(amp_dtype, str):
        return getattr(torch, amp_dtype)
    return amp_dtype


def default_device() -> torch.device:
    """CUDA if available, else Apple's MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TabOmicsEstimator(BaseEstimator):
    """Base of Classifier and Regressor (scikit-learn compatible: parameters are stored as given).

    Args:
        model: A published model's name ("v2.5"), a release directory or `.safetensors` file, a training checkpoint
            (`.pth`), or a TabOmicsModel. None: the default model of the estimator (the classifier:
            tabomics.hub.DEFAULT_CLASSIFIER, "v2.5").
        cache_dir: Where published models are downloaded; default Hugging Face's cache.
        device: Where to run; default CUDA, else MPS, else CPU.
        amp_dtype: Mixed precision: "auto" (the default) runs in bfloat16 on CUDA, much faster on wide tables, and in
            float32 elsewhere; torch.bfloat16 (or "bfloat16") always in bfloat16; None always in float32.
        low_memory: Low-memory inference (True, the default: as accurate, and correct on very large tables),
            the normal path (False), or "auto" (low-memory for large tables only, and as a retry after running out
            of GPU memory). It normalizes inside the memory chunks and keeps the activations in the input dtype
            (bf16 with amp_dtype), so no full-table float32 copy is made.
        num_mem_chunks: Pieces each step of a block (attention between columns, between rows, MLP) is applied in,
            writing each result in place: the peak memory of the intermediate tensors falls about this many times,
            at some cost in speed. Results are the same. Inference only (with gradients the step runs in one
            piece). With low_memory, the two together let tables of tens of thousands of columns fit.
        feature_clip: Bound of the standardized feature values; None keeps the model's own.
        permute_columns: Shuffle the columns once before the model; None keeps the model's own setting.
        column_seed: Seed of that permutation.
        categorical_features: Indices of the columns that are categorical (see FeaturePreprocessor).
        infer_categorical: Detect the other categorical columns.
        max_unique_for_categorical: Most distinct values of a numeric column inferred as categorical.
        min_samples_for_categorical_inference: Fewest observed rows for that inference.
        n_estimators: Number of ensemble members averaged, each a forward pass on a relabeled copy of the table. The
            first is the table as given; each other one reorders the columns and renumbers the category codes and,
            for the classifier, the class indices. 1, the default, is a single forward pass.
        random_state: Seed of those relabelings.

    After fit, `model_name_` says which model was used: a published model by its name ("v2.5"), a file or
    directory by its absolute path (for a model downloaded with huggingface_hub, the path holds the repository and
    the commit), and a TabOmicsModel passed as an object as "TabOmicsModel".
    """

    def __init__(
        self,
        model: TabOmicsModel | str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        amp_dtype: torch.dtype | str | None = "auto",
        low_memory: bool | str = True,
        num_mem_chunks: int = 8,
        feature_clip: float | None = None,
        permute_columns: bool | None = None,
        column_seed: int = 0,
        categorical_features: list[int] | None = None,
        infer_categorical: bool = True,
        max_unique_for_categorical: int = 10,
        min_samples_for_categorical_inference: int = 30,
        n_estimators: int = 1,
        random_state: int = 0,
    ):
        self.model = model
        self.cache_dir = cache_dir
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
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _prepare(self, X_train) -> np.ndarray:
        """Loads the model (once), applies the settings and fits the preprocessing; returns the training inputs."""
        if self.low_memory not in (True, False, "auto"):
            raise ValueError(f"low_memory must be True, False or 'auto', got {self.low_memory!r}")
        if getattr(self, "model_", None) is None or getattr(self, "_model_source", None) is not self.model:
            self.device_ = torch.device(self.device) if self.device is not None else default_device()
            source = self.model if self.model is not None else self._default_model()
            self.model_ = load_model(source, cache_dir=self.cache_dir).to(self.device_).eval()
            self.model_name_ = model_name(source)
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

    def _make_members(self, num_classes: int | None = None) -> list[EnsembleMember]:
        """The members of the ensemble: the table as given, then n_estimators - 1 random relabelings."""
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {self.n_estimators}")
        counts = self.preprocessor_.category_counts_
        identity = EnsembleMember(np.arange(len(counts)), {}, None)
        rng = np.random.default_rng(self.random_state)
        members = [identity]
        for _ in range(self.n_estimators - 1):
            columns = rng.permutation(len(counts))
            codes = {int(k): rng.permutation(int(n)) for k, n in enumerate(counts) if n > 1}
            classes = rng.permutation(num_classes) if num_classes and num_classes > 1 else None
            members.append(EnsembleMember(columns, codes, classes))
        return members

    def _passes(
        self, X_test, relabel_classes: bool = True, **extras
    ) -> Iterator[tuple[EnsembleMember, TabOmicsOutput]]:
        """Runs every member of the ensemble on the training rows and the test rows.

        Args:
            X_test: Test rows, in the original columns.
            relabel_classes: Apply the members' class relabelling (off for embeddings, see embeddings()).
            **extras: return_embeddings / return_label_attention, passed to TabOmicsModel.run.

        Yields:
            Each member and its output.
        """
        test = self.preprocessor_.transform(X_test)
        targets = self._model_targets()
        for member in self.members_:
            x = np.concatenate([member.inputs(self.X_train_), member.inputs(test)])
            y = member.classes[targets] if relabel_classes and member.classes is not None else targets
            yield member, self._forward(x, y, **extras)

    def _forward(self, x: np.ndarray, y_train: np.ndarray, **extras) -> TabOmicsOutput:
        """One forward pass, without gradients.

        Args:
            x: Training rows followed by test rows, preprocessed, shape (rows, columns).
            y_train: Training targets as the model reads them (class indices, or standardized values).
            **extras: return_embeddings / return_label_attention, passed to TabOmicsModel.run.
        """
        x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device_)
        y = torch.from_numpy(np.asarray(y_train)).unsqueeze(0).to(torch.float).to(self.device_)
        model, train_size = self.model_, len(self.X_train_)

        def forward(low_memory: bool) -> TabOmicsOutput:
            model.set_low_memory(low_memory)
            amp_dtype = resolve_amp_dtype(self.amp_dtype, self.device_)
            with torch.no_grad(), torch.autocast(self.device_.type, dtype=amp_dtype, enabled=amp_dtype is not None):
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

    def embeddings(self, X_test, reduce: str = "mean") -> np.ndarray:
        """Embedding of each test row: its label-column representation before the decoder.

        With several members, they reorder the columns and renumber the category codes only: a member with other
        class indices encodes the labels differently, and its embeddings would not be comparable.

        For embeddings of the training rows comparable with these, use out_of_fold_embeddings: a training row
        seen with its own label is not in the same situation as a test row.

        Args:
            X_test: Test rows.
            reduce: "mean" averages the members; "concat" puts them side by side.

        Returns:
            Shape (test rows, embedding size), or (test rows, n_estimators x embedding size) with "concat".
        """
        if reduce not in ("mean", "concat"):
            raise ValueError(f"reduce must be 'mean' or 'concat', got {reduce!r}")
        train_size = len(self.X_train_)
        pieces = [
            out.embeddings[0, train_size:].float().cpu().numpy()
            for _, out in self._passes(X_test, relabel_classes=False, return_embeddings=True)
        ]
        return np.mean(pieces, axis=0) if reduce == "mean" else np.concatenate(pieces, axis=1)

    def feature_importance(self, X_test, per_row: bool = False) -> np.ndarray:
        """Attention of the label column of the test rows to each feature, averaged over the heads and blocks.

        The importance measure of TabPFN-Wide, computed on the test rows only. TabPFN-Wide averages over all rows,
        training and test, whose label columns mean different things: a known label in training rows, the
        prediction being built in test rows. A test row's score is how much its prediction draws on each column,
        and it does not depend on the other test rows. For the training rows, use out_of_fold_feature_importance:
        a training row seen with its own label is not in the same situation as a test row.

        With several members each column has been at several positions, and the scores are averaged over the
        members, which removes the effect of the position. Constant columns, never seen by the model, get NaN.

        Args:
            X_test: Test rows.
            per_row: One score per test row instead of their mean.

        Returns:
            One score per original column, shape (n_features_in_,); with per_row, shape (test rows,
            n_features_in_), in the order of X_test, whose mean over the rows is the first.
        """
        total = None
        for member, out in self._passes(X_test, return_label_attention=True):
            attention = out.label_attention[0, :, :-1].cpu().numpy()  # (test rows, columns); the label is last
            if total is None:
                total = np.zeros((attention.shape[0], self.n_features_in_))
            total[:, self.preprocessor_.original_columns_[member.columns]] += attention
        scores = np.full_like(total, np.nan)
        kept = self.preprocessor_.original_columns_
        scores[:, kept] = total[:, kept] / len(self.members_)
        return scores if per_row else scores.mean(axis=0)

    def _default_model(self) -> str:
        raise ValueError("No model given.")

    def _model_targets(self) -> np.ndarray:
        raise NotImplementedError
