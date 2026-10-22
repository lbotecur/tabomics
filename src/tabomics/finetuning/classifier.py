# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""A classifier that fine-tunes the model on its training rows before predicting, as a scikit-learn estimator."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import torch

from tabomics.finetuning.loop import finetune
from tabomics.inference import Classifier
from tabomics.inference.base import default_device
from tabomics.model import TabOmicsModel
from tabomics.model.io import load_model


class FineTunedClassifier(Classifier):
    """Classifier that first fine-tunes the model on the training rows of the table (see tabomics.finetuning).

    `fit` holds out `validation_fraction` of the training rows, fine-tunes a copy of the model on episodes of the
    others until the loss on the held-out rows stops improving, and then classifies as Classifier does, with the
    fine-tuned model and all the training rows as context. Everything else (n_estimators, embeddings, feature
    importance) works as in Classifier. Inside a cross-validation, each fold fine-tunes on its own training rows
    only, so the test rows are never seen.

    Args:
        model, cache_dir, device, amp_dtype, low_memory, num_mem_chunks, feature_clip, permute_columns,
            column_seed, categorical_features, infer_categorical, max_unique_for_categorical,
            min_samples_for_categorical_inference, n_estimators, random_state: As in Classifier (random_state also
            seeds the fine-tuning).
        epochs: Most epochs of fine-tuning.
        steps_per_epoch: Batches per epoch.
        batch_size: Episodes per batch.
        accumulate: Batches per optimizer step; must divide steps_per_epoch.
        lr: Learning rate.
        validation_fraction: Share of the training rows held out for early stopping.
        patience: Epochs without improvement before stopping.
        max_features: Most columns in an episode (memory); None = no limit.
        freeze: Name prefixes of parameters not to train; None trains every weight.
        prior_fraction: Share of the batches made of synthetic tables.
        verbose: Print the losses of every epoch.

    Attributes:
        finetune_history_: One row per epoch: train_loss, val_loss, val_accuracy (epoch 0 = the pretrained model).
        best_epoch_: Epoch whose weights are used (0 = no epoch improved: the pretrained weights).
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
        epochs: int = 30,
        steps_per_epoch: int = 64,
        batch_size: int = 1,
        accumulate: int = 8,
        lr: float = 1e-5,
        validation_fraction: float = 0.1,
        patience: int = 5,
        max_features: int | None = None,
        freeze: Sequence[str] | None = None,
        prior_fraction: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(
            model=model, cache_dir=cache_dir, device=device, amp_dtype=amp_dtype, low_memory=low_memory,
            num_mem_chunks=num_mem_chunks, feature_clip=feature_clip, permute_columns=permute_columns,
            column_seed=column_seed, categorical_features=categorical_features, infer_categorical=infer_categorical,
            max_unique_for_categorical=max_unique_for_categorical,
            min_samples_for_categorical_inference=min_samples_for_categorical_inference, n_estimators=n_estimators,
            random_state=random_state,
        )  # fmt: skip
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.max_features = max_features
        self.freeze = freeze
        self.prior_fraction = prior_fraction
        self.verbose = verbose

    def fit(self, X_train, y_train) -> FineTunedClassifier:
        """Fine-tunes a copy of the model on the training rows, then stores them as Classifier.fit does."""
        device = torch.device(self.device) if self.device is not None else default_device()
        pretrained = load_model(self.model if self.model is not None else self._default_model(), self.cache_dir)
        if self.feature_clip is not None:  # fine-tune with the clip it will predict with
            pretrained = copy.deepcopy(pretrained)
            pretrained.feature_clip = self.feature_clip
        result = finetune(
            pretrained, [(X_train, y_train)], validation_fraction=self.validation_fraction, epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch, batch_size=self.batch_size, accumulate=self.accumulate, lr=self.lr,
            patience=self.patience, max_features=self.max_features, freeze=self.freeze,
            prior_fraction=self.prior_fraction, device=device, amp_dtype=self.amp_dtype,
            num_mem_chunks=self.num_mem_chunks, seed=self.random_state, verbose=self.verbose,
            categorical_features=self.categorical_features, infer_categorical=self.infer_categorical,
            max_unique_for_categorical=self.max_unique_for_categorical,
            min_samples_for_categorical_inference=self.min_samples_for_categorical_inference,
        )  # fmt: skip
        self.finetune_history_ = result.history
        self.best_epoch_ = result.best_epoch
        # the fine-tuned model replaces the loaded one; Classifier.fit then uses it as it is
        self.model_, self.device_, self._model_source = result.model.to(device).eval(), device, self.model
        return super().fit(X_train, y_train)
