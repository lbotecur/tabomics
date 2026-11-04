# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from nanoTabPFN (TFM-Playground, Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-TFM-Playground.txt). Modified (see inference/base.py).
"""The classifier."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from tabomics.inference.base import TabOmicsEstimator, check_targets


class Classifier(ClassifierMixin, TabOmicsEstimator):
    """Classifies the rows of a table in context, with a scikit-learn interface.

    `fit` only stores the training rows (no weights change); `predict_proba` runs the model once over the
    training and test rows. Labels can be any values (strings, non-contiguous integers); at most
    `num_outputs` classes (10 for v2 and v2.5).

    See TabOmicsEstimator for the parameters.

    Example:
        >>> clf = Classifier().fit(X_train, y_train)  # the default model, v2.5, downloaded once
        >>> proba = clf.predict_proba(X_test)
    """

    def fit(self, X_train, y_train) -> Classifier:
        """Stores the training rows and encodes the labels as 0..C-1 (original labels in `classes_`)."""
        self.X_train_ = self._prepare(X_train)
        self.label_encoder_ = LabelEncoder()
        self.y_train_ = self.label_encoder_.fit_transform(check_targets(y_train))
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) > self.model_.config.num_outputs:
            raise ValueError(
                f"This model supports at most {self.model_.config.num_outputs} classes, "
                f"but the training data has {len(self.classes_)}."
            )
        self.members_ = self._make_members(num_classes=len(self.classes_))
        return self

    def _default_model(self) -> str:
        from tabomics.hub import DEFAULT_CLASSIFIER

        return DEFAULT_CLASSIFIER

    def _model_targets(self) -> np.ndarray:
        return self.y_train_

    def predict_proba(self, X_test) -> np.ndarray:
        """Class probabilities of the test rows, shape (rows, classes), columns in the order of `classes_`.

        With several members, the probabilities of the members are averaged (each brought back to the original
        class indices first).
        """
        num_classes = len(self.classes_)
        total = 0.0
        for member, out in self._passes(X_test):
            # the model has num_outputs logits; keep those of the classes present in the training rows
            proba = torch.softmax(out.logits.float().squeeze(0)[:, :num_classes], dim=1).cpu().numpy()
            total = total + (proba if member.classes is None else proba[:, member.classes])
        return total / len(self.members_)

    def predict(self, X_test) -> np.ndarray:
        """Most probable class of each test row, as an original label."""
        return self.label_encoder_.inverse_transform(self.predict_proba(X_test).argmax(axis=1))
