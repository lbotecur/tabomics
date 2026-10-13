# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from nanoTabPFN (TFM-Playground, Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-nanoTabPFN.txt). Modified: declared and inferred
# categorical columns, missing values left to the model, optional column permutation.
"""From a user's table to the numbers the model reads.

Numeric columns pass as they are (strings holding numbers are converted); categorical columns become the codes
0..K-1 (as the synthetic training tables have them); constant columns are dropped; missing values stay NaN, for
the model to fill and flag. No standardization here: the model standardizes each column with the training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder


# Module-level functions, not lambdas, so that fitted estimators can be pickled.
def _to_pandas(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)


def _to_numeric(x: pd.DataFrame) -> np.ndarray:
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


class FeaturePreprocessor:
    """Turns a table into model inputs: numeric columns first, then categorical codes, optionally shuffled.

    A column is categorical if it is declared so, or, with inference on, if it is not numeric, or if it is
    numeric with at most `max_unique_for_categorical` distinct values and at least
    `min_samples_for_categorical_inference` observed rows (in a tiny table every column has few values).

    Args:
        categorical_features: Indices of columns that are categorical whatever their values.
        infer_categorical: Detect the other categorical columns. If False, undeclared columns are numeric and a
            non-numeric undeclared column is an error.
        max_unique_for_categorical: Most distinct values of a numeric column inferred as categorical.
        min_samples_for_categorical_inference: Fewest observed rows for that inference.
        permute_columns: Shuffle the columns once, with `column_seed`, before the model. Training tables have their
            columns in random order, while this preprocessing puts the categorical ones last.
        column_seed: Seed of that permutation.
    """

    def __init__(
        self,
        categorical_features: list[int] | None = None,
        infer_categorical: bool = True,
        max_unique_for_categorical: int = 10,
        min_samples_for_categorical_inference: int = 30,
        permute_columns: bool = False,
        column_seed: int = 0,
    ):
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical
        self.max_unique_for_categorical = max_unique_for_categorical
        self.min_samples_for_categorical_inference = min_samples_for_categorical_inference
        self.permute_columns = permute_columns
        self.column_seed = column_seed

    def _column_types(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks of the numeric and categorical columns (constant columns are in neither)."""
        declared = set(self.categorical_features or [])
        for index in declared:
            if not 0 <= index < X.shape[1]:
                raise ValueError(f"categorical_features index {index} is out of range for {X.shape[1]} columns.")
        numeric, categorical = [], []
        for position, name in enumerate(X.columns):
            column = X[name]
            values = column.dropna().unique()
            if len(values) <= 1:  # constant: nothing to learn from
                numeric.append(False)
                categorical.append(False)
                continue
            if position in declared:
                numeric.append(False)
                categorical.append(True)
                continue
            observed = column.notna().sum()
            is_numeric = observed == pd.to_numeric(column, errors="coerce").notna().sum()
            if not self.infer_categorical and not is_numeric:
                raise ValueError(
                    f"Column {position} is not numeric and was not declared categorical; with "
                    "infer_categorical=False every categorical column must be listed in categorical_features."
                )
            few_values = (
                self.infer_categorical
                and is_numeric
                and observed >= self.min_samples_for_categorical_inference
                and len(values) <= self.max_unique_for_categorical
            )
            numeric.append(bool(is_numeric and not few_values))
            categorical.append(not (is_numeric and not few_values))
        return np.array(numeric, dtype=bool), np.array(categorical, dtype=bool)

    def fit(self, X) -> FeaturePreprocessor:
        """Learns the column types, the category codes and the column order from the training rows."""
        X = _to_pandas(X)
        self.n_features_in_ = X.shape[1]
        self.numeric_mask_, self.categorical_mask_ = self._column_types(X)
        numeric = Pipeline(
            [("to_pandas", FunctionTransformer(_to_pandas)), ("to_numeric", FunctionTransformer(_to_numeric))]
        )
        categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        self.transformer_ = ColumnTransformer(
            [("num", numeric, self.numeric_mask_), ("cat", categorical, self.categorical_mask_)]
        ).fit(X)
        kept = np.concatenate([np.flatnonzero(self.numeric_mask_), np.flatnonzero(self.categorical_mask_)])
        order = np.arange(len(kept))
        if self.permute_columns:
            order = np.random.default_rng(self.column_seed).permutation(len(kept))
        self.column_order_ = order
        # original column index of each model input column
        self.original_columns_ = kept[order]
        # number of categories of each model input column (0 for numeric ones)
        counts = np.zeros(len(kept), dtype=int)
        if self.categorical_mask_.any():
            categories = self.transformer_.named_transformers_["cat"].categories_
            # the encoder lists a missing value as one more category (always last); it keeps its NaN code
            counts[self.numeric_mask_.sum() :] = [int((~pd.isna(c)).sum()) for c in categories]
        self.category_counts_ = counts[order]
        return self

    def transform(self, X) -> np.ndarray:
        """Model inputs for rows of the same table: shape (rows, kept columns), NaN for missing values.

        A category not seen in the training rows becomes NaN (missing).
        """
        return self.transformer_.transform(_to_pandas(X))[:, self.column_order_]

    def fit_transform(self, X) -> np.ndarray:
        """fit, then transform the same rows."""
        return self.fit(X).transform(X)
