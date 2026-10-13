# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TabICL 2.2.0, tabicl/prior/_graph_scm.py (https://github.com/soda-inria/tabicl),
# Copyright (c) 2025, Soda team @ Inria; see LICENSES/BSD-3-Clause-TabICL.txt. Changes: see CHANGES.md.
"""One synthetic table from a random causal graph (TabICL's graph prior).

TabICL builds a table as categorical columns (codes 0..K-1) followed by numeric columns and then, before handing it
over, clamps every column to about 4 robust standard deviations and z-scores it, over all the rows (training and
test). At inference nothing like that happens: numeric columns arrive as they are, categorical ones as codes
0..K-1, and the model standardizes each column with the training rows only. With `raw=True` (the default here)
the table is returned without those two steps, so that training tables look like inference tables; with
`raw=False` it is TabICL's table, bit for bit (same seed). Either way the columns that are categorical are
returned too (TabICL knows them but drops them).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tabomics.prior.graph._base import Context, DatasetProperties
from tabomics.prior.graph._config import PriorConfig
from tabomics.prior.graph._dataset import RandomDataset
from tabomics.prior.graph._properties import sample_categorical_sizes
from tabomics.prior.graph._scaling import outlier_removing, standard_scaling

# Largest number of categories of a categorical column (TabICL's value).
MAX_CATEGORIES = 200


class GraphSCM:
    """Generates one table (features, target, categorical mask) from a random causal graph.

    Every random draw happens in the order of TabICL's GraphSCM, so the same seed gives the same table.

    Args:
        regression: Numeric target instead of classes.
        seq_len: Rows of the table.
        num_features: Columns of the table.
        max_features: Width the table is padded to with zeros.
        num_classes: Classes of the target (classification).
        permute_features: Shuffle the columns (TabICL puts the categorical ones first).
        permute_labels: Shuffle the class indices.
        config: Settings of the graph prior (PriorConfig()).
        device: Device of the tensors.
        raw: Skip TabICL's clamp and z-score of the features (and of a regression target).
        **kwargs: Ignored (the batch generator passes the train size too).
    """

    def __init__(
        self,
        regression: bool = False,
        seq_len: int = 1024,
        num_features: int = 100,
        max_features: int = 100,
        num_classes: int = 2,
        permute_features: bool = True,
        permute_labels: bool = True,
        config: PriorConfig | None = None,
        device: str = "cpu",
        raw: bool = True,
        **kwargs,
    ):
        self.regression = regression
        self.seq_len = seq_len
        self.num_features = num_features
        self.max_features = max_features
        self.num_classes = num_classes
        self.permute_features = permute_features
        self.permute_labels = permute_labels
        self.config = config
        self.device = device
        self.raw = raw

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the table.

        Returns:
            X of shape (seq_len, max_features), y of shape (seq_len,) and a boolean mask of shape (max_features,),
            True on the categorical columns (False on the padding). A table whose generation produced NaN comes
            back as zeros with labels -100, as in TabICL (the batch generator discards it).
        """
        context = Context(config=self.config, device=self.device)
        properties = DatasetProperties(
            n_train=self.seq_len,
            n_test=0,
            cat_sizes={
                "x": sample_categorical_sizes(self.num_features, context, max_cat_size=MAX_CATEGORIES),
                "y": [0 if self.regression else self.num_classes],
            },
        )
        data = RandomDataset(context).sample(properties).get_concat_tensors()
        parts = [data[key] for key in ("x_cat", "x_num") if data.get(key) is not None]  # categorical first
        if not parts:
            raise ValueError("No features found in dataset")
        X = torch.cat(parts, dim=-1).float()
        num_categorical = data["x_cat"].shape[-1] if data.get("x_cat") is not None else 0
        categorical = torch.arange(self.num_features, device=X.device) < num_categorical

        if not self.raw:
            X = standard_scaling(outlier_removing(X, threshold=4))

        if self.permute_features:
            permutation = torch.randperm(self.num_features, device=self.device)
            X, categorical = X[..., permutation], categorical[permutation]

        if self.num_features < self.max_features:
            padding = self.max_features - self.num_features
            X = F.pad(X, (0, padding), mode="constant", value=0.0)
            categorical = F.pad(categorical, (0, padding), value=False)

        if self.regression:
            y = data["y_num"].float()
            if not self.raw:  # raw: the target is left for the training loop to standardize
                y = standard_scaling(outlier_removing(y, threshold=4))
            y = y.view(-1)
        else:
            y = data["y_cat"].view(-1)
            if self.permute_labels:
                class_permutation = torch.randperm(self.num_classes, device=self.device)
                y = class_permutation[y.long()]

        if torch.isnan(X).any() or torch.isnan(y).any():
            X = torch.zeros_like(X)
            y = torch.full_like(y, -100.0)
        return X, y, categorical
