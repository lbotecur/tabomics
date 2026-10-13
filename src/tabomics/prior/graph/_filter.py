# SPDX-License-Identifier: BSD-3-Clause
# Copied from TabICL 2.2.0, tabicl/prior/_dataset.py: should_filter only (https://github.com/soda-inria/tabicl),
# Copyright (c) 2025, Soda team @ Inria; see LICENSES/BSD-3-Clause-TabICL.txt. Changes: see CHANGES.md.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor

from tabomics.prior.graph._config import PriorConfig


def should_filter(
        X: torch.Tensor,
        y: torch.Tensor,
        config: PriorConfig,
        *,
        is_classif: bool,
        n_estimators: int = 25,
        n_boot: int = 200,
) -> bool:
    if not (config.remove_trivial_datasets or config.filter_unpredictable_datasets):
        return False

    if is_classif:
        y_int = y.to(torch.long)
        C = int(y_int.max().item() + 1)
        Y = F.one_hot(y_int, num_classes=C).to(torch.float32)  # (n, C)
        if C == 2:
            # drop one dimension to make it faster
            Y = Y[:, :1]
    else:
        Y = y.reshape(-1, 1).to(torch.float32)  # (n, 1)

    # sklearn needs numpy
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # do the squeeze to avoid a sklearn warning
    et = ExtraTreesRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        random_state=1,  # don't set to 0 because it doesn't give OOB scores for all samples for some 133 <= n <= 257
        max_depth=6,  # make it faster
    ).fit(X_np, np.squeeze(Y_np, axis=-1) if Y_np.shape[-1] == 1 else Y_np)

    Yhat = et.oob_prediction_  # (n, d)
    if len(Yhat.shape) == 1:
        Yhat = Yhat[:, None]
    mask = ~np.isnan(Yhat).any(axis=1)  # keep valid OOB rows
    Yv, Yhatv = Y_np[mask], Yhat[mask]

    baseline = Y_np.mean(axis=0, keepdims=True)  # (1, d)
    imp = ((Yv - baseline) ** 2).sum(axis=1) - ((Yv - Yhatv) ** 2).sum(axis=1)

    baseline_mse = ((Yv - baseline) ** 2).sum(axis=1).mean()
    extratrees_mse = ((Yv - Yhatv) ** 2).sum(axis=1).mean()

    # Vectorized bootstrap (no retraining)
    rng = np.random.default_rng(0)
    n = imp.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))

    if config.remove_trivial_datasets and np.sqrt(extratrees_mse) <= config.trivial_dataset_threshold * np.sqrt(baseline_mse):
        return True
    if config.filter_unpredictable_datasets and pval >= 0.05:
        return True
    return False
