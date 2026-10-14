"""Batches and augmentation settings compared by test_widening.py with TFM-Playground's training loop."""

import torch

BASE = dict(add_features_min=0, add_features_max=0, sparsity_max=0.05, noise_max=1.0, include_original_prob=0.5,
            max_cats=20, prob_no_widening=0.0, widen_max_rows=0, sparse_columns_prob=0.0,
            sparse_columns_max_fraction=1.0, missing_rate_max=0.0)  # fmt: skip
CASES = {
    "missing_only": dict(seed=1, train_size=70, with_masks=False, config=dict(BASE, missing_rate_max=0.1)),
    "widen_detected": dict(seed=2, train_size=70, with_masks=False,
                           config=dict(BASE, add_features_min=10, add_features_max=120, missing_rate_max=0.1)),
    "widen_masks_short": dict(seed=3, train_size=70, with_masks=True,
                              config=dict(BASE, add_features_min=10, add_features_max=120, widen_max_rows=60,
                                          include_original_prob=1.0, missing_rate_max=0.1)),
    "widen_sparse": dict(seed=4, train_size=70, with_masks=True,
                         config=dict(BASE, add_features_min=30, add_features_max=120, sparse_columns_prob=1.0,
                                     sparse_columns_max_fraction=0.8, prob_no_widening=0.2, missing_rate_max=0.05)),
}  # fmt: skip


def case_input(case: dict):
    """The batch of a case: 2 tables of 100 rows and 12 columns, some categorical (codes 0..4)."""
    g = torch.Generator().manual_seed(case["seed"])
    x = torch.randn(2, 100, 12, generator=g)
    masks = torch.rand(2, 12, generator=g) < 0.3
    x = torch.where(masks[:, None, :], torch.randint(0, 5, (2, 100, 12), generator=g).float(), x)
    y = torch.randint(0, 3, (2, 100), generator=g).float()
    return x, y, masks if case["with_masks"] else None
