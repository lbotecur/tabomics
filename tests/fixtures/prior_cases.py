"""The batches compared by test_prior.py with TabICL's graph prior (fixture written by make_prior_fixture.py).

`generate` takes the generator and configuration classes, TabICL's or TabOmics', so both sides draw exactly the
same batches. Generation depends on Python's hash seed: run it with PYTHONHASHSEED=0.
"""

import random

import numpy as np
import torch

CASES = {
    "classification": dict(
        settings=dict(batch_size=4, batch_size_per_gp=2, min_features=2, max_features=10, max_classes=4,
                      min_seq_len=40, max_seq_len=80),
        config=dict(),
    ),
    "filtered": dict(
        settings=dict(batch_size=4, batch_size_per_gp=4, batch_size_per_subgp=2, min_features=3, max_features=12,
                      max_classes=10, max_seq_len=90, log_seq_len=True, log_n_features=True),
        config=dict(filter_unpredictable_datasets=True, filter_unpredictable_graphs=True, fct_types="default,tree"),
    ),
    "regression": dict(
        settings=dict(regression=True, batch_size=3, batch_size_per_gp=3, max_features=6, max_seq_len=60,
                      min_train_size=20, max_train_size=40),
        config=dict(),
    ),
}  # fmt: skip
NAMES = ["X", "y", "num_features", "seq_lens", "train_sizes"]


def generate(prior_class, config_class, **extra) -> dict:
    """{case/name: array} of one batch per case, with the global generators seeded by the case's index."""
    out = {}
    for index, (case, spec) in enumerate(CASES.items()):
        random.seed(index)
        np.random.seed(index)
        torch.manual_seed(index)
        prior = prior_class(config=config_class(**spec["config"]), **spec["settings"], **extra)
        batch = prior.get_batch()
        # TabOmics adds the categorical masks as a sixth tensor: not compared
        for name, tensor in zip(NAMES, batch, strict=False):
            out[f"{case}/{name}"] = tensor.numpy()
    return out
