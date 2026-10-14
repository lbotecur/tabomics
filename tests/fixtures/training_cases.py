"""The small training run compared by test_training.py with TFM-Playground's train(): fixed batches (no prior, so
that the comparison does not depend on the environment), widening and missing values on, two epochs."""

import torch

SMALL = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)
SETTINGS = dict(epochs=2, accumulate=2, lr=1e-3, warmup_steps=3, grad_clip=1.0)
WIDENING = dict(add_features_min=5, add_features_max=40, prob_no_widening=0.3, sparse_columns_prob=0.5,
                sparse_columns_max_fraction=0.8)  # fmt: skip
MISSING_RATE_MAX = 0.1
SEED = 0


def batches() -> list[dict]:
    """Four batches of 3 tables (60 rows, 6 columns, 2 of them categorical), the first 40 rows for training."""
    g = torch.Generator().manual_seed(123)
    out = []
    for _ in range(4):
        x = torch.randn(3, 60, 6, generator=g)
        categorical = torch.zeros(3, 6, dtype=torch.bool)
        categorical[:, :2] = True
        x[:, :, :2] = torch.randint(0, 4, (3, 60, 2), generator=g).float()
        y = (x[:, :, 2] + x[:, :, 0] > 1).float() + (x[:, :, 3] > 0).float()  # 3 classes
        out.append({"x": x, "y": y, "train_size": 40, "categorical": categorical})
    return out


class Batches:
    """The batches as a loader of one epoch (TabOmics' keys)."""

    def __init__(self, data):
        self.data, self.num_steps = data, len(data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return self.num_steps
