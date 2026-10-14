"""Writes parity_widening.npz: batches augmented by TFM-Playground's training loop (widening, sparse columns and
missing values; tfmplayground/train.py and augmentation.py, where TabOmics was developed) for test_widening.py,
which checks that tabomics.prior.augment_batch gives the same tables. Run from the repository root with the path
of a TFM-Playground checkout:
    python tests/fixtures/make_widening_fixture.py ../TFM-Playground
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from widening_cases import CASES, case_input  # noqa: E402


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root = Path(sys.argv[1]) / "tfmplayground"
A = load(root / "augmentation.py", "tfm_augmentation")


def shorten_table(x, y, target_y, split, max_rows):  # tfmplayground/train.py
    rows = x.shape[1]
    n_train = min(max(2, round(split * max_rows / rows)), split, max_rows - 1)
    n_test = min(max_rows - n_train, rows - split)
    keep = torch.cat([torch.arange(n_train), torch.arange(split, split + n_test)])
    return x[:, keep.to(x.device)], y[:, keep], target_y[:, keep], n_train


def training_loop_block(x, y, split, w, missing_rate_max, masks):  # the augmentation lines of train.py's loop
    target = y.clone()
    widen = w["add_features_max"] > 0 and (w["prob_no_widening"] == 0 or torch.rand(1).item() >= w["prob_no_widening"])
    if widen and w["max_rows"] and x.shape[1] > w["max_rows"]:
        x, y, target, split = shorten_table(x, y, target, int(split), w["max_rows"])
    if widen:
        num_add = int(torch.randint(w["add_features_min"], w["add_features_max"] + 1, (1,)).item())
        sparsity = torch.rand(1).item() * w["sparsity_max"]
        noise = torch.rand(1).item() * w["noise_max"]
        x = A.add_mixed_widening_features(
            x,
            num_add,
            sparsity,
            noise,
            max_cats=w["max_cats"],
            include_original_prob=w["include_original_prob"],
            categorical_masks=masks,
            recode_categoricals=masks is not None,
        )
        if w["sparse_columns_prob"] > 0 and torch.rand(1).item() < w["sparse_columns_prob"]:
            x = A.sparsify_columns(x, torch.rand(1).item() * w["sparse_columns_max_fraction"])
    if missing_rate_max > 0:
        rate = torch.rand(1, device=x.device).item() * missing_rate_max
        x = A.inject_mcar_missingness(x, rate)
    return x, y, split


out = {}
for name, case in CASES.items():
    x, y, masks = case_input(case)
    torch.manual_seed(case["seed"])
    w = {("max_rows" if k == "widen_max_rows" else k): v for k, v in case["config"].items()}
    xa, ya, split = training_loop_block(x, y, case["train_size"], w, w.pop("missing_rate_max"), masks)
    out |= {f"{name}/x": xa.numpy(), f"{name}/y": ya.numpy(), f"{name}/train_size": np.array(split)}
np.savez_compressed(Path(__file__).parent / "parity_widening.npz", **out)
print("wrote parity_widening.npz")
