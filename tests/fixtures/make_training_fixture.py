"""Writes parity_training.npz: the per-batch losses and final weights of a small run of TFM-Playground's train()
(where TabOmics was developed) for test_training.py, which checks that tabomics.training.train gives the same. Run
from the repository root with the path of a TFM-Playground checkout and its dependencies installed:
    python tests/fixtures/make_training_fixture.py ../TFM-Playground
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from tfmplayground.models.nanotabpfn import NanoTabPFNModel  # noqa: E402
from tfmplayground.train import WideningConfig, train  # noqa: E402
from training_cases import MISSING_RATE_MAX, SEED, SETTINGS, SMALL, WIDENING, Batches, batches  # noqa: E402


class TfmBatches(Batches):  # TFM-Playground's keys
    def __iter__(self):
        for b in self.data:
            yield {"x": b["x"], "y": b["y"], "target_y": b["y"].clone(), "train_test_split_index": b["train_size"],
                   "categorical_mask": b["categorical"]}  # fmt: skip


os.chdir(tempfile.mkdtemp())
torch.manual_seed(SEED)
model = NanoTabPFNModel(**SMALL)
model.feature_encoder.clip = 10.0
model.gradient_checkpointing = True
train(model=model, prior=TfmBatches(batches()), criterion=nn.CrossEntropyLoss(), epochs=SETTINGS["epochs"],
      accumulate_gradients=SETTINGS["accumulate"], lr=SETTINGS["lr"], device=torch.device("cpu"), run_name="run",
      missing_rate_max=MISSING_RATE_MAX, widening=WideningConfig(**WIDENING), warmup_steps=SETTINGS["warmup_steps"],
      log_every=1, step_log_path="steps.csv", grad_clip=SETTINGS["grad_clip"])  # fmt: skip
state = torch.load("workdir/run/latest_checkpoint.pth", weights_only=False)["model"]
losses = pd.read_csv("steps.csv")["loss"].to_numpy()
np.savez_compressed(
    HERE / "parity_training.npz", losses=losses, **{f"weight::{k}": v.numpy() for k, v in state.items()}
)
print("wrote parity_training.npz")
