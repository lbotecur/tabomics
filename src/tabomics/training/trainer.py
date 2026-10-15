# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from TFM-Playground's train() (Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander Pfefferle,
# Apache License 2.0; see LICENSES/Apache-2.0-TFM-Playground.txt). Modified: TabOmics' prior loader and
# augmentations, mixed precision, gradient clipping, warmup, snapshots, step log, resuming; classification only.
"""Pretraining loop: the model learns to classify the test rows of synthetic tables from their training rows.

Each step takes a batch of prior tables, augments it (widening, sparse columns, missing values), runs the model on
all rows with the labels of the training rows only, and minimizes the cross-entropy on the test rows. The
optimizer is schedule-free AdamW (no learning-rate schedule to tune besides a linear warmup); checkpoints are saved
with its averaged weights, which are the ones meant for inference.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import schedulefree
import torch
from torch import nn

from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.prior import AugmentationConfig, augment_batch

LATEST = "latest_checkpoint.pth"


def save_checkpoint(path: Path, model: TabOmicsModel, optimizer, epoch: int) -> None:
    """Writes a training checkpoint: epoch, architecture (the model's config), weights and optimizer state.

    The format of TFM-Playground's checkpoints, so that its tools read these too; tabomics.model.io.load_checkpoint
    loads the model, scripts/export.py turns it into a release.
    """
    torch.save(
        {"epoch": epoch, "architecture": model.config.to_dict(), "model": model.state_dict(),
         "optimizer": optimizer.state_dict()},
        path,
    )  # fmt: skip


def load_training_state(path: str | Path) -> dict:
    """Reads a training checkpoint (with its optimizer state) to resume a run."""
    return torch.load(path, map_location="cpu", weights_only=False)


def model_from_checkpoint(state: dict) -> TabOmicsModel:
    """The model of a training checkpoint, with its weights."""
    model = TabOmicsModel(TabOmicsConfig.from_dict(state["architecture"]))
    model.load_state_dict(state["model"])
    return model


def train(
    model: TabOmicsModel,
    loader,
    run_dir: str | Path,
    epochs: int,
    *,
    lr: float = 1e-4,
    accumulate: int = 1,
    warmup_steps: int = 0,
    grad_clip: float = 1.0,
    augmentation: AugmentationConfig | None = None,
    device: str | torch.device = "cpu",
    amp_dtype: torch.dtype | None = None,
    snapshot_every: int = 0,
    log_every: int = 0,
    resume: dict | None = None,
    on_epoch_end: Callable[[int, float], bool] | None = None,
) -> float:
    """Pretrains a model on the batches of a prior loader.

    Args:
        model: The model (its weights are trained in place).
        loader: Iterable over the batches of one epoch, each a dict with "x" (B, T, H), "y" (B, T),
            "train_size" and "categorical" (B, H) (a PriorLoader), with `len` = batches per epoch.
        run_dir: Where the checkpoints and the step log go.
        epochs: Last epoch to train (a resumed run continues from its checkpoint's epoch).
        lr: Learning rate.
        accumulate: Batches whose gradients are summed per optimizer step; must divide the batches per epoch. A
            batch with missing labels is skipped; if that leaves the last group of an epoch short, it still makes
            a step, with its gradients rescaled to the mean over the batches it has.
        warmup_steps: Optimizer steps of linear warmup of the learning rate.
        grad_clip: Largest gradient norm.
        augmentation: Widening, sparse columns and missing values; None = none.
        device: Where the model trains.
        amp_dtype: Mixed precision (e.g. torch.bfloat16: matrix products in bf16, weights in fp32); None = fp32.
        snapshot_every: Also keep epoch_<n>.pth every this many epochs (latest_checkpoint.pth is overwritten every
            epoch); 0 = no snapshots.
        log_every: Print, and append to steps.csv, the mean loss every this many batches; 0 = only per epoch.
        resume: A training checkpoint (load_training_state) to continue from: optimizer state and epoch; the model
            must already hold its weights.
        on_epoch_end: Called after each epoch with (epoch, mean loss), the model holding the weights just saved
            (the optimizer's averages, in evaluation mode); returning True stops training (e.g. early stopping).

    Returns:
        The mean loss of the last epoch.
    """
    if len(loader) % accumulate:
        raise ValueError(f"accumulate={accumulate} must divide the {len(loader)} batches per epoch.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    augmentation = augmentation or AugmentationConfig()
    device = torch.device(device)
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0, warmup_steps=warmup_steps)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
    criterion = nn.CrossEntropyLoss()

    steps_log_path = run_dir / "steps.csv"
    new_log = not steps_log_path.exists()
    mean_loss = float("nan")
    tables_seen = 0
    with open(steps_log_path, "a") as steps_log:
        if new_log:
            steps_log.write("epoch,batch,tables,loss,data_wait_s,seconds,grad_norm,clipped\n")
        for epoch in range(resume["epoch"] + 1 if resume else 1, epochs + 1):
            start = time.time()
            model.train()
            optimizer.train()
            optimizer.zero_grad()
            total_loss, num_batches, skipped = 0.0, 0, 0
            window_loss, window_wait, window_norms = 0.0, 0.0, []
            batches = iter(loader)
            while True:
                fetch = time.time()
                batch = next(batches, None)
                if batch is None:
                    break
                window_wait += time.time() - fetch
                x, y, train_size = augment_batch(
                    batch["x"].to(device), batch["y"], batch["train_size"], augmentation, batch.get("categorical")
                )
                y_train = y[:, :train_size].to(device)
                if torch.isnan(y_train).any():  # missing features are the model's job; missing labels are not
                    skipped += 1
                    continue
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(x, y_train, train_size)
                targets = y[:, train_size:].to(device).reshape(-1).long()
                loss = criterion(logits.float().reshape(-1, logits.shape[-1]), targets) / accumulate
                loss.backward()
                batch_loss = loss.item() * accumulate
                total_loss += batch_loss
                window_loss += batch_loss
                num_batches += 1
                tables_seen += x.shape[0]
                if log_every and num_batches % log_every == 0:
                    elapsed = time.time() - start
                    norms = torch.tensor(window_norms or [float("nan")])
                    print(
                        f"epoch {epoch} | batch {num_batches}/{len(loader)} | loss {window_loss / log_every:.4f} | "
                        f"{elapsed:.0f}s | data wait {window_wait:.0f}s",
                        flush=True,
                    )
                    steps_log.write(
                        f"{epoch},{num_batches},{tables_seen},{window_loss / log_every:.5f},{window_wait:.2f},"
                        f"{elapsed:.1f},{norms.mean():.4f},{(norms > grad_clip).float().mean():.3f}\n"
                    )
                    steps_log.flush()
                    window_loss, window_wait, window_norms = 0.0, 0.0, []
                # step on trained batches, not on the loop index, so that a skipped batch does not shift the groups
                if num_batches % accumulate == 0:
                    # the norm before clipping is logged: if most steps are clipped, the clip sets the step size
                    window_norms.append(nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
                    optimizer.step()
                    optimizer.zero_grad()

            # After a skipped batch the last group is short and would never reach the step above; its gradients
            # would be cleared by the next epoch's zero_grad. Step on it instead, rescaled to a mean over the batches
            # it has (each loss was divided by accumulate), so that its step has the scale of the others.
            short = num_batches % accumulate
            if short:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulate / short)
                window_norms.append(nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
                optimizer.step()
                optimizer.zero_grad()

            mean_loss = total_loss / max(num_batches, 1)
            model.eval()
            optimizer.eval()  # the averaged weights: these are saved
            save_checkpoint(run_dir / LATEST, model, optimizer, epoch)
            if snapshot_every and epoch % snapshot_every == 0:
                save_checkpoint(run_dir / f"epoch_{epoch}.pth", model, optimizer, epoch)
            print(
                f"Epoch {epoch:5d} | Time {time.time() - start:5.2f}s | Mean Loss {mean_loss:5.2f} | "
                f"Skipped {skipped} (missing labels)",
                flush=True,
            )
            if on_epoch_end is not None and on_epoch_end(epoch, mean_loss):
                break
    return mean_loss
