# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Fine-tuning a pretrained model on real tables, with early stopping on held-out rows.

The model keeps its task: it still classifies the query rows of a table from its context rows, in one forward pass.
Fine-tuning only moves its weights towards the tables of a domain, with the same loop as pretraining
(tabomics.training.train) fed with episodes of real tables (tabomics.finetuning.episodes) instead of synthetic ones.

Every table holds out some rows for validation (10% by default, stratified). After each epoch the model classifies
each table's validation rows with all its pool rows as context, as at inference; the mean cross-entropy over the
tables decides when to stop: when it has not improved for `patience` epochs, fine-tuning stops and the weights of
its best epoch are kept. If no epoch improves on the pretrained model, the pretrained weights are returned
unchanged. The test rows of an evaluation must not be among the tables given here.

Defaults follow TabPFN's and TabICL's fine-tuning (learning rate 1e-5, 10% validation, every weight trained) and
Rubachev et al. (2025), who found full fine-tuning with early stopping the best choice and did not see it overfit.
"""

from __future__ import annotations

import contextlib
import copy
import io
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tabomics.finetuning.episodes import EpisodeLoader, FinetuneTable, prepare_table
from tabomics.inference.base import resolve_amp_dtype
from tabomics.model import TabOmicsModel
from tabomics.model.io import load_model
from tabomics.prior import PriorLoader
from tabomics.training import PriorSettings, train
from tabomics.training.trainer import LATEST


@dataclass
class FinetuneResult:
    """What fine-tuning returns.

    Attributes:
        model: The fine-tuned model (weights of the best epoch), in evaluation mode.
        history: One row per epoch (0 = the pretrained model): train_loss (cross-entropy over all the model's
            outputs, as in pretraining), val_loss and val_accuracy (over the table's classes, as at inference).
        best_epoch: Epoch whose weights the model holds (0 = no epoch improved: the pretrained weights).
    """

    model: TabOmicsModel
    history: pd.DataFrame
    best_epoch: int


class MixedLoader:
    """Batches of episodes, some of them replaced by batches of synthetic tables (prior mixing).

    Args:
        episodes: The episodes of the real tables.
        prior: Synthetic batches (a PriorLoader), drawn from as needed.
        prior_fraction: Share of the batches taken from the prior.
        seed: Seed of which batches are synthetic.
    """

    def __init__(self, episodes: EpisodeLoader, prior: PriorLoader, prior_fraction: float, seed: int = 0):
        self.episodes = episodes
        self.prior = prior
        self.prior_fraction = prior_fraction
        self.rng = np.random.default_rng(seed)
        self._prior_batches = None

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        for _ in range(len(self.episodes)):
            if self.rng.random() < self.prior_fraction:
                yield self._next_prior_batch()
            else:
                yield self.episodes.next_batch()

    def _next_prior_batch(self) -> dict:
        batch = next(self._prior_batches, None) if self._prior_batches is not None else None
        if batch is None:  # first call, or the prior loader's epoch is over: start another one
            self._prior_batches = iter(self.prior)
            batch = next(self._prior_batches)
        return batch


@torch.no_grad()
def validation_metrics(
    model: TabOmicsModel,
    tables: Sequence[FinetuneTable],
    device: str | torch.device = "cpu",
    amp_dtype: torch.dtype | None = None,
    num_mem_chunks: int = 8,
) -> tuple[float, float]:
    """Mean cross-entropy and accuracy on the validation rows of the tables, each with its pool as context.

    The model runs as at inference: eval mode, low-memory, all columns, and only the logits of the table's classes.
    Each table counts the same, whatever its number of validation rows.

    Returns:
        (loss, accuracy), means over the tables with validation rows; NaN if none has any.
    """
    device = torch.device(device)
    was_training = model.training
    previous = [block.low_memory for block in model.transformer_blocks]
    model.eval()
    model.set_low_memory(True)
    losses, accuracies = [], []
    try:
        for table in tables:
            if not len(table.y_val):
                continue
            x = torch.from_numpy(np.concatenate([table.X, table.X_val])).unsqueeze(0).to(device)
            y_train = torch.from_numpy(table.y).float().unsqueeze(0).to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model.run(x, y_train, len(table.y), num_mem_chunks=num_mem_chunks).logits
            logits = logits.float().squeeze(0)[:, : table.num_classes]
            target = torch.from_numpy(table.y_val).to(device)
            losses.append(F.cross_entropy(logits, target).item())
            accuracies.append((logits.argmax(dim=1) == target).float().mean().item())
    finally:
        for block, value in zip(model.transformer_blocks, previous, strict=True):
            block.low_memory = value
        model.train(was_training)
    if not losses:
        return float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(accuracies))


def freeze_parameters(model: TabOmicsModel, prefixes: Sequence[str]) -> int:
    """Stops training the parameters whose names start with any of the prefixes.

    Args:
        model: The model.
        prefixes: Name prefixes, e.g. ["feature_encoder", "transformer_blocks.0."] (see model.named_parameters()).

    Returns:
        Number of parameter tensors frozen.
    """
    frozen = 0
    for prefix in prefixes:
        matches = [p for name, p in model.named_parameters() if name.startswith(prefix)]
        if not matches:
            raise ValueError(f"No parameter name starts with {prefix!r}.")
        for parameter in matches:
            frozen += parameter.requires_grad
            parameter.requires_grad_(False)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Every parameter is frozen: nothing to fine-tune.")
    return frozen


def finetune(
    model: TabOmicsModel | str | Path,
    tables: Sequence,
    *,
    validation_fraction: float = 0.1,
    epochs: int = 30,
    steps_per_epoch: int = 64,
    batch_size: int = 1,
    accumulate: int = 8,
    lr: float = 1e-5,
    warmup_fraction: float = 0.1,
    grad_clip: float = 1.0,
    patience: int = 5,
    min_delta: float = 1e-4,
    query_fraction: float = 0.2,
    row_fraction: tuple[float, float] = (0.8, 1.0),
    feature_fraction: tuple[float, float] = (0.5, 1.0),
    max_features: int | None = None,
    freeze: Sequence[str] | None = None,
    prior_fraction: float = 0.0,
    prior: PriorSettings | None = None,
    device: str | torch.device | None = None,
    amp_dtype: torch.dtype | str | None = "auto",
    gradient_checkpointing: bool = True,
    num_mem_chunks: int = 8,
    seed: int = 0,
    verbose: bool = True,
    output: str | Path | None = None,
    **preprocessing,
) -> FinetuneResult:
    """Fine-tunes a pretrained model on real tables, stopping early on their held-out rows.

    For a model of a domain, give the training tables of that domain and evaluate the result on other tables. For
    one dataset, give only its training rows (never its test rows) and evaluate on its test rows.

    Args:
        model: The pretrained model: a published name ("v2.5"), a release, a checkpoint or a TabOmicsModel (which is
            not modified: a copy is fine-tuned).
        tables: The tables: (X, y) pairs, (X, y, X_val, y_val) with validation rows of one's own, or prepared
            FinetuneTable's.
        validation_fraction: Share of each table's rows held out for early stopping (stratified); 0 = no
            validation, no early stopping: every epoch is run and the last weights are kept.
        epochs: Most epochs.
        steps_per_epoch: Batches per epoch; the tables take turns, one table per batch.
        batch_size: Episodes per batch (all of one table). Wide tables need few: memory grows with
            batch_size x rows x columns.
        accumulate: Batches per optimizer step (effective batch: batch_size x accumulate episodes, of several
            tables); must divide steps_per_epoch.
        lr: Learning rate of schedule-free AdamW.
        warmup_fraction: Share of the optimizer steps of linear warmup.
        grad_clip: Largest gradient norm.
        patience: Epochs without improvement of the validation loss before stopping.
        min_delta: Smallest decrease of the validation loss that counts as an improvement.
        query_fraction: Share of an episode's rows to predict.
        row_fraction: Range of the share of a table's pool rows in an episode.
        feature_fraction: Range of the share of a table's columns in an episode.
        max_features: Most columns in an episode (memory); None = no limit. Validation uses all columns.
        freeze: Name prefixes of parameters not to train (see freeze_parameters); None trains every weight.
        prior_fraction: Share of the batches made of synthetic tables, to keep what pretraining taught.
        prior: The synthetic tables (PriorSettings); default PriorSettings().
        device: Where to train; default CUDA if available, else CPU.
        amp_dtype: Mixed precision, as in Classifier: "auto" (the default) runs in bfloat16 on CUDA and in float32
            elsewhere; torch.bfloat16 (or "bfloat16") always in bfloat16; None always in float32.
        gradient_checkpointing: Recompute each block in the backward pass (less memory, more compute).
        num_mem_chunks: Memory chunks of the validation passes (see Classifier).
        seed: Seed of the validation split, the episodes and the prior.
        verbose: Print the losses of every epoch.
        output: Directory for the fine-tuned checkpoint (finetuned.pth, loadable by Classifier), history.csv and
            steps.csv; None = nothing is written.
        **preprocessing: Settings of FeaturePreprocessor (categorical_features, infer_categorical, ...).

    Returns:
        The fine-tuned model, the history and the best epoch.
    """
    if steps_per_epoch % accumulate:
        raise ValueError(f"accumulate={accumulate} must divide steps_per_epoch={steps_per_epoch}.")
    if not 0 <= prior_fraction < 1:
        raise ValueError(f"prior_fraction must be in [0, 1), got {prior_fraction}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = resolve_amp_dtype(amp_dtype, device)

    prepared = [_prepare(table, validation_fraction, seed + i, preprocessing) for i, table in enumerate(tables)]
    model = copy.deepcopy(load_model(model)).to(device)
    for table in prepared:
        if table.num_classes > model.config.num_outputs:
            raise ValueError(f"A table has {table.num_classes} classes; the model supports {model.config.num_outputs}.")
    if freeze:
        freeze_parameters(model, freeze)
    model.gradient_checkpointing = gradient_checkpointing

    torch.manual_seed(seed)
    loader = EpisodeLoader(
        prepared, steps_per_epoch, batch_size=batch_size, query_fraction=query_fraction, row_fraction=row_fraction,
        feature_fraction=feature_fraction, max_features=max_features, seed=seed,
    )  # fmt: skip
    if prior_fraction > 0:
        p = prior or PriorSettings()
        if p.max_classes > model.config.num_outputs:
            raise ValueError(f"prior.max_classes={p.max_classes} exceeds the model's {model.config.num_outputs}.")
        synthetic = PriorLoader(
            num_steps=steps_per_epoch, batch_size=batch_size, min_rows=p.min_rows, max_rows=p.max_rows,
            min_features=p.min_features, max_features=p.max_features, max_classes=p.max_classes, raw=p.raw,
            log_seq_len=p.log_seq_len, min_train_size=p.min_train_size, max_train_size=p.max_train_size,
            batch_size_per_gp=p.batch_size_per_gp, config=p.graph_config(), seed=seed,
        )  # fmt: skip
        loader = MixedLoader(loader, synthetic, prior_fraction, seed=seed)

    def evaluate() -> tuple[float, float]:
        return validation_metrics(model, prepared, device, amp_dtype, num_mem_chunks)

    has_validation = any(len(table.y_val) for table in prepared)
    val_loss, val_accuracy = evaluate()
    history = [{"epoch": 0, "train_loss": float("nan"), "val_loss": val_loss, "val_accuracy": val_accuracy}]
    best = {"epoch": 0, "loss": val_loss, "state": _copy_state(model)}
    log = print if verbose else (lambda *args, **kwargs: None)
    log(f"Pretrained | Val Loss {val_loss:.4f} | Val Acc {val_accuracy:.4f}", flush=True)

    def on_epoch_end(epoch: int, train_loss: float) -> bool:
        val_loss, val_accuracy = evaluate()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
        log(f"Epoch {epoch:5d} | Val Loss {val_loss:.4f} | Val Acc {val_accuracy:.4f}", flush=True)
        if not has_validation:
            best.update(epoch=epoch, state=None)  # the last weights are kept
            return False
        if val_loss < best["loss"] - min_delta:
            best.update(epoch=epoch, loss=val_loss, state=_copy_state(model))
        return epoch - best["epoch"] >= patience

    optimizer_steps = epochs * steps_per_epoch // accumulate
    quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with tempfile.TemporaryDirectory() as scratch, quiet:
        run_dir = Path(output) if output is not None else Path(scratch)
        train(
            model, loader, run_dir, epochs, lr=lr, accumulate=accumulate,
            warmup_steps=round(warmup_fraction * optimizer_steps), grad_clip=grad_clip, device=device,
            amp_dtype=amp_dtype, on_epoch_end=on_epoch_end,
        )  # fmt: skip
        (run_dir / LATEST).unlink(missing_ok=True)  # the last epoch's weights, not the kept ones

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.gradient_checkpointing = False
    model.eval()
    history = pd.DataFrame(history)
    if output is not None:
        torch.save(
            {"epoch": best["epoch"], "architecture": model.config.to_dict(), "model": model.state_dict()},
            Path(output) / "finetuned.pth",
        )
        history.to_csv(Path(output) / "history.csv", index=False)
    log(f"Kept the weights of epoch {best['epoch']}" + (" (the pretrained ones)" if best["epoch"] == 0 else ""))
    return FinetuneResult(model=model, history=history, best_epoch=best["epoch"])


def _prepare(table, validation_fraction: float, random_state: int, preprocessing: dict) -> FinetuneTable:
    """A table as given to finetune, prepared."""
    if isinstance(table, FinetuneTable):
        return table
    if len(table) == 2:
        X, y = table
        return prepare_table(X, y, validation_fraction, random_state=random_state, **preprocessing)
    if len(table) == 4:
        X, y, X_val, y_val = table
        return prepare_table(X, y, X_val=X_val, y_val=y_val, random_state=random_state, **preprocessing)
    raise ValueError("Each table must be (X, y), (X, y, X_val, y_val) or a FinetuneTable.")


def _copy_state(model: TabOmicsModel) -> dict:
    """A copy of the weights, on the CPU."""
    return {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()}
