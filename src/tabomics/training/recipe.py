# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Training recipes: everything a pretraining stage needs, in one JSON file (configs/).

A recipe has four sections: "model" (TabOmicsConfig: the architecture, which anyone can change for their own
training), "prior" (the synthetic tables), "augmentation" (AugmentationConfig) and "training" (optimization and
bookkeeping), plus "init_from", the checkpoint a later stage starts from. Any value can be overridden from the
command line with a dotted key, e.g. `training.lr=3e-4`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from tabomics.model import TabOmicsConfig
from tabomics.prior import AugmentationConfig, PriorConfig, PriorLoader


@dataclass
class PriorSettings:
    """The synthetic tables (see tabomics.prior.PriorLoader).

    Attributes:
        min_rows: Fewest rows of a table.
        max_rows: Most rows of a table (exclusive).
        min_features: Fewest columns.
        max_features: Most columns.
        max_classes: Most classes (at most the model's num_outputs).
        raw: Tables in the form of inference tables (TabOmics' prior); False gives TabICL's (clamped, z-scored).
        log_seq_len: Draw the rows log-uniformly instead of uniformly.
        min_train_size: Smallest share of training rows.
        max_train_size: Largest share of training rows.
        batch_size_per_gp: Tables sharing their size and hyperparameter draws; the whole batch if None.
        fct_types: Random function types of the graphs (PriorConfig.fct_types); None = TabICL's default.
        filter_graphs: Resample graphs whose target shares no ancestor with any feature.
        filter_datasets: Resample tables an ExtraTrees cannot predict better than a constant.
    """

    min_rows: int = 40
    max_rows: int = 1024
    min_features: int = 2
    max_features: int = 100
    max_classes: int = 10
    raw: bool = True
    log_seq_len: bool = False
    min_train_size: float = 0.3
    max_train_size: float = 0.9
    batch_size_per_gp: int | None = None
    fct_types: str | None = None
    filter_graphs: bool = False
    filter_datasets: bool = False

    def graph_config(self) -> PriorConfig | None:
        """The PriorConfig of the graph prior, or None for its defaults."""
        if not (self.fct_types or self.filter_graphs or self.filter_datasets):
            return None
        return PriorConfig(
            fct_types=self.fct_types or "default",
            filter_unpredictable_graphs=self.filter_graphs,
            filter_unpredictable_datasets=self.filter_datasets,
        )


@dataclass
class TrainingSettings:
    """Optimization and bookkeeping of a stage.

    Attributes:
        epochs: Epochs of the stage.
        steps_per_epoch: Batches per epoch.
        batch_size: Tables per batch.
        accumulate: Batches per optimizer step (effective batch: batch_size x accumulate).
        lr: Learning rate of schedule-free AdamW.
        warmup_steps: Optimizer steps of linear warmup.
        grad_clip: Largest gradient norm.
        amp_dtype: "bfloat16" for mixed precision, None for float32.
        gradient_checkpointing: Recompute each block in the backward pass (less memory, ~30% more compute).
        seed: Fixes the initial weights, the tables and the augmentations; None = random.
        workers: Processes generating tables.
        snapshot_every: Keep epoch_<n>.pth every this many epochs.
        log_every: Log the mean loss every this many batches.
    """

    epochs: int = 100
    steps_per_epoch: int = 1000
    batch_size: int = 8
    accumulate: int = 1
    lr: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp_dtype: str | None = "bfloat16"
    gradient_checkpointing: bool = True
    seed: int | None = None
    workers: int = 0
    snapshot_every: int = 5
    log_every: int = 50


@dataclass
class Recipe:
    """A pretraining stage."""

    model: TabOmicsConfig = field(default_factory=TabOmicsConfig)
    prior: PriorSettings = field(default_factory=PriorSettings)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    init_from: str | None = None

    def to_dict(self) -> dict:
        """Plain dictionary, as in the JSON files."""
        return {
            "model": self.model.to_dict(),
            "prior": asdict(self.prior),
            "augmentation": asdict(self.augmentation),
            "training": asdict(self.training),
            "init_from": self.init_from,
        }

    @classmethod
    def from_dict(cls, values: dict) -> Recipe:
        """Builds a recipe; unknown keys are an error (a typo would otherwise be silently ignored)."""
        sections = {"model": TabOmicsConfig, "prior": PriorSettings, "augmentation": AugmentationConfig,
                    "training": TrainingSettings}  # fmt: skip
        unknown = set(values) - set(sections) - {"init_from", "description"}
        if unknown:
            raise ValueError(f"Unknown recipe sections: {sorted(unknown)}")
        built = {}
        for name, kind in sections.items():
            given = values.get(name, {})
            names = {f.name for f in fields(kind)}
            if set(given) - names:
                raise ValueError(f"Unknown keys in '{name}': {sorted(set(given) - names)}")
            built[name] = kind(**given)
        return cls(**built, init_from=values.get("init_from"))

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Recipe:
        """Reads a recipe JSON file and applies `section.key=value` overrides.

        Values are parsed as JSON if they can be, so 3e-4, true, null and 42 have their types; anything else is a
        string.
        """
        values = json.loads(Path(path).read_text())
        return cls.from_dict(apply_overrides(values, overrides or []))

    def make_loader(self, epoch: int = 0) -> PriorLoader:
        """The prior loader of this stage (epoch: epochs already done, when resuming)."""
        p, t = self.prior, self.training
        return PriorLoader(
            num_steps=t.steps_per_epoch, batch_size=t.batch_size, min_rows=p.min_rows, max_rows=p.max_rows,
            min_features=p.min_features, max_features=p.max_features, max_classes=p.max_classes, raw=p.raw,
            log_seq_len=p.log_seq_len, min_train_size=p.min_train_size, max_train_size=p.max_train_size,
            batch_size_per_gp=p.batch_size_per_gp, config=p.graph_config(), num_workers=t.workers, seed=t.seed,
            epoch=epoch,
        )  # fmt: skip


def apply_overrides(values: dict, overrides: list[str]) -> dict:
    """A copy of the recipe dictionary with `section.key=value` (or `init_from=value`) applied."""
    values = copy.deepcopy(values)
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"Override {override!r} is not key=value")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *path, last = key.split(".")
        target = values
        for part in path:
            target = target.setdefault(part, {})
        target[last] = value
    return values
