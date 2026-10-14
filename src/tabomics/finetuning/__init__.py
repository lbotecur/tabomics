# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Fine-tuning a pretrained model on real tables: episodes, the fine-tuning loop and early stopping."""

from tabomics.finetuning.episodes import EpisodeLoader, FinetuneTable, prepare_table, split_context_query
from tabomics.finetuning.loop import (
    FinetuneResult,
    MixedLoader,
    finetune,
    freeze_parameters,
    validation_metrics,
)

__all__ = [
    "EpisodeLoader",
    "FinetuneResult",
    "FinetuneTable",
    "MixedLoader",
    "finetune",
    "freeze_parameters",
    "prepare_table",
    "split_context_query",
    "validation_metrics",
]
