# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Loading models from files."""

from __future__ import annotations

from pathlib import Path

import torch

from tabomics.model.model import TabOmicsConfig, TabOmicsModel

# Checkpoints written before the feature clip was stored were trained and evaluated with this clip.
LEGACY_FEATURE_CLIP = 100.0


def load_checkpoint(path: str | Path) -> TabOmicsModel:
    """Loads a model from a PyTorch checkpoint written by the training code.

    The file is a dictionary with the architecture (`"architecture"`) and the weights (`"model"`); training
    checkpoints also hold an optimizer state, which is ignored. Only tensors and plain values are read
    (`weights_only=True`), so loading cannot run code.

    Args:
        path: The checkpoint file.

    Returns:
        The model, in evaluation mode, on the CPU.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    architecture = {"feature_clip": LEGACY_FEATURE_CLIP, **state["architecture"]}
    model = TabOmicsModel(TabOmicsConfig.from_dict(architecture))
    model.load_state_dict(state["model"])
    return model.eval()
