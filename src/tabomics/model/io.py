# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Saving and loading models.

Two formats:

- **Release** (what is published): a directory with `model.safetensors` (the weights: only tensors, so loading it
  cannot run code, unlike a pickled `.pth`) and `config.json` (the TabOmicsConfig: architecture, feature clip,
  column permutation, plus the task).
- **Training checkpoint** (what the training code writes): a `.pth` file with `"architecture"` and `"model"`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tabomics.model.model import TabOmicsConfig, TabOmicsModel

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
# Checkpoints written before the feature clip was stored were trained and evaluated with this clip.
LEGACY_FEATURE_CLIP = 100.0


def save_pretrained(model: TabOmicsModel, directory: str | Path, task: str = "classification") -> Path:
    """Writes a model as a release: `model.safetensors` and `config.json` in `directory` (created if needed).

    Args:
        model: The model.
        directory: Where to write.
        task: "classification" or "regression".

    Returns:
        The directory.
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(weights, directory / WEIGHTS_FILE)
    config = {"model_type": "tabomics", "task": task, **model.config.to_dict()}
    (directory / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
    return directory


def load_pretrained(path: str | Path) -> TabOmicsModel:
    """Loads a release.

    Args:
        path: A directory with `model.safetensors` and `config.json`, or the `.safetensors` file itself (with its
            `config.json` next to it).

    Returns:
        The model, in evaluation mode, on the CPU.
    """
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    weights_file = path if path.is_file() else directory / WEIGHTS_FILE
    config = json.loads((directory / CONFIG_FILE).read_text())
    model = TabOmicsModel(TabOmicsConfig.from_dict(config))
    model.load_state_dict(load_file(weights_file))
    return model.eval()


def load_checkpoint(path: str | Path) -> TabOmicsModel:
    """Loads a model from a training checkpoint (`.pth`).

    The file is a dictionary with the architecture (`"architecture"`) and the weights (`"model"`); training
    checkpoints also hold an optimizer state, which is ignored. Only tensors and plain values are read
    (`weights_only=True`), so loading cannot run code.

    Returns:
        The model, in evaluation mode, on the CPU.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    architecture = {"feature_clip": LEGACY_FEATURE_CLIP, **state["architecture"]}
    model = TabOmicsModel(TabOmicsConfig.from_dict(architecture))
    model.load_state_dict(state["model"])
    return model.eval()


def load_model(source: TabOmicsModel | str | Path, cache_dir: str | Path | None = None) -> TabOmicsModel:
    """Any model source to a model.

    Args:
        source: A TabOmicsModel (returned as is); the name of a published model ("v2.5", see tabomics.hub);
            a release directory or `.safetensors` file; or a training checkpoint (`.pth` / `.pt`).
        cache_dir: Where published models are downloaded (default: Hugging Face's cache).

    Returns:
        The model.
    """
    if isinstance(source, TabOmicsModel):
        return source
    from tabomics import hub  # imported here: hub needs huggingface_hub only when a name is used

    if isinstance(source, str) and source in hub.PUBLISHED_MODELS:
        return load_pretrained(hub.download(source, cache_dir=cache_dir))
    path = Path(source)
    if not path.exists():
        names = ", ".join(sorted(hub.PUBLISHED_MODELS))
        raise FileNotFoundError(f"{source!r} is neither a file nor a published model ({names}).")
    if path.is_dir() or path.suffix == ".safetensors":
        return load_pretrained(path)
    return load_checkpoint(path)
