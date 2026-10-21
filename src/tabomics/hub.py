# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Published models, downloaded from Hugging Face.

Each name points to a fixed commit of its Hugging Face repository and to the SHA-256 of its weights, never to
"the latest file": a model name always gives the same weights, and results never change silently. The default
model is fixed by the version of the package; a new model comes with a new version.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from tabomics.model.io import CONFIG_FILE, WEIGHTS_FILE


@dataclass(frozen=True)
class PublishedModel:
    """A model on Hugging Face.

    Attributes:
        repo_id: Hugging Face repository.
        revision: Commit of that repository.
        sha256: SHA-256 of its `model.safetensors`.
        task: "classification" or "regression".
        description: One line about the model.
    """

    repo_id: str
    revision: str | None
    sha256: str | None
    task: str
    description: str


PUBLISHED_MODELS: dict[str, PublishedModel] = {
    "v2": PublishedModel(
        repo_id="lbotecur/tabomics-v2",
        revision="9fb540dc21e0feb7f6550f87d9896a26ed9d9435",
        sha256="bedc7594adf4e11ce4292f450cd01fc7be5955fc263f1e9b9eca960720d834e6",
        task="classification",
        description="TabOmics v2: 7.3M parameters, trained on synthetic tables of up to 3350 columns.",
    ),
    "v2.5": PublishedModel(
        repo_id="lbotecur/tabomics-v2.5",
        revision="5b80160242101a612158820852125b2fb95940ea",
        sha256="9d3d93560f029dea44678a387c3fdd03776ea95ebf3e59861829b2f974f4dac7",
        task="classification",
        description="TabOmics v2.5: the architecture of v2 with a longer base stage of training (configs/v2.5).",
    ),
}
DEFAULT_CLASSIFIER = "v2.5"


def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file, as hexadecimal."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, cache_dir: str | Path | None = None) -> Path:
    """Downloads a published model (once: later calls use the cache, also offline) and checks its weights.

    Args:
        name: Its name, e.g. "v2.5".
        cache_dir: Where to keep it; default Hugging Face's cache (`~/.cache/huggingface/hub`, or `HF_HOME`).

    Returns:
        The directory with `model.safetensors` and `config.json`.
    """
    if name not in PUBLISHED_MODELS:
        raise KeyError(f"Unknown model {name!r}; published models: {', '.join(sorted(PUBLISHED_MODELS))}.")
    spec = PUBLISHED_MODELS[name]
    if spec.revision is None:
        raise RuntimeError(f"Model {name!r} is not published yet.")
    from huggingface_hub import hf_hub_download

    files = {
        filename: Path(hf_hub_download(spec.repo_id, filename, revision=spec.revision, cache_dir=cache_dir))
        for filename in (WEIGHTS_FILE, CONFIG_FILE)
    }
    if spec.sha256 is not None and sha256_of(files[WEIGHTS_FILE]) != spec.sha256:
        raise RuntimeError(f"The downloaded weights of {name!r} do not match their published SHA-256.")
    return files[WEIGHTS_FILE].parent
