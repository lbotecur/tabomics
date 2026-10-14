# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Pretrains TabOmics on synthetic tables, one stage per recipe (configs/).

    # a stage from scratch
    python -u scripts/train.py --config configs/v2/1_base.json --run-dir runs/v2_base --device cuda:0
    # the next stage, from the last one's checkpoint (or set init_from in the recipe)
    python -u scripts/train.py --config configs/v2/2_rows2048.json --run-dir runs/v2_rows2048 --device cuda:0 \
        --set init_from=runs/v2_base/epoch_100.pth
    # continue an interrupted run where it stopped
    python -u scripts/train.py --config configs/v2/1_base.json --run-dir runs/v2_base --device cuda:0 --resume
    # change anything in the recipe: the architecture, the prior, the learning rate...
    python -u scripts/train.py --config configs/v2/1_base.json --run-dir runs/small --set model.num_layers=6 \
        --set training.lr=3e-4

The run directory gets latest_checkpoint.pth (every epoch), epoch_<n>.pth snapshots, steps.csv (loss, waiting for
the prior, gradient norms) and the recipe as run (recipe.json, plus one record per launch with the command and the
git commit). scripts/export.py turns a checkpoint into a release.
"""

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # before torch is imported

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from tabomics.model import TabOmicsModel  # noqa: E402
from tabomics.model.io import load_model  # noqa: E402
from tabomics.training import Recipe, load_training_state, train  # noqa: E402
from tabomics.training.trainer import LATEST  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--config", required=True, help="recipe JSON (configs/)")
parser.add_argument("--run-dir", required=True, help="where checkpoints and logs go")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a recipe value, e.g. training.lr=3e-4 (repeatable)")  # fmt: skip
parser.add_argument("--resume", action="store_true", help="continue from the run directory's latest checkpoint")
args = parser.parse_args()
recipe = Recipe.load(args.config, args.set)
seed = recipe.training.seed

# The prior's tables depend on Python's string hashing, random per process unless PYTHONHASHSEED is set: with a
# seed, restart this process once with it set, so that every launch generates the same tables.
if seed is not None and os.environ.get("PYTHONHASHSEED") != str(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])

run_dir = Path(args.run_dir)
run_dir.mkdir(parents=True, exist_ok=True)
started = datetime.now().strftime("%Y%m%d-%H%M%S")
repo = Path(__file__).resolve().parents[1]


def git(*command: str) -> str:
    """Output of a git command in this repository (empty outside a git checkout)."""
    return subprocess.run(["git", "-C", str(repo), *command], capture_output=True, text=True).stdout.strip()


record = {
    "recipe": recipe.to_dict(),
    "command": " ".join(sys.argv),
    "started": started,
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
}
(run_dir / f"launch_{started}.json").write_text(json.dumps(record, indent=2))
if not (run_dir / "recipe.json").exists():
    (run_dir / "recipe.json").write_text(json.dumps(recipe.to_dict(), indent=2))

resume = load_training_state(run_dir / LATEST) if args.resume else None
loader = recipe.make_loader(epoch=resume["epoch"] if resume else 0)  # a seeded prior continues its sequence
if seed is not None:
    torch.manual_seed(seed)  # initial weights, and the augmentations' draws
model = TabOmicsModel(recipe.model).to(args.device)
if resume is not None:
    model.load_state_dict(resume["model"])
elif recipe.init_from:
    model.load_state_dict(load_model(recipe.init_from).state_dict())  # weights of an earlier stage
model.gradient_checkpointing = recipe.training.gradient_checkpointing
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, feature clip {model.feature_clip:g}")

t = recipe.training
train(
    model,
    loader,
    run_dir,
    t.epochs,
    lr=t.lr,
    accumulate=t.accumulate,
    warmup_steps=t.warmup_steps,
    grad_clip=t.grad_clip,
    augmentation=recipe.augmentation,
    device=args.device,
    amp_dtype=getattr(torch, t.amp_dtype) if t.amp_dtype else None,
    snapshot_every=t.snapshot_every,
    log_every=t.log_every,
    resume=resume,
)
print(f"Done. Checkpoint: {run_dir / LATEST}")
