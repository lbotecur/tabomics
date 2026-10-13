"""Writes parity_prior.npz: batches of TabICL 2.2.0's graph prior (GraphPrior, n_jobs=1) for test_prior.py, which
checks that tabomics.prior with raw=False gives the same tables bit for bit. Needs tabicl==2.2.0; run from the
repository root (it sets PYTHONHASHSEED=0 itself):
    python tests/fixtures/make_prior_fixture.py
"""

import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":  # generation iterates over a set of strings
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])

from importlib.metadata import version  # noqa: E402

import numpy as np  # noqa: E402
from tabicl.prior._dataset import GraphPrior  # noqa: E402
from tabicl.prior.graph_lib._config import PriorConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from prior_cases import generate  # noqa: E402

assert version("tabicl") == "2.2.0", version("tabicl")
np.savez_compressed(Path(__file__).parent / "parity_prior.npz", **generate(GraphPrior, PriorConfig, n_jobs=1))
print("wrote parity_prior.npz")
