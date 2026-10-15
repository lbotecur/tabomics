"""The examples run (each with --quick: a tiny synthetic table, and a tiny random model where one is needed)."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tabomics.model import TabOmicsConfig, TabOmicsModel
from tabomics.model.io import save_pretrained

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("model") / "release"
    small = TabOmicsConfig(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2)
    save_pretrained(TabOmicsModel(small), path)
    return path


@pytest.mark.parametrize(
    "example",
    ["01_classify.py", "02_embeddings.py", "03_interpretability.py", "04_train_and_continue.py", "05_finetune.py"],
)
def test_example_runs(example, tiny_model, tmp_path):
    command = [sys.executable, str(EXAMPLES / example), "--quick", "--device", "cpu", "--model", str(tiny_model)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
