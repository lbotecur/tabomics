# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Benchmark suites (HDLSS, MLOmics, Shamir, CC18), with fixed folds, and a runner for any model."""

from tabomics.benchmarks.runner import SUITES, run_benchmark, summarize

__all__ = ["SUITES", "run_benchmark", "summarize"]
