# Changes from TabICL

This folder derives from the graph prior of TabICL 2.2.0 (https://github.com/soda-inria/tabicl,
`tabicl/prior`), Copyright (c) 2025, Soda team @ Inria, BSD 3-Clause License (LICENSES/BSD-3-Clause-TabICL.txt).
Every file keeps that licence and says where it comes from in its header.

## Copied unchanged (files with a leading underscore)

Only the imports were changed, from `tabicl.prior.graph_lib.` to `tabomics.prior.graph.`. They are excluded from
the linter, so that they stay as they are.

- `_activation.py`, `_base.py`, `_config.py`, `_converter.py`, `_dataset.py`, `_function.py`, `_graph.py`,
  `_graph_function.py`, `_matrix.py`, `_multi_function.py`, `_node_function.py`, `_points.py`, `_properties.py`,
  `_weights.py`: all of `tabicl/prior/graph_lib/` (its `eval/` plotting scripts and README are not included).
- `_scaling.py`: `torch_nanstd`, `outlier_removing` and `standard_scaling` from `tabicl/prior/_reg2cls.py`.
- `_filter.py`: `should_filter` from `tabicl/prior/_dataset.py`.

## Adapted

- `scm.py`, from `GraphSCM` in `tabicl/prior/_graph_scm.py`:
  - option `raw` (default True): skips the clamp to 4 robust standard deviations and the z-score that TabICL
    applies to every column (and to a regression target) over all the rows. At inference the model gets numeric
    columns as they are and categorical ones as codes 0..K-1, and standardizes with the training rows only, so
    raw training tables have the form of inference tables. With `raw=False` the tables are TabICL's;
  - also returns which columns are categorical (TabICL knows it but drops it), permuted and padded with the
    columns.
- `generator.py`, from `Prior` and `GraphPrior` in `tabicl/prior/_dataset.py`:
  - generation in the calling process only (TabICL's `n_jobs=1`); parallelism comes from data-loader workers;
  - one number of rows and train size per batch (no `seq_len_per_gp`, `replay_small` or nested tensors);
  - the categorical masks are kept aligned through the removal of constant columns, and returned;
  - `get_batch` returns a `PriorBatch` (TabICL's five tensors plus the masks).

With `raw=False` and the same seeds (numpy, torch, random and `PYTHONHASHSEED`), the batches are TabICL's bit for
bit (tests/test_prior.py). With `raw=True` they are bit for bit those of TFM-Playground's `graph_scm_raw`, used to
train TabOmics v3.

## Known behaviour kept for now

- Generation depends on Python's hash seed: `_dataset.py` iterates over a set of strings (the feature groups).
  Fixing it would change the tables; it is kept so that they stay TabICL's.
