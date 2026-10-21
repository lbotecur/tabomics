# Recipes of TabOmics v2

The four pretraining stages that produced the published v2, from the arguments of the original runs (TFM-Playground,
where TabOmics was developed). Each stage starts from the previous one's weights (`init_from`) with a new optimizer.

| Stage | Tables | Columns | Rows | Batch | lr | Epochs | Seed |
|---|---|---|---|---|---|---|---|
| 1_base | graph prior | 2-100 | 40-1024 | 8 | 1e-4 | 100 | 42 |
| 2_rows2048 | graph prior | 2-100 | 40-2048 | 8 | 1e-4 | 100 | 1100 |
| 3_widening | graph prior, widened (30% of batches) by 200-3000 columns | 2-350 (+ added) | 40-1024 | 1 x 8 | 1e-4 | 30 | 1100 |
| 4_low_lr | as stage 3 | | | 1 x 8 | 3e-5 | 30 | 1030 |

All stages: TabICL's tables as TabICL gives them (`raw: false`: clamped and z-scored; TabOmics v3 uses raw
tables), rows drawn log-uniformly, graphs whose labels depend on no feature resampled, 30-90% training rows, up to
10% missing values, 1000 batches per epoch, schedule-free AdamW with 500 warmup steps, gradient norm clipped at 1,
bf16, gradient checkpointing. Widening detects categorical columns as those with at most 20 distinct values
(`"widening_categorical": "detect"`), as the original runs did; the categorical masks of the graph prior are not used.

The model was trained with feature clip 100 (the clip of the standardized feature values, inherited from
nanoTabPFN); the published v2 is used with 10, which bounds the rare extreme standardized values of sparse real
columns. So the last
checkpoint is exported with `--feature-clip 10`:

```bash
python -u scripts/train.py --config configs/v2/1_base.json --run-dir runs/v2/1_base --device cuda:0
python -u scripts/train.py --config configs/v2/2_rows2048.json --run-dir runs/v2/2_rows2048 --device cuda:0
python -u scripts/train.py --config configs/v2/3_widening.json --run-dir runs/v2/3_widening --device cuda:0
python -u scripts/train.py --config configs/v2/4_low_lr.json --run-dir runs/v2/4_low_lr --device cuda:0
python scripts/export.py --checkpoint runs/v2/4_low_lr/epoch_30.pth --output release/tabomics-v2 --name v2 \
    --feature-clip 10
```

With the same seeds and number of workers the recipes follow the original runs, but they do not reproduce their
exact weights: the original runs were not configured to be bit-for-bit deterministic on the GPU; they were run before
the hash seed of the Python process was fixed, so their tables cannot be regenerated exactly from the seeds alone; and
they had uncommitted changes (`git_dirty`), whose exact source was not archived.
