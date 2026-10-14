# Recipes of TabOmics v2

The four pretraining stages that produced the published v2, from the arguments of the original runs (TFM-Playground,
where TabOmics was developed). Each stage starts from the previous one's weights (`init_from`) with a new optimizer.

| Stage | Tables | Columns | Rows | Batch | lr | Epochs | Seed |
|---|---|---|---|---|---|---|---|
| 1_base | graph prior | 2-100 | 40-1024 | 8 | 1e-4 | 100 | 42 |
| 2_rows2048 | graph prior | 2-100 | 40-2048 | 8 | 1e-4 | 100 | 1100 |
| 3_widening | graph prior, widened (70% of batches) by 200-3000 columns | 2-350 (+ added) | 40-1024 | 1 x 8 | 1e-4 | 30 | 1100 |
| 4_low_lr | as stage 3 | | | 1 x 8 | 3e-5 | 30 | 1030 |

All stages: TabICL's tables as TabICL gives them (`raw: false`: clamped and z-scored; TabOmics v3 uses raw
tables), rows drawn log-uniformly, graphs whose labels depend on no feature resampled, 30-90% training rows, up to
10% missing values, 1000 batches per epoch, schedule-free AdamW with 500 warmup steps, gradient norm clipped at 1,
bf16, gradient checkpointing.

The model was trained with feature clip 100 (the clip of the standardized feature values, fixed at the time); the
published v2 is used with 10, which only bounds values outside the training distribution (see the paper). So the
last checkpoint is exported with `--feature-clip 10`:

```bash
python -u scripts/train.py --config configs/v2/1_base.json --run-dir runs/v2/1_base --device cuda:0
python -u scripts/train.py --config configs/v2/2_rows2048.json --run-dir runs/v2/2_rows2048 --device cuda:0
python -u scripts/train.py --config configs/v2/3_widening.json --run-dir runs/v2/3_widening --device cuda:0
python -u scripts/train.py --config configs/v2/4_low_lr.json --run-dir runs/v2/4_low_lr --device cuda:0
python scripts/export.py --checkpoint runs/v2/4_low_lr/epoch_30.pth --output release/tabomics-v2 --name v2 \
    --feature-clip 10
```

With the same seeds and number of workers the tables and augmentations are the same as in the original runs; training
on a GPU is not bit-for-bit deterministic, so the weights match statistically, not exactly. The original runs had
uncommitted changes (`git_dirty`), so the code of the time is not exactly known either.
