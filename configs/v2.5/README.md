# Recipes of TabOmics v2.5

The three pretraining stages that produced the published v2.5, from the arguments of the original runs
(TFM-Playground, where TabOmics was developed). Each stage starts from the previous one's weights (`init_from`) with a
new optimizer.

| Stage | Tables | Columns | Rows | Batch | lr | Epochs | Seed |
|---|---|---|---|---|---|---|---|
| 1_base | graph prior | 2-100 | 40-1024 | 8 | 1e-4 | 300 | 42 |
| 2_widening | graph prior, widened (30% of batches) by 200-3000 columns | 2-350 (+ added) | 40-1024 | 1 x 8 | 1e-4 | 30 | 1100 |
| 3_low_lr | as stage 2 | | | 1 x 8 | 3e-5 | 30 | 1030 |

v2.5 has the architecture of v2 and differs only in its training: its base stage is the same run as v2's, trained to
epoch 300 instead of 100, and it has no stage on 2048 rows; stages 2 and 3 are v2's stages 3 and 4 with the same
settings and seeds. All stages: TabICL's tables as TabICL gives them (`raw: false`: clamped and z-scored), rows drawn
log-uniformly, graphs in which the label shares no ancestor with any feature resampled, 30-90% training rows, up to 10% missing values,
1000 batches per epoch, schedule-free AdamW with 500 warmup steps, gradient norm clipped at 1, bf16, gradient
checkpointing. Widening detects categorical columns as those with at most 20 distinct values
(`"widening_categorical": "detect"`), as the original runs did; the categorical masks of the graph prior are not used.

As v2, the model was trained with feature clip 100 and is used with 10, so the last checkpoint is exported with
`--feature-clip 10`:

```bash
python -u scripts/train.py --config configs/v2.5/1_base.json --run-dir runs/v2.5/1_base --device cuda:0
python -u scripts/train.py --config configs/v2.5/2_widening.json --run-dir runs/v2.5/2_widening --device cuda:0
python -u scripts/train.py --config configs/v2.5/3_low_lr.json --run-dir runs/v2.5/3_low_lr --device cuda:0
python scripts/export.py --checkpoint runs/v2.5/3_low_lr/epoch_30.pth --output release/tabomics-v2.5 --name v2.5 \
    --feature-clip 10
```

With the same seeds and number of workers the recipes follow the original runs, but they do not reproduce their
exact weights: the original runs were not configured to be bit-for-bit deterministic on the GPU; its base stage was
run before the hash seed of the Python process was fixed, so its tables cannot be regenerated exactly from the seeds
alone (stages 2 and 3 were run with it fixed); and the runs had uncommitted changes (`git_dirty`), whose exact source
was not archived.
