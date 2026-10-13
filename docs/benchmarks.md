# Benchmarks

`scripts/evaluate.py` evaluates a TabOmics model on one or more suites, with fixed folds, and writes one row per
fold to a CSV (it resumes an interrupted run). It does not compare with other models: any scikit-learn-like
estimator can be evaluated on the same folds with `tabomics.benchmarks.run_benchmark`, tuned as its user sees fit.

```bash
python scripts/evaluate.py --model v2 --suite hdlss mlomics --data-root /path/to/data
```

| Suite | Tasks | Folds | Source |
|---|---|---|---|
| `hdlss` | 15 scikit-feature datasets (12 to 22,283 features) | stratified 3-fold x 10 (x 3 with >= 2500 rows), seed 42 | as TabPFN-Wide |
| `mlomics` | 5 TCGA cancers x (mRNA, CNV, methylation, miRNA, all four) | stratified 5-fold, seed 42 | as TabPFN-Wide |
| `shamir` | breast, gbm, sarcoma x (mRNA, methylation, miRNA, all three) | stratified 5-fold, seed 42 | Rappoport & Shamir (2018) |
| `cc18` | OpenML-CC18 tables with <= 10,000 rows, 500 features, 10 classes | stratified 10-fold, seed 42 | OpenML suite 99 |

`hdlss` and `mlomics` are prepared exactly as TabPFN-Wide does, so their folds are TabPFN-Wide's. madelon, arcene and
RELATHE (hdlss) and madelon (left out of cc18) were used for development decisions.

## Data

All suites read from one data folder (`--data-root`):

```
<data_root>/
├── HDLSS/<name>.mat                                   # https://jundongl.github.io/scikit-feature/datasets.html
├── Cancer-Multi-Omics-Benchmark/Main_Dataset/...      # scripts/download_mlomics.py
├── Shamir/<cancer>/{exp,methy,mirna}, Shamir/clinical/clinical/<cancer>
│                                                      # https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html
└── OpenML-CC18/<name>.npz, index.csv                  # scripts/download_cc18.py
```

The data are not redistributed here: each source has its own terms.

## Metrics

AUROC (binary: of class 1; multiclass: one-vs-rest, macro average; NaN when a test fold lacks a class) and accuracy,
per fold, plus the seconds per fold (fit and predict). `summarize` averages per dataset first, so every dataset
weighs the same.
