# Benchmarks

`scripts/evaluate.py` evaluates a TabOmics model on one or more suites, with fixed folds, and writes one row per
fold to a CSV (it resumes an interrupted run). It does not compare with other models: any scikit-learn-like
estimator can be evaluated on the same folds with `tabomics.benchmarks.run_benchmark`, tuned as its user sees fit.

```bash
python scripts/evaluate.py --model v2 --suite hdlss mlomics --data-root /path/to/data
```

| Suite | Tasks | Folds |
|---|---|---|
| `hdlss` | 15 high-dimensional datasets of scikit-feature [1] (12 to 22,283 features), the selection of TabPFN-Wide [2] | stratified 3-fold x 10 repeats (x 3 with >= 2500 rows), seed 42 |
| `mlomics` | 5 TCGA cancers of MLOmics [3] x (mRNA, CNV, methylation, miRNA, all four) | stratified 5-fold, seed 42 |
| `shamir` | breast, gbm and sarcoma of the TCGA benchmark of Rappoport and Shamir [4] x (mRNA, methylation, miRNA, all three) | stratified 5-fold, seed 42 |
| `cc18` | OpenML-CC18 [5] (OpenML suite 99) tables with <= 10,000 rows, 500 features and 10 classes | stratified 10-fold, seed 42 |

`hdlss`, `mlomics` and `shamir` are prepared as in the code of TabPFN-Wide [6], so the folds of `hdlss` and `mlomics`
are theirs. For `shamir`, two differences: classes with fewer patients than folds are left out, and samples are
sorted, since TabPFN-Wide's order (and so its folds) changes from run to run. The `cc18` folds are ours, not OpenML's
predefined splits.

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

## References

[1] J. Li, K. Cheng, S. Wang, F. Morstatter, R. P. Trevino, J. Tang and H. Liu. *Feature Selection: A Data
Perspective*. ACM Computing Surveys, 2017. [doi.org/10.1145/3136625](https://doi.org/10.1145/3136625)

[2] C. Kolberg, J. Kreuer, J. Huurdeman, S. Ouaari, K. Eggensperger and N. Pfeifer. *TabPFN-Wide: Continued
Pre-Training for Extreme Feature Counts*. arXiv:2510.06162, 2025. [arxiv.org/abs/2510.06162](https://arxiv.org/abs/2510.06162)

[3] Z. Yang, R. Kotoge, X. Piao, Z. Chen, L. Zhu, P. Gao, Y. Matsubara, Y. Sakurai and J. Sun. *MLOmics: Cancer
Multi-Omics Database for Machine Learning*. arXiv:2409.02143, 2024. [arxiv.org/abs/2409.02143](https://arxiv.org/abs/2409.02143)

[4] N. Rappoport and R. Shamir. *Multi-omic and multi-view clustering algorithms: review and cancer benchmark*.
Nucleic Acids Research, 2018. [doi.org/10.1093/nar/gky889](https://doi.org/10.1093/nar/gky889)

[5] B. Bischl, G. Casalicchio, M. Feurer, P. Gijsbers, F. Hutter, M. Lang, R. G. Mantovani, J. N. van Rijn and
J. Vanschoren. *OpenML Benchmarking Suites*. NeurIPS Datasets and Benchmarks, 2021.
[arxiv.org/abs/1708.03731](https://arxiv.org/abs/1708.03731)

[6] J. Kreuer (not-a-feature). *TabPFN-Wide*. GitHub repository.
[github.com/not-a-feature/TabPFN-Wide](https://github.com/not-a-feature/TabPFN-Wide)
