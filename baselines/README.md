# Baselines

Other models evaluated on the same folds as TabOmics (`scripts/evaluate.py`), for comparison. They are not installed
with TabOmics: install the requirements of the ones to run, preferably each group in its own environment, since
their packages pin different versions.

| Name | Model | Requirements |
|---|---|---|
| `random_forest` | Random forest, scikit-learn defaults; missing values imputed with the mode (as TabPFN-Wide [1]) | none |
| `logreg` | Logistic regression on standardized features | none |
| `logreg_en` | Elastic-net logistic regression, C chosen by 3-fold CV on the training rows; one-hot categorical columns | none |
| `xgboost` | XGBoost, 1000 trees, histogram method (as TabPFN-Wide [1]) | `classic.txt` |
| `tabpfn-wide-5k` | TabPFN-Wide [1], its 5k-feature model, one feature per group | `tabpfn.txt` |
| `tabpfn-v2-gn2p4bpt` | TabPFN v2 [2] checkpoint trained with one feature per group | `tabpfn.txt` |
| `tabpfn-3.5` | TabPFN-3.5 | `tabpfn.txt` |
| `tabicl-v2`, `tabicl-v1.1` | TabICL [3, 4] | `tabicl.txt` |

The foundation models run one forward pass by default (one ensemble member), as TabOmics; a suffix changes it:
`:n8` for 8 members, `:auto` for the package's default, and for TabPFN-3.5, which sees at most 768 features per
member, `:cover` for enough members (at least 8) for every feature to be seen.

```bash
pip install -r baselines/requirements/classic.txt
python baselines/evaluate.py --model xgboost --suite hdlss mlomics --data-root /path/to/data
python baselines/evaluate.py --model tabpfn-3.5:cover --suite hdlss --data-root /path/to/data --device cuda:1
```

Each run writes one row per fold to `results/<model>.csv` (resumable), with the same columns as
`scripts/evaluate.py`; `tabomics.benchmarks.summarize` averages them per dataset and suite.

## References

[1] C. Kolberg, J. Kreuer, J. Huurdeman, S. Ouaari, K. Eggensperger and N. Pfeifer. *TabPFN-Wide: Continued
Pre-Training for Extreme Feature Counts*. arXiv:2510.06162v2, 2026 (first version 2025). [arxiv.org/abs/2510.06162v2](https://arxiv.org/abs/2510.06162v2)

[2] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister and F. Hutter.
*Accurate predictions on small data with a tabular foundation model*. Nature, 2025.
[doi.org/10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)

[3] J. Qu, D. Holzmüller, G. Varoquaux and M. Le Morvan. *TabICL: A Tabular Foundation Model for In-Context Learning
on Large Data*. ICML, 2025. [arxiv.org/abs/2502.05564](https://arxiv.org/abs/2502.05564)

[4] J. Qu, D. Holzmüller, G. Varoquaux and M. Le Morvan. *TabICLv2: A better, faster, scalable, and open tabular
foundation model*. ICML, 2026. [arxiv.org/abs/2602.11139](https://arxiv.org/abs/2602.11139)
