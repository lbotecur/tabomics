# TabOmics

TabOmics is a tabular foundation model for wide omics data. Like TabPFN [1], it is pretrained on synthetic tables and
uses in-context learning: given labelled training rows and unlabelled test rows, it predicts the test rows in a single
forward pass, with no training or tuning on your data. It is designed for wide tables (thousands of features, few
samples, as in omics), also works on ordinary tables, and provides row embeddings and feature importance.

## Installation

```bash
pip install git+https://github.com/lbotecur/tabomics.git
```

Python 3.10 or later and PyTorch 2.4 or later; everything else comes with it. The model weights are downloaded from
Hugging Face the first time they are used. While the repositories are private, log in first (`hf auth login`).

## Quick start

```python
from tabomics import Classifier

clf = Classifier().fit(X_train, y_train)  # the published model, v2; fit only stores the training rows
proba = clf.predict_proba(X_test)  # one forward pass over training and test rows
labels = clf.predict(X_test)
```

`X` is a NumPy array or a pandas DataFrame; labels can be any values (up to 10 classes). Missing values are fine
(leave them as NaN). It is a scikit-learn estimator, so `cross_val_score`, pipelines and the like work as usual.

```python
import torch

clf = Classifier(
    device="cuda",  # default: CUDA if available, else CPU
    amp_dtype=torch.bfloat16,  # mixed precision on GPU: much faster on wide tables
    categorical_features=[],  # indices of categorical columns; [] with infer_categorical=False for omics
    infer_categorical=False,  # otherwise numeric columns with <= 10 distinct values are taken as categorical
    n_estimators=1,  # passes averaged (each one shuffles columns and relabels classes); 1 by default
)
```

**Row embeddings.** A vector per row, for clustering, plots or another model. A training row is seen together with
its own label, so embed the training rows out of fold:

```python
from tabomics import out_of_fold_embeddings

train_embeddings = out_of_fold_embeddings(clf, X_train, y_train)  # every row embedded as a test row
test_embeddings = clf.embeddings(X_test)  # clf is then fitted on all training rows
```

**Feature importance.** The attention of the label to each column, averaged over test rows, heads and layers; one score per column:

```python
importance = clf.fit(X_train, y_train).feature_importance(X_test)
```

**Large tables.** Tables of tens of thousands of columns fit on one GPU: inference runs in low-memory mode by
default (`low_memory=True`) and splits each step in pieces (`num_mem_chunks=8`); neither changes the results.

The [examples](examples) show all of this on a real gene-expression table, and the training of a model of your own.

## Models

| Name | Description |
|---|---|
| `v2` (default) | A transformer with 12 layers, an embedding size of 192 and 7.3M parameters, pretrained on synthetic tables. [Model card](https://huggingface.co/lbotecur/tabomics-v2) |

A name points to a fixed commit of the model's Hugging Face repository and to the SHA-256 of its weights, so it
always loads the same model. `Classifier(model=...)` also takes a release directory, a `.safetensors` file, a
training checkpoint (`.pth`) or a `TabOmicsModel`.

## Benchmarks

`scripts/evaluate.py` evaluates a model on fixed folds of the HDLSS, MLOmics, Shamir and OpenML-CC18 suites and
writes one row per fold to a CSV. See [docs/benchmarks.md](docs/benchmarks.md) for the suites and their data.

```bash
python scripts/evaluate.py --model v2 --suite hdlss mlomics --data-root /path/to/data
```

Other models (random forest, XGBoost, TabPFN, TabICL...) are evaluated on the same folds with
[baselines/evaluate.py](baselines), which installs nothing with TabOmics.

## Training your own model

Pretraining runs on synthetic tables only: a generator of random causal graphs, with
widening to thousands of columns. A stage of training is a JSON recipe: architecture, synthetic
tables, augmentations and optimization. `configs/v2/` has the four stages that produced v2 ([how to rerun
them](configs/v2/README.md)); change anything from the command line:

```bash
python scripts/train.py --config configs/v2/1_base.json --run-dir runs/mine --device cuda:0 \
    --set model.num_layers=6 --set training.lr=3e-4
python scripts/export.py --checkpoint runs/mine/latest_checkpoint.pth --output release/mine --name mine
```

or from Python with `tabomics.training.train` (see [examples/04_train_and_continue.py](examples/04_train_and_continue.py)).

## Licence and credits

TabOmics is free software, released under the GNU General Public License v3.0 or later (see [LICENSE](LICENSE)).
Its architecture derives from nanoTabPFN [2, 3], and its code from the TFM-Playground repository [4] (Apache License
2.0); its synthetic-table generator derives from the TabICL prior [5, 6, 7] (BSD 3-Clause License). Its feature
widening, Feature importance method and benchmark protocol follow TabPFN-Wide [8, 9]. See [NOTICE](NOTICE) and [LICENSES](LICENSES).

### References

[1] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister and F. Hutter.
*Accurate predictions on small data with a tabular foundation model*. Nature, 2025.
[doi.org/10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)

[2] A. Pfefferle, J. Hog, L. Purucker and F. Hutter. *nanoTabPFN: A Lightweight and Educational Reimplementation of
TabPFN*. arXiv:2511.03634, 2025. [arxiv.org/abs/2511.03634](https://arxiv.org/abs/2511.03634)

[3] AutoML. *nanoTabPFN*. GitHub repository. [github.com/automl/nanoTabPFN](https://github.com/automl/nanoTabPFN)

[4] A. Pfefferle, J. Hog, S. B. Öztürk, K. Kaya, Z. Türkmen and F. Hutter. *TFM-Playground: A Playground for Tabular
Foundation Models*. GitHub repository. [github.com/automl/TFM-Playground](https://github.com/automl/TFM-Playground)

[5] J. Qu, D. Holzmüller, G. Varoquaux and M. Le Morvan. *TabICL: A Tabular Foundation Model for In-Context Learning
on Large Data*. ICML, 2025. [arxiv.org/abs/2502.05564](https://arxiv.org/abs/2502.05564)

[6] J. Qu, D. Holzmüller, G. Varoquaux and M. Le Morvan. *TabICLv2: A better, faster, scalable, and open tabular
foundation model*. ICML, 2026. [arxiv.org/abs/2602.11139](https://arxiv.org/abs/2602.11139)

[7] The Soda research team, Inria. *TabICL*. GitHub repository.
[github.com/soda-inria/tabicl](https://github.com/soda-inria/tabicl)

[8] C. Kolberg, J. Kreuer, J. Huurdeman, S. Ouaari, K. Eggensperger and N. Pfeifer. *TabPFN-Wide: Continued
Pre-Training for Extreme Feature Counts*. arXiv:2510.06162, 2025. [arxiv.org/abs/2510.06162](https://arxiv.org/abs/2510.06162)

[9] J. Kreuer (not-a-feature). *TabPFN-Wide*. GitHub repository.
[github.com/not-a-feature/TabPFN-Wide](https://github.com/not-a-feature/TabPFN-Wide)
