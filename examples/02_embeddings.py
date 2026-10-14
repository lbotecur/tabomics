# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Row embeddings: a vector per patient, for clustering, plots or another model.

    python examples/02_embeddings.py               # leukemia; writes embeddings.png (needs matplotlib)

The embedding of a row is its representation inside the model just before the classification. A training row is
seen together with its own label, so its embedding "knows" the answer; a test row's does not. Embeddings of training
rows taken directly are therefore not comparable with those of new rows, and a model trained on them would be
evaluated on leaked labels. out_of_fold_embeddings embeds every row as a test row: the rows are split into folds and
each fold is embedded with the other folds as context.
"""

import numpy as np
from common import classifier_settings, load_data, parser
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tabomics import Classifier, out_of_fold_embeddings

args = parser(__doc__.split("\n")[0]).parse_args()
X, y = load_data(args)
clf = Classifier(**classifier_settings(args))

# Every row embedded as a test row (out of fold); the classifier is then fitted on all rows, ready for new ones
embeddings = out_of_fold_embeddings(clf, X, y, n_folds=3 if args.quick else 10, shuffle=True, random_state=0)
print(f"embeddings: {embeddings.shape}")  # (rows, 192)

# Leaked: training rows embedded with their own labels in the context
leaked = clf.fit(X, y).embeddings(X)

# How much each set tells about the labels, measured honestly (cross-validated logistic regression on top)
for name, values in [("out of fold", embeddings), ("leaked", leaked)]:
    score = cross_val_score(LogisticRegression(max_iter=2000), values, y, cv=3).mean()
    print(f"{name:12s} embeddings -> logistic regression accuracy {score:.3f}")
print("If the leaked ones score higher, it is only because they were made with the labels.")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib to plot them")
else:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, values) in zip(axes, [("out of fold", embeddings), ("leaked", leaked)], strict=True):
        points = PCA(n_components=2).fit_transform(values)
        for label in np.unique(y):
            ax.scatter(*points[y == label].T, s=15, label=f"class {label}")
        ax.set_title(f"{name} embeddings (PCA)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig("embeddings.png", dpi=150)
    print("wrote embeddings.png")
