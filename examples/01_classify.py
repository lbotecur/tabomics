# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Classify a wide table: fit, predict and cross-validate, as with any scikit-learn classifier.

    python examples/01_classify.py                 # leukemia (72 patients, 7129 genes), downloaded from OpenML
    python examples/01_classify.py --data my.csv --target subtype

TabOmics does not train on your table: `fit` only stores the training rows, and `predict_proba` reads them together
with the test rows in one forward pass of the pretrained model.
"""

import numpy as np
from common import classifier_settings, load_data, parser
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from tabomics import Classifier

args = parser(__doc__.split("\n")[0]).parse_args()
X, y = load_data(args)
print(f"{X.shape[0]} rows, {X.shape[1]} columns, {len(np.unique(y))} classes")

# 1. One split: fit stores the training rows, predict_proba classifies the test rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
clf = Classifier(**classifier_settings(args)).fit(X_train, y_train)
proba = clf.predict_proba(X_test)
print(f"test accuracy {accuracy_score(y_test, clf.predict(X_test)):.3f}")

# 2. Cross-validation, as with any scikit-learn estimator (5 folds, every patient predicted once)
folds = StratifiedKFold(n_splits=3 if args.quick else 5, shuffle=True, random_state=0)
proba = cross_val_predict(Classifier(**classifier_settings(args)), X, y, cv=folds, method="predict_proba")
auroc = roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2 else roc_auc_score(y, proba, multi_class="ovr")
print(f"cross-validated AUROC {auroc:.3f}, accuracy {accuracy_score(y, proba.argmax(axis=1)):.3f}")

# 3. Several ensemble members (n_estimators): each after the first reorders the columns and renumbers the classes,
#    and their probabilities are averaged. A little more accurate, n times slower; one member is the default.
ensemble = Classifier(**classifier_settings(args), n_estimators=4).fit(X_train, y_train)
print(f"test accuracy with 4 members {accuracy_score(y_test, ensemble.predict(X_test)):.3f}")
