# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Which columns does the model use? Feature importance from the attention of the label to each column.

    python examples/03_interpretability.py         # leukemia: the top genes; writes importance.csv

feature_importance(X_test) returns, for every column, how much the label column attends to it, averaged over the test
rows, the attention heads and the blocks (as TabPFN-Wide). First on a synthetic table whose informative columns are
known, to see that they come out on top; then on the real table, with the stability of the ranking across
ensemble members.
"""

import numpy as np
import pandas as pd
from common import classifier_settings, load_data, parser, synthetic
from scipy.stats import spearmanr, ttest_ind
from sklearn.model_selection import train_test_split

from tabomics import Classifier

args = parser(__doc__.split("\n")[0]).parse_args()

# 1. Known answer: a label made from 5 columns hidden among many noise columns
X, y = synthetic(n_rows=60 if args.quick else 200, n_features=200 if args.quick else 2000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
clf = Classifier(**classifier_settings(args)).fit(X_train, y_train)
importance = pd.Series(clf.feature_importance(X_test), index=X.columns).sort_values(ascending=False)
ranks = [importance.index.get_loc(f"informative_{i}") + 1 for i in range(5)]
print(f"synthetic, {X.shape[1]} columns: the 5 informative ones rank {sorted(ranks)}")
print(importance.head(8).round(5).to_string())

# 2. The real table: the columns the model attends to most
X, y = load_data(args)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
single = pd.Series(Classifier(**classifier_settings(args)).fit(X_train, y_train).feature_importance(X_test), X.columns)

# Stability: with one member a column's score also reflects its position in the table; several members put every
# column at several positions and average. Compare the rankings.
several = Classifier(**classifier_settings(args), n_estimators=4).fit(X_train, y_train)
averaged = pd.Series(several.feature_importance(X_test), X.columns)
top = 20
overlap = len(set(single.nlargest(top).index) & set(averaged.nlargest(top).index))
print(
    f"\nreal table: 1 member vs 4 members, Spearman {spearmanr(single, averaged, nan_policy='omit')[0]:.2f}, "
    f"top-{top} overlap {overlap}/{top}"
)

# A sanity check against a classical ranking: the genes most different between the classes (t-test, two classes)
if len(np.unique(y)) == 2:
    t = pd.Series(np.abs(ttest_ind(X_train[y_train == 0], X_train[y_train == 1], nan_policy="omit")[0]), X.columns)
    print(f"Spearman with |t| of a two-class t-test: {spearmanr(averaged, t, nan_policy='omit')[0]:.2f}")

table = pd.DataFrame({"importance_1_member": single, "importance_4_members": averaged})
table.sort_values("importance_4_members", ascending=False).to_csv("importance.csv")
print("\ntop columns (4 members):")
print(averaged.nlargest(10).round(5).to_string())
print("wrote importance.csv")
