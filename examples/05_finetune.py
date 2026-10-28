# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Fine-tune the pretrained model on real tables, and compare it with the model as published.

    python examples/05_finetune.py --device cuda      # leukemia; a few minutes on a GPU
    python examples/05_finetune.py --data my.csv --target subtype --device cuda

Fine-tuning trains a copy of the model on episodes of your tables (random subsets of rows and columns, classes
relabeled), stopping when the loss on 10% of held-out rows stops improving; if no epoch improves on the
pretrained model, the pretrained weights are kept. On one small table the gain is usually small (Rubachev et al.,
2025); a model fine-tuned on many tables of a domain (scripts/finetune.py) is the other use.
"""

import numpy as np
from common import classifier_settings, load_data, parser
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabomics import Classifier
from tabomics.finetuning import FineTunedClassifier, finetune

p = parser(__doc__.split("\n")[0])
p.add_argument("--output", default="runs/finetune_example", help="where 2. writes the fine-tuned model")
args = p.parse_args()
X, y = load_data(args)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
settings = classifier_settings(args)
quick = {"epochs": 2, "steps_per_epoch": 4, "accumulate": 2} if args.quick else {}


def report(name: str, proba: np.ndarray) -> None:
    """AUROC and log loss on the test rows."""
    auroc = (
        roc_auc_score(y_test, proba[:, 1]) if proba.shape[1] == 2 else roc_auc_score(y_test, proba, multi_class="ovr")
    )
    print(f"{name:<12} test AUROC {auroc:.3f}, log loss {log_loss(y_test, proba, labels=np.unique(y)):.3f}")


# 1. As a scikit-learn classifier: fit fine-tunes on the training rows, then classifies as Classifier does (also
#    inside cross_val_predict or a Pipeline: each fold fine-tunes on its own training rows only)
pretrained = Classifier(**settings).fit(X_train, y_train)
report("pretrained", pretrained.predict_proba(X_test))
tuned = FineTunedClassifier(**settings, verbose=True, **quick).fit(X_train, y_train)
report("fine-tuned", tuned.predict_proba(X_test))
print(f"weights of epoch {tuned.best_epoch_} (0 = the pretrained ones)")
print(tuned.finetune_history_.round(4).to_string(index=False))

# 2. The function, for a model to keep: finetuned.pth and history.csv in --output, usable wherever a model is
#    (Classifier(model=...), scripts/export.py). Several tables can be given: [(X1, y1), (X2, y2), ...]
finetune(
    settings["model"], [(X_train, y_train)], device=args.device, amp_dtype="auto", output=args.output, verbose=False,
    categorical_features=[], infer_categorical=False, **quick,
)  # fmt: skip
clf = Classifier(**{**settings, "model": f"{args.output}/finetuned.pth"}).fit(X_train, y_train)
report("saved model", clf.predict_proba(X_test))
