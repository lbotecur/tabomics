# Examples

Each example is a short script; run it from the repository root. By default they use the published model (`v2.5`,
downloaded from Hugging Face the first time) and the leukemia table of Golub et al. (1999): 72 patients, 7129 genes,
two classes (OpenML dataset 1104, downloaded the first time). `--data my.csv --target label` runs them on your own
table, `--device cpu` without a GPU, and `--quick` runs a tiny synthetic version (to check that it works).

| Example | What it shows |
|---|---|
| `01_classify.py` | Fit and predict as any scikit-learn classifier; cross-validation; several ensemble members (`n_estimators`). |
| `02_embeddings.py` | A vector per patient. Training rows are embedded out of fold: embedded directly, they have seen their own label. |
| `03_interpretability.py` | Feature importance (attention of the label to each column): the informative columns of a synthetic table come out on top; the top genes of the real table, and how stable the ranking is with 1 and 4 members. |
| `04_train_and_continue.py` | Your own architecture, pretrained from scratch on synthetic tables, then a second stage from its weights with widening. |
| `05_finetune.py` | Fine-tuning on the training rows (`FineTunedClassifier`), compared with the pretrained model; a fine-tuned model saved and reused. |

```bash
pip install -r examples/requirements.txt   # matplotlib for the plot of 02, openml for the leukemia table
python examples/01_classify.py
python examples/04_train_and_continue.py --device cuda
```
