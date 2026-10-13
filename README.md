# TabOmics

Tabular foundation model for wide omics data.

TabOmics is a small transformer (7.3M parameters) pretrained on synthetic tables. Given a table, it classifies
the test rows in context, in one forward pass over the training and test rows, with no training on your data. It
is built for wide data (thousands of features, few samples, as in omics), and also returns row embeddings and
feature importance.

> Work in progress: the package is being assembled. Installation and usage will be documented here.

## Licence and credits

Apache License 2.0 (see [LICENSE](LICENSE)). TabOmics derives from the nanoTabPFN architecture
(TFM-Playground, Pfefferle et al.) and uses a synthetic-table generator derived from TabICL's graph prior; see
[NOTICE](NOTICE).
