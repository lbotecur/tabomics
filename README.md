# TabOmics

Tabular foundation model for wide omics data.

TabOmics is a small transformer (7.3M parameters) pretrained on synthetic tables. Given a table, it classifies
the test rows in context, in one forward pass over the training and test rows, with no training on your data. It
is built for wide data (thousands of features, few samples, as in omics), and also returns row embeddings and
feature importance.

> Work in progress: the package is being assembled. Installation and usage will be documented here.

## Licence and credits

TabOmics is free software, released under the GNU General Public License v3.0 or later (see [LICENSE](LICENSE)).
Its architecture derives from nanoTabPFN [1, 2], and its code from the TFM-Playground repository [3] (Apache License
2.0); its synthetic-table generator derives from the TabICL prior [4, 5, 6] (BSD 3-Clause License). See
[NOTICE](NOTICE) and [LICENSES](LICENSES).

### References

[1] A. Pfefferle, J. Hog, L. Purucker and F. Hutter. *nanoTabPFN: A Lightweight and Educational Reimplementation of
TabPFN*. arXiv:2511.03634, 2025. [arxiv.org/abs/2511.03634](https://arxiv.org/abs/2511.03634)

[2] AutoML. *nanoTabPFN*. GitHub repository. [github.com/automl/nanoTabPFN](https://github.com/automl/nanoTabPFN)

[3] A. Pfefferle, J. Hog, S. B. Öztürk, K. Kaya, Z. Türkmen and F. Hutter. *TFM-Playground: A Playground for Tabular
Foundation Models*. GitHub repository, 2026. [github.com/automl/TFM-Playground](https://github.com/automl/TFM-Playground)

[4] J. Qu, D. Holzmüller, G. Varoquaux and M. Le Morvan. *TabICL: A Tabular Foundation Model for In-Context Learning
on Large Data*. ICML, 2025. [arxiv.org/abs/2502.05564](https://arxiv.org/abs/2502.05564)

[5] J. Qu, D. Holzmüller, G. Varoquaux and M. Le Morvan. *TabICLv2: A better, faster, scalable, and open tabular
foundation model*. ICML, 2026. [arxiv.org/abs/2602.11139](https://arxiv.org/abs/2602.11139)

[6] The Soda research team, Inria. *TabICL*. GitHub repository.
[github.com/soda-inria/tabicl](https://github.com/soda-inria/tabicl)
