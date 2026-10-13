"""TabOmics: a tabular foundation model for wide omics data.

A small transformer, pretrained on synthetic tables, that classifies a new table in context (one forward pass on
its training and test rows, no gradient steps), and gives row embeddings and feature importance on the way.
"""

__version__ = "0.1.0.dev0"
