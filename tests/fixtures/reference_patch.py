"""A patch of the reference implementation (TFM-Playground's nanoTabPFN) used by the fixture scripts."""

import torch
from tfmplayground.models.nanotabpfn import TransformerEncoderLayer


def average_test_rows_only(num_rows: int, train_size: int):
    """The reference's capture averages the target column's attention over all rows (training and test); with
    this patch it averages over the test rows only, as TabOmics does. The capture sums target_attention over the
    rows and divides by their number, so each test row is scaled by rows / test rows and each training row set to
    zero. Needs the rows in one piece (num_mem_chunks=1): row i of the piece is row i % num_rows of its table."""
    original = TransformerEncoderLayer.target_attention

    def patched(attention, x):
        weights = original(attention, x)
        test = (torch.arange(x.shape[0]) % num_rows >= train_size).to(weights.dtype)
        return weights * test[:, None] * (x.shape[0] / test.sum())

    TransformerEncoderLayer.target_attention = staticmethod(patched)
    return original
