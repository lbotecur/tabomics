# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from nanoTabPFN (TFM-Playground, Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-nanoTabPFN.txt). Modified: missing-value channel,
# configurable feature clip, test-row indicator for the targets, low-memory inference, label-attention capture.
"""Building blocks of TabOmicsModel: input encoders, the transformer block and the decoder.

Shapes use B for tables in a batch, R for rows (training rows first, then test rows), C for columns and E for
the embedding size.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

# Bound of the normalized feature values. 100 only guards against absurd values (a column constant in the
# training rows has std ~ eps, so any other value becomes ~1e7). 10 also bounds rare values: a binary column
# whose ones are a fraction p of the rows gives them z ~ sqrt((1 - p) / p), about 30 for p = 0.1%, which wide
# sparse tables (bag-of-words text) have in almost every column and synthetic tables almost never have.
DEFAULT_FEATURE_CLIP = 10.0


def normalize_features(x: torch.Tensor, train_size: int, clip: float = DEFAULT_FEATURE_CLIP) -> torch.Tensor:
    """Standardizes each column with its training rows, fills missing values and clips.

    The mean and standard deviation of each column come from the training rows only, ignoring missing values.
    Missing values become 0 (the training mean) and are flagged in a second channel, so the model sees
    "value, was missing" as one unit per cell.

    Args:
        x: Features, shape (B, R, C), with NaN for missing values.
        train_size: Number of training rows (the first rows of x).
        clip: Bound of the standardized values.

    Returns:
        Shape (B, R, C, 2): standardized value and missing-value indicator.
    """
    x = x.unsqueeze(-1)
    missing = torch.isnan(x).to(x.dtype)  # before filling the holes
    train = x[:, :train_size]
    mean = torch.nanmean(train, dim=1, keepdim=True)
    count = (~torch.isnan(train)).sum(dim=1, keepdim=True)
    sum_sq = torch.nansum((train - mean) ** 2, dim=1, keepdim=True)
    std = torch.sqrt(sum_sq / (count - 1)) + torch.finfo(torch.float32).eps  # unbiased, ignoring NaN
    # a single training row gives a NaN std, which clipping cannot repair: use 1
    std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
    x = (x - mean) / std
    x = torch.nan_to_num(x, nan=0.0)
    x = torch.clip(x, min=-clip, max=clip)
    return torch.cat([x, missing], dim=-1)


def pad_targets(y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Extends the training targets to every row and flags the test rows.

    Test rows get the mean training target as a placeholder, plus an indicator that is 1 on them. Without the
    indicator, a test row whose placeholder equals a class (mean 1.0 with classes 0, 1, 2) would look exactly like
    a training row of that class.

    Args:
        y_train: Training targets, shape (B, N, 1).
        num_rows: Total number of rows R (training plus test).

    Returns:
        Shape (B, R, 1, 2): target (or placeholder) and test-row indicator.
    """
    mean = torch.mean(y_train, dim=1, keepdim=True)
    y = torch.cat([y_train, mean.repeat(1, num_rows - y_train.shape[1], 1)], dim=1)
    is_test = torch.zeros_like(y)
    is_test[:, y_train.shape[1] :] = 1.0
    return torch.cat([y, is_test], dim=-1).unsqueeze(2)


class FeatureEncoder(nn.Module):
    """Normalizes the features (normalize_features) and embeds each cell with a linear layer."""

    def __init__(self, embedding_size: int, clip: float = DEFAULT_FEATURE_CLIP):
        super().__init__()
        self.linear_layer = nn.Linear(2, embedding_size)  # input: [value, missing indicator]
        # A plain attribute, not a weight: it can change at inference without touching the weights.
        self.clip = float(clip)

    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        """Embeds the features.

        Args:
            x: Features, shape (B, R, C).
            train_size: Number of training rows.

        Returns:
            Shape (B, R, C, E).
        """
        return self.linear_layer(normalize_features(x, train_size, self.clip))


class TargetEncoder(nn.Module):
    """Extends the training targets to every row (pad_targets) and embeds them with a linear layer."""

    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(2, embedding_size)  # input: [target, test-row indicator]

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """Embeds the targets.

        Args:
            y_train: Training targets, shape (B, N, 1).
            num_rows: Total number of rows R.

        Returns:
            Shape (B, R, 1, E).
        """
        return self.linear_layer(pad_targets(y_train, num_rows))


def apply_in_chunks(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, num_chunks: int) -> torch.Tensor:
    """Applies fn to x in num_chunks pieces along the first axis, writing each result in place, to save memory.

    In-place writes give wrong gradients, so with gradients enabled the input is processed in one piece (and a
    warning is raised if chunks were asked for).

    Args:
        fn: Function that keeps the shape of its input.
        x: Input, split along its first axis.
        num_chunks: Number of pieces; 1 or less processes x at once.

    Returns:
        fn applied to x (x itself when chunked).
    """
    if num_chunks <= 1 or x.shape[0] == 0:
        return fn(x)
    if torch.is_grad_enabled():
        warnings.warn(
            "Memory chunking is disabled while gradients are enabled (it would give wrong gradients); "
            "use torch.no_grad() at inference to enable it.",
            stacklevel=2,
        )
        return fn(x)
    chunk_size = max(1, math.ceil(x.shape[0] / num_chunks))
    for piece in torch.split(x, chunk_size, dim=0):
        piece[:] = fn(piece)
    return x


def label_attention_weights(attention: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    """Attention of the last token (the label column) to every token, averaged over heads, for every row.

    Only that row of the attention matrix is computed, so memory grows with the number of columns C, not C^2.
    Equals the head-averaged weights of attention(x, x, x, need_weights=True) at [:, -1].

    Args:
        attention: The attention between columns of a block.
        x: Its input, shape (N, C, E), the label column last.

    Returns:
        Shape (N, C).
    """
    n, c, e = x.shape
    h = attention.num_heads
    d = e // h
    w_q, w_k, _ = attention.in_proj_weight.chunk(3)
    b_q, b_k, _ = attention.in_proj_bias.chunk(3)
    q = F.linear(x[:, -1:, :], w_q, b_q).view(n, 1, h, d).transpose(1, 2)  # (N, H, 1, d)
    k = F.linear(x, w_k, b_k).view(n, c, h, d).transpose(1, 2)  # (N, H, C, d)
    weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), dim=-1)  # (N, H, 1, C)
    return weights.mean(dim=1)[:, 0, :]


def _no_fast_path(key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Key and value as new views of the same data, so that query, key and value are distinct tensor objects.

    When the three are one object and the module is in evaluation mode, nn.MultiheadAttention takes a fused
    "fast path" that rounds slightly differently (differences of ~1e-7). The published models were trained and
    evaluated on the regular path, so it is used always: results then do not depend on train/eval mode.
    """
    return key.view_as(key), value.view_as(value)


class TransformerBlock(nn.Module):
    """Attention between the columns of each row, attention between rows, and an MLP.

    Each step has a residual connection and a LayerNorm. Training rows attend to training rows; test rows attend
    to training rows only, so the test rows never influence each other or the training rows.
    """

    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int):
        super().__init__()
        self.self_attention_between_features = nn.MultiheadAttention(
            embedding_size, num_attention_heads, batch_first=True
        )
        self.self_attention_between_datapoints = nn.MultiheadAttention(
            embedding_size, num_attention_heads, batch_first=True
        )
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)
        # Inference only (ignored while gradients are enabled): apply the LayerNorms inside the memory chunks and
        # keep activations in the dtype of the input (bf16 under autocast) instead of full-size fp32 tensors.
        # Needed for tables with tens of thousands of columns; as accurate as the normal path.
        self.low_memory = False

    def forward(
        self,
        x: torch.Tensor,
        train_size: int,
        num_mem_chunks: int = 1,
        capture_label_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Applies the block.

        Args:
            x: Cell embeddings, shape (B, R, C, E), the label column last.
            train_size: Number of training rows.
            num_mem_chunks: Pieces the memory-heavy steps are split into (inference only, see apply_in_chunks).
            capture_label_attention: Also return the label column's attention to every column, averaged over
                rows and heads.

        Returns:
            The new embeddings, shape (B, R, C, E), and the label attention, shape (C,), or None.
        """
        batch_size, num_rows, num_columns, embedding_size = x.shape
        low_memory = self.low_memory and not torch.is_grad_enabled()

        def residual_norm(norm: nn.LayerNorm, out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
            # low-memory: normalize inside the chunk and keep the input dtype; otherwise normalize later, at once
            return norm(out).to(inp.dtype) if low_memory else out

        # attention between the columns of each row
        x = x.reshape(batch_size * num_rows, num_columns, embedding_size)
        attention_sum = [torch.zeros(num_columns, device=x.device), 0]  # exact mean whatever the chunks

        def between_columns(inp: torch.Tensor) -> torch.Tensor:
            # need_weights=False lets PyTorch use scaled_dot_product_attention, which never builds the (C, C)
            # weights; the label row of the attention is computed apart when it is asked for. Key and value are
            # views (distinct tensor objects): see _no_fast_path.
            out = self.self_attention_between_features(inp, *_no_fast_path(inp, inp), need_weights=False)[0]
            if capture_label_attention:
                weights = label_attention_weights(self.self_attention_between_features, inp)
                attention_sum[0] += weights.float().sum(dim=0)
                attention_sum[1] += inp.shape[0]
            return residual_norm(self.norm1, out + inp, inp)

        x = apply_in_chunks(between_columns, x, num_mem_chunks)
        label_attention = (attention_sum[0] / attention_sum[1]).detach() if capture_label_attention else None
        x = x.reshape(batch_size, num_rows, num_columns, embedding_size)
        if not low_memory:
            x = self.norm1(x)

        # attention between rows, in each column
        x = x.transpose(1, 2).reshape(batch_size * num_columns, num_rows, embedding_size)

        def between_rows(inp: torch.Tensor) -> torch.Tensor:
            context = inp[:, :train_size]
            train_out = self.self_attention_between_datapoints(
                context, *_no_fast_path(context, context), need_weights=False
            )[0]
            test_out = self.self_attention_between_datapoints(
                inp[:, train_size:], *_no_fast_path(context, context), need_weights=False
            )[0]
            return residual_norm(self.norm2, torch.cat([train_out, test_out], dim=1) + inp, inp)

        x = apply_in_chunks(between_rows, x, num_mem_chunks)
        x = x.reshape(batch_size, num_columns, num_rows, embedding_size).transpose(2, 1)
        if not low_memory:
            x = self.norm2(x)

        # MLP on every cell
        x = x.reshape(-1, embedding_size)

        def mlp(inp: torch.Tensor) -> torch.Tensor:
            return residual_norm(self.norm3, self.linear2(F.gelu(self.linear1(inp))) + inp, inp)

        x = apply_in_chunks(mlp, x, num_mem_chunks)
        x = x.reshape(batch_size, num_rows, num_columns, embedding_size)
        if not low_memory:
            x = self.norm3(x)
        return x, label_attention


class Decoder(nn.Module):
    """Two-layer MLP from the label-column embedding of a row to the class logits."""

    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps embeddings of shape (..., E) to logits of shape (..., num_outputs)."""
        return self.linear2(F.gelu(self.linear1(x)))
