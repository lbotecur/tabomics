# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
# Derived from nanoTabPFN (TFM-Playground, Copyright 2025 Prior Labs GmbH, AutoML Freiburg and Alexander
# Pfefferle, Apache License 2.0; see LICENSES/Apache-2.0-TFM-Playground.txt). Modified: column vectors, configurable
# feature clip, low-memory inference, gradient checkpointing, embeddings and label attention as outputs.
"""TabOmicsModel: the transformer that classifies the test rows of a table in context."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabomics.model.layers import DEFAULT_FEATURE_CLIP, Decoder, FeatureEncoder, TargetEncoder, TransformerBlock

# Seed of the fixed random vector given to each column position (see TabOmicsModel.add_column_vectors).
COLUMN_VECTOR_SEED = 42


@dataclass(frozen=True)
class TabOmicsConfig:
    """Architecture of a TabOmicsModel and how it is meant to be used (stored with the weights).

    Attributes:
        embedding_size: Size of each cell's embedding.
        num_attention_heads: Heads of both attentions.
        mlp_hidden_size: Hidden size of the MLPs of the blocks and of the decoder.
        num_layers: Number of transformer blocks.
        num_outputs: Maximum number of classes.
        feature_clip: Bound of the normalized feature values (see normalize_features).
        permute_columns: Whether the classifier shuffles the columns once, with a fixed seed, before the model
            (training tables have their columns in random order; the preprocessing puts categorical ones last).
    """

    embedding_size: int = 192
    num_attention_heads: int = 6
    mlp_hidden_size: int = 768
    num_layers: int = 12
    num_outputs: int = 10
    feature_clip: float = DEFAULT_FEATURE_CLIP
    permute_columns: bool = False

    def to_dict(self) -> dict:
        """Plain dictionary, as saved next to the weights."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> TabOmicsConfig:
        """Builds a configuration from a dictionary, ignoring unknown keys."""
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class TabOmicsOutput:
    """Result of TabOmicsModel.run.

    Attributes:
        logits: Class logits of the test rows, shape (B, R_test, num_outputs).
        embeddings: Label-column embedding of every row (training and test), shape (B, R, E), if asked for.
        label_attention: Attention of the label column to each column (the label itself last), averaged over
            rows, heads and blocks, shape (C + 1,), if asked for.
    """

    logits: torch.Tensor
    embeddings: torch.Tensor | None = None
    label_attention: torch.Tensor | None = None


class TabOmicsModel(nn.Module):
    """Transformer over the cells of a table (rows x columns), with the label as one more column.

    Each cell is embedded on its own; the blocks then alternate attention between the columns of a row and
    attention between rows (test rows only look at training rows); the decoder reads the label column of the test
    rows. It classifies a new table in one forward pass, without training on it.
    """

    def __init__(self, config: TabOmicsConfig | None = None):
        super().__init__()
        self.config = config or TabOmicsConfig()
        c = self.config
        self.feature_encoder = FeatureEncoder(c.embedding_size, clip=c.feature_clip)
        # projects the fixed random vector of each column position (see add_column_vectors)
        self.column_embedding = nn.Linear(c.embedding_size // 4, c.embedding_size)
        self.target_encoder = TargetEncoder(c.embedding_size)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(c.embedding_size, c.num_attention_heads, c.mlp_hidden_size) for _ in range(c.num_layers)
        )
        self.decoder = Decoder(c.embedding_size, c.mlp_hidden_size, c.num_outputs)
        # Training only: keep only each block's input and recompute the block in the backward pass (less memory,
        # about 30% more compute).
        self.gradient_checkpointing = False

    @property
    def feature_clip(self) -> float:
        """Bound of the normalized feature values; can be changed at inference."""
        return self.feature_encoder.clip

    @feature_clip.setter
    def feature_clip(self, value: float) -> None:
        self.feature_encoder.clip = float(value)

    def set_low_memory(self, enabled: bool) -> None:
        """Turns low-memory inference on or off in every block (see TransformerBlock.low_memory)."""
        for block in self.transformer_blocks:
            block.low_memory = enabled

    def add_column_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """Adds to every cell a vector that identifies its column position.

        Each position gets a fixed random vector (seed COLUMN_VECTOR_SEED, drawn on the CPU so that it is the same
        on every device), projected by a learned linear layer, as TabPFN v2 does: columns with similar values stay
        distinguishable in the attention between columns. Training tables have their columns in random order, so
        the vectors are identifiers, not meanings.

        Args:
            x: Cell embeddings, shape (B, R, C, E).

        Returns:
            Same shape.
        """
        generator = torch.Generator().manual_seed(COLUMN_VECTOR_SEED)
        vectors = torch.randn(x.shape[2], self.config.embedding_size // 4, generator=generator).to(x.device)
        return x + self.column_embedding(vectors).to(x.dtype)[None, None]

    def run(
        self,
        x: torch.Tensor,
        y_train: torch.Tensor,
        train_size: int,
        *,
        num_mem_chunks: int = 1,
        return_embeddings: bool = False,
        return_label_attention: bool = False,
    ) -> TabOmicsOutput:
        """Classifies the test rows of a batch of tables.

        Args:
            x: Features of the training rows followed by the test rows, shape (B, R, C), NaN for missing values.
            y_train: Class indices of the training rows, shape (B, N) or (B, N, 1), as floats.
            train_size: Number of training rows N (the first N rows of x).
            num_mem_chunks: Pieces the memory-heavy steps are split into, at inference (1 in training).
            return_embeddings: Also return the label-column embedding of every row.
            return_label_attention: Also return the label column's attention to each column.

        Returns:
            The logits of the test rows, and the extras asked for.
        """
        if y_train.dim() < x.dim():
            y_train = y_train.unsqueeze(-1)
        cells = self.add_column_vectors(self.feature_encoder(x, train_size))
        labels = self.target_encoder(y_train, cells.shape[1])
        table = torch.cat([cells, labels], dim=2)  # the label is the last column
        del cells, labels  # do not keep a second full-size copy of the table

        attentions = []
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                # The stored block input is fp32 under autocast (a LayerNorm output): cast it to the autocast
                # dtype first, half the memory. Training with mixed precision only.
                device_type = table.device.type
                if torch.is_autocast_enabled(device_type):
                    table = table.to(torch.get_autocast_dtype(device_type))
                table, _ = checkpoint(block, table, train_size, use_reentrant=False)
            else:
                table, attention = block(table, train_size, num_mem_chunks, return_label_attention)
                attentions.append(attention)

        return TabOmicsOutput(
            logits=self.decoder(table[:, train_size:, -1, :]),
            embeddings=table[:, :, -1, :].detach() if return_embeddings else None,
            label_attention=torch.stack(attentions).mean(dim=0) if return_label_attention else None,
        )

    def forward(self, x: torch.Tensor, y_train: torch.Tensor, train_size: int, num_mem_chunks: int = 1) -> torch.Tensor:
        """Class logits of the test rows, shape (B, R - N, num_outputs); see run for the arguments."""
        return self.run(x, y_train, train_size, num_mem_chunks=num_mem_chunks).logits
