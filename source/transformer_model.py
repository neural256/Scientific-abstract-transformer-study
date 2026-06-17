"""Manual Transformer encoder for scientific abstract classification.

This module intentionally does not use torch.nn.TransformerEncoder. The core
Transformer pieces are implemented directly so the architecture is easy to
study and discuss in a portfolio or interview setting.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the original Transformer paper."""

    def __init__(self, embedding_dim: int, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embedding_dim)
        )

        encoding = torch.zeros(max_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        sequence_length = token_embeddings.size(1)
        token_embeddings = token_embeddings + self.encoding[:, :sequence_length, :]
        return self.dropout(token_embeddings)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional padding mask."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)
        return context, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built from linear projections."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.attention = ScaledDotProductAttention(dropout)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.size()
        tensor = tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, sequence_length, self.embedding_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        context, _ = self.attention(query, key, value, attention_mask)
        context = self._merge_heads(context)
        return self.dropout(self.output(context))


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network used inside each encoder block."""

    def __init__(self, embedding_dim: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


class TransformerEncoderBlock(nn.Module):
    """One Transformer encoder block with residual connections and LayerNorm."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim, feed_forward_dim, dropout)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attention_output)

        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_norm(hidden_states + feed_forward_output)
        return hidden_states


class CustomTransformerClassifier(nn.Module):
    """Transformer encoder classifier for single-label text classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feed_forward_dim: int = 256,
        max_length: int = 256,
        dropout: float = 0.2,
        pad_token_id: int = 0,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id,
        )
        self.use_positional_encoding = use_positional_encoding
        self.position_encoding = (
            PositionalEncoding(embedding_dim, max_length, dropout)
            if use_positional_encoding
            else nn.Dropout(dropout)
        )
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def _masked_mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.position_encoding(hidden_states)

        for encoder_block in self.encoder_blocks:
            hidden_states = encoder_block(hidden_states, attention_mask)

        pooled = self._masked_mean_pool(hidden_states, attention_mask)
        return self.classifier(pooled)
