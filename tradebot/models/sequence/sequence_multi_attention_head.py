"""Attention pooling head for sequence classifiers."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.sequence_attention_block import SequenceAttentionBlock


class SequenceMultiAttentionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        n_classes: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError('SequenceMultiAttentionHead requires input_dim > 0.')
        if num_heads <= 0:
            raise ValueError('SequenceMultiAttentionHead requires num_heads > 0.')
        if num_layers <= 0:
            raise ValueError('SequenceMultiAttentionHead requires num_layers > 0.')

        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.model_dim = max(self.num_heads, self.input_dim)
        if self.model_dim % self.num_heads != 0:
            self.model_dim += self.num_heads - (self.model_dim % self.num_heads)

        self.input_projection = (
            nn.Linear(self.input_dim, self.model_dim) if self.model_dim != self.input_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            SequenceAttentionBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        )
        self.pool_projection = nn.Linear(self.model_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('SequenceMultiAttentionHead expects [batch, seq_len, channels] input.')

        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        pool_weights = torch.softmax(self.pool_projection(x).squeeze(-1), dim=-1)
        pooled = torch.sum(x * pool_weights.unsqueeze(-1), dim=1)
        summary = x.mean(dim=1)
        return self.classifier(torch.cat([pooled, summary], dim=1))
