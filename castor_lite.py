"""Compact Castor-style classifier used by the training pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mamba_lite import CausalDepthwiseConv1d, RMSNorm
from sequence_models import SequenceInstanceNorm, SequenceMultiAttentionHead


class CastorTemporalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv = CausalDepthwiseConv1d(self.d_inner, kernel_size=kernel_size)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_branch, gate_branch = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        x_branch = F.gelu(self.conv(x_branch))
        x_branch = x_branch * torch.sigmoid(gate_branch)
        return residual + self.dropout(self.out_proj(x_branch))


class CastorClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 48,
        hidden: int = 96,
        n_classes: int = 3,
        dropout: float = 0.1,
        n_layers: int = 3,
        kernel_size: int = 5,
        use_multihead_attention: bool = False,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.use_multihead_attention = bool(use_multihead_attention)
        self.backend_name = "castor-lite-attention" if self.use_multihead_attention else "castor-lite"
        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.embedding = nn.Linear(n_features, d_model) if d_model != n_features else nn.Identity()
        self.layers = nn.ModuleList(
            CastorTemporalBlock(
                d_model=d_model,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)
        if self.use_multihead_attention:
            self.head = SequenceMultiAttentionHead(
                input_dim=d_model,
                hidden=hidden,
                n_classes=n_classes,
                num_heads=attention_heads,
                num_layers=attention_layers,
                dropout=attention_dropout,
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        if self.use_multihead_attention:
            return self.head(encoded)
        return self.head(encoded[:, -1, :])


__all__ = [
    "CastorClassifier",
    "CastorTemporalBlock",
]
