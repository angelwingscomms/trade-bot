"""Newer gold temporal classifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tradebot.models.sequence.causal_conv1d import CausalConv1d
from tradebot.models.sequence.sequence_attention_block import SequenceAttentionBlock
from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm
from tradebot.models.sequence.temporal_attention_pooling import TemporalAttentionPooling


class GoldNewTemporalClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        hidden: int = 64,
        dense_hidden: int = 96,
        n_classes: int = 3,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError('GoldNewTemporalClassifier requires n_features > 0.')
        if channels <= 0:
            raise ValueError('GoldNewTemporalClassifier requires channels > 0.')
        if hidden <= 0:
            raise ValueError('GoldNewTemporalClassifier requires hidden > 0.')
        if dense_hidden <= 0:
            raise ValueError('GoldNewTemporalClassifier requires dense_hidden > 0.')
        if kernel_size <= 1:
            raise ValueError('GoldNewTemporalClassifier requires kernel_size > 1.')

        self.backend_name = 'gold-new-conv-gru-attention'
        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.conv_in = CausalConv1d(
            in_channels=n_features,
            out_channels=channels,
            kernel_size=kernel_size,
        )
        self.conv_mid = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
        )
        self.conv_residual = nn.Linear(n_features, channels) if channels != n_features else nn.Identity()
        self.conv_norm = nn.LayerNorm(channels)
        self.conv_dropout = nn.Dropout(dropout)
        self.recurrent = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.recurrent_norm = nn.LayerNorm(hidden)
        self.attention = SequenceAttentionBlock(
            model_dim=hidden,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )
        self.pool = TemporalAttentionPooling(hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, n_classes),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('GoldNewTemporalClassifier expects [batch, seq_len, channels] input.')

        residual = self.conv_residual(self.sequence_norm(x))
        x = self.conv_in(self.sequence_norm(x))
        x = F.gelu(x)
        x = self.conv_dropout(x)
        x = self.conv_mid(x)
        x = self.conv_dropout(F.gelu(x))
        x = self.conv_norm(x + residual)
        x, _state = self.recurrent(x)
        x = self.recurrent_norm(x)
        return self.attention(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        pooled = self.pool(encoded)
        last_state = encoded[:, -1, :]
        return self.classifier(torch.cat([last_state, pooled], dim=1))
