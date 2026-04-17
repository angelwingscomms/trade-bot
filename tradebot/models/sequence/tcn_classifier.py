"""Temporal convolution classifier."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm
from tradebot.models.sequence.sequence_multi_attention_head import SequenceMultiAttentionHead
from tradebot.models.sequence.temporal_conv_block import TemporalConvBlock


class TCNClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        hidden: int = 96,
        n_classes: int = 3,
        dropout: float = 0.1,
        n_layers: int = 4,
        kernel_size: int = 3,
        use_multihead_attention: bool = False,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError('TCNClassifier requires channels > 0.')
        if n_layers <= 0:
            raise ValueError('TCNClassifier requires n_layers > 0.')
        if kernel_size <= 1:
            raise ValueError('TCNClassifier requires kernel_size > 1.')

        self.use_multihead_attention = bool(use_multihead_attention)
        self.backend_name = 'tcn-attention' if self.use_multihead_attention else 'tcn'
        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.input_projection = nn.Linear(n_features, channels) if channels != n_features else nn.Identity()
        self.layers = nn.ModuleList(
            TemporalConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=2 ** layer_index,
                dropout=dropout,
            )
            for layer_index in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(channels)

        if self.use_multihead_attention:
            self.head = SequenceMultiAttentionHead(
                input_dim=channels,
                hidden=hidden,
                n_classes=n_classes,
                num_heads=attention_heads,
                num_layers=attention_layers,
                dropout=attention_dropout,
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        if self.use_multihead_attention:
            return self.head(encoded)
        return self.head(encoded[:, -1, :])
