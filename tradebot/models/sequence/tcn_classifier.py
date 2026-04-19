"""Strict temporal convolution classifier for the TCN pipeline."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm
from tradebot.models.sequence.temporal_conv_block import TemporalConvBlock


class TCNClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        dropout: float = 0.4,
        kernel_size: int = 3,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError("TCNClassifier requires n_features > 0.")
        if n_classes <= 1:
            raise ValueError("TCNClassifier requires n_classes > 1.")
        if kernel_size <= 1:
            raise ValueError("TCNClassifier requires kernel_size > 1.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("TCNClassifier requires dropout in [0, 1).")

        self.backend_name = "tcn"
        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.input_projection = (
            nn.Linear(n_features, 32) if n_features != 32 else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=kernel_size,
                    dilation=1,
                    dropout=dropout,
                ),
                TemporalConvBlock(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=kernel_size,
                    dilation=2,
                    dropout=dropout,
                ),
                TemporalConvBlock(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=kernel_size,
                    dilation=4,
                    dropout=dropout,
                ),
            ]
        )
        self.output_norm = nn.LayerNorm(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, n_classes)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "TCNClassifier expects [batch, seq_len, features] input."
            )

        x = self.sequence_norm(x)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        pooled = self.global_dropout(pooled)
        return self.classifier(pooled)
