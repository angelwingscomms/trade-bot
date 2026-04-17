"""Residual temporal convolution block."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tradebot.models.sequence.causal_conv1d import CausalConv1d


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv1(self.norm1(x))
        x = self.dropout(F.gelu(x))
        x = self.conv2(self.norm2(x))
        x = self.dropout(F.gelu(x))
        return residual + x
