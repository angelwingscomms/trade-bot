"""Temporal attention pooling."""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError('TemporalAttentionPooling requires input_dim > 0.')
        self.score = nn.Linear(int(input_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('TemporalAttentionPooling expects [batch, seq_len, channels] input.')

        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)
