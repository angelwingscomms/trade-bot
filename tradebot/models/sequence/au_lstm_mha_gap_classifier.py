"""AU architecture: LSTM -> projected MHA -> global average pool."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.projected_multihead_self_attention import ProjectedMultiHeadSelfAttention


class AuLSTMMultiheadAttentionClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        if n_features <= 0:
            raise ValueError('AuLSTMMultiheadAttentionClassifier requires n_features > 0.')

        self.backend_name = 'au-lstm-mha-gap'
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.attention = ProjectedMultiHeadSelfAttention(
            input_dim=64,
            num_heads=4,
            head_dim=64,
            dropout=0.0,
        )
        self.classifier = nn.Linear(64, n_classes)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _state = self.lstm(x)
        return self.attention(sequence_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
