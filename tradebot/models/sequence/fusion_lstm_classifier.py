"""Fusion LSTM classifier."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.mish_lstm_cell import MishLSTMCell
from tradebot.models.sequence.projected_multihead_self_attention import ProjectedMultiHeadSelfAttention
from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm


class FusionLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 20,
        n_classes: int = 3,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError('FusionLSTMClassifier requires n_features > 0.')
        if hidden <= 0:
            raise ValueError('FusionLSTMClassifier requires hidden > 0.')

        self.n_features = int(n_features)
        self.backend_name = 'fusion-lstm-attention'
        self.sequence_norm = SequenceInstanceNorm(self.n_features)
        self.recurrent_cell = MishLSTMCell(input_size=self.n_features, hidden_size=self.n_features)
        self.attention = ProjectedMultiHeadSelfAttention(
            input_dim=self.n_features,
            num_heads=attention_heads,
            head_dim=self.n_features,
            dropout=attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.n_features, hidden),
            nn.Mish(),
            nn.Linear(hidden, n_classes),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        batch_size, seq_len, _channels = x.shape
        hidden_state = x.new_zeros(batch_size, self.n_features)
        cell_state = x.new_zeros(batch_size, self.n_features)
        outputs = []
        for timestep in range(seq_len):
            hidden_state, cell_state = self.recurrent_cell(x[:, timestep, :], (hidden_state, cell_state))
            outputs.append(hidden_state)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        attended = self.attention(encoded)
        pooled = (encoded + attended).mean(dim=1)
        return self.classifier(pooled)
