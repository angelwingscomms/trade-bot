"""Legacy gold LSTM-attention classifier."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.mish_lstm_cell import MishLSTMCell
from tradebot.models.sequence.projected_multihead_self_attention import ProjectedMultiHeadSelfAttention


class GoldLegacyLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        dense_hidden: int = 20,
        n_classes: int = 3,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError('GoldLegacyLSTMAttentionClassifier requires n_features > 0.')
        if dense_hidden <= 0:
            raise ValueError('GoldLegacyLSTMAttentionClassifier requires dense_hidden > 0.')

        self.n_features = int(n_features)
        self.backend_name = 'gold-legacy-lstm-attention'
        self.recurrent_cell = MishLSTMCell(input_size=self.n_features, hidden_size=self.n_features)
        self.attention = ProjectedMultiHeadSelfAttention(
            input_dim=self.n_features,
            num_heads=attention_heads,
            head_dim=self.n_features,
            dropout=attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.n_features, dense_hidden),
            nn.Mish(),
            nn.Linear(dense_hidden, n_classes),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('GoldLegacyLSTMAttentionClassifier expects [batch, seq_len, channels] input.')

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
