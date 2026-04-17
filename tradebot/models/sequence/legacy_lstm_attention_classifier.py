"""Legacy LSTM-attention classifier."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.legacy_sequence_self_attention import LegacySequenceSelfAttention
from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm


class LegacyLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        dense_hidden: int = 20,
        n_classes: int = 3,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
        backend_name: str = 'legacy-lstm-attention',
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError('LegacyLSTMAttentionClassifier requires n_features > 0.')
        if dense_hidden <= 0:
            raise ValueError('LegacyLSTMAttentionClassifier requires dense_hidden > 0.')

        self.hidden_size = int(n_features)
        self.backend_name = backend_name
        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.recurrent = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.output_activation = nn.Mish()
        self.self_attention = LegacySequenceSelfAttention(
            model_dim=self.hidden_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, dense_hidden),
            nn.Mish(),
            nn.Linear(dense_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('LegacyLSTMAttentionClassifier expects [batch, seq_len, channels] input.')

        x = self.sequence_norm(x)
        sequence_output, _state = self.recurrent(x)
        sequence_output = self.output_activation(self.output_norm(sequence_output))
        attention_output = self.self_attention(sequence_output)
        pooled = (sequence_output + attention_output).mean(dim=1)
        return self.classifier(pooled)
