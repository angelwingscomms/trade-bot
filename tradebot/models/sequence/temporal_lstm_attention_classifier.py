"""Temporal conv + BiLSTM + attention classifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm
from tradebot.models.sequence.sequence_multi_attention_head import SequenceMultiAttentionHead


class TemporalLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: int = 128,
        lstm_hidden: int = 128,
        hidden: int = 96,
        n_classes: int = 3,
        dropout: float = 0.1,
        conv_kernel_size: int = 5,
        lstm_layers: int = 2,
        attention_heads: int = 8,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError('TemporalLSTMAttentionClassifier requires n_features > 0.')
        if conv_channels <= 0:
            raise ValueError('TemporalLSTMAttentionClassifier requires conv_channels > 0.')
        if lstm_hidden <= 0:
            raise ValueError('TemporalLSTMAttentionClassifier requires lstm_hidden > 0.')
        if lstm_layers <= 0:
            raise ValueError('TemporalLSTMAttentionClassifier requires lstm_layers > 0.')

        self.n_features = int(n_features)
        self.conv_channels = int(conv_channels)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.backend_name = 'tla-temporal-lstm-attention'

        self.sequence_norm = SequenceInstanceNorm(self.n_features)
        self.conv1 = nn.Conv1d(
            in_channels=self.n_features,
            out_channels=64,
            kernel_size=int(conv_kernel_size),
            padding=int(conv_kernel_size) // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=self.conv_channels,
            kernel_size=int(conv_kernel_size),
            padding=int(conv_kernel_size) // 2,
        )
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=self.conv_channels,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            dropout=dropout if self.lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        lstm_output_dim = self.lstm_hidden * 2
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        self.head = SequenceMultiAttentionHead(
            input_dim=lstm_output_dim,
            hidden=hidden,
            n_classes=n_classes,
            num_heads=attention_heads,
            num_layers=attention_layers,
            dropout=attention_dropout,
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.conv_dropout(x)
        x = F.relu(self.conv2(x))
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)
        lstm_output, _ = self.lstm(x)
        lstm_output = self.lstm_norm(lstm_output)
        return lstm_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        return self.head(encoded)
