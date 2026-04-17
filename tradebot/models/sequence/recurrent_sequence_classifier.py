"""Configurable recurrent sequence classifier."""

from __future__ import annotations

import torch
from torch import nn

from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm
from tradebot.models.sequence.sequence_multi_attention_head import SequenceMultiAttentionHead


class RecurrentSequenceClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        cell_type: str = 'lstm',
        hidden_size: int = 64,
        hidden: int = 96,
        n_classes: int = 3,
        dropout: float = 0.1,
        num_layers: int = 2,
        bidirectional: bool = False,
        use_multihead_attention: bool = False,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
        backend_name: str = '',
    ):
        super().__init__()
        recurrent_type = cell_type.strip().lower()
        if recurrent_type not in {'lstm', 'gru'}:
            raise ValueError(f'Unsupported recurrent cell_type: {cell_type}')
        if hidden_size <= 0:
            raise ValueError('RecurrentSequenceClassifier requires hidden_size > 0.')
        if num_layers <= 0:
            raise ValueError('RecurrentSequenceClassifier requires num_layers > 0.')

        self.cell_type = recurrent_type
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.use_multihead_attention = bool(use_multihead_attention)
        self.num_directions = 2 if self.bidirectional else 1
        output_dim = self.hidden_size * self.num_directions

        default_backend = self.cell_type
        if self.bidirectional:
            default_backend = f'bi{default_backend}'
        if self.use_multihead_attention:
            default_backend = f'{default_backend}-attention'
        self.backend_name = backend_name or default_backend

        recurrent_dropout = dropout if self.num_layers > 1 else 0.0
        recurrent_cls = nn.LSTM if self.cell_type == 'lstm' else nn.GRU

        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.recurrent = recurrent_cls(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.output_norm = nn.LayerNorm(output_dim)

        if self.use_multihead_attention:
            self.head = SequenceMultiAttentionHead(
                input_dim=output_dim,
                hidden=hidden,
                n_classes=n_classes,
                num_heads=attention_heads,
                num_layers=attention_layers,
                dropout=attention_dropout,
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

    def encode_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        x = self.sequence_norm(x)
        output, state = self.recurrent(x)
        output = self.output_norm(output)
        return output, state

    def encode_summary(self, x: torch.Tensor) -> torch.Tensor:
        _output, state = self.encode_sequence(x)
        hidden_state = state[0] if isinstance(state, tuple) else state
        hidden_state = hidden_state.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_size)
        last_hidden = hidden_state[-1].transpose(0, 1).contiguous().view(x.shape[0], -1)
        return last_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, state = self.encode_sequence(x)
        if self.use_multihead_attention:
            return self.head(sequence_output)

        hidden_state = state[0] if isinstance(state, tuple) else state
        hidden_state = hidden_state.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_size)
        last_hidden = hidden_state[-1].transpose(0, 1).contiguous().view(x.shape[0], -1)
        return self.head(last_hidden)
