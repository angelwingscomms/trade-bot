"""Mish-activated LSTM cell."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MishLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if input_size <= 0:
            raise ValueError('MishLSTMCell requires input_size > 0.')
        if hidden_size <= 0:
            raise ValueError('MishLSTMCell requires hidden_size > 0.')

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_projection = nn.Linear(self.input_size, self.hidden_size * 4)
        self.hidden_projection = nn.Linear(self.hidden_size, self.hidden_size * 4)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state, cell_state = state
        gates = self.input_projection(x) + self.hidden_projection(hidden_state)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_gate = F.mish(cell_gate)
        cell_state = forget_gate * cell_state + input_gate * cell_gate
        hidden_state = output_gate * F.mish(cell_state)
        return hidden_state, cell_state
