"""Projected multi-head self-attention block."""

from __future__ import annotations

import torch
from torch import nn


class ProjectedMultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, head_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if input_dim <= 0:
            raise ValueError('ProjectedMultiHeadSelfAttention requires input_dim > 0.')
        if num_heads <= 0:
            raise ValueError('ProjectedMultiHeadSelfAttention requires num_heads > 0.')

        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim if head_dim is not None else input_dim)
        if self.head_dim <= 0:
            raise ValueError('ProjectedMultiHeadSelfAttention requires head_dim > 0.')

        inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.input_dim, inner_dim)
        self.k_proj = nn.Linear(self.input_dim, inner_dim)
        self.v_proj = nn.Linear(self.input_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, self.input_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('ProjectedMultiHeadSelfAttention expects [batch, seq_len, channels] input.')

        batch_size, seq_len, _channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = self.attention_dropout(torch.softmax(attention_scores, dim=-1))
        attended = torch.matmul(attention_probs, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attended)
