"""Legacy self-attention block used by the classic LSTM architecture."""

from __future__ import annotations

import torch
from torch import nn


class LegacySequenceSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if model_dim <= 0:
            raise ValueError('LegacySequenceSelfAttention requires model_dim > 0.')
        if num_heads <= 0:
            raise ValueError('LegacySequenceSelfAttention requires num_heads > 0.')

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.attention_dim = max(self.num_heads, self.model_dim)
        if self.attention_dim % self.num_heads != 0:
            self.attention_dim += self.num_heads - (self.attention_dim % self.num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.input_projection = (
            nn.Linear(self.model_dim, self.attention_dim) if self.attention_dim != self.model_dim else nn.Identity()
        )
        self.q_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.k_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.v_proj = nn.Linear(self.attention_dim, self.attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = (
            nn.Linear(self.attention_dim, self.model_dim) if self.attention_dim != self.model_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('LegacySequenceSelfAttention expects [batch, seq_len, channels] input.')

        x = self.input_projection(x)
        batch_size, seq_len, _channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = self.dropout(torch.softmax(attention_scores, dim=-1))
        attended = torch.matmul(attention_probs, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)
        return self.output_projection(attended)
