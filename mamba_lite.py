"""Compact Mamba-inspired classifier used by the training pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from sequence_models import (
    SequenceInstanceNorm,
    SequenceMultiAttentionHead,
)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm.to(x.dtype) * self.weight


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class PortableMambaMixer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = max(1, (d_model + 15) // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        a = a + torch.rand_like(a) * 0.1
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)
        x_branch = F.silu(self.conv(x_branch))
        params = self.x_proj(x_branch)
        delta, b_term, c_term = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        y = self._selective_scan(x_branch, delta, b_term, c_term)
        y = y * F.silu(z_branch)
        return self.out_proj(y)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        b_term: torch.Tensor,
        c_term: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = u.shape
        a = -torch.exp(self.A_log).to(dtype=u.dtype, device=u.device)
        d = self.D.to(dtype=u.dtype, device=u.device)
        state = torch.zeros(
            (batch_size, self.d_inner, self.d_state),
            device=u.device,
            dtype=torch.float32,
        )
        outputs = []

        delta_a = torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0)).to(torch.float32)
        delta_b_u = ((delta * u).unsqueeze(-1) * b_term.unsqueeze(2)).to(torch.float32)
        c_term = c_term.to(torch.float32)
        u_f32 = u.to(torch.float32)
        d_f32 = d.to(torch.float32)

        for t in range(seq_len):
            state = delta_a[:, t] * state + delta_b_u[:, t]
            y_t = (state * c_term[:, t].unsqueeze(1)).sum(dim=-1) + d_f32 * u_f32[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(u.dtype)


class MambaLiteResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = PortableMambaMixer(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class MambaLiteClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        hidden: int = 64,
        n_classes: int = 3,
        dropout: float = 0.1,
        n_layers: int = 1,
        use_multihead_attention: bool = False,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_multihead_attention = bool(use_multihead_attention)
        self.backend_name = (
            "portable-mamba-lite-attention" if self.use_multihead_attention else "portable-mamba-lite"
        )
        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.embedding = nn.Linear(n_features, d_model) if d_model != n_features else nn.Identity()
        self.layers = nn.ModuleList(
            MambaLiteResidualBlock(d_model=d_model, dropout=dropout) for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)
        if self.use_multihead_attention:
            self.head = SequenceMultiAttentionHead(
                input_dim=d_model,
                hidden=hidden,
                n_classes=n_classes,
                num_heads=attention_heads,
                num_layers=attention_layers,
                dropout=attention_dropout,
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def encode_last(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(x)[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        if self.use_multihead_attention:
            return self.head(encoded)
        return self.head(encoded[:, -1, :])


GoldMambaLiteClassifier = MambaLiteClassifier

__all__ = [
    "CausalDepthwiseConv1d",
    "GoldMambaLiteClassifier",
    "MambaLiteClassifier",
    "RMSNorm",
    "SequenceMultiAttentionHead",
]
