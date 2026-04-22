from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KANLinear


class TKAN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        n_heads: int = 4,
        proj_dim: int = 512,
        dropout: float = 0.2,
        l1_lambda: float = 1e-4,
    ):
        super().__init__()
        self.l1_lambda = l1_lambda

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        lstm_out_dim = lstm_hidden * 2

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)

        self.latent_proj = nn.Linear(lstm_out_dim, 256)
        self.latent_norm = nn.LayerNorm(256)

        self.head = nn.Sequential(
            KANLinear(256, proj_dim, grid_size=3, spline_order=2),
            nn.SiLU(),
            nn.Dropout(dropout),
            KANLinear(proj_dim, n_classes, grid_size=3, spline_order=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.encoder(x)

        attn_out, _ = self.attention(lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)

        pooled = attn_out.mean(dim=1)

        latent = self.latent_proj(pooled)
        latent = self.latent_norm(latent)

        return self.head(latent)

    def l1_sparsity_penalty(self):
        l1 = 0.0
        for m in self.modules():
            if isinstance(m, KANLinear):
                for p in m.parameters():
                    if p.requires_grad:
                        l1 = l1 + p.abs().sum()
        return self.l1_lambda * l1
