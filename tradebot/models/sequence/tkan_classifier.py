
from __future__ import annotations

import torch
from torch import nn

from efficient_kan import KANLinear
from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm


class ScalperMicrostructureClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        hidden: int = 64,
        dense_hidden: int = 48,
        n_classes: int = 3,
        attention_heads: int = 2,
        attention_dropout: float = 0.1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sequence_norm = SequenceInstanceNorm(n_features)

        self.encoder = nn.LSTM(
            n_features,
            64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        latent_dim = 256

        classifier_in_dim = latent_dim + n_features
        self.dense_shared = nn.Sequential(
            KANLinear(classifier_in_dim, dense_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.state_branch = KANLinear(dense_hidden, 1)
        self.direction_branch = KANLinear(dense_hidden, n_classes)
        self.l1_lambda = 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.sequence_norm(x)
        _, (h_n, _) = self.encoder(normed)
        latent = h_n.permute(1, 0, 2).reshape(x.size(0), -1)
        final_vector = torch.cat([latent, x[:, -1, :]], dim=1)

        shared = self.dense_shared(final_vector)
        state_val = self.state_branch(shared)
        direction_val = self.direction_branch(shared)

        if direction_val.shape[1] > 1:
            return state_val + (direction_val - direction_val.mean(1, keepdim=True))
        return direction_val

    def l1_sparsity_penalty(self) -> torch.Tensor:
        return self.l1_lambda * sum(
            p.abs().sum()
            for m in self.modules()
            if isinstance(m, KANLinear)
            for p in m.parameters()
        )
