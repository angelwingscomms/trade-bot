from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tradebot.models.sequence.causal_conv1d import CausalConv1d
from tradebot.models.sequence.sequence_attention_block import SequenceAttentionBlock
from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm
from tradebot.models.sequence.temporal_attention_pooling import TemporalAttentionPooling

class EmbTCNClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int = 64, # TCN embedding width
        hidden: int = 64, # used for Transformer FFN
        dense_hidden: int = 48,
        n_classes: int = 3,
        attention_heads: int = 2,
        attention_dropout: float = 0.1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sequence_norm = SequenceInstanceNorm(n_features)

        # --- EmbTCN: dilated causal convolutions replace linear embedding ---
        # dilation 1,2,4,8 gives receptive field > 30 bars on 1m data
        self.tcn1 = CausalConv1d(n_features, channels, kernel_size=3, dilation=1)
        self.tcn2 = CausalConv1d(channels, channels, kernel_size=3, dilation=2)
        self.tcn3 = CausalConv1d(channels, channels, kernel_size=3, dilation=4)
        self.tcn4 = CausalConv1d(channels, channels, kernel_size=3, dilation=8)
        self.tcn_norm = nn.LayerNorm(channels)
        self.tcn_dropout = nn.Dropout(dropout)

        # --- Transformer backbone: global self-attention ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=attention_heads,
            dim_feedforward=hidden * 2,
            dropout=attention_dropout,
            batch_first=True,
            norm_first=True, # matches paper's LayerNorm stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Keep your existing attention block for compatibility
        self.attention = SequenceAttentionBlock(
            model_dim=channels,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )
        self.pool = TemporalAttentionPooling(channels)

        # --- Dueling head (unchanged interface) ---
        classifier_in_dim = (channels * 2) + n_features
        self.dense_shared = nn.Sequential(
            nn.Linear(classifier_in_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.state_branch = nn.Linear(dense_hidden, 1)
        self.direction_branch = nn.Linear(dense_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        normed = self.sequence_norm(x)

        # EmbTCN embedding
        t = normed.transpose(1, 2) # (B, F, T)
        t = F.gelu(self.tcn1(t))
        t = F.gelu(self.tcn2(t))
        t = F.gelu(self.tcn3(t))
        t = F.gelu(self.tcn4(t))
        t = t.transpose(1, 2) # (B, T, C)
        t = self.tcn_dropout(self.tcn_norm(t))

        # Transformer global context
        encoded = self.transformer(t) # (B, T, C)

        # Optional refinement (keeps your SequenceAttentionBlock)
        encoded = self.attention(encoded)

        pooled = self.pool(encoded)
        final_vector = torch.cat([encoded[:, -1, :], pooled, x[:, -1, :]], dim=1)

        shared = self.dense_shared(final_vector)
        state_val = self.state_branch(shared)
        direction_val = self.direction_branch(shared)

        if direction_val.shape[1] > 1:
            return state_val + (direction_val - direction_val.mean(dim=1, keepdim=True))
        return direction_val