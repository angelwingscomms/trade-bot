


from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tradebot.models.sequence.causal_conv1d import CausalConv1d
from tradebot.models.sequence.sequence_attention_block import SequenceAttentionBlock
from tradebot.models.sequence.sequence_instance_norm import SequenceInstanceNorm
from tradebot.models.sequence.temporal_attention_pooling import TemporalAttentionPooling

class ScalperMicrostructureClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int = 64,        # Reduced from 128 for i3 CPU speed
        hidden: int = 64,          # Reduced from 128 
        dense_hidden: int = 48,
        n_classes: int = 3,
        attention_heads: int = 2,  # Fewer heads = faster CPU inference
        attention_dropout: float = 0.1,
        dropout: float = 0.3,     
    ):
        super().__init__()
        self.sequence_norm = SequenceInstanceNorm(n_features)
        
        # Pointwise mixing is very fast on CPU
        self.pointwise_in = nn.Conv1d(n_features, channels, kernel_size=1)
        
        # Dilated Convolutions (Fast and effective for 9s timeframe)
        self.conv_d1 = CausalConv1d(channels, channels // 2, kernel_size=3, dilation=1)
        self.conv_d2 = CausalConv1d(channels, channels // 2, kernel_size=3, dilation=2)
        
        self.conv_norm = nn.LayerNorm(channels)
        self.conv_dropout = nn.Dropout(dropout)
        
        # Unidirectional GRU (Saves 50% compute over Bidirectional)
        self.recurrent = nn.GRU(
            input_size=channels,
            hidden_size=hidden, 
            num_layers=1,
            batch_first=True,
            bidirectional=False # Essential for CPU speed
        )
        
        self.attention = SequenceAttentionBlock(
            model_dim=hidden,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )
        self.pool = TemporalAttentionPooling(hidden)
        
        # Classifier with Skip Connection
        classifier_in_dim = (hidden * 2) + n_features
        self.dense_shared = nn.Sequential(
            nn.Linear(classifier_in_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.state_branch = nn.Linear(dense_hidden, 1)
        self.direction_branch = nn.Linear(dense_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard input processing
        normed_x = self.sequence_norm(x)
        x_t = normed_x.transpose(1, 2)
        
        # Feature extraction
        x_pt = F.gelu(self.pointwise_in(x_t).transpose(1, 2))
        x_conv = torch.cat([self.conv_d1(x_pt), self.conv_d2(x_pt)], dim=-1)
        x_merged = self.conv_norm(self.conv_dropout(x_conv) + x_pt)
        
        # RNN + Attention
        x_rnn, _ = self.recurrent(x_merged)
        encoded = self.attention(x_rnn)
        
        # Aggregation
        pooled = self.pool(encoded)
        final_vector = torch.cat([encoded[:, -1, :], pooled, x[:, -1, :]], dim=1)
        
        shared = self.dense_shared(final_vector)
        
        # Dueling Output (Handles 2 or 3 classes automatically)
        state_val = self.state_branch(shared)
        direction_val = self.direction_branch(shared)
        
        if direction_val.shape[1] > 1:
            return state_val + (direction_val - direction_val.mean(dim=1, keepdim=True))
        return direction_val
