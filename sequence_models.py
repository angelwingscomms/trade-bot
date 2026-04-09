from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SequenceInstanceNorm(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)


class SequenceAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = int(num_heads)
        self.model_dim = max(self.num_heads, int(model_dim))
        if self.model_dim % self.num_heads != 0:
            self.model_dim += self.num_heads - (self.model_dim % self.num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(self.model_dim)
        self.q_proj = nn.Linear(self.model_dim, self.model_dim)
        self.k_proj = nn.Linear(self.model_dim, self.model_dim)
        self.v_proj = nn.Linear(self.model_dim, self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.model_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(self.model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_dim * 2, self.model_dim),
        )

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = self.attention_dropout(torch.softmax(attention_scores, dim=-1))
        attended = torch.matmul(attention_probs, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        return self.out_proj(attended)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self._self_attention(self.norm1(x)))
        x = x + self.residual_dropout(self.feed_forward(self.norm2(x)))
        return x


class SequenceMultiAttentionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        n_classes: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("SequenceMultiAttentionHead requires input_dim > 0.")
        if num_heads <= 0:
            raise ValueError("SequenceMultiAttentionHead requires num_heads > 0.")
        if num_layers <= 0:
            raise ValueError("SequenceMultiAttentionHead requires num_layers > 0.")

        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.model_dim = max(self.num_heads, self.input_dim)
        if self.model_dim % self.num_heads != 0:
            self.model_dim += self.num_heads - (self.model_dim % self.num_heads)

        self.input_projection = (
            nn.Linear(self.input_dim, self.model_dim) if self.model_dim != self.input_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            SequenceAttentionBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        )
        self.pool_projection = nn.Linear(self.model_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("SequenceMultiAttentionHead expects [batch, seq_len, channels] input.")

        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        pool_weights = torch.softmax(self.pool_projection(x).squeeze(-1), dim=-1)
        pooled = torch.sum(x * pool_weights.unsqueeze(-1), dim=1)
        summary = x.mean(dim=1)
        return self.classifier(torch.cat([pooled, summary], dim=1))


class ProjectedMultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, head_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("ProjectedMultiHeadSelfAttention requires input_dim > 0.")
        if num_heads <= 0:
            raise ValueError("ProjectedMultiHeadSelfAttention requires num_heads > 0.")

        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim if head_dim is not None else input_dim)
        if self.head_dim <= 0:
            raise ValueError("ProjectedMultiHeadSelfAttention requires head_dim > 0.")

        inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.input_dim, inner_dim)
        self.k_proj = nn.Linear(self.input_dim, inner_dim)
        self.v_proj = nn.Linear(self.input_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, self.input_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("ProjectedMultiHeadSelfAttention expects [batch, seq_len, channels] input.")

        batch_size, seq_len, _channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = self.attention_dropout(torch.softmax(attention_scores, dim=-1))
        attended = torch.matmul(attention_probs, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attended)


class MishLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        if input_size <= 0:
            raise ValueError("MishLSTMCell requires input_size > 0.")
        if hidden_size <= 0:
            raise ValueError("MishLSTMCell requires hidden_size > 0.")

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


class FusionLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 20,
        n_classes: int = 3,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if n_features <= 0:
            raise ValueError("FusionLSTMClassifier requires n_features > 0.")
        if hidden <= 0:
            raise ValueError("FusionLSTMClassifier requires hidden > 0.")

        self.n_features = int(n_features)
        self.backend_name = "fusion-lstm-attention"
        self.sequence_norm = SequenceInstanceNorm(self.n_features)
        self.recurrent_cell = MishLSTMCell(input_size=self.n_features, hidden_size=self.n_features)
        self.attention = ProjectedMultiHeadSelfAttention(
            input_dim=self.n_features,
            num_heads=attention_heads,
            head_dim=self.n_features,
            dropout=attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.n_features, hidden),
            nn.Mish(),
            nn.Linear(hidden, n_classes),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        batch_size, seq_len, _channels = x.shape
        hidden_state = x.new_zeros(batch_size, self.n_features)
        cell_state = x.new_zeros(batch_size, self.n_features)
        outputs = []
        for timestep in range(seq_len):
            hidden_state, cell_state = self.recurrent_cell(x[:, timestep, :], (hidden_state, cell_state))
            outputs.append(hidden_state)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        attended = self.attention(encoded)
        pooled = (encoded + attended).mean(dim=1)
        return self.classifier(pooled)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.left_padding = max(0, (int(kernel_size) - 1) * int(dilation))
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv1(self.norm1(x))
        x = self.dropout(F.gelu(x))
        x = self.conv2(self.norm2(x))
        x = self.dropout(F.gelu(x))
        return residual + x


class RecurrentSequenceClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        cell_type: str = "lstm",
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
        backend_name: str = "",
    ):
        super().__init__()
        recurrent_type = cell_type.strip().lower()
        if recurrent_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported recurrent cell_type: {cell_type}")
        if hidden_size <= 0:
            raise ValueError("RecurrentSequenceClassifier requires hidden_size > 0.")
        if num_layers <= 0:
            raise ValueError("RecurrentSequenceClassifier requires num_layers > 0.")

        self.cell_type = recurrent_type
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.use_multihead_attention = bool(use_multihead_attention)
        self.num_directions = 2 if self.bidirectional else 1
        output_dim = self.hidden_size * self.num_directions

        default_backend = self.cell_type
        if self.bidirectional:
            default_backend = f"bi{default_backend}"
        if self.use_multihead_attention:
            default_backend = f"{default_backend}-attention"
        self.backend_name = backend_name or default_backend

        recurrent_dropout = dropout if self.num_layers > 1 else 0.0
        recurrent_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU

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


class TCNClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        hidden: int = 96,
        n_classes: int = 3,
        dropout: float = 0.1,
        n_layers: int = 4,
        kernel_size: int = 3,
        use_multihead_attention: bool = False,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("TCNClassifier requires channels > 0.")
        if n_layers <= 0:
            raise ValueError("TCNClassifier requires n_layers > 0.")
        if kernel_size <= 1:
            raise ValueError("TCNClassifier requires kernel_size > 1.")

        self.use_multihead_attention = bool(use_multihead_attention)
        self.backend_name = "tcn-attention" if self.use_multihead_attention else "tcn"
        self.sequence_norm = SequenceInstanceNorm(n_features)
        self.input_projection = nn.Linear(n_features, channels) if channels != n_features else nn.Identity()
        self.layers = nn.ModuleList(
            TemporalConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=2 ** layer_index,
                dropout=dropout,
            )
            for layer_index in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(channels)

        if self.use_multihead_attention:
            self.head = SequenceMultiAttentionHead(
                input_dim=channels,
                hidden=hidden,
                n_classes=n_classes,
                num_heads=attention_heads,
                num_layers=attention_layers,
                dropout=attention_dropout,
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(x)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        if self.use_multihead_attention:
            return self.head(encoded)
        return self.head(encoded[:, -1, :])


__all__ = [
    "CausalConv1d",
    "FusionLSTMClassifier",
    "MishLSTMCell",
    "ProjectedMultiHeadSelfAttention",
    "RecurrentSequenceClassifier",
    "SequenceAttentionBlock",
    "SequenceInstanceNorm",
    "SequenceMultiAttentionHead",
    "TCNClassifier",
    "TemporalConvBlock",
]
