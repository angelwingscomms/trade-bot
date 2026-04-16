"""MiniRocket feature transforms plus lightweight classifier/export wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


KERNEL_INDICES = np.asarray(list(combinations(range(9), 3)), dtype=np.int64)
BASE_KERNELS = np.full((len(KERNEL_INDICES), 9), -1.0, dtype=np.float32)
for kernel_index, combo in enumerate(KERNEL_INDICES):
    BASE_KERNELS[kernel_index, combo] = 2.0


@dataclass
class MiniRocketTransformParameters:
    num_channels_per_combination: np.ndarray
    channel_indices: np.ndarray
    dilations: np.ndarray
    num_features_per_dilation: np.ndarray
    biases: np.ndarray
    num_channels: int
    input_length: int

    @property
    def num_features(self) -> int:
        return int(len(self.biases))

    @property
    def num_tokens(self) -> int:
        return int(len(self.dilations))

    @property
    def max_features_per_dilation(self) -> int:
        if len(self.num_features_per_dilation) == 0:
            return 0
        return int(np.max(self.num_features_per_dilation))

    @property
    def token_feature_dim(self) -> int:
        return int(len(KERNEL_INDICES) * self.max_features_per_dilation)


def _fit_dilations(
    input_length: int,
    num_features: int,
    max_dilations_per_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_kernels = len(KERNEL_INDICES)
    num_features_per_kernel = max(1, num_features // num_kernels)
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)

    remainder = num_features_per_kernel - int(np.sum(num_features_per_dilation))
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations.astype(np.int32), num_features_per_dilation.astype(np.int32)


def _quantiles(count: int) -> np.ndarray:
    phi = (np.sqrt(5.0) + 1.0) / 2.0
    return np.asarray([(i * phi) % 1.0 for i in range(1, count + 1)], dtype=np.float32)


def _sample_channel_combinations(
    num_channels: int,
    num_combinations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    max_num_channels = min(num_channels, 9)
    max_exponent = np.log2(max_num_channels + 1)
    num_channels_per_combination = (2 ** rng.uniform(0.0, max_exponent, num_combinations)).astype(
        np.int32
    )
    num_channels_per_combination = np.clip(num_channels_per_combination, 1, max_num_channels)

    channel_indices = np.zeros(int(num_channels_per_combination.sum()), dtype=np.int32)
    offset = 0
    for combination_index in range(num_combinations):
        count = int(num_channels_per_combination[combination_index])
        next_offset = offset + count
        channel_indices[offset:next_offset] = rng.choice(num_channels, count, replace=False)
        offset = next_offset

    return num_channels_per_combination, channel_indices


def _compute_same_padded_response(
    sample: np.ndarray,
    channel_subset: np.ndarray,
    kernel_index: int,
    dilation: int,
) -> np.ndarray:
    selected = sample[channel_subset]
    if selected.ndim == 1:
        selected = selected[None, :]
    selected = selected.astype(np.float32, copy=False)

    kernel = BASE_KERNELS[kernel_index]
    input_length = selected.shape[1]
    padding = ((9 - 1) * dilation) // 2
    padded = np.pad(selected, ((0, 0), (padding, padding)), mode="constant")
    response = np.zeros(input_length, dtype=np.float32)

    for tap_index, weight in enumerate(kernel):
        start = tap_index * dilation
        end = start + input_length
        response += weight * padded[:, start:end].sum(axis=0)

    return response


def fit_minirocket(
    x_train_channels_first: np.ndarray,
    num_features: int = 10_080,
    max_dilations_per_kernel: int = 32,
    seed: int = 42,
) -> MiniRocketTransformParameters:
    x_train = np.asarray(x_train_channels_first, dtype=np.float32)
    if x_train.ndim != 3:
        raise ValueError("MiniRocket expects training input in [examples, channels, length] format.")

    num_examples, num_channels, input_length = x_train.shape
    if num_examples == 0:
        raise ValueError("MiniRocket fitting requires at least one training example.")
    if input_length < 9:
        raise ValueError("MiniRocket requires sequence length >= 9.")

    num_kernels = len(KERNEL_INDICES)
    rounded_features = max(num_kernels, num_kernels * ((num_features + num_kernels - 1) // num_kernels))
    dilations, num_features_per_dilation = _fit_dilations(
        input_length=input_length,
        num_features=rounded_features,
        max_dilations_per_kernel=max_dilations_per_kernel,
    )
    num_features_per_kernel = int(np.sum(num_features_per_dilation))
    quantiles = _quantiles(num_kernels * num_features_per_kernel)
    num_combinations = num_kernels * len(dilations)
    rng = np.random.default_rng(seed)
    num_channels_per_combination, channel_indices = _sample_channel_combinations(
        num_channels=num_channels,
        num_combinations=num_combinations,
        rng=rng,
    )

    biases = np.zeros(num_kernels * num_features_per_kernel, dtype=np.float32)
    feature_offset = 0
    channel_offset = 0
    combination_index = 0
    for dilation_index, dilation in enumerate(dilations):
        features_this_dilation = int(num_features_per_dilation[dilation_index])
        for kernel_index in range(num_kernels):
            feature_end = feature_offset + features_this_dilation
            channel_count = int(num_channels_per_combination[combination_index])
            channel_end = channel_offset + channel_count
            channel_subset = channel_indices[channel_offset:channel_end]
            sample = x_train[int(rng.integers(num_examples))]
            response = _compute_same_padded_response(
                sample=sample,
                channel_subset=channel_subset,
                kernel_index=kernel_index,
                dilation=int(dilation),
            )
            biases[feature_offset:feature_end] = np.quantile(
                response,
                quantiles[feature_offset:feature_end],
            ).astype(np.float32)

            feature_offset = feature_end
            channel_offset = channel_end
            combination_index += 1

    return MiniRocketTransformParameters(
        num_channels_per_combination=num_channels_per_combination,
        channel_indices=channel_indices,
        dilations=dilations,
        num_features_per_dilation=num_features_per_dilation,
        biases=biases,
        num_channels=num_channels,
        input_length=input_length,
    )


class MiniRocketFeatureExtractor(nn.Module):
    def __init__(
        self,
        parameters: MiniRocketTransformParameters,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        token_mean: np.ndarray | None = None,
        token_std: np.ndarray | None = None,
    ):
        super().__init__()
        self.num_channels = int(parameters.num_channels)
        self.input_length = int(parameters.input_length)
        self.num_features = int(parameters.num_features)
        self.num_kernels = len(KERNEL_INDICES)
        self.dilations = [int(v) for v in parameters.dilations.tolist()]
        self.num_features_per_dilation = [int(v) for v in parameters.num_features_per_dilation.tolist()]
        self.num_tokens = int(parameters.num_tokens)
        self.max_features_per_dilation = int(parameters.max_features_per_dilation)
        self.token_feature_dim = int(parameters.token_feature_dim)
        self.backend_name = "minirocket-multivariate"

        depthwise_weight = (
            torch.from_numpy(BASE_KERNELS).unsqueeze(1).repeat(self.num_channels, 1, 1).contiguous()
        )
        self.register_buffer("depthwise_weight", depthwise_weight)

        channel_offset = 0
        feature_offset = 0
        combination_index = 0
        for dilation_index, features_this_dilation in enumerate(self.num_features_per_dilation):
            channel_mask = np.zeros((len(KERNEL_INDICES), self.num_channels, 1), dtype=np.float32)
            bias_matrix = np.zeros((len(KERNEL_INDICES), features_this_dilation), dtype=np.float32)
            for kernel_index in range(len(KERNEL_INDICES)):
                count = int(parameters.num_channels_per_combination[combination_index])
                next_channel_offset = channel_offset + count
                channels = parameters.channel_indices[channel_offset:next_channel_offset]
                channel_mask[kernel_index, channels, 0] = 1.0

                next_feature_offset = feature_offset + features_this_dilation
                bias_matrix[kernel_index] = parameters.biases[feature_offset:next_feature_offset]

                channel_offset = next_channel_offset
                feature_offset = next_feature_offset
                combination_index += 1

            self.register_buffer(
                f"channel_mask_{dilation_index}",
                torch.from_numpy(channel_mask),
            )
            self.register_buffer(
                f"bias_matrix_{dilation_index}",
                torch.from_numpy(bias_matrix),
            )

        if feature_mean is None:
            feature_mean = np.zeros(self.num_features, dtype=np.float32)
        if feature_std is None:
            feature_std = np.ones(self.num_features, dtype=np.float32)
        feature_std = np.where(np.asarray(feature_std, dtype=np.float32) < 1e-6, 1.0, feature_std)
        self.register_buffer("feature_mean", torch.from_numpy(np.asarray(feature_mean, dtype=np.float32)))
        self.register_buffer("feature_std", torch.from_numpy(np.asarray(feature_std, dtype=np.float32)))

        if token_mean is None:
            token_mean = np.zeros((self.num_tokens, self.token_feature_dim), dtype=np.float32)
        if token_std is None:
            token_std = np.ones((self.num_tokens, self.token_feature_dim), dtype=np.float32)
        token_std = np.where(np.asarray(token_std, dtype=np.float32) < 1e-6, 1.0, token_std)
        self.register_buffer("token_mean", torch.from_numpy(np.asarray(token_mean, dtype=np.float32)))
        self.register_buffer("token_std", torch.from_numpy(np.asarray(token_std, dtype=np.float32)))

    def _extract_raw_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("MiniRocketFeatureExtractor expects [batch, seq_len, features] input.")

        x_channels_first = x.transpose(1, 2).contiguous()
        batch_size = x_channels_first.shape[0]
        flat_features: list[torch.Tensor] = []
        token_features: list[torch.Tensor] = []

        for dilation_index, dilation in enumerate(self.dilations):
            padding = ((9 - 1) * dilation) // 2
            conv = F.conv1d(
                x_channels_first,
                self.depthwise_weight,
                padding=padding,
                dilation=dilation,
                groups=self.num_channels,
            )
            conv = conv.view(batch_size, self.num_channels, self.num_kernels, -1).permute(0, 2, 1, 3)
            response = (conv * getattr(self, f"channel_mask_{dilation_index}").unsqueeze(0)).sum(dim=2)
            bias_matrix = getattr(self, f"bias_matrix_{dilation_index}")
            should_trim = padding > 0 and (self.input_length - (2 * padding)) > 0
            token_block = x.new_zeros((batch_size, self.num_kernels, self.max_features_per_dilation))

            for kernel_index in range(self.num_kernels):
                kernel_response = response[:, kernel_index, :]
                if (dilation_index + kernel_index) % 2 == 1 and should_trim:
                    kernel_response = kernel_response[:, padding:-padding]
                kernel_biases = bias_matrix[kernel_index].view(1, -1, 1)
                kernel_features = (kernel_response.unsqueeze(1) > kernel_biases).to(x.dtype).mean(dim=-1)
                flat_features.append(kernel_features)
                token_block[:, kernel_index, : kernel_features.shape[1]] = kernel_features

            token_features.append(token_block.reshape(batch_size, -1))

        flat_output = torch.cat(flat_features, dim=1)
        token_output = torch.stack(token_features, dim=1)
        return flat_output, token_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _token_output = self._extract_raw_features(x)
        return (output - self.feature_mean) / self.feature_std

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        _flat_output, token_output = self._extract_raw_features(x)
        return (token_output - self.token_mean) / self.token_std


class MiniRocketMultiAttentionHead(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        n_classes: int = 3,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("MiniRocketMultiAttentionHead requires at least one token.")
        if token_dim <= 0:
            raise ValueError("MiniRocketMultiAttentionHead requires token_dim > 0.")
        if num_heads <= 0:
            raise ValueError("MiniRocketMultiAttentionHead requires num_heads > 0.")
        if num_layers <= 0:
            raise ValueError("MiniRocketMultiAttentionHead requires num_layers > 0.")

        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.model_dim = max(self.num_heads, int(model_dim))
        if self.model_dim % self.num_heads != 0:
            self.model_dim += self.num_heads - (self.model_dim % self.num_heads)

        self.token_projection = nn.Linear(self.token_dim, self.model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, self.model_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            MiniRocketAttentionBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        )
        self.pool_projection = nn.Linear(self.model_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_dim, n_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("MiniRocketMultiAttentionHead expects [batch, tokens, token_dim] input.")

        x = self.input_dropout(self.token_projection(tokens) + self.position_embedding[:, : tokens.shape[1]])
        for layer in self.layers:
            x = layer(x)
        pool_weights = torch.softmax(self.pool_projection(x).squeeze(-1), dim=-1)
        pooled = torch.sum(x * pool_weights.unsqueeze(-1), dim=1)
        summary = x.mean(dim=1)
        return self.classifier(torch.cat([pooled, summary], dim=1))


class MiniRocketAttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
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
        batch_size, num_tokens, _channels = x.shape
        q = self.q_proj(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = self.attention_dropout(torch.softmax(attention_scores, dim=-1))
        attended = torch.matmul(attention_probs, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.model_dim)
        return self.out_proj(attended)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self._self_attention(self.norm1(x)))
        x = x + self.residual_dropout(self.feed_forward(self.norm2(x)))
        return x


class MiniRocketClassifier(nn.Module):
    def __init__(
        self,
        parameters: MiniRocketTransformParameters,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        n_classes: int = 3,
        token_mean: np.ndarray | None = None,
        token_std: np.ndarray | None = None,
        head_type: str = "multiattention",
        attention_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.head_type = str(head_type)
        self.backend_name = (
            "minirocket-multivariate-attention"
            if self.head_type == "multiattention"
            else "minirocket-multivariate"
        )
        self.extractor = MiniRocketFeatureExtractor(
            parameters=parameters,
            feature_mean=feature_mean,
            feature_std=feature_std,
            token_mean=token_mean,
            token_std=token_std,
        )
        if self.head_type == "linear":
            self.head = nn.Linear(self.extractor.num_features, n_classes)
        elif self.head_type == "multiattention":
            self.head = MiniRocketMultiAttentionHead(
                num_tokens=self.extractor.num_tokens,
                token_dim=self.extractor.token_feature_dim,
                n_classes=n_classes,
                model_dim=attention_dim,
                num_heads=attention_heads,
                num_layers=attention_layers,
                dropout=attention_dropout,
            )
        else:
            raise ValueError(f"Unsupported MiniRocket head_type: {self.head_type}")

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor.encode_tokens(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "linear":
            return self.head(self.encode_features(x))
        return self.head(self.encode_tokens(x))


@torch.no_grad()
def transform_sequences(
    parameters: MiniRocketTransformParameters,
    sequences: np.ndarray,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    extractor = MiniRocketFeatureExtractor(parameters=parameters).to(device)
    extractor.eval()
    features: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
        features.append(extractor(batch).cpu().numpy())
    if not features:
        return np.empty((0, extractor.num_features), dtype=np.float32)
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def transform_sequence_tokens(
    parameters: MiniRocketTransformParameters,
    sequences: np.ndarray,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    extractor = MiniRocketFeatureExtractor(parameters=parameters).to(device)
    extractor.eval()
    features: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
        features.append(extractor.encode_tokens(batch).cpu().numpy())
    if not features:
        return np.empty((0, extractor.num_tokens, extractor.token_feature_dim), dtype=np.float32)
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


__all__ = [
    "MiniRocketAttentionBlock",
    "MiniRocketClassifier",
    "MiniRocketFeatureExtractor",
    "MiniRocketMultiAttentionHead",
    "MiniRocketTransformParameters",
    "fit_minirocket",
    "transform_sequence_tokens",
    "transform_sequences",
]
