"""Chronos-Bolt loading and wrapper helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

DEFAULT_CHRONOS_BOLT_MODEL_ID = "amazon/chronos-bolt-tiny"
CHRONOS_BOLT_MODEL_IDS = (
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
)
CHRONOS_BOLT_REQUIRED_FEATURES = (
    "ret1",
    "spread_rel",
    "atr_rel",
)
LOGIT_EPS = 1e-6


def _feature_index_map(feature_columns: Sequence[str]) -> dict[str, int]:
    index_map = {name: idx for idx, name in enumerate(feature_columns)}
    missing = [name for name in CHRONOS_BOLT_REQUIRED_FEATURES if name not in index_map]
    if missing:
        raise ValueError(f"Chronos-Bolt backend requires features {CHRONOS_BOLT_REQUIRED_FEATURES}, missing {missing}")
    return index_map


def _quantile_weights(quantile_levels: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    quantiles = torch.tensor(list(quantile_levels), device=device, dtype=dtype)
    boundaries = torch.cat(
        [
            torch.tensor([0.0], device=device, dtype=dtype),
            quantiles,
            torch.tensor([1.0], device=device, dtype=dtype),
        ]
    )
    masses = (boundaries[2:] - boundaries[:-2]) / 2
    return masses / masses.sum()


def _normalize_context_tail_lengths(context_tail_lengths: Sequence[int] | None) -> tuple[int, ...]:
    if not context_tail_lengths:
        return (0,)

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_value in context_tail_lengths:
        value = max(0, int(raw_value))
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized) if normalized else (0,)


class OnnxSafeInstanceNorm(nn.Module):
    """ONNX-safe replacement for Chronos-Bolt instance normalization on dense contexts."""

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.use_arcsinh = bool(use_arcsinh)

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            # Our exported MT5 feature windows are fully observed, so plain mean/variance
            # matches Chronos-Bolt's masked normalization without relying on nanmean.
            loc = x.mean(dim=-1, keepdim=True)
            scale = (x - loc).square().mean(dim=-1, keepdim=True).sqrt()
            scale = torch.clamp_min(scale, self.eps)
        else:
            loc, scale = loc_scale

        scaled_x = (x - loc) / scale
        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)
        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        if self.use_arcsinh:
            x = torch.sinh(x)
        return (x * scale + loc).to(orig_dtype)


class OnnxSafePatch(nn.Module):
    """Patch dense sequences without `unfold`, which the legacy ONNX exporter rejects here."""

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        if self.patch_stride != self.patch_size:
            raise ValueError(
                "OnnxSafePatch currently supports only non-overlapping Chronos-Bolt patches "
                f"(patch_size={self.patch_size}, patch_stride={self.patch_stride})."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        remainder = length % self.patch_size
        if remainder != 0:
            padding = torch.zeros(
                (*x.shape[:-1], self.patch_size - remainder),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((padding, x), dim=-1)
        patch_count = x.shape[-1] // self.patch_size
        return x.contiguous().reshape(*x.shape[:-1], patch_count, self.patch_size)


class ChronosBoltBarrierClassifier(nn.Module):
    def __init__(
        self,
        bolt_model: nn.Module,
        quantile_levels: Sequence[float],
        median: Sequence[float],
        iqr: Sequence[float],
        feature_columns: Sequence[str],
        prediction_length: int,
        label_tp_multiplier: float,
        label_sl_multiplier: float,
        context_tail_lengths: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.bolt = bolt_model
        self.prediction_length = int(prediction_length)
        self.label_tp_multiplier = float(label_tp_multiplier)
        self.label_sl_multiplier = float(label_sl_multiplier)

        index_map = _feature_index_map(feature_columns)
        self.ret1_index = index_map["ret1"]
        self.spread_feature_index = index_map["spread_rel"]
        self.atr_feature_index = index_map["atr_rel"]

        model_prediction_length = int(getattr(self.bolt.chronos_config, "prediction_length"))
        if self.prediction_length > model_prediction_length:
            raise ValueError(
                f"Chronos-Bolt backend supports prediction_length <= {model_prediction_length}, "
                f"but TARGET_HORIZON={self.prediction_length}."
            )
        self.patch_size = int(getattr(self.bolt.chronos_config, "input_patch_size", 0))
        self.context_tail_lengths = (0,)
        self.set_context_tail_lengths(context_tail_lengths)

        self.register_buffer("input_median", torch.as_tensor(median, dtype=torch.float32))
        self.register_buffer("input_iqr", torch.as_tensor(iqr, dtype=torch.float32))
        self.register_buffer(
            "quantile_weights",
            _quantile_weights(quantile_levels, device=torch.device("cpu"), dtype=torch.float32),
        )

    def set_context_tail_lengths(self, context_tail_lengths: Sequence[int] | None) -> None:
        self.context_tail_lengths = _normalize_context_tail_lengths(context_tail_lengths)
        tail_label = "full" if self.context_tail_lengths == (0,) else "_".join(
            "full" if tail_length <= 0 else str(tail_length) for tail_length in self.context_tail_lengths
        )
        self.backend_name = f"chronos-bolt-zero-shot-close-barrier-context-{tail_label}"

    def _raw_features(self, x_scaled: torch.Tensor) -> torch.Tensor:
        median = self.input_median.to(device=x_scaled.device, dtype=x_scaled.dtype)
        iqr = self.input_iqr.to(device=x_scaled.device, dtype=x_scaled.dtype)
        return x_scaled * iqr.view(1, 1, -1) + median.view(1, 1, -1)

    def _build_context_series(self, x_raw: torch.Tensor) -> torch.Tensor:
        raw_ret1 = x_raw[:, :, self.ret1_index]
        # Chronos-Bolt's official inference API is univariate here, so we reconstruct
        # a relative log-price path from the ret1 feature and forecast that sequence.
        return torch.cumsum(raw_ret1, dim=1)

    def _crop_context_series(self, context_series: torch.Tensor, tail_length: int) -> torch.Tensor:
        if tail_length <= 0 or tail_length >= context_series.shape[1]:
            return context_series
        return context_series[:, -tail_length:]

    def _predict_price_quantiles(self, context_series: torch.Tensor) -> torch.Tensor:
        context_mask = torch.ones_like(context_series, dtype=context_series.dtype, device=context_series.device)
        quantile_preds = self.bolt(context=context_series, mask=context_mask).quantile_preds
        return quantile_preds[:, :, : self.prediction_length]

    def _quantile_signal_probs(self, x_raw: torch.Tensor, context_series: torch.Tensor, predicted_levels: torch.Tensor) -> torch.Tensor:
        device = predicted_levels.device
        dtype = predicted_levels.dtype
        batch_size, quantile_count, _horizon = predicted_levels.shape
        scenario_weights = self.quantile_weights.to(device=device, dtype=dtype).view(1, quantile_count)

        current_level = context_series[:, -1].unsqueeze(1).unsqueeze(2)
        future_close_rel = torch.exp(torch.clamp(predicted_levels - current_level, min=-20.0, max=20.0))

        last_bar = x_raw[:, -1, :]
        current_spread_rel = torch.clamp_min(last_bar[:, self.spread_feature_index], 0.0)
        current_atr_rel = torch.clamp_min(last_bar[:, self.atr_feature_index], LOGIT_EPS)

        long_tp = 1.0 + current_spread_rel + self.label_tp_multiplier * current_atr_rel
        long_sl = 1.0 + current_spread_rel - self.label_sl_multiplier * current_atr_rel
        short_tp = 1.0 - self.label_tp_multiplier * current_atr_rel
        short_sl = 1.0 + self.label_sl_multiplier * current_atr_rel
        valid_thresholds = (long_tp > long_sl) & (long_sl > 0.0) & (short_tp > 0.0) & (short_sl > 1.0)

        long_result = torch.zeros(batch_size, quantile_count, device=device, dtype=torch.int64)
        short_result = torch.zeros(batch_size, quantile_count, device=device, dtype=torch.int64)
        one = torch.ones_like(long_result)
        neg_one = -one

        for step in range(self.prediction_length):
            future_close = future_close_rel[:, :, step]

            unresolved_long = long_result == 0
            long_hit_tp = future_close >= long_tp.unsqueeze(1)
            long_hit_sl = future_close <= long_sl.unsqueeze(1)
            long_result = torch.where(unresolved_long & long_hit_tp & ~long_hit_sl, one, long_result)
            long_result = torch.where(unresolved_long & long_hit_sl, neg_one, long_result)

            unresolved_short = short_result == 0
            short_hit_tp = future_close <= short_tp.unsqueeze(1)
            short_hit_sl = future_close >= short_sl.unsqueeze(1)
            short_result = torch.where(unresolved_short & short_hit_tp & ~short_hit_sl, one, short_result)
            short_result = torch.where(unresolved_short & short_hit_sl, neg_one, short_result)

        buy_mask = (long_result == 1) & (short_result != 1)
        sell_mask = (short_result == 1) & (long_result != 1)
        hold_mask = ~(buy_mask | sell_mask)

        hold_prob = (hold_mask.to(dtype) * scenario_weights).sum(dim=1)
        buy_prob = (buy_mask.to(dtype) * scenario_weights).sum(dim=1)
        sell_prob = (sell_mask.to(dtype) * scenario_weights).sum(dim=1)
        probs = torch.stack([hold_prob, buy_prob, sell_prob], dim=1)

        invalid_probs = torch.zeros_like(probs)
        invalid_probs[:, 0] = 1.0
        probs = torch.where(valid_thresholds.unsqueeze(1), probs, invalid_probs)
        probs = torch.clamp_min(probs, LOGIT_EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        x_raw = self._raw_features(x_scaled)
        full_context_series = self._build_context_series(x_raw)
        member_probs = []

        for tail_length in self.context_tail_lengths:
            context_series = self._crop_context_series(full_context_series, tail_length)
            predicted_levels = self._predict_price_quantiles(context_series)
            probs = self._quantile_signal_probs(x_raw, context_series, predicted_levels)
            member_probs.append(probs)

        probs = torch.stack(member_probs, dim=0).mean(dim=0)
        probs = torch.clamp_min(probs, LOGIT_EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return torch.log(probs)


def load_chronos_bolt_barrier_model(
    *,
    device: torch.device,
    model_id: str,
    median: Sequence[float],
    iqr: Sequence[float],
    feature_columns: Sequence[str],
    prediction_length: int,
    use_atr_risk: bool,
    label_tp_multiplier: float,
    label_sl_multiplier: float,
    context_tail_lengths: Sequence[int] | None = None,
) -> ChronosBoltBarrierClassifier:
    if not use_atr_risk:
        raise ValueError(
            "Chronos-Bolt backend currently supports ATR-based label risk only. "
            "Fixed-risk labels require absolute price scale, which is not available in the exported MT5 feature tensor."
        )

    try:
        from chronos import BaseChronosPipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Chronos-Bolt backend requires the official `chronos-forecasting` package. "
            "Install dependencies from requirements.txt and rerun `nn.py --chronos-bolt`."
        ) from exc

    load_kwargs: dict[str, object] = {
        "device_map": device.type,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = torch.float32

    try:
        pipeline = BaseChronosPipeline.from_pretrained(model_id, **load_kwargs)
    except OSError as exc:
        message = str(exc).lower()
        if "paging file is too small" in message:
            raise RuntimeError(
                f"Loading {model_id} failed because Windows reported that the paging file is too small. "
                "Chronos-Bolt is much lighter than Chronos-2, but this machine still needs a larger page file or more free RAM/swap "
                "for the checkpoint load to complete."
            ) from exc
        raise RuntimeError(f"Failed to load {model_id}: {exc}") from exc

    bolt_model = pipeline.model.to(device).eval()
    if hasattr(bolt_model, "instance_norm"):
        original_instance_norm = bolt_model.instance_norm
        bolt_model.instance_norm = OnnxSafeInstanceNorm(
            eps=float(getattr(original_instance_norm, "eps", 1e-5)),
            use_arcsinh=bool(getattr(original_instance_norm, "use_arcsinh", False)),
        )
    if hasattr(bolt_model, "patch"):
        original_patch = bolt_model.patch
        bolt_model.patch = OnnxSafePatch(
            patch_size=int(getattr(original_patch, "patch_size", 0)),
            patch_stride=int(getattr(original_patch, "patch_stride", 0)),
        )
    for parameter in bolt_model.parameters():
        parameter.requires_grad_(False)

    return ChronosBoltBarrierClassifier(
        bolt_model=bolt_model,
        quantile_levels=tuple(pipeline.quantiles),
        median=median,
        iqr=iqr,
        feature_columns=tuple(feature_columns),
        prediction_length=prediction_length,
        label_tp_multiplier=label_tp_multiplier,
        label_sl_multiplier=label_sl_multiplier,
        context_tail_lengths=context_tail_lengths,
    )
