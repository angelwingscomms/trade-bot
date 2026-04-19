from __future__ import annotations

from .shared import *  # noqa: F401,F403


@dataclass(frozen=True)
class DiagnosticsConfig:
    current_config_name: str
    seq_len: int
    target_horizon: int
    primary_bar_seconds: int
    imbalance_min_ticks: int
    imbalance_ema_span: int
    feature_atr_period: int
    target_atr_period: int
    rv_period: int
    return_period: int
    warmup_bars: int
    default_fixed_move: float
    label_fixed_sl: float  # 0.0 means use default_fixed_move
    label_fixed_tp: float  # 0.0 means use default_fixed_move
    label_sl_multiplier: float
    label_tp_multiplier: float
    execution_sl_multiplier: float
    execution_tp_multiplier: float
    use_all_windows: bool
