"""Shared feature column groups."""

from __future__ import annotations

from typing import Final, Mapping

from common.past_dir_features import parse_past_dir_features

MINIMAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "ret1",
    "high_rel_prev",
    "low_rel_prev",
    "spread_rel",
    "close_in_range",
    "atr_rel",
    "rv",
    "ret_n",
    "tick_imbalance",
)
GOLD_CONTEXT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "usdx_ret1",
    "usdjpy_ret1",
)
EXTRA_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "ret_2",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_20",
    "open_rel_prev",
    "range_rel",
    "body_rel",
    "upper_wick_rel",
    "lower_wick_rel",
    "close_rel_sma_3",
    "close_rel_sma_9",
    "close_rel_sma_20",
    "sma_3_9_gap",
    "sma_5_20_gap",
    "sma_9_20_gap",
    "sma_slope_9",
    "sma_slope_20",
    "rv_3",
    "rv_6",
    "rv_18",
    "donchian_pos_9",
    "donchian_width_9",
    "donchian_pos_20",
    "donchian_width_20",
    "tick_count_rel_9",
    "tick_count_z_9",
    "tick_count_chg",
    "tick_imbalance_sma_5",
    "tick_imbalance_sma_9",
    "spread_z_9",
    "rsi_6",
    "rsi_14",
    "stoch_k_9",
    "stoch_d_3",
    "stoch_gap",
    "bollinger_pos_20",
    "bollinger_width_20",
    "atr_ratio_20",
    "regime_5m",
    "regime_15m",
    "above_sma200_1m",
    "ema_slope_5m",
    "regime_5m_usdx",
    "regime_15m_usdx",
    "regime_5m_usdjpy",
    "regime_15m_usdjpy",
    "hl_ratio",
    "lh_ratio",
    "high_before_low",
    "timing_signal",
    "bar_range_norm",
)
MAIN_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "ret1",
    "spread_abs",
    "bar_duration_ms",
    "upper_wick_rel",
    "lower_wick_rel",
    "range_rel",
    "close_in_range",
    "rsi_9",
    "rsi_18",
    "rsi_27",
    "atr_9",
    "atr_18",
    "atr_27",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "ema_gap_9",
    "ema_gap_18",
    "ema_gap_27",
    "ema_gap_54",
    "ema_gap_144",
    "cci_9",
    "cci_18",
    "cci_27",
    "willr_9",
    "willr_18",
    "willr_27",
    "mom_9",
    "mom_18",
    "mom_27",
    "usdx_pct_change",
    "usdjpy_pct_change",
    "bollinger_width_9",
    "bollinger_width_18",
    "bollinger_width_27",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "day_of_week_scaled",
)
MAIN_GOLD_CONTEXT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "usdx_pct_change",
    "usdjpy_pct_change",
)
_STATIC_ALL_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(
        MINIMAL_FEATURE_COLUMNS
        + GOLD_CONTEXT_FEATURE_COLUMNS
        + EXTRA_FEATURE_COLUMNS
        + MAIN_FEATURE_COLUMNS
    )
)


def resolve_all_feature_columns(
    values: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return all known feature names, including enabled dynamic `past_dir_*`.

    `ALL_FEATURE_COLUMNS` remains the static built-in catalog. Pass config values
    here when code needs the full runtime feature universe.
    """

    feature_columns = list(_STATIC_ALL_FEATURE_COLUMNS)
    if values is not None:
        feature_columns.extend(parse_past_dir_features(dict(values)))
    return tuple(dict.fromkeys(feature_columns))


ALL_FEATURE_COLUMNS: Final[tuple[str, ...]] = resolve_all_feature_columns()
