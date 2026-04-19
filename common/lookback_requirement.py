"""Feature lookback requirements."""

from __future__ import annotations

from common.main_periods import main_periods
from common.past_dir_features import parse_past_dir_spec, past_dir_lookback_bars


def lookback_requirement(values: dict, feature_name: str) -> int:
    feature_atr_period = int(values["FEATURE_ATR_PERIOD"])
    feature_atr_ratio_period = int(values["FEATURE_ATR_RATIO_PERIOD"])
    feature_bollinger_period = int(values["FEATURE_BOLLINGER_PERIOD"])
    feature_donchian_fast_period = int(values["FEATURE_DONCHIAN_FAST_PERIOD"])
    feature_donchian_slow_period = int(values["FEATURE_DONCHIAN_SLOW_PERIOD"])
    feature_ret_2_period = int(values["FEATURE_RET_2_PERIOD"])
    feature_ret_3_period = int(values["FEATURE_RET_3_PERIOD"])
    feature_ret_6_period = int(values["FEATURE_RET_6_PERIOD"])
    feature_ret_12_period = int(values["FEATURE_RET_12_PERIOD"])
    feature_ret_20_period = int(values["FEATURE_RET_20_PERIOD"])
    feature_rsi_fast_period = int(values["FEATURE_RSI_FAST_PERIOD"])
    feature_rsi_slow_period = int(values["FEATURE_RSI_SLOW_PERIOD"])
    feature_slope_shift = int(values["FEATURE_SMA_SLOPE_SHIFT"])
    feature_sma_fast_period = int(values["FEATURE_SMA_FAST_PERIOD"])
    feature_sma_mid_period = int(values["FEATURE_SMA_MID_PERIOD"])
    feature_sma_slow_period = int(values["FEATURE_SMA_SLOW_PERIOD"])
    feature_sma_trend_fast_period = int(values["FEATURE_SMA_TREND_FAST_PERIOD"])
    feature_spread_z_period = int(values["FEATURE_SPREAD_Z_PERIOD"])
    feature_stoch_period = int(values["FEATURE_STOCH_PERIOD"])
    feature_stoch_smooth_period = int(values["FEATURE_STOCH_SMOOTH_PERIOD"])
    feature_tick_count_period = int(values["FEATURE_TICK_COUNT_PERIOD"])
    feature_tick_imbalance_fast_period = int(
        values["FEATURE_TICK_IMBALANCE_FAST_PERIOD"]
    )
    feature_tick_imbalance_slow_period = int(
        values["FEATURE_TICK_IMBALANCE_SLOW_PERIOD"]
    )
    return_period = int(values["RETURN_PERIOD"])
    rv_period = int(values["RV_PERIOD"])
    (
        main_short_period,
        main_medium_period,
        main_long_period,
        main_xlong_period,
        main_xxlong_period,
    ) = main_periods(values)
    macd_fast_period = int(values.get("FEATURE_MACD_FAST_PERIOD", 12))
    macd_slow_period = int(values.get("FEATURE_MACD_SLOW_PERIOD", 26))
    macd_signal_period = int(values.get("FEATURE_MACD_SIGNAL_PERIOD", 9))

    requirements = {
        "ret1": 1,
        "high_rel_prev": 1,
        "low_rel_prev": 1,
        "spread_rel": 0,
        "close_in_range": 0,
        "atr_rel": feature_atr_period,
        "rv": rv_period,
        "ret_n": return_period,
        "tick_imbalance": 0,
        "usdx_ret1": 1,
        "usdjpy_ret1": 1,
        "ret_2": feature_ret_2_period,
        "ret_3": feature_ret_3_period,
        "ret_6": feature_ret_6_period,
        "ret_12": feature_ret_12_period,
        "ret_20": feature_ret_20_period,
        "open_rel_prev": 1,
        "range_rel": 0,
        "body_rel": 0,
        "upper_wick_rel": 0,
        "lower_wick_rel": 0,
        "close_rel_sma_3": feature_sma_fast_period,
        "close_rel_sma_9": feature_sma_mid_period,
        "close_rel_sma_20": feature_sma_slow_period,
        "sma_3_9_gap": max(feature_sma_fast_period, feature_sma_mid_period),
        "sma_5_20_gap": max(feature_sma_trend_fast_period, feature_sma_slow_period),
        "sma_9_20_gap": max(feature_sma_mid_period, feature_sma_slow_period),
        "sma_slope_9": feature_sma_mid_period + feature_slope_shift,
        "sma_slope_20": feature_sma_slow_period + feature_slope_shift,
        "rv_3": feature_ret_3_period,
        "rv_6": feature_ret_6_period,
        "rv_18": int(values["FEATURE_RV_LONG_PERIOD"]),
        "donchian_pos_9": feature_donchian_fast_period,
        "donchian_width_9": feature_donchian_fast_period,
        "donchian_pos_20": feature_donchian_slow_period,
        "donchian_width_20": feature_donchian_slow_period,
        "tick_count_rel_9": feature_tick_count_period,
        "tick_count_z_9": feature_tick_count_period,
        "tick_count_chg": 1,
        "tick_imbalance_sma_5": feature_tick_imbalance_fast_period,
        "tick_imbalance_sma_9": feature_tick_imbalance_slow_period,
        "spread_z_9": feature_spread_z_period,
        "rsi_6": feature_rsi_fast_period,
        "rsi_14": feature_rsi_slow_period,
        "stoch_k_9": feature_stoch_period,
        "stoch_d_3": feature_stoch_period + feature_stoch_smooth_period - 1,
        "stoch_gap": feature_stoch_period + feature_stoch_smooth_period - 1,
        "bollinger_pos_20": feature_bollinger_period,
        "bollinger_width_20": feature_bollinger_period,
        "atr_ratio_20": feature_atr_period + feature_atr_ratio_period - 1,
        "spread_abs": 0,
        "bar_duration_ms": 0,
        "rsi_9": main_short_period,
        "rsi_18": main_medium_period,
        "rsi_27": main_long_period,
        "atr_9": main_short_period,
        "atr_18": main_medium_period,
        "atr_27": main_long_period,
        "macd_line": macd_slow_period + macd_signal_period - 1,
        "macd_signal": macd_slow_period + macd_signal_period - 1,
        "macd_hist": macd_slow_period + macd_signal_period - 1,
        "ema_gap_9": main_short_period,
        "ema_gap_18": main_medium_period,
        "ema_gap_27": main_long_period,
        "ema_gap_54": main_xlong_period,
        "ema_gap_144": main_xxlong_period,
        "cci_9": main_short_period,
        "cci_18": main_medium_period,
        "cci_27": main_long_period,
        "willr_9": main_short_period,
        "willr_18": main_medium_period,
        "willr_27": main_long_period,
        "mom_9": main_short_period,
        "mom_18": main_medium_period,
        "mom_27": main_long_period,
        "usdx_pct_change": 1,
        "usdjpy_pct_change": 1,
        "bollinger_width_9": main_short_period,
        "bollinger_width_18": main_medium_period,
        "bollinger_width_27": main_long_period,
        "hour_sin": 0,
        "hour_cos": 0,
        "minute_sin": 0,
        "minute_cos": 0,
        "day_of_week_scaled": 0,
    }
    if feature_name in requirements:
        return requirements[feature_name]

    # Dynamic PAST_DIR_<N>_S / PAST_DIR_<N>_T features
    past_dir_spec = parse_past_dir_spec(feature_name)
    if past_dir_spec is not None:
        n, unit = past_dir_spec
        if unit == "T":
            return n
        # _S: need bar_seconds to convert wall-clock seconds to bars
        bar_seconds = int(values.get("PRIMARY_BAR_SECONDS", 0))
        return past_dir_lookback_bars(feature_name, bar_seconds)

    raise KeyError(f"Unknown feature name: {feature_name!r}")
