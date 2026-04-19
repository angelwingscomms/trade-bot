from __future__ import annotations

from .shared import *  # noqa: F401,F403


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    feature_atr_period: int
    feature_atr_ratio_period: int
    feature_bollinger_period: int
    feature_donchian_fast_period: int
    feature_donchian_slow_period: int
    feature_ret_2_period: int
    feature_ret_3_period: int
    feature_ret_6_period: int
    feature_ret_12_period: int
    feature_ret_20_period: int
    feature_rsi_fast_period: int
    feature_rsi_slow_period: int
    feature_rv_long_period: int
    feature_sma_fast_period: int
    feature_sma_mid_period: int
    feature_sma_slow_period: int
    feature_sma_slope_shift: int
    feature_sma_trend_fast_period: int
    feature_spread_z_period: int
    feature_stoch_period: int
    feature_stoch_smooth_period: int
    feature_tick_count_period: int
    feature_tick_imbalance_fast_period: int
    feature_tick_imbalance_slow_period: int
    rv_period: int
    return_period: int
    primary_bar_seconds: int
    main_short_period: int
    main_medium_period: int
    main_long_period: int
    main_xlong_period: int
    main_xxlong_period: int
    macd_fast_period: int
    macd_slow_period: int
    macd_signal_period: int

    @classmethod
    def from_values(cls, values: Mapping[str, Scalar]) -> "FeatureEngineeringConfig":
        return cls(
            feature_atr_period=int(values["FEATURE_ATR_PERIOD"]),
            feature_atr_ratio_period=int(values["FEATURE_ATR_RATIO_PERIOD"]),
            feature_bollinger_period=int(values["FEATURE_BOLLINGER_PERIOD"]),
            feature_donchian_fast_period=int(values["FEATURE_DONCHIAN_FAST_PERIOD"]),
            feature_donchian_slow_period=int(values["FEATURE_DONCHIAN_SLOW_PERIOD"]),
            feature_ret_2_period=int(values["FEATURE_RET_2_PERIOD"]),
            feature_ret_3_period=int(values["FEATURE_RET_3_PERIOD"]),
            feature_ret_6_period=int(values["FEATURE_RET_6_PERIOD"]),
            feature_ret_12_period=int(values["FEATURE_RET_12_PERIOD"]),
            feature_ret_20_period=int(values["FEATURE_RET_20_PERIOD"]),
            feature_rsi_fast_period=int(values["FEATURE_RSI_FAST_PERIOD"]),
            feature_rsi_slow_period=int(values["FEATURE_RSI_SLOW_PERIOD"]),
            feature_rv_long_period=int(values["FEATURE_RV_LONG_PERIOD"]),
            feature_sma_fast_period=int(values["FEATURE_SMA_FAST_PERIOD"]),
            feature_sma_mid_period=int(values["FEATURE_SMA_MID_PERIOD"]),
            feature_sma_slow_period=int(values["FEATURE_SMA_SLOW_PERIOD"]),
            feature_sma_slope_shift=int(values["FEATURE_SMA_SLOPE_SHIFT"]),
            feature_sma_trend_fast_period=int(values["FEATURE_SMA_TREND_FAST_PERIOD"]),
            feature_spread_z_period=int(values["FEATURE_SPREAD_Z_PERIOD"]),
            feature_stoch_period=int(values["FEATURE_STOCH_PERIOD"]),
            feature_stoch_smooth_period=int(values["FEATURE_STOCH_SMOOTH_PERIOD"]),
            feature_tick_count_period=int(values["FEATURE_TICK_COUNT_PERIOD"]),
            feature_tick_imbalance_fast_period=int(
                values["FEATURE_TICK_IMBALANCE_FAST_PERIOD"]
            ),
            feature_tick_imbalance_slow_period=int(
                values["FEATURE_TICK_IMBALANCE_SLOW_PERIOD"]
            ),
            rv_period=int(values["RV_PERIOD"]),
            return_period=int(values["RETURN_PERIOD"]),
            primary_bar_seconds=int(values.get("PRIMARY_BAR_SECONDS", 0)),
            main_short_period=int(values.get("FEATURE_MAIN_SHORT_PERIOD", 9)),
            main_medium_period=int(values.get("FEATURE_MAIN_MEDIUM_PERIOD", 18)),
            main_long_period=int(values.get("FEATURE_MAIN_LONG_PERIOD", 27)),
            main_xlong_period=int(values.get("FEATURE_MAIN_XLONG_PERIOD", 54)),
            main_xxlong_period=int(values.get("FEATURE_MAIN_XXLONG_PERIOD", 144)),
            macd_fast_period=int(values.get("FEATURE_MACD_FAST_PERIOD", 12)),
            macd_slow_period=int(values.get("FEATURE_MACD_SLOW_PERIOD", 26)),
            macd_signal_period=int(values.get("FEATURE_MACD_SIGNAL_PERIOD", 9)),
        )
