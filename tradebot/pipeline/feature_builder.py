"""Feature engineering for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from common.features import GOLD_CONTEXT_FEATURE_COLUMNS, MAIN_GOLD_CONTEXT_FEATURE_COLUMNS
from tradebot.config_io import Scalar


EPS = 1e-10


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
            feature_tick_imbalance_fast_period=int(values["FEATURE_TICK_IMBALANCE_FAST_PERIOD"]),
            feature_tick_imbalance_slow_period=int(values["FEATURE_TICK_IMBALANCE_SLOW_PERIOD"]),
            rv_period=int(values["RV_PERIOD"]),
            return_period=int(values["RETURN_PERIOD"]),
            main_short_period=int(values.get("FEATURE_MAIN_SHORT_PERIOD", 9)),
            main_medium_period=int(values.get("FEATURE_MAIN_MEDIUM_PERIOD", 18)),
            main_long_period=int(values.get("FEATURE_MAIN_LONG_PERIOD", 27)),
            main_xlong_period=int(values.get("FEATURE_MAIN_XLONG_PERIOD", 54)),
            main_xxlong_period=int(values.get("FEATURE_MAIN_XXLONG_PERIOD", 144)),
            macd_fast_period=int(values.get("FEATURE_MACD_FAST_PERIOD", 12)),
            macd_slow_period=int(values.get("FEATURE_MACD_SLOW_PERIOD", 26)),
            macd_signal_period=int(values.get("FEATURE_MACD_SIGNAL_PERIOD", 9)),
        )


def rolling_population_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = rolling_population_std(series, window)
    return (series - mean) / (std + EPS)


def simple_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.rolling(period, min_periods=period).mean()
    avg_loss = losses.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return (100.0 - (100.0 / (1.0 + rs)) - 50.0) / 50.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    high_values = high.to_numpy(dtype=np.float64, copy=False)
    low_values = low.to_numpy(dtype=np.float64, copy=False)
    close_values = close.to_numpy(dtype=np.float64, copy=False)

    tr = np.empty(len(close_values), dtype=np.float64)
    if len(tr) == 0:
        return pd.Series(dtype=np.float64, index=close.index)

    tr[0] = high_values[0] - low_values[0]
    for i in range(1, len(tr)):
        tr[i] = max(
            high_values[i] - low_values[i],
            abs(high_values[i] - close_values[i - 1]),
            abs(low_values[i] - close_values[i - 1]),
        )

    atr = np.full(len(tr), np.nan, dtype=np.float64)
    if len(tr) >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / period

    return pd.Series(atr, index=close.index, dtype=np.float64)


def simple_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def macd_components(close: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(close, fast_period)
    slow = ema(close, slow_period)
    line = fast - slow
    signal = line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = line - signal
    return line, signal, hist


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period, min_periods=period).mean()
    mean_deviation = (typical - mean).abs().rolling(period, min_periods=period).mean()
    return (typical - mean) / (0.015 * (mean_deviation + EPS))


def willr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    rolling_high = high.rolling(period, min_periods=period).max()
    rolling_low = low.rolling(period, min_periods=period).min()
    return -100.0 * (rolling_high - close) / (rolling_high - rolling_low + EPS)


def momentum(close: pd.Series, period: int) -> pd.Series:
    return close - close.shift(period)


def bollinger_width(close: pd.Series, period: int) -> pd.Series:
    mean = close.rolling(period, min_periods=period).mean()
    std = rolling_population_std(close, period)
    upper = mean + (2.0 * std)
    lower = mean - (2.0 * std)
    return (upper - lower) / (mean + EPS)


def _aux_context_required(feature_columns: tuple[str, ...]) -> bool:
    required_columns = GOLD_CONTEXT_FEATURE_COLUMNS + MAIN_GOLD_CONTEXT_FEATURE_COLUMNS
    return any(name in feature_columns for name in required_columns)


def _resolve_aux_series(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    requires_aux_context = _aux_context_required(feature_columns)
    usdx_bid = df.get("usdx_bid")
    usdjpy_bid = df.get("usdjpy_bid")
    if requires_aux_context:
        if usdx_bid is None or usdjpy_bid is None:
            raise ValueError("Auxiliary USDX/USDJPY features were requested but the bar data is missing those columns.")
        if usdx_bid.notna().sum() == 0 or usdjpy_bid.notna().sum() == 0:
            raise ValueError("Auxiliary USDX/USDJPY features were requested but the bar data is empty for them.")
        return usdx_bid.ffill().bfill(), usdjpy_bid.ffill().bfill()

    fallback = df["close"].astype(float)
    return (
        fallback if usdx_bid is None else usdx_bid.fillna(fallback),
        fallback if usdjpy_bid is None else usdjpy_bid.fillna(fallback),
    )


def compute_feature_frame(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: FeatureEngineeringConfig,
) -> pd.DataFrame:
    close = df["close"].astype(float)
    open_price = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tick_count = df["tick_count"].astype(float)
    tick_imbalance = df["tick_imbalance"].astype(float)
    prev_close = close.shift(1)
    ret1 = np.log(close / (prev_close + EPS))
    atr_feature = wilder_atr(high, low, close, period=config.feature_atr_period)
    spread_rel = df["spread"] / (close + EPS)
    spread_abs = df.get("spread_mean", df["spread"]).astype(float)

    sma_fast = close.rolling(config.feature_sma_fast_period, min_periods=config.feature_sma_fast_period).mean()
    sma_trend_fast = close.rolling(
        config.feature_sma_trend_fast_period,
        min_periods=config.feature_sma_trend_fast_period,
    ).mean()
    sma_mid = close.rolling(config.feature_sma_mid_period, min_periods=config.feature_sma_mid_period).mean()
    sma_slow = close.rolling(config.feature_sma_slow_period, min_periods=config.feature_sma_slow_period).mean()
    rv_3 = rolling_population_std(ret1, config.feature_ret_3_period)
    rv_6 = rolling_population_std(ret1, config.feature_ret_6_period)
    rv_18 = rolling_population_std(ret1, config.feature_rv_long_period)
    high_fast = high.rolling(config.feature_donchian_fast_period, min_periods=config.feature_donchian_fast_period).max()
    low_fast = low.rolling(config.feature_donchian_fast_period, min_periods=config.feature_donchian_fast_period).min()
    high_slow = high.rolling(config.feature_donchian_slow_period, min_periods=config.feature_donchian_slow_period).max()
    low_slow = low.rolling(config.feature_donchian_slow_period, min_periods=config.feature_donchian_slow_period).min()
    high_stoch = high.rolling(config.feature_stoch_period, min_periods=config.feature_stoch_period).max()
    low_stoch = low.rolling(config.feature_stoch_period, min_periods=config.feature_stoch_period).min()
    stoch_k_9 = (close - low_stoch) / (high_stoch - low_stoch + EPS)
    stoch_d_3 = stoch_k_9.rolling(
        config.feature_stoch_smooth_period,
        min_periods=config.feature_stoch_smooth_period,
    ).mean()
    bollinger_std_20 = rolling_population_std(close, config.feature_bollinger_period)
    main_macd_line, main_macd_signal, main_macd_hist = macd_components(
        close,
        fast_period=config.macd_fast_period,
        slow_period=config.macd_slow_period,
        signal_period=config.macd_signal_period,
    )
    usdx_bid, usdjpy_bid = _resolve_aux_series(df, feature_columns)
    time_open = df["time_open"].astype(np.int64)
    time_close = df["time_close"].astype(np.int64)
    time_index = pd.to_datetime(time_open, unit="ms", utc=True)

    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = ret1
    feat["high_rel_prev"] = np.log(high / (prev_close + EPS))
    feat["low_rel_prev"] = np.log(low / (prev_close + EPS))
    feat["spread_rel"] = spread_rel
    feat["close_in_range"] = (close - low) / (high - low + 1e-8)
    feat["atr_rel"] = atr_feature / (close + EPS)
    feat["rv"] = rolling_population_std(ret1, config.rv_period)
    feat["ret_n"] = np.log(close / (close.shift(config.return_period) + EPS))
    feat["tick_imbalance"] = tick_imbalance
    feat["usdx_ret1"] = np.log(usdx_bid / (usdx_bid.shift(1) + EPS))
    feat["usdjpy_ret1"] = np.log(usdjpy_bid / (usdjpy_bid.shift(1) + EPS))

    feat["ret_2"] = np.log(close / (close.shift(config.feature_ret_2_period) + EPS))
    feat["ret_3"] = np.log(close / (close.shift(config.feature_ret_3_period) + EPS))
    feat["ret_6"] = np.log(close / (close.shift(config.feature_ret_6_period) + EPS))
    feat["ret_12"] = np.log(close / (close.shift(config.feature_ret_12_period) + EPS))
    feat["ret_20"] = np.log(close / (close.shift(config.feature_ret_20_period) + EPS))
    feat["open_rel_prev"] = np.log(open_price / (prev_close + EPS))
    feat["range_rel"] = (high - low) / (close + EPS)
    feat["body_rel"] = (close - open_price) / (close + EPS)
    feat["upper_wick_rel"] = (high - np.maximum(open_price, close)) / (close + EPS)
    feat["lower_wick_rel"] = (np.minimum(open_price, close) - low) / (close + EPS)
    feat["close_rel_sma_3"] = np.log(close / (sma_fast + EPS))
    feat["close_rel_sma_9"] = np.log(close / (sma_mid + EPS))
    feat["close_rel_sma_20"] = np.log(close / (sma_slow + EPS))
    feat["sma_3_9_gap"] = np.log(sma_fast / (sma_mid + EPS))
    feat["sma_5_20_gap"] = np.log(sma_trend_fast / (sma_slow + EPS))
    feat["sma_9_20_gap"] = np.log(sma_mid / (sma_slow + EPS))
    feat["sma_slope_9"] = np.log(sma_mid / (sma_mid.shift(config.feature_sma_slope_shift) + EPS))
    feat["sma_slope_20"] = np.log(sma_slow / (sma_slow.shift(config.feature_sma_slope_shift) + EPS))
    feat["rv_3"] = rv_3
    feat["rv_6"] = rv_6
    feat["rv_18"] = rv_18
    feat["donchian_pos_9"] = (close - low_fast) / (high_fast - low_fast + EPS)
    feat["donchian_width_9"] = (high_fast - low_fast) / (close + EPS)
    feat["donchian_pos_20"] = (close - low_slow) / (high_slow - low_slow + EPS)
    feat["donchian_width_20"] = (high_slow - low_slow) / (close + EPS)
    tick_count_sma_9 = tick_count.rolling(
        config.feature_tick_count_period,
        min_periods=config.feature_tick_count_period,
    ).mean()
    feat["tick_count_rel_9"] = tick_count / (tick_count_sma_9 + EPS) - 1.0
    feat["tick_count_z_9"] = rolling_zscore(tick_count, config.feature_tick_count_period)
    feat["tick_count_chg"] = np.log((tick_count + 1.0) / (tick_count.shift(1) + 1.0))
    feat["tick_imbalance_sma_5"] = tick_imbalance.rolling(
        config.feature_tick_imbalance_fast_period,
        min_periods=config.feature_tick_imbalance_fast_period,
    ).mean()
    feat["tick_imbalance_sma_9"] = tick_imbalance.rolling(
        config.feature_tick_imbalance_slow_period,
        min_periods=config.feature_tick_imbalance_slow_period,
    ).mean()
    feat["spread_z_9"] = rolling_zscore(spread_rel, config.feature_spread_z_period)
    feat["rsi_6"] = simple_rsi(close, config.feature_rsi_fast_period)
    feat["rsi_14"] = simple_rsi(close, config.feature_rsi_slow_period)
    feat["stoch_k_9"] = stoch_k_9
    feat["stoch_d_3"] = stoch_d_3
    feat["stoch_gap"] = stoch_k_9 - stoch_d_3
    feat["bollinger_pos_20"] = (close - sma_slow) / (2.0 * bollinger_std_20 + EPS)
    feat["bollinger_width_20"] = (4.0 * bollinger_std_20) / (sma_slow + EPS)
    feat["atr_ratio_20"] = np.log(
        atr_feature
        / (
            atr_feature.rolling(
                config.feature_atr_ratio_period,
                min_periods=config.feature_atr_ratio_period,
            ).mean()
            + EPS
        )
    )

    feat["spread_abs"] = spread_abs
    feat["bar_duration_ms"] = (time_close - time_open).astype(float)
    feat["rsi_9"] = simple_rsi(close, config.main_short_period)
    feat["rsi_18"] = simple_rsi(close, config.main_medium_period)
    feat["rsi_27"] = simple_rsi(close, config.main_long_period)
    feat["atr_9"] = simple_atr(high, low, close, config.main_short_period)
    feat["atr_18"] = simple_atr(high, low, close, config.main_medium_period)
    feat["atr_27"] = simple_atr(high, low, close, config.main_long_period)
    feat["macd_line"] = main_macd_line
    feat["macd_signal"] = main_macd_signal
    feat["macd_hist"] = main_macd_hist
    feat["ema_gap_9"] = ema(close, config.main_short_period) - close
    feat["ema_gap_18"] = ema(close, config.main_medium_period) - close
    feat["ema_gap_27"] = ema(close, config.main_long_period) - close
    feat["ema_gap_54"] = ema(close, config.main_xlong_period) - close
    feat["ema_gap_144"] = ema(close, config.main_xxlong_period) - close
    feat["cci_9"] = cci(high, low, close, config.main_short_period)
    feat["cci_18"] = cci(high, low, close, config.main_medium_period)
    feat["cci_27"] = cci(high, low, close, config.main_long_period)
    feat["willr_9"] = willr(high, low, close, config.main_short_period)
    feat["willr_18"] = willr(high, low, close, config.main_medium_period)
    feat["willr_27"] = willr(high, low, close, config.main_long_period)
    feat["mom_9"] = momentum(close, config.main_short_period)
    feat["mom_18"] = momentum(close, config.main_medium_period)
    feat["mom_27"] = momentum(close, config.main_long_period)
    feat["usdx_pct_change"] = usdx_bid.pct_change()
    feat["usdjpy_pct_change"] = usdjpy_bid.pct_change()
    feat["bollinger_width_9"] = bollinger_width(close, config.main_short_period)
    feat["bollinger_width_18"] = bollinger_width(close, config.main_medium_period)
    feat["bollinger_width_27"] = bollinger_width(close, config.main_long_period)
    feat["hour_sin"] = np.sin(2.0 * np.pi * time_index.dt.hour / 24.0)
    feat["hour_cos"] = np.cos(2.0 * np.pi * time_index.dt.hour / 24.0)
    feat["minute_sin"] = np.sin(2.0 * np.pi * time_index.dt.minute / 60.0)
    feat["minute_cos"] = np.cos(2.0 * np.pi * time_index.dt.minute / 60.0)
    feat["day_of_week_scaled"] = ((time_index.dt.dayofweek + 1) % 7) / 6.0
    return feat


def compute_features(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: FeatureEngineeringConfig,
) -> np.ndarray:
    feat = compute_feature_frame(df, feature_columns=feature_columns, config=config)
    missing_features = [name for name in feature_columns if name not in feat.columns]
    if missing_features:
        raise KeyError(f"Missing computed features: {missing_features}")
    return feat.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False)
