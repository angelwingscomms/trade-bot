from __future__ import annotations

import numpy as np
import pandas as pd

from common.past_dir_features import parse_past_dir_spec

from tradebot.training.wavelet_regime_timing import (
    denoise_ohlc_dataframe,
    compute_regime_features,
    add_intrabar_timing_features,
    add_usdx_regime_features,
    add_usdjpy_regime_features,
    WAVELET_REGIME_TIMING_FEATURES,
)

from .shared import *  # noqa: F401,F403


def _apply_wavelet_regime_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply wavelet denoising, regime features, and timing features to bars."""
    df = df.copy()

    time_open = df["time_open"].astype(np.int64)
    df["_time_index"] = pd.to_datetime(time_open, unit="ms", utc=True)
    df = df.set_index("_time_index")

    ohlc_cols = ['open', 'high', 'low', 'close']
    available_ohlc = [c for c in ohlc_cols if c in df.columns]
    if len(available_ohlc) == len(ohlc_cols):
        df = denoise_ohlc_dataframe(df, price_columns=ohlc_cols)

    df = compute_regime_features(df, close_col='close')
    df = add_usdx_regime_features(df, usdx_col='usdx_bid')
    df = add_usdjpy_regime_features(df, usdjpy_col='usdjpy_bid')
    df = add_intrabar_timing_features(df)

    df = df.reset_index(drop=True)

    return df


def compute_feature_frame(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: FeatureEngineeringConfig,
) -> pd.DataFrame:
    needs_wavelet_regime_timing = any(
        feat in WAVELET_REGIME_TIMING_FEATURES for feat in feature_columns
    )
    if needs_wavelet_regime_timing:
        df = _apply_wavelet_regime_timing_features(df)

    close = df["close"].astype(float)
    open_price = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tick_count = df["tick_count"].astype(float)
    tick_imbalance = df["tick_imbalance"].astype(float)
    prev_close = close.shift(1)
    ret1 = np.log(close / (prev_close + EPS))

    close_z_250 = rolling_zscore(close, config.feature_normalize_period)
    ret_z_250 = rolling_zscore(ret1, config.feature_normalize_period)

    atr_feature = wilder_atr(high, low, close, period=config.feature_atr_period)
    spread_rel = df["spread"] / (close + EPS)
    spread_abs = df.get("spread_mean", df["spread"]).astype(float)

    sma_fast = close.rolling(
        config.feature_sma_fast_period, min_periods=config.feature_sma_fast_period
    ).mean()
    sma_trend_fast = close.rolling(
        config.feature_sma_trend_fast_period,
        min_periods=config.feature_sma_trend_fast_period,
    ).mean()
    sma_mid = close.rolling(
        config.feature_sma_mid_period, min_periods=config.feature_sma_mid_period
    ).mean()
    sma_slow = close.rolling(
        config.feature_sma_slow_period, min_periods=config.feature_sma_slow_period
    ).mean()
    rv_3 = rolling_population_std(ret1, config.feature_ret_3_period)
    rv_6 = rolling_population_std(ret1, config.feature_ret_6_period)
    rv_18 = rolling_population_std(ret1, config.feature_rv_long_period)
    high_fast = high.rolling(
        config.feature_donchian_fast_period,
        min_periods=config.feature_donchian_fast_period,
    ).max()
    low_fast = low.rolling(
        config.feature_donchian_fast_period,
        min_periods=config.feature_donchian_fast_period,
    ).min()
    high_slow = high.rolling(
        config.feature_donchian_slow_period,
        min_periods=config.feature_donchian_slow_period,
    ).max()
    low_slow = low.rolling(
        config.feature_donchian_slow_period,
        min_periods=config.feature_donchian_slow_period,
    ).min()
    high_stoch = high.rolling(
        config.feature_stoch_period, min_periods=config.feature_stoch_period
    ).max()
    low_stoch = low.rolling(
        config.feature_stoch_period, min_periods=config.feature_stoch_period
    ).min()
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
    feat["sma_slope_9"] = np.log(
        sma_mid / (sma_mid.shift(config.feature_sma_slope_shift) + EPS)
    )
    feat["sma_slope_20"] = np.log(
        sma_slow / (sma_slow.shift(config.feature_sma_slope_shift) + EPS)
    )
    feat["rv_3"] = rv_3
    feat["rv_6"] = rv_6
    feat["rv_18"] = rv_18
    feat["close_z_250"] = close_z_250
    feat["ret_z_250"] = ret_z_250
    feat["donchian_pos_9"] = (close - low_fast) / (high_fast - low_fast + EPS)
    feat["donchian_width_9"] = (high_fast - low_fast) / (close + EPS)
    feat["donchian_pos_20"] = (close - low_slow) / (high_slow - low_slow + EPS)
    feat["donchian_width_20"] = (high_slow - low_slow) / (close + EPS)
    tick_count_sma_9 = tick_count.rolling(
        config.feature_tick_count_period,
        min_periods=config.feature_tick_count_period,
    ).mean()
    feat["tick_count_rel_9"] = tick_count / (tick_count_sma_9 + EPS) - 1.0
    feat["tick_count_z_9"] = rolling_zscore(
        tick_count, config.feature_tick_count_period
    )
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

    # Dynamic PAST_DIR_<N>_S / PAST_DIR_<N>_T features.
    # For each requested column whose name matches the pattern, compute:
    #   tanh(log(close_now / close_n_bars_ago))
    # tanh maps any log-return into (-1, 1), symmetric around 0:
    #   near -1 -> strong drop,  0 -> flat,  near +1 -> strong rise.
    _bar_seconds: int = int(getattr(config, "primary_bar_seconds", 0))
    for _col in feature_columns:
        _spec = parse_past_dir_spec(_col)
        if _spec is None:
            continue
        _n, _unit = _spec
        if _unit == "T":
            # _T: lookback is in bars directly
            _shift = _n
        else:
            # _S: convert wall-clock seconds to bar count, rounding up
            if _bar_seconds > 0:
                import math as _math

                _shift = _math.ceil(_n / _bar_seconds)
            else:
                _shift = _n  # safe fallback when bar duration is unknown
        _log_ret = np.log(close / (close.shift(_shift) + EPS))
        feat[_col] = np.tanh(_log_ret)

    return feat
