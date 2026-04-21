from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

WAVELET_DENOISE_WAVELET = 'sym4'
WAVELET_DENOISE_LEVEL = 3
WAVELET_DENOISE_THRESHOLD_MODE = 'soft'

REGIME_FAST_EMA_PERIOD = 50
REGIME_SLOW_EMA_PERIOD = 200
REGIME_SMA_PERIOD = 200


def wavelet_denoise_series(series: np.ndarray,
                           wavelet: str = 'sym4',
                           level: int = 3,
                           threshold_mode: str = 'soft') -> np.ndarray:
    if len(series) < 2 ** (level + 1):
        return series.copy()

    if np.any(np.isnan(series)):
        raise ValueError("wavelet_denoise_series: input contains NaN values. "
                         "Fill NaNs before calling this function.")

    coeffs = pywt.wavedec(series, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    threshold = sigma * np.sqrt(2 * np.log(len(series)))

    denoised_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        denoised_coeffs.append(
            pywt.threshold(detail_level, value=threshold, mode=threshold_mode)
        )

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    return reconstructed[:len(series)]


def denoise_ohlc_dataframe(df: pd.DataFrame,
                           price_columns: list = None) -> pd.DataFrame:
    df = df.copy()

    if price_columns is None:
        price_columns = ['open', 'high', 'low', 'close']

    available_cols = [c for c in price_columns if c in df.columns]

    for col in available_cols:
        original_values = df[col].values

        if np.any(np.isnan(original_values)):
            original_values = pd.Series(original_values).ffill().bfill().values

        df[col] = wavelet_denoise_series(
            original_values,
            wavelet=WAVELET_DENOISE_WAVELET,
            level=WAVELET_DENOISE_LEVEL,
            threshold_mode=WAVELET_DENOISE_THRESHOLD_MODE
        )

    return df


def compute_regime_features(df: pd.DataFrame,
                            close_col: str = 'close',
                            fast_ema_period: int = REGIME_FAST_EMA_PERIOD,
                            slow_ema_period: int = REGIME_SLOW_EMA_PERIOD,
                            sma_period: int = REGIME_SMA_PERIOD) -> pd.DataFrame:
    df = df.copy()

    if close_col not in df.columns:
        raise ValueError(f"Column {close_col} not found in DataFrame")

    close_series = df[close_col]

    sma200_1m = close_series.rolling(window=sma_period, min_periods=1).mean()
    df['above_sma200_1m'] = (close_series > sma200_1m).astype(float)

    close_5m = close_series.resample('5min', label='right', closed='right').last()
    close_5m = close_5m.dropna()

    ema_fast_5m = close_5m.ewm(span=fast_ema_period, min_periods=fast_ema_period, adjust=False).mean()
    ema_slow_5m = close_5m.ewm(span=slow_ema_period, min_periods=slow_ema_period, adjust=False).mean()

    regime_5m = pd.Series(
        np.where(ema_fast_5m > ema_slow_5m, 1.0, -1.0),
        index=ema_fast_5m.index
    )

    ema_slope_5m = ema_fast_5m.pct_change(periods=3)

    if isinstance(df.index, pd.DatetimeIndex):
        df['regime_5m'] = (
            regime_5m
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
        df['ema_slope_5m'] = (
            ema_slope_5m
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
    else:
        df['regime_5m'] = 0.0
        df['ema_slope_5m'] = 0.0

    close_15m = close_series.resample('15min', label='right', closed='right').last()
    close_15m = close_15m.dropna()

    ema_fast_15m = close_15m.ewm(span=fast_ema_period, min_periods=fast_ema_period, adjust=False).mean()
    ema_slow_15m = close_15m.ewm(span=slow_ema_period, min_periods=slow_ema_period, adjust=False).mean()

    regime_15m = pd.Series(
        np.where(ema_fast_15m > ema_slow_15m, 1.0, -1.0),
        index=ema_fast_15m.index
    )

    if isinstance(df.index, pd.DatetimeIndex):
        df['regime_15m'] = (
            regime_15m
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
    else:
        df['regime_15m'] = 0.0

    slope_col = 'ema_slope_5m'
    if slope_col in df.columns:
        std = df[slope_col].std()
        if std > 0:
            df[slope_col] = df[slope_col].clip(-3 * std, 3 * std)
            max_abs = df[slope_col].abs().max()
            if max_abs > 0:
                df[slope_col] = df[slope_col] / max_abs

    return df


def add_intrabar_timing_features(df: pd.DataFrame,
                                open_col: str = 'open',
                                close_col: str = 'close',
                                high_col: str = 'high',
                                low_col: str = 'low') -> pd.DataFrame:
    df = df.copy()

    required_cols = [open_col, close_col, high_col, low_col]
    if not all(c in df.columns for c in required_cols):
        for col in ['hl_ratio', 'lh_ratio', 'high_before_low', 'timing_signal', 'bar_range_norm']:
            df[col] = 0.0
        return df

    open_p = df[open_col]
    close_p = df[close_col]
    high_p = df[high_col]
    low_p = df[low_col]

    bar_range = high_p - low_p
    bar_range = bar_range.replace(0, np.nan)

    close_position = (close_p - low_p) / bar_range
    df['hl_ratio'] = close_position.fillna(0.5)
    df['lh_ratio'] = 1 - close_position.fillna(0.5)
    df['high_before_low'] = (close_p < open_p).astype(float)
    df['timing_signal'] = (2 * close_position - 1).fillna(0.0).clip(-1, 1)

    df['bar_range_norm'] = (high_p - low_p) / (close_p + 1e-10)

    return df


def add_usdx_regime_features(df: pd.DataFrame,
                            usdx_col: str = 'usdx_bid') -> pd.DataFrame:
    df = df.copy()

    if usdx_col not in df.columns or df[usdx_col].notna().sum() == 0:
        df['regime_5m_usdx'] = 0.0
        df['regime_15m_usdx'] = 0.0
        return df

    usdx_series = df[usdx_col].ffill().bfill()

    close_5m = usdx_series.resample('5min', label='right', closed='right').last()
    close_5m = close_5m.dropna()

    ema_fast_5m = close_5m.ewm(span=REGIME_FAST_EMA_PERIOD, min_periods=REGIME_FAST_EMA_PERIOD, adjust=False).mean()
    ema_slow_5m = close_5m.ewm(span=REGIME_SLOW_EMA_PERIOD, min_periods=REGIME_SLOW_EMA_PERIOD, adjust=False).mean()

    regime_5m_usdx = pd.Series(
        np.where(ema_fast_5m > ema_slow_5m, 1.0, -1.0),
        index=ema_fast_5m.index
    )

    close_15m = usdx_series.resample('15min', label='right', closed='right').last()
    close_15m = close_15m.dropna()

    ema_fast_15m = close_15m.ewm(span=REGIME_FAST_EMA_PERIOD, min_periods=REGIME_FAST_EMA_PERIOD, adjust=False).mean()
    ema_slow_15m = close_15m.ewm(span=REGIME_SLOW_EMA_PERIOD, min_periods=REGIME_SLOW_EMA_PERIOD, adjust=False).mean()

    regime_15m_usdx = pd.Series(
        np.where(ema_fast_15m > ema_slow_15m, 1.0, -1.0),
        index=ema_fast_15m.index
    )

    if isinstance(df.index, pd.DatetimeIndex):
        df['regime_5m_usdx'] = (
            regime_5m_usdx
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
        df['regime_15m_usdx'] = (
            regime_15m_usdx
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
    else:
        df['regime_5m_usdx'] = 0.0
        df['regime_15m_usdx'] = 0.0

    return df


def add_usdjpy_regime_features(df: pd.DataFrame,
                               usdjpy_col: str = 'usdjpy_bid') -> pd.DataFrame:
    df = df.copy()

    if usdjpy_col not in df.columns or df[usdjpy_col].notna().sum() == 0:
        df['regime_5m_usdjpy'] = 0.0
        df['regime_15m_usdjpy'] = 0.0
        return df

    usdjpy_series = df[usdjpy_col].ffill().bfill()

    close_5m = usdjpy_series.resample('5min', label='right', closed='right').last()
    close_5m = close_5m.dropna()

    ema_fast_5m = close_5m.ewm(span=REGIME_FAST_EMA_PERIOD, min_periods=REGIME_FAST_EMA_PERIOD, adjust=False).mean()
    ema_slow_5m = close_5m.ewm(span=REGIME_SLOW_EMA_PERIOD, min_periods=REGIME_SLOW_EMA_PERIOD, adjust=False).mean()

    regime_5m_usdjpy = pd.Series(
        np.where(ema_fast_5m > ema_slow_5m, 1.0, -1.0),
        index=ema_fast_5m.index
    )

    close_15m = usdjpy_series.resample('15min', label='right', closed='right').last()
    close_15m = close_15m.dropna()

    ema_fast_15m = close_15m.ewm(span=REGIME_FAST_EMA_PERIOD, min_periods=REGIME_FAST_EMA_PERIOD, adjust=False).mean()
    ema_slow_15m = close_15m.ewm(span=REGIME_SLOW_EMA_PERIOD, min_periods=REGIME_SLOW_EMA_PERIOD, adjust=False).mean()

    regime_15m_usdjpy = pd.Series(
        np.where(ema_fast_15m > ema_slow_15m, 1.0, -1.0),
        index=ema_fast_15m.index
    )

    if isinstance(df.index, pd.DatetimeIndex):
        df['regime_5m_usdjpy'] = (
            regime_5m_usdjpy
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
        df['regime_15m_usdjpy'] = (
            regime_15m_usdjpy
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )
    else:
        df['regime_5m_usdjpy'] = 0.0
        df['regime_15m_usdjpy'] = 0.0

    return df


WAVELET_REGIME_TIMING_FEATURES = (
    'regime_5m',
    'regime_15m',
    'above_sma200_1m',
    'ema_slope_5m',
    'regime_5m_usdx',
    'regime_15m_usdx',
    'regime_5m_usdjpy',
    'regime_15m_usdjpy',
    'hl_ratio',
    'lh_ratio',
    'high_before_low',
    'timing_signal',
    'bar_range_norm',
)


def apply_wavelet_regime_timing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'time_open' in df.columns and 'time_msc' in df.columns:
        if isinstance(df['time_open'].iloc[0], (int, np.integer)):
            df['time'] = pd.to_datetime(df['time_open'], unit='ms')
        else:
            df['time'] = pd.to_datetime(df['time_open'])
        df.index = df['time']

    df = denoise_ohlc_dataframe(df)

    df = compute_regime_features(df)

    df = add_usdx_regime_features(df)

    df = add_usdjpy_regime_features(df)

    df = add_intrabar_timing_features(df)

    return df


def verify_denoising(df_original: pd.DataFrame, df_denoised: pd.DataFrame, col: str = 'close') -> bool:
    orig = df_original[col].values
    deno = df_denoised[col].values

    assert len(orig) == len(deno), f"Length mismatch: {len(orig)} vs {len(deno)}"

    orig_noise = np.std(np.diff(orig))
    deno_noise = np.std(np.diff(deno))

    if deno_noise >= orig_noise:
        print(f"[WARN] Denoising may not be reducing noise: denoised={deno_noise:.4f} >= original={orig_noise:.4f}")

    max_deviation = np.max(np.abs(orig - deno))
    price_scale = np.mean(np.abs(orig))
    relative_deviation = max_deviation / price_scale
    if relative_deviation > 0.05:
        print(f"[WARN] Denoising over-smoothed: max deviation is {relative_deviation:.2%} of price scale")

    assert not np.any(np.isnan(deno)), "Denoising introduced NaN values"
    assert not np.any(np.isinf(deno)), "Denoising introduced Inf values"

    noise_reduction = (1 - deno_noise / orig_noise) * 100
    print(f"[OK] Denoising verified for {col}: {noise_reduction:.1f}% noise reduction")

    return True
