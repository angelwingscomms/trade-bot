# Implementation Guide: Wavelet Denoising, Multi-Timeframe Regime Features, and Intra-Bar Timing

## For AI Agent Use — Read Every Section Before Writing Code

This guide is written for an AI coding agent implementing three specific techniques into
an existing ML trading pipeline (`nn.py`, `data.mq5`, `live.mq5`) for XAUUSD on MetaTrader 5.

The pipeline currently:
- Exports tick data from MT5 using `data.mq5`
- Builds tick imbalance bars in Python (`nn.py`)
- Computes 18 features per symbol (GOLD, USDX, USDJPY) = 54 total
- Trains a Mamba/SSM model with triple-barrier labels
- Serves predictions in MT5 via `live.mq5`

---

## PART 1: UNDERSTANDING MULTI-TIMEFRAME REGIME FEATURES

### What the Problem Is

A 1-minute model making decisions bar-by-bar is like a trader who stares at a single
candlestick through a microscope and ignores the fact that the entire chart is in a
confirmed downtrend. Every bullish 1-minute bar the model sees will look like a buy
opportunity, when in reality the higher-timeframe context says "don't buy."

This is called a **whipsaw** — and it's the primary cause of small-edge strategies
bleeding out. The model might have a 52% directional accuracy, but if 40% of its "Buy"
signals happen against a dominant downtrend, those trades are systematically worse.

### What Multi-Timeframe Regime Does

It adds features to each 1-minute bar that answer the question:
**"What is the dominant market condition at a higher zoom level right now?"**

These features are not predictions. They are *descriptions of the current macro-state*
that the model uses to modulate its confidence about 1-minute signals.

Think of it in three layers:
```
15-minute chart → "What REGIME are we in?" (trending up, trending down, sideways)
 5-minute chart → "What MOMENTUM does this regime have?" (strong, weakening, reversing)
  1-minute chart → "What ACTION does this bar suggest?" (the model's actual job)
```

When all three align (15m uptrend + 5m bullish momentum + 1m bullish bar), the model
gets a high-confidence signal. When they conflict, the model learns to suppress or
reduce position size.

### How the Features Are Computed

#### Step 1: Resample 1-minute data to higher timeframes

Your 1-minute bars already exist. You don't need separate data feeds.
You resample the 1-minute OHLC to 5-minute and 15-minute bars by:
- Taking the FIRST open of each 5-minute window as the 5m Open
- Taking the MAX high of all 1-minute bars in the window as the 5m High
- Taking the MIN low as the 5m Low
- Taking the LAST close as the 5m Close

This is exactly what `.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})` does in pandas.

#### Step 2: Compute trend indicators on the resampled data

On the 5m and 15m bars, compute:
- EMA(50) and EMA(200): standard trend identification
- The sign of (EMA50 - EMA200): +1 = bullish regime, -1 = bearish regime, 0 = neutral

#### Step 3: Forward-fill back to 1-minute resolution

A 5-minute bar covers bars 1 through 5. The regime computed from bars 1-5 should be
applied to bar 5 ONLY (when the 5m bar closes). Bars 1-4 should carry the PREVIOUS
5m regime value.

This is critical for avoiding lookahead bias. The regime of the current 5m bar
is not known until that bar closes.

Implementation: `regime_series.reindex(df_1m.index, method='ffill')`

This reindexes the 5m regime series (one value per 5 minutes) onto the 1m index,
filling forward — so each 1m bar carries the most recently confirmed regime.

#### Step 4: What the model sees

After this process, each 1-minute bar has extra columns like:

```
regime_5m_GOLD    = +1.0   # 5m EMA50 > EMA200 = bullish
regime_15m_GOLD   = +1.0   # 15m EMA50 > EMA200 = bullish
above_sma200_GOLD = 1.0    # price > 200-bar 1m SMA = bullish
regime_5m_USDX    = -1.0   # dollar index 5m is bearish (good for gold)
```

The model learns, through training, that when `regime_5m_GOLD = +1` and
`regime_15m_GOLD = +1` and `regime_5m_USDX = -1`, buy signals on 1m gold have
higher expected returns.

### Why This Works Without Lookahead

The critical invariant:
**The regime assigned to a 1m bar can only use information from bars that CLOSED before this bar.**

- 5m regime assigned at bar t is computed from the 5m candle that LAST CLOSED, not the current one
- This is enforced by `ffill()` — a 5m bar's regime propagates forward to the next 5 bars
- The 5m candle currently forming is never included in the regime calculation

### What Features to Add (Per Symbol)

For each of your 3 symbols (GOLD, USDX, USDJPY), add:

| Feature Name | Description | Values |
|---|---|---|
| `regime_5m_{sym}` | EMA50 vs EMA200 on 5m bars | +1, -1 |
| `regime_15m_{sym}` | EMA50 vs EMA200 on 15m bars | +1, -1 |
| `above_sma200_1m_{sym}` | Price > 200-bar SMA on 1m | 0, 1 |
| `ema_slope_5m_{sym}` | Rate of change of 5m EMA50 | float (normalized) |

That's 4 features × 3 symbols = 12 new features.
Your feature count goes from ~28 → ~40.

---

## PART 2: IMPLEMENTATION INSTRUCTIONS — WAVELET DENOISING

### What It Does

Wavelet denoising separates a price series into:
- **Approximation coefficients**: the "true" underlying trend
- **Detail coefficients**: the noise at multiple frequency bands

By shrinking or zeroing small detail coefficients (which represent random noise)
and reconstructing the signal, you get a cleaner price series with the same sharp
edges (breakouts, reversals) but without tick-level jitter.

This improves downstream indicator quality because RSI, MACD, ATR, etc. are all
computed FROM the price series. Noisy prices → noisy indicators → noisy features.

### Prerequisites

```bash
pip install PyWavelets --break-system-packages
```

Verify:
```python
import pywt
print(pywt.wavelist(kind='discrete'))  # Should print list of wavelets including 'sym4'
```

### Step-by-Step Implementation

#### Step 2.1: Write the core denoising function

Place this in `nn.py` near the top, after imports:

```python
import pywt
import numpy as np

def wavelet_denoise_series(series: np.ndarray,
                            wavelet: str = 'sym4',
                            level: int = 3,
                            threshold_mode: str = 'soft') -> np.ndarray:
    """
    Denoise a 1D price series using wavelet thresholding.

    ARGUMENTS:
        series         : 1D numpy array of price values (e.g., close prices)
                         Must have NO NaN values. Fill or drop NaNs before calling.
        wavelet        : Wavelet basis function. Use 'sym4' for trading (minimal phase
                         distortion = timing of signals is accurate).
                         Alternatives: 'db4' (captures transitions), 'haar' (fastest,
                         best for sharp breakout detection).
        level          : Decomposition depth.
                         level=2 → removes finest noise (~1-2 bar jitter)
                         level=3 → removes ~1-4 bar noise (RECOMMENDED for 1m scalping)
                         level=4 → smoother but adds small lag
        threshold_mode : 'soft' → shrinks coefficients toward zero (smooth output)
                         'hard' → zeros small coefficients (preserves spikes better)
                         Use 'soft' for OHLC price columns.
                         Use 'hard' if you want to preserve candle wicks exactly.

    RETURNS:
        Denoised 1D numpy array, same length as input.

    CRITICAL NOTES FOR THE AGENT:
        1. The output length may differ by 1-2 elements due to wavelet padding.
           ALWAYS slice the output: return result[:len(series)]
        2. Do NOT apply this to already-derived features like RSI or log returns.
           ONLY apply to raw OHLC price columns.
        3. Apply BEFORE computing any indicators. Indicators computed after
           denoising will benefit from cleaner inputs.
        4. Apply the SAME denoising parameters to all symbols for consistency.
    """
    # Validate input
    if len(series) < 2 ** (level + 1):
        # Series too short for this decomposition level — return unchanged
        return series.copy()

    if np.any(np.isnan(series)):
        raise ValueError("wavelet_denoise_series: input contains NaN values. "
                         "Fill NaNs before calling this function.")

    # Decompose into approximation + detail coefficients
    # coeffs[0]  = approximation (lowest frequency = trend)
    # coeffs[1]  = finest detail (highest frequency = noise)
    # coeffs[-1] = coarsest detail
    coeffs = pywt.wavedec(series, wavelet, level=level)

    # Estimate noise sigma from finest detail level using MAD estimator
    # (Donoho & Johnstone 1994 — robust to outliers unlike std)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Universal threshold: penalizes coefficients smaller than this
    threshold = sigma * np.sqrt(2 * np.log(len(series)))

    # Apply threshold to ALL detail levels (leave approximation [0] untouched)
    denoised_coeffs = [coeffs[0]]  # keep trend unchanged
    for detail_level in coeffs[1:]:
        denoised_coeffs.append(
            pywt.threshold(detail_level, value=threshold, mode=threshold_mode)
        )

    # Reconstruct signal from modified coefficients
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    # Slice to original length (waverec can return N+1 elements)
    return reconstructed[:len(series)]
```

#### Step 2.2: Write the DataFrame-level wrapper

```python
def denoise_ohlc_dataframe(df: pd.DataFrame,
                            symbols: list,
                            wavelet: str = 'sym4',
                            level: int = 3) -> pd.DataFrame:
    """
    Apply wavelet denoising to all OHLC columns for the given symbols.

    ARGUMENTS:
        df      : DataFrame with columns named like 'open_GOLD', 'high_GOLD',
                  'low_GOLD', 'close_GOLD', 'open_USDX', etc.
        symbols : List of symbol names, e.g., ['GOLD', 'USDX', 'USDJPY']
        wavelet : See wavelet_denoise_series docs. Use 'sym4'.
        level   : See wavelet_denoise_series docs. Use 3.

    RETURNS:
        New DataFrame with denoised OHLC values.
        All other columns (volume, time, computed features) are unchanged.

    AGENT INSTRUCTION:
        Call this function ONCE, immediately after loading and validating the
        raw OHLC data, and BEFORE calling any indicator computation function.

        Correct order:
            df = load_raw_bars(...)           # Load OHLC
            df = validate_no_nans(df)         # Ensure clean data
            df = denoise_ohlc_dataframe(df)   # <-- Denoise HERE
            df = compute_all_indicators(df)   # MACD, ATR, RSI, etc.
            df = compute_regime_features(df)  # Multi-TF regime
            df = add_intrabar_timing(df)      # Timing features
            df = build_labels(df)             # Triple barrier
    """
    df = df.copy()
    price_prefixes = ['open', 'high', 'low', 'close']

    for symbol in symbols:
        for prefix in price_prefixes:
            col = f'{prefix}_{symbol}'
            if col not in df.columns:
                print(f"[WARN] Column '{col}' not found, skipping denoising for it.")
                continue

            original_values = df[col].values

            # Check for NaNs and handle
            if np.any(np.isnan(original_values)):
                print(f"[WARN] NaNs in {col}, forward-filling before denoising.")
                original_values = pd.Series(original_values).ffill().bfill().values

            df[col] = wavelet_denoise_series(original_values, wavelet=wavelet, level=level)

    return df
```

#### Step 2.3: Where to call it in your existing pipeline

Find the function in `nn.py` where you currently build your feature DataFrame.
It likely looks something like:

```python
# EXISTING CODE (approximate — find the actual function name)
def build_features(df_raw):
    df = df_raw.copy()
    df['log_ret_GOLD'] = np.log(df['close_GOLD']).diff()
    df['rsi_GOLD'] = compute_rsi(df['close_GOLD'], period=14)
    df['macd_GOLD'] = compute_macd(df['close_GOLD'])
    # ... etc
    return df
```

Modify it to denoise FIRST:

```python
# MODIFIED CODE
def build_features(df_raw, symbols=['GOLD', 'USDX', 'USDJPY']):
    df = df_raw.copy()

    # =========================================================
    # STEP 1: WAVELET DENOISING (must be first, before indicators)
    # =========================================================
    df = denoise_ohlc_dataframe(df, symbols=symbols, wavelet='sym4', level=3)

    # =========================================================
    # STEP 2: Your existing indicator computation (now on clean prices)
    # =========================================================
    df['log_ret_GOLD'] = np.log(df['close_GOLD']).diff()
    df['rsi_GOLD'] = compute_rsi(df['close_GOLD'], period=14)
    df['macd_GOLD'] = compute_macd(df['close_GOLD'])
    # ... rest of existing code unchanged

    return df
```

#### Step 2.4: Verify the denoising worked correctly

After implementing, run this verification:

```python
def verify_denoising(df_original, df_denoised, symbol='GOLD'):
    """
    Run after denoising to confirm it worked correctly.
    All assertions should pass silently. Any failure = bug in implementation.
    """
    col = f'close_{symbol}'

    orig = df_original[col].values
    deno = df_denoised[col].values

    # 1. Lengths must match
    assert len(orig) == len(deno), \
        f"Length mismatch: {len(orig)} vs {len(deno)}"

    # 2. Denoised should have lower first-difference std (less noise)
    orig_noise = np.std(np.diff(orig))
    deno_noise = np.std(np.diff(deno))
    assert deno_noise < orig_noise, \
        f"Denoising FAILED: denoised noise ({deno_noise:.4f}) >= original ({orig_noise:.4f})"

    # 3. Denoised values should stay within reasonable range of original
    max_deviation = np.max(np.abs(orig - deno))
    price_scale = np.mean(np.abs(orig))
    relative_deviation = max_deviation / price_scale
    assert relative_deviation < 0.05, \
        f"Denoising OVER-SMOOTHED: max deviation is {relative_deviation:.2%} of price scale"

    # 4. No NaNs introduced
    assert not np.any(np.isnan(deno)), "Denoising introduced NaN values"

    # 5. No infinities
    assert not np.any(np.isinf(deno)), "Denoising introduced Inf values"

    noise_reduction = (1 - deno_noise / orig_noise) * 100
    print(f"[OK] Denoising verified for {col}: {noise_reduction:.1f}% noise reduction")

    return True
```

#### Step 2.5: Denoising in live.mq5

The denoising must be reproduced in `live.mq5` exactly. Because MQL5 doesn't have
PyWavelets, you have two options:

**Option A (Recommended): Pre-denoise in the Python serving script**
When `live.mq5` calls your Python inference endpoint, pass the last N bars of OHLC,
apply `denoise_ohlc_dataframe()` to them in Python before feature computation,
then serve the prediction. The MQL5 side never needs to know about wavelets.

**Option B: Approximate denoising in MQL5**
Use a simple Exponential Moving Average as a denoising proxy.
This is less accurate but avoids needing PyWavelets in MQL5.
Use only if you cannot run Python in the serving loop.

```mql5
// MQL5 approximation of wavelet denoising (Option B only)
// Apply to Close array before computing indicators
double SmoothPrice(double &prices[], int idx, int smoothing_period=3) {
    // Simple EWM approximation — less accurate than true wavelet but causal
    double alpha = 2.0 / (smoothing_period + 1);
    double smoothed = prices[idx];
    for (int i = 1; i < MathMin(idx, smoothing_period * 3); i++) {
        smoothed = alpha * prices[idx - i] + (1 - alpha) * smoothed;
    }
    return smoothed;
}
```

**AGENT DECISION**: Use Option A. Your `live.mq5` already sends data to a Python
socket server. Simply apply denoising on the Python side before feature computation.
Add this to your inference endpoint (wherever you currently compute features for
live inference):

```python
# In your live inference Python script, wherever you receive bars from MT5:
def prepare_features_for_inference(raw_bars_dict, symbols):
    """
    raw_bars_dict: dict with keys like 'open_GOLD', 'close_GOLD', etc.
                   as received from MT5 socket.
    """
    df = pd.DataFrame(raw_bars_dict)

    # Apply same denoising as training
    df = denoise_ohlc_dataframe(df, symbols=symbols, wavelet='sym4', level=3)

    # Then compute all other features (same as training)
    df = compute_all_indicators(df)
    df = compute_regime_features(df, symbols=symbols)
    df = add_intrabar_timing(df, symbols=symbols)

    # Return the last row as the feature vector
    return df.iloc[-1][FEATURE_COLUMNS].values
```

---

## PART 3: IMPLEMENTATION INSTRUCTIONS — MULTI-TIMEFRAME REGIME FEATURES

### Prerequisites

Your DataFrame MUST have a DatetimeIndex for resampling to work.
If it currently uses integer indexing, add a datetime index first:

```python
# If your df has a 'time' or 'datetime' column:
df.index = pd.to_datetime(df['time_GOLD'])  # or whatever your timestamp column is

# Verify:
assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex for resampling"
assert df.index.is_monotonic_increasing, "Index must be sorted chronologically"
```

### Step 3.1: Write the regime computation function

```python
def compute_regime_features(df: pd.DataFrame,
                              symbols: list,
                              fast_ema_period: int = 50,
                              slow_ema_period: int = 200,
                              sma_period: int = 200) -> pd.DataFrame:
    """
    Compute multi-timeframe trend regime features for each symbol.

    Adds the following columns per symbol:
        regime_5m_{sym}       : +1 if 5m EMA50 > EMA200, else -1
        regime_15m_{sym}      : +1 if 15m EMA50 > EMA200, else -1
        above_sma200_1m_{sym} : 1.0 if 1m close > 200-bar SMA, else 0.0
        ema_slope_5m_{sym}    : rate of change of 5m EMA50 (normalized)

    ARGUMENTS:
        df              : DataFrame with DatetimeIndex and 'close_{sym}' columns.
                          Must contain at least slow_ema_period * 5 rows to give the
                          slow EMA enough warmup on the 5m timeframe.
        symbols         : List like ['GOLD', 'USDX', 'USDJPY']
        fast_ema_period : Period for fast EMA (50 recommended)
        slow_ema_period : Period for slow EMA (200 recommended)
        sma_period      : Period for 1m SMA baseline (200 recommended)

    RETURNS:
        df with new regime columns added. DatetimeIndex preserved.

    CRITICAL — LOOKAHEAD PREVENTION:
        This function uses method='ffill' when reindexing from 5m/15m back to 1m.
        This means: a 5m bar's regime is ONLY applied to 1m bars that come AFTER
        that 5m bar closes. The currently-forming 5m bar contributes NOTHING.
        This is correct behavior. Do NOT change method='ffill' to 'nearest' or 'bfill'.

    AGENT NOTES:
        - The EMA warmup period on 5m data = slow_ema_period bars = 200 bars × 5 minutes
          = 1000 minutes = ~16.7 hours of data. Your earliest regime values will be NaN
          for the first ~1000 1m bars. Fill these with 0.0 (neutral) using .fillna(0.0).
        - For XAUUSD, the USDX regime is particularly important: dollar strength
          (regime_5m_USDX = +1) is typically BEARISH for gold. The model will learn
          this negative correlation if the features are computed correctly.
    """
    df = df.copy()

    for symbol in symbols:
        close_col = f'close_{symbol}'
        if close_col not in df.columns:
            print(f"[WARN] {close_col} not found, skipping regime for {symbol}")
            continue

        close_series = df[close_col]

        # -----------------------------------------------------------------
        # 1m SMA200: straightforward rolling mean on 1m bars
        # -----------------------------------------------------------------
        sma200_1m = close_series.rolling(window=sma_period, min_periods=1).mean()
        df[f'above_sma200_1m_{symbol}'] = (close_series > sma200_1m).astype(float)

        # -----------------------------------------------------------------
        # 5-minute regime
        # -----------------------------------------------------------------
        # Resample to 5m bars
        # IMPORTANT: label='right', closed='right' means the 5m bar at 10:05
        # represents the period 10:00-10:05, labeled at its CLOSE time (10:05).
        # This is critical for correctness — the label must be at bar close.
        close_5m = close_series.resample('5min',
                                          label='right',
                                          closed='right').last()
        close_5m = close_5m.dropna()

        # Compute EMAs on 5m bars
        ema_fast_5m = close_5m.ewm(span=fast_ema_period,
                                    min_periods=fast_ema_period,
                                    adjust=False).mean()
        ema_slow_5m = close_5m.ewm(span=slow_ema_period,
                                    min_periods=slow_ema_period,
                                    adjust=False).mean()

        # Regime signal: +1 (bullish) or -1 (bearish)
        regime_5m = pd.Series(
            np.where(ema_fast_5m > ema_slow_5m, 1.0, -1.0),
            index=ema_fast_5m.index
        )

        # EMA slope: rate of change of fast EMA (normalized by price)
        # Tells the model whether the trend is accelerating or weakening
        ema_slope_5m = ema_fast_5m.pct_change(periods=3)  # 3-bar momentum

        # Forward-fill from 5m index to 1m index
        # The .reindex(df.index) maps 5m timestamps to the 1m DatetimeIndex.
        # Any 1m bar that falls between 5m bar closes gets the PREVIOUS 5m value (ffill).
        # This is the lookahead prevention mechanism.
        df[f'regime_5m_{symbol}'] = (
            regime_5m
            .reindex(df.index, method='ffill')
            .fillna(0.0)  # NaN during EMA warmup → neutral
        )
        df[f'ema_slope_5m_{symbol}'] = (
            ema_slope_5m
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )

        # -----------------------------------------------------------------
        # 15-minute regime (same logic, different resample)
        # -----------------------------------------------------------------
        close_15m = close_series.resample('15min',
                                           label='right',
                                           closed='right').last()
        close_15m = close_15m.dropna()

        ema_fast_15m = close_15m.ewm(span=fast_ema_period,
                                      min_periods=fast_ema_period,
                                      adjust=False).mean()
        ema_slow_15m = close_15m.ewm(span=slow_ema_period,
                                      min_periods=slow_ema_period,
                                      adjust=False).mean()

        regime_15m = pd.Series(
            np.where(ema_fast_15m > ema_slow_15m, 1.0, -1.0),
            index=ema_fast_15m.index
        )

        df[f'regime_15m_{symbol}'] = (
            regime_15m
            .reindex(df.index, method='ffill')
            .fillna(0.0)
        )

    return df
```

### Step 3.2: Verify no lookahead bias

This is the most important correctness check:

```python
def verify_no_lookahead_regime(df: pd.DataFrame, symbol: str = 'GOLD') -> bool:
    """
    Verify that regime features do not use future information.

    HOW IT WORKS:
        We take the regime at 1m bar index t, and check what the most recent
        CLOSED 5m bar was at that point in time. The regime should exactly match
        what you'd compute if you only had data up to t.

    This test simulates the live environment where only past bars are available.
    """
    close_5m = df[f'close_GOLD'].resample('5min', label='right', closed='right').last()
    ema_fast = close_5m.ewm(span=50, min_periods=50, adjust=False).mean()
    ema_slow = close_5m.ewm(span=200, min_periods=200, adjust=False).mean()
    regime_5m_true = pd.Series(
        np.where(ema_fast > ema_slow, 1.0, -1.0),
        index=ema_fast.index
    )

    # At any 1m bar time t, the regime in df should equal the LAST 5m regime
    # at or before t. Sample 100 random bars to check.
    sample_indices = np.random.choice(len(df), size=min(100, len(df)), replace=False)

    for idx in sample_indices:
        t = df.index[idx]
        regime_in_df = df[f'regime_5m_{symbol}'].iloc[idx]

        # Find the last 5m bar that closed at or before t
        past_5m = regime_5m_true[regime_5m_true.index <= t]
        if len(past_5m) == 0:
            continue  # In warmup period, skip

        expected_regime = past_5m.iloc[-1]

        if np.isnan(expected_regime):
            continue  # In EMA warmup, skip

        assert regime_in_df == expected_regime, (
            f"LOOKAHEAD DETECTED at {t}: "
            f"df has regime={regime_in_df}, expected={expected_regime}"
        )

    print(f"[OK] No lookahead bias detected in regime_5m_{symbol}")
    return True
```

### Step 3.3: Normalize regime features before model input

The Mamba model expects normalized inputs. Regime features are already bounded:
- `regime_5m_{sym}` ∈ {-1.0, +1.0} — already normalized, leave as-is
- `above_sma200_1m_{sym}` ∈ {0.0, 1.0} — already normalized, leave as-is
- `ema_slope_5m_{sym}` — this is a pct_change, typically in range [-0.02, +0.02]
  for gold. It CAN have outliers during extreme moves. Clip it:

```python
# After compute_regime_features(), add this:
for symbol in symbols:
    slope_col = f'ema_slope_5m_{symbol}'
    if slope_col in df.columns:
        # Clip at 3 standard deviations to handle outliers without distorting scale
        std = df[slope_col].std()
        df[slope_col] = df[slope_col].clip(-3 * std, 3 * std)

        # Then normalize to [-1, 1]
        max_abs = df[slope_col].abs().max()
        if max_abs > 0:
            df[slope_col] = df[slope_col] / max_abs
```

### Step 3.4: Handle the train/val split correctly

**CRITICAL AGENT NOTE:**
Your current pipeline splits data into train/val after feature computation.
Regime features computed on the FULL dataset before splitting are NOT a source of
lookahead, because the regime at any time t only uses prices from before t.

However: if you re-fit any scaler (StandardScaler, MinMaxScaler) on regime features,
fit it ONLY on training data and transform both train and val:

```python
# CORRECT
from sklearn.preprocessing import StandardScaler

# Identify regime feature columns (they're already bounded, but if scaling anyway)
regime_cols = [c for c in df.columns if 'regime_' in c or 'above_sma200' in c]

scaler = StandardScaler()
df_train[regime_cols] = scaler.fit_transform(df_train[regime_cols])  # fit on train only
df_val[regime_cols] = scaler.transform(df_val[regime_cols])          # transform val

# Save scaler for live inference
import joblib
joblib.dump(scaler, 'regime_scaler.pkl')
```

Note: Since `regime_5m` and `above_sma200` are already bounded to {-1, 0, +1},
you may choose to skip scaling for those specific columns and only scale `ema_slope_5m`.

### Step 3.5: Add regime features to the FEATURE_COLUMNS list

Find where FEATURE_COLUMNS is defined in your pipeline and add:

```python
REGIME_FEATURES = []
for sym in ['GOLD', 'USDX', 'USDJPY']:
    REGIME_FEATURES.extend([
        f'regime_5m_{sym}',
        f'regime_15m_{sym}',
        f'above_sma200_1m_{sym}',
        f'ema_slope_5m_{sym}',
    ])

# Add to existing feature list
FEATURE_COLUMNS = EXISTING_FEATURES + REGIME_FEATURES
```

### Step 3.6: Reproduce for live inference in Python serving script

The live inference script must compute regime features the same way:

```python
class LiveFeatureBuilder:
    """
    Maintains a rolling buffer of 1m bars for live regime computation.
    Must hold at least 200 * 15 = 3000 bars for EMA warmup.
    """
    WARMUP_BARS = 3000  # 200 slow EMA periods × 15 min timeframe

    def __init__(self, symbols):
        self.symbols = symbols
        self.buffer = pd.DataFrame()  # Rolling buffer of historical bars

    def update(self, new_bar: dict) -> pd.DataFrame:
        """
        Add a new completed 1m bar to the buffer and recompute features.

        new_bar: dict with keys like 'open_GOLD', 'close_GOLD', 'time', etc.
                 'time' must be a datetime or parseable datetime string.

        Returns the feature vector for the LAST bar (for inference).
        """
        new_row = pd.DataFrame([new_bar])
        new_row.index = pd.to_datetime([new_bar['time']])
        self.buffer = pd.concat([self.buffer, new_row]).iloc[-self.WARMUP_BARS:]

        if len(self.buffer) < 250:
            return None  # Not enough data yet

        # Apply denoising to the buffer
        df = denoise_ohlc_dataframe(self.buffer, symbols=self.symbols)

        # Compute all features
        df = compute_all_indicators(df)
        df = compute_regime_features(df, symbols=self.symbols)
        df = add_intrabar_timing(df, symbols=self.symbols)

        # Return last row's feature vector
        return df[FEATURE_COLUMNS].iloc[-1].values
```

---

## PART 4: IMPLEMENTATION INSTRUCTIONS — INTRA-BAR TIMING FEATURES

### What It Captures

Every 1-minute bar has 4 prices: Open, High, Low, Close.
But within that minute, the price movement followed a specific path.

Consider two bars with identical OHLC values:
- Open=2000, High=2010, Low=1995, Close=2005

**Bar A trajectory**: Price opens at 2000, immediately spikes to 2010 (at second 5),
then falls to 1995 (at second 50), then recovers slightly to close at 2005.
Interpretation: **BEARISH** — the bar opened with a bull spike that completely failed,
fell to new lows, and barely recovered. Sellers dominated.

**Bar B trajectory**: Price opens at 2000, dips to 1995 (at second 5), then climbs
to 2010 (at second 55), closes at 2005.
Interpretation: **BULLISH** — the bar opened weak, buyers absorbed all selling,
and the high was made in the final seconds. Momentum is upward.

Standard OHLC features see these as identical. Intra-bar timing sees them as opposites.

### What Data You Need

Your tick imbalance bars should have access to the timestamps when the high and low
of each bar were reached (since you're building bars from tick data).

If you currently DON'T store `time_high_{sym}` and `time_low_{sym}`, you need to
add this to your bar-building code. See Step 4.1.

If you DO have these columns, skip to Step 4.2.

### Step 4.1: Modify bar-building to capture high/low timing

In your bar-building function in `nn.py` (where you aggregate ticks into bars):

```python
def build_tick_imbalance_bars(ticks_df: pd.DataFrame, symbol: str,
                               threshold: float = 100) -> pd.DataFrame:
    """
    MODIFIED bar builder that also records when high/low were reached.

    ticks_df must have columns: 'time', 'bid', 'ask' (or 'price')
    threshold: tick imbalance threshold (your existing logic)

    Returns DataFrame with standard OHLC plus timing columns:
        time_open_{symbol}  : timestamp of bar open
        time_high_{symbol}  : timestamp when bar's high was first reached
        time_low_{symbol}   : timestamp when bar's low was first reached
        time_close_{symbol} : timestamp of bar close (= next bar's open time)
    """
    bars = []
    current_bar = {
        'open': None, 'high': -np.inf, 'low': np.inf, 'close': None,
        'time_open': None, 'time_high': None, 'time_low': None,
        'imbalance': 0, 'tick_count': 0
    }

    for _, tick in ticks_df.iterrows():
        price = (tick['bid'] + tick['ask']) / 2  # mid price
        t = tick['time']

        if current_bar['open'] is None:
            # Start new bar
            current_bar['open'] = price
            current_bar['time_open'] = t
            current_bar['high'] = price
            current_bar['low'] = price
            current_bar['time_high'] = t
            current_bar['time_low'] = t

        # Update high and its timestamp
        if price > current_bar['high']:
            current_bar['high'] = price
            current_bar['time_high'] = t  # record WHEN this high was made

        # Update low and its timestamp
        if price < current_bar['low']:
            current_bar['low'] = price
            current_bar['time_low'] = t  # record WHEN this low was made

        # Update imbalance (your existing tick imbalance logic here)
        # current_bar['imbalance'] += ... (your existing code)
        current_bar['tick_count'] += 1
        current_bar['close'] = price

        # Check if bar is complete (your existing threshold logic)
        if abs(current_bar['imbalance']) >= threshold:
            bars.append({
                f'open_{symbol}':       current_bar['open'],
                f'high_{symbol}':       current_bar['high'],
                f'low_{symbol}':        current_bar['low'],
                f'close_{symbol}':      current_bar['close'],
                f'time_open_{symbol}':  current_bar['time_open'],
                f'time_high_{symbol}':  current_bar['time_high'],
                f'time_low_{symbol}':   current_bar['time_low'],
                f'time_close_{symbol}': t,
            })
            # Reset bar
            current_bar = {
                'open': None, 'high': -np.inf, 'low': np.inf, 'close': None,
                'time_open': None, 'time_high': None, 'time_low': None,
                'imbalance': 0, 'tick_count': 0
            }

    return pd.DataFrame(bars)
```

### Step 4.2: Compute intra-bar timing features

```python
def add_intrabar_timing_features(df: pd.DataFrame, symbols: list) -> pd.DataFrame:
    """
    Add intra-bar timing features for each symbol.

    For each symbol, requires these columns to already exist in df:
        time_open_{sym}   : timestamp when bar opened
        time_high_{sym}   : timestamp when bar's high was reached
        time_low_{sym}    : timestamp when bar's low was reached
        time_close_{sym}  : timestamp when bar closed
        open_{sym}        : bar open price
        close_{sym}       : bar close price
        high_{sym}        : bar high price
        low_{sym}         : bar low price

    Adds these features per symbol:
        hl_ratio_{sym}      : fractional position of high within bar [0, 1]
                              0.0 = high made at bar open
                              1.0 = high made at bar close (bullish)
        lh_ratio_{sym}      : fractional position of low within bar [0, 1]
                              0.0 = low made at bar open
                              1.0 = low made at bar close (bearish)
        high_before_low_{sym}: 1.0 if high came before low (bearish sequence)
                               0.0 if low came before high (bullish sequence)
        timing_signal_{sym} : composite timing signal in [-1, +1]
                              +1 = very bullish (low early, high late)
                              -1 = very bearish (high early, low late)
        bar_range_norm_{sym}: normalized bar range = (high - low) / close
                              proxy for intra-bar volatility

    AGENT NOTES:
        - If timing columns are not available, the function falls back to
          price-based proxies (see fallback section below).
        - All timing features are normalized to [0, 1] or [-1, +1].
          Do NOT re-normalize them in the scaler — they are already bounded.
        - For tick imbalance bars (not fixed 1m bars), 'bar duration' varies.
          The fractional timing features handle this correctly because they're
          relative (e.g., high_time - open_time) / (close_time - open_time).
    """
    df = df.copy()

    for symbol in symbols:
        time_open_col  = f'time_open_{symbol}'
        time_high_col  = f'time_high_{symbol}'
        time_low_col   = f'time_low_{symbol}'
        time_close_col = f'time_close_{symbol}'

        open_col  = f'open_{symbol}'
        close_col = f'close_{symbol}'
        high_col  = f'high_{symbol}'
        low_col   = f'low_{symbol}'

        has_timing = all(c in df.columns for c in [
            time_open_col, time_high_col, time_low_col, time_close_col
        ])

        if has_timing:
            # Convert to datetime if not already
            t_open  = pd.to_datetime(df[time_open_col])
            t_high  = pd.to_datetime(df[time_high_col])
            t_low   = pd.to_datetime(df[time_low_col])
            t_close = pd.to_datetime(df[time_close_col])

            # Bar duration in seconds
            bar_duration = (t_close - t_open).dt.total_seconds()
            bar_duration = bar_duration.replace(0, np.nan)  # avoid div by zero

            # Fractional position of high within bar [0, 1]
            # 0 = high at open, 1 = high at close (bullish)
            df[f'hl_ratio_{symbol}'] = (
                (t_high - t_open).dt.total_seconds() / bar_duration
            ).clip(0, 1).fillna(0.5)

            # Fractional position of low within bar [0, 1]
            # 0 = low at open, 1 = low at close (bearish)
            df[f'lh_ratio_{symbol}'] = (
                (t_low - t_open).dt.total_seconds() / bar_duration
            ).clip(0, 1).fillna(0.5)

            # Did the high come before the low?
            # 1.0 = high was made first → price spiked then fell → BEARISH sequence
            # 0.0 = low was made first → price dipped then rose → BULLISH sequence
            df[f'high_before_low_{symbol}'] = (
                (t_high < t_low).astype(float)
            )

            # Composite timing signal [-1, +1]
            # Logic:
            #   high late (hl_ratio → 1) = bullish → adds to signal
            #   low early (lh_ratio → 0) = bullish → adds to signal
            #   high before low = bearish → subtracts from signal
            timing_raw = (
                df[f'hl_ratio_{symbol}']           # [0, 1] high late = good
                - (1 - df[f'lh_ratio_{symbol}'])   # [0, 1] low early = good
                - 0.5 * df[f'high_before_low_{symbol}']  # penalty for high-before-low
            )
            # Normalize to [-1, 1]
            df[f'timing_signal_{symbol}'] = timing_raw.clip(-1, 1)

        else:
            # FALLBACK: No timing columns available
            # Use price structure as a proxy for timing
            print(f"[WARN] Timing columns not found for {symbol}. Using price proxy.")

            if all(c in df.columns for c in [open_col, close_col, high_col, low_col]):
                # Close position within bar range [0, 1]
                # 1.0 = closed at high (bullish), 0.0 = closed at low (bearish)
                bar_range = df[high_col] - df[low_col]
                bar_range = bar_range.replace(0, np.nan)

                close_position = (df[close_col] - df[low_col]) / bar_range
                df[f'hl_ratio_{symbol}']    = close_position.fillna(0.5)
                df[f'lh_ratio_{symbol}']    = 1 - close_position.fillna(0.5)
                df[f'high_before_low_{symbol}'] = (
                    (df[close_col] < df[open_col]).astype(float)
                )
                df[f'timing_signal_{symbol}'] = (
                    (2 * close_position - 1).fillna(0.0).clip(-1, 1)
                )
            else:
                # Cannot compute — fill with neutral
                for col in [f'hl_ratio_{symbol}', f'lh_ratio_{symbol}',
                            f'high_before_low_{symbol}', f'timing_signal_{symbol}']:
                    df[col] = 0.0

        # Bar range normalization (works with or without timing columns)
        if all(c in df.columns for c in [high_col, low_col, close_col]):
            df[f'bar_range_norm_{symbol}'] = (
                (df[high_col] - df[low_col]) / (df[close_col] + 1e-10)
            )
        else:
            df[f'bar_range_norm_{symbol}'] = 0.0

    return df
```

### Step 4.3: Verify timing feature correctness

```python
def verify_timing_features(df: pd.DataFrame, symbol: str = 'GOLD'):
    """
    Run manual inspection on timing features.
    Checks a few obvious cases to confirm the logic is correct.
    """
    timing_col = f'timing_signal_{symbol}'
    hl_col     = f'hl_ratio_{symbol}'
    lh_col     = f'lh_ratio_{symbol}'

    # Find strongly bullish bars: closed at top, low came early
    bullish_mask = (df[hl_col] > 0.8) & (df[lh_col] < 0.2)
    bullish_timing = df.loc[bullish_mask, timing_col]

    # All should be positive (bullish signal)
    assert (bullish_timing > 0).all(), \
        f"FAIL: Bullish bars have negative timing signal. Sample: {bullish_timing.head()}"

    # Find strongly bearish bars: high came early, closed at bottom
    bearish_mask = (df[hl_col] < 0.2) & (df[lh_col] > 0.8)
    bearish_timing = df.loc[bearish_mask, timing_col]

    # All should be negative (bearish signal)
    assert (bearish_timing < 0).all(), \
        f"FAIL: Bearish bars have positive timing signal. Sample: {bearish_timing.head()}"

    # Timing signal should always be in [-1, 1]
    assert df[timing_col].between(-1, 1).all(), \
        f"FAIL: timing_signal_{symbol} has values outside [-1, 1]"

    print(f"[OK] Timing features verified for {symbol}")
    print(f"     Bullish bar timing: {bullish_timing.mean():.3f} (should be > 0)")
    print(f"     Bearish bar timing: {bearish_timing.mean():.3f} (should be < 0)")
```

### Step 4.4: Add timing features to feature list

```python
TIMING_FEATURES = []
for sym in ['GOLD', 'USDX', 'USDJPY']:
    TIMING_FEATURES.extend([
        f'hl_ratio_{sym}',
        f'lh_ratio_{sym}',
        f'high_before_low_{sym}',
        f'timing_signal_{sym}',
        f'bar_range_norm_{sym}',
    ])

FEATURE_COLUMNS = EXISTING_FEATURES + REGIME_FEATURES + TIMING_FEATURES
```

---

## PART 5: COMPLETE PIPELINE INTEGRATION

### Final call order in nn.py

```python
def full_preprocessing_pipeline(df_raw: pd.DataFrame,
                                  symbols: list = ['GOLD', 'USDX', 'USDJPY'],
                                  is_training: bool = True,
                                  scaler=None) -> tuple:
    """
    Complete preprocessing pipeline.

    CALL ORDER IS MANDATORY. Changing the order will cause bugs.

    Returns:
        (df_features, scaler) where scaler is fitted on training data only.
    """
    # 1. Validate index
    assert isinstance(df_raw.index, pd.DatetimeIndex), \
        "df must have DatetimeIndex. Set: df.index = pd.to_datetime(df['time'])"
    assert df_raw.index.is_monotonic_increasing, \
        "Index must be sorted. Call: df = df.sort_index()"

    # 2. Drop NaNs in raw OHLC (before denoising)
    ohlc_cols = [f'{p}_{s}' for s in symbols for p in ['open','high','low','close']
                 if f'{p}_{s}' in df_raw.columns]
    df = df_raw.dropna(subset=ohlc_cols).copy()

    # 3. Wavelet denoising (FIRST — before any indicators)
    df = denoise_ohlc_dataframe(df, symbols=symbols, wavelet='sym4', level=3)

    # 4. Your existing indicators (MACD, ATR, RSI, etc.) — now on denoised prices
    df = compute_existing_indicators(df, symbols=symbols)

    # 5. NEW: Multi-timeframe regime features
    df = compute_regime_features(df, symbols=symbols)

    # 6. NEW: Intra-bar timing features
    df = add_intrabar_timing_features(df, symbols=symbols)

    # 7. Drop warmup rows (first 250 bars have NaN regime from EMA warmup)
    warmup_rows = 250
    df = df.iloc[warmup_rows:].copy()

    # 8. Drop any remaining NaN rows from indicator computation
    df = df.dropna(subset=FEATURE_COLUMNS)

    # 9. Scale features (fit scaler on train only, transform val)
    feature_matrix = df[FEATURE_COLUMNS].values

    if is_training:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        import joblib
        joblib.dump(scaler, 'feature_scaler.pkl')
    else:
        assert scaler is not None, "Must provide fitted scaler for validation/inference"
        feature_matrix = scaler.transform(feature_matrix)

    df[FEATURE_COLUMNS] = feature_matrix

    return df, scaler
```

### Checklist for the agent before declaring implementation complete

```
[ ] pip install PyWavelets — installed and importable
[ ] wavelet_denoise_series — implemented and verify_denoising() passes
[ ] denoise_ohlc_dataframe — called BEFORE compute_existing_indicators()
[ ] DatetimeIndex — df.index is DatetimeIndex, sorted ascending
[ ] compute_regime_features — implemented with label='right', closed='right'
[ ] verify_no_lookahead_regime — runs without assertion errors
[ ] ema_slope_5m — clipped and normalized to [-1, 1]
[ ] Regime scaler — fitted on train only, saved with joblib
[ ] Time columns — time_open_{sym}, time_high_{sym}, time_low_{sym} exist in bars
[ ] add_intrabar_timing_features — implemented with fallback for missing timing cols
[ ] verify_timing_features — bullish bars have positive signal, bearish bars negative
[ ] FEATURE_COLUMNS — updated to include all new features
[ ] WARMUP_BARS — first 250 rows dropped after feature computation
[ ] Live inference script — denoise + regime + timing computed in the same order
[ ] Regime features in live — LiveFeatureBuilder maintains 3000-bar rolling buffer
[ ] No data leakage — scaler fit on train, transform on val and live
[ ] Total feature count — verify expected count with: len(FEATURE_COLUMNS)
```

### Expected new feature count

| Group | Features per Symbol | Symbols | Total |
|---|---|---|---|
| Existing features | ~9 | 3 | ~27 |
| Wavelet (no new cols — modifies existing) | 0 | 3 | 0 |
| Regime features | 4 | 3 | 12 |
| Timing features | 5 | 3 | 15 |
| **Total** | | | **~54** |

You end up back near 54 features, but with significantly higher information density
per feature — the regime and timing features carry directional signal that raw OHLC
ratios cannot capture.
