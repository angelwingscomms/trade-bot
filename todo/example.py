"""
scalping_additions.py
=====================
Drop-in additions for your existing nn.py pipeline.
Implements the key techniques from the PDF:
  1. Wavelet denoising
  2. Intra-bar timing features (from tick bars)
  3. Short RSI + Bollinger width
  4. Multi-timeframe regime features
  5. EmbTCN-Transformer (optional Mamba replacement / comparison model)

Each section is independent — adopt what you want.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt  # pip install PyWavelets


# =============================================================================
# 1. WAVELET DENOISING
# =============================================================================
# Why: 1-minute OHLC is mostly noise. Wavelets separate real price structure
# (breakouts, trends) from random jitter — without the lag of a moving average.
# The MODWT variant is preferred because it's translation-invariant (no aliasing).
#
# In your pipeline: apply to each OHLC column BEFORE computing indicators.
# This gives your RSI, MACD, ATR cleaner inputs to work from.

def wavelet_denoise(series: np.ndarray,
                    wavelet: str = 'sym4',
                    level: int = 3,
                    threshold_mode: str = 'soft') -> np.ndarray:
    """
    Denoise a 1D price series using wavelet soft-thresholding.

    Args:
        series:         Raw price array (e.g. Close prices).
        wavelet:        Wavelet basis. 'sym4' has minimal phase distortion —
                        critical when signal TIMING matters (scalping).
                        'haar' is faster but rougher.
                        'db4' captures complex transitions.
        level:          Decomposition depth. 3 = strips ~1-3 bar noise.
                        4-5 for smoother but more lag.
        threshold_mode: 'soft' shrinks coefficients toward zero (smoother).
                        'hard' zeroes them outright (preserves spikes better).

    Returns:
        Denoised price array, same shape as input.
    """
    # Decompose into approximation + detail coefficients
    coeffs = pywt.wavedec(series, wavelet, level=level)

    # Universal threshold: sigma * sqrt(2 * log(N))
    # sigma estimated from finest detail level (Donoho & Johnstone 1994)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(series)))

    # Threshold detail coefficients (leave approximation [0] untouched)
    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=threshold_mode)
        for c in coeffs[1:]
    ]

    # Reconstruct
    return pywt.waverec(coeffs_thresh, wavelet)[:len(series)]


def denoise_ohlc(df: pd.DataFrame,
                 cols: list = None,
                 wavelet: str = 'sym4',
                 level: int = 3) -> pd.DataFrame:
    """
    Apply wavelet denoising to OHLC columns in-place.
    Call this BEFORE computing any indicators (RSI, MACD, ATR, etc.)

    Example in your existing pipeline (nn.py, inside build_features()):
        df = denoise_ohlc(df, cols=['close_GOLD', 'close_USDX', 'close_USDJPY'])
    """
    if cols is None:
        cols = [c for c in df.columns if c.startswith(('open_', 'high_', 'low_', 'close_'))]

    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = wavelet_denoise(df[col].values, wavelet=wavelet, level=level)
    return df


# =============================================================================
# 2. INTRA-BAR TIMING FEATURES
# =============================================================================
# Why: Two bars with identical OHLC can have opposite momentum.
#
#   Bar A: open=100, close=105, high reached at t=0:05 → BEARISH (spike then fade)
#   Bar B: open=100, close=105, high reached at t=0:55 → BULLISH (climbed all minute)
#
# Your tick imbalance bars already have timestamps. This is free alpha.
# The PDF cites a paper (arXiv:2509.16137) showing this consistently improves
# Transformer/RNN accuracy. For Mamba (SSM), it's equally useful.

def add_intrabar_timing_features(df: pd.DataFrame,
                                  symbol: str = 'GOLD') -> pd.DataFrame:
    """
    Add intra-bar timing features for a symbol.

    Requires columns:
        time_high_{symbol}: timestamp when the high of the bar was reached
        time_low_{symbol}:  timestamp when the low of the bar was reached
        time_{symbol}:      bar open timestamp (or use index)

    Adds:
        high_first_{symbol}: 1 if high came before low (bearish trap signal)
        hl_time_ratio_{symbol}: normalized position of high within bar [0,1]
        lh_time_ratio_{symbol}: normalized position of low within bar [0,1]
        bar_direction_{symbol}: +1 bullish, -1 bearish based on timing
    """
    df = df.copy()
    col_hi_t = f'time_high_{symbol}'
    col_lo_t = f'time_low_{symbol}'
    col_open_t = f'time_{symbol}'
    col_close = f'close_{symbol}'
    col_open = f'open_{symbol}'

    # If you have actual timestamps, compute fractional position within bar
    if col_hi_t in df.columns and col_lo_t in df.columns and col_open_t in df.columns:
        bar_start = pd.to_datetime(df[col_open_t])
        high_time = pd.to_datetime(df[col_hi_t])
        low_time  = pd.to_datetime(df[col_lo_t])
        bar_len   = pd.Timedelta('1min')

        df[f'hl_time_ratio_{symbol}'] = (
            (high_time - bar_start).dt.total_seconds() / bar_len.total_seconds()
        ).clip(0, 1)
        df[f'lh_time_ratio_{symbol}'] = (
            (low_time - bar_start).dt.total_seconds() / bar_len.total_seconds()
        ).clip(0, 1)
        df[f'high_first_{symbol}'] = (
            df[f'hl_time_ratio_{symbol}'] < df[f'lh_time_ratio_{symbol}']
        ).astype(float)

    # Simpler proxy if exact timestamps unavailable:
    # Use (close - open) direction combined with bar body position
    if col_close in df.columns and col_open in df.columns:
        df[f'bar_direction_{symbol}'] = np.sign(
            df[col_close] - df[col_open]
        )

    return df


# =============================================================================
# 3. SHORT-PERIOD RSI + BOLLINGER BAND WIDTH
# =============================================================================
# Why: Standard RSI(14) is too slow for 1-minute scalping.
# RSI(4-6) reacts within the current micro-move. Bollinger width tells you
# whether you're in a squeeze (breakout imminent) or expansion (momentum trade).
#
# These extend your existing ~28 features with 2-3 per symbol.

def add_scalping_indicators(df: pd.DataFrame,
                             symbol: str = 'GOLD',
                             rsi_period: int = 5,
                             bb_period: int = 20,
                             bb_std: float = 2.0) -> pd.DataFrame:
    """
    Add short-period RSI and Bollinger Band width for a symbol.

    Adds:
        rsi_fast_{symbol}:   RSI with short period (4-6 recommended)
        bb_width_{symbol}:   Bollinger Band width = (upper - lower) / middle
        bb_pct_b_{symbol}:   %B position within bands = (price - lower) / (upper - lower)
    """
    df = df.copy()
    close = df[f'close_{symbol}']

    # --- Fast RSI ---
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df[f'rsi_fast_{symbol}'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    mid   = close.rolling(bb_period).mean()
    std   = close.rolling(bb_period).std()
    upper = mid + bb_std * std
    lower = mid - bb_std * std

    df[f'bb_width_{symbol}'] = (upper - lower) / (mid + 1e-10)
    df[f'bb_pct_b_{symbol}'] = (close - lower) / (upper - lower + 1e-10)

    return df


# =============================================================================
# 4. MULTI-TIMEFRAME REGIME FILTER
# =============================================================================
# Why: A 1-minute model operating without context is "blind." If the 15-minute
# chart is in a strong downtrend, long scalps will fail systematically.
# The regime filter adds 3-5 features that tell the model "what market are we in."
#
# In your pipeline: compute from the same tick data, resampled to 5m/15m.
# Merge back to 1m bars by forward-filling (no lookahead).

def compute_regime_features(df_1m: pd.DataFrame,
                             symbol: str = 'GOLD',
                             fast_ema: int = 50,
                             slow_ema: int = 200) -> pd.DataFrame:
    """
    Compute multi-timeframe regime features from 1m data by resampling.

    Adds to df_1m:
        regime_5m_{symbol}:  EMA50/EMA200 trend direction on 5m bars  (+1/-1/0)
        regime_15m_{symbol}: EMA50/EMA200 trend direction on 15m bars (+1/-1/0)
        above_sma200_1m_{symbol}: whether 1m close is above 200-bar SMA
    """
    df = df_1m.copy()
    close_col = f'close_{symbol}'

    # --- 1m SMA200 ---
    df[f'above_sma200_1m_{symbol}'] = (
        df[close_col] > df[close_col].rolling(200).mean()
    ).astype(float)

    # --- 5m and 15m resampled regime ---
    # Assumes df has a DatetimeIndex
    for tf, label in [('5min', '5m'), ('15min', '15m')]:
        try:
            close_resampled = df[close_col].resample(tf).last().ffill()
            ema_fast = close_resampled.ewm(span=fast_ema).mean()
            ema_slow = close_resampled.ewm(span=slow_ema).mean()
            regime   = np.sign(ema_fast - ema_slow)

            # Forward-fill back to 1m (NO lookahead — reindex then ffill)
            df[f'regime_{label}_{symbol}'] = (
                regime.reindex(df.index, method='ffill')
            )
        except Exception:
            # If index isn't datetime-compatible, skip gracefully
            df[f'regime_{label}_{symbol}'] = 0.0

    return df


# =============================================================================
# 5. EmbTCN-TRANSFORMER (Optional replacement / comparison for Mamba)
# =============================================================================
# Why: The paper benchmarks this as SOTA for trajectory prediction on OHLC data.
# It beats standard Transformers by 10% and RNNs by 60%.
# Architecture: TCN replaces the linear embedding layer of a vanilla Transformer.
#   TCN = local temporal feature extractor (last 5-30 bars)
#   Transformer = global correlation finder (patterns across entire sequence)
#
# Your Mamba/SSM is also strong at sequence modeling. Run both and compare
# val loss + OOS Sharpe. Keep whichever wins on your XAUUSD data.

class CausalConv1d(nn.Module):
    """Causal (no lookahead) 1D convolution with padding on left only."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """Single TCN block: 2x CausalConv + residual + dropout."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, channels)
        residual = x
        out = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.drop(F.gelu(self.norm1(out)))
        out = self.conv2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.drop(F.gelu(self.norm2(out)))
        return out + residual


class TCNEmbedding(nn.Module):
    """
    TCN-based embedding: replaces the linear projection in a vanilla Transformer.
    Stacks dilated TCN blocks with exponentially growing dilation: 1, 2, 4, 8...
    This gives receptive field = (k-1) * (2^L - 1) bars without vanishing gradients.
    """
    def __init__(self, input_dim, d_model, n_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])
        # Receptive field info (printed at init)
        rf = (kernel_size - 1) * (2**n_layers - 1) + 1
        print(f"[TCNEmbedding] Receptive field: {rf} bars "
              f"(kernel={kernel_size}, layers={n_layers})")

    def forward(self, x):
        # x: (batch, seq, input_dim)
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        return out


class EmbTCNTransformer(nn.Module):
    """
    EmbTCN-Transformer: the SOTA architecture from the paper.

    Drop-in replacement for your Mamba encoder in nn.py.
    Input:  (batch, seq_len, n_features)  — same as your current model
    Output: (batch, n_classes)            — same interface

    Args:
        input_dim:    Number of input features (your ~28 features * 3 symbols)
        d_model:      Internal dimension (try 64-256)
        n_tcn_layers: TCN depth (4 = receptive field ~15 bars with kernel=3)
        n_heads:      Transformer attention heads (must divide d_model)
        n_tf_layers:  Transformer encoder layers
        n_classes:    Output classes (3 for Buy/Hold/Sell, or 1 for regression)
        dropout:      Regularization (0.1-0.2 for scalping)
        max_seq_len:  Sequence length for positional encoding

    Why the hybrid beats pure Transformer:
        Pure Transformer uses a linear embedding — it projects features independently
        per timestep, losing temporal context before attention even runs.
        TCN embedding gives each timestep LOCAL context first, then Transformer
        adds GLOBAL correlations on top. Two-level hierarchy = better features.
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 n_tcn_layers: int = 4,
                 n_heads: int = 8,
                 n_tf_layers: int = 2,
                 n_classes: int = 3,
                 dropout: float = 0.1,
                 max_seq_len: int = 256):
        super().__init__()

        # --- TCN Embedding (local temporal features) ---
        self.tcn_embed = TCNEmbedding(input_dim, d_model, n_tcn_layers, dropout=dropout)

        # --- Learnable positional encoding ---
        self.pos_enc = nn.Embedding(max_seq_len, d_model)

        # --- Transformer Encoder (global correlations) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-norm = more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_tf_layers)

        # --- Output head ---
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch, n_classes) — logits, apply softmax for probabilities
        """
        B, T, _ = x.shape

        # 1. TCN extracts local temporal features at each position
        out = self.tcn_embed(x)                          # (B, T, d_model)

        # 2. Add positional encoding (so Transformer knows bar order)
        positions = torch.arange(T, device=x.device)
        out = out + self.pos_enc(positions).unsqueeze(0)  # broadcast over batch

        # 3. Causal mask — Transformer must not look at future bars
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )
        out = self.transformer(out, mask=causal_mask,
                               is_causal=True)            # (B, T, d_model)

        # 4. Use only the last timestep for classification
        out = self.norm(out[:, -1, :])                    # (B, d_model)
        return self.head(out)                             # (B, n_classes)


# =============================================================================
# 6. COMPOSITE REWARD FUNCTION (for PPO or reward-shaped supervised training)
# =============================================================================
# Why: Transaction costs kill 1-minute strategies. A reward that doesn't
# penalize trading will overtrade and go net-negative after spread/commission.
# This implements: R = w1*R_ann - w2*sigma_down + w3*D_ret - w_cost*N_trades

def composite_reward(realized_pnl: float,
                     unrealized_pnl: float,
                     n_trades: int,
                     downside_returns: np.ndarray,
                     benchmark_return: float = 0.0,
                     w1: float = 1.0,
                     w2: float = 0.5,
                     w3: float = 0.3,
                     w_cost: float = 0.01,
                     spread_per_trade: float = 0.0002) -> float:
    """
    Composite reward for a trading step.

    Args:
        realized_pnl:       Closed trade P&L this step (normalized by account)
        unrealized_pnl:     Open position P&L (teaches agent to hold winners)
        n_trades:           Number of new trades opened this step (penalty)
        downside_returns:   Array of negative returns in window (downside risk)
        benchmark_return:   Buy-and-hold return for the window
        spread_per_trade:   Bid-ask spread cost per trade (XAUUSD ~0.2-0.5 pips)
        w1-w3, w_cost:      Weights (tune with Optuna)

    Returns:
        Scalar reward signal
    """
    # Annualized return proxy
    r_ann = realized_pnl + 0.1 * unrealized_pnl  # unrealized counts less

    # Downside deviation (penalizes losses more than gains)
    sigma_down = np.std(downside_returns[downside_returns < 0]) \
                 if len(downside_returns[downside_returns < 0]) > 0 else 0.0

    # Differential return vs benchmark
    d_ret = r_ann - benchmark_return

    # Trade cost penalty (spread + implicit cost)
    cost_penalty = n_trades * (w_cost + spread_per_trade)

    reward = w1 * r_ann - w2 * sigma_down + w3 * d_ret - cost_penalty
    return float(reward)


# =============================================================================
# 7. INTEGRATION SNIPPET FOR YOUR nn.py
# =============================================================================
# Copy-paste this into your existing build_features() or preprocess() function.

def integrate_into_pipeline(df: pd.DataFrame,
                             symbols: list = ['GOLD', 'USDX', 'USDJPY'],
                             apply_denoising: bool = True,
                             apply_timing: bool = True,
                             apply_regime: bool = True) -> pd.DataFrame:
    """
    Full pipeline integration: call this after your raw OHLC load,
    before your existing indicator computation.

    Recommended order in nn.py:
        1. Load raw OHLC                              ← existing
        2. integrate_into_pipeline(df)                ← NEW (this function)
        3. Compute MACD, ATR, etc.                    ← existing (now on denoised prices)
        4. Triple-barrier labeling                    ← existing
        5. Train/val split + scaling                  ← existing
    """
    close_cols = [f'close_{s}' for s in symbols if f'close_{s}' in df.columns]

    # Step 1: Wavelet denoising on OHLC columns
    if apply_denoising:
        all_price_cols = [
            f'{prefix}_{s}'
            for s in symbols
            for prefix in ['open', 'high', 'low', 'close']
            if f'{prefix}_{s}' in df.columns
        ]
        df = denoise_ohlc(df, cols=all_price_cols)

    # Step 2: Per-symbol feature additions
    for symbol in symbols:
        if f'close_{symbol}' not in df.columns:
            continue

        # Intra-bar timing (if timestamp columns exist)
        if apply_timing:
            df = add_intrabar_timing_features(df, symbol)

        # Short RSI + Bollinger width
        df = add_scalping_indicators(df, symbol)

        # Multi-timeframe regime
        if apply_regime and isinstance(df.index, pd.DatetimeIndex):
            df = compute_regime_features(df, symbol)

    return df


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == '__main__':
    # --- Test wavelet denoising ---
    np.random.seed(42)
    fake_close = np.cumsum(np.random.randn(500)) + 2000  # fake XAUUSD
    denoised = wavelet_denoise(fake_close)
    print(f"Original std: {np.std(np.diff(fake_close)):.4f}")
    print(f"Denoised std: {np.std(np.diff(denoised)):.4f}")
    print("→ Denoised has less noise (lower diff std)\n")

    # --- Test EmbTCN-Transformer ---
    batch, seq_len, n_features = 32, 128, 28
    model = EmbTCNTransformer(
        input_dim=n_features,
        d_model=128,
        n_tcn_layers=4,
        n_heads=8,
        n_tf_layers=2,
        n_classes=3,         # Buy / Hold / Sell
        dropout=0.1,
        max_seq_len=seq_len
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EmbTCN-Transformer params: {total_params:,}")

    x = torch.randn(batch, seq_len, n_features)
    logits = model(x)
    print(f"Output shape: {logits.shape}")  # (32, 3)
    probs  = torch.softmax(logits, dim=-1)
    print(f"Sample probabilities (Buy/Hold/Sell): {probs[0].detach().numpy().round(3)}\n")

    # --- Compare against your Mamba model ---
    # To benchmark:
    #   1. Train EmbTCN-Transformer on same data/splits as your Mamba
    #   2. Compare: val_loss, OOS Sharpe, OOS max drawdown
    #   3. Keep whichever wins on XAUUSD specifically
    print("Integration test passed. Add to nn.py before indicator computation.")