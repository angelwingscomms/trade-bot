import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import sys
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from shared_mamba import SharedMambaClassifier

torch.manual_seed(42)
np.random.seed(42)

# --- CONFIG ---
TICK_DENSITY  = 540   # ticks per bar for GOLD
SEQ_LEN       = 120
TARGET_HORIZON = 30
N_FEATURES    = 48    # 16 features × 3 symbols
DATA_FILE     = "gold_market_ticks.csv"
SYMBOL_ORDER  = ("XAUUSD", "$USDX", "USDJPY")

# ─────────────────────────────────────────────────────────────
# 1. BAR CONSTRUCTION (TIME-ALIGNED)
# ─────────────────────────────────────────────────────────────
def build_aligned_bars(csv_path, symbols, tick_density):
    print(f"[INFO] Loading combined tick CSV: {csv_path}…")
    df_all = pd.read_csv(csv_path)
    df_all["symbol"] = df_all["symbol"].astype(str).str.upper()

    sym_gold = symbols[0]
    df_gold = df_all[df_all["symbol"] == sym_gold].sort_values("time_msc").reset_index(drop=True)
    if df_gold.empty:
        raise ValueError(f"No ticks found for {sym_gold}")
        
    # Create tick bars for GOLD
    df_gold['bar_id'] = np.arange(len(df_gold)) // tick_density
    bar_ends = df_gold.groupby('bar_id')['time_msc'].last().values
    
    # Bins for exact time alignment. Start slightly before the first tick.
    bins = [-1] + list(bar_ends)
    
    bars_by_symbol = {}
    for sym in symbols:
        df_sym = df_all[df_all["symbol"] == sym].sort_values("time_msc").reset_index(drop=True)
        if df_sym.empty:
            raise ValueError(f"No ticks found for {sym}")
            
        if sym == sym_gold:
            df_sym_binned = df_gold
        else:
            # Assign each tick to the corresponding GOLD bar time window
            # np.searchsorted handles duplicate timestamps properly, unlike pd.cut
            bar_ids = np.searchsorted(bar_ends, df_sym['time_msc'].values, side='left')
            valid = bar_ids < len(bar_ends)
            df_sym_binned = df_sym[valid].copy()
            df_sym_binned['bar_id'] = bar_ids[valid]

        has_ask = 'ask' in df_sym_binned.columns
        agg = {'bid': ['first', 'max', 'min', 'last'], 'time_msc': 'first'}
        
        if has_ask:
            df_sym_binned['spread'] = df_sym_binned['ask'] - df_sym_binned['bid']
            agg_spread = df_sym_binned.groupby('bar_id')['spread'].last()
            agg['ask'] = ['max', 'min']

        df_bars = df_sym_binned.groupby('bar_id').agg(agg)
        
        if has_ask:
            df_bars.columns = ['open', 'high', 'low', 'close', 'time_open', 'ask_high', 'ask_low']
            df_bars['spread'] = agg_spread
        else:
            df_bars.columns = ['open', 'high', 'low', 'close', 'time_open']
            df_bars['spread'] = 0.0
            df_bars['ask_high'] = df_bars['high']
            df_bars['ask_low'] = df_bars['low']

        # Reindex to ensure all GOLD bar_ids are present
        df_bars = df_bars.reindex(np.arange(len(bar_ends)))
        
        # Forward fill if a symbol had no ticks during a GOLD bar
        df_bars['close'] = df_bars['close'].ffill().bfill()
        df_bars['open'] = df_bars['open'].fillna(df_bars['close'])
        df_bars['high'] = df_bars['high'].fillna(df_bars['close'])
        df_bars['low'] = df_bars['low'].fillna(df_bars['close'])
        
        # Spread should also be forward-filled, not zeroed
        df_bars['spread'] = df_bars['spread'].ffill().bfill()
        
        # Ask prices should fall back to close + spread if there were no ticks
        df_bars['ask_high'] = df_bars['ask_high'].fillna(df_bars['high'] + df_bars['spread'])
        df_bars['ask_low'] = df_bars['ask_low'].fillna(df_bars['low'] + df_bars['spread'])

        if sym != sym_gold:
            gold_time_open = df_gold.groupby('bar_id')['time_msc'].first()
            df_bars['time_open'] = df_bars['time_open'].fillna(gold_time_open).ffill().bfill()

        bars_by_symbol[sym] = df_bars.reset_index(drop=True)
        print(f"[INFO] {sym}: built {len(bars_by_symbol[sym])} aligned bars.")

    return bars_by_symbol


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING — 16 features per symbol
# ─────────────────────────────────────────────────────────────
# Feature index map (same layout for every symbol):
#   f0  = log return (close-to-close)
#   f1  = spread / close
#   f2  = bar duration in seconds
#   f3  = upper wick / close
#   f4  = lower wick / close
#   f5  = range / close
#   f6  = close position in range [0,1]
#   f7  = MACD line / close         (EMA12 - EMA26)
#   f8  = MACD signal / close       (EMA9 of MACD line)
#   f9  = MACD histogram / close    (MACD line - signal)
#   f10 = ATR(14) / close
#   f11 = sin(2π * UTC hour / 24)
#   f12 = cos(2π * UTC hour / 24)
#   f13 = sin(2π * UTC weekday / 7)
#   f14 = cos(2π * UTC weekday / 7)
#   f15 = RSI(14) / 100  — normalised to [0, 1]

def compute_features(df):
    """
    Computes 16 features for a bars DataFrame.
    Returns a (N, 16) numpy float32 array and the datetime series.
    """
    df = df.copy()
    # Interpret time_open as raw broker time (not UTC) to match MQL5 logic
    df['dt'] = pd.to_datetime(df['time_open'], unit='ms')

    c  = df['close']
    c1 = c.shift(1)

    feat = pd.DataFrame(index=df.index)

    # Price-based
    feat['f0']  = np.log(c / (c1 + 1e-10))
    feat['f1']  = df['spread'] / (c + 1e-10)
    feat['f2']  = df['dt'].diff().dt.total_seconds().fillna(0)
    feat['f3']  = (df['high'] - df[['open', 'close']].max(axis=1)) / (c + 1e-10)
    feat['f4']  = (df[['open', 'close']].min(axis=1) - df['low']) / (c + 1e-10)
    feat['f5']  = (df['high'] - df['low']) / (c + 1e-10)
    feat['f6']  = (c - df['low']) / (df['high'] - df['low'] + 1e-8)

    # MACD (EMA 12, 26, 9)
    # pandas_ta macd returns [MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9]
    m = ta.macd(c, 12, 26, 9)
    feat['f7']  = m.iloc[:, 0] / (c + 1e-10)   # MACD line
    feat['f8']  = m.iloc[:, 2] / (c + 1e-10)   # Signal line
    feat['f9']  = m.iloc[:, 1] / (c + 1e-10)   # Histogram

    # ATR(14)
    feat['f10'] = ta.atr(df['high'], df['low'], c, length=14) / (c + 1e-10)

    # Time cyclicals (Match MQL5 integer-hour logic)
    hours = df['dt'].dt.hour
    days  = df['dt'].dt.dayofweek # Mon=0, Sun=6 (Matches MQL5 (days+3)%7)
    
    feat['f11'] = np.sin(2 * np.pi * hours / 24.0)
    feat['f12'] = np.cos(2 * np.pi * hours / 24.0)
    feat['f13'] = np.sin(2 * np.pi * days / 7.0)
    feat['f14'] = np.cos(2 * np.pi * days / 7.0)

    # RSI(14) normalised
    feat['f15'] = ta.rsi(c, length=14) / 100.0

    return feat.values.astype(np.float32), df['dt']


# ─────────────────────────────────────────────────────────────
# 3. SYMMETRIC LABELING (GOLD only, ATR-based TP/SL)
# ─────────────────────────────────────────────────────────────
def get_symmetric_labels(df_gold, tp_mult=9.0, sl_mult=5.4):
    c   = df_gold['close'].values
    hi  = df_gold['high'].values
    lo  = df_gold['low'].values
    spr = df_gold['spread'].values
    ask_hi = df_gold['ask_high'].values if 'ask_high' in df_gold.columns else hi + spr
    ask_lo = df_gold['ask_low'].values if 'ask_low' in df_gold.columns else lo + spr
    atr = ta.atr(df_gold['high'], df_gold['low'], df_gold['close'], length=14).values
    labels = np.zeros(len(df_gold), dtype=np.int64)

    for i in range(len(df_gold) - TARGET_HORIZON):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
            
        # Long entry is at ask (close + spread), exits are evaluated at bid
        long_entry = c[i] + spr[i]
        b_tp = long_entry + tp_mult * atr[i]
        b_sl = long_entry - sl_mult * atr[i]
        
        # Short entry is at bid (close), exits are evaluated at ask (true ask_high / ask_low)
        short_entry = c[i]
        s_tp = short_entry - tp_mult * atr[i]
        s_sl = short_entry + sl_mult * atr[i]

        buy_done = sell_done = False
        buy_won  = sell_won  = False

        for j in range(i + 1, i + TARGET_HORIZON + 1):
            if not buy_done:
                hit_tp = hi[j] >= b_tp
                hit_sl = lo[j] <= b_sl
                # If both hit in same bar, assume worst case (SL)
                if hit_tp and hit_sl:
                    buy_done = True
                elif hit_tp:
                    buy_won = True;  buy_done = True
                elif hit_sl:
                    buy_done = True
                    
            if not sell_done:
                hit_tp = ask_lo[j] <= s_tp
                hit_sl = ask_hi[j] >= s_sl
                # If both hit in same bar, assume worst case (SL)
                if hit_tp and hit_sl:
                    sell_done = True
                elif hit_tp:
                    sell_won = True; sell_done = True
                elif hit_sl:
                    sell_done = True
                    
            if buy_done and sell_done:
                break

        if   buy_won and not sell_won:  labels[i] = 1
        elif sell_won and not buy_won:  labels[i] = 2

    return labels


def main():
    # ─────────────────────────────────────────────────────────────
    # 4. LOAD DATA & BUILD FEATURES
    # ─────────────────────────────────────────────────────────────
    bars_by_symbol = build_aligned_bars(DATA_FILE, SYMBOL_ORDER, TICK_DENSITY)
    df_gold   = bars_by_symbol[SYMBOL_ORDER[0]]
    df_usdx   = bars_by_symbol[SYMBOL_ORDER[1]]
    df_usdjpy = bars_by_symbol[SYMBOL_ORDER[2]]

    # Align all three to the same length (shortest wins)
    N = min(len(df_gold), len(df_usdx), len(df_usdjpy))
    df_gold   = df_gold.iloc[:N].reset_index(drop=True)
    df_usdx   = df_usdx.iloc[:N].reset_index(drop=True)
    df_usdjpy = df_usdjpy.iloc[:N].reset_index(drop=True)

    print(f"[INFO] Aligned bar count: {N}")

    feat_gold,  _ = compute_features(df_gold)
    feat_usdx,  _ = compute_features(df_usdx)
    feat_usdjpy,_ = compute_features(df_usdjpy)

    # Concatenate: [gold(16) | usdx(16) | usdjpy(16)] = 48 features per bar
    X = np.concatenate([feat_gold, feat_usdx, feat_usdjpy], axis=1)  # (N, 48)
    assert X.shape[1] == N_FEATURES, f"Expected 48 features, got {X.shape[1]}"

    # Labels from GOLD bars only
    y = get_symmetric_labels(df_gold)

    # ─────────────────────────────────────────────────────────────
    # 5. ROBUST SCALING (fit on train partition only — no leakage)
    # ─────────────────────────────────────────────────────────────
    # DROP WARMUP BARS (first 50 bars) to ensure indicators are stable
    WARMUP = 50
    X = X[WARMUP:]
    y = y[WARMUP:]
    N = len(X)
    
    raw_split = int(N * 0.9)

    # Calculate median and iqr ignoring NaNs on TRAINING SET ONLY
    median = np.nanmedian(X[:raw_split], axis=0)
    median = np.nan_to_num(median, nan=0.0) 
    iqr    = (np.nanpercentile(X[:raw_split], 75, axis=0)
            - np.nanpercentile(X[:raw_split], 25, axis=0))
    iqr    = np.nan_to_num(iqr, nan=1.0)
    iqr    = np.where(iqr < 1e-6, 1.0, iqr)

    # Scale the entire dataset using training statistics
    X_s = np.clip((X - median) / iqr, -10, 10).astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    # 6. SEQUENCE CONSTRUCTION  (N_seq, SEQ_LEN, 48)
    # ─────────────────────────────────────────────────────────────
    # Re-evaluate valid_mask after scaling/clipping
    valid_mask = ~np.isnan(X_s).any(axis=1)

    X_seq_train, y_seq_train = [], []
    for i in range(raw_split - SEQ_LEN):
        # A sequence is valid if all bars in the sequence are valid and the label is within the valid range
        if valid_mask[i : i + SEQ_LEN].all() and (i + SEQ_LEN - 1 + TARGET_HORIZON < N):
            X_seq_train.append(X_s[i : i + SEQ_LEN])
            y_seq_train.append(y[i + SEQ_LEN - 1])
        
    X_seq_val, y_seq_val = [], []
    # Prevent lookahead leakage: Validation features must not overlap with training label horizons
    for i in range(raw_split + TARGET_HORIZON, N - SEQ_LEN):
        if valid_mask[i : i + SEQ_LEN].all() and (i + SEQ_LEN - 1 + TARGET_HORIZON < N):
            X_seq_val.append(X_s[i : i + SEQ_LEN])
            y_seq_val.append(y[i + SEQ_LEN - 1])

    X_train = torch.tensor(np.array(X_seq_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_seq_train), dtype=torch.int64)
    X_val   = torch.tensor(np.array(X_seq_val), dtype=torch.float32)
    y_val   = torch.tensor(np.array(y_seq_val), dtype=torch.int64)

    # ─────────────────────────────────────────────────────────────
    # 7. DATASETS & CLASS WEIGHTS
    # ─────────────────────────────────────────────────────────────
    unique_classes = np.unique(y_seq_train)
    cw = compute_class_weight('balanced', classes=unique_classes, y=np.array(y_seq_train))
    weight_dict = {c: w for c, w in zip(unique_classes, cw)}
    cw_full = [weight_dict.get(i, 1.0) for i in range(3)]
    class_weights = torch.tensor(cw_full, dtype=torch.float32)
    print(f"[INFO] Class weights: {cw_full}")

    # For CPU, we use smaller batch size and num_workers=0 (default) for stability, but we can try to optimize
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32,  shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ─────────────────────────────────────────────────────────────
    # 8. MAMBA MODEL  (d_model = 128)
    # ─────────────────────────────────────────────────────────────
    # Reduce model size slightly for faster CPU training without losing too much capacity
    model     = SharedMambaClassifier(n_features=N_FEATURES, d_model=64, hidden=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ─────────────────────────────────────────────────────────────
    # 9. TRAINING WITH EARLY STOPPING
    # ─────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience, wait = 10, 0
    best_state = None

    for epoch in range(54):
        model.train()
        train_losses = []
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                # Use carriage return \r to overwrite the same line, preventing massive console spam
                print(f"\r  Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{len(train_loader)} | loss: {loss.item():.4f}", end="", flush=True)

        print() # New line after the epoch's batches are complete

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        
        # Guard against empty validation set
        if len(val_losses) == 0:
            print("[WARN] Validation set is empty! Check your dataset size.")
            val_loss = float('inf')
        else:
            val_loss = float(np.mean(val_losses))
            
        train_loss = float(np.mean(train_losses))
        print(f"Epoch {epoch:02d} Summary | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"✅ Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[WARN] No best state found. Using the last epoch's state.")

    # ─────────────────────────────────────────────────────────────
    # 10. ONNX EXPORT — input shape (1, 120, 48)
    # ─────────────────────────────────────────────────────────────
    model.eval()
    model.to("cpu")
    dummy = torch.randn(1, SEQ_LEN, N_FEATURES)
    torch.onnx.export(
        model, dummy, "gold_mamba.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}},
    )
    print("✅ ONNX saved: gold_mamba.onnx")

    # ─────────────────────────────────────────────────────────────
    # 11. SCALER OUTPUT FOR live.mq5
    # ─────────────────────────────────────────────────────────────
    print("\n--- PASTE THESE INTO live.mq5 ---")
    print(f"float medians[48] = {{{', '.join([f'{v:.8f}f' for v in median])}}};")
    print(f"float iqrs[48]    = {{{', '.join([f'{v:.8f}f' for v in iqr])}}};")

if __name__ == "__main__":
    main()
