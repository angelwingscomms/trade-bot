import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import sys
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from shared_mamba import SharedMambaClassifier

torch.manual_seed(42)
np.random.seed(42)

# --- CONFIG ---
TICK_DENSITY = 540
SEQ_LEN = 120
TARGET_HORIZON = 30

# 1. LOAD & BAR CONSTRUCTION
df_t = pd.read_csv('bitcoin_ticks.csv')
df_t['bar_id'] = np.arange(len(df_t)) // TICK_DENSITY
df = df_t.groupby('bar_id').agg({'bid': ['first', 'max', 'min', 'last'], 'time_msc': 'first'})
df.columns = ['open', 'high', 'low', 'close', 'time_open']
df['spread'] = df_t.groupby('bar_id').apply(lambda x: (x['ask'] - x['bid']).mean()).values

# 2. SYMMETRIC LABELING
def get_symmetric_labels(df, tp_mult=9, sl_mult=5.4):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    atr = ta.atr(df.high, df.low, df.close, length=18).values
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df) - TARGET_HORIZON):
        if np.isnan(atr[i]):
            continue
        b_tp, b_sl = c[i] + (tp_mult * atr[i]), c[i] - (sl_mult * atr[i])
        s_tp, s_sl = c[i] - (tp_mult * atr[i]), c[i] + (sl_mult * atr[i])
        buy_done, sell_done = False, False
        buy_won, sell_won = False, False
        for j in range(i + 1, i + TARGET_HORIZON):
            if not buy_done:
                if hi[j] >= b_tp:
                    buy_won = True; buy_done = True
                elif lo[j] <= b_sl:
                    buy_done = True
            if not sell_done:
                if lo[j] <= s_tp:
                    sell_won = True; sell_done = True
                elif hi[j] >= s_sl:
                    sell_done = True
            if buy_done and sell_done:
                break
        if buy_won and not sell_won:
            labels[i] = 1
        elif sell_won and not buy_won:
            labels[i] = 2
    return labels

# 3. FEATURE ENGINEERING (15 FEATURES)
df['dt'] = pd.to_datetime(df['time_open'], unit='ms', utc=True)
df['f0'] = np.log(df['close'] / df['close'].shift(1))
df['f1'] = df['spread']
df['f2'] = df['dt'].diff().dt.total_seconds().fillna(0)
df['f3'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['f4'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
df['f5'] = (df['high'] - df['low']) / df['close']
df['f6'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
m = ta.macd(df['close'], 12, 26, 9)
df['f7'] = m.iloc[:, 0] / df['close']   # MACD line
df['f8'] = m.iloc[:, 1] / df['close']   # Signal line (MACDs)
df['f9'] = m.iloc[:, 2] / df['close']   # Histogram (MACDh)
df['f10'] = ta.atr(df['high'], df['low'], df['close'], length=18) / df['close']
df['f11'] = np.sin(2 * np.pi * df['dt'].dt.hour / 24)
df['f12'] = np.cos(2 * np.pi * df['dt'].dt.hour / 24)
df['f13'] = np.sin(2 * np.pi * df['dt'].dt.dayofweek / 7)
df['f14'] = np.cos(2 * np.pi * df['dt'].dt.dayofweek / 7)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df['target'] = get_symmetric_labels(df)

X = df[[f'f{i}' for i in range(15)]].values
y = df['target'].values

# 4. ROBUST SCALING (fit on train rows only, no leakage)
raw_split = int(len(X) * 0.9)
split = raw_split - SEQ_LEN

median = np.median(X[:raw_split], axis=0)
iqr = np.percentile(X[:raw_split], 75, axis=0) - np.percentile(X[:raw_split], 25, axis=0)
iqr = np.where(iqr < 1e-6, 1.0, iqr)
X_s = np.clip((X - median) / iqr, -10, 10)

# 5. SEQUENCE CONSTRUCTION — shape (N, SEQ_LEN, 15), label at last bar
X_seq, y_seq = [], []
for i in range(len(X_s) - SEQ_LEN):
    X_seq.append(X_s[i:i + SEQ_LEN])
    y_seq.append(y[i + SEQ_LEN - 1])
X_seq = np.array(X_seq, dtype=np.float32)   # (N, 120, 15)
y_seq = np.array(y_seq, dtype=np.int64)

assert split > 0 and split < len(X_seq), f"Split {split} out of range for X_seq len {len(X_seq)}"

# 6. PYTORCH DATASETS
X_train, y_train = torch.from_numpy(X_seq[:split]), torch.from_numpy(y_seq[:split])
X_val,   y_val   = torch.from_numpy(X_seq[split:]), torch.from_numpy(y_seq[split:])

cw = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_seq[:split])
class_weights = torch.tensor(cw, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=256)

# 7. MAMBA MODEL
model = SharedMambaClassifier(n_features=15, hidden=128)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 8. TRAINING WITH EARLY STOPPING
best_val_loss = float('inf')
patience, wait = 10, 0
best_state = None

for epoch in tqdm(range(54), desc="Training"):
    model.train()
    for xb, yb in train_loader:
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            val_losses.append(criterion(out, yb).item())
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch:02d} | val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)

# 9. EXPORT TO ONNX — input shape (1, 120, 15)
model.eval()
dummy_input = torch.randn(1, SEQ_LEN, 15)
torch.onnx.export(
    model,
    dummy_input,
    "bitcoin_mamba_144.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}}
)
print("✅ ONNX saved: bitcoin_mamba_144.onnx")

# 10. SCALER VALUES FOR live.mq5
print("\n--- PASTE THESE INTO live.mq5 ---")
print(f"float medians[15] = {{{', '.join([f'{v:.8f}f' for v in median])}}};")
print(f"float iqrs[15] = {{{', '.join([f'{v:.8f}f' for v in iqr])}}};")
