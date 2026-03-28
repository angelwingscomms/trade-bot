import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
import tf2onnx
import os
import argparse
from sklearn.utils.class_weight import compute_class_weight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TICK_DATA = os.path.join(SCRIPT_DIR, 'bitcoin_ticks.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--tick-density', type=int, default=144)
args = parser.parse_args()

TICK_DENSITY = args.tick_density
OUTPUT_ONNX_MODEL = os.path.join(SCRIPT_DIR, f'bitcoin_{TICK_DENSITY}.onnx') 

df_t = pd.read_csv(INPUT_TICK_DATA)
df_t = df_t[(df_t['bid'] > 0) & (df_t['ask'] > 0)].copy() # Defend against -Inf Logs

print("Constructing Tick Bars...")
df_t['bar_id'] = np.arange(len(df_t)) // TICK_DENSITY
df_t['spread_tick'] = df_t['ask'] - df_t['bid']

df_agg = df_t.groupby('bar_id').agg({
    'bid':['first', 'max', 'min', 'last'],
    'time_msc': ['first', 'last'],
    'spread_tick': 'mean'
})

df = pd.DataFrame({
    'open': df_agg[('bid', 'first')].values,
    'high': df_agg[('bid', 'max')].values,
    'low': df_agg[('bid', 'min')].values,
    'close': df_agg[('bid', 'last')].values,
    'spread': df_agg[('spread_tick', 'mean')].values,
    'time_open': df_agg[('time_msc', 'first')].values,
    'time_close': df_agg[('time_msc', 'last')].values
})
df['duration'] = df['time_close'] - df['time_open']
df.dropna(inplace=True)

print("Building Features...")
df['f0'] = np.log(df['close'] / df['close'].shift(1))
df['f1'] = df['spread']
df['f2'] = df['duration'] / 1000.0 
df['f3'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['f4'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
df['f5'] = (df['high'] - df['low']) / df['close']
df['f6'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

for p, f_idx in zip([9, 18, 27], [7, 8, 9]): df[f'f{f_idx}'] = ta.rsi(df['close'], length=p)
for p, f_idx in zip([9, 18, 27],[10, 11, 12]): df[f'f{f_idx}'] = ta.atr(df['high'], df['low'], df['close'], length=p) / df['close']

# CRITICAL FIX: Align Python feature indexes with MQL5's native buffers
m = ta.macd(df['close'], 12, 26, 9)
df['f13'] = m.iloc[:, 0] / df['close'] # MACD Line
df['f14'] = m.iloc[:, 2] / df['close'] # Signal Line (Index 2)
df['f15'] = m.iloc[:, 1] / df['close'] # Histogram (Index 1)

for p, f_idx in zip([9, 18, 27, 54, 144],[16, 17, 18, 19, 20]): df[f'f{f_idx}'] = (ta.ema(df['close'], p) - df['close']) / df['close']
for p, f_idx in zip([9, 18, 27], [21, 22, 23]): df[f'f{f_idx}'] = ta.cci(df['high'], df['low'], df['close'], p)
for p, f_idx in zip([9, 18, 27], [24, 25, 26]): df[f'f{f_idx}'] = ta.willr(df['high'], df['low'], df['close'], p)
for p, f_idx in zip([9, 18, 27],[27, 28, 29]): df[f'f{f_idx}'] = df['close'].diff(p) / df['close']
for p, f_idx in zip([9, 18, 27],[30, 31, 32]):
    bb = ta.bbands(df['close'], length=p)
    df[f'f{f_idx}'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df['close']

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

def label(df, tp_mult, sl_mult, h=30):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    atr = ta.atr(df.high, df.low, df.close, length=18).values
    t = np.zeros(len(df), dtype=int)
    for i in range(len(df)-h):
        if np.isnan(atr[i]) or atr[i] == 0: continue
        up, lw = c[i]+(tp_mult*atr[i]), c[i]-(sl_mult*atr[i])
        for j in range(i+1, i+h+1):
            if hi[j] >= up: t[i] = 1; break 
            if lo[j] <= lw: t[i] = 2; break 
    return t

df['target'] = label(df, 2.7, 0.54, 30)
features =[f'f{i}' for i in range(33)]
X, y = df[features].values, df.target.values

train_end = int(len(X) * 0.70)
median = np.median(X[:train_end], axis=0)
iqr = np.percentile(X[:train_end], 75, axis=0) - np.percentile(X[:train_end], 25, axis=0)

X_s = (X - median) / (iqr + 1e-8)

def win(X, y):
    xs, ys = [],[]
    for i in range(len(X) - 120):
        xs.append(X[i:i+120])
        ys.append(y[i+119])
    return np.array(xs), np.array(ys)

X_seq, y_seq = win(X_s, y)
X_seq_flat = X_seq.reshape(-1, 3960)

# CRITICAL FIX: Architectural overhaul to prevent exploding gradients
in_lay = tf.keras.Input(shape=(3960,), name="input") 
rs = tf.keras.layers.Reshape((120, 33))(in_lay)

# Must use tanh to leverage CuDNN and prevent cell-state NaN explosion
ls = tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh')(rs)

# Self-attention must use Layer Normalization + Residuals
ln1 = tf.keras.layers.LayerNormalization()(ls)
at = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(ln1, ln1)
res = tf.keras.layers.Add()([ls, at])

ln2 = tf.keras.layers.LayerNormalization()(res)
pl = tf.keras.layers.GlobalAveragePooling1D()(ln2)
dp = tf.keras.layers.Dropout(0.3)(pl)
ou = tf.keras.layers.Dense(3, activation='softmax')(dp)

model = tf.keras.Model(in_lay, ou)

# Clipnorm prevents exploding gradients. 
opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')

# Compute weights to fix inherent neutral-class bias
classes = np.unique(y_seq)
weights = compute_class_weight('balanced', classes=classes, y=y_seq)
class_weight = {c: w for c, w in zip(classes, weights)}

print("Training Network...")
model.fit(X_seq_flat, y_seq, epochs=9, batch_size=64, class_weight=class_weight)

print("Exporting model to ONNX...")
spec = (tf.TensorSpec((None, 3960), tf.float32, name="input"),) 
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open(OUTPUT_ONNX_MODEL, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"float medians[33] = {{{', '.join([f'{m:.8f}f' for m in median])}}};")
print(f"float iqrs[33]    = {{{', '.join([f'{s:.8f}f' for s in iqr])}}};")