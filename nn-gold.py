import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
import tf2onnx
import os
import argparse

# 1. SETUP & PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Load tick data from MT5 Files directory
TICK_FILE_NAME = 'achilles_ticks.csv'
INPUT_TICK_DATA = os.path.join(SCRIPT_DIR, '..', '..', 'Files', TICK_FILE_NAME)

parser = argparse.ArgumentParser(description='Train Achilles neural network model')
parser.add_argument('--tick-density', type=int, default=144, help='Ticks per bar')
args = parser.parse_args()

TICK_DENSITY = args.tick_density
OUTPUT_ONNX_MODEL = os.path.join(SCRIPT_DIR, f'achilles_{TICK_DENSITY}.onnx') 

if not os.path.exists(INPUT_TICK_DATA):
    print(f"Error: {INPUT_TICK_DATA} not found. Ensure you exported data from MT5 first.")
    exit()

print("Loading data...")
df_t = pd.read_csv(INPUT_TICK_DATA)

# 2. TICK-BAR CONSTRUCTION
print("Constructing Tick Bars...")
df_t['bar_id'] = np.arange(len(df_t)) // TICK_DENSITY
df_t['spread_tick'] = df_t['ask'] - df_t['bid']

agg_dict = {
    'bid':['first', 'max', 'min', 'last'],
    'time_msc': ['first', 'last'],
    'spread_tick': 'mean'
}

if 'usdx' in df_t.columns: agg_dict['usdx'] = 'last'
if 'usdjpy' in df_t.columns: agg_dict['usdjpy'] = 'last'

df_agg = df_t.groupby('bar_id').agg(agg_dict)
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
df['usdx'] = df_agg[('usdx', 'last')].values if 'usdx' in df_t.columns else df['close']
df['usdjpy'] = df_agg[('usdjpy', 'last')].values if 'usdjpy' in df_t.columns else df['close']
df.dropna(inplace=True)

# 3. FEATURE ENGINEERING (Parity with MQL5 Built-ins)
print("Building 35 Features...")
# Returns & Spreads
df['f0'] = np.log(df['close'] / df['close'].shift(1))
df['f1'] = df['spread']
df['f2'] = df['duration'] / 1000.0 # Seconds
# Candle Shapes
df['f3'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['f4'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
df['f5'] = (df['high'] - df['low']) / df['close']
df['f6'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
# Oscillators (MQL5 RSI match)
df['f7'] = ta.rsi(df['close'], length=9)
df['f8'] = ta.rsi(df['close'], length=18)
df['f9'] = ta.rsi(df['close'], length=27)
# ATR (Relative)
df['f10'] = ta.atr(df['high'], df['low'], df['close'], length=9) / df['close']
df['f11'] = ta.atr(df['high'], df['low'], df['close'], length=18) / df['close']
df['f12'] = ta.atr(df['high'], df['low'], df['close'], length=27) / df['close']
# MACD (Relative)
m = ta.macd(df['close'], 12, 26, 9)
df['f13'] = m.iloc[:, 0] / df['close'] # MACD Line
df['f14'] = m.iloc[:, 1] / df['close'] # Signal Line
df['f15'] = m.iloc[:, 2] / df['close'] # Histogram
# Moving Averages (Relative)
df['f16'] = (ta.ema(df['close'], 9) - df['close']) / df['close']
df['f17'] = (ta.ema(df['close'], 18) - df['close']) / df['close']
df['f18'] = (ta.ema(df['close'], 27) - df['close']) / df['close']
df['f19'] = (ta.ema(df['close'], 54) - df['close']) / df['close']
df['f20'] = (ta.ema(df['close'], 144) - df['close']) / df['close']
# CCI
df['f21'] = ta.cci(df['high'], df['low'], df['close'], 9)
df['f22'] = ta.cci(df['high'], df['low'], df['close'], 18)
df['f23'] = ta.cci(df['high'], df['low'], df['close'], 27)
# Williams %R
df['f24'] = ta.willr(df['high'], df['low'], df['close'], 9)
df['f25'] = ta.willr(df['high'], df['low'], df['close'], 18)
df['f26'] = ta.willr(df['high'], df['low'], df['close'], 27)
# Momentum (Relative)
df['f27'] = df['close'].diff(9) / df['close']
df['f28'] = df['close'].diff(18) / df['close']
df['f29'] = df['close'].diff(27) / df['close']
# Correlations
df['f30'] = df['usdx'].pct_change()
df['f31'] = df['usdjpy'].pct_change()
# Bollinger Band Width
for p, f_idx in zip([9, 18, 27], [32, 33, 34]):
    bb = ta.bbands(df['close'], length=p)
    df[f'f{f_idx}'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df['close']

df.dropna(inplace=True)

# 4. TARGETING (Matching SL/TP Logic)
TP_MULTIPLIER, SL_MULTIPLIER, H = 2.7, 0.54, 30
def label(df, tp_mult, sl_mult, h):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    # Note: Using ATR18 for labeling to match execution
    atr = ta.atr(df.high, df.low, df.close, length=18).values
    t = np.zeros(len(df), dtype=int)
    for i in range(len(df)-h):
        if np.isnan(atr[i]): continue
        up, lw = c[i]+(tp_mult*atr[i]), c[i]-(sl_mult*atr[i])
        for j in range(i+1, i+h+1):
            if hi[j] >= up: t[i] = 1; break # Buy
            if lo[j] <= lw: t[i] = 2; break # Sell
    return t

df['target'] = label(df, TP_MULTIPLIER, SL_MULTIPLIER, H)
features =[f'f{i}' for i in range(35)]
X, y = df[features].values, df.target.values

# 5. SPLIT & SCALE
train_end = int(len(X) * 0.70)
X_train = X[:train_end]
median = np.median(X_train, axis=0)
iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)

def win(X, y):
    xs, ys = [],[]
    for i in range(len(X) - 120):
        xs.append(X[i:i+120])
        ys.append(y[i+119])
    return np.array(xs), np.array(ys)

X_s = (X - median) / (iqr + 1e-8)
X_seq, y_seq = win(X_s, y)

# --- UPDATE SECTION 6: MODEL ---
# New Input: 4200 flat numbers (120 bars * 35 features)
in_lay = tf.keras.Input(shape=(4200,), name="input") 

# Internally reshape to what LSTM needs: (Batch=1, Timesteps=120, Features=35)
rs = tf.keras.layers.Reshape((120, 35))(in_lay)

ls = tf.keras.layers.LSTM(64, return_sequences=True, activation='mish')(rs)
at = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(ls, ls)
pl = tf.keras.layers.GlobalAveragePooling1D()(at)
ou = tf.keras.layers.Dense(3, activation='softmax')(pl)
model = tf.keras.Model(in_lay, ou)
model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')

# Flatten the training data to match the new input shape
X_seq_flat = X_seq.reshape(-1, 4200)

print("Training...")
model.fit(X_seq_flat, y_seq, epochs=10, batch_size=64)

# 7. EXPORT TO ONNX (FLAT SHAPE)
print("Exporting model to ONNX...")
# Input is now just a flat vector of 4200
spec = (tf.TensorSpec((None, 4200), tf.float32, name="input"),) 
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open(OUTPUT_ONNX_MODEL, "wb") as f:
    f.write(model_proto.SerializeToString())

print("\n--- PASTE THESE INTO live.mq5 ---")
print(f"float medians[35] = {{{', '.join([f'{m:.8f}f' for m in median])}}};")
print(f"float iqrs[35] = {{{', '.join([f'{s:.8f}f' for s in iqr])}}};")