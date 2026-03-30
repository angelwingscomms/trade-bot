import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
import tf2onnx
import os
from tensorflow.keras import layers, Model, Input, regularizers

# Architecture Improvements: BiMT-TCN
from tensorflow.keras.layers import Bidirectional, LSTM, MultiHeadAttention, LayerNormalization, Add, Dense, Dropout, GlobalAveragePooling1D, Reshape

tf.keras.utils.set_random_seed(42)

# --- CONFIG ---
TICK_DENSITY = 144
SEQ_LEN = 120
TARGET_HORIZON = 30 # Bars

# 1. LOAD & SYMMETRIC LABELING
df_t = pd.read_csv('bitcoin_ticks.csv')
df_t['bar_id'] = np.arange(len(df_t)) // TICK_DENSITY
df = df_t.groupby('bar_id').agg({'bid':['first','max','min','last'], 'vol':'sum', 'time_msc':'first'})
df.columns = ['open','high','low','close','volume','time_open']
df['spread'] = df_t.groupby('bar_id').apply(lambda x: (x['ask']-x['bid']).mean()).values

def get_symmetric_labels(df, tp_mult=27, sl_mult=5.4):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    atr = ta.atr(df.high, df.low, df.close, length=18).values
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df)-TARGET_HORIZON):
        if np.isnan(atr[i]): continue
        b_tp, b_sl = c[i] + (tp_mult*atr[i]), c[i] - (sl_mult*atr[i])
        s_tp, s_sl = c[i] - (tp_mult*atr[i]), c[i] + (sl_mult*atr[i])

        buy_done, sell_done = False, False
        buy_won,  sell_won  = False, False
        for j in range(i+1, i+TARGET_HORIZON):
            if not buy_done:
                if   hi[j] >= b_tp: buy_won  = True;  buy_done  = True
                elif lo[j] <= b_sl:                    buy_done  = True
            if not sell_done:
                if   lo[j] <= s_tp: sell_won = True;  sell_done = True
                elif hi[j] >= s_sl:                    sell_done = True
            if buy_done and sell_done: break

        if   buy_won and not sell_won: labels[i] = 1
        elif sell_won and not buy_won: labels[i] = 2
        # both or neither → stays 0 (neutral), which is the correct safe label
    return labels

# 2. FEATURE ENGINEERING (17 FEATURES)
df['dt'] = pd.to_datetime(df['time_open'], unit='ms', utc=True)
df['f0'] = np.log(df['close'] / df['close'].shift(1))
df['f1'] = df['spread']
df['f2'] = df['dt'].diff().dt.total_seconds().fillna(0)
df['f3'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['f4'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
df['f5'] = (df['high'] - df['low']) / df['close']
df['f6'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
m = ta.macd(df['close'], 12, 26, 9)
df['f7'], df['f8'], df['f9'] = m.iloc[:, 0]/df['close'], m.iloc[:, 2]/df['close'], m.iloc[:, 1]/df['close']
df['f10'] = ta.atr(df['high'], df['low'], df['close'], length=18) / df['close']
df['f11'] = np.sin(2 * np.pi * df['dt'].dt.hour / 24)
df['f12'] = np.cos(2 * np.pi * df['dt'].dt.hour / 24)
df['f13'] = np.sin(2 * np.pi * df['dt'].dt.dayofweek / 7)
df['f14'] = np.cos(2 * np.pi * df['dt'].dt.dayofweek / 7)
df['f15'] = np.log(df['volume'] + 1)
tvwp = (df['close'] * df['volume']).rolling(144).sum() / (df['volume'].rolling(144).sum() + 1e-8)
df['f16'] = (df['close'] - tvwp) / df['close']

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df['target'] = get_symmetric_labels(df)
X = df[[f'f{i}' for i in range(17)]].values
y = df['target'].values

# 3. ROBUST SCALING
raw_split = int(len(X) * 0.9)
split = raw_split - SEQ_LEN
# Fit scaler on ALL training rows, not the sequence-adjusted subset
median = np.median(X[:raw_split], axis=0)
iqr = np.percentile(X[:raw_split], 75, axis=0) - np.percentile(X[:raw_split], 25, axis=0)
iqr = np.where(iqr < 1e-6, 1.0, iqr)
X_s = np.clip((X - median) / iqr, -10, 10)

X_seq, y_seq = [], []
for i in range(len(X_s)-SEQ_LEN):
    X_seq.append(X_s[i:i+SEQ_LEN])
    y_seq.append(y[i+SEQ_LEN-1])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# 4. BiMT-TCN MODEL
def build_bimt_tcn(seq_len=120, n_features=17):
    reg = regularizers.l2(1e-4)
    inp = Input(shape=(seq_len * n_features,), name="input")
    x = Reshape((seq_len, n_features))(inp)

    # Level 1: Bidirectional Context
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=reg))(x)
    
    # Level 2: Deep Causal TCN (Dilation coverage: 1,2,4,8,16 = 31 steps receptive field)
    for d in [1, 2, 4, 8, 16]:
        shortcut = x
        x = layers.Conv1D(128, 3, padding='causal', dilation_rate=d, activation='relu', kernel_regularizer=reg)(x)
        x = LayerNormalization()(x)
        x = layers.Conv1D(128, 3, padding='causal', dilation_rate=d, activation='relu', kernel_regularizer=reg)(x)
        x = LayerNormalization()(x)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(128, 1, padding='same')(shortcut)
        x = Add()([shortcut, x])

    # Level 3: Transformer Attention
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)
    out = Dense(3, activation='softmax', name="output")(x)
    return Model(inp, out)

model = build_bimt_tcn()
model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with Class Weights
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_seq[:split])
assert split > 0 and split < len(X_seq), f"Split {split} out of range for X_seq len {len(X_seq)}"
model.fit(X_seq[:split].reshape(-1, 2040), y_seq[:split], 
          validation_data=(X_seq[split:].reshape(-1, 2040), y_seq[split:]),
          epochs=54, batch_size=64, class_weight=dict(enumerate(cw)),
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# Export
spec = (tf.TensorSpec((1, 2040), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open("bitcoin_144.onnx", "wb") as f: f.write(model_proto.SerializeToString())

print("\n--- PASTE THESE INTO live.mq5 ---")
print(f"float medians[17] = {{{', '.join([f'{m:.8f}f' for m in median])}}};")
print(f"float iqrs[17] = {{{', '.join([f'{s:.8f}f' for s in iqr])}}};")