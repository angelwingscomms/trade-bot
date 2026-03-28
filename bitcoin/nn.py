import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
import tf2onnx
import os
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

tf.keras.utils.set_random_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TICK_DATA = os.path.join(SCRIPT_DIR, 'bitcoin_ticks.csv')
TICK_DENSITY = 144
SEQ_LEN = 120
TARGET_HORIZON = 30
OUTPUT_ONNX_MODEL = os.path.join(SCRIPT_DIR, f'bitcoin_{TICK_DENSITY}.onnx') 

print("Loading Microstructure Bars...")
df_t = pd.read_csv(INPUT_TICK_DATA)
df_t['vol'] = df_t['vol'].replace(0, 1.0) 
df_t['bar_id'] = np.arange(len(df_t)) // TICK_DENSITY

df = df_t.groupby('bar_id').agg({
    'bid':['first', 'max', 'min', 'last'],
    'vol': 'sum',
    'time_msc': 'first'
})
df.columns =['open', 'high', 'low', 'close', 'volume', 'time_open']
df['spread'] = df_t.groupby('bar_id').apply(lambda x: (x['ask']-x['bid']).mean()).values

print("Engineering 17 Orthogonal Features...")
df['tpv'] = df['close'] * df['volume']
df['tvwp'] = df['tpv'].rolling(144).sum() / (df['volume'].rolling(144).sum() + 1e-8)
df['dt'] = pd.to_datetime(df['time_open'], unit='ms', utc=True)

# [f0 - f6] Price & Microstructure
df['f0'] = np.log(df['close'] / df['close'].shift(1))
df['f1'] = df['spread']
df['f2'] = df['dt'].diff().dt.total_seconds().fillna(0)
df['f3'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['f4'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
df['f5'] = (df['high'] - df['low']) / df['close']
df['f6'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

# [f7 - f9] MACD Core
m = ta.macd(df['close'], 12, 26, 9)
df['f7'], df['f8'], df['f9'] = m.iloc[:, 0]/df['close'], m.iloc[:, 2]/df['close'], m.iloc[:, 1]/df['close']

# [f10] Volatility
df['f10'] = ta.atr(df['high'], df['low'], df['close'], length=18) / df['close']

# [f11 - f14] Time Embeddings (Strict UTC)
df['f11'] = np.sin(2 * np.pi * df['dt'].dt.hour / 24)
df['f12'] = np.cos(2 * np.pi * df['dt'].dt.hour / 24)
df['f13'] = np.sin(2 * np.pi * df['dt'].dt.dayofweek / 7)
df['f14'] = np.cos(2 * np.pi * df['dt'].dt.dayofweek / 7)

# [f15 - f16] Volume & TVWP Drift
df['f15'] = np.log(df['volume'] + 1)
df['f16'] = (df['close'] - df['tvwp']) / df['close'] 

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

def get_labels(df):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    atr = ta.atr(df.high, df.low, df.close, length=18).values
    t = np.zeros(len(df), dtype=int)
    for i in range(len(df)-TARGET_HORIZON):
        if np.isnan(atr[i]): continue
        up, lw = c[i]+(0.72*atr[i]), c[i]-(0.09*atr[i])
        for j in range(i+1, i+TARGET_HORIZON+1):
            if hi[j] >= up: t[i] = 1; break 
            if lo[j] <= lw: t[i] = 2; break 
    return t

df['target'] = get_labels(df)
X, y = df[[f'f{i}' for i in range(17)]].values, df['target'].values

split_time = int(len(X) * 0.9)
median = np.median(X[:split_time], axis=0)
iqr = np.percentile(X[:split_time], 75, axis=0) - np.percentile(X[:split_time], 25, axis=0) + 1e-8
X_s = (X - median) / iqr

X_seq, y_seq = [],[]
for i in range(len(X_s)-SEQ_LEN):
    X_seq.append(X_s[i:i+SEQ_LEN])
    y_seq.append(y[i+(SEQ_LEN-1)])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# ======================================================================
# 🚨 STRICT PURGED EMBARGO TIME-SERIES SPLIT
# ======================================================================
gap = SEQ_LEN + TARGET_HORIZON
total_seqs = len(X_seq)

print(f"[INFO] Total Temporal Sequences Generated: {total_seqs}")

if total_seqs <= gap + 32:
    print(f"❌ FATAL ERROR: Dimensional Starvation.")
    print(f"You have {total_seqs} sequences. The causality gap requires {gap} steps, leaving nothing for validation.")
    print(f"Fix: Export a vastly larger tick history from MQL5.")
    exit(1)

# Dynamically calculate split to guarantee AT LEAST 10% or a minimum of 32 samples for validation
val_size = max(32, int(total_seqs * 0.10))
train_idx = total_seqs - gap - val_size

if train_idx < 32:
    print("❌ FATAL ERROR: Training set would be reduced to less than 1 batch due to the required Embargo Gap.")
    exit(1)

X_train, y_train = X_seq[:train_idx], y_seq[:train_idx]
X_val, y_val     = X_seq[train_idx+gap:], y_seq[train_idx+gap:]

print(f"[INFO] Array Partitioning Complete:")
print(f"       -> Training Tensor:   {X_train.shape[0]} samples")
print(f"       -> Embargo Gap:       {gap} samples (Purged)")
print(f"       -> Validation Tensor: {X_val.shape[0]} samples")

X_train = X_train.reshape(-1, 2040)
X_val   = X_val.reshape(-1, 2040)

# ======================================================================
# CAUSAL TCN ARCHITECTURE
# ======================================================================
def tcn_block(x, filters, dilation):
    shortcut = layers.Conv1D(filters, 1, padding='same')(x)
    x = layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    return layers.Add()([shortcut, x])

inp = Input(shape=(2040,), name="input")
x = layers.Reshape((120, 17))(inp)
for d in [1, 2, 4, 8, 16]: 
    x = tcn_block(x, 64, d)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(3, activation='softmax', name="output")(x)

model = Model(inp, out)

# Added clipnorm to prevent TCN gradient explosion
model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, clipnorm=1.0), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Defensive weighting against mode collapse
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in zip(np.unique(y_train), cw)}

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, mode='min', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, mode='min', verbose=1)
]

print("\n[INFO] Commencing Training Phase...")
model.fit(X_train, y_train, 
          validation_data=(X_val, y_val), 
          epochs=144, 
          batch_size=32, 
          class_weight=class_weights, 
          callbacks=callbacks)

print("\n[INFO] Post-Training Empirical Evaluation:")
# Explicit check to prevent crashes if validation array is severely compromised
if len(X_val) > 0:
    y_pred = np.argmax(model.predict(X_val, batch_size=32, verbose=0), axis=1)
    
    # Check if all classes are predicted to avoid scikit-learn warnings
    labels_present = np.unique(np.concatenate((y_val, y_pred)))
    print(classification_report(y_val, y_pred, labels=labels_present, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred, labels=labels_present))
else:
    print("⚠️ WARNING: Validation tensor empty. Evaluation bypassed.")

# EXPORT (Batch Size 1)
spec = (tf.TensorSpec((1, 2040), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(OUTPUT_ONNX_MODEL, "wb") as f: f.write(model_proto.SerializeToString())

print("\n--- UPDATE live.mq5 WITH THESE 17-DIMENSIONAL SCALERS ---")
print(f"float medians[17] = {{{', '.join([f'{m:.8f}f' for m in median])}}};")
print(f"float iqrs[17]    = {{{', '.join([f'{s:.8f}f' for s in iqr])}}};")