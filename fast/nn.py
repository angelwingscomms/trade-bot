import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
import tf2onnx
import os

# 1. SETUP & PATHS
# Adjust these paths to where your files actually are on your computer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TICK_DATA = os.path.join(SCRIPT_DIR, 'achilles_ticks.csv')
OUTPUT_ONNX_MODEL = os.path.join(SCRIPT_DIR, 'achilles_144.onnx')
TICK_DENSITY = 144 

if not os.path.exists(INPUT_TICK_DATA):
    print(f"Error: {INPUT_TICK_DATA} not found. Please place the CSV in the same folder.")
    exit()

print("Loading data...")
df_t = pd.read_csv(INPUT_TICK_DATA)

# 2. TICK-BAR CONSTRUCTION
# FIX FLAW 1.1: Use mathematical groupby instead of iloc slicing
# This guarantees zero overlap and true OHLC tick-bar integrity
print("Constructing Tick Bars...")

# Assign bar_id to each tick - ticks 0-143 go to bar 0, ticks 144-287 go to bar 1, etc.
df_t['bar_id'] = np.arange(len(df_t)) // TICK_DENSITY

# Calculate spread per tick before aggregation
df_t['spread_tick'] = df_t['ask'] - df_t['bid']

# Aggregate ticks into bars using proper groupby
# open = first tick of bar, high = max of bar, low = min of bar, close = last tick of bar
agg_dict = {
    'bid': ['first', 'max', 'min', 'last'],
    'time_msc': ['first', 'last'],
    'spread_tick': 'mean'
}

# Include usdx and usdjpy in aggregation if they exist
if 'usdx' in df_t.columns:
    agg_dict['usdx'] = 'last'
if 'usdjpy' in df_t.columns:
    agg_dict['usdjpy'] = 'last'

df_agg = df_t.groupby('bar_id').agg(agg_dict)

# Flatten column names and assign to df
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

# Handle USDX/USDJPY columns from aggregation or use placeholders
if 'usdx' in df_t.columns:
    df['usdx'] = df_agg[('usdx', 'last')].values
else:
    df['usdx'] = df['close']  # Placeholder
    
if 'usdjpy' in df_t.columns:
    df['usdjpy'] = df_agg[('usdjpy', 'last')].values
else:
    df['usdjpy'] = df['close']  # Placeholder

df.dropna(inplace=True)

# 3. FEATURE ENGINEERING (35 FEATURES)
print("Building 35 Features...")
df['f0'] = np.log(df['close'] / df['close'].shift(1))
df['f1'] = df['spread']
df['f2'] = df['duration']
df['f3'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['f4'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
df['f5'] = (df['high'] - df['low']) / df['close']
df['f6'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

df['f7'] = ta.rsi(df['close'], length=9)
df['f8'] = ta.rsi(df['close'], length=18)
df['f9'] = ta.rsi(df['close'], length=27)
df['f10'] = ta.atr(df['high'], df['low'], df['close'], length=9)
df['f11'] = ta.atr(df['high'], df['low'], df['close'], length=18)
df['f12'] = ta.atr(df['high'], df['low'], df['close'], length=27)

m = ta.macd(df['close'], 9, 18, 9)
df['f13'], df['f14'], df['f15'] = m.iloc[:,0], m.iloc[:,2], m.iloc[:,1]

df['f16'] = ta.ema(df['close'], 9) - df['close']
df['f17'] = ta.ema(df['close'], 18) - df['close']
df['f18'] = ta.ema(df['close'], 27) - df['close']
df['f19'] = ta.ema(df['close'], 54) - df['close']
df['f20'] = ta.ema(df['close'], 144) - df['close']

df['f21'] = ta.cci(df['high'], df['low'], df['close'], 9)
df['f22'] = ta.cci(df['high'], df['low'], df['close'], 18)
df['f23'] = ta.cci(df['high'], df['low'], df['close'], 27)

df['f24'] = ta.willr(df['high'], df['low'], df['close'], 9)
df['f25'] = ta.willr(df['high'], df['low'], df['close'], 18)
df['f26'] = ta.willr(df['high'], df['low'], df['close'], 27)

df['f27'] = ta.mom(df['close'], 9)
df['f28'] = ta.mom(df['close'], 18)
df['f29'] = ta.mom(df['close'], 27)

df['f30'] = df['usdx'].pct_change()
df['f31'] = df['usdjpy'].pct_change()

for p, f_idx in zip([9, 18, 27], [32, 33, 34]):
    bb = ta.bbands(df['close'], length=p)
    df[f'f{f_idx}'] = (bb.iloc[:,2] - bb.iloc[:,0]) / (bb.iloc[:,1] + 1e-8)

df.dropna(inplace=True)

# 4. TARGETING
# FIX FLAW 1.3: Handle simultaneous TP/SL breaches properly
# If both TP and SL are breached in the same bar, default to Stop Loss (worst-case)
TP, SL, H = 1.44, 0.50, 30
def label(df, tp, sl, h):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    t = np.zeros(len(df), dtype=int)
    for i in range(len(df)-h):
        up, lw = c[i]+tp, c[i]-sl
        for j in range(i+1, i+h+1):
            tp_hit = hi[j] >= up
            sl_hit = lo[j] <= lw
            # If both hit in same bar, default to SL (worst-case execution)
            if tp_hit and sl_hit:
                t[i] = 2  # Stop Loss takes precedence
                break
            elif tp_hit:
                t[i] = 1  # Take Profit hit
                break
            elif sl_hit:
                t[i] = 2  # Stop Loss hit
                break
    return t

print("Labeling data...")
df['target'] = label(df, TP, SL, H)

# 5. MODEL PREP
# FIX FLAW 1.2: Split data BEFORE calculating normalization parameters
# to prevent future data leakage during training
features = [f'f{i}' for i in range(35)]
X = df[features].values
y = df.target.values

# Split data chronologically (train/val/test) before normalization
# 70% train, 15% val, 15% test
n_samples = len(X)
train_end = int(n_samples * 0.70)
val_end = int(n_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Calculate normalization parameters ONLY from training data
mean, std = X_train.mean(axis=0), X_train.std(axis=0)

# Transform all splits using training-only parameters
X_train_s = (X_train - mean) / (std + 1e-8)
X_val_s = (X_val - mean) / (std + 1e-8)
X_test_s = (X_test - mean) / (std + 1e-8)

def win(X, y, horizon=30):
    xs, ys = [], []
    # FIX: Ensure all labels are valid by limiting range
    # Label at position j is valid only if j <= len(df)-horizon-1
    # We use y[i+119], so i+119 <= len(X)-horizon-1, thus i <= len(X)-horizon-120
    max_i = len(X) - 120 - horizon
    for i in range(max_i + 1):  # +1 because range is exclusive
        xs.append(X[i:i+120]); ys.append(y[i+119])
    return np.array(xs), np.array(ys)

# Create sequences for each split separately
X_train_seq, y_train_seq = win(X_train_s, y_train, H)
X_val_seq, y_val_seq = win(X_val_s, y_val, H)
X_test_seq, y_test_seq = win(X_test_s, y_test, H)

# 6. MODEL ARCHITECTURE
# FIX FLAW 3.1: Contextual Last-Step Extraction
# WHY GlobalAveragePooling1D FAILS: Gives equal weight to Bar 1 and Bar 120, diluting
# current market state with stale data.
# WHY Flatten() FAILS: Multiplies feature dimension by 120 (4,200 inputs to Dense),
# causing parameter explosion and catastrophic overfitting on noisy financial data.
# WHY CONTEXTUAL LAST-STEP EXTRACTION IS SUPERIOR: MultiHeadAttention(ls, ls) means
# the output at T=120 is already a dynamically calculated, weighted sum of ALL
# previous 119 timesteps. Extracting the last timestep performs Attention Pooling
# conditioned exclusively on the most recent market state.
#
# Tensor shape pipeline:
# in_lay: (None, 120, 35) -> ls: (None, 120, 35) -> at: (None, 120, 35)
# -> res_add: (None, 120, 35) -> pl: (None, 35) -> ou: (None, 3)
in_lay = tf.keras.Input(shape=(120, 35))
ls = tf.keras.layers.LSTM(35, return_sequences=True, activation='mish')(in_lay)
at = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=35)(ls, ls)

# 1. Compute the Residual Connection (combining sequential memory + global attention)
res_add = tf.keras.layers.Add(name="Residual_Add")([ls, at])

# 2. Extract the final causal timestep (Index -1) to serve as the context vector
# tf2onnx perfectly supports this slicing operation via opset 13.
pl = tf.keras.layers.Lambda(lambda x: x[:, -1, :], name="Extract_Last_Step")(res_add)

ou = tf.keras.layers.Dense(3, activation='softmax')(tf.keras.layers.Dense(20, activation='mish')(pl))

model = tf.keras.Model(in_lay, ou)
model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# FIX FLAW 3.2: Calculate class weights to handle unbalanced multi-class targeting
# Financial data is heavily skewed - "Do Nothing" (Class 0) often represents 80%+ of data
# Without class weights, the model trivially predicts Class 0 almost every time
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train_seq)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_seq)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Class distribution - Class 0: {np.sum(y_train_seq==0)}, Class 1: {np.sum(y_train_seq==1)}, Class 2: {np.sum(y_train_seq==2)}")
print(f"Class weights: {class_weight_dict}")

print("Starting training...")
# Use explicit validation data instead of validation_split to ensure no leakage
# Pass class_weight to aggressively penalize the model for missing Class 1 (Buy) and Class 2 (Sell)
model.fit(X_train_seq, y_train_seq, epochs=54, batch_size=64, 
          validation_data=(X_val_seq, y_val_seq),
          class_weight=class_weight_dict)

# 7. EXPORT TO ONNX
print("Exporting model to ONNX...")
spec = (tf.TensorSpec((None, 120, 35), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open(OUTPUT_ONNX_MODEL, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model saved to {OUTPUT_ONNX_MODEL}")

# PRINT STATS FOR MQL5 / TRADING BOT
print("\n--- Copy these into your C++ / MQL5 code ---")
means_str = f"float means[35]={{{','.join([f'{m:.6f}f' for m in mean])}}};".replace(',', ', ')
stds_str = f"float stds[35]={{{','.join([f'{s:.6f}f' for s in std])}}};"

print(means_str)
print(stds_str)

# Save to file for easy reference
with open(os.path.join(SCRIPT_DIR, 'normalization_params.txt'), 'w') as f:
    f.write("// Copy these lines into live.mq5 (lines 14-15)\n\n")
    f.write(means_str + "\n")
    f.write(stds_str + "\n")

# Auto-generate updated live.mq5 with correct normalization params
print("\n✅ Generated live_updated.mq5 with correct normalization parameters")
live_template = os.path.join(SCRIPT_DIR, 'live.mq5')
if os.path.exists(live_template):
    with open(live_template, 'r') as f_in:
        content = f_in.read()
    
    # Replace placeholder lines 14-15
    content_updated = content.replace(
        'float means[35] = {0.0f}; // ⚠️ PASTE FROM PYTHON',
        means_str
    ).replace(
        'float stds[35]  = {1.0f}; // ⚠️ PASTE FROM PYTHON',
        stds_str
    )
    
    output_file = os.path.join(SCRIPT_DIR, 'live_updated.mq5')
    with open(output_file, 'w') as f_out:
        f_out.write(content_updated)
    
    print(f"   📄 File saved: {output_file}")
    print("   ℹ️  Replace live.mq5 with live_updated.mq5 or copy lines 14-15 from it")