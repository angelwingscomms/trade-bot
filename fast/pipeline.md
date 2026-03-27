```cpp
#property script_show_inputs // Show settings window
input int ticks_to_export = 2160000; // Total ticks (~5 days of Gold)
input string USDX_Symbol = "$USDX"; // Name of USD Index
input string USDJPY_Symbol = "USDJPY"; // Name of USDJPY

// FLAW 4.4 FIX: Optimized tick data exporting with StringFormat and Two-Pointer Merge

//+------------------------------------------------------------------+
//| Logging helper functions                                          |
//+------------------------------------------------------------------+
void LogInfo(string message) {
   Print("[INFO] ", message);
}

void LogSuccess(string message) {
   Print("✅ [SUCCESS] ", message);
}

void LogWarning(string message) {
   Print("⚠️ [WARNING] ", message);
}

void LogError(string message) {
   Print("❌ [ERROR] ", message);
}

void LogProgress(string stage, int current, int total, string extra = "") {
   int percent = (int)((double)current / total * 100);
   Print("📊 [PROGRESS] ", stage, ": ", current, "/", total, " (", percent, "%)", extra);
}

void LogSeparator() {
   Print("═══════════════════════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| Main script function                                              |
//+------------------------------------------------------------------+
void OnStart() {
   ulong start_time = GetTickCount64(); // Script start timestamp
   LogSeparator();
   LogInfo("ACHILLES TICK DATA EXPORTER - Starting execution");
   LogSeparator();
   LogInfo(StringFormat("Parameters: ticks_to_export=%d, USDX='%s', USDJPY='%s'", 
                        ticks_to_export, USDX_Symbol, USDJPY_Symbol));
   LogInfo(StringFormat("Main symbol: %s", _Symbol));
   
   MqlTick ticks[], usdx_ticks[], usdjpy_ticks[]; // Arrays to hold tick data
   
   // === SYMBOL SELECTION PHASE ===
   LogInfo("Phase 1: Symbol Selection");
   LogInfo(StringFormat("  Attempting to select USDX symbol: '%s'", USDX_Symbol));
   bool usdx_available = SymbolSelect(USDX_Symbol, true);
   if(usdx_available) {
      LogSuccess(StringFormat("  USDX symbol '%s' selected successfully", USDX_Symbol));
   } else {
      LogWarning(StringFormat("  USDX symbol '%s' NOT available - will use placeholder (0.0)", USDX_Symbol));
   }
   
   LogInfo(StringFormat("  Attempting to select USDJPY symbol: '%s'", USDJPY_Symbol));
   bool usdjpy_available = SymbolSelect(USDJPY_Symbol, true);
   if(usdjpy_available) {
      LogSuccess(StringFormat("  USDJPY symbol '%s' selected successfully", USDJPY_Symbol));
   } else {
      LogWarning(StringFormat("  USDJPY symbol '%s' NOT available - will use placeholder (0.0)", USDJPY_Symbol));
   }
   
   // === TICK DATA COPYING PHASE ===
   LogSeparator();
   LogInfo("Phase 2: Tick Data Acquisition");
   LogInfo(StringFormat("  Copying %d ticks for main symbol '%s'...", ticks_to_export, _Symbol));
   
   ulong copy_start = GetTickCount64();
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, ticks_to_export);
   ulong copy_time = GetTickCount64() - copy_start;
   
   if(copied <= 0) {
      LogError(StringFormat("  Failed to copy ticks for '%s'! Error code: %d", _Symbol, GetLastError()));
      LogError("  Script terminated - no data to export");
      return;
   }
   LogSuccess(StringFormat("  Copied %d ticks for '%s' in %llu ms", copied, _Symbol, copy_time));
   
   // Log tick data time range
   if(copied > 0) {
      datetime first_time = (datetime)(ticks[0].time_msc / 1000);
      datetime last_time = (datetime)(ticks[copied-1].time_msc / 1000);
      LogInfo(StringFormat("  Tick time range: %s to %s", 
                           TimeToString(first_time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
                           TimeToString(last_time, TIME_DATE|TIME_MINUTES|TIME_SECONDS)));
   }
   
   // Get tick data for auxiliary symbols if available
   int usdx_copied = 0, usdjpy_copied = 0;
   
   if(usdx_available) {
      LogInfo(StringFormat("  Copying %d ticks for USDX '%s'...", ticks_to_export, USDX_Symbol));
      copy_start = GetTickCount64();
      usdx_copied = CopyTicks(USDX_Symbol, usdx_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      copy_time = GetTickCount64() - copy_start;
      
      if(usdx_copied <= 0) {
         LogWarning(StringFormat("  USDX ticks not available (error: %d), using placeholder", GetLastError()));
         usdx_available = false;
      } else {
         LogSuccess(StringFormat("  Copied %d USDX ticks in %llu ms", usdx_copied, copy_time));
      }
   }
   
   if(usdjpy_available) {
      LogInfo(StringFormat("  Copying %d ticks for USDJPY '%s'...", ticks_to_export, USDJPY_Symbol));
      copy_start = GetTickCount64();
      usdjpy_copied = CopyTicks(USDJPY_Symbol, usdjpy_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      copy_time = GetTickCount64() - copy_start;
      
      if(usdjpy_copied <= 0) {
         LogWarning(StringFormat("  USDJPY ticks not available (error: %d), using placeholder", GetLastError()));
         usdjpy_available = false;
      } else {
         LogSuccess(StringFormat("  Copied %d USDJPY ticks in %llu ms", usdjpy_copied, copy_time));
      }
   }
   
    // === FILE CREATION PHASE ===
    LogSeparator();
    LogInfo("Phase 3: File Creation");
    LogInfo("  Creating output file: fast/achilles_ticks.csv");
    LogInfo("  (MQL5 sandbox restricts to MQL5\\Files, run move_ticks.py after export)");
    
    int h = FileOpen("fast/achilles_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) {
      LogError(StringFormat("  Failed to create file! Error code: %d", GetLastError()));
      LogError("  Script terminated - cannot write data");
      return;
   }
   LogSuccess("  File opened successfully");
   
   FileWrite(h, "time_msc,bid,ask,usdx,usdjpy"); // Write CSV header
   LogInfo("  CSV header written: time_msc,bid,ask,usdx,usdjpy");
   
   // === DATA PROCESSING PHASE ===
   LogSeparator();
   LogInfo("Phase 4: Data Processing & Export");
   LogInfo(StringFormat("  Processing %d ticks with Two-Pointer Merge algorithm...", copied));
   LogInfo("  Algorithm complexity: O(N) - linear time");
   
   // FLAW 4.4 FIX: Two-Pointer Merge algorithm for O(N) timestamp alignment
   int usdx_idx = 0, usdjpy_idx = 0; // Indices for auxiliary tick arrays
   double usdx_bid = 0.0, usdjpy_bid = 0.0; // Current matched prices
   
   int usdx_matches = 0, usdjpy_matches = 0; // Count of successful matches
   int progress_interval = copied / 10; // Report progress every 10%
   if(progress_interval < 1000) progress_interval = 1000; // Minimum 1000 ticks between reports
   
   ulong process_start = GetTickCount64();
   
   for(int i = 0; i < copied; i++) {
      ulong t = ticks[i].time_msc; // Current tick timestamp
      
      // FLAW 4.4 FIX: Two-Pointer Merge for USDX
      if(usdx_available && usdx_copied > 0) {
         int prev_idx = usdx_idx;
         while(usdx_idx < usdx_copied - 1 && usdx_ticks[usdx_idx + 1].time_msc <= t) {
            usdx_idx++;
         }
         if(usdx_idx != prev_idx) usdx_matches++;
         usdx_bid = usdx_ticks[usdx_idx].bid;
      }
      
      // FLAW 4.4 FIX: Two-Pointer Merge for USDJPY
      if(usdjpy_available && usdjpy_copied > 0) {
         int prev_idx = usdjpy_idx;
         while(usdjpy_idx < usdjpy_copied - 1 && usdjpy_ticks[usdjpy_idx + 1].time_msc <= t) {
            usdjpy_idx++;
         }
         if(usdjpy_idx != prev_idx) usdjpy_matches++;
         usdjpy_bid = usdjpy_ticks[usdjpy_idx].bid;
      }
      
      // Use StringFormat for efficient string building
      string row = StringFormat("%lld,%.5f,%.5f,%.5f,%.5f",
                                ticks[i].time_msc,
                                ticks[i].bid,
                                ticks[i].ask,
                                usdx_bid,
                                usdjpy_bid);
      FileWrite(h, row);
      
      // Progress reporting
      if(progress_interval > 0 && (i + 1) % progress_interval == 0) {
         int percent = (int)((double)(i + 1) / copied * 100);
         ulong elapsed = GetTickCount64() - process_start;
         int estimated_total = (int)((double)elapsed / (i + 1) * copied / 1000);
         int estimated_remaining = (int)((double)elapsed / (i + 1) * (copied - i - 1) / 1000);
         LogProgress("Export", i + 1, copied, 
                     StringFormat(" | Elapsed: %ds | ETA: %ds", 
                                  (int)(elapsed / 1000), estimated_remaining));
      }
   }
   
   ulong process_time = GetTickCount64() - process_start;
   
   // === FILE FINALIZATION PHASE ===
   LogSeparator();
   LogInfo("Phase 5: File Finalization");
   FileClose(h);
   LogSuccess("  File closed successfully");
   
   // Calculate file size estimate (approximate)
   long file_size_estimate = copied * 60L; // ~60 bytes per row estimate
   LogInfo(StringFormat("  Estimated file size: ~%.2f MB", (double)file_size_estimate / 1024 / 1024));
   
   // === FINAL SUMMARY ===
   LogSeparator();
   LogInfo("EXECUTION SUMMARY");
   LogSeparator();
   
   ulong total_time = GetTickCount64() - start_time;
   
    LogSuccess(StringFormat("Exported %d ticks to fast/achilles_ticks.csv", copied));
    LogInfo("  Run 'python fast/move_ticks.py' to move file to project directory");
    LogInfo(StringFormat("  Main symbol (%s): %d ticks", _Symbol, copied));
   
   if(usdx_available) {
      LogInfo(StringFormat("  USDX (%s): %d ticks loaded, %d timestamp matches", 
                           USDX_Symbol, usdx_copied, usdx_matches));
   } else {
      LogInfo("  USDX: Not available (used placeholder 0.0)");
   }
   
   if(usdjpy_available) {
      LogInfo(StringFormat("  USDJPY (%s): %d ticks loaded, %d timestamp matches", 
                           USDJPY_Symbol, usdjpy_copied, usdjpy_matches));
   } else {
      LogInfo("  USDJPY: Not available (used placeholder 0.0)");
   }
   
   LogInfo(StringFormat("Processing time: %llu ms (%.2f seconds)", process_time, (double)process_time / 1000));
   LogInfo(StringFormat("Throughput: %.0f ticks/second", (double)copied / (process_time / 1000.0)));
   LogInfo(StringFormat("Total script execution time: %llu ms (%.2f seconds)", total_time, (double)total_time / 1000));
   
   LogSeparator();
   LogSuccess("SCRIPT COMPLETED SUCCESSFULLY");
   LogSeparator();
}
```

```python
import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
import tf2onnx
import os
import argparse

# 1. SETUP & PATHS
# Adjust these paths to where your files actually are on your computer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TICK_DATA = os.path.join(SCRIPT_DIR, 'achilles_ticks.csv')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train Achilles neural network model')
parser.add_argument('--tick-density', type=int, default=144,
                    help='Tick bar density - number of ticks per bar (default: 144)')
args = parser.parse_args()

TICK_DENSITY = args.tick_density
OUTPUT_ONNX_MODEL = os.path.join(SCRIPT_DIR, f'achilles_{TICK_DENSITY}.onnx') 

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
```

```cpp
//+------------------------------------------------------------------+
//|                                  Live_Achilles_TickBar.mq5       |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>

#resource "\\Files\\achilles_144.onnx" as uchar model_buffer[]

input int    TICK_DENSITY  = 144;      // Variable Bar Density
input double TP_POINTS     = 1.44;     // Target Profit
input double SL_POINTS     = 0.0;      // 0 = ABSOLUTELY NO STOP LOSS
input string USDX_Symbol   = "$USDX";  // USD Index Symbol
input string USDJPY_Symbol = "USDJPY"; // Yield Proxy Symbol
input int    MAGIC_NUMBER  = 144144;   // FLAW 4.3 FIX: Magic Number for position identification

float means[35] = {0.0f}; // ⚠️ PASTE FROM PYTHON
float stds[35]  = {1.0f}; // ⚠️ PASTE FROM PYTHON

CTrade trade;
long onnx = INVALID_HANDLE;
float input_data[4200]; 
// FLAW 4.2 FIX: Ring Buffer implementation for O(1) operations
// Arrays sized at 512 (power of 2) for efficient modulo via bitwise AND
#define RING_SIZE 512
#define RING_MASK 511  // RING_SIZE - 1 for fast modulo
double o_a[RING_SIZE], h_a[RING_SIZE], l_a[RING_SIZE], c_a[RING_SIZE];
double s_a[RING_SIZE], d_a[RING_SIZE], dx_a[RING_SIZE], jp_a[RING_SIZE];
int head_index = 0;  // Points to next write position
int ticks_in_bar = 0, bars = 0;
double b_open, b_high, b_low, b_spread;
ulong b_start_time;

// FLAW 2.1 FIX: Global EMA running state variables
// EMA formula: EMA_t = (Price_t - EMA_{t-1}) * K + EMA_{t-1}, where K = 2/(period+1)
double ema9_state = 0.0, ema18_state = 0.0, ema27_state = 0.0, ema54_state = 0.0, ema144_state = 0.0;
bool ema_initialized = false;

// Historical EMA arrays for 120-bar lookback in Predict()
// FLAW 4.2 FIX: Use RING_SIZE for ring buffer compatibility
double ema9_a[RING_SIZE], ema18_a[RING_SIZE], ema27_a[RING_SIZE], ema54_a[RING_SIZE], ema144_a[RING_SIZE];

// FLAW 2.2 FIX: MACD Signal line state (EMA of MACD line with period 9)
// MACD = EMA9 - EMA18, Signal = EMA(MACD, 9), Histogram = MACD - Signal
double macd_signal_state = 0.0;
double macd_a[RING_SIZE], macd_signal_a[RING_SIZE], macd_hist_a[RING_SIZE];

// FLAW 2.2 FIX: CCI state arrays for running calculations
// CCI = (TP - SMA_TP) / (0.015 * Mean_Deviation), where TP = (H+L+C)/3
double cci9_a[RING_SIZE], cci18_a[RING_SIZE], cci27_a[RING_SIZE];

int OnInit() {
   if(!SymbolSelect(USDX_Symbol, true) || !SymbolSelect(USDJPY_Symbol, true)) return(INIT_FAILED);
   onnx = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   if(onnx == INVALID_HANDLE) return(INIT_FAILED);
   long in[]={1,120,35}; OnnxSetInputShape(onnx,0,in);
   long out[]={1,3}; OnnxSetOutputShape(onnx,0,out);
   
   // FLAW 4.3 FIX: Set magic number for trade identification
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   
   // FLAW 2.1 FIX: Bootstrap EMAs with historical data warm-up
   // Pre-load historical bars to let EMAs "warm up" to match Python output
   InitializeEMAs();
   
   return(INIT_SUCCEEDED);
}

// FLAW 2.1 FIX: Initialize EMAs by pre-loading historical tick data
// FLAW 4.1 FIX: Fully bootstrap all arrays so EA is ready to trade on Tick 1
// FLAW 4.2 FIX: Use ring buffer for O(1) operations
void InitializeEMAs() {
   MqlTick ticks[];
   // FLAW 4.1 FIX: Use COPY_TICKS_ALL to get both bid and ask for spread calculation
   int count = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, 150000); // Get ~1000+ bars worth of ticks
   if(count < TICK_DENSITY * 270) {
      Print("Warning: Not enough historical ticks for full bootstrap. Need ", TICK_DENSITY * 270, ", got ", count);
      // Continue with what we have instead of returning
   }
   
   // FLAW 4.1 FIX: Also load historical ticks for USDX and USDJPY
   MqlTick usdx_ticks[], usdjpy_ticks[];
   int usdx_count = CopyTicks(USDX_Symbol, usdx_ticks, COPY_TICKS_BID, 0, 150000);
   int usdjpy_count = CopyTicks(USDJPY_Symbol, usdjpy_ticks, COPY_TICKS_BID, 0, 150000);
   
   // Build historical tick-bars
   int bar_count = 0;
   int tick_idx = 0;
   
   // Arrays to store bar data before final placement
   double tmp_o[512], tmp_h[512], tmp_l[512], tmp_c[512], tmp_s[512], tmp_d[512];
   ulong tmp_time[512]; // Bar start time for aligning USDX/USDJPY
   
   while(tick_idx < count && bar_count < RING_SIZE) {
      double bar_open = ticks[tick_idx].bid;
      double bar_high = ticks[tick_idx].bid;
      double bar_low = ticks[tick_idx].bid;
      double bar_spread_sum = 0;
      ulong bar_start_time = ticks[tick_idx].time_msc;
      int ticks_in_this_bar = 0;
      
      while(tick_idx < count && ticks_in_this_bar < TICK_DENSITY) {
         bar_high = MathMax(bar_high, ticks[tick_idx].bid);
         bar_low = MathMin(bar_low, ticks[tick_idx].bid);
         bar_spread_sum += (ticks[tick_idx].ask - ticks[tick_idx].bid);
         ticks_in_this_bar++;
         tick_idx++;
      }
      
      double bar_close = ticks[tick_idx-1].bid;
      ulong bar_end_time = ticks[tick_idx-1].time_msc;
      double bar_spread = (ticks_in_this_bar > 0) ? bar_spread_sum / ticks_in_this_bar : 0;
      double bar_duration = (double)(bar_end_time - bar_start_time); // Duration in milliseconds
      
      // Store in temp arrays (oldest first, will be reversed)
      int idx = bar_count;
      if(idx < RING_SIZE) {
         tmp_o[idx] = bar_open;
         tmp_h[idx] = bar_high;
         tmp_l[idx] = bar_low;
         tmp_c[idx] = bar_close;
         tmp_s[idx] = bar_spread;
         tmp_d[idx] = bar_duration;
         tmp_time[idx] = bar_start_time;
      }
      bar_count++;
   }
   
   // FLAW 4.1 FIX: Align USDX and USDJPY data with primary symbol bars
   // Use bar start time to find corresponding prices
   int usdx_idx = 0, usdjpy_idx = 0;
   
   // FLAW 4.2 FIX: Initialize ring buffer - fill from oldest to newest
   // head_index will point to next write position after initialization
   head_index = 0;
   
   // Fill ring buffer: oldest data at position 0, newest at bar_count-1
   // Then head_index = bar_count % RING_SIZE
   for(int i = 0; i < bar_count && i < RING_SIZE; i++) {
      int src_idx = bar_count - 1 - i; // Reverse: last bar = most recent
      int ring_idx = i; // Fill sequentially in ring buffer
      
      o_a[ring_idx] = tmp_o[src_idx];
      h_a[ring_idx] = tmp_h[src_idx];
      l_a[ring_idx] = tmp_l[src_idx];
      c_a[ring_idx] = tmp_c[src_idx];
      s_a[ring_idx] = tmp_s[src_idx];
      d_a[ring_idx] = tmp_d[src_idx];
      
      // FLAW 4.1 FIX: Find USDX price closest to bar time
      ulong bar_time = tmp_time[src_idx];
      if(usdx_count > 0) {
         while(usdx_idx < usdx_count - 1 && usdx_ticks[usdx_idx].time_msc < bar_time) {
            usdx_idx++;
         }
         dx_a[ring_idx] = usdx_ticks[usdx_idx].bid;
      } else {
         dx_a[ring_idx] = SymbolInfoDouble(USDX_Symbol, SYMBOL_BID);
      }
      
      // FLAW 4.1 FIX: Find USDJPY price closest to bar time
      if(usdjpy_count > 0) {
         while(usdjpy_idx < usdjpy_count - 1 && usdjpy_ticks[usdjpy_idx].time_msc < bar_time) {
            usdjpy_idx++;
         }
         jp_a[ring_idx] = usdjpy_ticks[usdjpy_idx].bid;
      } else {
         jp_a[ring_idx] = SymbolInfoDouble(USDJPY_Symbol, SYMBOL_BID);
      }
   }
   
   // Set head_index to next write position
   head_index = bar_count % RING_SIZE;
   bars = bar_count;
   
   // Initialize EMAs using SMA of oldest 'period' values as seed (matching pandas_ta behavior)
   // With ring buffer, oldest data is at index 0, newest at bar_count-1
   // EMA9: seed with SMA of first 9 closes (oldest 9 bars)
   double sum = 0;
   for(int i = 0; i < 9 && i < bar_count; i++) sum += c_a[i];
   ema9_state = sum / 9.0;
   
   // EMA18: seed with SMA of first 18 closes
   sum = 0;
   for(int i = 0; i < 18 && i < bar_count; i++) sum += c_a[i];
   ema18_state = sum / 18.0;
   
   // EMA27: seed with SMA of first 27 closes
   sum = 0;
   for(int i = 0; i < 27 && i < bar_count; i++) sum += c_a[i];
   ema27_state = sum / 27.0;
   
   // EMA54: seed with SMA of first 54 closes
   sum = 0;
   for(int i = 0; i < 54 && i < bar_count; i++) sum += c_a[i];
   ema54_state = sum / 54.0;
   
   // EMA144: seed with SMA of first 144 closes
   sum = 0;
   for(int i = 0; i < 144 && i < bar_count; i++) sum += c_a[i];
   ema144_state = sum / 144.0;
   
   // FLAW 2.2 FIX: Initialize MACD signal state with SMA of first 9 MACD values
   double initial_macd = ema9_state - ema18_state;
   macd_signal_state = initial_macd;
   
   // Now propagate EMAs through remaining historical data
   // This "warms up" the EMAs to match what pandas_ta would produce
   // Also store EMA values in historical arrays for Predict() lookback
   for(int i = 9; i < bar_count && i < RING_SIZE; i++) {
      double price = c_a[i];
      ema9_state   = (price - ema9_state)   * (2.0/10.0)  + ema9_state;
      ema18_state  = (price - ema18_state)  * (2.0/19.0)  + ema18_state;
      ema27_state  = (price - ema27_state)  * (2.0/28.0)  + ema27_state;
      ema54_state  = (price - ema54_state)  * (2.0/55.0)  + ema54_state;
      ema144_state = (price - ema144_state) * (2.0/145.0) + ema144_state;
      
      // Store EMA values in historical arrays
      ema9_a[i]   = ema9_state;
      ema18_a[i]  = ema18_state;
      ema27_a[i]  = ema27_state;
      ema54_a[i]  = ema54_state;
      ema144_a[i] = ema144_state;
      
      // FLAW 2.2 FIX: Calculate MACD, Signal, and Histogram
      double macd = ema9_state - ema18_state;
      macd_a[i] = macd;
      
      // Signal line: EMA of MACD with period 9 (K = 2/10 = 0.2)
      macd_signal_state = (macd - macd_signal_state) * 0.2 + macd_signal_state;
      macd_signal_a[i] = macd_signal_state;
      macd_hist_a[i] = macd - macd_signal_state;
      
      // FLAW 2.2 FIX: Calculate CCI for periods 9, 18, 27
      cci9_a[i]  = CalcCCI(i, 9);
      cci18_a[i] = CalcCCI(i, 18);
      cci27_a[i] = CalcCCI(i, 27);
   }
   
   ema_initialized = true;
}

void OnDeinit(const int reason) { if(onnx != INVALID_HANDLE) OnnxRelease(onnx); }

// FLAW 4.3 FIX: Check if EA has an open position for this symbol with our magic number
// Returns true if we have an open position, false otherwise
bool HasOpenPosition() {
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0) {
         // Check if position matches our symbol AND magic number
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER) {
            return true;
         }
      }
   }
   return false;
}

void OnTick() {
   MqlTick t; if(!SymbolInfoTick(_Symbol, t)) return;
   if(ticks_in_bar == 0) { b_open=t.bid; b_high=t.bid; b_low=t.bid; b_spread=0; b_start_time=t.time_msc; }
   b_high=MathMax(b_high, t.bid); b_low=MathMin(b_low, t.bid);
   b_spread += (t.ask-t.bid); ticks_in_bar++;

   if(ticks_in_bar >= TICK_DENSITY) {
      Shift(b_open, b_high, b_low, t.bid, b_spread/(double)TICK_DENSITY, (double)(t.time_msc - b_start_time));
      ticks_in_bar = 0;
      // Inference requires 120 bars + 144 bar indicator depth = 264 minimum
      if(bars >= 270) Predict(); 
   }
}

// FLAW 4.2 FIX: O(1) ring buffer push - no array shifting needed
void Shift(double o, double h, double l, double c, double s, double d) {
   // Write new data at head_index position (O(1) operation)
   int idx = head_index;
   o_a[idx]=o; h_a[idx]=h; l_a[idx]=l; c_a[idx]=c; s_a[idx]=s; d_a[idx]=d;
   dx_a[idx]=SymbolInfoDouble(USDX_Symbol, SYMBOL_BID);
   jp_a[idx]=SymbolInfoDouble(USDJPY_Symbol, SYMBOL_BID);
   
   // FLAW 2.1 FIX: Update EMA running states with new close price
   // EMA_t = (Price_t - EMA_{t-1}) * K + EMA_{t-1}
   if(ema_initialized) {
      ema9_state   = (c - ema9_state)   * (2.0/10.0)  + ema9_state;
      ema18_state  = (c - ema18_state)  * (2.0/19.0)  + ema18_state;
      ema27_state  = (c - ema27_state)  * (2.0/28.0)  + ema27_state;
      ema54_state  = (c - ema54_state)  * (2.0/55.0)  + ema54_state;
      ema144_state = (c - ema144_state) * (2.0/145.0) + ema144_state;
      
      // Store current EMA values at current ring buffer position
      ema9_a[idx]   = ema9_state;
      ema18_a[idx]  = ema18_state;
      ema27_a[idx]  = ema27_state;
      ema54_a[idx]  = ema54_state;
      ema144_a[idx] = ema144_state;
      
      // FLAW 2.2 FIX: Update MACD, Signal, and Histogram
      double macd = ema9_state - ema18_state;
      macd_signal_state = (macd - macd_signal_state) * 0.2 + macd_signal_state;
      macd_a[idx] = macd;
      macd_signal_a[idx] = macd_signal_state;
      macd_hist_a[idx] = macd - macd_signal_state;
      
      // FLAW 2.2 FIX: Update CCI values
      cci9_a[idx]  = CalcCCI(idx, 9);
      cci18_a[idx] = CalcCCI(idx, 18);
      cci27_a[idx] = CalcCCI(idx, 27);
   }
   
   // Advance head_index with wrap-around (O(1) operation)
   head_index = (head_index + 1) & RING_MASK;
   bars++;
}

// FLAW 4.2 FIX: Helper function to convert logical index to ring buffer index
// logical 0 = most recent bar, logical 1 = one bar ago, etc.
int RingIdx(int logical) {
   return (head_index - 1 - logical) & RING_MASK;
}

void Predict() {
   for(int i=0; i<120; i++) {
      // FLAW 4.2 FIX: Use ring buffer indexing
      int x = RingIdx(119-i);  // Most recent bar for this sequence position
      int x1 = RingIdx(119-i+1);  // One bar older for comparisons
      float f[35];
      f[0]=(float)MathLog(c_a[x]/(c_a[x1]+1e-8)); 
      f[1]=(float)s_a[x]; 
      f[2]=(float)d_a[x];
      f[3]=(float)((h_a[x]-MathMax(o_a[x],c_a[x]))/(c_a[x]+1e-8)); 
      f[4]=(float)((MathMin(o_a[x],c_a[x])-l_a[x])/(c_a[x]+1e-8));
      f[5]=(float)((h_a[x]-l_a[x])/(c_a[x]+1e-8)); 
      f[6]=(float)((c_a[x]-l_a[x])/(h_a[x]-l_a[x]+1e-8));
      f[7]=CRSI(x,9); f[8]=CRSI(x,18); f[9]=CRSI(x,27);
      f[10]=CATR(x,9); f[11]=CATR(x,18); f[12]=CATR(x,27);
      // FLAW 2.1 FIX: Use global EMA arrays (running state) instead of truncated CEMA
      double e9=ema9_a[x], e18=ema18_a[x], e27=ema27_a[x], e54=ema54_a[x], e144=ema144_a[x];
      // FLAW 2.2 FIX: Use proper MACD values from running state arrays
      // f13 = MACD line, f14 = Signal line, f15 = Histogram
      f[13]=(float)macd_a[x]; f[14]=(float)macd_signal_a[x]; f[15]=(float)macd_hist_a[x];
      f[16]=(float)(e9-c_a[x]); f[17]=(float)(e18-c_a[x]); f[18]=(float)(e27-c_a[x]); 
      f[19]=(float)(e54-c_a[x]); f[20]=(float)(e144-c_a[x]);
      // FLAW 2.2 FIX: Use proper CCI values from running state arrays
      f[21]=(float)cci9_a[x]; f[22]=(float)cci18_a[x]; f[23]=(float)cci27_a[x];
      f[24]=CWPR(x,9); f[25]=CWPR(x,18); f[26]=CWPR(x,27);
      // FLAW 4.2 FIX: Use ring buffer indexing for lookback
      f[27]=(float)(c_a[x]-c_a[RingIdx(119-i+9)]); 
      f[28]=(float)(c_a[x]-c_a[RingIdx(119-i+18)]); 
      f[29]=(float)(c_a[x]-c_a[RingIdx(119-i+27)]);
      f[30]=(float)((dx_a[x]-dx_a[x1])/(dx_a[x1]+1e-8)); 
      f[31]=(float)((jp_a[x]-jp_a[x1])/(jp_a[x1]+1e-8));
      f[32]=CBBW(x,9); f[33]=CBBW(x,18); f[34]=CBBW(x,27);
      for(int k=0; k<35; k++) input_data[i*35+k]=(f[k]-means[k])/(stds[k]+1e-8f);
   }
   float out[3]; OnnxRun(onnx, ONNX_DEFAULT, input_data, out);
   if(out[0]>0.5) return;
   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK), bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);
   // FLAW 4.3 FIX: Use HasOpenPosition() instead of PositionsTotal()==0
   // This ensures we only check positions for this symbol and magic number
   if(out[1]>0.55 && !HasOpenPosition()) Execute(ORDER_TYPE_BUY, ask);
   if(out[2]>0.55 && !HasOpenPosition()) Execute(ORDER_TYPE_SELL, bid);
}

void Execute(ENUM_ORDER_TYPE type, double p) {
   double sl = (SL_POINTS <= 0) ? 0 : (type==ORDER_TYPE_BUY ? p-SL_POINTS : p+SL_POINTS);
   double tp = (type==ORDER_TYPE_BUY ? p+TP_POINTS : p-TP_POINTS);
   if(type==ORDER_TYPE_BUY) trade.Buy(0.01,_Symbol,p,sl,tp); else trade.Sell(0.01,_Symbol,p,sl,tp);
}

// FLAW 4.2 FIX: All indicator functions now use ring buffer indexing
// x is already a ring buffer index, x+i needs to be wrapped
float CRSI(int x, int p) { 
   double u=0, d=0; 
   for(int i=0; i<p; i++) { 
      int idx0 = (x + i) & RING_MASK;
      int idx1 = (x + i + 1) & RING_MASK;
      double df=c_a[idx0]-c_a[idx1]; 
      if(df>0) u+=df; else d-=df; 
   } 
   return (d==0)?100:(float)(100-(100/(1+u/(d+1e-8)))); 
}

float CATR(int x, int p) { 
   double s=0; 
   for(int i=0; i<p; i++) { 
      int idx0 = (x + i) & RING_MASK;
      int idx1 = (x + i + 1) & RING_MASK;
      s+=MathMax(h_a[idx0]-l_a[idx0], MathAbs(h_a[idx0]-c_a[idx1])); 
   } 
   return (float)(s/p); 
}

// FLAW 2.1 FIX: CEMA removed - now using global EMA running state with warm-up
float CWPR(int x, int p) { 
   double h=h_a[x], l=l_a[x]; 
   for(int i=1; i<p; i++) { 
      int idx = (x + i) & RING_MASK;
      h=MathMax(h,h_a[idx]); 
      l=MathMin(l,l_a[idx]); 
   } 
   return (h==l)?0:(float)(-100*(h-c_a[x])/(h-l+1e-8)); 
}

// FLAW 2.3 FIX: Use Bessel's correction (divide by p-1 instead of p) for sample standard deviation
float CBBW(int x, int p) { 
   double s=0, sq=0; 
   for(int i=0; i<p; i++) {
      int idx = (x + i) & RING_MASK;
      s+=c_a[idx]; 
   }
   double m=s/p; 
   for(int i=0; i<p; i++) {
      int idx = (x + i) & RING_MASK;
      sq+=MathPow(c_a[idx]-m,2); 
   }
   return (float)((MathSqrt(sq/(p-1))*4)/(m+1e-8)); 
}

// FLAW 2.2 FIX: CCI (Commodity Channel Index) implementation
// CCI = (TP - SMA_TP) / (0.015 * Mean_Deviation)
// TP (Typical Price) = (High + Low + Close) / 3
// FLAW 4.2 FIX: Updated for ring buffer indexing
float CalcCCI(int x, int p) {
   // Calculate Typical Prices and SMA of TP
   double tp_sum = 0;
   double tp[];
   ArrayResize(tp, p);
   
   for(int i = 0; i < p; i++) {
      int idx = (x + i) & RING_MASK;
      tp[i] = (h_a[idx] + l_a[idx] + c_a[idx]) / 3.0;
      tp_sum += tp[i];
   }
   
   double sma_tp = tp_sum / p;
   
   // Calculate Mean Deviation
   double mean_dev = 0;
   for(int i = 0; i < p; i++) {
      mean_dev += MathAbs(tp[i] - sma_tp);
   }
   mean_dev /= p;
   
   // CCI formula
   double current_tp = (h_a[x] + l_a[x] + c_a[x]) / 3.0;
   if(mean_dev < 1e-10) return 0;  // Avoid division by zero
   return (float)((current_tp - sma_tp) / (0.015 * mean_dev));
}
```
