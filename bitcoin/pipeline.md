data.mq5
```cpp
#property script_show_inputs
input int ticks_to_export = 2160000;
input int days_lookback   = 180;     
input int chunk_size      = 100000;

void OnStart() {
   Print("[INFO] Initializing Absolute-Chronological Tick Export...");
   int h = FileOpen("bitcoin_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) return;
   
   FileWrite(h, "time_msc", "bid", "ask", "vol"); 
   MqlTick ticks[];
   int total_copied = 0;
   
   ulong anchor_msc = ((ulong)TimeCurrent() - ((ulong)days_lookback * 86400ull)) * 1000ull;
   ulong last_time  = anchor_msc;
   
   while(total_copied < ticks_to_export) {
      int to_copy = MathMin(chunk_size, ticks_to_export - total_copied);
      int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, last_time, to_copy);
      
      if(copied <= 0) break;
      last_time = ticks[copied-1].time_msc + 1; 
      
      for(int i = 0; i < copied; i++) {
         if(ticks[i].bid <= 0.0 || ticks[i].ask < ticks[i].bid) continue;
         // Ensure volume is at least 0.01 to prevent math errors downstream
         double v = (ticks[i].volume > 0) ? (double)ticks[i].volume : 0.01;
         FileWrite(h, ticks[i].time_msc, ticks[i].bid, ticks[i].ask, v);
      }
      total_copied += copied;
      if(last_time >= (ulong)TimeCurrent() * 1000ull) break;
   }
   FileClose(h);
   PrintFormat("✅ Exported %d ticks.", total_copied);
}
```

nn.py
```python
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

def get_symmetric_labels(df, tp_mult=2.7, sl_mult=0.54):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    atr = ta.atr(df.high, df.low, df.close, length=18).values
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df)-TARGET_HORIZON):
        if np.isnan(atr[i]): continue
        b_tp, b_sl = c[i] + (tp_mult*atr[i]), c[i] - (sl_mult*atr[i])
        s_tp, s_sl = c[i] - (tp_mult*atr[i]), c[i] + (sl_mult*atr[i])

        buy_done, sell_done = False, False
        for j in range(i+1, i+TARGET_HORIZON):
            if not buy_done:
                if hi[j] >= b_tp: labels[i] = 1; buy_done = True
                elif lo[j] <= b_sl: buy_done = True
            if not sell_done:
                if lo[j] <= s_tp: labels[i] = 2; sell_done = True
                elif hi[j] >= s_sl: sell_done = True
            if buy_done and sell_done: break
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
median = np.median(X[:split], axis=0)
iqr = np.percentile(X[:split], 75, axis=0) - np.percentile(X[:split], 25, axis=0)
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
cw = compute_class_weight('balanced', classes=np.unique(y_seq[:split]), y=y_seq[:split])
assert split > 0 and split < len(X_seq), f"Split {split} out of range for X_seq len {len(X_seq)}"
model.fit(X_seq[:split].reshape(-1, 2040), y_seq[:split], 
          validation_data=(X_seq[split:].reshape(-1, 2040), y_seq[split:]),
          epochs=100, batch_size=64, class_weight=dict(enumerate(cw)),
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# Export
spec = (tf.TensorSpec((1, 2040), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open("bitcoin_144.onnx", "wb") as f: f.write(model_proto.SerializeToString())

print(f"Medians: {list(median)}")
print(f"IQRs: {list(iqr)}")
```

live.mq5
```cpp
﻿#include <Trade\Trade.mqh>
#resource "\\Experts\\nn\\bitcoin_144.onnx" as uchar model_buffer[]

input int TICK_DENSITY = 144;
long onnx_handle = INVALID_HANDLE;
CTrade trade;

// PASTE FROM PYTHON OUTPUT
float medians[17] = {0.0f...}; 
float iqrs[17]    = {1.0f...};

struct Bar { 
   double o, h, l, c, v, spread, tvwp, atr18, macd_ema12, macd_ema26, macd_sig; 
   ulong time_msc; 
};
Bar history[200];
Bar cur_b;
int ticks_in_bar = 0;
float input_data[2040];
float output_data[3];

int OnInit() {
   onnx_handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   const long in_shape[] = {1, 2040};
   const long out_shape[] = {1, 3};
   OnnxSetInputShape(onnx_handle, 0, in_shape);
   OnnxSetOutputShape(onnx_handle, 0, out_shape);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
}

void OnTick() {
   MqlTick t; 
   if(!SymbolInfoTick(_Symbol, t)) return;
   
   if(ticks_in_bar == 0) {
      cur_b.o = t.bid; cur_b.h = t.bid; cur_b.l = t.bid; cur_b.v = 0; cur_b.spread = 0; cur_b.time_msc = t.time_msc;
   }
   cur_b.h = MathMax(cur_b.h, t.bid); cur_b.l = MathMin(cur_b.l, t.bid); cur_b.c = t.bid;
   cur_b.v += (t.volume > 0 ? (double)t.volume : 0.01);
   cur_b.spread += (t.ask - t.bid);
   ticks_in_bar++;

   if(ticks_in_bar >= TICK_DENSITY) {
      cur_b.spread /= TICK_DENSITY;
      UpdateIndicators(cur_b);
      for(int i=199; i>0; i--) history[i] = history[i-1];
      history[0] = cur_b;
      ticks_in_bar = 0;
      if(history[149].c > 0) Predict();
   }
}

void UpdateIndicators(Bar &b) {
   Bar p = history[0];
   if(p.c <= 0) { // EMA COLD START FIX
      b.macd_ema12 = b.c; b.macd_ema26 = b.c; b.macd_sig = 0; b.atr18 = b.h-b.l; return;
   }
   double tr = MathMax(b.h-b.l, MathMax(MathAbs(b.h-p.c), MathAbs(b.l-p.c)));
   b.atr18 = (tr - p.atr18)/18.0 + p.atr18;
   b.macd_ema12 = (b.c - p.macd_ema12)*(2.0/13.0) + p.macd_ema12;
   b.macd_ema26 = (b.c - p.macd_ema26)*(2.0/27.0) + p.macd_ema26;
   double macd_raw = b.macd_ema12 - b.macd_ema26;
   b.macd_sig = (macd_raw - p.macd_sig)*(2.0/10.0) + p.macd_sig;
   
   double p_sum=0, v_sum=0;
   for(int i=0; i<143; i++) { p_sum+=(history[i].c*history[i].v); v_sum+=history[i].v; }
   b.tvwp = (p_sum + b.c*b.v)/(v_sum + b.v + 1e-8);
}

void Predict() {
   for(int i=0; i<120; i++) {
      int h = 119 - i;
      float f[17];
      double cl = history[h].c;
      // DOW FIX: Thursday is 3 in Python's dayofweek
      double utc_h = (double)((history[h].time_msc / 3600000) % 24);
      double utc_d = (double)(((history[h].time_msc / 86400000) + 3) % 7);

      f[0]=(float)MathLog(cl/history[h+1].c); f[1]=(float)history[h].spread;
      f[2]=(float)((history[h].time_msc - history[h+1].time_msc)/1000.0);
      f[3]=(float)((history[h].h-MathMax(history[h].o,cl))/cl);
      f[4]=(float)((MathMin(history[h].o,cl)-history[h].l)/cl);
      f[5]=(float)((history[h].h-history[h].l)/cl);
      f[6]=(float)((cl-history[h].l)/(history[h].h-history[h].l+1e-8));
      double macd = history[h].macd_ema12 - history[h].macd_ema26;
      f[7]=(float)(macd/cl); f[8]=(float)(history[h].macd_sig/cl); f[9]=(float)((macd-history[h].macd_sig)/cl);
      f[10]=(float)(history[h].atr18/cl);
      f[11]=(float)MathSin(2*M_PI*utc_h/24.0); f[12]=(float)MathCos(2*M_PI*utc_h/24.0);
      f[13]=(float)MathSin(2*M_PI*utc_d/7.0); f[14]=(float)MathCos(2*M_PI*utc_d/7.0);
      f[15]=(float)MathLog(history[h].v + 1.0); f[16]=(float)((cl-history[h].tvwp)/cl);

      for(int k=0; k<17; k++) {
         float scaled = (f[k] - medians[k]) / iqrs[k];
         input_data[i*17+k] = MathMax(-10.0f, MathMin(10.0f, scaled)); // CLIPPING FIX
      }
   }
   if(OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
      int sig = ArrayMaximum(output_data);
      if(sig > 0 && output_data[sig] > 0.75) Execute(sig);
   }
}

void Execute(int sig) {
   if(PositionSelect(_Symbol)) return;
   double p   = (sig==1) ? SymbolInfoDouble(_Symbol,SYMBOL_ASK) : SymbolInfoDouble(_Symbol,SYMBOL_BID);
   double sl  = (sig==1) ? (p - history[0].atr18*0.54) : (p + history[0].atr18*0.54);
   double tp  = (sig==1) ? (p + history[0].atr18*2.7)  : (p - history[0].atr18*2.7);
   
   // NEW: validate stops are non-degenerate
   double min_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(MathAbs(p - sl) < min_dist || MathAbs(tp - p) < min_dist) {
      Print("[WARN] Stop/TP too close to price, skipping trade.");
      return;
   }
   trade.PositionOpen(_Symbol,(sig==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL),0.1,p,sl,tp);
}
```
