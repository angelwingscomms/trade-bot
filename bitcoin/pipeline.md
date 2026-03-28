data.mq5
```cpp
#property script_show_inputs
input int ticks_to_export = 2160000;

void OnStart() {
   Print("[INFO] Starting Microstructure Tick Export...");
   MqlTick ticks[];
   // ticks_to_export is int, CopyTicks expects uint for count
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, (uint)ticks_to_export);
   
   if(copied <= 0) {
      Print("❌ Error: No ticks copied. Check Symbol name and History.");
      return;
   }
   
   int h = FileOpen("fast/bitcoin_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) return;
   
   FileWrite(h, "time_msc", "bid", "ask", "vol"); 
   
   for(int i = 0; i < copied; i++) {
      if(ticks[i].bid <= 0.0 || ticks[i].ask <= 0.0) continue;
      
      // FIX: Use .volume instead of .tick_volume. 
      // Add fallback to 1.0 if volume is not provided by broker.
      double v = (ticks[i].volume > 0) ? (double)ticks[i].volume : 1.0;
      
      FileWrite(h, ticks[i].time_msc, ticks[i].bid, ticks[i].ask, v);
   }
   FileClose(h);
   PrintFormat("✅ Exported %d ticks with Micro-Volume data.", copied);
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
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TICK_DATA = os.path.join(SCRIPT_DIR, 'bitcoin_ticks.csv')
TICK_DENSITY = 144
OUTPUT_ONNX_MODEL = os.path.join(SCRIPT_DIR, f'bitcoin_{TICK_DENSITY}.onnx') 

if not os.path.exists(INPUT_TICK_DATA):
    print(f"❌ Error: {INPUT_TICK_DATA} not found.")
    exit()

print("Loading and preparing Microstructure Bars...")
df_t = pd.read_csv(INPUT_TICK_DATA)
df_t['vol'] = df_t['vol'].replace(0, 1.0) 
df_t['bar_id'] = np.arange(len(df_t)) // TICK_DENSITY

df = df_t.groupby('bar_id').agg({
    'bid': ['first', 'max', 'min', 'last'],
    'vol': 'sum',
    'time_msc': 'first'
})
df.columns = ['open', 'high', 'low', 'close', 'volume', 'time_open']
df['spread'] = df_t.groupby('bar_id').apply(lambda x: (x['ask']-x['bid']).mean()).values

print("Engineering 39 Institutional Features...")
df['tpv'] = df['close'] * df['volume']
df['tvwp'] = df['tpv'].rolling(144).sum() / (df['volume'].rolling(144).sum() + 1e-8)
df['f38'] = (df['close'] - df['tvwp']) / df['close'] 
df['dt'] = pd.to_datetime(df['time_open'], unit='ms')
df['f33'] = np.sin(2 * np.pi * df['dt'].dt.hour / 24)
df['f34'] = np.cos(2 * np.pi * df['dt'].dt.hour / 24)
df['f35'] = np.sin(2 * np.pi * df['dt'].dt.dayofweek / 7)
df['f36'] = np.cos(2 * np.pi * df['dt'].dt.dayofweek / 7)
df['f37'] = np.log(df['volume'] + 1)
df['f0'] = np.log(df['close'] / df['close'].shift(1))
df['f1'] = df['spread']
df['f2'] = df['dt'].diff().dt.total_seconds().fillna(0) / 1000.0
df['f3'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
df['f4'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
df['f5'] = (df['high'] - df['low']) / df['close']
df['f6'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

for p, f_idx in zip([9, 18, 27], [7, 8, 9]): df[f'f{f_idx}'] = ta.rsi(df['close'], length=p)
for p, f_idx in zip([9, 18, 27], [10, 11, 12]): df[f'f{f_idx}'] = ta.atr(df['high'], df['low'], df['close'], length=p) / df['close']
m = ta.macd(df['close'], 12, 26, 9)
df['f13'], df['f14'], df['f15'] = m.iloc[:, 0]/df['close'], m.iloc[:, 2]/df['close'], m.iloc[:, 1]/df['close']

for p, f_idx in zip([9, 18, 27, 54, 144],[16, 17, 18, 19, 20]): df[f'f{f_idx}'] = (ta.ema(df['close'], p) - df['close']) / df['close']
for p, f_idx in zip([9, 18, 27], [21, 22, 23]): df[f'f{f_idx}'] = ta.cci(df['high'], df['low'], df['close'], p)
for p, f_idx in zip([9, 18, 27], [24, 25, 26]): df[f'f{f_idx}'] = ta.willr(df['high'], df['low'], df['close'], p)
for p, f_idx in zip([9, 18, 27],[27, 28, 29]): df[f'f{f_idx}'] = df['close'].diff(p) / df['close']
for p, f_idx in zip([9, 18, 27],[30, 31, 32]):
    bb = ta.bbands(df['close'], length=p)
    df[f'f{f_idx}'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df['close']

df.dropna(inplace=True)

def get_labels(df):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    atr = ta.atr(df.high, df.low, df.close, length=18).values
    t = np.zeros(len(df), dtype=int)
    for i in range(len(df)-30):
        if np.isnan(atr[i]): continue
        up, lw = c[i]+(2.7*atr[i]), c[i]-(0.54*atr[i])
        for j in range(i+1, i+31):
            if hi[j] >= up: t[i] = 1; break 
            if lo[j] <= lw: t[i] = 2; break 
    return t

df['target'] = get_labels(df)
features = [f'f{i}' for i in range(39)]
X, y = df[features].values, df['target'].values

train_end = int(len(X) * 0.7)
median = np.median(X[:train_end], axis=0)
iqr = np.percentile(X[:train_end], 75, axis=0) - np.percentile(X[:train_end], 25, axis=0) + 1e-8
X_s = (X - median) / iqr

X_seq, y_seq = [], []
for i in range(len(X_s)-120):
    X_seq.append(X_s[i:i+120])
    y_seq.append(y[i+119])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# 7. TCN MODEL (RELU FOR ONNX COMPATIBILITY)
def tcn_block(x, filters, dilation):
    shortcut = layers.Conv1D(filters, 1, padding='same')(x)
    # Changed 'gelu' to 'relu' to avoid Erfc operator issues in MT5
    x = layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, 3, padding='causal', dilation_rate=dilation, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([shortcut, x])

inp = Input(shape=(4680,), name="input")
x = layers.Reshape((120, 39))(inp)
for d in [1, 2, 4, 8, 16]: x = tcn_block(x, 64, d)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(3, activation='softmax', name="output")(x)

model = Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss='sparse_categorical_crossentropy')

# 8. TRAINING
split = int(len(X_seq) * 0.85)
X_train, X_val = X_seq[:split].reshape(-1, 4680), X_seq[split:].reshape(-1, 4680)
y_train, y_val = y_seq[:split], y_seq[split:]

cw = compute_class_weight('balanced', classes=np.unique(y_seq), y=y_seq)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

model.fit(X_train, y_train, validation_data=(X_val, y_val), 
          epochs=144, batch_size=64, class_weight=dict(enumerate(cw)), callbacks=callbacks)

# 9. EXPORT (OPSET 13)
spec = (tf.TensorSpec((None, 4680), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(OUTPUT_ONNX_MODEL, "wb") as f: f.write(model_proto.SerializeToString())

print("\n--- NEW PARAMETERS ---")
print(f"float medians[39] = {{{', '.join([f'{m:.8f}f' for m in median])}}};")
print(f"float iqrs[39]    = {{{', '.join([f'{s:.8f}f' for s in iqr])}}};")
```

live.mq5
```cpp
﻿#include <Trade\Trade.mqh>

// Ensure the new .onnx file is in this exact directory
#resource "\\Experts\\nn\\bitcoin\\bitcoin_144.onnx" as uchar model_buffer[]

input int    TICK_DENSITY  = 144;      
input double TP_MULTIPLIER = 2.7;      
input double SL_MULTIPLIER = 0.54;     
input int    MAGIC_NUMBER  = 144144;   

// --- UPDATE THESE FROM nn.py OUTPUT ---
float medians[39] = {-0.00000706f, 27.00000000f, 0.08866700f, 0.00023250f, 0.00023166f, 0.00110964f, 0.50314465f, 50.11774674f, 49.96337380f, 49.98915654f, 0.00118782f, 0.00119898f, 0.00121380f, -0.00000017f, 0.00000290f, -0.00000432f, 0.00000795f, -0.00000911f, -0.00001203f, 0.00000781f, -0.00005802f, -40339.64831130f, -3706.85532102f, 11088.86575694f, -50.19305019f, -50.85714286f, -50.36036036f, -0.00002190f, 0.00000722f, -0.00003368f, 0.00331873f, 0.00460767f, 0.00562355f, 0.00000000f, -0.25881905f, 0.43388374f, -0.22252093f, 4.97673374f, -0.00003784f};
float iqrs[39]    = {0.00098919f, 0.14583334f, 0.04820751f, 0.00034463f, 0.00034863f, 0.00069907f, 0.56836275f, 19.92276396f, 13.68942723f, 10.99076822f, 0.00041188f, 0.00038550f, 0.00037337f, 0.00115702f, 0.00109603f, 0.00035454f, 0.00130605f, 0.00197353f, 0.00245342f, 0.00352335f, 0.00594199f, 92184.52250596f, 55292.12940509f, 42884.82809495f, 54.29349140f, 52.84053607f, 52.70318415f, 0.00301596f, 0.00435615f, 0.00544657f, 0.00243473f, 0.00318811f, 0.00385159f, 1.41421357f, 1.20710679f, 1.21571523f, 1.52445868f, 0.00000001f, 0.00688223f};

// Handles
int hRSI9, hRSI18, hRSI27, hATR9, hATR18, hATR27, hMACD, hEMA9, hEMA18, hEMA27, hEMA54, hEMA144, hCCI9, hCCI18, hCCI27, hWPR9, hWPR18, hWPR27, hBB9, hBB18, hBB27;
long onnx_handle = INVALID_HANDLE;
CTrade trade;

// Buffers
float input_data[4680]; 
float output_data[3];   
datetime last_loss_time = 0;

struct Bar { double o, h, l, c, v, spread, tvwp; datetime time; };
Bar history[200]; 
int ticks_in_bar = 0;
Bar cur_b;

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit() {
   if(ArraySize(model_buffer) == 0) {
      Print("❌ FATAL: ONNX model buffer is empty.");
      return(INIT_FAILED);
   }

   onnx_handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) {
      Print("❌ FATAL: OnnxCreateFromBuffer failed. Error: ", GetLastError());
      return(INIT_FAILED);
   }

   const long in_shape[] = {1, 4680};
   const long out_shape[] = {1, 3};
   
   if(!OnnxSetInputShape(onnx_handle, 0, in_shape)) {
      Print("❌ FATAL: OnnxSetInputShape failed. Error: ", GetLastError());
      return(INIT_FAILED);
   }
   if(!OnnxSetOutputShape(onnx_handle, 0, out_shape)) {
      Print("❌ FATAL: OnnxSetOutputShape failed. Error: ", GetLastError());
      return(INIT_FAILED);
   }

   // Indicators
   hRSI9 = iRSI(_Symbol,_Period,9,PRICE_CLOSE); hRSI18 = iRSI(_Symbol,_Period,18,PRICE_CLOSE); hRSI27 = iRSI(_Symbol,_Period,27,PRICE_CLOSE);
   hATR18 = iATR(_Symbol,_Period,18); hATR9 = iATR(_Symbol,_Period,9); hATR27 = iATR(_Symbol,_Period,27);
   hMACD = iMACD(_Symbol,_Period,12,26,9,PRICE_CLOSE);
   hEMA9 = iMA(_Symbol,_Period,9,0,MODE_EMA,PRICE_CLOSE); hEMA18 = iMA(_Symbol,_Period,18,0,MODE_EMA,PRICE_CLOSE);
   hEMA27 = iMA(_Symbol,_Period,27,0,MODE_EMA,PRICE_CLOSE); hEMA54 = iMA(_Symbol,_Period,54,0,MODE_EMA,PRICE_CLOSE);
   hEMA144 = iMA(_Symbol,_Period,144,0,MODE_EMA,PRICE_CLOSE);
   hCCI9 = iCCI(_Symbol,_Period,9,PRICE_TYPICAL); hCCI18 = iCCI(_Symbol,_Period,18,PRICE_TYPICAL); hCCI27 = iCCI(_Symbol,_Period,27,PRICE_TYPICAL);
   hWPR9 = iWPR(_Symbol,_Period,9); hWPR18 = iWPR(_Symbol,_Period,18); hWPR27 = iWPR(_Symbol,_Period,27);
   hBB9 = iBands(_Symbol,_Period,9,0,2.0,PRICE_CLOSE); hBB18 = iBands(_Symbol,_Period,18,0,2.0,PRICE_CLOSE); hBB27 = iBands(_Symbol,_Period,27,0,2.0,PRICE_CLOSE);

   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   Print("✅ ONNX Pipeline Ready (ReLU Mode).");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Tick Processing                                                  |
//+------------------------------------------------------------------+
void OnTick() {
   MqlTick t; if(!SymbolInfoTick(_Symbol, t)) return;
   if(ticks_in_bar == 0) {
      cur_b.o = t.bid; cur_b.h = t.bid; cur_b.l = t.bid; cur_b.v = 0; cur_b.spread = 0; cur_b.time = TimeCurrent();
   }
   cur_b.h = MathMax(cur_b.h, t.bid); cur_b.l = MathMin(cur_b.l, t.bid); cur_b.c = t.bid;
   double tv = (t.volume > 0) ? (double)t.volume : 1.0;
   cur_b.v += tv; cur_b.spread += (t.ask - t.bid);
   ticks_in_bar++;

   if(ticks_in_bar >= TICK_DENSITY) {
      cur_b.spread /= TICK_DENSITY;
      for(int i=199; i>0; i--) history[i] = history[i-1];
      history[0] = cur_b;
      
      double sum_pv = 0, sum_v = 0;
      for(int j=0; j<144; j++) { sum_pv += (history[j].c * history[j].v); sum_v += history[j].v; }
      history[0].tvwp = sum_pv / (sum_v + 1e-8);
      
      ticks_in_bar = 0;
      static int bc = 0; bc++;
      if(bc >= 150) Predict();
   }
}

//+------------------------------------------------------------------+
//| Inference & Execution                                             |
//+------------------------------------------------------------------+
void Predict() {
   if(onnx_handle == INVALID_HANDLE) return;
   if(TimeCurrent() < last_loss_time + 3600) return; 

   double r9[120], r18[120], r27[120], a9[120], a18[120], a27[120], mm[120], ms[120], e9[120], e18[120], e27[120], e54[120], e144[120], c9[120], c18[120], c27[120], w9[120], w18[120], w27[120], b9u[120], b9l[120], b18u[120], b18l[120], b27u[120], b27l[120];

   if(CopyBuffer(hRSI9,0,0,120,r9)<120 || CopyBuffer(hMACD,0,0,120,mm)<120 || CopyBuffer(hMACD,1,0,120,ms)<120) return;
   CopyBuffer(hRSI18,0,0,120,r18); CopyBuffer(hRSI27,0,0,120,r27);
   CopyBuffer(hATR9,0,0,120,a9); CopyBuffer(hATR18,0,0,120,a18); CopyBuffer(hATR27,0,0,120,a27);
   CopyBuffer(hEMA9,0,0,120,e9); CopyBuffer(hEMA18,0,0,120,e18); CopyBuffer(hEMA27,0,0,120,e27);
   CopyBuffer(hEMA54,0,0,120,e54); CopyBuffer(hEMA144,0,0,120,e144);
   CopyBuffer(hCCI9,0,0,120,c9); CopyBuffer(hCCI18,0,0,120,c18); CopyBuffer(hCCI27,0,0,120,c27);
   CopyBuffer(hWPR9,0,0,120,w9); CopyBuffer(hWPR18,0,0,120,w18); CopyBuffer(hWPR27,0,0,120,w27);
   CopyBuffer(hBB9,1,0,120,b9u); CopyBuffer(hBB9,2,0,120,b9l);
   CopyBuffer(hBB18,1,0,120,b18u); CopyBuffer(hBB18,2,0,120,b18l);
   CopyBuffer(hBB27,1,0,120,b27u); CopyBuffer(hBB27,2,0,120,b27l);

   for(int i=0; i<120; i++) {
      int h = 119 - i; int b = 119 - i;
      float f[39];
      double cl = history[h].c;
      MqlDateTime dt; TimeToStruct(history[h].time, dt);

      f[0]=(float)MathLog(cl/(history[h+1].c+1e-8)); f[1]=(float)history[h].spread; f[2]=(float)(history[h].time-history[h+1].time);
      f[3]=(float)((history[h].h-MathMax(history[h].o,cl))/cl); f[4]=(float)((MathMin(history[h].o,cl)-history[h].l)/cl);
      f[5]=(float)((history[h].h-history[h].l)/cl); f[6]=(float)((cl-history[h].l)/(history[h].h-history[h].l+1e-8));
      f[7]=(float)r9[b]; f[8]=(float)r18[b]; f[9]=(float)r27[b];
      f[10]=(float)(a9[b]/cl); f[11]=(float)(a18[b]/cl); f[12]=(float)(a27[b]/cl);
      f[13]=(float)(mm[b]/cl); f[14]=(float)(ms[b]/cl); f[15]=(float)((mm[b]-ms[b])/cl);
      f[16]=(float)((e9[b]-cl)/cl); f[17]=(float)((e18[b]-cl)/cl); f[18]=(float)((e27[b]-cl)/cl);
      f[19]=(float)((e54[b]-cl)/cl); f[20]=(float)((e144[b]-cl)/cl);
      f[21]=(float)c9[b]; f[22]=(float)c18[b]; f[23]=(float)c27[b];
      f[24]=(float)w9[b]; f[25]=(float)w18[b]; f[26]=(float)w27[b];
      f[27]=(float)((cl-history[h+9].c)/cl); f[28]=(float)((cl-history[h+18].c)/cl); f[29]=(float)((cl-history[h+27].c)/cl);
      f[30]=(float)((b9u[b]-b9l[b])/cl); f[31]=(float)((b18u[b]-b18l[b])/cl); f[32]=(float)((b27u[b]-b27l[b])/cl);
      f[33]=(float)MathSin(2.0*M_PI*dt.hour/24.0); f[34]=(float)MathCos(2.0*M_PI*dt.hour/24.0);
      f[35]=(float)MathSin(2.0*M_PI*dt.day_of_week/7.0); f[36]=(float)MathCos(2.0*M_PI*dt.day_of_week/7.0);
      f[37]=(float)MathLog(history[h].v + 1.0); 
      f[38]=(float)((cl - history[h].tvwp)/cl);

      for(int k=0; k<39; k++) input_data[i*39+k] = (f[k]-medians[k])/(iqrs[k]);
   }

   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) return;
   int sig = ArrayMaximum(output_data);
   if(output_data[sig] > 0.65 && !HasPos()) Exec(sig == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
}

void Exec(ENUM_ORDER_TYPE type) {
   double atr[1]; if(CopyBuffer(hATR18,0,0,1,atr)<1) return;
   double p = (type==ORDER_TYPE_BUY)?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
   double sl = (type==ORDER_TYPE_BUY)?(p-atr[0]*SL_MULTIPLIER):(p+atr[0]*SL_MULTIPLIER);
   double tp = (type==ORDER_TYPE_BUY)?(p+atr[0]*TP_MULTIPLIER):(p-atr[0]*TP_MULTIPLIER);
   trade.PositionOpen(_Symbol,type,0.1,p,sl,tp);
}

bool HasPos() {
   for(int i=PositionsTotal()-1; i>=0; i--)
      if(PositionGetSymbol(i)==_Symbol && PositionGetInteger(POSITION_MAGIC)==MAGIC_NUMBER) return true;
   return false;
}

void OnDeinit(const int reason) { if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle); }
```
