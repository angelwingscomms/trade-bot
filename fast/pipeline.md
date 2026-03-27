```cpp
#property script_show_inputs // Show settings window
input int ticks_to_export = 2160000; // Total ticks (~5 days of Gold)
input string USDX_Symbol = "$USDX"; // Name of USD Index
input string USDJPY_Symbol = "USDJPY"; // Name of USDJPY

void OnStart() { // Main script function
   MqlTick ticks[], usdx_ticks[], usdjpy_ticks[]; // Arrays to hold tick data
   
   // Enable symbols and check if they're available
   bool usdx_available = SymbolSelect(USDX_Symbol, true);
   bool usdjpy_available = SymbolSelect(USDJPY_Symbol, true);
   
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, ticks_to_export); // Get main symbol ticks
   if(copied <= 0) { Print("❌ Failed to copy ticks"); return; } // Error check
   
   // Get tick data for auxiliary symbols if available
   int usdx_copied = 0, usdjpy_copied = 0;
   if(usdx_available) {
      usdx_copied = CopyTicks(USDX_Symbol, usdx_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      if(usdx_copied <= 0) { Print("⚠️ USDX ticks not available, using placeholder"); usdx_available = false; }
   }
   if(usdjpy_available) {
      usdjpy_copied = CopyTicks(USDJPY_Symbol, usdjpy_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      if(usdjpy_copied <= 0) { Print("⚠️ USDJPY ticks not available, using placeholder"); usdjpy_available = false; }
   }
   
   int h = FileOpen("achilles_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ","); // Create file
   FileWrite(h, "time_msc,bid,ask,usdx,usdjpy"); // Write CSV header
   
   int usdx_idx = 0, usdjpy_idx = 0; // Indices for auxiliary tick arrays
   
   for(int i=0; i<copied; i++) { // Loop through every tick
      ulong t = ticks[i].time_msc; // Current tick timestamp
      
      // Find matching USDX tick (closest timestamp <= current tick)
      double usdx_bid = 0.0;
      if(usdx_available && usdx_copied > 0) {
         while(usdx_idx < usdx_copied - 1 && usdx_ticks[usdx_idx].time_msc <= t) usdx_idx++;
         if(usdx_idx > 0 && usdx_ticks[usdx_idx].time_msc > t) usdx_idx--;
         usdx_bid = usdx_ticks[usdx_idx].bid;
      }
      
      // Find matching USDJPY tick (closest timestamp <= current tick)
      double usdjpy_bid = 0.0;
      if(usdjpy_available && usdjpy_copied > 0) {
         while(usdjpy_idx < usdjpy_copied - 1 && usdjpy_ticks[usdjpy_idx].time_msc <= t) usdjpy_idx++;
         if(usdjpy_idx > 0 && usdjpy_ticks[usdjpy_idx].time_msc > t) usdjpy_idx--;
         usdjpy_bid = usdjpy_ticks[usdjpy_idx].bid;
      }
      
      FileWrite(h, IntegerToString(ticks[i].time_msc) + "," + // Time in milliseconds
                   DoubleToString(ticks[i].bid, 5) + "," + // Current Bid
                   DoubleToString(ticks[i].ask, 5) + "," + // Current Ask
                   DoubleToString(usdx_bid, 5) + "," + // USDX bid (matched by time)
                   DoubleToString(usdjpy_bid, 5)); // USDJPY bid (matched by time)
   }
   FileClose(h); // Close file
   Print("✅ Exported ", copied, " ticks to MQL5\\Files\\achilles_ticks.csv"); // Success message
   if(usdx_available) Print("   USDX: ", usdx_copied, " ticks matched");
   if(usdjpy_available) Print("   USDJPY: ", usdjpy_copied, " ticks matched");
}
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

float means[35] = {0.0f}; // ⚠️ PASTE FROM PYTHON
float stds[35]  = {1.0f}; // ⚠️ PASTE FROM PYTHON

CTrade trade;
long onnx = INVALID_HANDLE;
float input_data[4200]; 
// Arrays increased to 300 to prevent 'Array Out of Range' for 144-period indicators
double o_a[300], h_a[300], l_a[300], c_a[300], s_a[300], d_a[300], dx_a[300], jp_a[300];
int ticks_in_bar = 0, bars = 0;
double b_open, b_high, b_low, b_spread;
ulong b_start_time;

int OnInit() {
   if(!SymbolSelect(USDX_Symbol, true) || !SymbolSelect(USDJPY_Symbol, true)) return(INIT_FAILED);
   onnx = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   if(onnx == INVALID_HANDLE) return(INIT_FAILED);
   long in[]={1,120,35}; OnnxSetInputShape(onnx,0,in);
   long out[]={1,3}; OnnxSetOutputShape(onnx,0,out);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { if(onnx != INVALID_HANDLE) OnnxRelease(onnx); }

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

void Shift(double o, double h, double l, double c, double s, double d) {
   for(int i=299; i>0; i--) { 
      o_a[i]=o_a[i-1]; h_a[i]=h_a[i-1]; l_a[i]=l_a[i-1]; c_a[i]=c_a[i-1]; 
      s_a[i]=s_a[i-1]; d_a[i]=d_a[i-1]; dx_a[i]=dx_a[i-1]; jp_a[i]=jp_a[i-1]; 
   }
   o_a[0]=o; h_a[0]=h; l_a[0]=l; c_a[0]=c; s_a[0]=s; d_a[0]=d;
   dx_a[0]=SymbolInfoDouble(USDX_Symbol, SYMBOL_BID);
   jp_a[0]=SymbolInfoDouble(USDJPY_Symbol, SYMBOL_BID);
   bars++;
}

void Predict() {
   for(int i=0; i<120; i++) {
      int x = 119-i; float f[35];
      f[0]=(float)MathLog(c_a[x]/(c_a[x+1]+1e-8)); 
      f[1]=(float)s_a[x]; 
      f[2]=(float)d_a[x];
      f[3]=(float)((h_a[x]-MathMax(o_a[x],c_a[x]))/(c_a[x]+1e-8)); 
      f[4]=(float)((MathMin(o_a[x],c_a[x])-l_a[x])/(c_a[x]+1e-8));
      f[5]=(float)((h_a[x]-l_a[x])/(c_a[x]+1e-8)); 
      f[6]=(float)((c_a[x]-l_a[x])/(h_a[x]-l_a[x]+1e-8));
      f[7]=CRSI(x,9); f[8]=CRSI(x,18); f[9]=CRSI(x,27);
      f[10]=CATR(x,9); f[11]=CATR(x,18); f[12]=CATR(x,27);
      double e9=CEMA(x,9), e18=CEMA(x,18), e27=CEMA(x,27), e54=CEMA(x,54), e144=CEMA(x,144);
      f[13]=(float)(e9-e18); f[14]=f[13]; f[15]=0;
      f[16]=(float)(e9-c_a[x]); f[17]=(float)(e18-c_a[x]); f[18]=(float)(e27-c_a[x]); 
      f[19]=(float)(e54-c_a[x]); f[20]=(float)(e144-c_a[x]);
      f[21]=0; f[22]=0; f[23]=0;
      f[24]=CWPR(x,9); f[25]=CWPR(x,18); f[26]=CWPR(x,27);
      f[27]=(float)(c_a[x]-c_a[MathMin(x+9, 119)]); f[28]=(float)(c_a[x]-c_a[MathMin(x+18, 119)]); f[29]=(float)(c_a[x]-c_a[MathMin(x+27, 119)]);
      f[30]=(float)((dx_a[x]-dx_a[x+1])/(dx_a[x+1]+1e-8)); f[31]=(float)((jp_a[x]-jp_a[x+1])/(jp_a[x+1]+1e-8));
      f[32]=CBBW(x,9); f[33]=CBBW(x,18); f[34]=CBBW(x,27);
      for(int k=0; k<35; k++) input_data[i*35+k]=(f[k]-means[k])/(stds[k]+1e-8f);
   }
   float out[3]; OnnxRun(onnx, ONNX_DEFAULT, input_data, out);
   if(out[0]>0.5) return;
   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK), bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);
   if(out[1]>0.55 && PositionsTotal()==0) Execute(ORDER_TYPE_BUY, ask);
   if(out[2]>0.55 && PositionsTotal()==0) Execute(ORDER_TYPE_SELL, bid);
}

void Execute(ENUM_ORDER_TYPE type, double p) {
   double sl = (SL_POINTS <= 0) ? 0 : (type==ORDER_TYPE_BUY ? p-SL_POINTS : p+SL_POINTS);
   double tp = (type==ORDER_TYPE_BUY ? p+TP_POINTS : p-TP_POINTS);
   if(type==ORDER_TYPE_BUY) trade.Buy(0.01,_Symbol,p,sl,tp); else trade.Sell(0.01,_Symbol,p,sl,tp);
}

float CRSI(int x, int p) { double u=0, d=0; for(int i=0; i<p; i++) { double df=c_a[x+i]-c_a[x+i+1]; if(df>0) u+=df; else d-=df; } return (d==0)?100:(float)(100-(100/(1+u/(d+1e-8)))); }
float CATR(int x, int p) { double s=0; for(int i=0; i<p; i++) s+=MathMax(h_a[x+i]-l_a[x+i], MathAbs(h_a[x+i]-c_a[x+i+1])); return (float)(s/p); }
float CEMA(int x, int p) { double m=2.0/(p+1); double e=c_a[x+p]; for(int i=x+p-1; i>=x; i--) e=(c_a[i]-e)*m+e; return (float)e; }
float CWPR(int x, int p) { double h=h_a[x], l=l_a[x]; for(int i=1; i<p; i++) { h=MathMax(h,h_a[x+i]); l=MathMin(l,l_a[x+i]); } return (h==l)?0:(float)(-100*(h-c_a[x])/(h-l+1e-8)); }
float CBBW(int x, int p) { double s=0, sq=0; for(int i=0; i<p; i++) s+=c_a[i+x]; double m=s/p; for(int i=0; i<p; i++) sq+=MathPow(c_a[i+x]-m,2); return (float)((MathSqrt(sq/p)*4)/(m+1e-8)); }
```

```python
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
print("Constructing Tick Bars...")
df = df_t.iloc[::TICK_DENSITY].copy()
df['open'] = df_t['bid'].iloc[::TICK_DENSITY].values  # First tick of each bar
df['high'] = df_t['bid'].rolling(TICK_DENSITY).max().iloc[::TICK_DENSITY].values
df['low'] = df_t['bid'].rolling(TICK_DENSITY).min().iloc[::TICK_DENSITY].values
# FIX: Use tick at index (TICK_DENSITY-1) as close (last tick of each bar)
# Previously used shift(-1) which leaked future data from the next bar
df['close'] = df_t['bid'].iloc[TICK_DENSITY-1::TICK_DENSITY].values
df['spread'] = (df_t['ask'] - df_t['bid']).rolling(TICK_DENSITY).mean().iloc[::TICK_DENSITY].values
df['duration'] = df_t['time_msc'].diff(TICK_DENSITY).iloc[::TICK_DENSITY].values

# Dummy columns for USDX/USDJPY if they don't exist in your specific tick file 
# (Based on your original code f30/f31)
if 'usdx' not in df.columns: df['usdx'] = df['close'] # Placeholder
if 'usdjpy' not in df.columns: df['usdjpy'] = df['close'] # Placeholder

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
TP, SL, H = 1.44, 0.50, 30
def label(df, tp, sl, h):
    c, hi, lo = df.close.values, df.high.values, df.low.values
    t = np.zeros(len(df), dtype=int)
    for i in range(len(df)-h):
        up, lw = c[i]+tp, c[i]-sl
        for j in range(i+1, i+h+1):
            if hi[j] >= up: t[i]=1; break
            if lo[j] <= lw: t[i]=2; break
    return t

print("Labeling data...")
df['target'] = label(df, TP, SL, H)

# 5. MODEL PREP
features = [f'f{i}' for i in range(35)]
X = df[features].values
mean, std = X.mean(axis=0), X.std(axis=0)
X_s = (X - mean) / (std + 1e-8)

def win(X, y, horizon=30):
    xs, ys = [], []
    # FIX: Ensure all labels are valid by limiting range
    # Label at position j is valid only if j <= len(df)-horizon-1
    # We use y[i+119], so i+119 <= len(X)-horizon-1, thus i <= len(X)-horizon-120
    max_i = len(X) - 120 - horizon
    for i in range(max_i + 1):  # +1 because range is exclusive
        xs.append(X[i:i+120]); ys.append(y[i+119])
    return np.array(xs), np.array(ys)

X_seq, y_seq = win(X_s, df.target.values, H)

# 6. MODEL ARCHITECTURE
in_lay = tf.keras.Input(shape=(120, 35))
ls = tf.keras.layers.LSTM(35, return_sequences=True, activation='mish')(in_lay)
at = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=35)(ls, ls)
pl = tf.keras.layers.GlobalAveragePooling1D()(tf.keras.layers.Add()([ls, at]))
ou = tf.keras.layers.Dense(3, activation='softmax')(tf.keras.layers.Dense(20, activation='mish')(pl))

model = tf.keras.Model(in_lay, ou)
model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(X_seq, y_seq, epochs=54, batch_size=64, validation_split=0.2)

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
