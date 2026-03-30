#include <Trade\Trade.mqh>
#resource "\\Experts\\nn\\bitcoin\\bitcoin_144.onnx" as uchar model_buffer[]

input int TICK_DENSITY = 144;
input double SL_MULTIPLIER = 5.4;
input double TP_MULTIPLIER = 27;
long onnx_handle = INVALID_HANDLE;
CTrade trade;

// PASTE FROM PYTHON OUTPUT
float medians[17] = {0.00000000f, 27.00000000f, 70.73900000f, 0.00012056f, 0.00012359f, 0.00068856f, 0.50000000f, -0.00000263f, -0.00000316f, -0.00000581f, 0.00071232f, 0.00000000f, -0.00000000f, 0.00000000f, -0.22252093f, 0.89199804f, 0.00026568f};
float iqrs[17] = {0.00065647f, 0.03472222f, 30.58600000f, 0.00020881f, 0.00021586f, 0.00048541f, 0.62345182f, 0.00075497f, 0.00072372f, 0.00021932f, 0.00032388f, 1.41421356f, 1.41421356f, 1.56366296f, 1.52445867f, 1.00000000f, 0.00450346f};

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
   if(onnx_handle == INVALID_HANDLE) {
      Print("[FATAL] OnnxCreateFromBuffer failed: ", GetLastError());
      return(INIT_FAILED);
   }
   const long in_shape[] = {1, 2040};
   const long out_shape[] = {1, 3};
   if(!OnnxSetInputShape(onnx_handle, 0, in_shape) ||
      !OnnxSetOutputShape(onnx_handle, 0, out_shape)) {
      Print("[FATAL] OnnxSetShape failed: ", GetLastError());
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return(INIT_FAILED);
   }
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
      if(history[120].c > 0) Predict();  // h goes up to 119; h+1 = 120 is the deepest access
   }
}

// Add to Bar struct: int bar_count;  (initialise to 0 in OnInit / cold-start branch)
// Requires a small ring buffer for TR accumulation – shown inline below.

static double tr_buf[18];
static int    tr_buf_n = 0;

void UpdateIndicators(Bar &b) {
   Bar p = history[0];
   if(p.c <= 0) {
      b.macd_ema12 = b.c; b.macd_ema26 = b.c; b.macd_sig = 0;
      double tr0 = b.h - b.l;
      tr_buf[0] = tr0; tr_buf_n = 1;
      b.atr18 = tr0;   // will be replaced once 18 TRs are collected
      return;
   }
   double tr = MathMax(b.h-b.l, MathMax(MathAbs(b.h-p.c), MathAbs(b.l-p.c)));
   if(tr_buf_n < 18) {
      tr_buf[tr_buf_n++] = tr;
      // Seed phase: use plain SMA, matching pandas_ta initialisation
      double sum = 0;
      for(int k = 0; k < tr_buf_n; k++) sum += tr_buf[k];
      b.atr18 = sum / tr_buf_n;
   } else {
      // Wilder smoothing — identical to pandas_ta after the seed window
      b.atr18 = (tr - p.atr18) / 18.0 + p.atr18;
   }
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
   double sl  = (sig==1) ? (p - history[0].atr18*SL_MULTIPLIER) : (p + history[0].atr18*SL_MULTIPLIER);
   double tp  = (sig==1) ? (p + history[0].atr18*TP_MULTIPLIER)  : (p - history[0].atr18*TP_MULTIPLIER);
   
   // NEW: validate stops are non-degenerate
   double min_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(MathAbs(p - sl) < min_dist || MathAbs(tp - p) < min_dist) {
      Print("[WARN] Stop/TP too close to price, skipping trade.");
      return;
   }
   trade.PositionOpen(_Symbol,(sig==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL),0.1,p,sl,tp);
}