// live.mq5 - GOLD Mamba EA
//
// Feature block layout (mirrors gold/nn.py exactly):
//   indices  0-14 -> GOLD   : f0, f1, f2, f3, f4, f5, f6, f7, f8, f10, f11, f12, f13, f14, f15
//   indices 15-22 -> USDX   : f0, f1, f5, f6, f7, f8, f10, f15
//   indices 23-30 -> USDJPY : f0, f1, f5, f6, f7, f8, f10, f15
//
// Removed features:
//   - f9 for all symbols (MACD histogram is linearly redundant with f7/f8)
//   - f11-f14 from USDX/USDJPY (shared once via GOLD)
//   - f2, f3, f4 from USDX/USDJPY

#include <Trade\Trade.mqh>
#resource "\\Experts\\nn\\gold\\gold_mamba.onnx" as uchar model_buffer[]

#define GOLD_FEATURE_COUNT 15
#define AUX_FEATURE_COUNT 8
#define TOTAL_FEATURE_COUNT 31
#define INPUT_BUFFER_SIZE 3720

input int    TICK_DENSITY  = 540;
input double SL_MULTIPLIER = 5.4;
input double TP_MULTIPLIER = 9.0;
input double LOT_SIZE      = 0.01;
input double CONFIDENCE    = 0.72;
input int    MAGIC_NUMBER  = 777777;

input string USDX_SYMBOL   = "USDX";
input string USDJPY_SYMBOL = "USDJPY";

long   onnx_handle = INVALID_HANDLE;
CTrade trade;

// TO BE PASTED FROM PYTHON OUTPUT AFTER TRAINING, NOT AN ISSUE
float medians[TOTAL_FEATURE_COUNT] = {0.0f};
float iqrs[TOTAL_FEATURE_COUNT]    = {1.0f};

struct Bar {
   double o, h, l, c, spread;
   double atr14;
   double ema12, ema26;
   double macd_sig;
   double rsi_gain;
   double rsi_loss;
   ulong  time_msc;
   bool   valid;
};

Bar history[3][200];
Bar cur_b[3];
int ticks_in_bar[3];
bool bar_started[3];
ulong last_tick_time[3];

int    warmup_count[3];
double warmup_sum[3];
int    rsi_warmup[3];
double rsi_gain_acc[3];
double rsi_loss_acc[3];

float input_data[INPUT_BUFFER_SIZE];
float output_data[3];

string SymbolForIdx(int s);
void UpdateIndicators(int s, Bar &b);
double ComputeRSI(Bar &b);
float ScaleAndClip(float value, int feature_index);
void ExtractGoldFeatures(int h, float &f[]);
void ExtractAuxFeatures(int s, int h, float &f[]);
void Predict();
void Execute(int sig);
void LoadHistory();
void ProcessSymbolSnapshotToTime(int s, ulong end_time_msc);
void CloseBar();

int OnInit() {
   onnx_handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) {
      Print("[FATAL] OnnxCreateFromBuffer failed: ", GetLastError());
      return INIT_FAILED;
   }

   const long in_shape[]  = {1, 120, TOTAL_FEATURE_COUNT};
   const long out_shape[] = {1, 3};
   if(!OnnxSetInputShape(onnx_handle, 0, in_shape) ||
      !OnnxSetOutputShape(onnx_handle, 0, out_shape)) {
      Print("[FATAL] OnnxSetShape failed: ", GetLastError());
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return INIT_FAILED;
   }

   ArrayInitialize(ticks_in_bar, 0);
   ArrayInitialize(bar_started, false);
   ArrayInitialize(last_tick_time, 0);
   ArrayInitialize(warmup_count, 0);
   ArrayInitialize(warmup_sum, 0);
   ArrayInitialize(rsi_warmup, 0);
   ArrayInitialize(rsi_gain_acc, 0);
   ArrayInitialize(rsi_loss_acc, 0);
   ArrayInitialize(input_data, 0);

   for(int s = 0; s < 3; s++) {
      for(int b = 0; b < 200; b++) {
         history[s][b].valid = false;
      }
   }

   for(int s = 0; s < 3; s++) {
      MqlTick t;
      if(SymbolInfoTick(SymbolForIdx(s), t)) {
         last_tick_time[s] = t.time_msc;
      } else {
         last_tick_time[s] = TimeCurrent() * 1000ULL;
      }
   }

   Print("[INFO] EA initialised. Symbols: XAUUSD | ", USDX_SYMBOL, " | ", USDJPY_SYMBOL);
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   LoadHistory();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(onnx_handle != INVALID_HANDLE) {
      OnnxRelease(onnx_handle);
   }
}

string SymbolForIdx(int s) {
   if(s == 0) {
      return _Symbol;
   }
   if(s == 1) {
      return USDX_SYMBOL;
   }
   return USDJPY_SYMBOL;
}

void ProcessTick(int s, MqlTick &t) {
   if(t.bid <= 0.0) {
      return;
   }

   if(!bar_started[s]) {
      cur_b[s].o = t.bid;
      cur_b[s].h = t.bid;
      cur_b[s].l = t.bid;
      cur_b[s].c = t.bid;
      cur_b[s].spread = 0.0;
      cur_b[s].time_msc = t.time_msc;
      ticks_in_bar[s] = 0;
      bar_started[s] = true;
   }

   cur_b[s].h = MathMax(cur_b[s].h, t.bid);
   cur_b[s].l = MathMin(cur_b[s].l, t.bid);
   cur_b[s].c = t.bid;
   cur_b[s].spread = t.ask - t.bid;
   ticks_in_bar[s]++;
}

void ProcessSymbolSnapshotToTime(int s, ulong end_time_msc) {
   if(last_tick_time[s] >= end_time_msc) {
      return;
   }

   MqlTick ticks[];
   int count = CopyTicksRange(SymbolForIdx(s), ticks, COPY_TICKS_ALL, last_tick_time[s] + 1, end_time_msc);
   if(count > 0) {
      for(int i = 0; i < count; i++) {
         if(ticks[i].bid > 0.0) {
            ProcessTick(s, ticks[i]);
         }
      }
   }

   last_tick_time[s] = end_time_msc;
}

void CloseBar() {
   for(int s = 0; s < 3; s++) {
      if(ticks_in_bar[s] == 0) {
         if(history[s][0].valid || history[s][0].c > 0.0) {
            double prev_c = history[s][0].c;
            cur_b[s].o = prev_c;
            cur_b[s].h = prev_c;
            cur_b[s].l = prev_c;
            cur_b[s].c = prev_c;
            cur_b[s].spread = history[s][0].spread;
         } else {
            MqlTick fallback;
            if(SymbolInfoTick(SymbolForIdx(s), fallback) && fallback.bid > 0.0) {
               cur_b[s].o = fallback.bid;
               cur_b[s].h = fallback.bid;
               cur_b[s].l = fallback.bid;
               cur_b[s].c = fallback.bid;
               cur_b[s].spread = fallback.ask - fallback.bid;
            }
         }

         cur_b[s].time_msc = cur_b[0].time_msc;
      }

      UpdateIndicators(s, cur_b[s]);

      for(int i = 199; i > 0; i--) {
         history[s][i] = history[s][i - 1];
      }
      history[s][0] = cur_b[s];

      ticks_in_bar[s] = 0;
      bar_started[s] = false;
   }
}

void OnTick() {
   MqlTick gold_ticks[];
   int count = CopyTicks(_Symbol, gold_ticks, COPY_TICKS_ALL, last_tick_time[0] + 1, 100000);
   if(count <= 0) {
      return;
   }

   for(int i = 0; i < count; i++) {
      if(gold_ticks[i].bid <= 0.0) {
         continue;
      }

      ProcessTick(0, gold_ticks[i]);
      last_tick_time[0] = gold_ticks[i].time_msc;

      if(ticks_in_bar[0] >= TICK_DENSITY) {
         ProcessSymbolSnapshotToTime(1, last_tick_time[0]);
         ProcessSymbolSnapshotToTime(2, last_tick_time[0]);
         CloseBar();

         if(history[0][120].valid && history[1][120].valid && history[2][120].valid) {
            Predict();
         }
      }
   }
}

void LoadHistory() {
   Print("[INFO] Pre-loading history...");

   ulong start_time_msc = (TimeCurrent() - 86400 * 3) * 1000ULL;
   MqlTick hist_ticks[];
   int copied = CopyTicks(_Symbol, hist_ticks, COPY_TICKS_ALL, start_time_msc, 250000);

   if(copied <= 0) {
      Print("[WARN] Failed to load history ticks for GOLD. Trying 1 day...");
      start_time_msc = (TimeCurrent() - 86400 * 1) * 1000ULL;
      copied = CopyTicks(_Symbol, hist_ticks, COPY_TICKS_ALL, start_time_msc, 250000);
   }

   if(copied <= 0) {
      Print("[ERROR] No history ticks found.");
      return;
   }

   last_tick_time[0] = hist_ticks[0].time_msc - 1;
   last_tick_time[1] = hist_ticks[0].time_msc - 1;
   last_tick_time[2] = hist_ticks[0].time_msc - 1;

   for(int i = 0; i < copied; i++) {
      if(hist_ticks[i].bid <= 0.0) {
         continue;
      }

      ProcessTick(0, hist_ticks[i]);
      last_tick_time[0] = hist_ticks[i].time_msc;

      if(ticks_in_bar[0] >= TICK_DENSITY) {
         ProcessSymbolSnapshotToTime(1, last_tick_time[0]);
         ProcessSymbolSnapshotToTime(2, last_tick_time[0]);
         CloseBar();
      }
   }

   Print("[INFO] History loaded. Buffer status: ", history[0][120].valid ? "VALID" : "INCOMPLETE");
}

void UpdateIndicators(int s, Bar &b) {
   Bar p = history[s][0];
   bool is_first = (warmup_count[s] == 0);

   double tr;
   if(is_first) {
      tr = b.h - b.l;
   } else {
      tr = MathMax(b.h - b.l, MathMax(MathAbs(b.h - p.c), MathAbs(b.l - p.c)));
   }

   if(warmup_count[s] < 14) {
      warmup_sum[s] += tr;
      warmup_count[s]++;
      b.atr14 = warmup_sum[s] / warmup_count[s];
   } else {
      double prev_atr = (p.atr14 > 0.0 ? p.atr14 : tr);
      b.atr14 = (tr - prev_atr) / 14.0 + prev_atr;
   }

   if(is_first) {
      b.ema12 = b.c;
      b.ema26 = b.c;
      b.macd_sig = 0.0;
   } else {
      b.ema12 = (b.c - p.ema12) * (2.0 / 13.0) + p.ema12;
      b.ema26 = (b.c - p.ema26) * (2.0 / 27.0) + p.ema26;
      double macd_raw = b.ema12 - b.ema26;
      b.macd_sig = (macd_raw - p.macd_sig) * (2.0 / 10.0) + p.macd_sig;
   }

   if(is_first) {
      b.rsi_gain = 0.0;
      b.rsi_loss = 0.0;
   } else {
      double chg = b.c - p.c;
      double gain = (chg > 0.0 ? chg : 0.0);
      double loss = (chg < 0.0 ? -chg : 0.0);

      if(rsi_warmup[s] < 14) {
         rsi_gain_acc[s] += gain;
         rsi_loss_acc[s] += loss;
         rsi_warmup[s]++;
         if(rsi_warmup[s] == 14) {
            b.rsi_gain = rsi_gain_acc[s] / 14.0;
            b.rsi_loss = rsi_loss_acc[s] / 14.0;
         } else {
            b.rsi_gain = 0.0;
            b.rsi_loss = 0.0;
         }
      } else {
         b.rsi_gain = (p.rsi_gain * 13.0 + gain) / 14.0;
         b.rsi_loss = (p.rsi_loss * 13.0 + loss) / 14.0;
      }
   }

   b.valid = (warmup_count[s] >= 14 && rsi_warmup[s] >= 14);
}

double ComputeRSI(Bar &b) {
   if(b.rsi_loss < 1e-10) {
      return (b.rsi_gain > 0.0 ? 100.0 : 50.0);
   }

   double rs = b.rsi_gain / b.rsi_loss;
   return 100.0 - (100.0 / (1.0 + rs));
}

float ScaleAndClip(float value, int feature_index) {
   float iqr = (iqrs[feature_index] > 1e-6f ? iqrs[feature_index] : 1.0f);
   float raw = (value - medians[feature_index]) / iqr;
   return MathMax(-10.0f, MathMin(10.0f, raw));
}

void ExtractGoldFeatures(int h, float &f[]) {
   Bar b = history[0][h];
   Bar bp = history[0][h + 1];

   double cl = b.c;
   double broker_h = (double)((b.time_msc / 3600000ULL) % 24);
   double broker_d = (double)(((b.time_msc / 86400000ULL) + 3) % 7);
   double macd = b.ema12 - b.ema26;
   double rsi_val = ComputeRSI(b);

   f[0]  = (float)MathLog(cl / (bp.c + 1e-10));
   f[1]  = (float)(b.spread / (cl + 1e-10));
   f[2]  = (float)((double)(b.time_msc - bp.time_msc) / 1000.0);
   f[3]  = (float)((b.h - MathMax(b.o, cl)) / (cl + 1e-10));
   f[4]  = (float)((MathMin(b.o, cl) - b.l) / (cl + 1e-10));
   f[5]  = (float)((b.h - b.l) / (cl + 1e-10));
   f[6]  = (float)((cl - b.l) / (b.h - b.l + 1e-8));
   f[7]  = (float)(macd / (cl + 1e-10));
   f[8]  = (float)(b.macd_sig / (cl + 1e-10));
   f[9]  = (float)(b.atr14 / (cl + 1e-10));
   f[10] = (float)MathSin(2.0 * M_PI * broker_h / 24.0);
   f[11] = (float)MathCos(2.0 * M_PI * broker_h / 24.0);
   f[12] = (float)MathSin(2.0 * M_PI * broker_d / 7.0);
   f[13] = (float)MathCos(2.0 * M_PI * broker_d / 7.0);
   f[14] = (float)(rsi_val / 100.0);
}

void ExtractAuxFeatures(int s, int h, float &f[]) {
   Bar b = history[s][h];
   Bar bp = history[s][h + 1];

   double cl = b.c;
   double macd = b.ema12 - b.ema26;
   double rsi_val = ComputeRSI(b);

   f[0] = (float)MathLog(cl / (bp.c + 1e-10));
   f[1] = (float)(b.spread / (cl + 1e-10));
   f[2] = (float)((b.h - b.l) / (cl + 1e-10));
   f[3] = (float)((cl - b.l) / (b.h - b.l + 1e-8));
   f[4] = (float)(macd / (cl + 1e-10));
   f[5] = (float)(b.macd_sig / (cl + 1e-10));
   f[6] = (float)(b.atr14 / (cl + 1e-10));
   f[7] = (float)(rsi_val / 100.0);
}

void Predict() {
   float f_gold[GOLD_FEATURE_COUNT];
   float f_usdx[AUX_FEATURE_COUNT];
   float f_usdjpy[AUX_FEATURE_COUNT];
   const int usdx_offset = GOLD_FEATURE_COUNT;
   const int usdjpy_offset = GOLD_FEATURE_COUNT + AUX_FEATURE_COUNT;

   for(int i = 0; i < 120; i++) {
      int h = 119 - i;
      int base = i * TOTAL_FEATURE_COUNT;

      ExtractGoldFeatures(h, f_gold);
      ExtractAuxFeatures(1, h, f_usdx);
      ExtractAuxFeatures(2, h, f_usdjpy);

      for(int k = 0; k < GOLD_FEATURE_COUNT; k++) {
         input_data[base + k] = ScaleAndClip(f_gold[k], k);
      }
      for(int k = 0; k < AUX_FEATURE_COUNT; k++) {
         input_data[base + usdx_offset + k] = ScaleAndClip(f_usdx[k], usdx_offset + k);
         input_data[base + usdjpy_offset + k] = ScaleAndClip(f_usdjpy[k], usdjpy_offset + k);
      }
   }

   if(OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
      float max_val = MathMax(output_data[0], MathMax(output_data[1], output_data[2]));
      float sum_exp = 0.0f;
      for(int i = 0; i < 3; i++) {
         output_data[i] = (float)MathExp(output_data[i] - max_val);
         sum_exp += output_data[i];
      }
      for(int i = 0; i < 3; i++) {
         output_data[i] /= sum_exp;
      }

      int sig = ArrayMaximum(output_data);
      if(sig > 0 && output_data[sig] > (float)CONFIDENCE) {
         Execute(sig);
      }
   }
}

void Execute(int sig) {
   bool has_position = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER) {
         has_position = true;
         break;
      }
   }
   if(has_position) {
      return;
   }

   double atr = history[0][0].atr14;
   double p = (sig == 1 ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID));
   double sl = (sig == 1 ? (p - atr * SL_MULTIPLIER) : (p + atr * SL_MULTIPLIER));
   double tp = (sig == 1 ? (p + atr * TP_MULTIPLIER) : (p - atr * TP_MULTIPLIER));

   double min_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   if(MathAbs(p - sl) < min_dist || MathAbs(tp - p) < min_dist) {
      Print("[WARN] Stop/TP too close to price, skipping trade.");
      return;
   }

   ENUM_ORDER_TYPE order = (sig == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   trade.PositionOpen(_Symbol, order, LOT_SIZE, p, sl, tp, (sig == 1 ? "GOLD BUY" : "GOLD SELL"));
}
