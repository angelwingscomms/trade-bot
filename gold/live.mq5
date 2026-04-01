// live.mq5 — GOLD Mamba EA  (48 features: GOLD[16] | USDX[16] | USDJPY[16])
//
// Feature block layout (mirrors nn.py exactly):
//   indices  0-15  →  GOLD   features f0..f15
//   indices 16-31  →  USDX   features f0..f15
//   indices 32-47  →  USDJPY features f0..f15
//
// Per-symbol feature index offsets:
//   f0  log return          f1  spread/close      f2  bar seconds
//   f3  upper wick          f4  lower wick         f5  range/close
//   f6  close-in-range      f7  MACD line          f8  MACD signal
//   f9  MACD histogram      f10 ATR14/close        f11 sin(hour)
//   f12 cos(hour)           f13 sin(weekday)       f14 cos(weekday)
//   f15 RSI14/100

#include <Trade\Trade.mqh>
#resource "\\Experts\\nn\\gold\\gold_mamba.onnx" as uchar model_buffer[]

input int    TICK_DENSITY  = 540;
input double SL_MULTIPLIER = 5.4;
input double TP_MULTIPLIER = 9.0;
input double LOT_SIZE      = 0.01;
input double CONFIDENCE    = 0.72;
input int    MAGIC_NUMBER  = 777777;

// Change to match your broker's Dollar Index and USDJPY symbols if needed
input string USDX_SYMBOL   = "USDX";
input string USDJPY_SYMBOL = "USDJPY";

long   onnx_handle = INVALID_HANDLE;
CTrade trade;

// PASTE FROM PYTHON OUTPUT
float medians[48] = {0.0f};   // replace with Python output
float iqrs[48]    = {1.0f};   // replace with Python output

// ─── Bar accumulator ─────────────────────────────────────────────
struct Bar {
   double o, h, l, c, spread;
   double atr14;           // Wilder ATR(14)
   double ema12, ema26;    // for MACD
   double macd_sig;        // MACD signal EMA(9)
   double rsi_gain;        // Wilder smoothed avg gain
   double rsi_loss;        // Wilder smoothed avg loss
   ulong  time_msc;
   bool   valid;           // false until first bar after warmup
};

// 3 symbols × 200 bars ring buffer
//   GOLD=0  USDX=1  USDJPY=2
Bar history[3][200];
Bar cur_b[3];
int ticks_in_bar[3];
bool bar_started[3];
ulong last_tick_time[3];

// Per-symbol ATR warmup (need 14 bars before Wilder kicks in)
int    warmup_count[3];
double warmup_sum[3];       // accumulator for simple average during warmup

// RSI warmup: need 14 bars of gain/loss before first valid RSI
int    rsi_warmup[3];
double rsi_gain_acc[3];
double rsi_loss_acc[3];

float input_data[5760];   // 1 × 120 × 48 = 5760 floats, row-major
float output_data[3];

void UpdateIndicators(int s, Bar &b);
void Predict();
void Execute(int sig);
void LoadHistory();
void ProcessSymbolSnapshotToTime(int s, ulong end_time_msc);
void CloseBar();

//+------------------------------------------------------------------+
int OnInit() {
   onnx_handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) {
      Print("[FATAL] OnnxCreateFromBuffer failed: ", GetLastError());
      return INIT_FAILED;
   }

   const long in_shape[]  = {1, 120, 48};
   const long out_shape[] = {1, 3};
   if(!OnnxSetInputShape(onnx_handle, 0, in_shape) ||
      !OnnxSetOutputShape(onnx_handle, 0, out_shape)) {
      Print("[FATAL] OnnxSetShape failed: ", GetLastError());
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return INIT_FAILED;
   }

   // Reset all state
   ArrayInitialize(ticks_in_bar, 0);
   ArrayInitialize(bar_started, false);
   ArrayInitialize(last_tick_time, 0);
   ArrayInitialize(warmup_count, 0);
   ArrayInitialize(warmup_sum,   0);
   ArrayInitialize(rsi_warmup,   0);
   ArrayInitialize(rsi_gain_acc, 0);
   ArrayInitialize(rsi_loss_acc, 0);
   ArrayInitialize(input_data,    0);
   for(int s = 0; s < 3; s++)
      for(int b = 0; b < 200; b++) history[s][b].valid = false;

   // Initialize last_tick_time to current time to avoid 1970 bug
   for(int s = 0; s < 3; s++) {
      MqlTick t;
      if(SymbolInfoTick(SymbolForIdx(s), t)) {
         last_tick_time[s] = t.time_msc;
      } else {
         last_tick_time[s] = TimeCurrent() * 1000LL;
      }
   }

   Print("[INFO] EA initialised. Symbols: XAUUSD | ", USDX_SYMBOL, " | ", USDJPY_SYMBOL);
   
   trade.SetExpertMagicNumber(777777); // Set magic number to avoid interfering with other EAs/manual trades
   
   // Pre-load history to avoid 18-hour wait time
   LoadHistory();
   
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
}

//+------------------------------------------------------------------+
// Map symbol index → broker symbol string
string SymbolForIdx(int s) {
   if(s == 0) return _Symbol;
   if(s == 1) return USDX_SYMBOL;
   return USDJPY_SYMBOL;
}

//+------------------------------------------------------------------+
void ProcessTick(int s, MqlTick &t) {
   if(t.bid <= 0.0) return;
   
   if(!bar_started[s]) {
      cur_b[s].o        = t.bid;
      cur_b[s].h        = t.bid;
      cur_b[s].l        = t.bid;
      cur_b[s].c        = t.bid;
      cur_b[s].spread   = 0;
      cur_b[s].time_msc = t.time_msc;
      ticks_in_bar[s]   = 0;
      bar_started[s]    = true;
   }

   cur_b[s].h = MathMax(cur_b[s].h, t.bid);
   cur_b[s].l = MathMin(cur_b[s].l, t.bid);
   cur_b[s].c = t.bid;
   cur_b[s].spread = (t.ask - t.bid); // Store last spread to match Python's .last()
   ticks_in_bar[s]++;
}

void ProcessSymbolSnapshotToTime(int s, ulong end_time_msc) {
   if (last_tick_time[s] >= end_time_msc) return;
   
   MqlTick ticks[];
   int count = CopyTicksRange(SymbolForIdx(s), ticks, COPY_TICKS_ALL, last_tick_time[s] + 1, end_time_msc);
   if(count > 0) {
      for(int i = 0; i < count; i++) {
         if(ticks[i].bid > 0.0) {
            ProcessTick(s, ticks[i]);
         }
      }
   }
   // Always advance time pointer so we don't query empty intervals repeatedly
   last_tick_time[s] = end_time_msc;
}

void CloseBar() {
   for(int s = 0; s < 3; s++) {
      if(ticks_in_bar[s] == 0) {
         // No ticks received for this symbol during GOLD's bar. Forward fill from history.
         if(history[s][0].valid || history[s][0].c > 0) {
            double prev_c = history[s][0].c;
            cur_b[s].o = prev_c;
            cur_b[s].h = prev_c;
            cur_b[s].l = prev_c;
            cur_b[s].c = prev_c;
            cur_b[s].spread = history[s][0].spread;
         } else {
            // Fallback if very first bar has no ticks (rare but possible for illiquid crosses)
            MqlTick fallback;
            if(SymbolInfoTick(SymbolForIdx(s), fallback) && fallback.bid > 0) {
               cur_b[s].o = cur_b[s].h = cur_b[s].l = cur_b[s].c = fallback.bid;
               cur_b[s].spread = fallback.ask - fallback.bid;
            }
         }
         cur_b[s].time_msc = cur_b[0].time_msc; // align time with GOLD ONLY if empty
      }
      
      UpdateIndicators(s, cur_b[s]);
      
      for(int i = 199; i > 0; i--) history[s][i] = history[s][i-1];
      history[s][0] = cur_b[s];
      
      // Reset for next bar
      ticks_in_bar[s] = 0;
      bar_started[s] = false;
   }
}

void OnTick() {
   MqlTick gold_ticks[];
   // Fetch all GOLD ticks since last_tick_time[0]
   int count = CopyTicks(_Symbol, gold_ticks, COPY_TICKS_ALL, last_tick_time[0] + 1, 100000);
   if(count <= 0) return;

   for(int i = 0; i < count; i++) {
      if(gold_ticks[i].bid <= 0.0) continue;
      
      ProcessTick(0, gold_ticks[i]);
      last_tick_time[0] = gold_ticks[i].time_msc;
      
      if(ticks_in_bar[0] >= TICK_DENSITY) {
         // GOLD bar complete.
         // Now fetch USDX and USDJPY up to THIS exact time_msc
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
   // We need enough bars for SEQ_LEN(120) + WARMUP(50) = 170 bars. 
   // Fetching last 3 days of ticks should be more than enough for GOLD.
   ulong start_time_msc = (TimeCurrent() - 86400 * 3) * 1000LL; 
   
   MqlTick hist_ticks[];
   int copied = CopyTicks(_Symbol, hist_ticks, COPY_TICKS_ALL, start_time_msc, 250000);
   if (copied <= 0) {
      Print("[WARN] Failed to load history ticks for GOLD. Trying 1 day...");
      start_time_msc = (TimeCurrent() - 86400 * 1) * 1000LL;
      copied = CopyTicks(_Symbol, hist_ticks, COPY_TICKS_ALL, start_time_msc, 250000);
   }
   
   if (copied <= 0) {
      Print("[ERROR] No history ticks found.");
      return;
   }
   
   last_tick_time[0] = hist_ticks[0].time_msc - 1;
   last_tick_time[1] = hist_ticks[0].time_msc - 1;
   last_tick_time[2] = hist_ticks[0].time_msc - 1;
   
   for(int i = 0; i < copied; i++) {
      if(hist_ticks[i].bid <= 0.0) continue;
      
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

//+------------------------------------------------------------------+
// UpdateIndicators: incremental Wilder ATR(14), MACD EMAs, RSI Wilder(14)
void UpdateIndicators(int s, Bar &b) {
   Bar &p = history[s][0];   // previous bar (newest in buffer before shift)

   bool is_first = (warmup_count[s] == 0);

   // ── TR / ATR(14) ──────────────────────────────────────────────
   double tr;
   if(is_first) {
      tr    = b.h - b.l;
   } else {
      tr = MathMax(b.h - b.l,
           MathMax(MathAbs(b.h - p.c),
                   MathAbs(b.l - p.c)));
   }

   if(warmup_count[s] < 14) {
      warmup_sum[s] += tr;
      warmup_count[s]++;
      b.atr14 = warmup_sum[s] / warmup_count[s];
   } else {
      double prev_atr = (p.atr14 > 0 ? p.atr14 : tr); // guard
      b.atr14 = (tr - prev_atr) / 14.0 + prev_atr;    // Wilder smoothing
   }

   // ── MACD EMAs ─────────────────────────────────────────────────
   if(is_first) {
      b.ema12    = b.c;
      b.ema26    = b.c;
      b.macd_sig = 0;
   } else {
      b.ema12    = (b.c - p.ema12)    * (2.0 / 13.0) + p.ema12;
      b.ema26    = (b.c - p.ema26)    * (2.0 / 27.0) + p.ema26;
      double macd_raw = b.ema12 - b.ema26;
      b.macd_sig = (macd_raw - p.macd_sig) * (2.0 / 10.0) + p.macd_sig;
   }

   // ── RSI(14) Wilder ────────────────────────────────────────────
   if(is_first) {
      b.rsi_gain = 0;
      b.rsi_loss = 0;
   } else {
      double chg   = b.c - p.c;
      double gain  = (chg > 0) ? chg : 0.0;
      double loss  = (chg < 0) ? -chg : 0.0;

      if(rsi_warmup[s] < 14) {
         rsi_gain_acc[s] += gain;
         rsi_loss_acc[s] += loss;
         rsi_warmup[s]++;
         if(rsi_warmup[s] == 14) {
            b.rsi_gain = rsi_gain_acc[s] / 14.0;
            b.rsi_loss = rsi_loss_acc[s] / 14.0;
         } else {
            b.rsi_gain = 0;
            b.rsi_loss = 0;
         }
      } else {
         b.rsi_gain = (p.rsi_gain * 13.0 + gain) / 14.0;
         b.rsi_loss = (p.rsi_loss * 13.0 + loss) / 14.0;
      }
   }

   b.valid = (warmup_count[s] >= 14 && rsi_warmup[s] >= 14);
}

//+------------------------------------------------------------------+
// ComputeRSI: returns RSI value in [0,100] from bar's Wilder state
double ComputeRSI(Bar &b) {
   if(b.rsi_loss < 1e-10) return (b.rsi_gain > 0) ? 100.0 : 50.0;
   double rs = b.rsi_gain / b.rsi_loss;
   return 100.0 - (100.0 / (1.0 + rs));
}

//+------------------------------------------------------------------+
// ExtractFeatures: fills f[16] for one symbol at bar index h in buffer
// 'h'   = history index (0 = newest)
// 'h+1' = previous bar for log-return denominator
void ExtractFeatures(int s, int h, float &f[]) {
   Bar &b  = history[s][h];
   Bar &bp = history[s][h + 1];   // previous bar

   double cl      = b.c;
   // Ensure time_msc is treated consistently with Python (broker time integer hour/weekday)
   double utc_h   = (double)((b.time_msc / 3600000ULL) % 24);
   double utc_d   = (double)(((b.time_msc / 86400000ULL) + 3) % 7);
   double macd    = b.ema12 - b.ema26;
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
   f[9]  = (float)((macd - b.macd_sig) / (cl + 1e-10));
   f[10] = (float)(b.atr14 / (cl + 1e-10));
   f[11] = (float)MathSin(2 * M_PI * utc_h / 24.0);
   f[12] = (float)MathCos(2 * M_PI * utc_h / 24.0);
   f[13] = (float)MathSin(2 * M_PI * utc_d / 7.0);
   f[14] = (float)MathCos(2 * M_PI * utc_d / 7.0);
   f[15] = (float)(rsi_val / 100.0);
}

//+------------------------------------------------------------------+
void Predict() {
   float f_gold[16], f_usdx[16], f_usdjpy[16];

   // Build sequence: oldest bar → index 0, newest bar → index 119
   for(int i = 0; i < 120; i++) {
      int h = 119 - i;

      ExtractFeatures(0, h, f_gold);
      ExtractFeatures(1, h, f_usdx);
      ExtractFeatures(2, h, f_usdjpy);

      int base = i * 48;
      for(int k = 0; k < 16; k++) {
         float raw;
         // Safety: ensure iqr > 0 to avoid division by zero
         float iqr_g = (iqrs[k] > 1e-6f ? iqrs[k] : 1.0f);
         float iqr_x = (iqrs[16+k] > 1e-6f ? iqrs[16+k] : 1.0f);
         float iqr_j = (iqrs[32+k] > 1e-6f ? iqrs[32+k] : 1.0f);

         raw = (f_gold[k]   - medians[k])    / iqr_g;
         input_data[base + k]      = MathMax(-10.0f, MathMin(10.0f, raw));
         
         raw = (f_usdx[k]   - medians[16+k]) / iqr_x;
         input_data[base + 16 + k] = MathMax(-10.0f, MathMin(10.0f, raw));
         
         raw = (f_usdjpy[k] - medians[32+k]) / iqr_j;
         input_data[base + 32 + k] = MathMax(-10.0f, MathMin(10.0f, raw));
      }
   }

   if(OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
      // Apply Softmax to output_data
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
      if(sig > 0 && output_data[sig] > (float)CONFIDENCE) Execute(sig);
   }
}

//+------------------------------------------------------------------+
void Execute(int sig) {
   bool has_position = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER) {
         has_position = true;
         break;
      }
   }
   if(has_position) return;

   double atr  = history[0][0].atr14;
   double p    = (sig == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                            : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl   = (sig == 1) ? (p - atr * SL_MULTIPLIER) : (p + atr * SL_MULTIPLIER);
   double tp   = (sig == 1) ? (p + atr * TP_MULTIPLIER) : (p - atr * TP_MULTIPLIER);

   double min_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL)
                   * SymbolInfoDouble (_Symbol, SYMBOL_POINT);

   if(MathAbs(p - sl) < min_dist || MathAbs(tp - p) < min_dist) {
      Print("[WARN] Stop/TP too close to price, skipping trade.");
      return;
   }

   ENUM_ORDER_TYPE order = (sig == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   trade.PositionOpen(_Symbol, order, LOT_SIZE, p, sl, tp,
                      (sig == 1 ? "GOLD BUY" : "GOLD SELL"));
}
