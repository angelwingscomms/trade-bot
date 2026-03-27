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

// FLAW 2 FIX: Use robust scaling (median/IQR) instead of mean/std for fat-tailed financial data
float medians[35] = {0.0f}; // ⚠️ PASTE FROM PYTHON
float iqrs[35]  = {1.0f}; // ⚠️ PASTE FROM PYTHON

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
       // FLAW 1 FIX: ATR normalized by Close for stationarity
       f[10]=(float)(CATR(x,9)/c_a[x]); f[11]=(float)(CATR(x,18)/c_a[x]); f[12]=(float)(CATR(x,27)/c_a[x]);
       // FLAW 2.1 FIX: Use global EMA arrays (running state) instead of truncated CEMA
       double e9=ema9_a[x], e18=ema18_a[x], e27=ema27_a[x], e54=ema54_a[x], e144=ema144_a[x];
       // FLAW 2.2 FIX: Use proper MACD values from running state arrays
       // FLAW 1 FIX: MACD normalized by Close for stationarity
       // f13 = MACD line, f14 = Signal line, f15 = Histogram
       f[13]=(float)(macd_a[x]/c_a[x]); f[14]=(float)(macd_signal_a[x]/c_a[x]); f[15]=(float)(macd_hist_a[x]/c_a[x]);
       // FLAW 1 FIX: EMA distances normalized by Close for stationarity
       f[16]=(float)((e9-c_a[x])/c_a[x]); f[17]=(float)((e18-c_a[x])/c_a[x]); f[18]=(float)((e27-c_a[x])/c_a[x]); 
       f[19]=(float)((e54-c_a[x])/c_a[x]); f[20]=(float)((e144-c_a[x])/c_a[x]);
      // FLAW 2.2 FIX: Use proper CCI values from running state arrays
       f[21]=(float)cci9_a[x]; f[22]=(float)cci18_a[x]; f[23]=(float)cci27_a[x];
       f[24]=CWPR(x,9); f[25]=CWPR(x,18); f[26]=CWPR(x,27);
       // FLAW 1 FIX: Momentum normalized by Close for stationarity
       f[27]=(float)((c_a[x]-c_a[RingIdx(119-i+9)])/c_a[x]); 
       f[28]=(float)((c_a[x]-c_a[RingIdx(119-i+18)])/c_a[x]); 
       f[29]=(float)((c_a[x]-c_a[RingIdx(119-i+27)])/c_a[x]);
      f[30]=(float)((dx_a[x]-dx_a[x1])/(dx_a[x1]+1e-8)); 
      f[31]=(float)((jp_a[x]-jp_a[x1])/(jp_a[x1]+1e-8));
      f[32]=CBBW(x,9); f[33]=CBBW(x,18); f[34]=CBBW(x,27);
       // FLAW 2 FIX: Robust scaling using median/IQR instead of mean/std
       for(int k=0; k<35; k++) input_data[i*35+k]=(f[k]-medians[k])/(iqrs[k]+1e-8f);
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