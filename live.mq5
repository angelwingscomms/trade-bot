#include <Trade\Trade.mqh>
// @active-model-reference begin
#define ACTIVE_MODEL_SYMBOL "XAUUSD"
#define ACTIVE_MODEL_VERSION "16_04_2026-17_53__42-au"
#include "symbols/xauusd/models/16_04_2026-17_53__42-au/config.mqh"
#resource "symbols\\xauusd\\models\\16_04_2026-17_53__42-au\\model.onnx" as uchar model_buffer[]
// @active-model-reference end

#ifndef MODEL_USE_ATR_RISK
#define MODEL_USE_ATR_RISK 1
#endif

#ifndef MODEL_USE_FIXED_TIME_BARS
#define MODEL_USE_FIXED_TIME_BARS 0
#endif

#ifndef MODEL_USE_FIXED_TICK_BARS
#define MODEL_USE_FIXED_TICK_BARS 0
#endif

#define INPUT_BUFFER_SIZE (SEQ_LEN * MODEL_FEATURE_COUNT)
#define HISTORY_SIZE (REQUIRED_HISTORY_INDEX + 1)
#define PRIMARY_BAR_MILLISECONDS ((ulong)PRIMARY_BAR_SECONDS * 1000)

input bool R = (MODEL_USE_ATR_RISK == 0);
input double FIXED_MOVE = DEFAULT_FIXED_MOVE;
input double SL_MULTIPLIER = DEFAULT_SL_MULTIPLIER;
input double TP_MULTIPLIER = DEFAULT_TP_MULTIPLIER;
input double LOT_SIZE = DEFAULT_LOT_SIZE;
input double LOT_SIZE_CAP = DEFAULT_LOT_SIZE_CAP;
input double RISK_PERCENT = DEFAULT_RISK_PERCENT;
input double BROKER_MIN_LOT_SIZE = DEFAULT_BROKER_MIN_LOT_SIZE;
input bool USE_BROKER_MIN_LOT = USE_BROKER_MIN_LOT_SIZE;
input bool USE_LOT_SIZE_CAP_INPUT = USE_LOT_SIZE_CAP;
input bool USE_RISK_PERCENT_INPUT = USE_RISK_PERCENT;
input int MAGIC_NUMBER = 777777;
input bool DEBUG_LOG = true;
input string USDX_SYMBOL = "$USDX";
input string USDJPY_SYMBOL = "USDJPY";

long onnx_handle = INVALID_HANDLE;
CTrade trade;

struct Bar {
   double o;
   double h;
   double l;
   double c;
   double spread;
   double spread_mean;
   double tick_imbalance;
   int tick_count;
   double usdx_bid;
   double usdjpy_bid;
   double atr_feature;
   double atr_trade;
   ulong time_open_msc;
   ulong time_close_msc;
   bool valid;
};

Bar history[HISTORY_SIZE];
Bar current_bar;
int ticks_in_bar = 0;
bool bar_started = false;
ulong current_bar_bucket = 0;
ulong last_tick_time = 0;
double tick_imbalance_sum = 0.0;
double spread_sum = 0.0;
double last_bid = 0.0;
int last_sign = 1;
bool usdx_available = false;
bool usdjpy_available = false;
double last_usdx_bid = 0.0;
double last_usdjpy_bid = 0.0;
double primary_expected_abs_theta = 60.0;
int warmup_count = 0;
double warmup_sum_feature = 0.0;
double warmup_sum_trade = 0.0;
float input_data[INPUT_BUFFER_SIZE];
float output_data[3];
int prediction_count = 0;
int hold_skip_count = 0;
int confidence_skip_count = 0;
int position_skip_count = 0;
int stops_too_close_skip_count = 0;
int volume_skip_count = 0;
int trade_open_failed_count = 0;
int trades_opened_count = 0;
int closed_trade_count = 0;
int closed_win_count = 0;
int closed_loss_count = 0;
double realized_pnl = 0.0;

int UpdateTickSign(double bid);
ulong BarBucket(ulong time_msc);
ulong BarOpenTime(ulong bar_bucket);
void StartBar(MqlTick &tick, ulong bar_bucket);
void StartImbalanceBar(MqlTick &tick);
bool RollFixedTimeBarIfNeeded(ulong next_bar_bucket, int &closed_tick_count);
void ProcessTick(MqlTick &tick, ulong bar_bucket);
void UpdateIndicators(Bar &bar);
double ResolveImbalanceThresholdBase();
bool ShouldClosePrimaryBar(double &observed_abs_theta);
void UpdatePrimaryImbalanceThreshold(double observed_abs_theta);
void CloseBar();
void LoadHistory();
float ScaleAndClip(float value, int feature_index);
double SafeLogRatio(double num, double den);
double LogReturnAt(int h);
double ReturnOverBars(int h, int bars);
double RollingStdReturn(int h, int window);
double MeanClose(int h, int window);
double StdClose(int h, int window);
double MaxHigh(int h, int window);
double MinLow(int h, int window);
double MeanTickCount(int h, int window);
double StdTickCount(int h, int window);
double MeanTickImbalance(int h, int window);
double MeanSpreadRel(int h, int window);
double StdSpreadRel(int h, int window);
double MeanAtrFeature(int h, int window);
double TrueRangeAt(int h);
double SimpleAtr(int h, int period);
double EmaClose(int h, int period);
void MacdAt(int h, double &line, double &signal, double &hist);
double TypicalPrice(int h);
double SimpleCci(int h, int period);
double WilliamsR(int h, int period);
double SimpleRsi(int h, int period);
double StochK(int h, int period);
double StochD(int h, int period);
void ExtractFeatures(int h, float &features[]);
void Softmax(const float &logits[], float &probs[]);
void Predict();
void Execute(int signal);
void DebugPrint(string message);
string SignalName(int signal);
double StopDistance();
double TargetDistance();
double ResolveMinimumVolume();
double NormalizeVolume(double volume);
double CalculateTradeVolume(int signal, double price, double sl);
void PrintRunSummary();
double ResolveAuxBid(string symbol, bool &available, double &last_value, double fallback);

int OnInit() {
   DebugPrint("Model reference: " + ACTIVE_MODEL_SYMBOL + "/" + ACTIVE_MODEL_VERSION);

   onnx_handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) {
      Print("[FATAL] OnnxCreateFromBuffer failed: ", GetLastError());
      return INIT_FAILED;
   }

   long input_shape[3];
   long output_shape[2];
   input_shape[0] = 1;
   input_shape[1] = SEQ_LEN;
   input_shape[2] = MODEL_FEATURE_COUNT;
   output_shape[0] = 1;
   output_shape[1] = 3;
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape) || !OnnxSetOutputShape(onnx_handle, 0, output_shape)) {
      Print("[FATAL] OnnxSetShape failed: ", GetLastError());
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return INIT_FAILED;
   }

   for(int i = 0; i < HISTORY_SIZE; i++) {
      history[i].valid = false;
   }

   ArrayInitialize(input_data, 0.0f);
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   primary_expected_abs_theta = ResolveImbalanceThresholdBase();
   if(MODEL_USE_FIXED_TICK_BARS != 0 && PRIMARY_TICK_DENSITY <= 0) {
      Print("[FATAL] PRIMARY_TICK_DENSITY must be positive for fixed-tick bars.");
      return INIT_FAILED;
   }
   DebugPrint(
      StringFormat(
         "init seq=%d horizon=%d history=%d bar_mode=%s imbalance_min_ticks=%d imbalance_span=%d bar_seconds=%d tick_density=%d risk_mode=%s fixed_move=%.2f sl=%.2f tp=%.2f lot=%.2f risk_pct=%.3f primary_conf=%.2f",
         SEQ_LEN,
         TARGET_HORIZON,
         REQUIRED_HISTORY_INDEX,
         (MODEL_USE_FIXED_TICK_BARS != 0 ? "FIXED_TICK" : (MODEL_USE_FIXED_TIME_BARS != 0 ? "FIXED_TIME" : "IMBALANCE")),
         IMBALANCE_MIN_TICKS,
         IMBALANCE_EMA_SPAN,
         PRIMARY_BAR_SECONDS,
         PRIMARY_TICK_DENSITY,
         (R ? "FIXED" : "ATR"),
         FIXED_MOVE,
         SL_MULTIPLIER,
         TP_MULTIPLIER,
         LOT_SIZE,
         RISK_PERCENT,
         PRIMARY_CONFIDENCE
      )
   );
   if(PRIMARY_CONFIDENCE > 1.0) {
      Print("[INFO] Live trading disabled because the active model failed the trainer quality gate.");
   }

   MqlTick tick;
   if(SymbolInfoTick(_Symbol, tick)) {
      last_tick_time = tick.time_msc;
   } else {
      last_tick_time = TimeCurrent() * 1000ULL;
   }
   usdx_available = SymbolSelect(USDX_SYMBOL, true);
   usdjpy_available = SymbolSelect(USDJPY_SYMBOL, true);

   LoadHistory();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   PrintRunSummary();
   if(onnx_handle != INVALID_HANDLE) {
      OnnxRelease(onnx_handle);
   }
}

void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal == 0) {
      return;
   }
   if(!HistoryDealSelect(trans.deal)) {
      return;
   }
   if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) {
      return;
   }
   if((int)HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != MAGIC_NUMBER) {
      return;
   }

   long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY && entry != DEAL_ENTRY_INOUT) {
      return;
   }

   double pnl =
      HistoryDealGetDouble(trans.deal, DEAL_PROFIT) +
      HistoryDealGetDouble(trans.deal, DEAL_SWAP) +
      HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   realized_pnl += pnl;
   closed_trade_count++;
   if(pnl > 0.0) {
      closed_win_count++;
   } else if(pnl < 0.0) {
      closed_loss_count++;
   }
}

void DebugPrint(string message) {
   if(DEBUG_LOG) {
      Print("[DEBUG] ", message);
   }
}

string SignalName(int signal) {
   #ifdef USE_NO_HOLD
      if(signal == 0) {
         return "BUY";
      }
      return "SELL";
   #else
      if(signal == 1) {
         return "BUY";
      }
      if(signal == 2) {
         return "SELL";
      }
      return "HOLD";
   #endif
}

ulong BarBucket(ulong time_msc) {
   return time_msc / PRIMARY_BAR_MILLISECONDS;
}

ulong BarOpenTime(ulong bar_bucket) {
   return bar_bucket * PRIMARY_BAR_MILLISECONDS;
}

void StartBar(MqlTick &tick, ulong bar_bucket) {
   current_bar.o = tick.bid;
   current_bar.h = tick.bid;
   current_bar.l = tick.bid;
   current_bar.c = tick.bid;
   current_bar.spread = tick.ask - tick.bid;
   current_bar.spread_mean = 0.0;
   current_bar.tick_imbalance = 0.0;
   current_bar.tick_count = 0;
   current_bar.usdx_bid = 0.0;
   current_bar.usdjpy_bid = 0.0;
   current_bar.atr_feature = 0.0;
   current_bar.atr_trade = 0.0;
   current_bar.time_open_msc = tick.time_msc;
   current_bar.time_close_msc = tick.time_msc;
   current_bar.valid = false;
   ticks_in_bar = 0;
   tick_imbalance_sum = 0.0;
   spread_sum = 0.0;
   current_bar_bucket = bar_bucket;
   bar_started = true;
}

void StartImbalanceBar(MqlTick &tick) {
   current_bar.o = tick.bid;
   current_bar.h = tick.bid;
   current_bar.l = tick.bid;
   current_bar.c = tick.bid;
   current_bar.spread = tick.ask - tick.bid;
   current_bar.spread_mean = 0.0;
   current_bar.tick_imbalance = 0.0;
   current_bar.tick_count = 0;
   current_bar.usdx_bid = 0.0;
   current_bar.usdjpy_bid = 0.0;
   current_bar.atr_feature = 0.0;
   current_bar.atr_trade = 0.0;
   current_bar.time_open_msc = tick.time_msc;
   current_bar.time_close_msc = tick.time_msc;
   current_bar.valid = false;
   ticks_in_bar = 0;
   tick_imbalance_sum = 0.0;
   spread_sum = 0.0;
   current_bar_bucket = 0;
   bar_started = true;
}

bool RollFixedTimeBarIfNeeded(ulong next_bar_bucket, int &closed_tick_count) {
   closed_tick_count = 0;
   if(!bar_started || next_bar_bucket == current_bar_bucket) {
      return false;
   }

   closed_tick_count = ticks_in_bar;
   CloseBar();
   return true;
}

double ResolveImbalanceThresholdBase() {
   if(USE_IMBALANCE_EMA_THRESHOLD || !USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD) {
      return MathMax(2.0, (double)IMBALANCE_MIN_TICKS);
   }
   return MathMax(2.0, (double)MathMax(2, IMBALANCE_MIN_TICKS / 3));
}

double StopDistance() {
   if(R) {
      return FIXED_MOVE;
   }
   return history[0].atr_trade * SL_MULTIPLIER;
}

double TargetDistance() {
   if(R) {
      return FIXED_MOVE;
   }
   return history[0].atr_trade * TP_MULTIPLIER;
}

double ResolveMinimumVolume() {
   double fallback_min_volume = BROKER_MIN_LOT_SIZE;
   if(fallback_min_volume <= 0.0) {
      Print("[WARN] BROKER_MIN_LOT_SIZE is <= 0. Falling back to 0.01 lots.");
      fallback_min_volume = 0.01;
   }

   if(!USE_BROKER_MIN_LOT) {
      return fallback_min_volume;
   }

   double broker_min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(broker_min_volume > 0.0) {
      return broker_min_volume;
   }

   Print(
      StringFormat(
         "[WARN] SYMBOL_VOLUME_MIN lookup failed or returned %.8f. Falling back to %.8f.",
         broker_min_volume,
         fallback_min_volume
      )
   );
   return fallback_min_volume;
}

double NormalizeVolume(double volume) {
   double min_volume = ResolveMinimumVolume();
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(min_volume <= 0.0 || max_volume <= 0.0) {
      return 0.0;
   }
   if(step <= 0.0) {
      step = min_volume;
   }
   if(volume < min_volume - 1e-12) {
      return 0.0;
   }

   volume = MathMin(volume, max_volume);
   double steps = MathFloor(volume / step + 1e-9);
   double normalized = steps * step;
   if(normalized < min_volume) {
      return 0.0;
   }
   if(normalized > max_volume) {
      normalized = max_volume;
   }
   return NormalizeDouble(normalized, 8);
}

double CalculateTradeVolume(int signal, double price, double sl) {
   if(!USE_RISK_PERCENT_INPUT) {
      return NormalizeVolume(LOT_SIZE);
   }

   double risk_amount = AccountInfoDouble(ACCOUNT_BALANCE) * (RISK_PERCENT / 100.0);
   if(risk_amount <= 0.0) {
      return 0.0;
   }

   double one_lot_pnl = 0.0;
   ENUM_ORDER_TYPE order_type = (signal == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   if(!OrderCalcProfit(order_type, _Symbol, 1.0, price, sl, one_lot_pnl)) {
      DebugPrint(
         StringFormat(
            "skip trade: OrderCalcProfit failed retcode=%d last_error=%d",
            trade.ResultRetcode(),
            GetLastError()
         )
      );
      return 0.0;
   }

   double one_lot_loss = MathAbs(one_lot_pnl);
   if(one_lot_loss <= 0.0) {
      return 0.0;
   }

   double volume = risk_amount / one_lot_loss;
   if(USE_LOT_SIZE_CAP_INPUT) {
      double manual_cap = NormalizeVolume(LOT_SIZE_CAP);
      if(manual_cap > 0.0) {
         volume = MathMin(volume, manual_cap);
      }
   }
   return NormalizeVolume(volume);
}

void PrintRunSummary() {
   Print(
      StringFormat(
         "[SUMMARY] bar_mode=%s risk_mode=%s fixed_move=%.2f risk_pct=%.3f predictions=%d hold_skips=%d confidence_skips=%d position_skips=%d stops_too_close=%d volume_skips=%d open_failures=%d trades_opened=%d trades_closed=%d wins=%d losses=%d realized_pnl=%.2f balance=%.2f",
         (MODEL_USE_FIXED_TICK_BARS != 0 ? "FIXED_TICK" : (MODEL_USE_FIXED_TIME_BARS != 0 ? "FIXED_TIME" : "IMBALANCE")),
         (R ? "FIXED" : "ATR"),
         FIXED_MOVE,
         RISK_PERCENT,
         prediction_count,
         hold_skip_count,
         confidence_skip_count,
         position_skip_count,
         stops_too_close_skip_count,
         volume_skip_count,
         trade_open_failed_count,
         trades_opened_count,
         closed_trade_count,
         closed_win_count,
         closed_loss_count,
         realized_pnl,
         AccountInfoDouble(ACCOUNT_BALANCE)
      )
   );
}

int UpdateTickSign(double bid) {
   int sign = last_sign;
   if(last_bid <= 0.0) {
      sign = 1;
   } else {
      double diff = bid - last_bid;
      if(diff > 0.0) {
         sign = 1;
      } else if(diff < 0.0) {
         sign = -1;
      }
   }

   last_bid = bid;
   last_sign = sign;
   return sign;
}

void ProcessTick(MqlTick &tick, ulong bar_bucket) {
   if(tick.bid <= 0.0) {
      return;
   }

   if(!bar_started) {
      if(MODEL_USE_FIXED_TIME_BARS != 0) {
         StartBar(tick, bar_bucket);
      } else {
         StartImbalanceBar(tick);
      }
   }

   int tick_sign = UpdateTickSign(tick.bid);
   current_bar.h = MathMax(current_bar.h, tick.bid);
   current_bar.l = MathMin(current_bar.l, tick.bid);
   current_bar.c = tick.bid;
   current_bar.spread = tick.ask - tick.bid;
   current_bar.time_close_msc = tick.time_msc;
   current_bar.usdx_bid = ResolveAuxBid(USDX_SYMBOL, usdx_available, last_usdx_bid, tick.bid);
   current_bar.usdjpy_bid = ResolveAuxBid(USDJPY_SYMBOL, usdjpy_available, last_usdjpy_bid, tick.bid);
   ticks_in_bar++;
   tick_imbalance_sum += tick_sign;
   spread_sum += current_bar.spread;
}

double ResolveAuxBid(string symbol, bool &available, double &last_value, double fallback) {
   if(!available) {
      return (last_value > 0.0 ? last_value : fallback);
   }
   MqlTick aux;
   if(SymbolInfoTick(symbol, aux) && aux.bid > 0.0) {
      last_value = aux.bid;
      return aux.bid;
   }
   available = false;
   return (last_value > 0.0 ? last_value : fallback);
}

void UpdateIndicators(Bar &bar) {
   Bar prev = history[0];
   double tr = (warmup_count == 0)
      ? (bar.h - bar.l)
      : MathMax(bar.h - bar.l, MathMax(MathAbs(bar.h - prev.c), MathAbs(bar.l - prev.c)));
   int next_count = warmup_count + 1;

   if(next_count <= FEATURE_ATR_PERIOD) {
      warmup_sum_feature += tr;
      bar.atr_feature = warmup_sum_feature / next_count;
   } else {
      double prev_atr_feature = (prev.atr_feature > 0.0 ? prev.atr_feature : tr);
      bar.atr_feature = prev_atr_feature + (tr - prev_atr_feature) / FEATURE_ATR_PERIOD;
   }

   if(next_count <= TARGET_ATR_PERIOD) {
      warmup_sum_trade += tr;
      bar.atr_trade = warmup_sum_trade / next_count;
   } else {
      double prev_atr_trade = (prev.atr_trade > 0.0 ? prev.atr_trade : tr);
      bar.atr_trade = prev_atr_trade + (tr - prev_atr_trade) / TARGET_ATR_PERIOD;
   }

   warmup_count = next_count;
   bar.valid = (warmup_count >= WARMUP_BARS);
}

bool ShouldClosePrimaryBar(double &observed_abs_theta) {
   if(ticks_in_bar < IMBALANCE_MIN_TICKS) {
      observed_abs_theta = 0.0;
      return false;
   }

   observed_abs_theta = MathAbs(tick_imbalance_sum);
   double threshold = USE_IMBALANCE_EMA_THRESHOLD
      ? primary_expected_abs_theta
      : ResolveImbalanceThresholdBase();
   return (observed_abs_theta >= threshold);
}

void UpdatePrimaryImbalanceThreshold(double observed_abs_theta) {
   if(observed_abs_theta <= 0.0 || !USE_IMBALANCE_EMA_THRESHOLD) {
      return;
   }

   double alpha = 2.0 / (MathMax(1, IMBALANCE_EMA_SPAN) + 1.0);
   double observed = MathMax(2.0, observed_abs_theta);
   primary_expected_abs_theta = (1.0 - alpha) * primary_expected_abs_theta + alpha * observed;
}

void CloseBar() {
   current_bar.tick_imbalance = tick_imbalance_sum / MathMax(1, ticks_in_bar);
   current_bar.tick_count = ticks_in_bar;
   current_bar.spread_mean = spread_sum / MathMax(1, ticks_in_bar);
   UpdateIndicators(current_bar);

   for(int i = HISTORY_SIZE - 1; i > 0; i--) {
      history[i] = history[i - 1];
   }
   history[0] = current_bar;

   ticks_in_bar = 0;
   tick_imbalance_sum = 0.0;
   spread_sum = 0.0;
   current_bar_bucket = 0;
   bar_started = false;
}

void OnTick() {
   MqlTick ticks[];
   int count = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, last_tick_time + 1, 100000);
   if(count <= 0) {
      return;
   }

   for(int i = 0; i < count; i++) {
      if(ticks[i].bid <= 0.0) {
         continue;
      }

      if(MODEL_USE_FIXED_TIME_BARS != 0) {
         ulong tick_bucket = BarBucket(ticks[i].time_msc);
         int closed_tick_count = 0;
         if(RollFixedTimeBarIfNeeded(tick_bucket, closed_tick_count)) {
            DebugPrint(
               StringFormat(
                  "bar closed mode=FIXED_TIME seconds=%d ticks=%d atr_trade=%.5f close=%.5f",
                  PRIMARY_BAR_SECONDS,
                  closed_tick_count,
                  history[0].atr_trade,
                  history[0].c
               )
            );
            if(history[REQUIRED_HISTORY_INDEX].valid) {
               Predict();
            } else {
               DebugPrint(
                  StringFormat(
                     "history not ready yet: need index %d valid before predicting",
                     REQUIRED_HISTORY_INDEX
                  )
               );
            }
         }

         ProcessTick(ticks[i], tick_bucket);
         last_tick_time = ticks[i].time_msc;
         continue;
      }

      if(MODEL_USE_FIXED_TICK_BARS != 0) {
         if(!bar_started) {
            StartImbalanceBar(ticks[i]);
         }
         ProcessTick(ticks[i], 0);
         last_tick_time = ticks[i].time_msc;
         if(ticks_in_bar >= PRIMARY_TICK_DENSITY) {
            int closed_tick_count = ticks_in_bar;
            CloseBar();
            DebugPrint(
               StringFormat(
                  "bar closed mode=FIXED_TICK ticks=%d atr_trade=%.5f close=%.5f",
                  closed_tick_count,
                  history[0].atr_trade,
                  history[0].c
               )
            );
            if(history[REQUIRED_HISTORY_INDEX].valid) {
               Predict();
            } else {
               DebugPrint(
                  StringFormat(
                     "history not ready yet: need index %d valid before predicting",
                     REQUIRED_HISTORY_INDEX
                  )
               );
            }
         }
         continue;
      }

      ProcessTick(ticks[i], 0);
      last_tick_time = ticks[i].time_msc;

      double observed_abs_theta = 0.0;
      if(ShouldClosePrimaryBar(observed_abs_theta)) {
         int closed_tick_count = ticks_in_bar;
         CloseBar();
         UpdatePrimaryImbalanceThreshold(observed_abs_theta);
         DebugPrint(
            StringFormat(
               "bar closed mode=IMBALANCE ticks=%d theta=%.2f next_threshold=%.2f atr_trade=%.5f close=%.5f",
               closed_tick_count,
               observed_abs_theta,
               primary_expected_abs_theta,
               history[0].atr_trade,
               history[0].c
            )
         );
         if(history[REQUIRED_HISTORY_INDEX].valid) {
            Predict();
         } else {
            DebugPrint(
               StringFormat(
                  "history not ready yet: need index %d valid before predicting",
                  REQUIRED_HISTORY_INDEX
               )
            );
         }
      }
   }
}

void LoadHistory() {
   ulong start_time_msc = (TimeCurrent() - 86400 * 3) * 1000ULL;
   MqlTick ticks[];
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, start_time_msc, 250000);
   if(copied <= 0) {
      start_time_msc = (TimeCurrent() - 86400) * 1000ULL;
      copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, start_time_msc, 250000);
   }
   if(copied <= 0) {
      return;
   }

   last_tick_time = ticks[0].time_msc - 1;
   for(int i = 0; i < copied; i++) {
      if(ticks[i].bid <= 0.0) {
         continue;
      }

      if(MODEL_USE_FIXED_TIME_BARS != 0) {
         ulong tick_bucket = BarBucket(ticks[i].time_msc);
         int closed_tick_count = 0;
         RollFixedTimeBarIfNeeded(tick_bucket, closed_tick_count);
         ProcessTick(ticks[i], tick_bucket);
         last_tick_time = ticks[i].time_msc;
         continue;
      }

      if(MODEL_USE_FIXED_TICK_BARS != 0) {
         if(!bar_started) {
            StartImbalanceBar(ticks[i]);
         }
         ProcessTick(ticks[i], 0);
         last_tick_time = ticks[i].time_msc;
         if(ticks_in_bar >= PRIMARY_TICK_DENSITY) {
            CloseBar();
         }
         continue;
      }

      ProcessTick(ticks[i], 0);
      last_tick_time = ticks[i].time_msc;

      double observed_abs_theta = 0.0;
      if(ShouldClosePrimaryBar(observed_abs_theta)) {
         CloseBar();
         UpdatePrimaryImbalanceThreshold(observed_abs_theta);
      }
   }
}

float ScaleAndClip(float value, int feature_index) {
   float iqr = (iqrs[feature_index] > 1e-6f ? iqrs[feature_index] : 1.0f);
   float scaled = (value - medians[feature_index]) / iqr;
   return MathMax(-10.0f, MathMin(10.0f, scaled));
}

double SafeLogRatio(double num, double den) {
   return MathLog((num + 1e-10) / (den + 1e-10));
}

double LogReturnAt(int h) {
   return SafeLogRatio(history[h].c, history[h + 1].c);
}

double ReturnOverBars(int h, int bars) {
   return SafeLogRatio(history[h].c, history[h + bars].c);
}

double RollingStdReturn(int h, int window) {
   double values[RV_PERIOD];
   double mean = 0.0;
   for(int i = 0; i < window; i++) {
      values[i] = LogReturnAt(h + i);
      mean += values[i];
   }
   mean /= window;

   double var = 0.0;
   for(int i = 0; i < window; i++) {
      double diff = values[i] - mean;
      var += diff * diff;
   }
   return MathSqrt(var / window);
}

double MeanClose(int h, int window) {
   double sum = 0.0;
   for(int i = 0; i < window; i++) {
      sum += history[h + i].c;
   }
   return sum / window;
}

double StdClose(int h, int window) {
   double mean = MeanClose(h, window);
   double var = 0.0;
   for(int i = 0; i < window; i++) {
      double diff = history[h + i].c - mean;
      var += diff * diff;
   }
   return MathSqrt(var / window);
}

double MaxHigh(int h, int window) {
   double maxv = history[h].h;
   for(int i = 1; i < window; i++) {
      maxv = MathMax(maxv, history[h + i].h);
   }
   return maxv;
}

double MinLow(int h, int window) {
   double minv = history[h].l;
   for(int i = 1; i < window; i++) {
      minv = MathMin(minv, history[h + i].l);
   }
   return minv;
}

double MeanTickCount(int h, int window) {
   double sum = 0.0;
   for(int i = 0; i < window; i++) {
      sum += history[h + i].tick_count;
   }
   return sum / window;
}

double StdTickCount(int h, int window) {
   double mean = MeanTickCount(h, window);
   double var = 0.0;
   for(int i = 0; i < window; i++) {
      double diff = history[h + i].tick_count - mean;
      var += diff * diff;
   }
   return MathSqrt(var / window);
}

double MeanTickImbalance(int h, int window) {
   double sum = 0.0;
   for(int i = 0; i < window; i++) {
      sum += history[h + i].tick_imbalance;
   }
   return sum / window;
}

double MeanSpreadRel(int h, int window) {
   double sum = 0.0;
   for(int i = 0; i < window; i++) {
      double close = history[h + i].c;
      double spread_rel = history[h + i].spread / (close + 1e-10);
      sum += spread_rel;
   }
   return sum / window;
}

double StdSpreadRel(int h, int window) {
   double mean = MeanSpreadRel(h, window);
   double var = 0.0;
   for(int i = 0; i < window; i++) {
      double close = history[h + i].c;
      double spread_rel = history[h + i].spread / (close + 1e-10);
      double diff = spread_rel - mean;
      var += diff * diff;
   }
   return MathSqrt(var / window);
}

double MeanAtrFeature(int h, int window) {
   double sum = 0.0;
   for(int i = 0; i < window; i++) {
      sum += history[h + i].atr_feature;
   }
   return sum / window;
}

double TrueRangeAt(int h) {
   Bar bar = history[h];
   if(h + 1 > REQUIRED_HISTORY_INDEX) {
      return bar.h - bar.l;
   }
   double prev_close = history[h + 1].c;
   return MathMax(bar.h - bar.l, MathMax(MathAbs(bar.h - prev_close), MathAbs(bar.l - prev_close)));
}

double SimpleAtr(int h, int period) {
   double sum = 0.0;
   for(int i = 0; i < period; i++) {
      sum += TrueRangeAt(h + i);
   }
   return sum / period;
}

double EmaClose(int h, int period) {
   int oldest = MathMin(REQUIRED_HISTORY_INDEX, h + period - 1);
   double alpha = 2.0 / (period + 1.0);
   double ema = history[oldest].c;
   for(int i = oldest - 1; i >= h; i--) {
      ema = alpha * history[i].c + (1.0 - alpha) * ema;
   }
   return ema;
}

void MacdAt(int h, double &line, double &signal, double &hist) {
   int oldest = MathMin(REQUIRED_HISTORY_INDEX, h + FEATURE_MACD_SLOW_PERIOD + FEATURE_MACD_SIGNAL_PERIOD - 2);
   double fast_alpha = 2.0 / (FEATURE_MACD_FAST_PERIOD + 1.0);
   double slow_alpha = 2.0 / (FEATURE_MACD_SLOW_PERIOD + 1.0);
   double signal_alpha = 2.0 / (FEATURE_MACD_SIGNAL_PERIOD + 1.0);
   double fast_ema = history[oldest].c;
   double slow_ema = history[oldest].c;
   double signal_ema = 0.0;
   bool signal_ready = false;

   for(int i = oldest; i >= h; i--) {
      if(i != oldest) {
         fast_ema = fast_alpha * history[i].c + (1.0 - fast_alpha) * fast_ema;
         slow_ema = slow_alpha * history[i].c + (1.0 - slow_alpha) * slow_ema;
      }
      double current_line = fast_ema - slow_ema;
      if(!signal_ready) {
         signal_ema = current_line;
         signal_ready = true;
      } else {
         signal_ema = signal_alpha * current_line + (1.0 - signal_alpha) * signal_ema;
      }
      if(i == h) {
         line = current_line;
         signal = signal_ema;
         hist = current_line - signal_ema;
         return;
      }
   }

   line = 0.0;
   signal = 0.0;
   hist = 0.0;
}

double TypicalPrice(int h) {
   return (history[h].h + history[h].l + history[h].c) / 3.0;
}

double SimpleCci(int h, int period) {
   double typicals[512];
   double mean = 0.0;
   for(int i = 0; i < period; i++) {
      typicals[i] = TypicalPrice(h + i);
      mean += typicals[i];
   }
   mean /= period;

   double mean_deviation = 0.0;
   for(int i = 0; i < period; i++) {
      mean_deviation += MathAbs(typicals[i] - mean);
   }
   mean_deviation /= period;
   return (typicals[0] - mean) / (0.015 * (mean_deviation + 1e-10));
}

double WilliamsR(int h, int period) {
   double high = MaxHigh(h, period);
   double low = MinLow(h, period);
   return -100.0 * (high - history[h].c) / (high - low + 1e-10);
}

double SimpleRsi(int h, int period) {
   double gain = 0.0;
   double loss = 0.0;
   for(int i = 0; i < period; i++) {
      double delta = history[h + i].c - history[h + i + 1].c;
      if(delta > 0.0) {
         gain += delta;
      } else if(delta < 0.0) {
         loss -= delta;
      }
   }
   double avg_gain = gain / period;
   double avg_loss = loss / period;
   double rs = avg_gain / (avg_loss + 1e-10);
   return (100.0 - (100.0 / (1.0 + rs)) - 50.0) / 50.0;
}

double StochK(int h, int period) {
   double high = MaxHigh(h, period);
   double low = MinLow(h, period);
   return (history[h].c - low) / (high - low + 1e-10);
}

double StochD(int h, int period) {
   double sum = 0.0;
   for(int i = 0; i < period; i++) {
      sum += StochK(h + i, FEATURE_STOCH_PERIOD);
   }
   return sum / period;
}

void ExtractFeatures(int h, float &features[]) {
   Bar bar = history[h];
   Bar prev = history[h + 1];
   double close = bar.c;

   features[FEATURE_IDX_RET1] = ScaleAndClip((float)LogReturnAt(h), FEATURE_IDX_RET1);
   features[FEATURE_IDX_HIGH_REL_PREV] = ScaleAndClip((float)SafeLogRatio(bar.h, prev.c), FEATURE_IDX_HIGH_REL_PREV);
   features[FEATURE_IDX_LOW_REL_PREV] = ScaleAndClip((float)SafeLogRatio(bar.l, prev.c), FEATURE_IDX_LOW_REL_PREV);
   features[FEATURE_IDX_SPREAD_REL] = ScaleAndClip((float)(bar.spread / (close + 1e-10)), FEATURE_IDX_SPREAD_REL);
   features[FEATURE_IDX_CLOSE_IN_RANGE] = ScaleAndClip(
      (float)((close - bar.l) / (bar.h - bar.l + 1e-8)),
      FEATURE_IDX_CLOSE_IN_RANGE
   );
   features[FEATURE_IDX_ATR_REL] = ScaleAndClip((float)(bar.atr_feature / (close + 1e-10)), FEATURE_IDX_ATR_REL);
   features[FEATURE_IDX_RV] = ScaleAndClip((float)RollingStdReturn(h, RV_PERIOD), FEATURE_IDX_RV);
   features[FEATURE_IDX_RETURN_N] = ScaleAndClip((float)ReturnOverBars(h, RETURN_PERIOD), FEATURE_IDX_RETURN_N);
   features[FEATURE_IDX_TICK_IMBALANCE] = ScaleAndClip((float)bar.tick_imbalance, FEATURE_IDX_TICK_IMBALANCE);
#ifdef FEATURE_IDX_RET_2
   features[FEATURE_IDX_RET_2] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_2_PERIOD), FEATURE_IDX_RET_2);
#endif
#ifdef FEATURE_IDX_RET_3
   features[FEATURE_IDX_RET_3] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_3_PERIOD), FEATURE_IDX_RET_3);
#endif
#ifdef FEATURE_IDX_RET_6
   features[FEATURE_IDX_RET_6] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_6_PERIOD), FEATURE_IDX_RET_6);
#endif
#ifdef FEATURE_IDX_RET_12
   features[FEATURE_IDX_RET_12] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_12_PERIOD), FEATURE_IDX_RET_12);
#endif
#ifdef FEATURE_IDX_RET_20
   features[FEATURE_IDX_RET_20] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_20_PERIOD), FEATURE_IDX_RET_20);
#endif
#ifdef FEATURE_IDX_RANGE_REL
   features[FEATURE_IDX_RANGE_REL] = ScaleAndClip((float)((bar.h - bar.l) / (close + 1e-10)), FEATURE_IDX_RANGE_REL);
#endif
#ifdef FEATURE_IDX_BODY_REL
   features[FEATURE_IDX_BODY_REL] = ScaleAndClip((float)((bar.c - bar.o) / (close + 1e-10)), FEATURE_IDX_BODY_REL);
#endif
#ifdef FEATURE_IDX_UPPER_WICK_REL
   features[FEATURE_IDX_UPPER_WICK_REL] = ScaleAndClip(
      (float)((bar.h - MathMax(bar.o, bar.c)) / (close + 1e-10)),
      FEATURE_IDX_UPPER_WICK_REL
   );
#endif
#ifdef FEATURE_IDX_LOWER_WICK_REL
   features[FEATURE_IDX_LOWER_WICK_REL] = ScaleAndClip(
      (float)((MathMin(bar.o, bar.c) - bar.l) / (close + 1e-10)),
      FEATURE_IDX_LOWER_WICK_REL
   );
#endif
#ifdef FEATURE_IDX_CLOSE_REL_SMA_9
   features[FEATURE_IDX_CLOSE_REL_SMA_9] = ScaleAndClip(
      (float)SafeLogRatio(close, MeanClose(h, FEATURE_SMA_MID_PERIOD)),
      FEATURE_IDX_CLOSE_REL_SMA_9
   );
#endif
#ifdef FEATURE_IDX_CLOSE_REL_SMA_20
   features[FEATURE_IDX_CLOSE_REL_SMA_20] = ScaleAndClip(
      (float)SafeLogRatio(close, MeanClose(h, FEATURE_SMA_SLOW_PERIOD)),
      FEATURE_IDX_CLOSE_REL_SMA_20
   );
#endif
#ifdef FEATURE_IDX_CLOSE_REL_SMA_3
   features[FEATURE_IDX_CLOSE_REL_SMA_3] = ScaleAndClip(
      (float)SafeLogRatio(close, MeanClose(h, FEATURE_SMA_FAST_PERIOD)),
      FEATURE_IDX_CLOSE_REL_SMA_3
   );
#endif
#ifdef FEATURE_IDX_SMA_3_9_GAP
   features[FEATURE_IDX_SMA_3_9_GAP] = ScaleAndClip(
      (float)SafeLogRatio(MeanClose(h, FEATURE_SMA_FAST_PERIOD), MeanClose(h, FEATURE_SMA_MID_PERIOD)),
      FEATURE_IDX_SMA_3_9_GAP
   );
#endif
#ifdef FEATURE_IDX_SMA_5_20_GAP
   features[FEATURE_IDX_SMA_5_20_GAP] = ScaleAndClip(
      (float)SafeLogRatio(MeanClose(h, FEATURE_SMA_TREND_FAST_PERIOD), MeanClose(h, FEATURE_SMA_SLOW_PERIOD)),
      FEATURE_IDX_SMA_5_20_GAP
   );
#endif
#ifdef FEATURE_IDX_SMA_9_20_GAP
   features[FEATURE_IDX_SMA_9_20_GAP] = ScaleAndClip(
      (float)SafeLogRatio(MeanClose(h, FEATURE_SMA_MID_PERIOD), MeanClose(h, FEATURE_SMA_SLOW_PERIOD)),
      FEATURE_IDX_SMA_9_20_GAP
   );
#endif
#ifdef FEATURE_IDX_SMA_SLOPE_9
   features[FEATURE_IDX_SMA_SLOPE_9] = ScaleAndClip(
      (float)SafeLogRatio(
         MeanClose(h, FEATURE_SMA_MID_PERIOD),
         MeanClose(h + FEATURE_SMA_SLOPE_SHIFT, FEATURE_SMA_MID_PERIOD)
      ),
      FEATURE_IDX_SMA_SLOPE_9
   );
#endif
#ifdef FEATURE_IDX_SMA_SLOPE_20
   features[FEATURE_IDX_SMA_SLOPE_20] = ScaleAndClip(
      (float)SafeLogRatio(
         MeanClose(h, FEATURE_SMA_SLOW_PERIOD),
         MeanClose(h + FEATURE_SMA_SLOPE_SHIFT, FEATURE_SMA_SLOW_PERIOD)
      ),
      FEATURE_IDX_SMA_SLOPE_20
   );
#endif
#ifdef FEATURE_IDX_RSI_6
   features[FEATURE_IDX_RSI_6] = ScaleAndClip((float)SimpleRsi(h, FEATURE_RSI_FAST_PERIOD), FEATURE_IDX_RSI_6);
#endif
#ifdef FEATURE_IDX_RSI_14
   features[FEATURE_IDX_RSI_14] = ScaleAndClip((float)SimpleRsi(h, FEATURE_RSI_SLOW_PERIOD), FEATURE_IDX_RSI_14);
#endif
#ifdef FEATURE_IDX_STOCH_K_9
   features[FEATURE_IDX_STOCH_K_9] = ScaleAndClip((float)StochK(h, FEATURE_STOCH_PERIOD), FEATURE_IDX_STOCH_K_9);
#endif
#ifdef FEATURE_IDX_STOCH_D_3
   features[FEATURE_IDX_STOCH_D_3] = ScaleAndClip((float)StochD(h, FEATURE_STOCH_SMOOTH_PERIOD), FEATURE_IDX_STOCH_D_3);
#endif
#ifdef FEATURE_IDX_STOCH_GAP
   features[FEATURE_IDX_STOCH_GAP] = ScaleAndClip(
      (float)(StochK(h, FEATURE_STOCH_PERIOD) - StochD(h, FEATURE_STOCH_SMOOTH_PERIOD)),
      FEATURE_IDX_STOCH_GAP
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_POS_20
   {
      double sma20 = MeanClose(h, FEATURE_BOLLINGER_PERIOD);
      double std20 = StdClose(h, FEATURE_BOLLINGER_PERIOD);
      features[FEATURE_IDX_BOLLINGER_POS_20] = ScaleAndClip(
         (float)((close - sma20) / (2.0 * std20 + 1e-10)),
         FEATURE_IDX_BOLLINGER_POS_20
      );
   }
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_20
   {
      double sma20 = MeanClose(h, FEATURE_BOLLINGER_PERIOD);
      double std20 = StdClose(h, FEATURE_BOLLINGER_PERIOD);
      features[FEATURE_IDX_BOLLINGER_WIDTH_20] = ScaleAndClip(
         (float)((4.0 * std20) / (sma20 + 1e-10)),
         FEATURE_IDX_BOLLINGER_WIDTH_20
      );
   }
#endif
#ifdef FEATURE_IDX_ATR_RATIO_20
   {
      double mean_atr = MeanAtrFeature(h, FEATURE_ATR_RATIO_PERIOD);
      features[FEATURE_IDX_ATR_RATIO_20] = ScaleAndClip(
         (float)SafeLogRatio(bar.atr_feature, mean_atr),
         FEATURE_IDX_ATR_RATIO_20
      );
   }
#endif
#ifdef FEATURE_IDX_RV_18
   features[FEATURE_IDX_RV_18] = ScaleAndClip((float)RollingStdReturn(h, FEATURE_RV_LONG_PERIOD), FEATURE_IDX_RV_18);
#endif
#ifdef FEATURE_IDX_DONCHIAN_POS_20
   {
      double high20 = MaxHigh(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      double low20 = MinLow(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      features[FEATURE_IDX_DONCHIAN_POS_20] = ScaleAndClip(
         (float)((close - low20) / (high20 - low20 + 1e-10)),
         FEATURE_IDX_DONCHIAN_POS_20
      );
   }
#endif
#ifdef FEATURE_IDX_DONCHIAN_POS_9
   {
      double high9 = MaxHigh(h, FEATURE_DONCHIAN_FAST_PERIOD);
      double low9 = MinLow(h, FEATURE_DONCHIAN_FAST_PERIOD);
      features[FEATURE_IDX_DONCHIAN_POS_9] = ScaleAndClip(
         (float)((close - low9) / (high9 - low9 + 1e-10)),
         FEATURE_IDX_DONCHIAN_POS_9
      );
   }
#endif
#ifdef FEATURE_IDX_DONCHIAN_WIDTH_9
   {
      double high9 = MaxHigh(h, FEATURE_DONCHIAN_FAST_PERIOD);
      double low9 = MinLow(h, FEATURE_DONCHIAN_FAST_PERIOD);
      features[FEATURE_IDX_DONCHIAN_WIDTH_9] = ScaleAndClip(
         (float)((high9 - low9) / (close + 1e-10)),
         FEATURE_IDX_DONCHIAN_WIDTH_9
      );
   }
#endif
#ifdef FEATURE_IDX_DONCHIAN_WIDTH_20
   {
      double high20 = MaxHigh(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      double low20 = MinLow(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      features[FEATURE_IDX_DONCHIAN_WIDTH_20] = ScaleAndClip(
         (float)((high20 - low20) / (close + 1e-10)),
         FEATURE_IDX_DONCHIAN_WIDTH_20
      );
   }
#endif
#ifdef FEATURE_IDX_TICK_COUNT_Z_9
   {
      double mean_tc = MeanTickCount(h, FEATURE_TICK_COUNT_PERIOD);
      double std_tc = StdTickCount(h, FEATURE_TICK_COUNT_PERIOD);
      double z = (std_tc > 1e-10) ? ((bar.tick_count - mean_tc) / std_tc) : 0.0;
      features[FEATURE_IDX_TICK_COUNT_Z_9] = ScaleAndClip((float)z, FEATURE_IDX_TICK_COUNT_Z_9);
   }
#endif
#ifdef FEATURE_IDX_TICK_COUNT_REL_9
   {
      double mean_tc = MeanTickCount(h, FEATURE_TICK_COUNT_PERIOD);
      features[FEATURE_IDX_TICK_COUNT_REL_9] = ScaleAndClip(
         (float)(bar.tick_count / (mean_tc + 1e-10) - 1.0),
         FEATURE_IDX_TICK_COUNT_REL_9
      );
   }
#endif
#ifdef FEATURE_IDX_TICK_COUNT_CHG
   features[FEATURE_IDX_TICK_COUNT_CHG] = ScaleAndClip(
      (float)MathLog((bar.tick_count + 1.0) / (history[h + 1].tick_count + 1.0)),
      FEATURE_IDX_TICK_COUNT_CHG
   );
#endif
#ifdef FEATURE_IDX_TICK_IMBALANCE_SMA_9
   features[FEATURE_IDX_TICK_IMBALANCE_SMA_9] = ScaleAndClip(
      (float)MeanTickImbalance(h, FEATURE_TICK_IMBALANCE_SLOW_PERIOD),
      FEATURE_IDX_TICK_IMBALANCE_SMA_9
   );
#endif
#ifdef FEATURE_IDX_TICK_IMBALANCE_SMA_5
   features[FEATURE_IDX_TICK_IMBALANCE_SMA_5] = ScaleAndClip(
      (float)MeanTickImbalance(h, FEATURE_TICK_IMBALANCE_FAST_PERIOD),
      FEATURE_IDX_TICK_IMBALANCE_SMA_5
   );
#endif
#ifdef FEATURE_IDX_SPREAD_Z_9
   {
      double mean_spread = MeanSpreadRel(h, FEATURE_SPREAD_Z_PERIOD);
      double std_spread = StdSpreadRel(h, FEATURE_SPREAD_Z_PERIOD);
      double spread_rel = bar.spread / (close + 1e-10);
      double z = (std_spread > 1e-10) ? ((spread_rel - mean_spread) / std_spread) : 0.0;
      features[FEATURE_IDX_SPREAD_Z_9] = ScaleAndClip((float)z, FEATURE_IDX_SPREAD_Z_9);
   }
#endif
#ifdef FEATURE_IDX_USDX_RET1
   features[FEATURE_IDX_USDX_RET1] = ScaleAndClip(
      (float)SafeLogRatio(bar.usdx_bid, prev.usdx_bid),
      FEATURE_IDX_USDX_RET1
   );
#endif
#ifdef FEATURE_IDX_USDJPY_RET1
   features[FEATURE_IDX_USDJPY_RET1] = ScaleAndClip(
      (float)SafeLogRatio(bar.usdjpy_bid, prev.usdjpy_bid),
      FEATURE_IDX_USDJPY_RET1
   );
#endif
#ifdef FEATURE_IDX_SPREAD_ABS
   features[FEATURE_IDX_SPREAD_ABS] = ScaleAndClip((float)bar.spread_mean, FEATURE_IDX_SPREAD_ABS);
#endif
#ifdef FEATURE_IDX_BAR_DURATION_MS
   features[FEATURE_IDX_BAR_DURATION_MS] = ScaleAndClip(
      (float)(bar.time_close_msc - bar.time_open_msc),
      FEATURE_IDX_BAR_DURATION_MS
   );
#endif
#ifdef FEATURE_IDX_RSI_9
   features[FEATURE_IDX_RSI_9] = ScaleAndClip((float)SimpleRsi(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_RSI_9);
#endif
#ifdef FEATURE_IDX_RSI_18
   features[FEATURE_IDX_RSI_18] = ScaleAndClip((float)SimpleRsi(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_RSI_18);
#endif
#ifdef FEATURE_IDX_RSI_27
   features[FEATURE_IDX_RSI_27] = ScaleAndClip((float)SimpleRsi(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_RSI_27);
#endif
#ifdef FEATURE_IDX_ATR_9
   features[FEATURE_IDX_ATR_9] = ScaleAndClip((float)SimpleAtr(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_ATR_9);
#endif
#ifdef FEATURE_IDX_ATR_18
   features[FEATURE_IDX_ATR_18] = ScaleAndClip((float)SimpleAtr(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_ATR_18);
#endif
#ifdef FEATURE_IDX_ATR_27
   features[FEATURE_IDX_ATR_27] = ScaleAndClip((float)SimpleAtr(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_ATR_27);
#endif
#if defined(FEATURE_IDX_MACD_LINE) || defined(FEATURE_IDX_MACD_SIGNAL) || defined(FEATURE_IDX_MACD_HIST)
   {
      double macd_line = 0.0;
      double macd_signal = 0.0;
      double macd_hist = 0.0;
      MacdAt(h, macd_line, macd_signal, macd_hist);
      #ifdef FEATURE_IDX_MACD_LINE
         features[FEATURE_IDX_MACD_LINE] = ScaleAndClip((float)macd_line, FEATURE_IDX_MACD_LINE);
      #endif
      #ifdef FEATURE_IDX_MACD_SIGNAL
         features[FEATURE_IDX_MACD_SIGNAL] = ScaleAndClip((float)macd_signal, FEATURE_IDX_MACD_SIGNAL);
      #endif
      #ifdef FEATURE_IDX_MACD_HIST
         features[FEATURE_IDX_MACD_HIST] = ScaleAndClip((float)macd_hist, FEATURE_IDX_MACD_HIST);
      #endif
   }
#endif
#ifdef FEATURE_IDX_EMA_GAP_9
   features[FEATURE_IDX_EMA_GAP_9] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_SHORT_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_9
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_18
   features[FEATURE_IDX_EMA_GAP_18] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_MEDIUM_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_18
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_27
   features[FEATURE_IDX_EMA_GAP_27] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_LONG_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_27
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_54
   features[FEATURE_IDX_EMA_GAP_54] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_XLONG_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_54
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_144
   features[FEATURE_IDX_EMA_GAP_144] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_XXLONG_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_144
   );
#endif
#ifdef FEATURE_IDX_CCI_9
   features[FEATURE_IDX_CCI_9] = ScaleAndClip((float)SimpleCci(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_CCI_9);
#endif
#ifdef FEATURE_IDX_CCI_18
   features[FEATURE_IDX_CCI_18] = ScaleAndClip((float)SimpleCci(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_CCI_18);
#endif
#ifdef FEATURE_IDX_CCI_27
   features[FEATURE_IDX_CCI_27] = ScaleAndClip((float)SimpleCci(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_CCI_27);
#endif
#ifdef FEATURE_IDX_WILLR_9
   features[FEATURE_IDX_WILLR_9] = ScaleAndClip((float)WilliamsR(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_WILLR_9);
#endif
#ifdef FEATURE_IDX_WILLR_18
   features[FEATURE_IDX_WILLR_18] = ScaleAndClip((float)WilliamsR(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_WILLR_18);
#endif
#ifdef FEATURE_IDX_WILLR_27
   features[FEATURE_IDX_WILLR_27] = ScaleAndClip((float)WilliamsR(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_WILLR_27);
#endif
#ifdef FEATURE_IDX_MOM_9
   features[FEATURE_IDX_MOM_9] = ScaleAndClip(
      (float)(bar.c - history[h + FEATURE_MAIN_SHORT_PERIOD].c),
      FEATURE_IDX_MOM_9
   );
#endif
#ifdef FEATURE_IDX_MOM_18
   features[FEATURE_IDX_MOM_18] = ScaleAndClip(
      (float)(bar.c - history[h + FEATURE_MAIN_MEDIUM_PERIOD].c),
      FEATURE_IDX_MOM_18
   );
#endif
#ifdef FEATURE_IDX_MOM_27
   features[FEATURE_IDX_MOM_27] = ScaleAndClip(
      (float)(bar.c - history[h + FEATURE_MAIN_LONG_PERIOD].c),
      FEATURE_IDX_MOM_27
   );
#endif
#ifdef FEATURE_IDX_USDX_PCT_CHANGE
   features[FEATURE_IDX_USDX_PCT_CHANGE] = ScaleAndClip(
      (float)((bar.usdx_bid / (prev.usdx_bid + 1e-10)) - 1.0),
      FEATURE_IDX_USDX_PCT_CHANGE
   );
#endif
#ifdef FEATURE_IDX_USDJPY_PCT_CHANGE
   features[FEATURE_IDX_USDJPY_PCT_CHANGE] = ScaleAndClip(
      (float)((bar.usdjpy_bid / (prev.usdjpy_bid + 1e-10)) - 1.0),
      FEATURE_IDX_USDJPY_PCT_CHANGE
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_9
   features[FEATURE_IDX_BOLLINGER_WIDTH_9] = ScaleAndClip(
      (float)((4.0 * StdClose(h, FEATURE_MAIN_SHORT_PERIOD)) / (MeanClose(h, FEATURE_MAIN_SHORT_PERIOD) + 1e-10)),
      FEATURE_IDX_BOLLINGER_WIDTH_9
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_18
   features[FEATURE_IDX_BOLLINGER_WIDTH_18] = ScaleAndClip(
      (float)((4.0 * StdClose(h, FEATURE_MAIN_MEDIUM_PERIOD)) / (MeanClose(h, FEATURE_MAIN_MEDIUM_PERIOD) + 1e-10)),
      FEATURE_IDX_BOLLINGER_WIDTH_18
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_27
   features[FEATURE_IDX_BOLLINGER_WIDTH_27] = ScaleAndClip(
      (float)((4.0 * StdClose(h, FEATURE_MAIN_LONG_PERIOD)) / (MeanClose(h, FEATURE_MAIN_LONG_PERIOD) + 1e-10)),
      FEATURE_IDX_BOLLINGER_WIDTH_27
   );
#endif
#if defined(FEATURE_IDX_HOUR_SIN) || defined(FEATURE_IDX_HOUR_COS) || defined(FEATURE_IDX_MINUTE_SIN) || defined(FEATURE_IDX_MINUTE_COS) || defined(FEATURE_IDX_DAY_OF_WEEK_SCALED)
   {
      MqlDateTime parts;
      TimeToStruct((datetime)(bar.time_open_msc / 1000ULL), parts);
      double hour_angle = 2.0 * M_PI * parts.hour / 24.0;
      double minute_angle = 2.0 * M_PI * parts.min / 60.0;
      #ifdef FEATURE_IDX_HOUR_SIN
         features[FEATURE_IDX_HOUR_SIN] = ScaleAndClip((float)MathSin(hour_angle), FEATURE_IDX_HOUR_SIN);
      #endif
      #ifdef FEATURE_IDX_HOUR_COS
         features[FEATURE_IDX_HOUR_COS] = ScaleAndClip((float)MathCos(hour_angle), FEATURE_IDX_HOUR_COS);
      #endif
      #ifdef FEATURE_IDX_MINUTE_SIN
         features[FEATURE_IDX_MINUTE_SIN] = ScaleAndClip((float)MathSin(minute_angle), FEATURE_IDX_MINUTE_SIN);
      #endif
      #ifdef FEATURE_IDX_MINUTE_COS
         features[FEATURE_IDX_MINUTE_COS] = ScaleAndClip((float)MathCos(minute_angle), FEATURE_IDX_MINUTE_COS);
      #endif
      #ifdef FEATURE_IDX_DAY_OF_WEEK_SCALED
         features[FEATURE_IDX_DAY_OF_WEEK_SCALED] = ScaleAndClip(
            (float)(parts.day_of_week / 6.0),
            FEATURE_IDX_DAY_OF_WEEK_SCALED
         );
      #endif
   }
#endif
}

void Softmax(const float &logits[], float &probs[]) {
   #ifdef USE_NO_HOLD
      double max_logit = MathMax(logits[0], logits[1]);
      double e0 = MathExp(logits[0] - max_logit);
      double e1 = MathExp(logits[1] - max_logit);
      double sum = e0 + e1;
      probs[0] = (float)(e0 / sum);
      probs[1] = (float)(e1 / sum);
      DebugPrint(StringFormat("binary-softmax e0=%.4f e1=%.4f sum=%.4f probs=[%.4f, %.4f]", e0, e1, sum, probs[0], probs[1]));
   #else
      double max_logit = MathMax(logits[0], MathMax(logits[1], logits[2]));
      double e0 = MathExp(logits[0] - max_logit);
      double e1 = MathExp(logits[1] - max_logit);
      double e2 = MathExp(logits[2] - max_logit);
      double sum = e0 + e1 + e2;
      probs[0] = (float)(e0 / sum);
      probs[1] = (float)(e1 / sum);
      probs[2] = (float)(e2 / sum);
   #endif
}

void Predict() {
   prediction_count++;
   for(int i = 0; i < SEQ_LEN; i++) {
      int h = SEQ_LEN - 1 - i;
      int offset = i * MODEL_FEATURE_COUNT;
      float features[MODEL_FEATURE_COUNT];
      ExtractFeatures(h, features);
      for(int k = 0; k < MODEL_FEATURE_COUNT; k++) {
         input_data[offset + k] = features[k];
      }
   }

   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
      DebugPrint(StringFormat("OnnxRun failed err=%d", GetLastError()));
      return;
   }

   #ifdef USE_NO_HOLD
      float probs[2];
   #else
      float probs[3];
   #endif
   #ifdef USE_NO_HOLD
      DebugPrint("binary-mode USE_NO_HOLD=true");
   #else
      DebugPrint("ternary-mode USE_NO_HOLD=false");
   #endif
   Softmax(output_data, probs);
   int signal = ArrayMaximum(probs);
   #ifdef USE_NO_HOLD
      DebugPrint(
         StringFormat(
            "predict probs=[%.4f, %.4f] signal=%s conf=%.4f",
            probs[0],
            probs[1],
            SignalName(signal),
            probs[signal]
         )
      );
      if(signal < 0 || signal > 1) {
         DebugPrint(StringFormat("ERROR: invalid binary signal %d", signal));
         hold_skip_count++;
         return;
      }
   #else
      DebugPrint(
         StringFormat(
            "predict probs=[%.4f, %.4f, %.4f] signal=%s conf=%.4f",
            probs[0],
            probs[1],
            probs[2],
            SignalName(signal),
            probs[signal]
         )
      );
      if(signal <= 0) {
         hold_skip_count++;
         DebugPrint("skip trade: model chose HOLD");
         return;
      }
   #endif
   if(probs[signal] < PRIMARY_CONFIDENCE) {
      confidence_skip_count++;
      DebugPrint(
         StringFormat(
            "skip trade: confidence %.4f below threshold %.4f",
            probs[signal],
            PRIMARY_CONFIDENCE
         )
      );
      return;
   }

   Execute(signal);
}

void Execute(int signal) {
   if(PositionSelect(_Symbol)) {
      position_skip_count++;
      DebugPrint("skip trade: a position is already open on this symbol");
      return;
   }

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(bid <= 0.0 || ask <= 0.0) {
      trade_open_failed_count++;
      DebugPrint(StringFormat("skip trade: invalid bid/ask bid=%.5f ask=%.5f", bid, ask));
      return;
   }

    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double trigger_price = (signal == 1) ? bid : ask;
    double price = (signal == 1) ? ask : bid;
    double sl_distance = StopDistance();
    double tp_distance = TargetDistance();
    if(sl_distance <= 0.0 || tp_distance <= 0.0) {
       trade_open_failed_count++;
       DebugPrint(
          StringFormat(
             "skip trade: invalid risk distances sl_distance=%.5f tp_distance=%.5f",
             sl_distance,
             tp_distance
          )
       );
       return;
    }
    double sl = (signal == 1)
       ? (trigger_price - sl_distance)
       : (trigger_price + sl_distance);
    double tp = (signal == 1)
       ? (trigger_price + tp_distance)
       : (trigger_price - tp_distance);
    price = NormalizeDouble(price, digits);
    sl = NormalizeDouble(sl, digits);
    tp = NormalizeDouble(tp, digits);

    double min_stop_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
    double freeze_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * point;
    double min_dist = MathMax(min_stop_dist, freeze_dist);
   double sl_gap = (signal == 1) ? (trigger_price - sl) : (sl - trigger_price);
   double tp_gap = (signal == 1) ? (tp - trigger_price) : (trigger_price - tp);
   if(sl_gap < min_dist || tp_gap < min_dist) {
      stops_too_close_skip_count++;
      DebugPrint(
         StringFormat(
            "skip trade: stops too close bid=%.5f ask=%.5f price=%.5f trigger=%.5f sl=%.5f tp=%.5f sl_gap=%.5f tp_gap=%.5f min_dist=%.5f",
            bid,
            ask,
            price,
            trigger_price,
            sl,
            tp,
            sl_gap,
            tp_gap,
            min_dist
         )
      );
      return;
   }

   double volume = CalculateTradeVolume(signal, price, sl);
   DebugPrint(
      StringFormat(
         "risk_pct=%.3f lot_cap=%.2f use_risk_pct=%s use_broker_min_lot=%s",
         RISK_PERCENT,
         LOT_SIZE_CAP,
         (USE_RISK_PERCENT_INPUT ? "true" : "false"),
         (USE_BROKER_MIN_LOT ? "true" : "false")
      )
   );
   if(volume <= 0.0) {
      volume_skip_count++;
      DebugPrint(
         StringFormat(
            "skip trade: no valid volume price=%.5f sl=%.5f risk_pct=%.3f lot_cap=%.2f",
            price,
            sl,
            RISK_PERCENT,
            LOT_SIZE_CAP
         )
      );
      return;
   }

   double sl_pct_change = (sl - price) / price * 100.0;
   double tp_pct_change = (tp - price) / price * 100.0;
   DebugPrint(
      StringFormat(
         "Intent to place trade: volume=%.2f sl=%.5f tp=%.5f sl_pct_change=%.3f%% tp_pct_change=%.3f%%",
         volume,
         sl,
         tp,
         sl_pct_change,
         tp_pct_change
      )
   );

   bool opened = trade.PositionOpen(_Symbol, (signal == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), volume, price, sl, tp);
   if(opened) {
      trades_opened_count++;
      DebugPrint(
         StringFormat(
            "trade opened %s lot=%.2f price=%.5f sl=%.5f tp=%.5f",
            SignalName(signal),
            volume,
            price,
            sl,
            tp
         )
      );
   } else {
      trade_open_failed_count++;
      DebugPrint(
         StringFormat(
            "trade open failed %s retcode=%d last_error=%d",
            SignalName(signal),
            trade.ResultRetcode(),
            GetLastError()
         )
      );
   }
}
