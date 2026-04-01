#include <Trade\Trade.mqh>
#include "gold_model_config.mqh"

#resource "\\Experts\\nn\\gold\\gold_mamba.onnx" as uchar model_buffer[]

#define SEQ_LEN 54
#define FEATURE_COUNT 9
#define REQUIRED_HISTORY_INDEX (SEQ_LEN + 7)
#define HISTORY_SIZE (REQUIRED_HISTORY_INDEX + 1)
#define INPUT_BUFFER_SIZE (SEQ_LEN * MODEL_FEATURE_COUNT)

input double SL_MULTIPLIER = 0.54;
input double TP_MULTIPLIER = 0.54;
input double LOT_SIZE = 0.54;
input int MAGIC_NUMBER = 777777;

long onnx_handle = INVALID_HANDLE;
CTrade trade;

struct Bar {
   double o;
   double h;
   double l;
   double c;
   double spread;
   double tick_imbalance;
   double atr14;
   double atr9;
   ulong time_msc;
   bool valid;
};

Bar history[HISTORY_SIZE];
Bar current_bar;
int ticks_in_bar = 0;
bool bar_started = false;
ulong last_tick_time = 0;
double tick_imbalance_sum = 0.0;
double last_bid = 0.0;
int last_sign = 1;
double primary_expected_abs_theta = 60.0;
int warmup_count = 0;
double warmup_sum14 = 0.0;
double warmup_sum9 = 0.0;
float input_data[INPUT_BUFFER_SIZE];
float output_data[3];

int UpdateTickSign(double bid);
void ProcessTick(MqlTick &tick);
void UpdateIndicators(Bar &bar);
bool ShouldClosePrimaryBar(double &observed_abs_theta);
void UpdatePrimaryImbalanceThreshold(double observed_abs_theta);
void CloseBar();
void LoadHistory();
float ScaleAndClip(float value, int feature_index);
double SafeLogRatio(double num, double den);
double LogReturnAt(int h);
double ReturnOverBars(int h, int bars);
double RollingStdReturn(int h, int window);
void ExtractFeatures(int h, float &features[]);
void Softmax(const float &logits[], float &probs[]);
void Predict();
void Execute(int signal);

int OnInit() {
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
   primary_expected_abs_theta = MathMax(2.0, (double)MathMax(2, IMBALANCE_MIN_TICKS / 3));

   MqlTick tick;
   if(SymbolInfoTick(_Symbol, tick)) {
      last_tick_time = tick.time_msc;
   } else {
      last_tick_time = TimeCurrent() * 1000ULL;
   }

   LoadHistory();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(onnx_handle != INVALID_HANDLE) {
      OnnxRelease(onnx_handle);
   }
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

void ProcessTick(MqlTick &tick) {
   if(tick.bid <= 0.0) {
      return;
   }

   if(!bar_started) {
      current_bar.o = tick.bid;
      current_bar.h = tick.bid;
      current_bar.l = tick.bid;
      current_bar.c = tick.bid;
      current_bar.spread = tick.ask - tick.bid;
      current_bar.tick_imbalance = 0.0;
      current_bar.time_msc = tick.time_msc;
      ticks_in_bar = 0;
      tick_imbalance_sum = 0.0;
      bar_started = true;
   }

   int tick_sign = UpdateTickSign(tick.bid);
   current_bar.h = MathMax(current_bar.h, tick.bid);
   current_bar.l = MathMin(current_bar.l, tick.bid);
   current_bar.c = tick.bid;
   current_bar.spread = tick.ask - tick.bid;
   ticks_in_bar++;
   tick_imbalance_sum += tick_sign;
}

void UpdateIndicators(Bar &bar) {
   Bar prev = history[0];
   double tr = (warmup_count == 0)
      ? (bar.h - bar.l)
      : MathMax(bar.h - bar.l, MathMax(MathAbs(bar.h - prev.c), MathAbs(bar.l - prev.c)));
   int next_count = warmup_count + 1;

   if(next_count <= 14) {
      warmup_sum14 += tr;
      bar.atr14 = warmup_sum14 / next_count;
   } else {
      double prev_atr14 = (prev.atr14 > 0.0 ? prev.atr14 : tr);
      bar.atr14 = prev_atr14 + (tr - prev_atr14) / 14.0;
   }

   if(next_count <= 9) {
      warmup_sum9 += tr;
      bar.atr9 = warmup_sum9 / next_count;
   } else {
      double prev_atr9 = (prev.atr9 > 0.0 ? prev.atr9 : tr);
      bar.atr9 = prev_atr9 + (tr - prev_atr9) / 9.0;
   }

   warmup_count = next_count;
   bar.valid = (warmup_count >= 16);
}

bool ShouldClosePrimaryBar(double &observed_abs_theta) {
   if(ticks_in_bar < IMBALANCE_MIN_TICKS) {
      observed_abs_theta = 0.0;
      return false;
   }

   observed_abs_theta = MathAbs(tick_imbalance_sum);
   return (observed_abs_theta >= primary_expected_abs_theta);
}

void UpdatePrimaryImbalanceThreshold(double observed_abs_theta) {
   if(observed_abs_theta <= 0.0) {
      return;
   }

   double alpha = 2.0 / (MathMax(1, IMBALANCE_EMA_SPAN) + 1.0);
   double observed = MathMax(2.0, observed_abs_theta);
   primary_expected_abs_theta = (1.0 - alpha) * primary_expected_abs_theta + alpha * observed;
}

void CloseBar() {
   current_bar.tick_imbalance = tick_imbalance_sum / MathMax(1, ticks_in_bar);
   UpdateIndicators(current_bar);

   for(int i = HISTORY_SIZE - 1; i > 0; i--) {
      history[i] = history[i - 1];
   }
   history[0] = current_bar;

   ticks_in_bar = 0;
   tick_imbalance_sum = 0.0;
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

      ProcessTick(ticks[i]);
      last_tick_time = ticks[i].time_msc;

      double observed_abs_theta = 0.0;
      if(ShouldClosePrimaryBar(observed_abs_theta)) {
         CloseBar();
         UpdatePrimaryImbalanceThreshold(observed_abs_theta);
         if(history[REQUIRED_HISTORY_INDEX].valid) {
            Predict();
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

      ProcessTick(ticks[i]);
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
   double values[8];
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

void ExtractFeatures(int h, float &features[]) {
   Bar bar = history[h];
   Bar prev = history[h + 1];
   double close = bar.c;

   features[0] = ScaleAndClip((float)LogReturnAt(h), 0);
   features[1] = ScaleAndClip((float)SafeLogRatio(bar.h, prev.c), 1);
   features[2] = ScaleAndClip((float)SafeLogRatio(bar.l, prev.c), 2);
   features[3] = ScaleAndClip((float)(bar.spread / (close + 1e-10)), 3);
   features[4] = ScaleAndClip((float)((close - bar.l) / (bar.h - bar.l + 1e-8)), 4);
   features[5] = ScaleAndClip((float)(bar.atr14 / (close + 1e-10)), 5);
   features[6] = ScaleAndClip((float)RollingStdReturn(h, 4), 6);
   features[7] = ScaleAndClip((float)ReturnOverBars(h, 8), 7);
   features[8] = ScaleAndClip((float)bar.tick_imbalance, 8);
}

void Softmax(const float &logits[], float &probs[]) {
   double max_logit = MathMax(logits[0], MathMax(logits[1], logits[2]));
   double e0 = MathExp(logits[0] - max_logit);
   double e1 = MathExp(logits[1] - max_logit);
   double e2 = MathExp(logits[2] - max_logit);
   double sum = e0 + e1 + e2;
   probs[0] = (float)(e0 / sum);
   probs[1] = (float)(e1 / sum);
   probs[2] = (float)(e2 / sum);
}

void Predict() {
   for(int i = 0; i < SEQ_LEN; i++) {
      int h = SEQ_LEN - 1 - i;
      int offset = i * MODEL_FEATURE_COUNT;
      float features[FEATURE_COUNT];
      ExtractFeatures(h, features);
      for(int k = 0; k < FEATURE_COUNT; k++) {
         input_data[offset + k] = features[k];
      }
   }

   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
      return;
   }

   float probs[3];
   Softmax(output_data, probs);
   int signal = ArrayMaximum(probs);
   if(signal <= 0) {
      return;
   }
   if(probs[signal] < PRIMARY_CONFIDENCE) {
      return;
   }

   Execute(signal);
}

void Execute(int signal) {
   if(PositionSelect(_Symbol)) {
      return;
   }

   double price = (signal == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = (signal == 1) ? (price - history[0].atr9 * SL_MULTIPLIER) : (price + history[0].atr9 * SL_MULTIPLIER);
   double tp = (signal == 1) ? (price + history[0].atr9 * TP_MULTIPLIER) : (price - history[0].atr9 * TP_MULTIPLIER);

   double min_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(MathAbs(price - sl) < min_dist || MathAbs(tp - price) < min_dist) {
      return;
   }

   trade.PositionOpen(_Symbol, (signal == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), LOT_SIZE, price, sl, tp);
}
