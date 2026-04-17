#include <Trade\Trade.mqh>
// @active-model-reference begin
#define ACTIVE_MODEL_SYMBOL "XAUUSD"
#define ACTIVE_MODEL_VERSION "17_04_2026-09_54__09-submi"
#include "symbols/xauusd/models/17_04_2026-09_54__09-submi/config.mqh"
#resource "symbols\\xauusd\\models\\17_04_2026-09_54__09-submi\\model.onnx" as uchar model_buffer[]
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


#include "live/functions/OnInit.mqh"
#include "live/functions/OnDeinit.mqh"
#include "live/functions/OnTradeTransaction.mqh"
#include "live/functions/DebugPrint.mqh"
#include "live/functions/SignalName.mqh"
#include "live/functions/BarBucket.mqh"
#include "live/functions/BarOpenTime.mqh"
#include "live/functions/StartBar.mqh"
#include "live/functions/StartImbalanceBar.mqh"
#include "live/functions/RollFixedTimeBarIfNeeded.mqh"
#include "live/functions/ResolveImbalanceThresholdBase.mqh"
#include "live/functions/StopDistance.mqh"
#include "live/functions/TargetDistance.mqh"
#include "live/functions/ResolveMinimumVolume.mqh"
#include "live/functions/NormalizeVolume.mqh"
#include "live/functions/CalculateTradeVolume.mqh"
#include "live/functions/PrintRunSummary.mqh"
#include "live/functions/UpdateTickSign.mqh"
#include "live/functions/ProcessTick.mqh"
#include "live/functions/ResolveAuxBid.mqh"
#include "live/functions/UpdateIndicators.mqh"
#include "live/functions/ShouldClosePrimaryBar.mqh"
#include "live/functions/UpdatePrimaryImbalanceThreshold.mqh"
#include "live/functions/CloseBar.mqh"
#include "live/functions/OnTick.mqh"
#include "live/functions/LoadHistory.mqh"
#include "live/functions/ScaleAndClip.mqh"
#include "live/functions/SafeLogRatio.mqh"
#include "live/functions/LogReturnAt.mqh"
#include "live/functions/ReturnOverBars.mqh"
#include "live/functions/RollingStdReturn.mqh"
#include "live/functions/MeanClose.mqh"
#include "live/functions/StdClose.mqh"
#include "live/functions/MaxHigh.mqh"
#include "live/functions/MinLow.mqh"
#include "live/functions/MeanTickCount.mqh"
#include "live/functions/StdTickCount.mqh"
#include "live/functions/MeanTickImbalance.mqh"
#include "live/functions/MeanSpreadRel.mqh"
#include "live/functions/StdSpreadRel.mqh"
#include "live/functions/MeanAtrFeature.mqh"
#include "live/functions/TrueRangeAt.mqh"
#include "live/functions/SimpleAtr.mqh"
#include "live/functions/EmaClose.mqh"
#include "live/functions/MacdAt.mqh"
#include "live/functions/TypicalPrice.mqh"
#include "live/functions/SimpleCci.mqh"
#include "live/functions/WilliamsR.mqh"
#include "live/functions/SimpleRsi.mqh"
#include "live/functions/StochK.mqh"
#include "live/functions/StochD.mqh"
#include "live/functions/ExtractFeatures.mqh"
#include "live/functions/Softmax.mqh"
#include "live/functions/Predict.mqh"
#include "live/functions/Execute.mqh"
