#include <Trade\Trade.mqh>
// @active-model-reference begin
#define ACTIVE_MODEL_SYMBOL "XAUUSD"
#define ACTIVE_MODEL_VERSION "20_04_2026-22_07__24-au-54"
#include "../symbols/xauusd/models/20_04_2026-22_07__24-au-54/config.mqh"
#resource "..\symbols\\xauusd\\models\\20_04_2026-22_07__24-au-54\\model.onnx" as uchar model_buffer[]
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

// Allow separate fixed-point SL and TP distances (in points).
// If the config does not define them they fall back to DEFAULT_FIXED_MOVE.
#ifndef DEFAULT_FIXED_SL
#define DEFAULT_FIXED_SL DEFAULT_FIXED_MOVE
#endif
#ifndef DEFAULT_FIXED_TP
#define DEFAULT_FIXED_TP DEFAULT_FIXED_MOVE
#endif

#define INPUT_BUFFER_SIZE (SEQ_LEN * MODEL_FEATURE_COUNT)
#define HISTORY_SIZE (REQUIRED_HISTORY_INDEX + 1)
#define PRIMARY_BAR_MILLISECONDS ((ulong)PRIMARY_BAR_SECONDS * 1000)

input bool R = (MODEL_USE_ATR_RISK == 0);
input double FIXED_MOVE = DEFAULT_FIXED_MOVE;
input double FIXED_SL   = DEFAULT_FIXED_SL;   // fixed-mode stop-loss distance (points)
input double FIXED_TP   = DEFAULT_FIXED_TP;   // fixed-mode take-profit distance (points)
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
double PastDirBarAt(int h, int n_bars);
double PastDirSecondsAt(int h, int n_seconds);
double ResolveMinimumVolume();
double NormalizeVolume(double volume);
double CalculateTradeVolume(int signal, double price, double sl);
void PrintRunSummary();
double ResolveAuxBid(string symbol, bool &available, double &last_value, double fallback);


#include "functions/OnInit.mqh"
#include "functions/OnDeinit.mqh"
#include "functions/OnTradeTransaction.mqh"
#include "functions/DebugPrint.mqh"
#include "functions/SignalName.mqh"
#include "functions/BarBucket.mqh"
#include "functions/BarOpenTime.mqh"
#include "functions/StartBar.mqh"
#include "functions/StartImbalanceBar.mqh"
#include "functions/RollFixedTimeBarIfNeeded.mqh"
#include "functions/ResolveImbalanceThresholdBase.mqh"
#include "functions/StopDistance.mqh"
#include "functions/TargetDistance.mqh"
#include "functions/ResolveMinimumVolume.mqh"
#include "functions/NormalizeVolume.mqh"
#include "functions/CalculateTradeVolume.mqh"
#include "functions/PrintRunSummary.mqh"
#include "functions/UpdateTickSign.mqh"
#include "functions/ProcessTick.mqh"
#include "functions/ResolveAuxBid.mqh"
#include "functions/UpdateIndicators.mqh"
#include "functions/ShouldClosePrimaryBar.mqh"
#include "functions/UpdatePrimaryImbalanceThreshold.mqh"
#include "functions/CloseBar.mqh"
#include "functions/OnTick.mqh"
#include "functions/LoadHistory.mqh"
#include "functions/ScaleAndClip.mqh"
#include "functions/SafeLogRatio.mqh"
#include "functions/LogReturnAt.mqh"
#include "functions/ReturnOverBars.mqh"
#include "functions/RollingStdReturn.mqh"
#include "functions/MeanClose.mqh"
#include "functions/StdClose.mqh"
#include "functions/MaxHigh.mqh"
#include "functions/MinLow.mqh"
#include "functions/MeanTickCount.mqh"
#include "functions/StdTickCount.mqh"
#include "functions/MeanTickImbalance.mqh"
#include "functions/MeanSpreadRel.mqh"
#include "functions/StdSpreadRel.mqh"
#include "functions/MeanAtrFeature.mqh"
#include "functions/TrueRangeAt.mqh"
#include "functions/SimpleAtr.mqh"
#include "functions/EmaClose.mqh"
#include "functions/MacdAt.mqh"
#include "functions/TypicalPrice.mqh"
#include "functions/SimpleCci.mqh"
#include "functions/WilliamsR.mqh"
#include "functions/SimpleRsi.mqh"
#include "functions/StochK.mqh"
#include "functions/StochD.mqh"
#include "functions/ExtractFeatures.mqh"
#include "functions/Softmax.mqh"
#include "functions/Predict.mqh"
#include "functions/Execute.mqh"
#include "functions/PastDirAt.mqh"
