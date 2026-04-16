// Single source of truth for values shared by nn.py, live.mq5, data.mq5, and test.py.
#define SYMBOL "BTCUSD"

#define SEQ_LEN 54
#define TARGET_HORIZON 27

// Default feature layout. The active model snapshot may override MODEL_FEATURE_COUNT
// and FEATURE_IDX_* macros when nn.py exports an architecture-specific feature pack.
#define MODEL_FEATURE_COUNT 9
#define FEATURE_IDX_RET1 0
#define FEATURE_IDX_HIGH_REL_PREV 1
#define FEATURE_IDX_LOW_REL_PREV 2
#define FEATURE_IDX_SPREAD_REL 3
#define FEATURE_IDX_CLOSE_IN_RANGE 4
#define FEATURE_IDX_ATR_REL 5
#define FEATURE_IDX_RV 6
#define FEATURE_IDX_RETURN_N 7
#define FEATURE_IDX_TICK_IMBALANCE 8

#define FEATURE_ATR_PERIOD 9
#define TARGET_ATR_PERIOD 9
#define RV_PERIOD 9
#define RETURN_PERIOD 9
#define MAX_FEATURE_LOOKBACK 22
#define WARMUP_BARS 22
#define REQUIRED_HISTORY_INDEX (SEQ_LEN + MAX_FEATURE_LOOKBACK - 1)

// Default primary bars are imbalance bars.
#define IMBALANCE_MIN_TICKS 9
#define IMBALANCE_EMA_SPAN 9

// Fixed-time bars are optional and only used when nn.py runs with -i.
#define PRIMARY_BAR_SECONDS 9

// Fixed stop/target distance in symbol points, used when the trainer runs with -r
// and when live.mq5 input R is true.
#define DEFAULT_FIXED_MOVE 144

// Training labels: ATR multipliers used in the default ATR-risk mode.
#define LABEL_SL_MULTIPLIER 5.4
#define LABEL_TP_MULTIPLIER 5.4

// Live execution defaults used when live.mq5 input R is false / ATR-risk mode is enabled.
#define DEFAULT_SL_MULTIPLIER 5.4
#define DEFAULT_TP_MULTIPLIER 5.4
#define DEFAULT_LOT_SIZE 0.54
#define DEFAULT_RISK_PERCENT 0.00
#define DEFAULT_LOT_MIN 0.54

// Trainer window usage: 1 = use every eligible window in each split, 0 = obey the max_* caps below.
#define USE_ALL_WINDOWS 0

#define DEFAULT_EPOCHS 54
#define DEFAULT_BATCH_SIZE 54
#define DEFAULT_MAX_TRAIN_WINDOWS 5400
#define DEFAULT_MAX_EVAL_WINDOWS 540
#define DEFAULT_PATIENCE 27
