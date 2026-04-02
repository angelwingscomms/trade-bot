// Single source of truth for values shared by gold/nn.py and gold/live.mq5.
#define SEQ_LEN 54
#define TARGET_HORIZON 27

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
#define WARMUP_BARS 9
#define REQUIRED_HISTORY_INDEX (SEQ_LEN + RETURN_PERIOD - 1)

#define IMBALANCE_MIN_TICKS 9
#define IMBALANCE_EMA_SPAN 9

// Fixed stop/target distance in absolute price units, used when the trainer runs with -r
// and when live.mq5 input R is true.
#define DEFAULT_FIXED_MOVE 0.54

// Training labels: ATR multipliers used in the default ATR-risk mode.
#define LABEL_SL_MULTIPLIER 0.54
#define LABEL_TP_MULTIPLIER 0.54

// Live execution defaults used when live.mq5 input R is false / ATR-risk mode is enabled.
#define DEFAULT_SL_MULTIPLIER 0.54
#define DEFAULT_TP_MULTIPLIER 0.54
#define DEFAULT_LOT_SIZE 0.54

// Trainer window usage: 1 = use every eligible window in each split, 0 = obey the max_* caps below.
#define USE_ALL_WINDOWS 0
#define DEFAULT_EPOCHS 18
#define DEFAULT_BATCH_SIZE 54
#define DEFAULT_MAX_TRAIN_WINDOWS 5400
#define DEFAULT_MAX_EVAL_WINDOWS 540
#define DEFAULT_PATIENCE 9
