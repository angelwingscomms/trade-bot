# Model Diagnostics

## Run
- symbol: BTCUSD
- backend: minirocket-multivariate
- focal_gamma: 2.00

## Shared Config
- seq_len: 54
- target_horizon: 27
- bar_mode: FIXED_TIME
- primary_bar_seconds: 9
- feature_atr_period: 9
- target_atr_period: 9
- rv_period: 9
- return_period: 9
- warmup_bars: 9
- label_risk_mode: FIXED
- fixed_move: 1.44
- label_sl_multiplier: 0.54
- label_tp_multiplier: 0.54
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 0
- selected_primary_confidence: 1.0100
- deployed_primary_confidence: 1.0100
- quality_gate_passed: 0
- quality_gate_reason: validation selected trades 0 < required 12; validation selected-trade precision unavailable

## Bar Stats
- bars: 486840
- ticks_per_bar min=1.00
- ticks_per_bar p50=14.00
- ticks_per_bar p90=22.00
- ticks_per_bar p99=25.00
- ticks_per_bar mean=13.74
- ticks_per_bar max=26.00
- bar_duration_ms min=9000.00
- bar_duration_ms p50=9000.00
- bar_duration_ms p90=9000.00
- bar_duration_ms p99=9000.00
- bar_duration_ms mean=9000.00
- bar_duration_ms max=9000.00

## Label Counts
- full bars:
  - HOLD: 486313
  - BUY: 258
  - SELL: 260
- train windows:
  - HOLD: 19981
  - BUY: 13
  - SELL: 6
- validation windows:
  - HOLD: 4094
  - BUY: 1
  - SELL: 1
- holdout windows:
  - HOLD: 4094
  - BUY: 1
  - SELL: 1

## Window Usage
- train_available: 340701
- train_used: 20000
- validation_available: 72891
- validation_used: 4096
- holdout_available: 72891
- holdout_used: 4096

## Validation
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.5202

## Holdout
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.5190

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- shared_config_snapshot.mqh
- model_config_snapshot.mqh

## Note
- Bars are fixed-duration time buckets aligned to epoch time. Change PRIMARY_BAR_SECONDS in shared_config.mqh to retune them, for example to 27 or 9 seconds.
- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.
- In fixed mode, labels use the same DEFAULT_FIXED_MOVE distance for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.
