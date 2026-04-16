# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: fusion-lstm-attention
- feature_profile: core
- feature_count: 9
- loss_mode: cross-entropy
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
- warmup_bars: 22
- label_risk_mode: FIXED
- point_size: 0.00100000
- fixed_move_points: 1440.00
- fixed_move_price: 1.44000000
- label_sl_multiplier: 0.54
- label_tp_multiplier: 0.54
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 0
- selected_primary_confidence: 0.4000
- deployed_primary_confidence: 0.4000
- quality_gate_passed: 0
- quality_gate_reason: validation selected trades 5 < required 12

## Bar Stats
- bars: 344048
- ticks_per_bar min=1.00
- ticks_per_bar p50=35.00
- ticks_per_bar p90=45.00
- ticks_per_bar p99=48.00
- ticks_per_bar mean=33.27
- ticks_per_bar max=54.00
- bar_duration_ms min=9000.00
- bar_duration_ms p50=9000.00
- bar_duration_ms p90=9000.00
- bar_duration_ms p99=9000.00
- bar_duration_ms mean=9000.00
- bar_duration_ms max=9000.00

## Label Counts
- full bars:
  - HOLD: 37234
  - BUY: 152362
  - SELL: 154430
- train windows:
  - HOLD: 637
  - BUY: 2358
  - SELL: 2405
- validation windows:
  - HOLD: 34
  - BUY: 245
  - SELL: 261
- holdout windows:
  - HOLD: 59
  - BUY: 251
  - SELL: 230

## Window Usage
- train_available: 240738
- train_used: 5400
- validation_available: 51470
- validation_used: 540
- holdout_available: 51470
- holdout_used: 540

## Validation
- selected_trades: 5
- trade_coverage: 0.0093
- selected_trade_precision: 0.6000
- selected_trade_mean_confidence: 0.4516
- mean_confidence_all_predictions: 0.5716

## Holdout
- selected_trades: 2
- trade_coverage: 0.0037
- selected_trade_precision: 1.0000
- selected_trade_mean_confidence: 0.4827
- mean_confidence_all_predictions: 0.5805

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- active_features.txt
- shared_config_snapshot.mqh
- model_config_snapshot.mqh

## Note
- Bars are fixed-duration time buckets aligned to epoch time. Change PRIMARY_BAR_SECONDS in shared_config.mqh to retune them, for example to 27 or 9 seconds.
- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.
- In fixed mode, labels use the same DEFAULT_FIXED_MOVE value in symbol points for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.
