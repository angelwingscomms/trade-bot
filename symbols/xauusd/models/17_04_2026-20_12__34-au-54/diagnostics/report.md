# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: au-lstm-mha-gap
- feature_profile: minimal
- feature_count: 9
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 9
- target_horizon: 1
- bar_mode: FIXED_TIME
- primary_bar_seconds: 54
- feature_atr_period: 1
- target_atr_period: 14
- rv_period: 1
- return_period: 1
- warmup_bars: 1
- label_risk_mode: FIXED
- point_size: 0.00100000
- fixed_move_points: 1440.00
- fixed_move_price: 1.44000000
- label_sl_multiplier: 1.00
- label_tp_multiplier: 1.00
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 0
- selected_primary_confidence: 0.4500
- deployed_primary_confidence: 0.4500
- quality_gate_passed: 0
- quality_gate_reason: validation selected trades 0 < required 1; validation selected-trade precision unavailable

## Bar Stats
- bars: 512
- ticks_per_bar min=4.00
- ticks_per_bar p50=119.00
- ticks_per_bar p90=130.00
- ticks_per_bar p99=135.00
- ticks_per_bar mean=113.84
- ticks_per_bar max=138.00
- bar_duration_ms min=1400.00
- bar_duration_ms p50=53573.00
- bar_duration_ms p90=53642.80
- bar_duration_ms p99=53959.78
- bar_duration_ms mean=53157.45
- bar_duration_ms max=53970.00

## Label Counts
- full bars:
  - HOLD: 215
  - BUY: 149
  - SELL: 147
- train windows:
  - HOLD: 94
  - BUY: 81
  - SELL: 81
- validation windows:
  - HOLD: 37
  - BUY: 10
  - SELL: 12
- holdout windows:
  - HOLD: 31
  - BUY: 13
  - SELL: 15

## Window Usage
- train_available: 348
- train_used: 256
- validation_available: 59
- validation_used: 59
- holdout_available: 59
- holdout_used: 59

## Validation
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.3658

## Holdout
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.3649

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- active_features.txt
- config.mqh

## Note
- Bars are fixed-duration time buckets aligned to epoch time. Change PRIMARY_BAR_SECONDS in testrun.config to retune them.
- In ATR mode, labels use the label_sl_multiplier and label_tp_multiplier settings.
- In fixed mode, labels use DEFAULT_FIXED_MOVE for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples down to the configured train/eval caps.
