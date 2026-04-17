# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: au-lstm-mha-gap
- feature_profile: main
- feature_count: 40
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 144
- target_horizon: 30
- bar_mode: FIXED_TICK
- primary_tick_density: 144
- feature_atr_period: 9
- target_atr_period: 14
- rv_period: 9
- return_period: 9
- warmup_bars: 144
- label_risk_mode: ATR
- point_size: 0.00100000
- fixed_move_points: 1440.00
- fixed_move_price: 1.44000000
- label_sl_multiplier: 1.10
- label_tp_multiplier: 2.00
- execution_sl_multiplier: 5.40
- execution_tp_multiplier: 5.40
- use_all_windows: 0
- selected_primary_confidence: 0.4000
- deployed_primary_confidence: 0.4000
- quality_gate_passed: 0
- quality_gate_reason: validation selected trades 0 < required 12; validation selected-trade precision unavailable

## Bar Stats
- bars: 89305
- ticks_per_bar min=102.00
- ticks_per_bar p50=144.00
- ticks_per_bar p90=144.00
- ticks_per_bar p99=144.00
- ticks_per_bar mean=144.00
- ticks_per_bar max=144.00
- bar_duration_ms min=25814.00
- bar_duration_ms p50=35469.00
- bar_duration_ms p90=55151.20
- bar_duration_ms p99=86474.96
- bar_duration_ms mean=57732.07
- bar_duration_ms max=263100326.00

## Label Counts
- full bars:
  - HOLD: 27984
  - BUY: 29970
  - SELL: 31207
- train windows:
  - HOLD: 4569
  - BUY: 4732
  - SELL: 5099
- validation windows:
  - HOLD: 424
  - BUY: 513
  - SELL: 503
- holdout windows:
  - HOLD: 480
  - BUY: 477
  - SELL: 483

## Window Usage
- train_available: 62239
- train_used: 14400
- validation_available: 13057
- validation_used: 1440
- holdout_available: 13058
- holdout_used: 1440

## Validation
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.3505

## Holdout
- selected_trades: 1
- trade_coverage: 0.0007
- selected_trade_precision: 0.0000
- selected_trade_mean_confidence: 0.4011
- mean_confidence_all_predictions: 0.3505

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- active_features.txt
- config.mqh

## Note
- Fixed-tick bars use PRIMARY_TICK_DENSITY in au.config to set ticks per bar.
- In ATR mode, labels use the label_sl_multiplier and label_tp_multiplier settings.
- In fixed mode, labels use DEFAULT_FIXED_MOVE for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples down to the configured train/eval caps.
