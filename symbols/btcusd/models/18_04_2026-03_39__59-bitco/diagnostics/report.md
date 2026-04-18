# Model Diagnostics

## Run
- symbol: BTCUSD
- backend: au-lstm-mha-gap
- feature_profile: minimal
- feature_count: 9
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 144
- target_horizon: 5
- bar_mode: FIXED_TICK
- primary_tick_density: 18
- feature_atr_period: 7
- target_atr_period: 7
- rv_period: 5
- return_period: 5
- warmup_bars: 7
- label_risk_mode: FIXED
- point_size: 0.10000000
- fixed_move_points: 144000.00
- fixed_move_price: 14400.00000000
- label_sl_multiplier: 1.00
- label_tp_multiplier: 1.20
- execution_sl_multiplier: 2.00
- execution_tp_multiplier: 2.50
- use_all_windows: 0
- selected_primary_confidence: 0.4000
- deployed_primary_confidence: 0.4000
- quality_gate_passed: 0
- quality_gate_reason: validation selected trades 0 < required 15; validation selected-trade precision unavailable

## Bar Stats
- bars: 337053
- ticks_per_bar min=8.00
- ticks_per_bar p50=18.00
- ticks_per_bar p90=18.00
- ticks_per_bar p99=18.00
- ticks_per_bar mean=18.00
- ticks_per_bar max=18.00
- bar_duration_ms min=4094.00
- bar_duration_ms p50=10133.00
- bar_duration_ms p90=21072.00
- bar_duration_ms p99=42707.00
- bar_duration_ms mean=13101.22
- bar_duration_ms max=32832036.00

## Label Counts
- full bars:
  - HOLD: 337046
  - BUY: 0
  - SELL: 0
- train windows:
  - HOLD: 54000
  - BUY: 0
  - SELL: 0
- validation windows:
  - HOLD: 5400
  - BUY: 0
  - SELL: 0
- holdout windows:
  - HOLD: 5400
  - BUY: 0
  - SELL: 0

## Window Usage
- train_available: 235784
- train_used: 54000
- validation_available: 50265
- validation_used: 5400
- holdout_available: 50265
- holdout_used: 5400

## Validation
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 1.0000

## Holdout
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 1.0000

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- active_features.txt
- config.mqh

## Note
- Fixed-tick bars use PRIMARY_TICK_DENSITY in bitcoin.config to set ticks per bar.
- In ATR mode, labels use the label_sl_multiplier and label_tp_multiplier settings.
- In fixed mode, labels use DEFAULT_FIXED_MOVE for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples down to the configured train/eval caps.
