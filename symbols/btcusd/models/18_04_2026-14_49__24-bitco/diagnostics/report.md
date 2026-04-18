# Model Diagnostics

## Run
- symbol: BTCUSD
- backend: hft-tcn-bigru-attention
- feature_profile: minimal
- feature_count: 9
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 9
- target_horizon: 27
- bar_mode: FIXED_TIME
- primary_bar_seconds: 54
- feature_atr_period: 7
- target_atr_period: 9
- rv_period: 5
- return_period: 5
- warmup_bars: 7
- label_risk_mode: FIXED
- point_size: 0.01000000
- fixed_move_points: 14400.00
- fixed_move_price: 144.00000000
- label_sl_multiplier: 1.00
- label_tp_multiplier: 1.20
- execution_sl_multiplier: 2.00
- execution_tp_multiplier: 2.50
- use_all_windows: 0
- selected_primary_confidence: 0.5059
- deployed_primary_confidence: 0.5059
- quality_gate_passed: 1
- quality_gate_reason: -

## Bar Stats
- bars: 81956
- ticks_per_bar min=1.00
- ticks_per_bar p50=74.00
- ticks_per_bar p90=123.00
- ticks_per_bar p99=145.00
- ticks_per_bar mean=76.63
- ticks_per_bar max=155.00
- bar_duration_ms min=0.00
- bar_duration_ms p50=52993.00
- bar_duration_ms p90=53661.00
- bar_duration_ms p99=53929.00
- bar_duration_ms mean=51920.52
- bar_duration_ms max=53998.00

## Label Counts
- full bars:
  - BUY: 36375
  - SELL: 22268
- train windows:
  - BUY: 4157
  - SELL: 4376
- validation windows:
  - BUY: 1269
  - SELL: 1489
- holdout windows:
  - BUY: 1236
  - SELL: 1005

## Window Usage
- train_available: 57329
- train_used: 14400
- validation_available: 12230
- validation_used: 5400
- holdout_available: 12231
- holdout_used: 5400

## Validation
- selected_trades: 2269
- trade_coverage: 0.8227
- selected_trade_precision: 0.5320
- selected_trade_mean_confidence: 0.5082
- mean_confidence_all_predictions: 0.5073

## Holdout
- selected_trades: 1819
- trade_coverage: 0.8117
- selected_trade_precision: 0.4585
- selected_trade_mean_confidence: 0.5082
- mean_confidence_all_predictions: 0.5072

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- active_features.txt
- config.mqh

## Note
- Bars are fixed-duration time buckets aligned to epoch time. Change PRIMARY_BAR_SECONDS in bitcoin.config to retune them.
- In ATR mode, labels use the label_sl_multiplier and label_tp_multiplier settings.
- In fixed mode, labels use DEFAULT_FIXED_MOVE for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples down to the configured train/eval caps.
