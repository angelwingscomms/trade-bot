# Model Diagnostics

## Run
- symbol: BTCUSD
- backend: au-lstm-mha-gap
- feature_profile: minimal
- feature_count: 9
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 27
- target_horizon: 9
- bar_mode: FIXED_TICK
- primary_tick_density: 18
- feature_atr_period: 7
- target_atr_period: 7
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
- selected_primary_confidence: 0.8538
- deployed_primary_confidence: 0.8538
- quality_gate_passed: 0
- quality_gate_reason: validation selected-trade precision 0.1111 < required 0.5200

## Bar Stats
- bars: 348915
- ticks_per_bar min=8.00
- ticks_per_bar p50=18.00
- ticks_per_bar p90=18.00
- ticks_per_bar p99=18.00
- ticks_per_bar mean=18.00
- ticks_per_bar max=18.00
- bar_duration_ms min=4909.00
- bar_duration_ms p50=9664.00
- bar_duration_ms p90=19940.00
- bar_duration_ms p99=40721.86
- bar_duration_ms mean=12653.32
- bar_duration_ms max=32832036.00

## Label Counts
- full bars:
  - HOLD: 334548
  - BUY: 7323
  - SELL: 7037
- train windows:
  - HOLD: 13776
  - BUY: 326
  - SELL: 298
- validation windows:
  - HOLD: 5215
  - BUY: 92
  - SELL: 93
- holdout windows:
  - HOLD: 5204
  - BUY: 107
  - SELL: 89

## Window Usage
- train_available: 244200
- train_used: 14400
- validation_available: 52274
- validation_used: 5400
- holdout_available: 52275
- holdout_used: 5400

## Validation
- selected_trades: 18
- trade_coverage: 0.0033
- selected_trade_precision: 0.1111
- selected_trade_mean_confidence: 0.8718
- mean_confidence_all_predictions: 0.5996

## Holdout
- selected_trades: 20
- trade_coverage: 0.0037
- selected_trade_precision: 0.1000
- selected_trade_mean_confidence: 0.8733
- mean_confidence_all_predictions: 0.6065

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
