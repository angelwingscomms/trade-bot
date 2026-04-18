# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: gold-legacy-lstm-attention
- feature_profile: full
- feature_count: 48
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 60
- target_horizon: 5
- bar_mode: FIXED_TICK
- primary_tick_density: 18
- feature_atr_period: 7
- target_atr_period: 7
- rv_period: 5
- return_period: 5
- warmup_bars: 20
- label_risk_mode: FIXED
- point_size: 0.00100000
- fixed_move_points: 300.00
- fixed_move_price: 0.30000000
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
- bars: 714438
- ticks_per_bar min=12.00
- ticks_per_bar p50=18.00
- ticks_per_bar p90=18.00
- ticks_per_bar p99=18.00
- ticks_per_bar mean=18.00
- ticks_per_bar max=18.00
- bar_duration_ms min=1981.00
- bar_duration_ms p50=4102.00
- bar_duration_ms p90=6761.00
- bar_duration_ms p99=11719.00
- bar_duration_ms mean=6966.14
- bar_duration_ms max=263048801.00

## Label Counts
- full bars:
  - HOLD: 229501
  - BUY: 243300
  - SELL: 241617
- train windows:
  - HOLD: 1247
  - BUY: 1357
  - SELL: 1396
- validation windows:
  - HOLD: 254
  - BUY: 292
  - SELL: 254
- holdout windows:
  - HOLD: 291
  - BUY: 239
  - SELL: 270

## Window Usage
- train_available: 500028
- train_used: 4000
- validation_available: 107039
- validation_used: 800
- holdout_available: 107039
- holdout_used: 800

## Validation
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.3661

## Holdout
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.3615

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
