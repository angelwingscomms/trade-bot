# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: gold-legacy-lstm-attention
- feature_profile: full
- feature_count: 48
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 144
- target_horizon: 3
- bar_mode: FIXED_TICK
- primary_tick_density: 27
- feature_atr_period: 9
- target_atr_period: 9
- rv_period: 9
- return_period: 9
- warmup_bars: 28
- label_risk_mode: FIXED
- point_size: 0.00100000
- fixed_move_points: 1440.00
- fixed_move_price: 1.44000000
- label_sl_multiplier: 5.40
- label_tp_multiplier: 5.40
- execution_sl_multiplier: 5.40
- execution_tp_multiplier: 5.40
- use_all_windows: 0
- selected_primary_confidence: 0.4800
- deployed_primary_confidence: 0.4800
- quality_gate_passed: 0
- quality_gate_reason: validation selected-trade precision 0.3529 < required 0.5000

## Bar Stats
- bars: 476292
- ticks_per_bar min=21.00
- ticks_per_bar p50=27.00
- ticks_per_bar p90=27.00
- ticks_per_bar p99=27.00
- ticks_per_bar mean=27.00
- ticks_per_bar max=27.00
- bar_duration_ms min=2628.00
- bar_duration_ms p50=6314.00
- bar_duration_ms p90=10248.00
- bar_duration_ms p99=17206.00
- bar_duration_ms mean=10585.74
- bar_duration_ms max=263050557.00

## Label Counts
- full bars:
  - HOLD: 283273
  - BUY: 96791
  - SELL: 96200
- train windows:
  - HOLD: 8645
  - BUY: 2906
  - SELL: 2849
- validation windows:
  - HOLD: 781
  - BUY: 332
  - SELL: 327
- holdout windows:
  - HOLD: 899
  - BUY: 285
  - SELL: 256

## Window Usage
- train_available: 333238
- train_used: 14400
- validation_available: 71150
- validation_used: 1440
- holdout_available: 71150
- holdout_used: 1440

## Validation
- selected_trades: 51
- trade_coverage: 0.0354
- selected_trade_precision: 0.3529
- selected_trade_mean_confidence: 0.4951
- mean_confidence_all_predictions: 0.4301

## Holdout
- selected_trades: 58
- trade_coverage: 0.0403
- selected_trade_precision: 0.4828
- selected_trade_mean_confidence: 0.5003
- mean_confidence_all_predictions: 0.4195

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- active_features.txt
- config.mqh

## Note
- Fixed-tick bars use PRIMARY_TICK_DENSITY in config.mqh to set ticks per bar.
- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.
- In fixed mode, labels use the same DEFAULT_FIXED_MOVE value in symbol points for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.
