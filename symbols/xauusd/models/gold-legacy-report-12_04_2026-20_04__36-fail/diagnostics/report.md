# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: gold-legacy-lstm-attention
- feature_profile: gold_context_extended
- feature_count: 38
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 120
- target_horizon: 27
- bar_mode: FIXED_TICK
- primary_tick_density: 27
- feature_atr_period: 9
- target_atr_period: 9
- rv_period: 9
- return_period: 9
- warmup_bars: 22
- label_risk_mode: ATR
- point_size: 0.00100000
- fixed_move_points: 1440.00
- fixed_move_price: 1.44000000
- label_sl_multiplier: 0.54
- label_tp_multiplier: 0.54
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 1
- selected_primary_confidence: 0.4000
- deployed_primary_confidence: 0.4000
- quality_gate_passed: 0
- quality_gate_reason: validation selected trades 0 < required 12; validation selected-trade precision unavailable

## Bar Stats
- bars: 14856
- ticks_per_bar min=27.00
- ticks_per_bar p50=27.00
- ticks_per_bar p90=27.00
- ticks_per_bar p99=27.00
- ticks_per_bar mean=27.00
- ticks_per_bar max=27.00
- bar_duration_ms min=4313.00
- bar_duration_ms p50=10527.00
- bar_duration_ms p90=14004.00
- bar_duration_ms p99=20316.35
- bar_duration_ms mean=10377.48
- bar_duration_ms max=3703107.00

## Label Counts
- full bars:
  - HOLD: 3078
  - BUY: 5860
  - SELL: 5896
- train windows:
  - HOLD: 2260
  - BUY: 3957
  - SELL: 4015
- validation windows:
  - HOLD: 293
  - BUY: 845
  - SELL: 821
- holdout windows:
  - HOLD: 385
  - BUY: 800
  - SELL: 775

## Window Usage
- train_available: 10232
- train_used: 10232
- validation_available: 1959
- validation_used: 1959
- holdout_available: 1960
- holdout_used: 1960

## Validation
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.3943

## Holdout
- selected_trades: 0
- trade_coverage: 0.0000
- selected_trade_precision: n/a
- selected_trade_mean_confidence: n/a
- mean_confidence_all_predictions: 0.4268

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
- Fixed-tick bars use PRIMARY_TICK_DENSITY in shared_config_gold.mqh to set ticks per bar.
- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.
- In fixed mode, labels use the same DEFAULT_FIXED_MOVE value in symbol points for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.
