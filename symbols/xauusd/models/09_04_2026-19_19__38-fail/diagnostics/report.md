# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: tla-temporal-lstm-attention
- feature_profile: core
- feature_count: 9
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 54
- target_horizon: 27
- bar_mode: IMBALANCE
- imbalance_min_ticks: 3
- imbalance_ema_span: 3
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
- use_all_windows: 0
- selected_primary_confidence: 0.4100
- deployed_primary_confidence: 0.4100
- quality_gate_passed: 0
- quality_gate_reason: validation selected-trade precision 0.3416 < required 0.5000

## Bar Stats
- bars: 1649557
- ticks_per_bar min=3.00
- ticks_per_bar p50=5.00
- ticks_per_bar p90=13.00
- ticks_per_bar p99=25.00
- ticks_per_bar mean=6.94
- ticks_per_bar max=81.00
- bar_duration_ms min=1.00
- bar_duration_ms p50=1097.00
- bar_duration_ms p90=3489.00
- bar_duration_ms p99=7944.00
- bar_duration_ms mean=2488.91
- bar_duration_ms max=263045234.00

## Label Counts
- full bars:
  - HOLD: 759715
  - BUY: 451791
  - SELL: 438029
- train windows:
  - HOLD: 6812
  - BUY: 3867
  - SELL: 3721
- validation windows:
  - HOLD: 542
  - BUY: 438
  - SELL: 460
- holdout windows:
  - HOLD: 701
  - BUY: 394
  - SELL: 345

## Window Usage
- train_available: 1154594
- train_used: 14400
- validation_available: 247296
- validation_used: 1440
- holdout_available: 247297
- holdout_used: 1440

## Validation
- selected_trades: 606
- trade_coverage: 0.4208
- selected_trade_precision: 0.3416
- selected_trade_mean_confidence: 0.4394
- mean_confidence_all_predictions: 0.3935

## Holdout
- selected_trades: 607
- trade_coverage: 0.4215
- selected_trade_precision: 0.3344
- selected_trade_mean_confidence: 0.4382
- mean_confidence_all_predictions: 0.3931

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
- Imbalance bars are variable by design. Lowering imbalance_min_ticks makes them smaller on average, but it does not force a fixed tick count per bar.
- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.
- In fixed mode, labels use the same DEFAULT_FIXED_MOVE value in symbol points for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.
