# Model Diagnostics

## Run
- symbol: BTCUSD
- backend: gru
- feature_profile: core
- feature_count: 9
- loss_mode: focal
- focal_gamma: 2.00

## Shared Config
- seq_len: 54
- target_horizon: 27
- bar_mode: IMBALANCE
- imbalance_min_ticks: 9
- imbalance_ema_span: 9
- feature_atr_period: 9
- target_atr_period: 9
- rv_period: 9
- return_period: 9
- warmup_bars: 22
- label_risk_mode: ATR
- point_size: 0.01000000
- fixed_move_points: 144.00
- fixed_move_price: 1.44000000
- label_sl_multiplier: 5.40
- label_tp_multiplier: 5.40
- execution_sl_multiplier: 5.40
- execution_tp_multiplier: 5.40
- use_all_windows: 0
- selected_primary_confidence: 0.4000
- deployed_primary_confidence: 0.4000
- quality_gate_passed: 0
- quality_gate_reason: validation selected-trade precision 0.0714 < required 0.5000

## Bar Stats
- bars: 81683
- ticks_per_bar min=9.00
- ticks_per_bar p50=61.00
- ticks_per_bar p90=171.00
- ticks_per_bar p99=345.36
- ticks_per_bar mean=81.92
- ticks_per_bar max=1129.00
- bar_duration_ms min=2769.00
- bar_duration_ms p50=36001.00
- bar_duration_ms p90=114905.00
- bar_duration_ms p99=289167.72
- bar_duration_ms mean=56466.07
- bar_duration_ms max=32921752.00

## Label Counts
- full bars:
  - HOLD: 60764
  - BUY: 10118
  - SELL: 10779
- train windows:
  - HOLD: 18
  - BUY: 10
  - SELL: 4
- validation windows:
  - HOLD: 14
  - BUY: 1
  - SELL: 1
- holdout windows:
  - HOLD: 13
  - BUY: 2
  - SELL: 1

## Window Usage
- train_available: 57082
- train_used: 32
- validation_available: 12115
- validation_used: 16
- holdout_available: 12116
- holdout_used: 16

## Validation
- selected_trades: 14
- trade_coverage: 0.8750
- selected_trade_precision: 0.0714
- selected_trade_mean_confidence: 0.5290
- mean_confidence_all_predictions: 0.5104

## Holdout
- selected_trades: 15
- trade_coverage: 0.9375
- selected_trade_precision: 0.0667
- selected_trade_mean_confidence: 0.5025
- mean_confidence_all_predictions: 0.4960

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
