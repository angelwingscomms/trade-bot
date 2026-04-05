# Model Diagnostics

## Run
- symbol: BTCUSD
- backend: minirocket-multivariate
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
- warmup_bars: 9
- label_risk_mode: ATR
- fixed_move: 1.44
- label_sl_multiplier: 0.54
- label_tp_multiplier: 0.54
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 0
- selected_primary_confidence: 0.5862
- deployed_primary_confidence: 1.0100
- quality_gate_passed: 0
- quality_gate_reason: validation selected-trade precision 0.2857 < required 0.5000

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
  - HOLD: 67418
  - BUY: 7162
  - SELL: 7094
- train windows:
  - HOLD: 32598
  - BUY: 3779
  - SELL: 3623
- validation windows:
  - HOLD: 6805
  - BUY: 670
  - SELL: 717
- holdout windows:
  - HOLD: 7104
  - BUY: 518
  - SELL: 570

## Window Usage
- train_available: 57091
- train_used: 40000
- validation_available: 12117
- validation_used: 8192
- holdout_available: 12118
- holdout_used: 8192

## Validation
- selected_trades: 14
- trade_coverage: 0.0017
- selected_trade_precision: 0.2857
- selected_trade_mean_confidence: 0.6062
- mean_confidence_all_predictions: 0.6711

## Holdout
- selected_trades: 13
- trade_coverage: 0.0016
- selected_trade_precision: 0.3077
- selected_trade_mean_confidence: 0.6243
- mean_confidence_all_predictions: 0.7109

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- shared_config_snapshot.mqh
- model_config_snapshot.mqh

## Note
- Imbalance bars are variable by design. Lowering imbalance_min_ticks makes them smaller on average, but it does not force a fixed tick count per bar.
- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.
- In fixed mode, labels use the same DEFAULT_FIXED_MOVE distance for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.
