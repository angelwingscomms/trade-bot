# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: gold-new-conv-gru-attention
- feature_profile: main
- feature_count: 40
- loss_mode: cross-entropy
- focal_gamma: 2.00

## Shared Config
- seq_len: 144
- target_horizon: 3
- bar_mode: IMBALANCE
- imbalance_min_ticks: 3
- imbalance_ema_span: 3
- feature_atr_period: 9
- target_atr_period: 14
- rv_period: 9
- return_period: 9
- warmup_bars: 144
- label_risk_mode: FIXED
- point_size: 0.00100000
- fixed_move_points: 1440.00
- fixed_move_price: 1.44000000
- label_sl_multiplier: 1.00
- label_tp_multiplier: 1.00
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 0
- selected_primary_confidence: 0.6800
- deployed_primary_confidence: 0.6800
- quality_gate_passed: 1
- quality_gate_reason: -

## Bar Stats
- bars: 1845896
- ticks_per_bar min=3.00
- ticks_per_bar p50=5.00
- ticks_per_bar p90=13.00
- ticks_per_bar p99=27.00
- ticks_per_bar mean=6.97
- ticks_per_bar max=77.00
- bar_duration_ms min=0.00
- bar_duration_ms p50=1101.00
- bar_duration_ms p90=3546.00
- bar_duration_ms p99=8268.05
- bar_duration_ms mean=2174.55
- bar_duration_ms max=180250276.00

## Label Counts
- full bars:
  - BUY: 1671110
  - SELL: 86797
- train windows:
  - BUY: 3495
  - SELL: 3565
- validation windows:
  - BUY: 396
  - SELL: 382
- holdout windows:
  - BUY: 272
  - SELL: 279

## Window Usage
- train_available: 1291880
- train_used: 72000
- validation_available: 276573
- validation_used: 7200
- holdout_available: 276573
- holdout_used: 7200

## Validation
- selected_trades: 18
- trade_coverage: 0.0231
- selected_trade_precision: 0.8333
- selected_trade_mean_confidence: 0.7068
- mean_confidence_all_predictions: 0.5549

## Holdout
- selected_trades: 9
- trade_coverage: 0.0163
- selected_trade_precision: 0.3333
- selected_trade_mean_confidence: 0.7048
- mean_confidence_all_predictions: 0.5530

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- active_features.txt
- config.mqh

## Note
- Imbalance bars are variable by design. Lower imbalance thresholds make smaller bars on average.
- In ATR mode, labels use the label_sl_multiplier and label_tp_multiplier settings.
- In fixed mode, labels use DEFAULT_FIXED_MOVE for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples down to the configured train/eval caps.
