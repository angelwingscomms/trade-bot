# Model Diagnostics

## Run
- symbol: BTCUSD
- backend: minirocket-multivariate-attention
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
- label_sl_multiplier: 0.72
- label_tp_multiplier: 0.12
- execution_sl_multiplier: 0.72
- execution_tp_multiplier: 0.12
- use_all_windows: 0
- selected_primary_confidence: 0.5900
- deployed_primary_confidence: 0.5900
- quality_gate_passed: 1
- quality_gate_reason: -

## Bar Stats
- bars: 50537
- ticks_per_bar min=9.00
- ticks_per_bar p50=57.00
- ticks_per_bar p90=165.00
- ticks_per_bar p99=329.00
- ticks_per_bar mean=78.11
- ticks_per_bar max=751.00
- bar_duration_ms min=2790.00
- bar_duration_ms p50=36681.00
- bar_duration_ms p90=119783.60
- bar_duration_ms p99=306697.48
- bar_duration_ms mean=58057.88
- bar_duration_ms max=32922072.00

## Label Counts
- full bars:
  - HOLD: 27735
  - BUY: 11447
  - SELL: 11346
- train windows:
  - HOLD: 6416
  - BUY: 2802
  - SELL: 2782
- validation windows:
  - HOLD: 1211
  - BUY: 399
  - SELL: 438
- holdout windows:
  - HOLD: 1213
  - BUY: 418
  - SELL: 417

## Window Usage
- train_available: 35289
- train_used: 12000
- validation_available: 7445
- validation_used: 2048
- holdout_available: 7446
- holdout_used: 2048

## Validation
- selected_trades: 3
- trade_coverage: 0.0015
- selected_trade_precision: 1.0000
- selected_trade_mean_confidence: 0.6066
- mean_confidence_all_predictions: 0.6443

## Holdout
- selected_trades: 2
- trade_coverage: 0.0010
- selected_trade_precision: 0.5000
- selected_trade_mean_confidence: 0.6006
- mean_confidence_all_predictions: 0.6282

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
