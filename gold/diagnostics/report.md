# Gold Diagnostics

## Shared Config
- seq_len: 54
- target_horizon: 27
- imbalance_min_ticks: 27
- imbalance_ema_span: 27
- feature_atr_period: 9
- target_atr_period: 9
- rv_period: 9
- return_period: 9
- warmup_bars: 9
- label_risk_mode: ATR
- fixed_move: 0.54
- label_sl_multiplier: 0.54
- label_tp_multiplier: 0.54
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 0
- primary_confidence: 0.4000

## Bar Stats
- bars: 33298
- ticks_per_bar min=27.00
- ticks_per_bar p50=346.00
- ticks_per_bar p90=1109.00
- ticks_per_bar p99=2454.03
- ticks_per_bar mean=511.46
- ticks_per_bar max=6790.00
- bar_duration_ms min=4957.00
- bar_duration_ms p50=72237.00
- bar_duration_ms p90=232290.60
- bar_duration_ms p99=531122.01
- bar_duration_ms mean=151895.12
- bar_duration_ms max=180432472.00

## Label Counts
- full bars:
  - HOLD: 27
  - BUY: 10948
  - SELL: 22314
- train windows:
  - HOLD: 0
  - BUY: 214
  - SELL: 326
- validation windows:
  - HOLD: 0
  - BUY: 20
  - SELL: 34
- holdout windows:
  - HOLD: 0
  - BUY: 2
  - SELL: 52

## Window Usage
- train_available: 23222
- train_used: 540
- validation_available: 4859
- validation_used: 54
- holdout_available: 4860
- holdout_used: 54

## Validation
- selected_trades: 54
- trade_coverage: 1.0000
- mean_confidence: 0.9973

## Holdout
- selected_trades: 54
- trade_coverage: 1.0000
- mean_confidence: 0.9972

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- shared_config_snapshot.mqh

## Note
- Imbalance bars are variable by design. Lowering imbalance_min_ticks makes them smaller on average, but it does not force a fixed tick count per bar.
- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.
- In fixed mode, labels use the same DEFAULT_FIXED_MOVE distance for both stop loss and take profit.
- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.
