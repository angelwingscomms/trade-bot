# Gold Diagnostics

## Shared Config
- seq_len: 54
- target_horizon: 27
- imbalance_min_ticks: 27
- imbalance_ema_span: 18
- feature_atr_period: 9
- target_atr_period: 9
- rv_period: 9
- return_period: 9
- warmup_bars: 9
- sl_multiplier: 0.54
- tp_multiplier: 0.54
- primary_confidence: 0.4000

## Bar Stats
- bars: 34264
- ticks_per_bar min=27.00
- ticks_per_bar p50=346.00
- ticks_per_bar p90=1060.00
- ticks_per_bar p99=2310.00
- ticks_per_bar mean=497.04
- ticks_per_bar max=6846.00
- bar_duration_ms min=4902.00
- bar_duration_ms p50=70753.50
- bar_duration_ms p90=222512.10
- bar_duration_ms p99=521419.56
- bar_duration_ms mean=147619.40
- bar_duration_ms max=180630043.00

## Label Counts
- full bars:
  - HOLD: 27
  - BUY: 10781
  - SELL: 23447
- train windows:
  - HOLD: 0
  - BUY: 548
  - SELL: 910
- validation windows:
  - HOLD: 0
  - BUY: 112
  - SELL: 266
- holdout windows:
  - HOLD: 0
  - BUY: 29
  - SELL: 349

## Validation
- selected_trades: 378
- trade_coverage: 1.0000
- mean_confidence: 0.9997

## Holdout
- selected_trades: 378
- trade_coverage: 1.0000
- mean_confidence: 0.9977

## Files
- bars.csv
- validation_predictions.csv
- holdout_predictions.csv
- validation_confusion_matrix.csv
- holdout_confusion_matrix.csv
- shared_config_snapshot.mqh

## Note
- Imbalance bars are variable by design. Lowering imbalance_min_ticks makes them smaller on average, but it does not force a fixed tick count per bar.
