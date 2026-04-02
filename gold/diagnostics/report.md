# Gold Diagnostics

## Shared Config
- seq_len: 54
- target_horizon: 27
- imbalance_min_ticks: 9
- imbalance_ema_span: 9
- feature_atr_period: 9
- target_atr_period: 9
- rv_period: 9
- return_period: 9
- warmup_bars: 9
- label_risk_mode: ATR
- fixed_move: 1.44
- label_sl_multiplier: 0.01
- label_tp_multiplier: 0.54
- execution_sl_multiplier: 0.54
- execution_tp_multiplier: 0.54
- use_all_windows: 0
- primary_confidence: 0.7250

## Bar Stats
- bars: 161775
- ticks_per_bar min=9.00
- ticks_per_bar p50=79.00
- ticks_per_bar p90=219.00
- ticks_per_bar p99=437.00
- ticks_per_bar mean=105.27
- ticks_per_bar max=1635.00
- bar_duration_ms min=976.00
- bar_duration_ms p50=15431.00
- bar_duration_ms p90=45871.60
- bar_duration_ms p99=103174.30
- bar_duration_ms mean=31078.81
- bar_duration_ms max=180298897.00

## Label Counts
- full bars:
  - HOLD: 24168
  - BUY: 20501
  - SELL: 117097
- train windows:
  - HOLD: 869
  - BUY: 721
  - SELL: 3810
- validation windows:
  - HOLD: 93
  - BUY: 67
  - SELL: 380
- holdout windows:
  - HOLD: 52
  - BUY: 37
  - SELL: 451

## Window Usage
- train_available: 113156
- train_used: 5400
- validation_available: 24131
- validation_used: 540
- holdout_available: 24131
- holdout_used: 540

## Validation
- selected_trades: 432
- trade_coverage: 0.8000
- mean_confidence: 0.9755

## Holdout
- selected_trades: 477
- trade_coverage: 0.8833
- mean_confidence: 0.9759

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
