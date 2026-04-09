# Model Diagnostics

## Run
- symbol: XAUUSD
- backend: bilstm-attention
- feature_profile: sequence_extended
- feature_count: 36
- loss_mode: focal
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
- label_risk_mode: FIXED
- point_size: 0.00100000
- fixed_move_points: 144.00
- fixed_move_price: 0.14400000
- label_sl_multiplier: 5.40
- label_tp_multiplier: 5.40
- execution_sl_multiplier: 5.40
- execution_tp_multiplier: 5.40
- use_all_windows: 0
- selected_primary_confidence: 0.4000
- deployed_primary_confidence: 0.4000
- quality_gate_passed: 0
- quality_gate_reason: validation selected-trade precision 0.2955 < required 0.5000

## Bar Stats
- bars: 1425304
- ticks_per_bar min=2.00
- ticks_per_bar p50=5.00
- ticks_per_bar p90=13.00
- ticks_per_bar p99=27.00
- ticks_per_bar mean=6.98
- ticks_per_bar max=77.00
- bar_duration_ms min=0.00
- bar_duration_ms p50=1078.00
- bar_duration_ms p90=3416.00
- bar_duration_ms p99=7770.00
- bar_duration_ms mean=2254.65
- bar_duration_ms max=180250276.00

## Label Counts
- full bars:
  - HOLD: 700765
  - BUY: 368274
  - SELL: 356243
- train windows:
  - HOLD: 7049
  - BUY: 3755
  - SELL: 3596
- validation windows:
  - HOLD: 610
  - BUY: 424
  - SELL: 406
- holdout windows:
  - HOLD: 757
  - BUY: 350
  - SELL: 333

## Window Usage
- train_available: 997612
- train_used: 14400
- validation_available: 213658
- validation_used: 1440
- holdout_available: 213659
- holdout_used: 1440

## Validation
- selected_trades: 44
- trade_coverage: 0.0306
- selected_trade_precision: 0.2955
- selected_trade_mean_confidence: 0.4035
- mean_confidence_all_predictions: 0.3620

## Holdout
- selected_trades: 30
- trade_coverage: 0.0208
- selected_trade_precision: 0.3667
- selected_trade_mean_confidence: 0.4024
- mean_confidence_all_predictions: 0.3610

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
