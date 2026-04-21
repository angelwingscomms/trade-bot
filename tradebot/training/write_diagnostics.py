from __future__ import annotations

from .shared import *  # noqa: F401,F403

def write_diagnostics(
    diagnostics_dir: Path,
    bars: pd.DataFrame,
    y_full: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    selected_primary_confidence: float,
    deployed_primary_confidence: float,
    validation_gate: dict[str, float | int],
    holdout_gate: dict[str, float | int],
    quality_gate_passed: bool,
    quality_gate_reason: str,
    available_window_counts: dict[str, int],
    used_window_counts: dict[str, int],
    use_atr_risk: bool,
    use_fixed_time_bars: bool,
    symbol: str,
    model_backend: str,
    loss_mode: str,
    focal_gamma: float,
    model_config_text: str,
    feature_columns: tuple[str, ...],
    feature_profile: str,
    point_size: float,
    fixed_move_price: float,
    use_fixed_tick_bars: bool,
    tick_density: int,
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    active_label_names = LABEL_NAMES_BINARY if val_probs.shape[1] == 2 else LABEL_NAMES
    val_predictions = build_prediction_frame(y_val, val_probs, selected_primary_confidence)
    test_predictions = build_prediction_frame(y_test, test_probs, selected_primary_confidence)
    val_confusion = confusion_matrix_df(
        y_val, val_predictions["pred_label"].to_numpy(dtype=np.int64), active_label_names
    )
    test_confusion = confusion_matrix_df(
        y_test, test_predictions["pred_label"].to_numpy(dtype=np.int64), active_label_names
    )

    bar_stats = bars.loc[
        :,
        ["time_open", "time_close", "tick_count", "spread", "close", "tick_imbalance"],
    ].copy()
    bar_stats.insert(0, "bar_index", np.arange(len(bar_stats), dtype=np.int64))
    bar_stats["duration_ms"] = bar_stats["time_close"] - bar_stats["time_open"]

    bar_stats.to_csv(diagnostics_dir / "bars.csv", index=False)
    val_predictions.to_csv(diagnostics_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(diagnostics_dir / "holdout_predictions.csv", index=False)
    val_confusion.to_csv(diagnostics_dir / "validation_confusion_matrix.csv")
    test_confusion.to_csv(diagnostics_dir / "holdout_confusion_matrix.csv")
    (diagnostics_dir / "config.mqh").write_text(model_config_text, encoding="utf-8")
    (diagnostics_dir / "active_features.txt").write_text("\n".join(feature_columns) + "\n", encoding="utf-8")

    report_lines = [
        "# Model Diagnostics",
        "",
        "## Run",
        f"- symbol: {symbol}",
        f"- backend: {model_backend}",
        f"- feature_profile: {feature_profile}",
        f"- feature_count: {len(feature_columns)}",
        f"- loss_mode: {loss_mode}",
        f"- focal_gamma: {focal_gamma:.2f}",
        "",
        "## Shared Config",
        f"- seq_len: {SEQ_LEN}",
        f"- label_timeout_bars: {LABEL_TIMEOUT_BARS}",
        f"- bar_mode: {'FIXED_TICK' if use_fixed_tick_bars else ('FIXED_TIME' if use_fixed_time_bars else 'IMBALANCE')}",
        *(
            [f"- primary_bar_seconds: {PRIMARY_BAR_SECONDS}"]
            if use_fixed_time_bars
            else (
                [f"- primary_tick_density: {tick_density}"]
                if use_fixed_tick_bars
                else [
                    f"- imbalance_min_ticks: {IMBALANCE_MIN_TICKS}",
                    f"- imbalance_ema_span: {IMBALANCE_EMA_SPAN}",
                ]
            )
        ),
        f"- feature_atr_period: {FEATURE_ATR_PERIOD}",
        f"- target_atr_period: {TARGET_ATR_PERIOD}",
        f"- rv_period: {RV_PERIOD}",
        f"- return_period: {RETURN_PERIOD}",
        f"- warmup_bars: {WARMUP_BARS}",
        f"- label_risk_mode: {'ATR' if use_atr_risk else 'FIXED'}",
        f"- point_size: {point_size:.8f}",
        f"- fixed_move_points: {DEFAULT_FIXED_MOVE:.2f}",
        f"- fixed_move_price: {fixed_move_price:.8f}",
        f"- label_sl_multiplier: {LABEL_SL_MULTIPLIER:.2f}",
        f"- label_tp_multiplier: {LABEL_TP_MULTIPLIER:.2f}",
        f"- execution_sl_multiplier: {EXECUTION_SL_MULTIPLIER:.2f}",
        f"- execution_tp_multiplier: {EXECUTION_TP_MULTIPLIER:.2f}",
        f"- use_all_windows: {int(USE_ALL_WINDOWS)}",
        f"- selected_primary_confidence: {selected_primary_confidence:.4f}",
        f"- deployed_primary_confidence: {deployed_primary_confidence:.4f}",
        f"- quality_gate_passed: {int(quality_gate_passed)}",
        f"- quality_gate_reason: {quality_gate_reason or '-'}",
        "",
        "## Bar Stats",
        f"- bars: {len(bars)}",
        *[f"- {line}" for line in summarize_numeric(bar_stats["tick_count"].to_numpy(), "ticks_per_bar")],
        *[f"- {line}" for line in summarize_numeric(bar_stats["duration_ms"].to_numpy(), "bar_duration_ms")],
        "",
        "## Label Counts",
        "- full bars:",
        *[f"  - {line}" for line in class_count_lines(y_full, active_label_names)],
        "- train windows:",
        *[f"  - {line}" for line in class_count_lines(y_train, active_label_names)],
        "- validation windows:",
        *[f"  - {line}" for line in class_count_lines(y_val, active_label_names)],
        "- holdout windows:",
        *[f"  - {line}" for line in class_count_lines(y_test, active_label_names)],
        "",
        "## Window Usage",
        f"- train_available: {available_window_counts['train']}",
        f"- train_used: {used_window_counts['train']}",
        f"- validation_available: {available_window_counts['validation']}",
        f"- validation_used: {used_window_counts['validation']}",
        f"- holdout_available: {available_window_counts['holdout']}",
        f"- holdout_used: {used_window_counts['holdout']}",
        "",
        "## Validation",
        f"- selected_trades: {int(validation_gate['selected_trades'])}",
        f"- trade_coverage: {float(validation_gate['trade_coverage']):.4f}",
        f"- selected_trade_precision: {format_metric(float(validation_gate['precision']))}",
        f"- selected_trade_mean_confidence: {format_metric(float(validation_gate['selected_mean_confidence']))}",
        f"- mean_confidence_all_predictions: {float(validation_gate['mean_confidence']):.4f}",
        "",
        "## Holdout",
        f"- selected_trades: {int(holdout_gate['selected_trades'])}",
        f"- trade_coverage: {float(holdout_gate['trade_coverage']):.4f}",
        f"- selected_trade_precision: {format_metric(float(holdout_gate['precision']))}",
        f"- selected_trade_mean_confidence: {format_metric(float(holdout_gate['selected_mean_confidence']))}",
        f"- mean_confidence_all_predictions: {float(holdout_gate['mean_confidence']):.4f}",
        "",
        "## Files",
        "- bars.csv",
        "- validation_predictions.csv",
        "- holdout_predictions.csv",
        "- validation_confusion_matrix.csv",
        "- holdout_confusion_matrix.csv",
        "- active_features.txt",
        "- config.mqh",
        "",
        "## Note",
        *(
            [f"- Bars are fixed-duration time buckets aligned to epoch time. Change PRIMARY_BAR_SECONDS in {CURRENT_CONFIG_PATH.name} to retune them, for example to 27 or 9 seconds."]
            if use_fixed_time_bars
            else (
                [f"- Fixed-tick bars use PRIMARY_TICK_DENSITY in {CURRENT_CONFIG_PATH.name} to set ticks per bar."]
                if use_fixed_tick_bars
                else ["- Imbalance bars are variable by design. Lowering imbalance_min_ticks makes them smaller on average, but it does not force a fixed tick count per bar."]
            )
        ),
        "- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.",
        "- In fixed mode, labels use the same DEFAULT_FIXED_MOVE value in symbol points for both stop loss and take profit.",
        "- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.",
    ]
    (diagnostics_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
