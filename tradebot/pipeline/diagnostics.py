"""Diagnostics and report-writing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiagnosticsConfig:
    current_config_name: str
    seq_len: int
    target_horizon: int
    primary_bar_seconds: int
    imbalance_min_ticks: int
    imbalance_ema_span: int
    feature_atr_period: int
    target_atr_period: int
    rv_period: int
    return_period: int
    warmup_bars: int
    default_fixed_move: float
    label_sl_multiplier: float
    label_tp_multiplier: float
    execution_sl_multiplier: float
    execution_tp_multiplier: float
    use_all_windows: bool


def class_count_lines(labels: np.ndarray, label_names: tuple[str, ...]) -> list[str]:
    counts = np.bincount(labels, minlength=len(label_names))
    return [f"{label_names[i]}: {int(counts[i])}" for i in range(len(label_names))]


def confusion_matrix_df(labels: np.ndarray, preds: np.ndarray, label_names: tuple[str, ...]) -> pd.DataFrame:
    matrix = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    for true_label, pred_label in zip(labels.astype(np.int64), preds.astype(np.int64)):
        matrix[true_label, pred_label] += 1
    return pd.DataFrame(
        matrix,
        index=[f"true_{name.lower()}" for name in label_names],
        columns=[f"pred_{name.lower()}" for name in label_names],
    )


def summarize_numeric(values: np.ndarray, label: str) -> list[str]:
    array = np.asarray(values, dtype=np.float64)
    return [
        f"{label} min={array.min():.2f}",
        f"{label} p50={np.percentile(array, 50):.2f}",
        f"{label} p90={np.percentile(array, 90):.2f}",
        f"{label} p99={np.percentile(array, 99):.2f}",
        f"{label} mean={array.mean():.2f}",
        f"{label} max={array.max():.2f}",
    ]


def format_metric(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def build_prediction_frame(labels: np.ndarray, probs: np.ndarray, threshold: float, label_names: tuple[str, ...]) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    selected = confidences >= threshold if probs.shape[1] == 2 else (preds > 0) & (confidences >= threshold)
    frame = pd.DataFrame(
        {
            "true_label": labels.astype(np.int64),
            "pred_label": preds.astype(np.int64),
            "true_name": [label_names[int(v)] for v in labels],
            "pred_name": [label_names[int(v)] for v in preds],
            "confidence": confidences,
            "selected_trade": selected.astype(np.int64),
            "correct": (preds == labels).astype(np.int64),
        }
    )
    if probs.shape[1] == 3:
        frame["prob_hold"] = probs[:, 0]
        frame["prob_buy"] = probs[:, 1]
        frame["prob_sell"] = probs[:, 2]
    else:
        frame["prob_buy"] = probs[:, 0]
        frame["prob_sell"] = probs[:, 1]
    return frame


def write_diagnostics(
    diagnostics_dir: Path,
    *,
    config: DiagnosticsConfig,
    bars: pd.DataFrame,
    y_full: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    label_names: tuple[str, ...],
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

    val_predictions = build_prediction_frame(y_val, val_probs, selected_primary_confidence, label_names)
    test_predictions = build_prediction_frame(y_test, test_probs, selected_primary_confidence, label_names)
    val_confusion = confusion_matrix_df(
        y_val,
        val_predictions["pred_label"].to_numpy(dtype=np.int64),
        label_names,
    )
    test_confusion = confusion_matrix_df(
        y_test,
        test_predictions["pred_label"].to_numpy(dtype=np.int64),
        label_names,
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
        f"- seq_len: {config.seq_len}",
        f"- target_horizon: {config.target_horizon}",
        f"- bar_mode: {'FIXED_TICK' if use_fixed_tick_bars else ('FIXED_TIME' if use_fixed_time_bars else 'IMBALANCE')}",
        *(
            [f"- primary_bar_seconds: {config.primary_bar_seconds}"]
            if use_fixed_time_bars
            else (
                [f"- primary_tick_density: {tick_density}"]
                if use_fixed_tick_bars
                else [
                    f"- imbalance_min_ticks: {config.imbalance_min_ticks}",
                    f"- imbalance_ema_span: {config.imbalance_ema_span}",
                ]
            )
        ),
        f"- feature_atr_period: {config.feature_atr_period}",
        f"- target_atr_period: {config.target_atr_period}",
        f"- rv_period: {config.rv_period}",
        f"- return_period: {config.return_period}",
        f"- warmup_bars: {config.warmup_bars}",
        f"- label_risk_mode: {'ATR' if use_atr_risk else 'FIXED'}",
        f"- point_size: {point_size:.8f}",
        f"- fixed_move_points: {config.default_fixed_move:.2f}",
        f"- fixed_move_price: {fixed_move_price:.8f}",
        f"- label_sl_multiplier: {config.label_sl_multiplier:.2f}",
        f"- label_tp_multiplier: {config.label_tp_multiplier:.2f}",
        f"- execution_sl_multiplier: {config.execution_sl_multiplier:.2f}",
        f"- execution_tp_multiplier: {config.execution_tp_multiplier:.2f}",
        f"- use_all_windows: {int(config.use_all_windows)}",
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
        *[f"  - {line}" for line in class_count_lines(y_full, label_names)],
        "- train windows:",
        *[f"  - {line}" for line in class_count_lines(y_train, label_names)],
        "- validation windows:",
        *[f"  - {line}" for line in class_count_lines(y_val, label_names)],
        "- holdout windows:",
        *[f"  - {line}" for line in class_count_lines(y_test, label_names)],
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
            [f"- Bars are fixed-duration time buckets aligned to epoch time. Change PRIMARY_BAR_SECONDS in {config.current_config_name} to retune them."]
            if use_fixed_time_bars
            else (
                [f"- Fixed-tick bars use PRIMARY_TICK_DENSITY in {config.current_config_name} to set ticks per bar."]
                if use_fixed_tick_bars
                else ["- Imbalance bars are variable by design. Lower imbalance thresholds make smaller bars on average."]
            )
        ),
        "- In ATR mode, labels use the label_sl_multiplier and label_tp_multiplier settings.",
        "- In fixed mode, labels use DEFAULT_FIXED_MOVE for both stop loss and take profit.",
        "- When use_all_windows is 0, the trainer evenly subsamples down to the configured train/eval caps.",
    ]
    (diagnostics_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
