from __future__ import annotations

import argparse
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gold_mamba_lite import GoldMambaLiteClassifier
from model_archive import (
    ACTIVE_DIAGNOSTICS_DIR,
    ACTIVE_MODEL_CONFIG_PATH,
    ACTIVE_ONNX_PATH,
    ACTIVE_SHARED_CONFIG_PATH,
    DEFAULT_METAEDITOR_PATH,
    LIVE_EX5_PATH,
    MODELS_DIR,
    compile_live_expert,
    ensure_default_test_config,
    format_model_stamp,
    read_text_best_effort,
    resolve_metaeditor_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gold.nn")

EPS = 1e-10
DEFAULT_DATA_FILE = "gold_market_ticks.csv"
DEFAULT_OUTPUT_FILE = ACTIVE_ONNX_PATH.name
SHARED_CONFIG_PATH = ACTIVE_SHARED_CONFIG_PATH
DEFINE_PATTERN = re.compile(r"^\s*#define\s+([A-Z0-9_]+)\s+(.+?)\s*$")
FEATURE_MACRO_TO_NAME = {
    "FEATURE_IDX_RET1": "ret1",
    "FEATURE_IDX_HIGH_REL_PREV": "high_rel_prev",
    "FEATURE_IDX_LOW_REL_PREV": "low_rel_prev",
    "FEATURE_IDX_SPREAD_REL": "spread_rel",
    "FEATURE_IDX_CLOSE_IN_RANGE": "close_in_range",
    "FEATURE_IDX_ATR_REL": "atr_rel",
    "FEATURE_IDX_RV": "rv",
    "FEATURE_IDX_RETURN_N": "ret_n",
    "FEATURE_IDX_TICK_IMBALANCE": "tick_imbalance",
}


def parse_define_value(raw_value: str, known_values: dict[str, int | float | str]) -> int | float | str:
    value = raw_value.split("//", 1)[0].strip()
    if value.endswith("f"):
        value = value[:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return eval(value, {"__builtins__": {}}, dict(known_values))


def load_shared_config(path: Path) -> dict[str, int | float | str]:
    values: dict[str, int | float | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = DEFINE_PATTERN.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        values[name] = parse_define_value(raw_value, values)
    return values


SHARED = load_shared_config(SHARED_CONFIG_PATH)
SEQ_LEN = int(SHARED["SEQ_LEN"])
TARGET_HORIZON = int(SHARED["TARGET_HORIZON"])
MODEL_FEATURE_COUNT = int(SHARED["MODEL_FEATURE_COUNT"])
FEATURE_ATR_PERIOD = int(SHARED["FEATURE_ATR_PERIOD"])
TARGET_ATR_PERIOD = int(SHARED["TARGET_ATR_PERIOD"])
RV_PERIOD = int(SHARED["RV_PERIOD"])
RETURN_PERIOD = int(SHARED["RETURN_PERIOD"])
WARMUP_BARS = int(SHARED["WARMUP_BARS"])
IMBALANCE_MIN_TICKS = int(SHARED["IMBALANCE_MIN_TICKS"])
IMBALANCE_EMA_SPAN = int(SHARED["IMBALANCE_EMA_SPAN"])
PRIMARY_BAR_SECONDS = int(SHARED["PRIMARY_BAR_SECONDS"])
BAR_DURATION_MS = PRIMARY_BAR_SECONDS * 1000
DEFAULT_FIXED_MOVE = float(SHARED["DEFAULT_FIXED_MOVE"])
LABEL_SL_MULTIPLIER = float(SHARED["LABEL_SL_MULTIPLIER"])
LABEL_TP_MULTIPLIER = float(SHARED["LABEL_TP_MULTIPLIER"])
EXECUTION_SL_MULTIPLIER = float(SHARED["DEFAULT_SL_MULTIPLIER"])
EXECUTION_TP_MULTIPLIER = float(SHARED["DEFAULT_TP_MULTIPLIER"])
USE_ALL_WINDOWS = bool(int(SHARED["USE_ALL_WINDOWS"]))
DEFAULT_EPOCHS = int(SHARED["DEFAULT_EPOCHS"])
DEFAULT_BATCH_SIZE = int(SHARED["DEFAULT_BATCH_SIZE"])
DEFAULT_MAX_TRAIN_WINDOWS = int(SHARED["DEFAULT_MAX_TRAIN_WINDOWS"])
DEFAULT_MAX_EVAL_WINDOWS = int(SHARED["DEFAULT_MAX_EVAL_WINDOWS"])
DEFAULT_PATIENCE = int(SHARED["DEFAULT_PATIENCE"])

FEATURE_COLUMNS = [None] * MODEL_FEATURE_COUNT
for macro_name, feature_name in FEATURE_MACRO_TO_NAME.items():
    feature_index = int(SHARED[macro_name])
    FEATURE_COLUMNS[feature_index] = feature_name
if any(name is None for name in FEATURE_COLUMNS):
    raise RuntimeError("gold_shared_config.mqh does not define a complete feature order.")
FEATURE_COLUMNS = tuple(FEATURE_COLUMNS)
LABEL_NAMES = ("HOLD", "BUY", "SELL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GOLD model using gold_shared_config.mqh as the shared source of truth."
    )
    parser.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE, help="CSV with time_msc,bid,ask.")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="ONNX output file.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=DEFAULT_MAX_TRAIN_WINDOWS,
        help="Training window cap.",
    )
    parser.add_argument(
        "--max-eval-windows",
        type=int,
        default=DEFAULT_MAX_EVAL_WINDOWS,
        help="Validation/test window cap.",
    )
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="", help="Optional torch device override.")
    parser.add_argument(
        "-r",
        "--use-fixed-risk",
        "--use-fixed-stops",
        dest="use_fixed_risk",
        action="store_true",
        help="Use fixed label stops/targets. Without this flag, training labels use ATR-based barriers.",
    )
    parser.add_argument(
        "-i",
        "--use-fixed-time-bars",
        action="store_true",
        help="Build, train, and compile using fixed-time bars instead of the default imbalance bars.",
    )
    parser.add_argument(
        "--metaeditor-path",
        type=str,
        default=str(DEFAULT_METAEDITOR_PATH),
        help="Path to MetaEditor used to compile gold/live.mq5 after training.",
    )
    parser.add_argument(
        "--skip-live-compile",
        action="store_true",
        help="Skip automatic live.mq5 compile after exporting ONNX and config.",
    )
    return parser.parse_args()


def resolve_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def compute_tick_signs(prices: np.ndarray) -> np.ndarray:
    signs = np.empty(len(prices), dtype=np.int8)
    last_sign = 1
    prev_price = float(prices[0]) if len(prices) else 0.0
    for i, price in enumerate(prices):
        if i > 0:
            diff = float(price) - prev_price
            if diff > 0.0:
                last_sign = 1
            elif diff < 0.0:
                last_sign = -1
        signs[i] = last_sign
        prev_price = float(price)
    return signs


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    high_values = high.to_numpy(dtype=np.float64, copy=False)
    low_values = low.to_numpy(dtype=np.float64, copy=False)
    close_values = close.to_numpy(dtype=np.float64, copy=False)

    tr = np.empty(len(close_values), dtype=np.float64)
    if len(tr) == 0:
        return pd.Series(dtype=np.float64, index=close.index)

    tr[0] = high_values[0] - low_values[0]
    for i in range(1, len(tr)):
        tr[i] = max(
            high_values[i] - low_values[i],
            abs(high_values[i] - close_values[i - 1]),
            abs(low_values[i] - close_values[i - 1]),
        )

    atr = np.full(len(tr), np.nan, dtype=np.float64)
    if len(tr) >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / period

    return pd.Series(atr, index=close.index, dtype=np.float64)


def build_primary_bar_ids(df_ticks: pd.DataFrame) -> np.ndarray:
    prices = df_ticks["bid"].to_numpy(dtype=np.float64, copy=False)
    tick_signs = compute_tick_signs(prices)
    alpha = 2.0 / (max(1, IMBALANCE_EMA_SPAN) + 1.0)
    expected_abs_theta = max(2.0, float(max(2, IMBALANCE_MIN_TICKS // 3)))
    bar_ids = np.empty(len(prices), dtype=np.int64)
    current_bar = 0
    ticks_in_bar = 0
    theta = 0.0

    for i, sign in enumerate(tick_signs):
        bar_ids[i] = current_bar
        ticks_in_bar += 1
        theta += float(sign)
        if ticks_in_bar >= IMBALANCE_MIN_TICKS and abs(theta) >= expected_abs_theta:
            observed = max(2.0, abs(theta))
            expected_abs_theta = (1.0 - alpha) * expected_abs_theta + alpha * observed
            current_bar += 1
            ticks_in_bar = 0
            theta = 0.0

    return bar_ids


def build_time_bar_ids(time_msc: np.ndarray) -> np.ndarray:
    if PRIMARY_BAR_SECONDS <= 0:
        raise ValueError("PRIMARY_BAR_SECONDS must be positive.")
    return time_msc // BAR_DURATION_MS


def build_gold_bars(csv_path: Path, use_fixed_time_bars: bool) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(
        csv_path,
        usecols=["time_msc", "bid", "ask"],
        dtype={"time_msc": np.int64, "bid": np.float64, "ask": np.float64},
    ).sort_values("time_msc").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No ticks found in {csv_path}")

    df["tick_sign"] = compute_tick_signs(df["bid"].to_numpy(dtype=np.float64, copy=False))
    df["spread"] = df["ask"] - df["bid"]

    if use_fixed_time_bars:
        df["bar_id"] = build_time_bar_ids(df["time_msc"].to_numpy(dtype=np.int64, copy=False))
        grouped = (
            df.groupby("bar_id", sort=True)
            .agg(
                open=("bid", "first"),
                high=("bid", "max"),
                low=("bid", "min"),
                close=("bid", "last"),
                tick_count=("bid", "size"),
                tick_imbalance=("tick_sign", "mean"),
                ask_high=("ask", "max"),
                ask_low=("ask", "min"),
                spread=("spread", "last"),
            )
            .reset_index()
        )
        grouped["time_open"] = grouped["bar_id"] * BAR_DURATION_MS
        grouped["time_close"] = grouped["time_open"] + BAR_DURATION_MS
        grouped = grouped.drop(columns=["bar_id"])
        log.info(
            "Built %d gold bars in %.2fs using fixed %ds bars",
            len(grouped),
            time.time() - t0,
            PRIMARY_BAR_SECONDS,
        )
        return grouped

    df["bar_id"] = build_primary_bar_ids(df)
    grouped = (
        df.groupby("bar_id")
        .agg(
            open=("bid", "first"),
            high=("bid", "max"),
            low=("bid", "min"),
            close=("bid", "last"),
            time_open=("time_msc", "first"),
            time_close=("time_msc", "last"),
            tick_count=("bid", "size"),
            tick_imbalance=("tick_sign", "mean"),
            ask_high=("ask", "max"),
            ask_low=("ask", "min"),
            spread=("spread", "last"),
        )
        .reset_index(drop=True)
    )
    log.info(
        "Built %d gold bars in %.2fs using imbalance bars min_ticks=%d span=%d",
        len(grouped),
        time.time() - t0,
        IMBALANCE_MIN_TICKS,
        IMBALANCE_EMA_SPAN,
    )
    return grouped


def compute_features(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    ret1 = np.log(close / (prev_close + EPS))
    atr_feature = wilder_atr(df["high"], df["low"], close, period=FEATURE_ATR_PERIOD)

    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = ret1
    feat["high_rel_prev"] = np.log(df["high"] / (prev_close + EPS))
    feat["low_rel_prev"] = np.log(df["low"] / (prev_close + EPS))
    feat["spread_rel"] = df["spread"] / (close + EPS)
    feat["close_in_range"] = (close - df["low"]) / (df["high"] - df["low"] + 1e-8)
    feat["atr_rel"] = atr_feature / (close + EPS)
    feat["rv"] = ret1.rolling(RV_PERIOD, min_periods=RV_PERIOD).std(ddof=0)
    feat["ret_n"] = np.log(close / (close.shift(RETURN_PERIOD) + EPS))
    feat["tick_imbalance"] = df["tick_imbalance"].astype(float)
    return feat.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)


def get_triple_barrier_labels(df_gold: pd.DataFrame, use_atr_risk: bool) -> np.ndarray:
    close = df_gold["close"].to_numpy(dtype=np.float64, copy=False)
    high = df_gold["high"].to_numpy(dtype=np.float64, copy=False)
    low = df_gold["low"].to_numpy(dtype=np.float64, copy=False)
    spread = df_gold["spread"].to_numpy(dtype=np.float64, copy=False)
    ask_high = df_gold["ask_high"].to_numpy(dtype=np.float64, copy=False)
    ask_low = df_gold["ask_low"].to_numpy(dtype=np.float64, copy=False)
    atr_target = None
    if use_atr_risk:
        atr_target = wilder_atr(
            df_gold["high"], df_gold["low"], df_gold["close"], period=TARGET_ATR_PERIOD
        ).to_numpy(dtype=np.float64, copy=False)

    labels = np.zeros(len(df_gold), dtype=np.int64)
    for i in range(len(df_gold) - TARGET_HORIZON):
        long_entry = close[i] + spread[i]
        short_entry = close[i]
        if use_atr_risk:
            vol = atr_target[i]
            if not np.isfinite(vol) or vol <= 0.0:
                continue
            long_tp = long_entry + LABEL_TP_MULTIPLIER * vol
            long_sl = long_entry - LABEL_SL_MULTIPLIER * vol
            short_tp = short_entry - LABEL_TP_MULTIPLIER * vol
            short_sl = short_entry + LABEL_SL_MULTIPLIER * vol
        else:
            long_tp = long_entry + DEFAULT_FIXED_MOVE
            long_sl = long_entry - DEFAULT_FIXED_MOVE
            short_tp = short_entry - DEFAULT_FIXED_MOVE
            short_sl = short_entry + DEFAULT_FIXED_MOVE

        long_result = 0
        short_result = 0
        for j in range(i + 1, i + TARGET_HORIZON + 1):
            if long_result == 0:
                hit_tp = high[j] >= long_tp
                hit_sl = low[j] <= long_sl
                if hit_tp and not hit_sl:
                    long_result = 1
                elif hit_sl:
                    long_result = -1

            if short_result == 0:
                hit_tp = ask_low[j] <= short_tp
                hit_sl = ask_high[j] >= short_sl
                if hit_tp and not hit_sl:
                    short_result = 1
                elif hit_sl:
                    short_result = -1

            if long_result != 0 and short_result != 0:
                break

        if long_result == 1 and short_result != 1:
            labels[i] = 1
        elif short_result == 1 and long_result != 1:
            labels[i] = 2

    return labels


def choose_evenly_spaced(indices: np.ndarray, max_count: int) -> np.ndarray:
    if len(indices) <= max_count:
        return indices.astype(np.int64, copy=False)
    positions = np.linspace(0, len(indices) - 1, max_count)
    return indices[np.unique(np.round(positions).astype(np.int64))]


def maybe_cap_windows(indices: np.ndarray, max_count: int, use_all_windows: bool) -> np.ndarray:
    if use_all_windows:
        return indices.astype(np.int64, copy=False)
    return choose_evenly_spaced(indices, max_count)


def build_segment_end_indices(
    valid_mask: np.ndarray,
    start_bar: int,
    end_bar: int,
    seq_len: int,
    horizon: int,
) -> np.ndarray:
    first_end = start_bar + seq_len - 1
    last_end = end_bar - horizon - 1
    if last_end < first_end:
        return np.empty(0, dtype=np.int64)

    ends = []
    for end_idx in range(first_end, last_end + 1):
        start_idx = end_idx - seq_len + 1
        if valid_mask[start_idx : end_idx + 1].all():
            ends.append(end_idx)
    return np.asarray(ends, dtype=np.int64)


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    end_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.empty((len(end_indices), seq_len, features.shape[1]), dtype=np.float32)
    ys = np.empty(len(end_indices), dtype=np.int64)
    for i, end_idx in enumerate(end_indices):
        start_idx = end_idx - seq_len + 1
        xs[i] = features[start_idx : end_idx + 1]
        ys[i] = labels[end_idx]
    return xs, ys


def make_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = np.ones(3, dtype=np.float32)
    total = counts.sum()
    for cls in range(3):
        if counts[cls] > 0:
            weights[cls] = total / (3.0 * counts[cls])
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    log.info("evaluate_model: starting - %d batches", len(loader))
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if batch_idx % 100 == 0:
                log.info("evaluate_model: batch %d/%d", batch_idx, len(loader))
            logits_list.append(model(xb.to(device)).cpu().numpy())
            labels_list.append(yb.numpy())
    log.info("evaluate_model: done - concatenating %d arrays", len(logits_list))
    result = np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)
    log.info("evaluate_model: result shapes - logits=%s, labels=%s", result[0].shape, result[1].shape)
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def choose_confidence_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    candidate_mask = preds > 0
    if not candidate_mask.any():
        return 0.60

    min_selected = max(12, int(0.03 * len(labels)))
    best_threshold = 0.60
    best_precision = -1.0
    best_coverage = -1.0
    confidences = probs.max(axis=1)

    for threshold in np.linspace(0.40, 0.85, 19):
        selected = candidate_mask & (confidences >= threshold)
        if selected.sum() < min_selected:
            continue

        precision = float((preds[selected] == labels[selected]).mean())
        coverage = float(selected.mean())
        if precision > best_precision + 1e-12 or (
            abs(precision - best_precision) <= 1e-12 and coverage > best_coverage
        ):
            best_threshold = float(threshold)
            best_precision = precision
            best_coverage = coverage

    print ("Chosen confidence threshold: %.2f with precision %.4f and coverage %.4f" % (best_threshold, best_precision, best_coverage))
    return best_threshold


def summarize_gate(name: str, probs: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    preds = probs.argmax(axis=1)
    selected = (preds > 0) & (probs.max(axis=1) >= threshold)
    coverage = float(selected.mean())
    if selected.any():
        precision = float((preds[selected] == labels[selected]).mean())
        log.info(
            "%s: threshold=%.2f precision=%.4f coverage=%.4f trades=%d",
            name,
            threshold,
            precision,
            coverage,
            int(selected.sum()),
        )
    else:
        log.warning("%s: threshold=%.2f produced no trades.", name, threshold)


def class_count_lines(labels: np.ndarray) -> list[str]:
    counts = np.bincount(labels, minlength=len(LABEL_NAMES))
    return [f"{LABEL_NAMES[i]}: {int(counts[i])}" for i in range(len(LABEL_NAMES))]


def confusion_matrix_df(labels: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    matrix = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)
    for true_label, pred_label in zip(labels.astype(np.int64), preds.astype(np.int64)):
        matrix[true_label, pred_label] += 1
    return pd.DataFrame(
        matrix,
        index=[f"true_{name.lower()}" for name in LABEL_NAMES],
        columns=[f"pred_{name.lower()}" for name in LABEL_NAMES],
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


def build_prediction_frame(labels: np.ndarray, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    selected = (preds > 0) & (confidences >= threshold)
    frame = pd.DataFrame(
        {
            "true_label": labels.astype(np.int64),
            "pred_label": preds.astype(np.int64),
            "true_name": [LABEL_NAMES[int(v)] for v in labels],
            "pred_name": [LABEL_NAMES[int(v)] for v in preds],
            "prob_hold": probs[:, 0],
            "prob_buy": probs[:, 1],
            "prob_sell": probs[:, 2],
            "confidence": confidences,
            "selected_trade": selected.astype(np.int64),
            "correct": (preds == labels).astype(np.int64),
        }
    )
    return frame


def write_diagnostics(
    diagnostics_dir: Path,
    df_gold: pd.DataFrame,
    y_full: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    primary_confidence: float,
    available_window_counts: dict[str, int],
    used_window_counts: dict[str, int],
    use_atr_risk: bool,
    use_fixed_time_bars: bool,
    model_config_text: str,
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    val_predictions = build_prediction_frame(y_val, val_probs, primary_confidence)
    test_predictions = build_prediction_frame(y_test, test_probs, primary_confidence)
    val_confusion = confusion_matrix_df(y_val, val_predictions["pred_label"].to_numpy(dtype=np.int64))
    test_confusion = confusion_matrix_df(y_test, test_predictions["pred_label"].to_numpy(dtype=np.int64))

    bar_stats = df_gold.loc[
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
    (diagnostics_dir / "shared_config_snapshot.mqh").write_text(
        SHARED_CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (diagnostics_dir / "model_config_snapshot.mqh").write_text(model_config_text, encoding="utf-8")

    selected_val = val_predictions["selected_trade"].to_numpy(dtype=np.int64)
    selected_test = test_predictions["selected_trade"].to_numpy(dtype=np.int64)
    report_lines = [
        "# Gold Diagnostics",
        "",
        "## Shared Config",
        f"- seq_len: {SEQ_LEN}",
        f"- target_horizon: {TARGET_HORIZON}",
        f"- bar_mode: {'FIXED_TIME' if use_fixed_time_bars else 'IMBALANCE'}",
        *(
            [f"- primary_bar_seconds: {PRIMARY_BAR_SECONDS}"]
            if use_fixed_time_bars
            else [
                f"- imbalance_min_ticks: {IMBALANCE_MIN_TICKS}",
                f"- imbalance_ema_span: {IMBALANCE_EMA_SPAN}",
            ]
        ),
        f"- feature_atr_period: {FEATURE_ATR_PERIOD}",
        f"- target_atr_period: {TARGET_ATR_PERIOD}",
        f"- rv_period: {RV_PERIOD}",
        f"- return_period: {RETURN_PERIOD}",
        f"- warmup_bars: {WARMUP_BARS}",
        f"- label_risk_mode: {'ATR' if use_atr_risk else 'FIXED'}",
        f"- fixed_move: {DEFAULT_FIXED_MOVE:.2f}",
        f"- label_sl_multiplier: {LABEL_SL_MULTIPLIER:.2f}",
        f"- label_tp_multiplier: {LABEL_TP_MULTIPLIER:.2f}",
        f"- execution_sl_multiplier: {EXECUTION_SL_MULTIPLIER:.2f}",
        f"- execution_tp_multiplier: {EXECUTION_TP_MULTIPLIER:.2f}",
        f"- use_all_windows: {int(USE_ALL_WINDOWS)}",
        f"- primary_confidence: {primary_confidence:.4f}",
        "",
        "## Bar Stats",
        f"- bars: {len(df_gold)}",
        *[f"- {line}" for line in summarize_numeric(bar_stats["tick_count"].to_numpy(), "ticks_per_bar")],
        *[f"- {line}" for line in summarize_numeric(bar_stats["duration_ms"].to_numpy(), "bar_duration_ms")],
        "",
        "## Label Counts",
        "- full bars:",
        *[f"  - {line}" for line in class_count_lines(y_full)],
        "- train windows:",
        *[f"  - {line}" for line in class_count_lines(y_train)],
        "- validation windows:",
        *[f"  - {line}" for line in class_count_lines(y_val)],
        "- holdout windows:",
        *[f"  - {line}" for line in class_count_lines(y_test)],
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
        f"- selected_trades: {int(selected_val.sum())}",
        f"- trade_coverage: {float(selected_val.mean()):.4f}",
        f"- mean_confidence: {float(val_predictions['confidence'].mean()):.4f}",
        "",
        "## Holdout",
        f"- selected_trades: {int(selected_test.sum())}",
        f"- trade_coverage: {float(selected_test.mean()):.4f}",
        f"- mean_confidence: {float(test_predictions['confidence'].mean()):.4f}",
        "",
        "## Files",
        "- bars.csv",
        "- validation_predictions.csv",
        "- holdout_predictions.csv",
        "- validation_confusion_matrix.csv",
        "- holdout_confusion_matrix.csv",
        "- shared_config_snapshot.mqh",
        "- model_config_snapshot.mqh",
        "",
        "## Note",
        *(
            [f"- Bars are fixed-duration time buckets aligned to epoch time. Change PRIMARY_BAR_SECONDS in gold_shared_config.mqh to retune them, for example to 27 or 9 seconds."]
            if use_fixed_time_bars
            else ["- Imbalance bars are variable by design. Lowering imbalance_min_ticks makes them smaller on average, but it does not force a fixed tick count per bar."]
        ),
        "- In ATR mode, labels use the stricter label_sl_multiplier and label_tp_multiplier values, so a BUY/SELL label means price reached the target before making more than a tiny adverse move.",
        "- In fixed mode, labels use the same DEFAULT_FIXED_MOVE distance for both stop loss and take profit.",
        "- When use_all_windows is 0, the trainer evenly subsamples window endpoints down to the max_train_windows and max_eval_windows caps to keep runs fast.",
    ]
    (diagnostics_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def format_float_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(v):.8f}f" for v in values)


def build_mql_config(
    median: np.ndarray,
    iqr: np.ndarray,
    primary_confidence: float,
    use_atr_risk: bool,
    use_fixed_time_bars: bool,
) -> str:
    return "\n".join(
        [
            "// Auto-generated by gold/nn.py. Re-run training to refresh these values.",
            "// Shared static values live in gold_shared_config.mqh.",
            f"#define MODEL_USE_ATR_RISK {1 if use_atr_risk else 0}",
            f"#define MODEL_USE_FIXED_TIME_BARS {1 if use_fixed_time_bars else 0}",
            f"#define PRIMARY_CONFIDENCE {primary_confidence:.8f}",
            f"float medians[MODEL_FEATURE_COUNT] = {{{format_float_array(median)}}};",
            f"float iqrs[MODEL_FEATURE_COUNT] = {{{format_float_array(iqr)}}};",
        ]
    )


def sync_directory_contents(source_dir: Path, destination_dir: Path) -> None:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def main() -> None:
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    use_atr_risk = not bool(args.use_fixed_risk)
    use_fixed_time_bars = bool(args.use_fixed_time_bars)
    if use_fixed_time_bars and PRIMARY_BAR_SECONDS <= 0:
        raise ValueError("PRIMARY_BAR_SECONDS must be positive.")
    if DEFAULT_FIXED_MOVE <= 0.0:
        raise ValueError("DEFAULT_FIXED_MOVE must be positive.")

    data_path = resolve_local_path(args.data_file)
    output_path = resolve_local_path(args.output_file)
    active_output_path = ACTIVE_ONNX_PATH
    config_path = ACTIVE_MODEL_CONFIG_PATH
    active_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using device: %s", device)
    log.info(
        "Shared config | seq_len=%d horizon=%d atr_feature=%d atr_target=%d rv=%d ret=%d "
        "bar_mode=%s imbalance_min_ticks=%d imbalance_ema_span=%d bar_seconds=%d risk_mode=%s fixed_move=%.2f "
        "label_sl=%.2f label_tp=%.2f exec_sl=%.2f exec_tp=%.2f use_all_windows=%d",
        SEQ_LEN,
        TARGET_HORIZON,
        FEATURE_ATR_PERIOD,
        TARGET_ATR_PERIOD,
        RV_PERIOD,
        RETURN_PERIOD,
        "FIXED_TIME" if use_fixed_time_bars else "IMBALANCE",
        IMBALANCE_MIN_TICKS,
        IMBALANCE_EMA_SPAN,
        PRIMARY_BAR_SECONDS,
        "ATR" if use_atr_risk else "FIXED",
        DEFAULT_FIXED_MOVE,
        LABEL_SL_MULTIPLIER,
        LABEL_TP_MULTIPLIER,
        EXECUTION_SL_MULTIPLIER,
        EXECUTION_TP_MULTIPLIER,
        int(USE_ALL_WINDOWS),
    )

    df_gold = build_gold_bars(data_path, use_fixed_time_bars=use_fixed_time_bars)
    x_all = compute_features(df_gold)
    y_all = get_triple_barrier_labels(df_gold, use_atr_risk=use_atr_risk)

    x = x_all[WARMUP_BARS:]
    y = y_all[WARMUP_BARS:]
    n_rows = len(x)
    embargo = max(SEQ_LEN, TARGET_HORIZON)

    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    train_range = (0, train_end)
    val_range = (train_end + embargo, val_end)
    test_range = (val_end + embargo, n_rows)
    if test_range[0] >= test_range[1]:
        raise ValueError("Dataset is too small for leakage-safe train/val/test splits.")

    median = np.nanmedian(x[: train_range[1]], axis=0)
    median = np.nan_to_num(median, nan=0.0)
    iqr = np.nanpercentile(x[: train_range[1]], 75, axis=0) - np.nanpercentile(x[: train_range[1]], 25, axis=0)
    iqr = np.nan_to_num(iqr, nan=1.0)
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    x_scaled = np.clip((x - median) / iqr, -10.0, 10.0).astype(np.float32)
    valid_mask = ~np.isnan(x_scaled).any(axis=1)

    train_end_idx_all = build_segment_end_indices(valid_mask, *train_range, SEQ_LEN, TARGET_HORIZON)
    val_end_idx_all = build_segment_end_indices(valid_mask, *val_range, SEQ_LEN, TARGET_HORIZON)
    test_end_idx_all = build_segment_end_indices(valid_mask, *test_range, SEQ_LEN, TARGET_HORIZON)
    train_end_idx = maybe_cap_windows(train_end_idx_all, args.max_train_windows, USE_ALL_WINDOWS)
    val_end_idx = maybe_cap_windows(val_end_idx_all, args.max_eval_windows, USE_ALL_WINDOWS)
    test_end_idx = maybe_cap_windows(test_end_idx_all, args.max_eval_windows, USE_ALL_WINDOWS)
    if min(len(train_end_idx), len(val_end_idx), len(test_end_idx)) == 0:
        raise ValueError("One or more leakage-safe splits ended up empty.")
    log.info(
        "Window usage | train=%d/%d val=%d/%d test=%d/%d",
        len(train_end_idx),
        len(train_end_idx_all),
        len(val_end_idx),
        len(val_end_idx_all),
        len(test_end_idx),
        len(test_end_idx_all),
    )

    x_train, y_train = build_windows(x_scaled, y, train_end_idx, SEQ_LEN)
    x_val, y_val = build_windows(x_scaled, y, val_end_idx, SEQ_LEN)
    x_test, y_test = build_windows(x_scaled, y, test_end_idx, SEQ_LEN)
    log.info("Window counts | train=%d val=%d test=%d", len(x_train), len(x_val), len(x_test))

    class_weights = make_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = GoldMambaLiteClassifier(n_features=MODEL_FEATURE_COUNT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, max(args.batch_size, DEFAULT_BATCH_SIZE), shuffle=False)
    test_loader = make_loader(x_test, y_test, max(args.batch_size, DEFAULT_BATCH_SIZE), shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_logits, val_labels = evaluate_model(model, val_loader, device)
        val_loss = float(
            criterion(
                torch.tensor(val_logits, dtype=torch.float32, device=device),
                torch.tensor(val_labels, dtype=torch.long, device=device),
            ).item()
        )
        log.info(
            "Epoch %02d | train_loss=%.4f val_loss=%.4f wait=%d/%d",
            epoch,
            float(np.mean(train_losses)),
            val_loss,
            wait,
            args.patience,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    val_logits, val_labels = evaluate_model(model, val_loader, device)
    val_probs = softmax(val_logits)
    primary_confidence = choose_confidence_threshold(val_probs, val_labels)
    summarize_gate("validation", val_probs, val_labels, primary_confidence)

    test_logits, test_labels = evaluate_model(model, test_loader, device)
    test_probs = softmax(test_logits)
    summarize_gate("holdout", test_probs, test_labels, primary_confidence)
    completed_at = datetime.now()
    model_stamp = format_model_stamp(completed_at)
    model_dir = MODELS_DIR / model_stamp
    model_dir.mkdir(parents=True, exist_ok=False)
    model_diagnostics_dir = model_dir / "diagnostics"
    model_config_dir = model_dir / "config"
    model_test_dir = model_dir / "test"
    model_config_dir.mkdir(parents=True, exist_ok=True)
    model_test_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to("cpu")
    dummy = torch.randn(1, SEQ_LEN, MODEL_FEATURE_COUNT)
    torch.onnx.export(
        model,
        dummy,
        str(active_output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    if output_path != active_output_path:
        shutil.copy2(active_output_path, output_path)

    model_config_text = (
        build_mql_config(
            median=median,
            iqr=iqr,
            primary_confidence=primary_confidence,
            use_atr_risk=use_atr_risk,
            use_fixed_time_bars=use_fixed_time_bars,
        )
    )
        + "\n"
    )
    config_path.write_text(model_config_text, encoding="utf-8")
    write_diagnostics(
        diagnostics_dir=model_diagnostics_dir,
        df_gold=df_gold,
        y_full=y,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        val_probs=val_probs,
        test_probs=test_probs,
        primary_confidence=primary_confidence,
        available_window_counts={
            "train": len(train_end_idx_all),
            "validation": len(val_end_idx_all),
            "holdout": len(test_end_idx_all),
        },
        used_window_counts={
            "train": len(train_end_idx),
            "validation": len(val_end_idx),
            "holdout": len(test_end_idx),
        },
        use_atr_risk=use_atr_risk,
        use_fixed_time_bars=use_fixed_time_bars,
        model_config_text=model_config_text,
    )
    sync_directory_contents(model_diagnostics_dir, ACTIVE_DIAGNOSTICS_DIR)
    shutil.copy2(active_output_path, model_dir / active_output_path.name)
    shutil.copy2(config_path, model_dir / config_path.name)
    shutil.copy2(SHARED_CONFIG_PATH, model_dir / SHARED_CONFIG_PATH.name)
    ensure_default_test_config(model_config_dir)
    if not args.skip_live_compile:
        metaeditor_path = resolve_metaeditor_path(args.metaeditor_path)
        compile_log_path = compile_live_expert(metaeditor_path)
        warnings_match = re.search(
            r"Result:\s+(\d+)\s+errors?,\s+(\d+)\s+warnings?",
            read_text_best_effort(compile_log_path),
        )
        warnings = int(warnings_match.group(2)) if warnings_match else 0
        log.info(
            "Compiled live EA successfully with %d warnings. Log: %s",
            warnings,
            compile_log_path,
        )
        shutil.copy2(compile_log_path, model_dir / compile_log_path.name)
        if LIVE_EX5_PATH.exists():
            shutil.copy2(LIVE_EX5_PATH, model_dir / LIVE_EX5_PATH.name)
        log.info("Saved live compile log to %s", compile_log_path)
    log.info("Saved ONNX to %s", active_output_path)
    if output_path != active_output_path:
        log.info("Copied ONNX to %s", output_path)
    log.info("Saved config to %s", config_path)
    log.info("Saved diagnostics to %s", model_diagnostics_dir)
    log.info("Archived model artifacts to %s", model_dir)
    log.info("Total runtime: %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
