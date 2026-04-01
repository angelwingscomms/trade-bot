from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nn")

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from shared_mamba import SharedMambaClassifier

EPS = 1e-10
SEQ_LEN = 120
TARGET_HORIZON = 30
SYMBOL_ORDER = ("XAUUSD", "$USDX", "USDJPY")
DEFAULT_DATA_FILE = "gold_market_ticks.csv"
DEFAULT_OUTPUT_FILE = "gold_mamba.onnx"
GOLD_FEATURE_COLUMNS = (
    "ret1",
    "high_rel_prev",
    "low_rel_prev",
    "spread_rel",
    "duration_s",
    "close_in_range",
    "atr14_rel",
    "rv4",
    "rv16",
    "ret8",
    "hour_sin",
    "hour_cos",
)
AUX_FEATURE_COLUMNS = ("ret1", "close_in_range", "atr14_rel", "ret8")
N_FEATURES = len(GOLD_FEATURE_COLUMNS) + 2 * len(AUX_FEATURE_COLUMNS)
PRETRAIN_TARGET_INDICES = (0, 1, 2, 6, 12, 16)
META_FEATURE_INPUT_INDICES = (3, 6, 7, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the laptop-safe GOLD Mamba model.")
    parser.add_argument(
        "-i",
        "--pretrain-init",
        action="store_true",
        help="Enable the cheap self-supervised warmup before the main supervised training.",
    )
    parser.add_argument("--tick-density", type=int, default=540, help="Ticks per primary XAUUSD bar.")
    parser.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE, help="Combined CSV with symbol,time_msc,bid,ask.")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="ONNX output file.")
    parser.add_argument("--epochs", type=int, default=28, help="Max fine-tuning epochs.")
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=3,
        help="Cheap self-supervised warmup epochs. Only used when -i is enabled.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Fine-tuning batch size.")
    parser.add_argument("--max-train-windows", type=int, default=3072, help="Training window cap for slow laptops.")
    parser.add_argument("--max-eval-windows", type=int, default=1024, help="Validation/calibration/test window cap.")
    parser.add_argument("--device", type=str, default="", help="Optional torch device override.")
    return parser.parse_args()


def build_aligned_bars(csv_path: str, symbols: tuple[str, ...], tick_density: int) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    log.info(f"Loading combined tick CSV: {csv_path}...")
    df_all = pd.read_csv(csv_path)
    log.info(f"CSV loaded: {len(df_all)} rows, columns={list(df_all.columns)}, elapsed={time.time()-t0:.2f}s")
    df_all["symbol"] = df_all["symbol"].astype(str).str.upper()

    sym_gold = symbols[0]
    df_gold = df_all[df_all["symbol"] == sym_gold].sort_values("time_msc").reset_index(drop=True)
    if df_gold.empty:
        raise ValueError(f"No ticks found for {sym_gold}")
    log.info(f"Gold ticks: {len(df_gold)}, time_msc range=[{df_gold['time_msc'].iloc[0]}, {df_gold['time_msc'].iloc[-1]}]")

    df_gold["bar_id"] = np.arange(len(df_gold)) // tick_density
    n_bars_gold = df_gold["bar_id"].nunique()
    bar_ends = df_gold.groupby("bar_id")["time_msc"].last().values
    log.info(f"Gold bar_id assigned: {n_bars_gold} bars, tick_density={tick_density}")

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        t_sym = time.time()
        df_sym = df_all[df_all["symbol"] == sym].sort_values("time_msc").reset_index(drop=True)
        if df_sym.empty:
            raise ValueError(f"No ticks found for {sym}")
        log.info(f"{sym}: {len(df_sym)} ticks loaded, binning to gold bars...")

        if sym == sym_gold:
            df_sym_binned = df_gold
        else:
            bar_ids = np.searchsorted(bar_ends, df_sym["time_msc"].values, side="left")
            valid = bar_ids < len(bar_ends)
            df_sym_binned = df_sym[valid].copy()
            df_sym_binned["bar_id"] = bar_ids[valid]
            log.info(f"{sym}: {valid.sum()}/{len(valid)} ticks within bar range, {df_sym_binned['bar_id'].nunique()} unique bars")

        has_ask = "ask" in df_sym_binned.columns
        log.info(f"{sym}: has_ask={has_ask}")
        agg = {"bid": ["first", "max", "min", "last"], "time_msc": "first"}
        if has_ask:
            df_sym_binned["spread"] = df_sym_binned["ask"] - df_sym_binned["bid"]
            agg_spread = df_sym_binned.groupby("bar_id")["spread"].last()
            agg["ask"] = ["max", "min"]

        df_bars = df_sym_binned.groupby("bar_id").agg(agg)
        if has_ask:
            df_bars.columns = ["open", "high", "low", "close", "time_open", "ask_high", "ask_low"]
            df_bars["spread"] = agg_spread
        else:
            df_bars.columns = ["open", "high", "low", "close", "time_open"]
            df_bars["spread"] = 0.0
            df_bars["ask_high"] = df_bars["high"]
            df_bars["ask_low"] = df_bars["low"]

        df_bars = df_bars.reindex(np.arange(len(bar_ends)))
        n_nan_close = df_bars["close"].isna().sum()
        df_bars["close"] = df_bars["close"].ffill().bfill()
        df_bars["open"] = df_bars["open"].fillna(df_bars["close"])
        df_bars["high"] = df_bars["high"].fillna(df_bars["close"])
        df_bars["low"] = df_bars["low"].fillna(df_bars["close"])
        df_bars["spread"] = df_bars["spread"].ffill().bfill().fillna(0.0)
        df_bars["ask_high"] = df_bars["ask_high"].fillna(df_bars["high"] + df_bars["spread"])
        df_bars["ask_low"] = df_bars["ask_low"].fillna(df_bars["low"] + df_bars["spread"])
        log.info(f"{sym}: NaN close filled={n_nan_close}, reindexed to {len(bar_ends)} bars")

        if sym != sym_gold:
            gold_time_open = df_gold.groupby("bar_id")["time_msc"].first()
            df_bars["time_open"] = df_bars["time_open"].fillna(gold_time_open).ffill().bfill()

        bars_by_symbol[sym] = df_bars.reset_index(drop=True)
        log.info(f"{sym}: built {len(bars_by_symbol[sym])} aligned bars, elapsed={time.time()-t_sym:.2f}s")

    log.info(f"build_aligned_bars total elapsed={time.time()-t0:.2f}s")
    return bars_by_symbol


def rolling_std(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, min_periods=window).std(ddof=0)


def compute_features(df: pd.DataFrame, symbol_idx: int = 0) -> np.ndarray:
    t0 = time.time()
    sym_name = SYMBOL_ORDER[symbol_idx] if symbol_idx < len(SYMBOL_ORDER) else f"idx={symbol_idx}"
    log.info(f"compute_features({sym_name}): starting on {len(df)} bars...")
    df = df.copy()
    df["dt"] = pd.to_datetime(df["time_open"], unit="ms", utc=True)

    c = df["close"].astype(float)
    prev_c = c.shift(1)
    ret1 = np.log(c / (prev_c + EPS))

    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = ret1
    feat["close_in_range"] = (c - df["low"]) / (df["high"] - df["low"] + 1e-8)
    feat["atr14_rel"] = ta.atr(df["high"], df["low"], c, length=14) / (c + EPS)
    feat["ret8"] = np.log(c / (c.shift(8) + EPS))

    if symbol_idx == 0:
        feat["high_rel_prev"] = np.log(df["high"] / (prev_c + EPS))
        feat["low_rel_prev"] = np.log(df["low"] / (prev_c + EPS))
        feat["spread_rel"] = df["spread"] / (c + EPS)
        feat["duration_s"] = df["dt"].diff().dt.total_seconds().fillna(0.0)
        feat["rv4"] = rolling_std(ret1, 4)
        feat["rv16"] = rolling_std(ret1, 16)
        hours = df["dt"].dt.hour + (df["dt"].dt.minute / 60.0)
        feat["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        feat["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        feature_columns = GOLD_FEATURE_COLUMNS
    else:
        feature_columns = AUX_FEATURE_COLUMNS

    result = feat.loc[:, feature_columns].values.astype(np.float32)
    nan_count = int(np.isnan(result).sum())
    log.info(f"compute_features({sym_name}): done, shape={result.shape}, NaNs={nan_count}, elapsed={time.time()-t0:.2f}s")
    return result


def get_triple_barrier_labels(
    df_gold: pd.DataFrame,
    tp_mult: float = 9.0,
    sl_mult: float = 5.4,
    horizon: int = TARGET_HORIZON,
) -> np.ndarray:
    t0 = time.time()
    n = len(df_gold)
    log.info(f"get_triple_barrier_labels: starting on {n} bars, horizon={horizon}, tp_mult={tp_mult}, sl_mult={sl_mult}...")
    c = df_gold["close"].values
    hi = df_gold["high"].values
    lo = df_gold["low"].values
    spr = df_gold["spread"].values
    ask_hi = df_gold["ask_high"].values if "ask_high" in df_gold.columns else hi + spr
    ask_lo = df_gold["ask_low"].values if "ask_low" in df_gold.columns else lo + spr
    atr = ta.atr(df_gold["high"], df_gold["low"], df_gold["close"], length=14).values
    labels = np.zeros(n, dtype=np.int64)

    report_interval = max(1, (n - horizon) // 20)
    n_long_tp = 0
    n_short_tp = 0
    n_skipped_vol = 0

    for i in range(n - horizon):
        if (i + 1) % report_interval == 0 or i == n - horizon - 1:
            elapsed = time.time() - t0
            pct = (i + 1) / (n - horizon) * 100
            log.info(f"  triple_barrier {i+1}/{n-horizon} ({pct:.1f}%) elapsed={elapsed:.1f}s long_tp={n_long_tp} short_tp={n_short_tp} skipped_vol={n_skipped_vol}")

        vol = atr[i]
        if not np.isfinite(vol) or vol <= 0.0:
            n_skipped_vol += 1
            continue

        long_entry = c[i] + spr[i]
        short_entry = c[i]

        long_tp = long_entry + tp_mult * vol
        long_sl = long_entry - sl_mult * vol
        short_tp = short_entry - tp_mult * vol
        short_sl = short_entry + sl_mult * vol

        long_result = 0
        short_result = 0

        for j in range(i + 1, i + horizon + 1):
            if long_result == 0:
                hit_tp = hi[j] >= long_tp
                hit_sl = lo[j] <= long_sl
                if hit_tp and not hit_sl:
                    long_result = 1
                elif hit_sl and not hit_tp:
                    long_result = -1
                elif hit_tp and hit_sl:
                    long_result = -1

            if short_result == 0:
                hit_tp = ask_lo[j] <= short_tp
                hit_sl = ask_hi[j] >= short_sl
                if hit_tp and not hit_sl:
                    short_result = 1
                elif hit_sl and not hit_tp:
                    short_result = -1
                elif hit_tp and hit_sl:
                    short_result = -1

            if long_result != 0 and short_result != 0:
                break

        if long_result == 1 and short_result != 1:
            labels[i] = 1
            n_long_tp += 1
        elif short_result == 1 and long_result != 1:
            labels[i] = 2
            n_short_tp += 1

    class_counts = {int(v): int((labels == v).sum()) for v in np.unique(labels)}
    log.info(f"get_triple_barrier_labels: done, elapsed={time.time()-t0:.2f}s, class_counts={class_counts}, skipped_vol={n_skipped_vol}")
    return labels


def choose_evenly_spaced(indices: np.ndarray, max_count: int) -> np.ndarray:
    if len(indices) <= max_count:
        return indices.astype(np.int64)
    positions = np.linspace(0, len(indices) - 1, max_count)
    return indices[np.unique(np.round(positions).astype(np.int64))]


def build_segment_end_indices(
    valid_mask: np.ndarray,
    start_bar: int,
    end_bar: int,
    seq_len: int,
    horizon: int,
) -> np.ndarray:
    t0 = time.time()
    first_end = start_bar + seq_len - 1
    last_end = end_bar - horizon - 1
    log.debug(f"build_segment_end_indices: range=[{start_bar},{end_bar}), seq_len={seq_len}, horizon={horizon}, search=[{first_end},{last_end}]")
    if last_end < first_end:
        log.debug(f"build_segment_end_indices: empty (last_end < first_end)")
        return np.empty(0, dtype=np.int64)

    ends: list[int] = []
    for end_idx in range(first_end, last_end + 1):
        start_idx = end_idx - seq_len + 1
        if valid_mask[start_idx : end_idx + 1].all():
            ends.append(end_idx)
    result = np.asarray(ends, dtype=np.int64)
    log.debug(f"build_segment_end_indices: found {len(result)} valid windows, elapsed={time.time()-t0:.3f}s")
    return result


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    end_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    n = len(end_indices)
    log.debug(f"build_windows: building {n} windows, seq_len={seq_len}...")
    xs = np.empty((n, seq_len, features.shape[1]), dtype=np.float32)
    ys = np.empty(n, dtype=np.int64)
    for i, end_idx in enumerate(end_indices):
        start_idx = end_idx - seq_len + 1
        xs[i] = features[start_idx : end_idx + 1]
        ys[i] = labels[end_idx]
    log.debug(f"build_windows: done, xs_shape={xs.shape}, ys_shape={ys.shape}, elapsed={time.time()-t0:.3f}s")
    return xs, ys


class MaskedNextBarHead(nn.Module):
    def __init__(self, backbone: SharedMambaClassifier, target_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.d_model, backbone.d_model),
            nn.SiLU(),
            nn.Linear(backbone.d_model, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.encode_last(x))


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_eval), torch.from_numpy(y_eval)),
        batch_size=max(batch_size, 64),
        shuffle=False,
    )
    return train_loader, eval_loader


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    n_batches = len(loader)
    log.debug(f"evaluate_model: starting, {n_batches} batches...")
    with torch.no_grad():
        for bi, (xb, yb) in enumerate(loader):
            logits = model(xb.to(device)).cpu().numpy()
            logits_list.append(logits)
            labels_list.append(yb.numpy())
            if (bi + 1) % max(1, n_batches // 5) == 0 or bi == n_batches - 1:
                log.debug(f"  evaluate_model: batch {bi+1}/{n_batches}")
    result_logits = np.concatenate(logits_list, axis=0)
    result_labels = np.concatenate(labels_list, axis=0)
    log.debug(f"evaluate_model: done, {len(result_labels)} samples, elapsed={time.time()-t0:.2f}s")
    return result_logits, result_labels


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    t0 = time.time()
    log.info(f"fit_temperature: starting on {len(logits)} samples...")
    if len(logits) == 0:
        log.warning("fit_temperature: empty logits, returning 1.0")
        return 1.0

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp_min(1e-3)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    raw_t = float(torch.exp(log_temperature).item())
    final_t = float(torch.exp(log_temperature).clamp(0.5, 5.0).item())
    log.info(f"fit_temperature: raw_T={raw_t:.6f}, clamped_T={final_t:.6f}, elapsed={time.time()-t0:.3f}s")
    return final_t


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    log.debug(f"apply_temperature: T={temperature:.4f}, logits shape={logits.shape}")
    scaled = logits / max(temperature, 1e-3)
    scaled -= scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def build_meta_features(probs: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
    max_prob = probs.max(axis=1)
    second_prob = np.partition(probs, -2, axis=1)[:, -2]
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    last_step = x_seq[:, -1, :]
    extras = last_step[:, META_FEATURE_INPUT_INDICES]
    return np.column_stack([probs, max_prob, max_prob - second_prob, entropy, extras]).astype(np.float32)


def train_meta_classifier(probs: np.ndarray, labels: np.ndarray, x_seq: np.ndarray) -> LogisticRegression | None:
    t0 = time.time()
    preds = probs.argmax(axis=1)
    candidate_mask = preds > 0
    n_candidates = int(candidate_mask.sum())
    log.info(f"train_meta_classifier: {len(probs)} samples, {n_candidates} candidates (pred>0)")
    if n_candidates < 40:
        log.info(f"train_meta_classifier: skipped (only {n_candidates} candidates, need >=40), elapsed={time.time()-t0:.3f}s")
        return None

    features = build_meta_features(probs[candidate_mask], x_seq[candidate_mask])
    targets = (preds[candidate_mask] == labels[candidate_mask]).astype(np.int64)
    unique_targets = np.unique(targets)
    n_correct = int(targets.sum())
    n_incorrect = len(targets) - n_correct
    log.info(f"train_meta_classifier: meta targets 0={n_incorrect}, 1={n_correct}, unique={len(unique_targets)}")
    if len(unique_targets) < 2:
        log.info(f"train_meta_classifier: skipped (single class in targets), elapsed={time.time()-t0:.3f}s")
        return None

    model = LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs")
    model.fit(features, targets)
    train_acc = float(model.score(features, targets))
    log.info(f"train_meta_classifier: trained, train_acc={train_acc:.4f}, elapsed={time.time()-t0:.3f}s")
    return model


def choose_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    x_seq: np.ndarray,
    meta_model: LogisticRegression | None,
) -> tuple[float, float]:
    t0 = time.time()
    log.info(f"choose_thresholds: searching grid on {len(probs)} samples...")
    preds = probs.argmax(axis=1)
    candidate_mask = preds > 0
    base_conf = probs.max(axis=1)
    meta_probs = np.ones(len(probs), dtype=np.float32)
    meta_grid = [0.0]

    if meta_model is not None and candidate_mask.any():
        candidate_features = build_meta_features(probs[candidate_mask], x_seq[candidate_mask])
        meta_probs[candidate_mask] = meta_model.predict_proba(candidate_features)[:, 1]
        meta_grid = list(np.linspace(0.45, 0.90, 19))
        log.info(f"  choose_thresholds: meta model applied, meta_grid size={len(meta_grid)}")

    min_selected = max(12, int(0.03 * len(labels)))
    best_choice: tuple[float, float] | None = None
    best_precision = -1.0
    best_coverage = -1.0
    n_checked = 0

    for primary_thr in np.linspace(0.40, 0.85, 19):
        for meta_thr in meta_grid:
            selected = candidate_mask & (base_conf >= primary_thr) & (meta_probs >= meta_thr)
            if selected.sum() < min_selected:
                continue
            n_checked += 1

            precision = float((preds[selected] == labels[selected]).mean())
            coverage = float(selected.mean())
            if precision > best_precision + 1e-12 or (
                abs(precision - best_precision) <= 1e-12 and coverage > best_coverage
            ):
                best_choice = (float(primary_thr), float(meta_thr))
                best_precision = precision
                best_coverage = coverage

    if best_choice is None:
        log.info(f"choose_thresholds: no grid point passed, using defaults, elapsed={time.time()-t0:.3f}s")
        return 0.60, 0.55 if meta_model is not None else 0.0
    log.info(f"choose_thresholds: best={best_choice}, precision={best_precision:.4f}, coverage={best_coverage:.4f}, checked={n_checked}, elapsed={time.time()-t0:.3f}s")
    return best_choice


def summarize_gate(
    name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    x_seq: np.ndarray,
    primary_thr: float,
    meta_model: LogisticRegression | None,
    meta_thr: float,
) -> None:
    preds = probs.argmax(axis=1)
    candidate_mask = preds > 0
    selected = candidate_mask & (probs.max(axis=1) >= primary_thr)

    if meta_model is not None and candidate_mask.any():
        meta_features = build_meta_features(probs[candidate_mask], x_seq[candidate_mask])
        meta_probs = meta_model.predict_proba(meta_features)[:, 1]
        selected_candidates = meta_probs >= meta_thr
        selected = candidate_mask.copy()
        selected[candidate_mask] = selected[candidate_mask] & selected_candidates & (
            probs[candidate_mask].max(axis=1) >= primary_thr
        )

    coverage = float(selected.mean())
    if selected.any():
        precision = float((preds[selected] == labels[selected]).mean())
        log.info(f"{name}: selected precision={precision:.4f} coverage={coverage:.4f} trades={int(selected.sum())}")
    else:
        log.warning(f"{name}: no trades passed the abstention gate.")


def format_float_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(v):.8f}f" for v in values)


def build_export_block(
    tick_density: int,
    medians: np.ndarray,
    iqrs: np.ndarray,
    temperature: float,
    primary_thr: float,
    meta_thr: float,
    meta_model: LogisticRegression | None,
) -> str:
    meta_weights = np.zeros(3 + 1 + 1 + 1 + len(META_FEATURE_INPUT_INDICES), dtype=np.float32)
    meta_bias = 0.0
    if meta_model is not None:
        meta_weights = meta_model.coef_[0].astype(np.float32)
        meta_bias = float(meta_model.intercept_[0])

    return "\n".join(
        [
            "--- PASTE THESE INTO gold/live.mq5 ---",
            f"input int    TICK_DENSITY        = {tick_density};",
            f"input double TEMPERATURE         = {temperature:.8f};",
            f"input double PRIMARY_CONFIDENCE  = {primary_thr:.8f};",
            f"input double META_THRESHOLD      = {meta_thr:.8f};",
            f"float medians[{N_FEATURES}] = {{{format_float_array(medians)}}};",
            f"float iqrs[{N_FEATURES}]    = {{{format_float_array(iqrs)}}};",
            f"float meta_weights[{len(meta_weights)}] = {{{format_float_array(meta_weights)}}};",
            f"float meta_bias = {meta_bias:.8f}f;",
        ]
    )


def main() -> None:
    main_t0 = time.time()
    try:
        _main_inner()
    except Exception:
        log.exception(f"FATAL after {time.time()-main_t0:.2f}s")
        raise
    log.info(f"=== TOTAL RUNTIME: {time.time()-main_t0:.2f}s ===")


def _main_inner() -> None:
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Using device: {device}")
    log.info(f"Args: tick_density={args.tick_density}, data_file={args.data_file}, output_file={args.output_file}")
    log.info(
        f"Args: epochs={args.epochs}, pretrain_init={args.pretrain_init}, "
        f"pretrain_epochs={args.pretrain_epochs}, batch_size={args.batch_size}"
    )
    log.info(f"Args: max_train_windows={args.max_train_windows}, max_eval_windows={args.max_eval_windows}")

    t_step = time.time()
    bars_by_symbol = build_aligned_bars(args.data_file, SYMBOL_ORDER, args.tick_density)
    df_gold = bars_by_symbol[SYMBOL_ORDER[0]]
    df_usdx = bars_by_symbol[SYMBOL_ORDER[1]]
    df_usdjpy = bars_by_symbol[SYMBOL_ORDER[2]]

    n_bars = min(len(df_gold), len(df_usdx), len(df_usdjpy))
    df_gold = df_gold.iloc[:n_bars].reset_index(drop=True)
    df_usdx = df_usdx.iloc[:n_bars].reset_index(drop=True)
    df_usdjpy = df_usdjpy.iloc[:n_bars].reset_index(drop=True)
    log.info(f"Aligned bar count: {n_bars}")

    log.info("Computing features for all symbols...")
    feat_gold = compute_features(df_gold, symbol_idx=0)
    feat_usdx = compute_features(df_usdx, symbol_idx=1)
    feat_usdjpy = compute_features(df_usdjpy, symbol_idx=2)
    X = np.concatenate([feat_gold, feat_usdx, feat_usdjpy], axis=1)
    assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X.shape[1]}"
    log.info(f"Feature matrix X shape={X.shape}, NaN_count={int(np.isnan(X).sum())}")

    log.info("Computing triple-barrier labels...")
    y = get_triple_barrier_labels(df_gold)

    warmup = 20
    X = X[warmup:]
    y = y[warmup:]
    n_rows = len(X)
    embargo = max(SEQ_LEN, TARGET_HORIZON)

    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.82)
    calib_end = int(n_rows * 0.91)

    train_range = (0, train_end)
    val_range = (train_end + embargo, val_end)
    calib_range = (val_end + embargo, calib_end)
    test_range = (calib_end + embargo, n_rows)
    log.info(f"Splits: n_rows={n_rows}, embargo={embargo}")
    log.info(f"  train=[{train_range[0]},{train_range[1]}) ({train_end} rows)")
    log.info(f"  val=[{val_range[0]},{val_range[1]}) ({val_end-train_end-embargo} rows)")
    log.info(f"  calib=[{calib_range[0]},{calib_range[1]}) ({calib_end-val_end-embargo} rows)")
    log.info(f"  test=[{test_range[0]},{test_range[1]}) ({n_rows-calib_end-embargo} rows)")
    if test_range[0] >= test_range[1]:
        raise ValueError("Dataset is too small for leakage-safe train/val/calibration/test splits.")

    log.info("Normalizing features (median/IQR)...")
    median = np.nanmedian(X[: train_range[1]], axis=0)
    median = np.nan_to_num(median, nan=0.0)
    iqr = np.nanpercentile(X[: train_range[1]], 75, axis=0) - np.nanpercentile(X[: train_range[1]], 25, axis=0)
    iqr = np.nan_to_num(iqr, nan=1.0)
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    X_s = np.clip((X - median) / iqr, -10.0, 10.0).astype(np.float32)
    valid_mask = ~np.isnan(X_s).any(axis=1)
    n_invalid = int((~valid_mask).sum())
    log.info(f"Normalized: X_s shape={X_s.shape}, NaN rows={n_invalid}/{len(X_s)}")

    log.info("Building segment end indices for all splits...")
    train_end_idx = choose_evenly_spaced(
        build_segment_end_indices(valid_mask, *train_range, SEQ_LEN, TARGET_HORIZON),
        args.max_train_windows,
    )
    log.info(f"  train_end_idx: {len(train_end_idx)} windows")
    val_end_idx = choose_evenly_spaced(
        build_segment_end_indices(valid_mask, *val_range, SEQ_LEN, TARGET_HORIZON),
        args.max_eval_windows,
    )
    log.info(f"  val_end_idx: {len(val_end_idx)} windows")
    calib_end_idx = choose_evenly_spaced(
        build_segment_end_indices(valid_mask, *calib_range, SEQ_LEN, TARGET_HORIZON),
        args.max_eval_windows,
    )
    log.info(f"  calib_end_idx: {len(calib_end_idx)} windows")
    test_end_idx = choose_evenly_spaced(
        build_segment_end_indices(valid_mask, *test_range, SEQ_LEN, TARGET_HORIZON),
        args.max_eval_windows,
    )
    log.info(f"  test_end_idx: {len(test_end_idx)} windows")

    if min(len(train_end_idx), len(val_end_idx), len(calib_end_idx), len(test_end_idx)) == 0:
        raise ValueError("One or more leakage-safe splits ended up empty. Try a smaller tick density or more data.")

    log.info("Building windows...")
    x_train, y_train = build_windows(X_s, y, train_end_idx, SEQ_LEN)
    x_val, y_val = build_windows(X_s, y, val_end_idx, SEQ_LEN)
    x_calib, y_calib = build_windows(X_s, y, calib_end_idx, SEQ_LEN)
    x_test, y_test = build_windows(X_s, y, test_end_idx, SEQ_LEN)
    log.info(
        f"Window counts | train={len(x_train)} val={len(x_val)} calib={len(x_calib)} test={len(x_test)} "
        f"(embargo={embargo})"
    )

    del X
    gc.collect()

    unique_classes = np.unique(y_train)
    class_weights_raw = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(unique_classes, class_weights_raw)}
    class_weights = torch.tensor([weight_dict.get(i, 1.0) for i in range(3)], dtype=torch.float32, device=device)
    log.info(f"Class weights: {[round(float(v), 4) for v in class_weights.cpu().numpy()]}")

    log.info("Building SharedMambaClassifier model...")
    model = SharedMambaClassifier(
        n_features=N_FEATURES,
        d_model=48,
        hidden=96,
        dropout=0.20,
        n_layers=2,
        use_sequence_norm=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {n_params:,}")

    if args.pretrain_init and args.pretrain_epochs > 0:
        t_pretrain = time.time()
        pretrain_count = min(len(x_train), args.max_train_windows)
        pretrain_inputs = x_train[:pretrain_count].copy()
        pretrain_targets = pretrain_inputs[:, -1, PRETRAIN_TARGET_INDICES].copy()
        pretrain_inputs[:, -1, :] = 0.0

        pretrain_model = MaskedNextBarHead(model, target_dim=len(PRETRAIN_TARGET_INDICES)).to(device)
        pretrain_loader = DataLoader(
            TensorDataset(torch.from_numpy(pretrain_inputs), torch.from_numpy(pretrain_targets.astype(np.float32))),
            batch_size=args.batch_size,
            shuffle=True,
        )
        n_pretrain_batches = len(pretrain_loader)
        pretrain_optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=6e-4, weight_decay=1e-4)
        pretrain_criterion = nn.SmoothL1Loss()

        log.info(f"=== SELF-SUPERVISED WARMUP START ===")
        log.info(f"  windows={pretrain_count}, epochs={args.pretrain_epochs}, batches/epoch={n_pretrain_batches}, batch_size={args.batch_size}")
        log.info(f"  target_indices={PRETRAIN_TARGET_INDICES}, target_dim={len(PRETRAIN_TARGET_INDICES)}")
        log.info(f"  optimizer=AdamW(lr=6e-4, wd=1e-4), criterion=SmoothL1Loss")
        for epoch in range(args.pretrain_epochs):
            epoch_t0 = time.time()
            pretrain_model.train()
            losses = []
            for bi, (xb, yb) in enumerate(pretrain_loader):
                batch_t0 = time.time()
                xb = xb.to(device)
                yb = yb.to(device)
                pred = pretrain_model(xb)
                loss = pretrain_criterion(pred, yb)
                pretrain_optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1.0)
                pretrain_optimizer.step()
                batch_loss = float(loss.item())
                losses.append(batch_loss)
                batch_dt = time.time() - batch_t0

                if (bi + 1) % max(1, n_pretrain_batches // 10) == 0 or bi == 0 or bi == n_pretrain_batches - 1:
                    running_mean = np.mean(losses)
                    log.info(
                        f"  pretrain epoch {epoch+1:02d} | batch {bi+1:04d}/{n_pretrain_batches} | "
                        f"loss={batch_loss:.6f} | running_mean={running_mean:.6f} | "
                        f"grad_norm={float(grad_norm):.4f} | pred_mean={float(pred.mean()):.6f} | "
                        f"pred_std={float(pred.std()):.6f} | batch_dt={batch_dt*1000:.1f}ms"
                    )

                # check for NaN/Inf
                if not np.isfinite(batch_loss):
                    log.error(f"  !! NaN/Inf loss detected at pretrain epoch {epoch+1} batch {bi+1}: loss={batch_loss}")
                    log.error(f"     pred stats: mean={float(pred.mean())}, std={float(pred.std())}, min={float(pred.min())}, max={float(pred.max())}")
                    log.error(f"     yb stats: mean={float(yb.mean())}, std={float(yb.std())}, min={float(yb.min())}, max={float(yb.max())}")

            epoch_dt = time.time() - epoch_t0
            log.info(f"  pretrain epoch {epoch+1:02d} COMPLETE | mean_loss={np.mean(losses):.6f} | min_loss={np.min(losses):.6f} | max_loss={np.max(losses):.6f} | epoch_dt={epoch_dt:.2f}s")

        log.info(f"=== SELF-SUPERVISED WARMUP DONE | total_dt={time.time()-t_pretrain:.2f}s ===")

        del pretrain_inputs, pretrain_targets, pretrain_model, pretrain_loader
        gc.collect()
        log.info(f"Pretrain cleanup done, gc collected")
    else:
        log.info("Pretraining disabled, skipping self-supervised warmup")

    log.info("Building train/val data loaders...")
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val, args.batch_size)
    n_train_batches = len(train_loader)
    log.info(f"  train_loader: {n_train_batches} batches, val_loader: {len(val_loader)} batches")
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)

    best_val_loss = float("inf")
    best_state = None
    patience = 6
    wait = 0

    log.info(f"=== SUPERVISED FINE-TUNING START | epochs={args.epochs}, patience={patience} ===")
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        model.train()
        train_losses: list[float] = []
        for bi, (xb, yb) in enumerate(train_loader):
            batch_t0 = time.time()
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_loss = float(loss.item())
            train_losses.append(batch_loss)
            batch_dt = time.time() - batch_t0

            if (bi + 1) % max(1, n_train_batches // 5) == 0 or bi == 0 or bi == n_train_batches - 1:
                running_mean = np.mean(train_losses)
                log.info(
                    f"  epoch {epoch:02d} | batch {bi+1:04d}/{n_train_batches} | "
                    f"loss={batch_loss:.6f} | running_mean={running_mean:.6f} | "
                    f"grad_norm={float(grad_norm):.4f} | batch_dt={batch_dt*1000:.1f}ms"
                )

            if not np.isfinite(batch_loss):
                log.error(f"  !! NaN/Inf loss at epoch {epoch} batch {bi+1}: {batch_loss}")
                log.error(f"     logits stats: mean={float(logits.mean())}, std={float(logits.std())}, min={float(logits.min())}, max={float(logits.max())}")

        log.info(f"  epoch {epoch:02d} evaluating on val set...")
        val_logits, val_labels = evaluate_model(model, val_loader, device)
        val_loss = float(F.cross_entropy(torch.tensor(val_logits), torch.tensor(val_labels), weight=class_weights.cpu()).item())
        train_loss = float(np.mean(train_losses))
        epoch_dt = time.time() - epoch_t0
        log.info(f"  Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | epoch_dt={epoch_dt:.2f}s | wait={wait}/{patience}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            log.info(f"  --> New best val_loss={best_val_loss:.4f}, checkpoint saved")
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stopping at epoch {epoch} (val_loss did not improve for {patience} epochs)")
                break

    log.info(f"=== SUPERVISED FINE-TUNING DONE ===")

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    log.info(f"Loading best checkpoint (val_loss={best_val_loss:.4f})...")
    model.load_state_dict(best_state)
    model.to(device)

    calib_mid = max(1, len(x_calib) // 2)
    x_temp, y_temp = x_calib[:calib_mid], y_calib[:calib_mid]
    x_meta, y_meta = x_calib[calib_mid:], y_calib[calib_mid:]
    if len(x_meta) == 0:
        x_meta, y_meta = x_temp, y_temp
    log.info(f"Calibration split: temp={len(x_temp)} samples, meta={len(x_meta)} samples")

    log.info("Building calibration/test loaders...")
    temp_loader = DataLoader(TensorDataset(torch.from_numpy(x_temp), torch.from_numpy(y_temp)), batch_size=128, shuffle=False)
    meta_loader = DataLoader(TensorDataset(torch.from_numpy(x_meta), torch.from_numpy(y_meta)), batch_size=128, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=128, shuffle=False)

    log.info("Evaluating on temperature calibration set...")
    temp_logits, temp_labels = evaluate_model(model, temp_loader, device)
    temperature = fit_temperature(temp_logits, temp_labels)
    log.info(f"Temperature scaling fitted with T={temperature:.4f}")

    log.info("Evaluating on meta calibration set...")
    meta_logits, meta_labels = evaluate_model(model, meta_loader, device)
    log.info(f"Applying temperature scaling to meta logits...")
    meta_probs = apply_temperature(meta_logits, temperature)
    log.info(f"Meta probs shape={meta_probs.shape}, mean_conf={meta_probs.max(axis=1).mean():.4f}")
    meta_model = train_meta_classifier(meta_probs, meta_labels, x_meta)
    if meta_model is None:
        log.warning("Meta-label gate skipped because the calibration slice was too small or one-sided.")
    else:
        log.info("Meta-label gate trained.")

    log.info("Choosing thresholds...")
    primary_thr, meta_thr = choose_thresholds(meta_probs, meta_labels, x_meta, meta_model)
    log.info(f"Selected thresholds | primary={primary_thr:.3f} meta={meta_thr:.3f}")
    summarize_gate("calibration", meta_probs, meta_labels, x_meta, primary_thr, meta_model, meta_thr)

    log.info("Evaluating on holdout test set...")
    test_logits, test_labels = evaluate_model(model, test_loader, device)
    test_probs = apply_temperature(test_logits, temperature)
    summarize_gate("holdout", test_probs, test_labels, x_test, primary_thr, meta_model, meta_thr)

    log.info("Exporting to ONNX...")
    model.eval()
    model.to("cpu")
    dummy = torch.randn(1, SEQ_LEN, N_FEATURES)
    torch.onnx.export(
        model,
        dummy,
        args.output_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch"}},
        dynamo=False,
    )
    log.info(f"ONNX saved: {args.output_file}")

    export_block = build_export_block(args.tick_density, median, iqr, temperature, primary_thr, meta_thr, meta_model)
    export_path = Path("gold_export_values.txt")
    export_path.write_text(export_block + "\n", encoding="utf-8")
    log.info("Export block:")
    for line in export_block.split("\n"):
        log.info(f"  {line}")
    log.info(f"Export values also written to {export_path}")


if __name__ == "__main__":
    main()
