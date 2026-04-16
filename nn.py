"""Train, export, archive, and activate the MT5 model from `config.mqh`."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

from chronos_backend import (
    CHRONOS_BOLT_MODEL_IDS,
    DEFAULT_CHRONOS_BOLT_MODEL_ID,
    load_chronos_bolt_barrier_model,
)
from castor_lite import CastorClassifier
from mamba_lite import MambaLiteClassifier
from minirocket_classifier import (
    MiniRocketClassifier,
    MiniRocketMultiAttentionHead,
    fit_minirocket,
    transform_sequences,
    transform_sequence_tokens,
)
from mt5_runtime import resolve_mt5_runtime
from sequence_models import (
    FusionLSTMClassifier,
    GoldLegacyLSTMAttentionClassifier,
    GoldNewTemporalClassifier,
    LegacyLSTMAttentionClassifier,
    RecurrentSequenceClassifier,
    TCNClassifier,
    TemporalLSTMAttentionClassifier,
)
from tradebot.config_io import load_define_file, read_text_best_effort, render_define_value
from tradebot.project_config import (
    EXTRA_FEATURE_COLUMNS,
    GOLD_CONTEXT_FEATURE_COLUMNS,
    MINIMAL_FEATURE_COLUMNS,
    ResolvedProjectConfig,
    feature_macro_name as project_feature_macro_name,
    max_feature_lookback,
    resolve_active_project_config,
)
from tradebot.workspace import (
    ACTIVE_CONFIG_PATH,
    ACTIVE_DIAGNOSTICS_DIR,
    compile_live_expert,
    ensure_default_test_config,
    format_model_dir_name,
    model_diagnostics_dir,
    sanitize_model_name,
    set_live_model_reference,
    symbol_models_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nn")

EPS = 1e-10
DEFAULT_DATA_FILE = ""
DEFAULT_OUTPUT_FILE = "model.onnx"
SHARED_CONFIG_PATH = ACTIVE_CONFIG_PATH
DEFAULT_MINIROCKET_FEATURES = 10_080
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_MIN_SELECTED_TRADES = 12
DEFAULT_MIN_TRADE_PRECISION = 0.50
DEFAULT_MAMBA_LR = 6e-4
DEFAULT_MINIROCKET_LR = 1e-4
DEFAULT_SEQUENCE_LR = 1e-3
DEFAULT_MAMBA_WEIGHT_DECAY = 1e-4
DEFAULT_MINIROCKET_WEIGHT_DECAY = 0.0
DEFAULT_SEQUENCE_WEIGHT_DECAY = 1e-4
DEFAULT_MINIROCKET_ATTENTION_DIM = 128
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_ATTENTION_LAYERS = 2
DEFAULT_ATTENTION_DROPOUT = 0.1
DEFAULT_SEQUENCE_HIDDEN_SIZE = 64
DEFAULT_SEQUENCE_LAYERS = 2
DEFAULT_SEQUENCE_DROPOUT = 0.1
DEFAULT_TCN_LEVELS = 4
DEFAULT_TCN_KERNEL_SIZE = 3
DEFAULT_CONFIDENCE_SEARCH_MIN = 0.40
DEFAULT_CONFIDENCE_SEARCH_MAX = 0.99
DEFAULT_CONFIDENCE_SEARCH_STEPS = 60
DEFAULT_PRIMARY_TICK_DENSITY = 27
GOLD_CONTEXT_FEATURE_COLUMNS = (
    "usdx_ret1",
    "usdjpy_ret1",
)
GOLD_CONTEXT_TICK_COLUMNS = (
    "usdx_bid",
    "usdjpy_bid",
)
GOLD_WARNING_SYMBOL = "XAUUSD"
GOLD_PROFILE_SEQ_LEN = 120
GOLD_PROFILE_TICK_DENSITY = 27
GOLD_LEGACY_OLD_TICKS = 2_160_000
GOLD_LEGACY_OLD_TICK_DENSITY = 144
GOLD_LEGACY_OLD_RAW_BARS = 15_000
GOLD_LEGACY_OLD_CANDLES = 72000
GOLD_LEGACY_OLD_WINDOWS = 72000
GOLD_LEGACY_EQUIVALENT_27_TICK_BARS = 79_856

CURRENT_CONFIG_PATH = ACTIVE_CONFIG_PATH
ACTIVE_PROJECT: ResolvedProjectConfig | None = None
SHARED: dict[str, bool | int | float | str] = {}
SYMBOL = "XAUUSD"
SEQ_LEN = 0
TARGET_HORIZON = 0
FEATURE_ATR_PERIOD = 0
FEATURE_ATR_RATIO_PERIOD = 0
FEATURE_BOLLINGER_PERIOD = 0
FEATURE_DONCHIAN_FAST_PERIOD = 0
FEATURE_DONCHIAN_SLOW_PERIOD = 0
FEATURE_RET_2_PERIOD = 0
FEATURE_RET_3_PERIOD = 0
FEATURE_RET_6_PERIOD = 0
FEATURE_RET_12_PERIOD = 0
FEATURE_RET_20_PERIOD = 0
FEATURE_RSI_FAST_PERIOD = 0
FEATURE_RSI_SLOW_PERIOD = 0
FEATURE_RV_LONG_PERIOD = 0
FEATURE_SMA_FAST_PERIOD = 0
FEATURE_SMA_MID_PERIOD = 0
FEATURE_SMA_SLOW_PERIOD = 0
FEATURE_SMA_SLOPE_SHIFT = 0
FEATURE_SMA_TREND_FAST_PERIOD = 0
FEATURE_SPREAD_Z_PERIOD = 0
FEATURE_STOCH_PERIOD = 0
FEATURE_STOCH_SMOOTH_PERIOD = 0
FEATURE_TICK_COUNT_PERIOD = 0
FEATURE_TICK_IMBALANCE_FAST_PERIOD = 0
FEATURE_TICK_IMBALANCE_SLOW_PERIOD = 0
TARGET_ATR_PERIOD = 0
RV_PERIOD = 0
RETURN_PERIOD = 0
MAX_FEATURE_LOOKBACK = 0
WARMUP_BARS = 0
IMBALANCE_MIN_TICKS = 0
IMBALANCE_EMA_SPAN = 0
USE_IMBALANCE_EMA_THRESHOLD = False
USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD = False
PRIMARY_BAR_SECONDS = 0
PRIMARY_TICK_DENSITY = 0
BAR_DURATION_MS = 0
DEFAULT_FIXED_MOVE = 0.0
LABEL_SL_MULTIPLIER = 0.0
LABEL_TP_MULTIPLIER = 0.0
EXECUTION_SL_MULTIPLIER = 0.0
EXECUTION_TP_MULTIPLIER = 0.0
USE_ALL_WINDOWS = False
DEFAULT_EPOCHS = 0
DEFAULT_BATCH_SIZE = 0
DEFAULT_MAX_TRAIN_WINDOWS = 0
DEFAULT_MAX_EVAL_WINDOWS = 0
DEFAULT_PATIENCE = 0
DEFAULT_LOSS_MODE = "cross-entropy"
USE_NO_HOLD = False
LABEL_NAMES = ("HOLD", "BUY", "SELL")
LABEL_NAMES_BINARY = ("BUY", "SELL")

def apply_shared_settings(
    shared: dict[str, bool | int | float | str],
    *,
    project: ResolvedProjectConfig | None = None,
    shared_config_path: Path | None = None,
) -> None:
    global SHARED
    global ACTIVE_PROJECT
    global CURRENT_CONFIG_PATH
    global SYMBOL
    global SEQ_LEN
    global TARGET_HORIZON
    global FEATURE_ATR_PERIOD
    global FEATURE_ATR_RATIO_PERIOD
    global FEATURE_BOLLINGER_PERIOD
    global FEATURE_DONCHIAN_FAST_PERIOD
    global FEATURE_DONCHIAN_SLOW_PERIOD
    global FEATURE_RET_2_PERIOD
    global FEATURE_RET_3_PERIOD
    global FEATURE_RET_6_PERIOD
    global FEATURE_RET_12_PERIOD
    global FEATURE_RET_20_PERIOD
    global FEATURE_RSI_FAST_PERIOD
    global FEATURE_RSI_SLOW_PERIOD
    global FEATURE_RV_LONG_PERIOD
    global FEATURE_SMA_FAST_PERIOD
    global FEATURE_SMA_MID_PERIOD
    global FEATURE_SMA_SLOW_PERIOD
    global FEATURE_SMA_SLOPE_SHIFT
    global FEATURE_SMA_TREND_FAST_PERIOD
    global FEATURE_SPREAD_Z_PERIOD
    global FEATURE_STOCH_PERIOD
    global FEATURE_STOCH_SMOOTH_PERIOD
    global FEATURE_TICK_COUNT_PERIOD
    global FEATURE_TICK_IMBALANCE_FAST_PERIOD
    global FEATURE_TICK_IMBALANCE_SLOW_PERIOD
    global TARGET_ATR_PERIOD
    global RV_PERIOD
    global RETURN_PERIOD
    global MAX_FEATURE_LOOKBACK
    global WARMUP_BARS
    global IMBALANCE_MIN_TICKS
    global IMBALANCE_EMA_SPAN
    global USE_IMBALANCE_EMA_THRESHOLD
    global USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD
    global PRIMARY_BAR_SECONDS
    global PRIMARY_TICK_DENSITY
    global BAR_DURATION_MS
    global DEFAULT_FIXED_MOVE
    global LABEL_SL_MULTIPLIER
    global LABEL_TP_MULTIPLIER
    global EXECUTION_SL_MULTIPLIER
    global EXECUTION_TP_MULTIPLIER
    global USE_ALL_WINDOWS
    global DEFAULT_EPOCHS
    global DEFAULT_BATCH_SIZE
    global DEFAULT_MAX_TRAIN_WINDOWS
    global DEFAULT_MAX_EVAL_WINDOWS
    global DEFAULT_PATIENCE
    global DEFAULT_LOSS_MODE

    SHARED = dict(shared)
    ACTIVE_PROJECT = project
    if shared_config_path is not None:
        CURRENT_CONFIG_PATH = shared_config_path

    SYMBOL = str(SHARED.get("SYMBOL", "XAUUSD")).strip() or "XAUUSD"
    SEQ_LEN = int(SHARED["SEQ_LEN"])
    TARGET_HORIZON = int(SHARED["TARGET_HORIZON"])
    FEATURE_ATR_PERIOD = int(SHARED["FEATURE_ATR_PERIOD"])
    FEATURE_ATR_RATIO_PERIOD = int(SHARED["FEATURE_ATR_RATIO_PERIOD"])
    FEATURE_BOLLINGER_PERIOD = int(SHARED["FEATURE_BOLLINGER_PERIOD"])
    FEATURE_DONCHIAN_FAST_PERIOD = int(SHARED["FEATURE_DONCHIAN_FAST_PERIOD"])
    FEATURE_DONCHIAN_SLOW_PERIOD = int(SHARED["FEATURE_DONCHIAN_SLOW_PERIOD"])
    FEATURE_RET_2_PERIOD = int(SHARED["FEATURE_RET_2_PERIOD"])
    FEATURE_RET_3_PERIOD = int(SHARED["FEATURE_RET_3_PERIOD"])
    FEATURE_RET_6_PERIOD = int(SHARED["FEATURE_RET_6_PERIOD"])
    FEATURE_RET_12_PERIOD = int(SHARED["FEATURE_RET_12_PERIOD"])
    FEATURE_RET_20_PERIOD = int(SHARED["FEATURE_RET_20_PERIOD"])
    FEATURE_RSI_FAST_PERIOD = int(SHARED["FEATURE_RSI_FAST_PERIOD"])
    FEATURE_RSI_SLOW_PERIOD = int(SHARED["FEATURE_RSI_SLOW_PERIOD"])
    FEATURE_RV_LONG_PERIOD = int(SHARED["FEATURE_RV_LONG_PERIOD"])
    FEATURE_SMA_FAST_PERIOD = int(SHARED["FEATURE_SMA_FAST_PERIOD"])
    FEATURE_SMA_MID_PERIOD = int(SHARED["FEATURE_SMA_MID_PERIOD"])
    FEATURE_SMA_SLOW_PERIOD = int(SHARED["FEATURE_SMA_SLOW_PERIOD"])
    FEATURE_SMA_SLOPE_SHIFT = int(SHARED["FEATURE_SMA_SLOPE_SHIFT"])
    FEATURE_SMA_TREND_FAST_PERIOD = int(SHARED["FEATURE_SMA_TREND_FAST_PERIOD"])
    FEATURE_SPREAD_Z_PERIOD = int(SHARED["FEATURE_SPREAD_Z_PERIOD"])
    FEATURE_STOCH_PERIOD = int(SHARED["FEATURE_STOCH_PERIOD"])
    FEATURE_STOCH_SMOOTH_PERIOD = int(SHARED["FEATURE_STOCH_SMOOTH_PERIOD"])
    FEATURE_TICK_COUNT_PERIOD = int(SHARED["FEATURE_TICK_COUNT_PERIOD"])
    FEATURE_TICK_IMBALANCE_FAST_PERIOD = int(SHARED["FEATURE_TICK_IMBALANCE_FAST_PERIOD"])
    FEATURE_TICK_IMBALANCE_SLOW_PERIOD = int(SHARED["FEATURE_TICK_IMBALANCE_SLOW_PERIOD"])
    TARGET_ATR_PERIOD = int(SHARED["TARGET_ATR_PERIOD"])
    RV_PERIOD = int(SHARED["RV_PERIOD"])
    RETURN_PERIOD = int(SHARED["RETURN_PERIOD"])
    active_features = project.feature_columns if project is not None else MINIMAL_FEATURE_COLUMNS
    MAX_FEATURE_LOOKBACK = max_feature_lookback(SHARED, active_features)
    WARMUP_BARS = MAX_FEATURE_LOOKBACK
    IMBALANCE_MIN_TICKS = int(SHARED["IMBALANCE_MIN_TICKS"])
    IMBALANCE_EMA_SPAN = int(SHARED["IMBALANCE_EMA_SPAN"])
    USE_IMBALANCE_EMA_THRESHOLD = bool(SHARED["USE_IMBALANCE_EMA_THRESHOLD"])
    USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD = bool(SHARED["USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD"])
    PRIMARY_BAR_SECONDS = int(SHARED["PRIMARY_BAR_SECONDS"])
    PRIMARY_TICK_DENSITY = int(SHARED.get("PRIMARY_TICK_DENSITY", DEFAULT_PRIMARY_TICK_DENSITY))
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
    DEFAULT_LOSS_MODE = str(SHARED.get("DEFAULT_LOSS_MODE", "cross-entropy"))


apply_shared_settings(load_define_file(ACTIVE_CONFIG_PATH), shared_config_path=ACTIVE_CONFIG_PATH)


def symbol_ticks_path(symbol: str) -> Path:
    return Path("data") / symbol.strip().upper() / "ticks.csv"


def feature_macro_name(feature_name: str) -> str:
    return f"FEATURE_IDX_{project_feature_macro_name(feature_name)}"


def parse_args() -> argparse.Namespace:
    project = resolve_active_project_config(ACTIVE_CONFIG_PATH)
    apply_shared_settings(project.values, project=project, shared_config_path=project.config_path)
    values = project.values
    use_custom_lr = bool(values.get("USE_CUSTOM_LEARNING_RATE", False))
    use_custom_weight_decay = bool(values.get("USE_CUSTOM_WEIGHT_DECAY", False))
    use_max_bars = bool(values.get("USE_MAX_BARS", False))
    return argparse.Namespace(
        config_path=str(project.config_path),
        config_project=project,
        symbol=str(values.get("SYMBOL", "XAUUSD")).strip() or "XAUUSD",
        data_file=str(values.get("DATA_FILE", DEFAULT_DATA_FILE)),
        output_file=DEFAULT_OUTPUT_FILE,
        name=str(values.get("MODEL_NAME", "")).strip(),
        epochs=int(values.get("DEFAULT_EPOCHS", DEFAULT_EPOCHS)),
        batch_size=int(values.get("DEFAULT_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        max_train_windows=int(values.get("DEFAULT_MAX_TRAIN_WINDOWS", DEFAULT_MAX_TRAIN_WINDOWS)),
        max_eval_windows=int(values.get("DEFAULT_MAX_EVAL_WINDOWS", DEFAULT_MAX_EVAL_WINDOWS)),
        max_bars=int(values.get("MAX_BARS", 0)) if use_max_bars else 0,
        patience=int(values.get("DEFAULT_PATIENCE", DEFAULT_PATIENCE)),
        device=str(values.get("DEVICE", "cpu")).strip() or "cpu",
        focal_gamma=float(values.get("FOCAL_GAMMA", DEFAULT_FOCAL_GAMMA)),
        use_fixed_risk=bool(values.get("USE_FIXED_TARGETS", False)),
        use_fixed_time_bars=bool(values.get("USE_SECOND_BARS", False)),
        use_fixed_tick_bars=bool(values.get("USE_FIXED_TICK_BARS", False)),
        primary_tick_density=int(values.get("PRIMARY_TICK_DENSITY", DEFAULT_PRIMARY_TICK_DENSITY)),
        use_extended_features=not bool(values.get("USE_MINIMAL_FEATURE_SET", False)),
        no_hold=bool(values.get("USE_NO_HOLD", False)),
        config_architecture=project.architecture,
        gold=False,
        gold_new=False,
        gold_context=bool(values.get("USE_GOLD_CONTEXT", False)),
        use_minirocket_encoder=False,
        use_castor_encoder=False,
        ela=False,
        use_fusion_lstm_encoder=False,
        use_bilstm_encoder=False,
        use_gru_encoder=False,
        use_tcn_encoder=False,
        use_legacy_lstm_attention=False,
        use_tla_encoder=False,
        use_chronos_bolt=False,
        chronos_bolt_model=str(values.get("CHRONOS_BOLT_MODEL", DEFAULT_CHRONOS_BOLT_MODEL_ID)),
        chronos_patch_aligned_context=bool(values.get("USE_CHRONOS_PATCH_ALIGNED_CONTEXT", False)),
        chronos_auto_context=bool(values.get("USE_CHRONOS_AUTO_CONTEXT", False)),
        chronos_ensemble_contexts=bool(values.get("USE_CHRONOS_ENSEMBLE_CONTEXTS", False)),
        use_multihead_attention=bool(values.get("USE_MULTIHEAD_ATTENTION", False)),
        minirocket_features=int(values.get("MINIROCKET_FEATURES", DEFAULT_MINIROCKET_FEATURES)),
        attention_dim=int(values.get("ATTENTION_DIM", DEFAULT_MINIROCKET_ATTENTION_DIM)),
        attention_heads=int(values.get("ATTENTION_HEADS", DEFAULT_ATTENTION_HEADS)),
        attention_layers=int(values.get("ATTENTION_LAYERS", DEFAULT_ATTENTION_LAYERS)),
        attention_dropout=float(values.get("ATTENTION_DROPOUT", DEFAULT_ATTENTION_DROPOUT)),
        sequence_hidden_size=int(values.get("SEQUENCE_HIDDEN_SIZE", DEFAULT_SEQUENCE_HIDDEN_SIZE)),
        sequence_layers=int(values.get("SEQUENCE_LAYERS", DEFAULT_SEQUENCE_LAYERS)),
        sequence_dropout=float(values.get("SEQUENCE_DROPOUT", DEFAULT_SEQUENCE_DROPOUT)),
        tcn_levels=int(values.get("TCN_LEVELS", DEFAULT_TCN_LEVELS)),
        tcn_kernel_size=int(values.get("TCN_KERNEL_SIZE", DEFAULT_TCN_KERNEL_SIZE)),
        metaeditor_path=str(values.get("METAEDITOR_PATH", "")).strip(),
        skip_live_compile=bool(values.get("SKIP_LIVE_COMPILE", False)),
        archive_only=False,
        loss_mode=str(values.get("LOSS_MODE", "auto")).strip() or "auto",
        lr=float(values.get("LEARNING_RATE", 0.0)) if use_custom_lr else 0.0,
        weight_decay=float(values.get("WEIGHT_DECAY", 0.0)) if use_custom_weight_decay else -1.0,
        min_selected_trades=int(values.get("MIN_SELECTED_TRADES", DEFAULT_MIN_SELECTED_TRADES)),
        min_trade_precision=float(values.get("MIN_TRADE_PRECISION", DEFAULT_MIN_TRADE_PRECISION)),
        confidence_search_min=float(values.get("CONFIDENCE_SEARCH_MIN", DEFAULT_CONFIDENCE_SEARCH_MIN)),
        confidence_search_max=float(values.get("CONFIDENCE_SEARCH_MAX", DEFAULT_CONFIDENCE_SEARCH_MAX)),
        confidence_search_steps=int(values.get("CONFIDENCE_SEARCH_STEPS", DEFAULT_CONFIDENCE_SEARCH_STEPS)),
    )


def resolve_architecture(args: argparse.Namespace) -> str:
    configured_architecture = str(getattr(args, "config_architecture", "")).strip()
    if configured_architecture:
        return configured_architecture
    if args.use_chronos_bolt:
        return "chronos_bolt"
    if args.gold:
        return "gold_legacy"
    if args.gold_new:
        return "gold_new"
    if args.use_legacy_lstm_attention:
        return "legacy_lstm_attention"
    if args.ela:
        return "ela"
    if args.use_fusion_lstm_encoder:
        return "fusion_lstm"
    if args.use_bilstm_encoder:
        return "bilstm"
    if args.use_gru_encoder:
        return "gru"
    if args.use_tcn_encoder:
        return "tcn"
    if args.use_tla_encoder:
        return "tla"
    if args.use_minirocket_encoder:
        return "minirocket"
    if args.use_castor_encoder:
        return "castor"
    return "mamba"


def resolve_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def export_onnx_model(model: nn.Module, dummy_input: torch.Tensor, output_path: Path) -> None:
    export_attempts = (
        ("legacy", {"dynamo": False}),
        ("dynamo", {"dynamo": True}),
    )
    last_error: Exception | None = None

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except ValueError:
                pass

    for exporter_name, exporter_kwargs in export_attempts:
        try:
            log.info("ONNX export | trying %s exporter (opset=14)", exporter_name)
            torch.onnx.export(
                model,
                (dummy_input,),
                str(output_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=14,
                **exporter_kwargs,
            )
            log.info("ONNX export | succeeded with %s exporter", exporter_name)
            return
        except ModuleNotFoundError as exc:
            last_error = exc
            log.warning("ONNX export | %s exporter missing dependency: %s", exporter_name, exc)
        except Exception as exc:
            last_error = exc
            log.warning("ONNX export | %s exporter failed: %s", exporter_name, exc)

    if last_error is None:
        raise RuntimeError("ONNX export failed for an unknown reason.")
    raise RuntimeError(f"ONNX export failed with all available exporters: {last_error}") from last_error


def chronos_patch_aligned_tail_length(sequence_length: int, patch_size: int) -> int:
    if patch_size <= 0:
        return 0
    aligned = (int(sequence_length) // int(patch_size)) * int(patch_size)
    if aligned <= 0 or aligned >= int(sequence_length):
        return 0
    return aligned


def chronos_context_variants(args: argparse.Namespace, sequence_length: int, patch_size: int) -> tuple[tuple[int, ...], ...]:
    patch_aligned_tail = chronos_patch_aligned_tail_length(sequence_length, patch_size)

    if args.chronos_auto_context:
        variants: list[tuple[int, ...]] = [(0,)]
        if patch_size > 0:
            tail = (sequence_length // patch_size) * patch_size
            while tail >= patch_size:
                variant = (0,) if tail >= sequence_length else (tail,)
                if variant not in variants:
                    variants.append(variant)
                tail -= patch_size
        return tuple(variants)

    if args.chronos_ensemble_contexts and patch_aligned_tail > 0:
        return ((0, patch_aligned_tail),)

    if args.chronos_patch_aligned_context and patch_aligned_tail > 0:
        return ((patch_aligned_tail,),)

    return ((0,),)


def chronos_context_label(context_tail_lengths: Sequence[int]) -> str:
    return "+".join("full" if int(tail_length) <= 0 else str(int(tail_length)) for tail_length in context_tail_lengths)


def chronos_context_score(metrics: dict[str, float | int], min_selected: int) -> tuple[float, float, int, float]:
    selected_trades = int(metrics["selected_trades"])
    precision = float(metrics["precision"])
    precision_score = precision if np.isfinite(precision) else -1.0
    return (
        float(selected_trades >= min_selected),
        precision_score,
        selected_trades,
        float(metrics["trade_coverage"]),
    )


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
    if USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD:
        base_threshold = max(2.0, float(max(2, IMBALANCE_MIN_TICKS // 3)))
    else:
        base_threshold = max(2.0, float(IMBALANCE_MIN_TICKS))
    expected_abs_theta = base_threshold
    bar_ids = np.empty(len(prices), dtype=np.int64)
    current_bar = 0
    ticks_in_bar = 0
    theta = 0.0

    for i, sign in enumerate(tick_signs):
        bar_ids[i] = current_bar
        ticks_in_bar += 1
        theta += float(sign)
        threshold = expected_abs_theta if USE_IMBALANCE_EMA_THRESHOLD else base_threshold
        if ticks_in_bar >= IMBALANCE_MIN_TICKS and abs(theta) >= threshold:
            if USE_IMBALANCE_EMA_THRESHOLD:
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


def build_tick_bar_ids(tick_count: int, tick_density: int) -> np.ndarray:
    if tick_density <= 0:
        raise ValueError("PRIMARY_TICK_DENSITY must be positive.")
    return np.arange(tick_count, dtype=np.int64) // int(tick_density)


def infer_point_size_from_ticks(df_ticks: pd.DataFrame, max_samples: int = 200_000) -> float:
    prices = np.concatenate(
        [
            df_ticks["bid"].to_numpy(dtype=np.float64, copy=False),
            df_ticks["ask"].to_numpy(dtype=np.float64, copy=False),
        ]
    )
    prices = prices[np.isfinite(prices)]
    if len(prices) == 0:
        return 1.0

    sample = np.round(prices[: max_samples], 8)
    unique_prices = np.unique(sample)
    if len(unique_prices) < 2:
        return 1.0

    scaled = np.rint(unique_prices * 1e8).astype(np.int64)
    diffs = np.diff(scaled)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0

    gcd_points = int(np.gcd.reduce(diffs[: min(len(diffs), 50_000)]))
    point_size = gcd_points / 1e8 if gcd_points > 0 else 1.0
    return float(point_size if point_size > 0.0 else 1.0)


def build_market_bars(
    csv_path: Path,
    use_fixed_time_bars: bool,
    use_fixed_tick_bars: bool,
    tick_density: int,
    max_bars: int,
    require_gold_context: bool = False,
) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    chunks = []
    extended_usecols = ["time_msc", "bid", "ask", *GOLD_CONTEXT_TICK_COLUMNS]
    legacy_usecols = ["time_msc", "bid", "ask", "usdx", "usdjpy"]
    base_usecols = ["time_msc", "bid", "ask"]
    read_csv_kwargs = {
        "filepath_or_buffer": csv_path,
        "usecols": extended_usecols,
        "dtype": {
            "time_msc": np.int64,
            "bid": np.float64,
            "ask": np.float64,
            "usdx_bid": np.float64,
            "usdjpy_bid": np.float64,
        },
        "chunksize": 50000,
    }
    try:
        for chunk in pd.read_csv(**read_csv_kwargs):
            chunks.append(chunk)
    except ValueError:
        chunks.clear()
        try:
            read_csv_kwargs["usecols"] = legacy_usecols
            read_csv_kwargs["dtype"] = {
                "time_msc": np.int64,
                "bid": np.float64,
                "ask": np.float64,
                "usdx": np.float64,
                "usdjpy": np.float64,
            }
            for chunk in pd.read_csv(**read_csv_kwargs):
                chunks.append(chunk)
        except ValueError:
            chunks.clear()
            read_csv_kwargs["usecols"] = base_usecols
            read_csv_kwargs["dtype"] = {"time_msc": np.int64, "bid": np.float64, "ask": np.float64}
            for chunk in pd.read_csv(**read_csv_kwargs):
                chunks.append(chunk)
    except pd.errors.ParserError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        log.warning("Default CSV parser ran out of memory for %s; retrying with engine=python.", csv_path)
        chunks.clear()
        try:
            for chunk in pd.read_csv(**read_csv_kwargs, engine="python"):
                chunks.append(chunk)
        except ValueError:
            chunks.clear()
            try:
                read_csv_kwargs["usecols"] = legacy_usecols
                read_csv_kwargs["dtype"] = {
                    "time_msc": np.int64,
                    "bid": np.float64,
                    "ask": np.float64,
                    "usdx": np.float64,
                    "usdjpy": np.float64,
                }
                for chunk in pd.read_csv(**read_csv_kwargs, engine="python"):
                    chunks.append(chunk)
            except ValueError:
                chunks.clear()
                read_csv_kwargs["usecols"] = base_usecols
                read_csv_kwargs["dtype"] = {"time_msc": np.int64, "bid": np.float64, "ask": np.float64}
                for chunk in pd.read_csv(**read_csv_kwargs, engine="python"):
                    chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    if not df["time_msc"].is_monotonic_increasing:
        df = df.sort_values("time_msc").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No ticks found in {csv_path}")
    if "usdx_bid" not in df.columns:
        if "usdx" in df.columns:
            df["usdx_bid"] = df["usdx"]
        else:
            df["usdx_bid"] = np.nan
    if "usdjpy_bid" not in df.columns:
        if "usdjpy" in df.columns:
            df["usdjpy_bid"] = df["usdjpy"]
        else:
            df["usdjpy_bid"] = np.nan
    if require_gold_context:
        missing_columns = [name for name in GOLD_CONTEXT_TICK_COLUMNS if name not in df.columns]
        if missing_columns:
            raise ValueError(
                "Gold training requires auxiliary columns "
                f"{missing_columns} in {csv_path}. Re-export with `python export_data.py --profile gold --symbol XAUUSD`."
            )
        empty_columns = [name for name in GOLD_CONTEXT_TICK_COLUMNS if df[name].notna().sum() == 0]
        if empty_columns:
            raise ValueError(
                "Gold training found empty auxiliary columns "
                f"{empty_columns} in {csv_path}. Re-export with `python export_data.py --profile gold --symbol XAUUSD`."
            )
    point_size = infer_point_size_from_ticks(df)

    df["tick_sign"] = compute_tick_signs(df["bid"].to_numpy(dtype=np.float64, copy=False))
    df["spread"] = df["ask"] - df["bid"]

    if use_fixed_tick_bars:
        df["bar_id"] = build_tick_bar_ids(len(df), tick_density)
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
                time_open=("time_msc", "first"),
                time_close=("time_msc", "last"),
                usdx_bid=("usdx_bid", "last"),
                usdjpy_bid=("usdjpy_bid", "last"),
            )
            .reset_index(drop=True)
        )
        if max_bars > 0 and len(grouped) > max_bars:
            grouped = grouped.iloc[:max_bars].reset_index(drop=True)
            log.info("Capped fixed-tick bars to %d rows.", max_bars)
        log.info(
            "Built %d bars in %.2fs using fixed %d-tick bars | point_size=%.8f",
            len(grouped),
            time.time() - t0,
            tick_density,
            point_size,
        )
        return grouped, point_size

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
                usdx_bid=("usdx_bid", "last"),
                usdjpy_bid=("usdjpy_bid", "last"),
            )
            .reset_index()
        )
        grouped["time_open"] = grouped["bar_id"] * BAR_DURATION_MS
        grouped["time_close"] = grouped["time_open"] + BAR_DURATION_MS
        grouped = grouped.drop(columns=["bar_id"])
        if max_bars > 0 and len(grouped) > max_bars:
            grouped = grouped.iloc[:max_bars].reset_index(drop=True)
            log.info("Capped fixed-time bars to %d rows.", max_bars)
        log.info(
            "Built %d bars in %.2fs using fixed %ds bars | point_size=%.8f",
            len(grouped),
            time.time() - t0,
            PRIMARY_BAR_SECONDS,
            point_size,
        )
        return grouped, point_size

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
            usdx_bid=("usdx_bid", "last"),
            usdjpy_bid=("usdjpy_bid", "last"),
        )
        .reset_index(drop=True)
    )
    if max_bars > 0 and len(grouped) > max_bars:
        grouped = grouped.iloc[:max_bars].reset_index(drop=True)
        log.info("Capped imbalance bars to %d rows.", max_bars)
    log.info(
        "Built %d bars in %.2fs using imbalance bars min_ticks=%d span=%d | point_size=%.8f",
        len(grouped),
        time.time() - t0,
        IMBALANCE_MIN_TICKS,
        IMBALANCE_EMA_SPAN,
        point_size,
    )
    return grouped, point_size


def rolling_population_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = rolling_population_std(series, window)
    return (series - mean) / (std + EPS)


def simple_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.rolling(period, min_periods=period).mean()
    avg_loss = losses.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return (100.0 - (100.0 / (1.0 + rs)) - 50.0) / 50.0


def fixed_move_price_distance(fixed_move_points: float, point_size: float) -> float:
    return float(fixed_move_points) * float(point_size)


def compute_features(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    close = df["close"].astype(float)
    open_price = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tick_count = df["tick_count"].astype(float)
    tick_imbalance = df["tick_imbalance"].astype(float)
    prev_close = close.shift(1)
    ret1 = np.log(close / (prev_close + EPS))
    atr_feature = wilder_atr(df["high"], df["low"], close, period=FEATURE_ATR_PERIOD)
    spread_rel = df["spread"] / (close + EPS)

    sma_fast = close.rolling(FEATURE_SMA_FAST_PERIOD, min_periods=FEATURE_SMA_FAST_PERIOD).mean()
    sma_trend_fast = close.rolling(FEATURE_SMA_TREND_FAST_PERIOD, min_periods=FEATURE_SMA_TREND_FAST_PERIOD).mean()
    sma_mid = close.rolling(FEATURE_SMA_MID_PERIOD, min_periods=FEATURE_SMA_MID_PERIOD).mean()
    sma_slow = close.rolling(FEATURE_SMA_SLOW_PERIOD, min_periods=FEATURE_SMA_SLOW_PERIOD).mean()
    rv_3 = rolling_population_std(ret1, FEATURE_RET_3_PERIOD)
    rv_6 = rolling_population_std(ret1, FEATURE_RET_6_PERIOD)
    rv_18 = rolling_population_std(ret1, FEATURE_RV_LONG_PERIOD)
    high_fast = high.rolling(FEATURE_DONCHIAN_FAST_PERIOD, min_periods=FEATURE_DONCHIAN_FAST_PERIOD).max()
    low_fast = low.rolling(FEATURE_DONCHIAN_FAST_PERIOD, min_periods=FEATURE_DONCHIAN_FAST_PERIOD).min()
    high_slow = high.rolling(FEATURE_DONCHIAN_SLOW_PERIOD, min_periods=FEATURE_DONCHIAN_SLOW_PERIOD).max()
    low_slow = low.rolling(FEATURE_DONCHIAN_SLOW_PERIOD, min_periods=FEATURE_DONCHIAN_SLOW_PERIOD).min()
    high_stoch = high.rolling(FEATURE_STOCH_PERIOD, min_periods=FEATURE_STOCH_PERIOD).max()
    low_stoch = low.rolling(FEATURE_STOCH_PERIOD, min_periods=FEATURE_STOCH_PERIOD).min()
    stoch_k_9 = (close - low_stoch) / (high_stoch - low_stoch + EPS)
    stoch_d_3 = stoch_k_9.rolling(FEATURE_STOCH_SMOOTH_PERIOD, min_periods=FEATURE_STOCH_SMOOTH_PERIOD).mean()
    bollinger_std_20 = rolling_population_std(close, FEATURE_BOLLINGER_PERIOD)

    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = ret1
    feat["high_rel_prev"] = np.log(high / (prev_close + EPS))
    feat["low_rel_prev"] = np.log(low / (prev_close + EPS))
    feat["spread_rel"] = spread_rel
    feat["close_in_range"] = (close - low) / (high - low + 1e-8)
    feat["atr_rel"] = atr_feature / (close + EPS)
    feat["rv"] = rolling_population_std(ret1, RV_PERIOD)
    feat["ret_n"] = np.log(close / (close.shift(RETURN_PERIOD) + EPS))
    feat["tick_imbalance"] = tick_imbalance

    requires_gold_context = any(name in feature_columns for name in GOLD_CONTEXT_FEATURE_COLUMNS)
    usdx_bid = df.get("usdx_bid")
    usdjpy_bid = df.get("usdjpy_bid")
    if requires_gold_context:
        if usdx_bid is None or usdjpy_bid is None:
            raise ValueError("Gold-context features requested but auxiliary columns are missing from the bar data.")
        if usdx_bid.notna().sum() == 0 or usdjpy_bid.notna().sum() == 0:
            raise ValueError("Gold-context features requested but auxiliary columns are empty after bar construction.")
        usdx_bid = usdx_bid.ffill().bfill()
        usdjpy_bid = usdjpy_bid.ffill().bfill()
    else:
        if usdx_bid is None:
            usdx_bid = close
        else:
            usdx_bid = usdx_bid.fillna(close)
        if usdjpy_bid is None:
            usdjpy_bid = close
        else:
            usdjpy_bid = usdjpy_bid.fillna(close)
    feat["usdx_ret1"] = np.log(usdx_bid / (usdx_bid.shift(1) + EPS))
    feat["usdjpy_ret1"] = np.log(usdjpy_bid / (usdjpy_bid.shift(1) + EPS))

    feat["ret_2"] = np.log(close / (close.shift(FEATURE_RET_2_PERIOD) + EPS))
    feat["ret_3"] = np.log(close / (close.shift(FEATURE_RET_3_PERIOD) + EPS))
    feat["ret_6"] = np.log(close / (close.shift(FEATURE_RET_6_PERIOD) + EPS))
    feat["ret_12"] = np.log(close / (close.shift(FEATURE_RET_12_PERIOD) + EPS))
    feat["ret_20"] = np.log(close / (close.shift(FEATURE_RET_20_PERIOD) + EPS))
    feat["open_rel_prev"] = np.log(open_price / (prev_close + EPS))
    feat["range_rel"] = (high - low) / (close + EPS)
    feat["body_rel"] = (close - open_price) / (close + EPS)
    feat["upper_wick_rel"] = (high - np.maximum(open_price, close)) / (close + EPS)
    feat["lower_wick_rel"] = (np.minimum(open_price, close) - low) / (close + EPS)
    feat["close_rel_sma_3"] = np.log(close / (sma_fast + EPS))
    feat["close_rel_sma_9"] = np.log(close / (sma_mid + EPS))
    feat["close_rel_sma_20"] = np.log(close / (sma_slow + EPS))
    feat["sma_3_9_gap"] = np.log(sma_fast / (sma_mid + EPS))
    feat["sma_5_20_gap"] = np.log(sma_trend_fast / (sma_slow + EPS))
    feat["sma_9_20_gap"] = np.log(sma_mid / (sma_slow + EPS))
    feat["sma_slope_9"] = np.log(sma_mid / (sma_mid.shift(FEATURE_SMA_SLOPE_SHIFT) + EPS))
    feat["sma_slope_20"] = np.log(sma_slow / (sma_slow.shift(FEATURE_SMA_SLOPE_SHIFT) + EPS))
    feat["rv_3"] = rv_3
    feat["rv_6"] = rv_6
    feat["rv_18"] = rv_18
    feat["donchian_pos_9"] = (close - low_fast) / (high_fast - low_fast + EPS)
    feat["donchian_width_9"] = (high_fast - low_fast) / (close + EPS)
    feat["donchian_pos_20"] = (close - low_slow) / (high_slow - low_slow + EPS)
    feat["donchian_width_20"] = (high_slow - low_slow) / (close + EPS)
    tick_count_sma_9 = tick_count.rolling(FEATURE_TICK_COUNT_PERIOD, min_periods=FEATURE_TICK_COUNT_PERIOD).mean()
    feat["tick_count_rel_9"] = tick_count / (tick_count_sma_9 + EPS) - 1.0
    feat["tick_count_z_9"] = rolling_zscore(tick_count, FEATURE_TICK_COUNT_PERIOD)
    feat["tick_count_chg"] = np.log((tick_count + 1.0) / (tick_count.shift(1) + 1.0))
    feat["tick_imbalance_sma_5"] = tick_imbalance.rolling(
        FEATURE_TICK_IMBALANCE_FAST_PERIOD,
        min_periods=FEATURE_TICK_IMBALANCE_FAST_PERIOD,
    ).mean()
    feat["tick_imbalance_sma_9"] = tick_imbalance.rolling(
        FEATURE_TICK_IMBALANCE_SLOW_PERIOD,
        min_periods=FEATURE_TICK_IMBALANCE_SLOW_PERIOD,
    ).mean()
    feat["spread_z_9"] = rolling_zscore(spread_rel, FEATURE_SPREAD_Z_PERIOD)
    feat["rsi_6"] = simple_rsi(close, FEATURE_RSI_FAST_PERIOD)
    feat["rsi_14"] = simple_rsi(close, FEATURE_RSI_SLOW_PERIOD)
    feat["stoch_k_9"] = stoch_k_9
    feat["stoch_d_3"] = stoch_d_3
    feat["stoch_gap"] = stoch_k_9 - stoch_d_3
    feat["bollinger_pos_20"] = (close - sma_slow) / (2.0 * bollinger_std_20 + EPS)
    feat["bollinger_width_20"] = (4.0 * bollinger_std_20) / (sma_slow + EPS)
    feat["atr_ratio_20"] = np.log(
        atr_feature
        / (
            atr_feature.rolling(FEATURE_ATR_RATIO_PERIOD, min_periods=FEATURE_ATR_RATIO_PERIOD).mean() + EPS
        )
    )

    missing_features = [name for name in feature_columns if name not in feat.columns]
    if missing_features:
        raise KeyError(f"Missing computed features: {missing_features}")
    return feat.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False)


def get_triple_barrier_labels(
    bars: pd.DataFrame,
    use_atr_risk: bool,
    fixed_move_price: float,
) -> np.ndarray:
    close = bars["close"].to_numpy(dtype=np.float64, copy=False)
    high = bars["high"].to_numpy(dtype=np.float64, copy=False)
    low = bars["low"].to_numpy(dtype=np.float64, copy=False)
    spread = bars["spread"].to_numpy(dtype=np.float64, copy=False)
    ask_high = bars["ask_high"].to_numpy(dtype=np.float64, copy=False)
    ask_low = bars["ask_low"].to_numpy(dtype=np.float64, copy=False)
    atr_target = None
    if use_atr_risk:
        atr_target = wilder_atr(
            bars["high"], bars["low"], bars["close"], period=TARGET_ATR_PERIOD
        ).to_numpy(dtype=np.float64, copy=False)

    labels = np.zeros(len(bars), dtype=np.int64)
    for i in range(len(bars) - TARGET_HORIZON):
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
            long_tp = long_entry + fixed_move_price
            long_sl = long_entry - fixed_move_price
            short_tp = short_entry - fixed_move_price
            short_sl = short_entry + fixed_move_price

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


def make_sample_weights(labels: np.ndarray) -> torch.Tensor:
    class_weights = make_class_weights(labels).to(torch.float64).numpy()
    sample_weights = class_weights[labels.astype(np.int64)]
    sample_weights /= max(sample_weights.mean(), 1e-12)
    return torch.tensor(sample_weights, dtype=torch.double)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.to(torch.float32))
        else:
            self.register_buffer("alpha", None)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1.0 - pt).pow(self.gamma)
        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = self.alpha[targets]
        loss = -alpha_t * focal_term * log_pt
        return loss.mean()


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sample_weights: torch.Tensor | None = None,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
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


def gate_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    if probs.shape[1] == 2:
        selected = confidences >= threshold
    else:
        selected = (preds > 0) & (confidences >= threshold)
    selected_trades = int(selected.sum())
    precision = float((preds[selected] == labels[selected]).mean()) if selected_trades else float("nan")
    selected_mean_confidence = float(confidences[selected].mean()) if selected_trades else float("nan")
    return {
        "selected_trades": selected_trades,
        "trade_coverage": float(selected.mean()),
        "precision": precision,
        "mean_confidence": float(confidences.mean()),
        "selected_mean_confidence": selected_mean_confidence,
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def choose_confidence_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    min_selected: int,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> float:
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    is_binary = probs.shape[1] == 2
    if is_binary:
        candidate_mask = np.ones(len(preds), dtype=bool)
    else:
        candidate_mask = preds > 0
    threshold_min = min(max(0.0, float(threshold_min)), 0.999999)
    threshold_max = min(max(threshold_min, float(threshold_max)), 0.999999)
    threshold_steps = max(2, int(threshold_steps))
    if is_binary:
        candidate_count = len(preds)
    else:
        candidate_count = int(candidate_mask.sum())
    if candidate_count == 0:
        log.warning(
            "Confidence gate selection: model produced no BUY/SELL predictions; falling back to threshold %.2f.",
            threshold_min,
        )
        return threshold_min

    min_selected = max(1, min_selected)
    best_threshold = threshold_min
    best_precision = -1.0
    best_selected = -1
    best_coverage = -1.0
    relaxed_threshold = threshold_min
    relaxed_precision = -1.0
    relaxed_selected = -1
    relaxed_coverage = -1.0
    found_candidate = False
    found_relaxed_candidate = False

    for threshold in np.linspace(threshold_min, threshold_max, threshold_steps):
        selected = candidate_mask & (confidences >= threshold)
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue

        precision = float((preds[selected] == labels[selected]).mean())
        coverage = float(selected.mean())
        found_relaxed_candidate = True
        if precision > relaxed_precision + 1e-12 or (
            abs(precision - relaxed_precision) <= 1e-12
            and (selected_count > relaxed_selected or (selected_count == relaxed_selected and coverage > relaxed_coverage))
        ):
            relaxed_threshold = float(threshold)
            relaxed_precision = precision
            relaxed_selected = selected_count
            relaxed_coverage = coverage

        if selected_count < min_selected:
            continue
        found_candidate = True
        if precision > best_precision + 1e-12 or (
            abs(precision - best_precision) <= 1e-12
            and (selected_count > best_selected or (selected_count == best_selected and coverage > best_coverage))
        ):
            best_threshold = float(threshold)
            best_precision = precision
            best_selected = selected_count
            best_coverage = coverage

    if found_candidate:
        print("Chosen confidence threshold: %.2f with precision %.4f and coverage %.4f" % (best_threshold, best_precision, best_coverage))
        return best_threshold

    if found_relaxed_candidate:
        log.warning(
            "Confidence gate selection: no threshold produced at least %d BUY/SELL trades; "
            "falling back to threshold %.2f with %d trades and precision %.4f.",
            min_selected,
            relaxed_threshold,
            relaxed_selected,
            relaxed_precision,
        )
        print(
            "Chosen confidence threshold: %.2f with precision %.4f and coverage %.4f"
            % (relaxed_threshold, relaxed_precision, relaxed_coverage)
        )
        return relaxed_threshold

    log.warning(
        "Confidence gate selection: no threshold selected any BUY/SELL trades; falling back to threshold %.2f.",
        threshold_min,
    )
    return threshold_min


def summarize_gate(name: str, probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics = gate_metrics(labels, probs, threshold)
    if metrics["selected_trades"]:
        log.info(
            "%s: threshold=%.2f precision=%.4f coverage=%.4f trades=%d mean_selected_conf=%.4f",
            name,
            threshold,
            float(metrics["precision"]),
            float(metrics["trade_coverage"]),
            int(metrics["selected_trades"]),
            float(metrics["selected_mean_confidence"]),
        )
    else:
        log.warning("%s: threshold=%.2f produced no trades.", name, threshold)
    return metrics


def class_count_lines(labels: np.ndarray, label_names: tuple[str, ...] | None = None) -> list[str]:
    if label_names is None:
        label_names = LABEL_NAMES
    counts = np.bincount(labels, minlength=len(label_names))
    return [f"{label_names[i]}: {int(counts[i])}" for i in range(len(label_names))]


def confusion_matrix_df(labels: np.ndarray, preds: np.ndarray, label_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    if label_names is None:
        label_names = LABEL_NAMES
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


def build_prediction_frame(labels: np.ndarray, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    if probs.shape[1] == 2:
        active_names = LABEL_NAMES_BINARY
        selected = (confidences >= threshold)
    else:
        active_names = LABEL_NAMES
        selected = (preds > 0) & (confidences >= threshold)
    frame = pd.DataFrame(
        {
            "true_label": labels.astype(np.int64),
            "pred_label": preds.astype(np.int64),
            "true_name": [active_names[int(v)] for v in labels],
            "pred_name": [active_names[int(v)] for v in preds],
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
        f"- target_horizon: {TARGET_HORIZON}",
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


def format_float_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(v):.8f}f" for v in values)


def build_mql_config(
    project: ResolvedProjectConfig,
    median: np.ndarray,
    iqr: np.ndarray,
    primary_confidence: float,
    use_atr_risk: bool,
    use_fixed_time_bars: bool,
    architecture: str,
    use_multihead_attention: bool,
    feature_columns: tuple[str, ...],
    feature_profile: str,
    use_extended_features: bool,
    use_fixed_tick_bars: bool,
) -> str:
    base_text = project.config_path.read_text(encoding="utf-8").rstrip()
    def override_define(name: str, value: bool | int | float | str) -> list[str]:
        return [
            f"#ifdef {name}",
            f"#undef {name}",
            "#endif",
            f"#define {name} {render_define_value(value)}",
        ]

    override_lines: list[str] = [
        "",
        "// --- Resolved Model Overrides ---",
    ]
    for name, value in (
        ("MAX_FEATURE_LOOKBACK", MAX_FEATURE_LOOKBACK),
        ("WARMUP_BARS", WARMUP_BARS),
        ("MODEL_ARCHITECTURE", architecture),
        ("MODEL_FEATURE_PROFILE", feature_profile),
        ("MODEL_USE_ATR_RISK", 1 if use_atr_risk else 0),
        ("MODEL_USE_FIXED_TICK_BARS", 1 if use_fixed_tick_bars else 0),
        ("MODEL_USE_FIXED_TIME_BARS", 1 if use_fixed_time_bars else 0),
        ("MODEL_USE_MULTIHEAD_ATTENTION", 1 if use_multihead_attention else 0),
        ("MODEL_USE_EXTENDED_FEATURES", 1 if use_extended_features else 0),
        ("MODEL_USE_MINIROCKET", 1 if architecture == "minirocket" else 0),
        ("MODEL_USE_CASTOR", 1 if architecture == "castor" else 0),
        ("MODEL_USE_ELA", 1 if architecture == "ela" else 0),
        ("MODEL_USE_FUSION_LSTM", 1 if architecture == "fusion_lstm" else 0),
        ("MODEL_USE_BILSTM", 1 if architecture == "bilstm" else 0),
        ("MODEL_USE_GRU", 1 if architecture == "gru" else 0),
        ("MODEL_USE_TCN", 1 if architecture == "tcn" else 0),
        ("MODEL_USE_TLA", 1 if architecture == "tla" else 0),
        ("MODEL_USE_LEGACY_LSTM_ATTENTION", 1 if architecture == "legacy_lstm_attention" else 0),
        ("MODEL_USE_CHRONOS", 1 if architecture == "chronos_bolt" else 0),
        ("MODEL_USE_CHRONOS_BOLT", 1 if architecture == "chronos_bolt" else 0),
    ):
        override_lines.extend(override_define(name, value))
    override_lines.extend(
        [
            "#ifdef REQUIRED_HISTORY_INDEX",
            "#undef REQUIRED_HISTORY_INDEX",
            "#endif",
            "#define REQUIRED_HISTORY_INDEX (SEQ_LEN + MAX_FEATURE_LOOKBACK - 1)",
        ]
    )
    if project.architecture_config_path is not None:
        override_lines.extend(
            override_define(
                "MODEL_ARCHITECTURE_CONFIG",
                str(project.architecture_config_path.relative_to(ACTIVE_CONFIG_PATH.parent)),
            )
        )
    feature_macro_lines = [
        "#ifdef MODEL_FEATURE_COUNT",
        "#undef MODEL_FEATURE_COUNT",
        "#endif",
        f"#define MODEL_FEATURE_COUNT {len(feature_columns)}",
    ]
    for feature_index, feature_name in enumerate(feature_columns):
        macro_name = feature_macro_name(feature_name)
        feature_macro_lines.extend(
            [
                f"#ifdef {macro_name}",
                f"#undef {macro_name}",
                "#endif",
                f"#define {macro_name} {feature_index}",
            ]
        )

    return "\n".join(
        [
            "// Auto-generated by nn.py. Re-run training to refresh these values.",
            base_text,
            *override_lines,
            (
                "// Gold legacy reference: old/nn.py trained on 2,160,000 ticks (tick_density=144) -> "
                "14,856 bars -> 14,707 windows (seq_len=120, horizon=30)."
                if architecture == "gold_legacy"
                else ""
            ),
            *feature_macro_lines,
            f"#define PRIMARY_CONFIDENCE {primary_confidence:.8f}",
            f"float medians[MODEL_FEATURE_COUNT] = {{{format_float_array(median)}}};",
            f"float iqrs[MODEL_FEATURE_COUNT] = {{{format_float_array(iqr)}}};",
        ]
    )


def resolve_loss_mode(_architecture: str, requested_mode: str) -> str:
    if _architecture == "chronos_bolt":
        return "zero-shot"
    if requested_mode != "auto":
        return requested_mode
    return DEFAULT_LOSS_MODE


def main() -> None:
    t0 = time.time()
    args = parse_args()
    project = args.config_project
    apply_shared_settings(project.values, project=project, shared_config_path=project.config_path)
    torch.manual_seed(42)
    np.random.seed(42)
    architecture = resolve_architecture(args)
    requested_model_name = args.name.strip()
    model_name = sanitize_model_name(requested_model_name)
    if requested_model_name and not model_name:
        log.warning("Model name %r sanitized to empty; using a timestamp-only archive folder.", requested_model_name)
    elif requested_model_name and model_name != requested_model_name:
        log.info("Model folder prefix sanitized from %r to %r.", requested_model_name, model_name)
    use_extended_features = bool(args.use_extended_features)
    use_atr_risk = not bool(args.use_fixed_risk)
    use_fixed_time_bars = bool(args.use_fixed_time_bars)
    use_fixed_tick_bars = bool(args.use_fixed_tick_bars)
    use_no_hold = bool(args.no_hold)
    active_label_names = LABEL_NAMES_BINARY if use_no_hold else LABEL_NAMES
    if architecture == "chronos_bolt" and use_extended_features:
        log.warning("Chronos-Bolt backend uses the minimal live feature pack; ignoring extra feature switches.")
        use_extended_features = False
    feature_columns = project.feature_columns
    feature_profile = project.feature_profile
    feature_count = len(feature_columns)
    use_multihead_attention = bool(args.use_multihead_attention)
    if use_fixed_tick_bars and use_fixed_time_bars:
        raise ValueError("Choose only one bar mode: fixed-time or fixed-tick.")
    if architecture == "legacy_lstm_attention":
        if use_multihead_attention:
            log.info("Legacy LSTM attention architecture has built-in self-attention; the -a flag is redundant.")
        else:
            log.info("Legacy LSTM attention architecture includes self-attention by design.")
        use_multihead_attention = True
        if args.sequence_hidden_size != DEFAULT_SEQUENCE_HIDDEN_SIZE:
            log.warning(
                "Legacy LSTM attention fixes its recurrent width to the active feature count; ignoring --sequence-hidden-size=%d.",
                args.sequence_hidden_size,
            )
        if args.sequence_layers != DEFAULT_SEQUENCE_LAYERS:
            log.warning(
                "Legacy LSTM attention uses a single recurrent layer; ignoring --sequence-layers=%d.",
                args.sequence_layers,
            )
        if abs(args.sequence_dropout - DEFAULT_SEQUENCE_DROPOUT) > 1e-12:
            log.warning(
                "Legacy LSTM attention does not use the recurrent dropout path; ignoring --sequence-dropout=%.3f.",
                args.sequence_dropout,
            )
        if args.attention_layers != DEFAULT_ATTENTION_LAYERS:
            log.warning(
                "Legacy LSTM attention uses a single self-attention block; ignoring --attention-layers=%d.",
                args.attention_layers,
            )
    if architecture == "ela":
        if not use_multihead_attention:
            log.info("ELA uses the multihead attention head by design; enabling attention automatically.")
        use_multihead_attention = True
    if architecture == "fusion_lstm":
        if not use_multihead_attention:
            log.info("Fusion-LSTM includes its residual self-attention block by design; enabling attention automatically.")
        use_multihead_attention = True
        if args.sequence_hidden_size != DEFAULT_SEQUENCE_HIDDEN_SIZE:
            log.warning(
                "Fusion-LSTM fixes its recurrent width to the active feature count; ignoring --sequence-hidden-size=%d.",
                args.sequence_hidden_size,
            )
        if args.sequence_layers != DEFAULT_SEQUENCE_LAYERS:
            log.warning("Fusion-LSTM uses a single Mish-LSTM layer; ignoring --sequence-layers=%d.", args.sequence_layers)
        if abs(args.sequence_dropout - DEFAULT_SEQUENCE_DROPOUT) > 1e-12:
            log.warning(
                "Fusion-LSTM does not use the recurrent dropout path; ignoring --sequence-dropout=%.3f.",
                args.sequence_dropout,
            )
        if args.attention_layers != DEFAULT_ATTENTION_LAYERS:
            log.warning(
                "Fusion-LSTM uses a single self-attention block; ignoring --attention-layers=%d.",
                args.attention_layers,
            )
    if architecture == "chronos_bolt" and use_multihead_attention:
        log.warning("Chronos-Bolt backend ignores multihead-attention settings.")
        use_multihead_attention = False
    if architecture != "chronos_bolt" and (
        args.chronos_patch_aligned_context or args.chronos_auto_context or args.chronos_ensemble_contexts
    ):
        log.warning("Chronos context switches are ignored unless MODEL_ARCHITECTURE is chronos_bolt.")
    if use_fixed_time_bars and PRIMARY_BAR_SECONDS <= 0:
        raise ValueError("PRIMARY_BAR_SECONDS must be positive.")
    if use_fixed_tick_bars and args.primary_tick_density <= 0:
        raise ValueError("PRIMARY_TICK_DENSITY must be positive.")
    if DEFAULT_FIXED_MOVE <= 0.0:
        raise ValueError("DEFAULT_FIXED_MOVE must be positive in points.")

    data_path = resolve_local_path(args.data_file)
    archive_only = bool(args.archive_only)
    if archive_only and not args.skip_live_compile:
        log.info("archive-only mode implies skip-live-compile; local live references will still be updated.")

    requested_device = str(args.device).strip() or "cpu"
    device = torch.device(requested_device)
    log.info("Using device: %s", device)
    log.info(
        "Shared config | path=%s seq_len=%d horizon=%d atr_feature=%d atr_target=%d rv=%d ret=%d "
        "bar_mode=%s imbalance_min_ticks=%d imbalance_ema_span=%d bar_seconds=%d tick_density=%d risk_mode=%s fixed_move_points=%.2f "
        "label_sl=%.2f label_tp=%.2f exec_sl=%.2f exec_tp=%.2f use_all_windows=%d",
        CURRENT_CONFIG_PATH,
        SEQ_LEN,
        TARGET_HORIZON,
        FEATURE_ATR_PERIOD,
        TARGET_ATR_PERIOD,
        RV_PERIOD,
        RETURN_PERIOD,
        "FIXED_TICK" if use_fixed_tick_bars else ("FIXED_TIME" if use_fixed_time_bars else "IMBALANCE"),
        IMBALANCE_MIN_TICKS,
        IMBALANCE_EMA_SPAN,
        PRIMARY_BAR_SECONDS,
        args.primary_tick_density,
        "ATR" if use_atr_risk else "FIXED",
        DEFAULT_FIXED_MOVE,
        LABEL_SL_MULTIPLIER,
        LABEL_TP_MULTIPLIER,
        EXECUTION_SL_MULTIPLIER,
        EXECUTION_TP_MULTIPLIER,
        int(USE_ALL_WINDOWS),
    )
    log.info(
        "Run config | symbol=%s architecture=%s attention=%d focal_gamma=%.2f feature_profile=%s feature_count=%d",
        SYMBOL,
        architecture.upper(),
        int(use_multihead_attention),
        args.focal_gamma,
        feature_profile,
        feature_count,
    )

    bars, point_size = build_market_bars(
        data_path,
        use_fixed_time_bars=use_fixed_time_bars,
        use_fixed_tick_bars=use_fixed_tick_bars,
        tick_density=args.primary_tick_density,
        max_bars=int(args.max_bars),
        require_gold_context=bool(args.gold_context) and not bool(SHARED.get("USE_MINIMAL_FEATURE_SET", False)),
    )
    fixed_move_price = fixed_move_price_distance(DEFAULT_FIXED_MOVE, point_size)
    log.info(
        "Fixed risk config | fixed_move_points=%.2f point_size=%.8f fixed_move_price=%.8f",
        DEFAULT_FIXED_MOVE,
        point_size,
        fixed_move_price,
    )
    x_all = compute_features(bars, feature_columns=feature_columns)
    y_all = get_triple_barrier_labels(bars, use_atr_risk=use_atr_risk, fixed_move_price=fixed_move_price)

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

    if use_no_hold:
        train_mask = y_train > 0
        val_mask = y_val > 0
        test_mask = y_test > 0
        x_train = x_train[train_mask]
        y_train = y_train[train_mask] - 1
        x_val = x_val[val_mask]
        y_val = y_val[val_mask] - 1
        x_test = x_test[test_mask]
        y_test = y_test[test_mask] - 1
        log.info(
            "No-hold mode | filtered train=%d val=%d test=%d",
            len(x_train),
            len(x_val),
            len(x_test),
        )

    loss_mode = resolve_loss_mode(architecture, args.loss_mode)
    export_model: nn.Module | None = None

    if architecture == "chronos_bolt":
        if args.loss_mode != "auto":
            log.warning("Chronos-Bolt backend is zero-shot; ignoring --loss-mode=%s.", args.loss_mode)
        if args.lr > 0.0 or args.weight_decay >= 0.0:
            log.warning("Chronos-Bolt backend is zero-shot; ignoring optimizer overrides.")
        log.info(
            "Chronos-Bolt backend | model_id=%s feature_profile=%s prediction_length=%d",
            args.chronos_bolt_model,
            feature_profile,
            TARGET_HORIZON,
        )
        chronos_bolt_batch_size = max(1, min(args.batch_size, 16))
        if chronos_bolt_batch_size != args.batch_size:
            log.info(
                "Chronos-Bolt batch size capped to %d tasks to keep CPU memory use predictable.",
                chronos_bolt_batch_size,
            )
        export_model = load_chronos_bolt_barrier_model(
            device=device,
            model_id=args.chronos_bolt_model,
            median=median,
            iqr=iqr,
            feature_columns=feature_columns,
            prediction_length=TARGET_HORIZON,
            use_atr_risk=use_atr_risk,
            label_tp_multiplier=LABEL_TP_MULTIPLIER,
            label_sl_multiplier=LABEL_SL_MULTIPLIER,
            context_tail_lengths=(0,),
        ).to(device)
        val_loader = make_loader(
            x_val,
            y_val,
            chronos_bolt_batch_size,
            shuffle=False,
        )
        test_loader = make_loader(
            x_test,
            y_test,
            chronos_bolt_batch_size,
            shuffle=False,
        )
        context_variants = chronos_context_variants(
            args,
            sequence_length=SEQ_LEN,
            patch_size=int(getattr(export_model, "patch_size", 0)),
        )
        selected_context_variant = context_variants[0]

        if args.chronos_auto_context and len(context_variants) > 1:
            best_score: tuple[float, float, int, float] | None = None
            best_val_logits: np.ndarray | None = None
            best_val_labels: np.ndarray | None = None

            for candidate_context_variant in context_variants:
                export_model.set_context_tail_lengths(candidate_context_variant)
                candidate_val_logits, candidate_val_labels = evaluate_model(export_model, val_loader, device)
                candidate_val_probs = softmax(candidate_val_logits)
                candidate_threshold = choose_confidence_threshold(
                    candidate_val_probs,
                    candidate_val_labels,
                    min_selected=max(1, args.min_selected_trades),
                    threshold_min=args.confidence_search_min,
                    threshold_max=args.confidence_search_max,
                    threshold_steps=args.confidence_search_steps,
                )
                candidate_validation_gate = gate_metrics(candidate_val_labels, candidate_val_probs, candidate_threshold)
                candidate_score = chronos_context_score(
                    candidate_validation_gate,
                    min_selected=max(1, args.min_selected_trades),
                )
                log.info(
                    "Chronos auto-context | candidate=%s threshold=%.2f precision=%s coverage=%.4f trades=%d",
                    chronos_context_label(candidate_context_variant),
                    candidate_threshold,
                    format_metric(float(candidate_validation_gate["precision"])),
                    float(candidate_validation_gate["trade_coverage"]),
                    int(candidate_validation_gate["selected_trades"]),
                )
                if best_score is None or candidate_score > best_score:
                    best_score = candidate_score
                    selected_context_variant = candidate_context_variant
                    best_val_logits = candidate_val_logits
                    best_val_labels = candidate_val_labels

            if best_val_logits is None or best_val_labels is None:
                raise RuntimeError("Chronos auto-context search did not evaluate any candidates.")
            export_model.set_context_tail_lengths(selected_context_variant)
            log.info(
                "Chronos auto-context selected %s.",
                chronos_context_label(selected_context_variant),
            )
            val_logits, val_labels = best_val_logits, best_val_labels
        else:
            export_model.set_context_tail_lengths(selected_context_variant)
            if architecture == "chronos_bolt":
                log.info(
                    "Chronos context mode | selected=%s",
                    chronos_context_label(selected_context_variant),
                )
            val_logits, val_labels = evaluate_model(export_model, val_loader, device)

        model_backend = getattr(export_model, "backend_name", "chronos-bolt-zero-shot-close-barrier")
        test_logits, test_labels = evaluate_model(export_model, test_loader, device)
    else:
        scheduler = None
        feature_mean: np.ndarray | None = None
        feature_std: np.ndarray | None = None
        token_mean: np.ndarray | None = None
        token_std: np.ndarray | None = None
        minirocket_parameters = None
        train_sample_weights = make_sample_weights(y_train)

        if architecture == "minirocket":
            transform_batch_size = max(args.batch_size, DEFAULT_BATCH_SIZE)
            minirocket_parameters = fit_minirocket(
                x_train.transpose(0, 2, 1),
                num_features=args.minirocket_features,
                seed=42,
            )
            learning_rate = args.lr if args.lr > 0.0 else DEFAULT_MINIROCKET_LR
            weight_decay = args.weight_decay if args.weight_decay >= 0.0 else DEFAULT_MINIROCKET_WEIGHT_DECAY

            if use_multihead_attention:
                train_inputs = transform_sequence_tokens(
                    minirocket_parameters,
                    x_train,
                    batch_size=transform_batch_size,
                    device=device,
                )
                val_inputs = transform_sequence_tokens(
                    minirocket_parameters,
                    x_val,
                    batch_size=transform_batch_size,
                    device=device,
                )
                test_inputs = transform_sequence_tokens(
                    minirocket_parameters,
                    x_test,
                    batch_size=transform_batch_size,
                    device=device,
                )

                token_mean = train_inputs.mean(axis=0).astype(np.float32)
                token_std = np.where(train_inputs.std(axis=0) < 1e-6, 1.0, train_inputs.std(axis=0)).astype(np.float32)
                train_inputs = ((train_inputs - token_mean) / token_std).astype(np.float32, copy=False)
                val_inputs = ((val_inputs - token_mean) / token_std).astype(np.float32, copy=False)
                test_inputs = ((test_inputs - token_mean) / token_std).astype(np.float32, copy=False)

                training_model = MiniRocketMultiAttentionHead(
                    num_tokens=train_inputs.shape[1],
                    token_dim=train_inputs.shape[2],
                    n_classes=len(active_label_names),
                    model_dim=args.attention_dim,
                    num_heads=args.attention_heads,
                    num_layers=args.attention_layers,
                    dropout=args.attention_dropout,
                ).to(device)
                model_backend = "minirocket-multivariate-attention"
            else:
                train_inputs = transform_sequences(
                    minirocket_parameters,
                    x_train,
                    batch_size=transform_batch_size,
                    device=device,
                )
                val_inputs = transform_sequences(
                    minirocket_parameters,
                    x_val,
                    batch_size=transform_batch_size,
                    device=device,
                )
                test_inputs = transform_sequences(
                    minirocket_parameters,
                    x_test,
                    batch_size=transform_batch_size,
                    device=device,
                )

                feature_mean = train_inputs.mean(axis=0).astype(np.float32)
                feature_std = np.where(train_inputs.std(axis=0) < 1e-6, 1.0, train_inputs.std(axis=0)).astype(np.float32)
                train_inputs = ((train_inputs - feature_mean) / feature_std).astype(np.float32, copy=False)
                val_inputs = ((val_inputs - feature_mean) / feature_std).astype(np.float32, copy=False)
                test_inputs = ((test_inputs - feature_mean) / feature_std).astype(np.float32, copy=False)

                training_model = nn.Linear(train_inputs.shape[1], len(active_label_names)).to(device)
                model_backend = "minirocket-multivariate"

            optimizer = torch.optim.AdamW(training_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                min_lr=1e-8,
                patience=max(1, args.patience // 2),
            )
            train_loader = make_loader(
                train_inputs,
                y_train,
                args.batch_size,
                shuffle=True,
                sample_weights=train_sample_weights,
            )
            val_loader = make_loader(
                val_inputs,
                y_val,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
            test_loader = make_loader(
                test_inputs,
                y_test,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
        else:
            if architecture in {
                "ela",
                "fusion_lstm",
                "bilstm",
                "gru",
                "tcn",
                "tla",
                "legacy_lstm_attention",
                "gold_legacy",
                "gold_new",
            }:
                learning_rate = args.lr if args.lr > 0.0 else DEFAULT_SEQUENCE_LR
                weight_decay = args.weight_decay if args.weight_decay >= 0.0 else DEFAULT_SEQUENCE_WEIGHT_DECAY
                if architecture == "legacy_lstm_attention":
                    training_model = LegacyLSTMAttentionClassifier(
                        n_features=feature_count,
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "gold_legacy":
                    training_model = GoldLegacyLSTMAttentionClassifier(
                        n_features=feature_count,
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "gold_new":
                    training_model = GoldNewTemporalClassifier(
                        n_features=feature_count,
                        channels=max(16, min(64, args.sequence_hidden_size)),
                        hidden=max(32, min(64, args.sequence_hidden_size)),
                        dense_hidden=max(64, feature_count * 2),
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                        dropout=args.sequence_dropout,
                    ).to(device)
                elif architecture == "fusion_lstm":
                    training_model = FusionLSTMClassifier(
                        n_features=feature_count,
                        hidden=20,
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "tcn":
                    training_model = TCNClassifier(
                        n_features=feature_count,
                        channels=args.sequence_hidden_size,
                        hidden=max(args.sequence_hidden_size, feature_count * 4),
                        dropout=args.sequence_dropout,
                        n_layers=args.tcn_levels,
                        kernel_size=args.tcn_kernel_size,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "tla":
                    training_model = TemporalLSTMAttentionClassifier(
                        n_features=feature_count,
                        conv_channels=128,
                        lstm_hidden=args.sequence_hidden_size,
                        hidden=max(args.sequence_hidden_size, feature_count * 4),
                        dropout=args.sequence_dropout,
                        conv_kernel_size=5,
                        lstm_layers=args.sequence_layers,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                else:
                    recurrent_cell_type = "gru" if architecture == "gru" else "lstm"
                    recurrent_bidirectional = architecture == "bilstm"
                    backend_name = {
                        "ela": "ela-lstm-attention",
                        "bilstm": "bilstm-attention" if use_multihead_attention else "bilstm",
                        "gru": "gru-attention" if use_multihead_attention else "gru",
                    }[architecture]
                    training_model = RecurrentSequenceClassifier(
                        n_features=feature_count,
                        cell_type=recurrent_cell_type,
                        hidden_size=args.sequence_hidden_size,
                        hidden=max(args.sequence_hidden_size, feature_count * 4),
                        dropout=args.sequence_dropout,
                        num_layers=args.sequence_layers,
                        bidirectional=recurrent_bidirectional,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                        backend_name=backend_name,
                    ).to(device)
            else:
                learning_rate = args.lr if args.lr > 0.0 else DEFAULT_MAMBA_LR
                weight_decay = args.weight_decay if args.weight_decay >= 0.0 else DEFAULT_MAMBA_WEIGHT_DECAY
                if architecture == "castor":
                    training_model = CastorClassifier(
                        n_features=feature_count,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                else:
                    training_model = MambaLiteClassifier(
                        n_features=feature_count,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
            optimizer = torch.optim.AdamW(training_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            train_loader = make_loader(
                x_train,
                y_train,
                args.batch_size,
                shuffle=True,
                sample_weights=train_sample_weights,
            )
            val_loader = make_loader(
                x_val,
                y_val,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
            test_loader = make_loader(
                x_test,
                y_test,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
            model_backend = getattr(training_model, "backend_name", "portable-mamba-lite")

        class_weights = make_class_weights(y_train).to(device)
        if loss_mode == "cross-entropy":
            criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights).to(device)
        else:
            criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma).to(device)
        log.info(
            "Optimization | loss=%s lr=%.6g weight_decay=%.6g balanced_sampling=1 confidence_search=[%.2f, %.2f]x%d",
            loss_mode,
            learning_rate,
            weight_decay,
            args.confidence_search_min,
            args.confidence_search_max,
            args.confidence_search_steps,
        )

        best_state = None
        best_val_loss = float("inf")
        wait = 0

        for epoch in tqdm(range(args.epochs), desc="Training"):
            training_model.train()
            train_losses = []
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
                logits = training_model(xb.to(device))
                loss = criterion(logits, yb.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            val_logits, val_labels = evaluate_model(training_model, val_loader, device)
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
            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in training_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a valid checkpoint.")

        training_model.load_state_dict(best_state)
        training_model.to(device)

        val_logits, val_labels = evaluate_model(training_model, val_loader, device)
        test_logits, test_labels = evaluate_model(training_model, test_loader, device)

        if architecture == "minirocket":
            if minirocket_parameters is None:
                raise RuntimeError("MiniRocket export requested without fitted parameters.")
            if use_multihead_attention:
                export_model = MiniRocketClassifier(
                    parameters=minirocket_parameters,
                    n_classes=len(active_label_names),
                    token_mean=token_mean,
                    token_std=token_std,
                    head_type="multiattention",
                    attention_dim=args.attention_dim,
                    attention_heads=args.attention_heads,
                    attention_layers=args.attention_layers,
                    attention_dropout=args.attention_dropout,
                )
            else:
                export_model = MiniRocketClassifier(
                    parameters=minirocket_parameters,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    n_classes=len(active_label_names),
                    head_type="linear",
                )
            export_model.head.load_state_dict(training_model.state_dict())
        else:
            export_model = training_model
    val_probs = softmax(val_logits)
    selected_primary_confidence = choose_confidence_threshold(
        val_probs,
        val_labels,
        min_selected=max(1, args.min_selected_trades),
        threshold_min=args.confidence_search_min,
        threshold_max=args.confidence_search_max,
        threshold_steps=args.confidence_search_steps,
    )
    validation_gate = summarize_gate("validation", val_probs, val_labels, selected_primary_confidence)

    test_probs = softmax(test_logits)
    holdout_gate = summarize_gate("holdout", test_probs, test_labels, selected_primary_confidence)

    quality_gate_reasons: list[str] = []
    if int(validation_gate["selected_trades"]) < args.min_selected_trades:
        quality_gate_reasons.append(
            "validation selected trades "
            f"{int(validation_gate['selected_trades'])} < required {args.min_selected_trades}"
        )
    validation_precision = float(validation_gate["precision"])
    if not np.isfinite(validation_precision):
        quality_gate_reasons.append("validation selected-trade precision unavailable")
    elif validation_precision < args.min_trade_precision:
        quality_gate_reasons.append(
            f"validation selected-trade precision {validation_precision:.4f} < required {args.min_trade_precision:.4f}"
        )
    quality_gate_passed = len(quality_gate_reasons) == 0
    quality_gate_reason = "; ".join(quality_gate_reasons)
    deployed_primary_confidence = selected_primary_confidence
    if not quality_gate_passed:
        log.warning(
            "Model failed the live quality gate (%s). Keeping PRIMARY_CONFIDENCE=%.2f.",
            quality_gate_reason,
            deployed_primary_confidence,
        )

    if export_model is None:
        raise RuntimeError("Model export path was not initialized.")

    completed_at = datetime.now()
    model_dir_name = format_model_dir_name(
        value=completed_at,
        name=model_name,
        failed_quality_gate=not quality_gate_passed,
        symbol=SYMBOL,
    )
    model_dir = symbol_models_dir(SYMBOL) / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_dir = model_diagnostics_dir(model_dir)
    model_test_dir = model_dir / "tests"
    model_test_dir.mkdir(parents=True, exist_ok=True)
    archive_output_path = model_dir / "model.onnx"
    combined_model_config_path = model_dir / "config.mqh"

    export_model.eval()
    export_model.to("cpu")
    dummy = torch.randn(1, SEQ_LEN, feature_count)
    export_onnx_model(export_model, dummy, archive_output_path)

    model_config_text = (
        build_mql_config(
            project=project,
            median=median,
            iqr=iqr,
            primary_confidence=deployed_primary_confidence,
            use_atr_risk=use_atr_risk,
            use_fixed_time_bars=use_fixed_time_bars,
            architecture=architecture,
            use_multihead_attention=use_multihead_attention,
            feature_columns=feature_columns,
            feature_profile=feature_profile,
            use_extended_features=use_extended_features,
            use_fixed_tick_bars=use_fixed_tick_bars,
        )
        + "\n"
    )
    combined_model_config_path.write_text(model_config_text, encoding="utf-8")
    write_diagnostics(
        diagnostics_dir=diagnostics_dir,
        bars=bars,
        y_full=y,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        val_probs=val_probs,
        test_probs=test_probs,
        selected_primary_confidence=selected_primary_confidence,
        deployed_primary_confidence=deployed_primary_confidence,
        validation_gate=validation_gate,
        holdout_gate=holdout_gate,
        quality_gate_passed=quality_gate_passed,
        quality_gate_reason=quality_gate_reason,
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
        symbol=SYMBOL,
        model_backend=model_backend,
        loss_mode=loss_mode,
        focal_gamma=args.focal_gamma,
        model_config_text=model_config_text,
        feature_columns=feature_columns,
        feature_profile=feature_profile,
        point_size=point_size,
        fixed_move_price=fixed_move_price,
        use_fixed_tick_bars=use_fixed_tick_bars,
        tick_density=args.primary_tick_density,
    )
    if not archive_only:
        shutil.rmtree(ACTIVE_DIAGNOSTICS_DIR, ignore_errors=True)
        shutil.copytree(diagnostics_dir, ACTIVE_DIAGNOSTICS_DIR)
    ensure_default_test_config(model_test_dir, symbol=SYMBOL)

    set_live_model_reference(model_dir)

    if not archive_only and not args.skip_live_compile:
        runtime_paths = resolve_mt5_runtime(metaeditor_path_override=args.metaeditor_path)
        compile_log_path = compile_live_expert(runtime_paths, model_dir=model_dir)
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
        shutil.copy2(compile_log_path, diagnostics_dir / compile_log_path.name)
        log.info("Saved live compile log to %s", compile_log_path)
    log.info("Archived ONNX to %s", archive_output_path)
    log.info("Saved combined model config to %s", combined_model_config_path)
    log.info("Saved diagnostics to %s", diagnostics_dir)
    log.info("Archived model artifacts to %s", model_dir)
    log.info("Total runtime: %.2fs", time.time() - t0)
    print(model_dir.name)


if __name__ == "__main__":
    main()
