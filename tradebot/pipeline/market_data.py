"""Bar building and label generation for the training pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.bars import build_primary_bar_ids, build_tick_bar_ids, build_time_bar_ids, compute_tick_signs, infer_point_size_from_ticks
from common.types import GOLD_CONTEXT_TICK_COLUMNS


log = logging.getLogger("nn")


def fixed_move_price_distance(fixed_move_points: float, point_size: float) -> float:
    return float(fixed_move_points) * float(point_size)


def build_market_bars(
    csv_path: Path,
    *,
    use_fixed_time_bars: bool,
    use_fixed_tick_bars: bool,
    tick_density: int,
    max_bars: int,
    bar_duration_ms: int,
    imbalance_min_ticks: int,
    imbalance_ema_span: int,
    use_imbalance_ema_threshold: bool,
    use_imbalance_min_ticks_div3_threshold: bool,
    require_gold_context: bool = False,
) -> tuple[pd.DataFrame, float]:
    t0 = time.time()
    chunks: list[pd.DataFrame] = []
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
        "chunksize": 50_000,
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
        df["usdx_bid"] = df["usdx"] if "usdx" in df.columns else np.nan
    if "usdjpy_bid" not in df.columns:
        df["usdjpy_bid"] = df["usdjpy"] if "usdjpy" in df.columns else np.nan
    if require_gold_context:
        missing_columns = [name for name in GOLD_CONTEXT_TICK_COLUMNS if name not in df.columns]
        if missing_columns:
            raise ValueError(
                "Gold-context training requires auxiliary columns "
                f"{missing_columns} in {csv_path}. Re-export gold ticks first."
            )
        empty_columns = [name for name in GOLD_CONTEXT_TICK_COLUMNS if df[name].notna().sum() == 0]
        if empty_columns:
            raise ValueError(
                "Gold-context training found empty auxiliary columns "
                f"{empty_columns} in {csv_path}. Re-export gold ticks first."
            )

    point_size = infer_point_size_from_ticks(df)
    df["tick_sign"] = compute_tick_signs(df["bid"].to_numpy(dtype=np.float64, copy=False))
    df["spread"] = df["ask"] - df["bid"]

    if use_fixed_tick_bars:
        df["bar_id"] = build_tick_bar_ids(len(df), tick_density)
    elif use_fixed_time_bars:
        if bar_duration_ms <= 0:
            raise ValueError("PRIMARY_BAR_SECONDS must be positive.")
        df["bar_id"] = build_time_bar_ids(df["time_msc"].to_numpy(dtype=np.int64, copy=False), bar_duration_ms)
    else:
        df["bar_id"] = build_primary_bar_ids(
            df["tick_sign"].to_numpy(dtype=np.int8, copy=False),
            imbalance_min_ticks=imbalance_min_ticks,
            imbalance_ema_span=imbalance_ema_span,
            use_imbalance_ema_threshold=use_imbalance_ema_threshold,
            use_imbalance_min_ticks_div3_threshold=use_imbalance_min_ticks_div3_threshold,
        )

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
            spread_mean=("spread", "mean"),
            time_open=("time_msc", "first"),
            time_close=("time_msc", "last"),
            usdx_bid=("usdx_bid", "last"),
            usdjpy_bid=("usdjpy_bid", "last"),
        )
        .reset_index(drop=True)
    )

    if max_bars > 0 and len(grouped) > max_bars:
        grouped = grouped.iloc[:max_bars].reset_index(drop=True)
        log.info("Capped bars to %d rows.", max_bars)

    if use_fixed_tick_bars:
        log.info(
            "Built %d bars in %.2fs using fixed %d-tick bars | point_size=%.8f",
            len(grouped),
            time.time() - t0,
            tick_density,
            point_size,
        )
    elif use_fixed_time_bars:
        log.info(
            "Built %d bars in %.2fs using fixed %dms bars | point_size=%.8f",
            len(grouped),
            time.time() - t0,
            bar_duration_ms,
            point_size,
        )
    else:
        log.info(
            "Built %d bars in %.2fs using imbalance bars min_ticks=%d span=%d | point_size=%.8f",
            len(grouped),
            time.time() - t0,
            imbalance_min_ticks,
            imbalance_ema_span,
            point_size,
        )
    return grouped, point_size


def get_triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    use_atr_risk: bool,
    fixed_move_price: float,
    target_horizon: int,
    target_atr_period: int,
    label_tp_multiplier: float,
    label_sl_multiplier: float,
) -> np.ndarray:
    close = bars["close"].to_numpy(dtype=np.float64, copy=False)
    high = bars["high"].to_numpy(dtype=np.float64, copy=False)
    low = bars["low"].to_numpy(dtype=np.float64, copy=False)
    spread = bars["spread"].to_numpy(dtype=np.float64, copy=False)
    ask_high = bars["ask_high"].to_numpy(dtype=np.float64, copy=False)
    ask_low = bars["ask_low"].to_numpy(dtype=np.float64, copy=False)
    atr_target = None
    if use_atr_risk:
        from tradebot.pipeline.feature_builder import wilder_atr

        atr_target = wilder_atr(
            bars["high"],
            bars["low"],
            bars["close"],
            period=target_atr_period,
        ).to_numpy(dtype=np.float64, copy=False)

    labels = np.zeros(len(bars), dtype=np.int64)
    for i in range(len(bars) - target_horizon):
        long_entry = close[i] + spread[i]
        short_entry = close[i]
        if use_atr_risk:
            vol = atr_target[i]
            if not np.isfinite(vol) or vol <= 0.0:
                continue
            long_tp = long_entry + label_tp_multiplier * vol
            long_sl = long_entry - label_sl_multiplier * vol
            short_tp = short_entry - label_tp_multiplier * vol
            short_sl = short_entry + label_sl_multiplier * vol
        else:
            long_tp = long_entry + fixed_move_price
            long_sl = long_entry - fixed_move_price
            short_tp = short_entry - fixed_move_price
            short_sl = short_entry + fixed_move_price

        long_result = 0
        short_result = 0
        for j in range(i + 1, i + target_horizon + 1):
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
