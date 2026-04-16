"""Bar construction utilities shared between Python training and MQL5 live."""

from __future__ import annotations

import numpy as np


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


def build_primary_bar_ids(
    tick_signs: np.ndarray,
    imbalance_min_ticks: int,
    imbalance_ema_span: int,
    use_imbalance_ema_threshold: bool,
    use_imbalance_min_ticks_div3_threshold: bool,
) -> np.ndarray:
    if use_imbalance_min_ticks_div3_threshold:
        base_threshold = max(2.0, float(max(2, imbalance_min_ticks // 3)))
    else:
        base_threshold = max(2.0, float(imbalance_min_ticks))
    expected_abs_theta = base_threshold
    bar_ids = np.empty(len(tick_signs), dtype=np.int64)
    current_bar = 0
    ticks_in_bar = 0
    theta = 0.0
    alpha = 2.0 / (max(1, imbalance_ema_span) + 1.0)

    for i, sign in enumerate(tick_signs):
        bar_ids[i] = current_bar
        ticks_in_bar += 1
        theta += float(sign)
        threshold = expected_abs_theta if use_imbalance_ema_threshold else base_threshold
        if ticks_in_bar >= imbalance_min_ticks and abs(theta) >= threshold:
            if use_imbalance_ema_threshold:
                observed = max(2.0, abs(theta))
                expected_abs_theta = (1.0 - alpha) * expected_abs_theta + alpha * observed
            current_bar += 1
            ticks_in_bar = 0
            theta = 0.0

    return bar_ids


def build_time_bar_ids(time_msc: np.ndarray, bar_duration_ms: int) -> np.ndarray:
    return time_msc // bar_duration_ms


def build_tick_bar_ids(tick_count: int, tick_density: int) -> np.ndarray:
    return np.arange(tick_count, dtype=np.int64) // int(tick_density)


def infer_point_size_from_ticks(df_ticks, max_samples: int = 200_000) -> float:
    prices = np.concatenate([
        df_ticks["bid"].to_numpy(dtype=np.float64, copy=False),
        df_ticks["ask"].to_numpy(dtype=np.float64, copy=False),
    ])
    prices = prices[np.isfinite(prices)]
    if len(prices) == 0:
        return 1.0

    sample = np.round(prices[:max_samples], 8)
    unique_prices = np.unique(sample)
    if len(unique_prices) < 2:
        return 1.0

    scaled = np.rint(unique_prices * 1e8).astype(np.int64)
    diffs = np.diff(scaled)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0

    gcd_points = int(np.gcd.reduce(diffs[:min(len(diffs), 50_000)]))
    point_size = gcd_points / 1e8 if gcd_points > 0 else 1.0
    return float(point_size if point_size > 0.0 else 1.0)
