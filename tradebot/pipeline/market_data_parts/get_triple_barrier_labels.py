from __future__ import annotations

from .shared import *  # noqa: F401,F403


def get_triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    use_atr_risk: bool,
    fixed_move_price: float,
    label_timeout_bars: int,
    target_atr_period: int,
    label_tp_multiplier: float,
    label_sl_multiplier: float,
    fixed_sl_price: float | None = None,
    fixed_tp_price: float | None = None,
) -> np.ndarray:
    """Generate triple-barrier labels.

    Scans forward up to label_timeout_bars bars. If neither TP nor SL is hit,
    labels the bar as 0 (hold).

    In ATR mode: SL/TP are label_sl_multiplier/label_tp_multiplier * ATR.
    In fixed mode: SL uses fixed_sl_price (falls back to fixed_move_price),
                   TP uses fixed_tp_price (falls back to fixed_move_price).
    """
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

    _fixed_sl = fixed_sl_price if fixed_sl_price is not None else fixed_move_price
    _fixed_tp = fixed_tp_price if fixed_tp_price is not None else fixed_move_price

    labels = np.zeros(len(bars), dtype=np.int64)
    for i in range(len(bars)):
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
            long_tp = long_entry + _fixed_tp
            long_sl = long_entry - _fixed_sl
            short_tp = short_entry - _fixed_tp
            short_sl = short_entry + _fixed_sl

        long_result = 0
        short_result = 0
        # Scan forward up to label_timeout_bars bars for TP/SL hits
        max_j = min(i + label_timeout_bars + 1, len(bars))
        for j in range(i + 1, max_j):
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
