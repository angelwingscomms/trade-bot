from __future__ import annotations

from .shared import *  # noqa: F401,F403

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
    for i in range(len(bars)):
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
        # Scan forward up to LABEL_TIMEOUT_BARS bars
        max_j = min(i + LABEL_TIMEOUT_BARS + 1, len(bars))
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
