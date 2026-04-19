"""Utilities for dynamic PAST_DIR_* price-direction features.

Config convention
-----------------
Users add one or more defines to their .config file:

    #define PAST_DIR_5400_S true   ; price change over 5400 seconds ago
    #define PAST_DIR_200_T  true   ; price change over 200 ticks ago

The suffix  _S  means the lookback is measured in seconds (wall-clock).
The suffix  _T  means the lookback is measured in bars (ticks in the bar
sense, i.e. primary-bar count).

Naming
------
Feature name  : past_dir_5400_s  /  past_dir_200_t
Config switch : PAST_DIR_5400_S  /  PAST_DIR_200_T
MQL macro     : PAST_DIR_5400_S  /  PAST_DIR_200_T  (same as switch)
MQL idx macro : FEATURE_IDX_PAST_DIR_5400_S  /  FEATURE_IDX_PAST_DIR_200_T

Value encoding
--------------
value = tanh(log(close_now / close_then))

tanh maps any real log-return into (-1, 1):
  - strongly negative  -> near -1  (price fell a lot)
  - zero               -> 0        (no change)
  - strongly positive  -> near +1  (price rose a lot)

This is superior to a linear [0, 1] mapping because:
  1. It is symmetric around zero.
  2. It saturates gracefully for extreme moves.
  3. Networks with tanh activations naturally work in this range.
"""

from __future__ import annotations

import re
from typing import Final

_PAST_DIR_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^PAST_DIR_(\d+)_(S|T)$", re.IGNORECASE
)


def parse_past_dir_features(values: dict) -> list[str]:
    """Return sorted list of enabled PAST_DIR feature names from config values.

    Only entries whose value is truthy (True / 1 / "true") are returned.
    """
    results: list[str] = []
    for key, val in values.items():
        m = _PAST_DIR_PATTERN.match(str(key))
        if m is None:
            continue
        # Accept bool True, int 1, or string "true" / "1"
        if isinstance(val, bool):
            enabled = val
        elif isinstance(val, int):
            enabled = bool(val)
        else:
            enabled = str(val).strip().lower() in ("true", "1")
        if not enabled:
            continue
        n = int(m.group(1))
        unit = m.group(2).upper()
        results.append(f"past_dir_{n}_{unit.lower()}")
    return sorted(results, key=_sort_key)


def parse_past_dir_spec(feature_name: str) -> tuple[int, str] | None:
    """Parse a feature name like 'past_dir_5400_s' into (5400, 'S').

    Returns None if the name does not match the pattern.
    """
    m = _PAST_DIR_PATTERN.match(feature_name.upper())
    if m is None:
        return None
    return int(m.group(1)), m.group(2).upper()


def past_dir_lookback_bars(feature_name: str, bar_seconds: int) -> int:
    """Return the number of bars needed as lookback for this feature.

    For _S features the lookback is ceil(seconds / bar_seconds).
    For _T features the lookback is the tick count directly (it is already
    in bar units).

    bar_seconds is only used for _S features; pass 0 or any value for _T.
    """
    spec = parse_past_dir_spec(feature_name)
    if spec is None:
        raise ValueError(f"Not a PAST_DIR feature: {feature_name!r}")
    n, unit = spec
    if unit == "T":
        return n
    # _S: convert seconds to bars, rounding up
    if bar_seconds <= 0:
        # Cannot convert without bar duration; return n as a safe upper bound
        return n
    import math

    return math.ceil(n / bar_seconds)


def _sort_key(name: str) -> tuple[str, int]:
    spec = parse_past_dir_spec(name)
    if spec is None:
        return ("", 0)
    n, unit = spec
    return (unit, n)
