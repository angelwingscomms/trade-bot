"""High-level config resolution for training and live feature selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.features import (
    EXTRA_FEATURE_COLUMNS,
    GOLD_CONTEXT_FEATURE_COLUMNS,
    MAIN_FEATURE_COLUMNS,
    MAIN_GOLD_CONTEXT_FEATURE_COLUMNS,
    MINIMAL_FEATURE_COLUMNS,
    feature_macro_name,
    feature_switch_name,
    max_feature_lookback,
    minimal_feature_switch_name,
)
from tradebot.config_io import Scalar, load_define_file


ROOT_DIR = Path(__file__).resolve().parents[2]
