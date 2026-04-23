from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tradebot.training as _impl
from tradebot.workspace import ROOT_DIR
from tradebot.workspace_parts.resolve_active_config_path import set_override_config_path
from tradebot.config_io import load_define_file

log = logging.getLogger("i.py")

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _get_config_keys(d: dict, prefix: str = "") -> set[str]:
    keys = set()
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.update(_get_config_keys(v, full_key))
            else:
                keys.add(full_key)
    return keys


def _warn_missing_configs(config_path: Path) -> None:
    default_yaml = ROOT_DIR / "config" / "default.yaml"
    if not default_yaml.exists():
        log.warning("Default config not found: %s", default_yaml)
        return

    default_values = load_define_file(default_yaml)

    if config_path.exists():
        user_values = load_define_file(config_path)
    else:
        user_values = {}

    required_keys = {
        "SYMBOL", "SEQ_LEN", "LABEL_TIMEOUT_BARS", "FEATURE_ATR_PERIOD",
        "FEATURE_ATR_RATIO_PERIOD", "FEATURE_BOLLINGER_PERIOD",
        "FEATURE_DONCHIAN_FAST_PERIOD", "FEATURE_DONCHIAN_SLOW_PERIOD",
        "FEATURE_RET_2_PERIOD", "FEATURE_RET_3_PERIOD", "FEATURE_RET_6_PERIOD",
        "FEATURE_RET_12_PERIOD", "FEATURE_RET_20_PERIOD", "FEATURE_RSI_FAST_PERIOD",
        "FEATURE_RSI_SLOW_PERIOD", "FEATURE_RV_LONG_PERIOD", "FEATURE_SMA_FAST_PERIOD",
        "FEATURE_SMA_MID_PERIOD", "FEATURE_SMA_SLOW_PERIOD", "FEATURE_SMA_SLOPE_SHIFT",
        "FEATURE_SMA_TREND_FAST_PERIOD", "FEATURE_SPREAD_Z_PERIOD", "FEATURE_STOCH_PERIOD",
        "FEATURE_STOCH_SMOOTH_PERIOD", "FEATURE_TICK_COUNT_PERIOD",
        "FEATURE_TICK_IMBALANCE_FAST_PERIOD", "FEATURE_TICK_IMBALANCE_SLOW_PERIOD",
        "TARGET_ATR_PERIOD", "RV_PERIOD", "RETURN_PERIOD", "IMBALANCE_MIN_TICKS",
        "IMBALANCE_EMA_SPAN", "USE_IMBALANCE_EMA_THRESHOLD",
        "USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD", "PRIMARY_BAR_SECONDS",
        "PRIMARY_TICK_DENSITY", "DEFAULT_FIXED_MOVE", "DEFAULT_FIXED_SL",
        "DEFAULT_FIXED_TP", "LABEL_SL_MULTIPLIER", "LABEL_TP_MULTIPLIER",
        "DEFAULT_SL_MULTIPLIER", "DEFAULT_TP_MULTIPLIER", "USE_ALL_WINDOWS",
        "DEFAULT_EPOCHS", "DEFAULT_BATCH_SIZE", "DEFAULT_MAX_TRAIN_WINDOWS",
        "DEFAULT_MAX_EVAL_WINDOWS", "DEFAULT_PATIENCE", "SEQUENCE_DROPOUT",
        "DEFAULT_LOSS_MODE",
    }

    missing = sorted(required_keys - user_values.keys())
    for key in missing:
        log.warning("Missing config: %s - default will be used", key)

    if missing:
        log.warning("Defaults will be used for %d configs not provided", len(missing))


def _override_from_argv() -> Path | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file (relative to config/)")
    args, _ = parser.parse_known_args()
    if args.config:
        config_path = ROOT_DIR / "config" / args.config
        if config_path.exists():
            set_override_config_path(config_path)
            return config_path
        else:
            config_yaml = config_path.with_suffix(".yaml")
            if config_yaml.exists():
                set_override_config_path(config_yaml)
                return config_yaml
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")
    return None


if __name__ == "__main__":
    _config_path = _override_from_argv() or (ROOT_DIR / "config" / "default.yaml")
    _warn_missing_configs(_config_path)
    main()
