"""High-level config resolution for training and live feature selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.features import (
    EXTRA_FEATURE_COLUMNS,
    GOLD_CONTEXT_FEATURE_COLUMNS,
    MAIN_FEATURE_COLUMNS,
    MINIMAL_FEATURE_COLUMNS,
    feature_macro_name,
    feature_switch_name,
    max_feature_lookback,
    minimal_feature_switch_name,
)
from tradebot.config_io import Scalar, load_define_file


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ResolvedProjectConfig:
    """Fully resolved config values used by the Python pipeline."""

    config_path: Path
    architecture_config_path: Path | None
    values: dict[str, Scalar]
    architecture: str
    feature_columns: tuple[str, ...]
    feature_profile: str


def resolve_architecture(values: dict[str, Scalar]) -> str:
    """Return the configured model architecture string."""

    architecture = str(values.get("MODEL_ARCHITECTURE", "mamba")).strip().lower()
    if not architecture:
        raise ValueError("MODEL_ARCHITECTURE cannot be empty.")
    return architecture


def _feature_enabled(values: dict[str, Scalar], feature_name: str) -> bool:
    return bool(values.get(feature_switch_name(feature_name), False))


def _minimal_feature_enabled(values: dict[str, Scalar], feature_name: str) -> bool:
    return bool(values.get(minimal_feature_switch_name(feature_name), feature_name in MINIMAL_FEATURE_COLUMNS))


def resolve_feature_columns(values: dict[str, Scalar], architecture: str) -> tuple[str, ...]:
    """Resolve the exact ordered feature list used for training and export."""

    if architecture == "chronos_bolt":
        return MINIMAL_FEATURE_COLUMNS

    if bool(values.get("USE_MAIN_FEATURE_SET", False)):
        return MAIN_FEATURE_COLUMNS

    if bool(values.get("USE_MINIMAL_FEATURE_SET", False)):
        selected = tuple(
            feature_name
            for feature_name in MINIMAL_FEATURE_COLUMNS
            if _minimal_feature_enabled(values, feature_name)
        )
        if not selected:
            raise ValueError("USE_MINIMAL_FEATURE_SET is true but no minimal features are enabled.")
        return selected

    selected = list(MINIMAL_FEATURE_COLUMNS)
    if bool(values.get("USE_GOLD_CONTEXT", False)):
        for feature_name in GOLD_CONTEXT_FEATURE_COLUMNS:
            if _feature_enabled(values, feature_name):
                selected.append(feature_name)
    for feature_name in EXTRA_FEATURE_COLUMNS:
        if _feature_enabled(values, feature_name):
            selected.append(feature_name)
    return tuple(selected)


def resolve_feature_profile(values: dict[str, Scalar], feature_columns: tuple[str, ...]) -> str:
    """Return a short human-readable label for diagnostics and reports."""

    if bool(values.get("USE_MAIN_FEATURE_SET", False)):
        return "main"
    if bool(values.get("USE_MINIMAL_FEATURE_SET", False)):
        return "minimal"
    enabled_gold = any(name in feature_columns for name in GOLD_CONTEXT_FEATURE_COLUMNS)
    enabled_extra = any(name in feature_columns for name in EXTRA_FEATURE_COLUMNS)
    if enabled_gold and enabled_extra:
        return "full"
    if enabled_extra:
        return "custom-extra"
    if enabled_gold:
        return "gold-context"
    return "minimal-plus-required"


def config_path_value(values: dict[str, Scalar], key: str) -> Path | None:
    """Resolve an optional repo-relative path from the config values."""

    raw = str(values.get(key, "")).strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else ROOT_DIR / path


def default_data_file(symbol: str) -> str:
    """Return the default CSV path for a symbol."""

    return f"data/{symbol.upper()}/ticks.csv"


def resolve_active_project_config(config_path: Path) -> ResolvedProjectConfig:
    """Load the root config plus any optional architecture-only overlay."""

    values = load_define_file(config_path)
    architecture_config_path = config_path_value(values, "ARCHITECTURE_CONFIG")
    if architecture_config_path is not None:
        architecture_values = load_define_file(architecture_config_path)
        values = {**values, **architecture_values}

    symbol = str(values.get("SYMBOL", "XAUUSD")).strip() or "XAUUSD"
    if not str(values.get("DATA_FILE", "")).strip():
        values["DATA_FILE"] = default_data_file(symbol)

    architecture = resolve_architecture(values)
    feature_columns = resolve_feature_columns(values, architecture=architecture)
    feature_profile = resolve_feature_profile(values, feature_columns)
    return ResolvedProjectConfig(
        config_path=config_path,
        architecture_config_path=architecture_config_path,
        values=values,
        architecture=architecture,
        feature_columns=feature_columns,
        feature_profile=feature_profile,
    )


__all__ = [
    "EXTRA_FEATURE_COLUMNS",
    "GOLD_CONTEXT_FEATURE_COLUMNS",
    "MAIN_FEATURE_COLUMNS",
    "MINIMAL_FEATURE_COLUMNS",
    "ResolvedProjectConfig",
    "config_path_value",
    "default_data_file",
    "feature_macro_name",
    "feature_switch_name",
    "max_feature_lookback",
    "minimal_feature_switch_name",
    "resolve_active_project_config",
    "resolve_architecture",
    "resolve_feature_columns",
    "resolve_feature_profile",
]
