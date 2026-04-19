from __future__ import annotations

from common.past_dir_features import parse_past_dir_spec

from .shared import *  # noqa: F401,F403


def resolve_feature_profile(
    values: dict[str, Scalar], feature_columns: tuple[str, ...]
) -> str:
    """Return a short human-readable label for diagnostics and reports."""

    if bool(values.get("USE_MAIN_FEATURE_SET", False)):
        return "main"
    if bool(values.get("USE_MINIMAL_FEATURE_SET", False)):
        # Check if any past_dir features were also appended
        has_past_dir = any(parse_past_dir_spec(n) is not None for n in feature_columns)
        return "minimal+past_dir" if has_past_dir else "minimal"
    enabled_gold = any(name in feature_columns for name in GOLD_CONTEXT_FEATURE_COLUMNS)
    # past_dir_* names count as extra features for profile purposes
    enabled_extra = any(
        name in feature_columns for name in EXTRA_FEATURE_COLUMNS
    ) or any(parse_past_dir_spec(n) is not None for n in feature_columns)
    if enabled_gold and enabled_extra:
        return "full"
    if enabled_extra:
        return "custom-extra"
    if enabled_gold:
        return "gold-context"
    return "minimal-plus-required"
