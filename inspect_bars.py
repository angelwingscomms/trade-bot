"""Show how many bars the active `config.mqh` would build without training."""

from __future__ import annotations

from pathlib import Path

from nn import apply_shared_settings, build_market_bars
from tradebot.workspace import ACTIVE_CONFIG_PATH
from tradebot.project_config import resolve_active_project_config


def main() -> None:
    project = resolve_active_project_config(ACTIVE_CONFIG_PATH)
    apply_shared_settings(project.values, project=project, shared_config_path=project.config_path)

    values = project.values
    data_path = Path(str(values["DATA_FILE"]))
    if not data_path.is_absolute():
        data_path = ACTIVE_CONFIG_PATH.parent / data_path
    if not data_path.exists():
        raise FileNotFoundError(
            f"Tick CSV not found: {data_path}. Export data first or point DATA_FILE at an existing CSV."
        )

    bars, point_size = build_market_bars(
        data_path,
        use_fixed_time_bars=bool(values["USE_SECOND_BARS"]),
        use_fixed_tick_bars=bool(values["USE_FIXED_TICK_BARS"]),
        tick_density=int(values["PRIMARY_TICK_DENSITY"]),
        max_bars=int(values["MAX_BARS"]) if bool(values.get("USE_MAX_BARS", False)) else 0,
        require_gold_context=bool(values["USE_GOLD_CONTEXT"]) and not bool(values["USE_MINIMAL_FEATURE_SET"]),
    )
    print(f"config={ACTIVE_CONFIG_PATH}")
    print(f"symbol={values['SYMBOL']}")
    print(f"bars={len(bars)}")
    print(f"point_size={point_size:.8f}")


if __name__ == "__main__":
    main()
