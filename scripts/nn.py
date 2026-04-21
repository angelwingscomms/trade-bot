from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tradebot.training as _impl
from tradebot.workspace import ROOT_DIR
from tradebot.workspace_parts.resolve_active_config_path import set_override_config_path

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _override_from_argv() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file (relative to config/)")
    args, _ = parser.parse_known_args()
    if args.config:
        config_path = ROOT_DIR / "config" / args.config
        if config_path.exists():
            set_override_config_path(config_path)
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")


if __name__ == "__main__":
    _override_from_argv()
    main()
