from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (parent of scripts/) is on sys.path so that
# `tradebot` and `common` are importable regardless of cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tradebot.training as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

if __name__ == "__main__":
    main()
