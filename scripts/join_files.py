from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: ensure the project root (parent of scripts/) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tradebot.root_modules.join_files as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

if __name__ == "__main__":
    main()
