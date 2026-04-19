from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so tradebot/common are importable
# regardless of which directory the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tradebot.root_modules.test_cli as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

if __name__ == "__main__":
    main()
