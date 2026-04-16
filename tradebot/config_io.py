"""Helpers for reading and resolving MQL-style config files.

The repo stores user-editable configuration in `.mqh`/`.config` files so the
same values can be consumed by Python training code and the MQL5 runtime.
This module keeps that bridge in one place.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final


Scalar = bool | int | float | str

DEFINE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*#define\s+([A-Z0-9_]+)\s+(.+?)\s*$")
SAFE_SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_symbol(symbol: str) -> str:
    """Return a filesystem-safe, lower-case symbol directory name."""

    cleaned = SAFE_SYMBOL_PATTERN.sub("_", symbol.strip())
    return (cleaned or "unknown").lower()


def parse_define_value(raw_value: str, known_values: dict[str, Scalar]) -> Scalar:
    """Parse a single `#define` literal or expression from an MQL config file."""

    value = raw_value.split("//", 1)[0].strip()
    if value.endswith("f"):
        value = value[:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    safe_names = dict(known_values)
    safe_names.update({"true": True, "false": False})
    return eval(value, {"__builtins__": {}}, safe_names)


def load_define_file(path: Path) -> dict[str, Scalar]:
    """Load all `#define` entries from a config-style file."""

    values: dict[str, Scalar] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = DEFINE_PATTERN.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        values[name] = parse_define_value(raw_value, values)
    return values


def read_text_best_effort(path: Path) -> str:
    """Read text files that may be written as UTF-8, UTF-16, or cp1252."""

    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def render_define_value(value: Scalar) -> str:
    """Render a Python scalar back into a valid MQL `#define` literal."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)

