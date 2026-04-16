"""Shared type definitions and stubs for Python ↔ MQL5 interop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

Scalar = bool | int | float | str

ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent

GOLD_CONTEXT_TICK_COLUMNS: Final[tuple[str, ...]] = (
    "usdx_bid",
    "usdjpy_bid",
)

LABEL_NAMES: Final[tuple[str, ...]] = ("HOLD", "BUY", "SELL")
LABEL_NAMES_BINARY: Final[tuple[str, ...]] = ("BUY", "SELL")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    symbols: Path
    diagnostics: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        if root is None:
            root = ROOT_DIR
        return cls(
            root=root,
            data=root / "data",
            symbols=root / "symbols",
            diagnostics=root / "diagnostics",
        )


@dataclass(frozen=True)
class BarMode:
    USE_FIXED_TIME_BARS: bool
    USE_FIXED_TICK_BARS: bool

    @property
    def is_time(self) -> bool:
        return self.USE_FIXED_TIME_BARS

    @property
    def is_tick(self) -> bool:
        return self.USE_FIXED_TICK_BARS

    @property
    def is_imbalance(self) -> bool:
        return not (self.USE_FIXED_TIME_BARS or self.USE_FIXED_TICK_BARS)
