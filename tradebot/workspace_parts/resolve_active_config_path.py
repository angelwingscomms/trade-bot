from __future__ import annotations

from .shared import *  # noqa: F401,F403

_OVERRIDE_CONFIG_PATH: Path | None = None


def set_override_config_path(path: Path | None) -> None:
    global _OVERRIDE_CONFIG_PATH
    _OVERRIDE_CONFIG_PATH = path


def resolve_active_config_path() -> Path:
    """Return the config file pointed to by the pointer file, or the default config.

    If `.active_config` exists in ROOT_DIR, its first line is read as the path
    to the active config (absolute or relative to ROOT_DIR). Otherwise, falls
    back to `config.mqh`.

    Can be overridden via `set_override_config_path()`.
    """
    global _OVERRIDE_CONFIG_PATH
    if _OVERRIDE_CONFIG_PATH is not None:
        return _OVERRIDE_CONFIG_PATH
    if ACTIVE_CONFIG_POINTER.exists():
        raw = ACTIVE_CONFIG_POINTER.read_text(encoding="utf-8").strip()
        if raw:
            path = Path(raw.strip())
            if path.is_absolute():
                return path
            resolved = (ROOT_DIR / path).resolve()
            if resolved.exists():
                return resolved
    return ACTIVE_CONFIG_PATH
