from __future__ import annotations

from .shared import *  # noqa: F401,F403


def build_live_model_reference_block(model_dir: Path) -> str:
    """Build the live.mq5 include/resource block for the active model."""

    try:
        relative_model_dir = model_dir.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Model directory must live under {ROOT_DIR}: {model_dir}"
        ) from exc

    symbol_value = sanitize_symbol(model_dir.parent.parent.name).upper()
    version = model_dir.name
    include_dir = relative_model_dir.as_posix()
    resource_literal = _resource_literal_for_relative_model_dir(relative_model_dir)
    return "\n".join(
        [
            LIVE_MODEL_BLOCK_BEGIN,
            f'#define ACTIVE_MODEL_SYMBOL "{symbol_value}"',
            f'#define ACTIVE_MODEL_VERSION "{version}"',
            f'#include "../{include_dir}/{MODEL_CONFIG_NAME}"',
            f'#resource "..\\{resource_literal}" as uchar model_buffer[]',
            LIVE_MODEL_BLOCK_END,
        ]
    )
