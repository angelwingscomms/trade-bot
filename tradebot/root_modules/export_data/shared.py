"""Compile and run the MT5 data-export scripts, then collect the resulting CSV."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from tradebot.config_io import load_define_file
from tradebot.root_modules.mt5_runtime import (
    PROJECT_DIR_NAME,
    build_metaeditor_compile_command,
    build_terminal_command,
    read_text_best_effort,
    resolve_mt5_runtime,
    runtime_env,
    stop_terminal_best_effort,
)
from tradebot.workspace import (
    ACTIVE_CONFIG_PATH,
    configured_symbol,
    symbol_default_config_path,
)

SCRIPT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_FILE = "market_ticks.csv"
COMPILE_LOG_PATH = SCRIPT_DIR / "mt5" / "logs" / "data.compile.log"
STARTUP_CONFIG_DIR = Path(tempfile.gettempdir()) / "mt5_export_configs"
DATA_PROFILE_SCRIPTS = {
    "default": SCRIPT_DIR / "mt5" / "scripts" / "data.mq5",
    "gold": SCRIPT_DIR / "mt5" / "scripts" / "data_gold.mq5",
}
