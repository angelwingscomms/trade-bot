"""Workspace layout and live-model deployment helpers.

This module centralizes the repository structure so the trainer, tester,
exporter, and MT5 compilation flow all agree on where configs, archived
models, and live sources live.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from tradebot.config_io import load_define_file, read_text_best_effort, sanitize_symbol
from tradebot.root_modules.mt5_runtime import (
    DEFAULT_WINDOWS_METAEDITOR_PATH,
    PROJECT_DIR_NAME,
    Mt5RuntimePaths,
    build_metaeditor_compile_command,
    ensure_runtime_dirs,
    runtime_env,
    to_windows_path,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
SYMBOLS_DIR = ROOT_DIR / "symbols"
ACTIVE_CONFIG_PATH = ROOT_DIR / "config" / "active.mqh"
ACTIVE_CONFIG_POINTER = ROOT_DIR / "config" / ".active_config"
ACTIVE_DIAGNOSTICS_DIR = ROOT_DIR / "diagnostics"
LIVE_MQ5_PATH = ROOT_DIR / "live" / "live.mq5"
LIVE_EX5_PATH = ROOT_DIR / "mt5" / "compiled" / "live.ex5"
LIVE_COMPILE_LOG_PATH = ROOT_DIR / "mt5" / "logs" / "live.compile.log"

DEFAULT_METAEDITOR_PATH = DEFAULT_WINDOWS_METAEDITOR_PATH
DEFAULT_MODEL_STAMP_FORMAT = "%m%d-%H%M%S"
MODEL_STAMP_FORMATS = (
    DEFAULT_MODEL_STAMP_FORMAT,
    "%m%d-%H%M_%S",
)
MODEL_STAMP_PREFIX_PATTERN = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}\d{2}\d{2})(?:-|$)"
)
MODEL_STAMP_SUFFIX_PATTERN = re.compile(
    r"(?P<stamp>\d{4}-\d{2}\d{2}\d{2})(?:-fail)?$"
)
COMPILE_RESULT_PATTERN = re.compile(r"Result:\s+(\d+)\s+errors?,\s+(\d+)\s+warnings?")

MODEL_FILE_NAME = "model.onnx"
MODEL_CONFIG_NAME = "config.mqh"
DEFAULT_TEST_CONFIG_NAME = "backtest_config.json"
LIVE_MODEL_BLOCK_BEGIN = "// @active-model-reference begin"
LIVE_MODEL_BLOCK_END = "// @active-model-reference end"
LIVE_MODEL_BLOCK_PATTERN = re.compile(
    rf"{re.escape(LIVE_MODEL_BLOCK_BEGIN)}.*?{re.escape(LIVE_MODEL_BLOCK_END)}",
    re.DOTALL,
)
RESOURCE_PATH_MAX_CHARS = 63
