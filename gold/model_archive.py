from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
ACTIVE_ONNX_PATH = SCRIPT_DIR / "gold_mamba.onnx"
ACTIVE_MODEL_CONFIG_PATH = SCRIPT_DIR / "gold_model_config.mqh"
ACTIVE_SHARED_CONFIG_PATH = SCRIPT_DIR / "gold_shared_config.mqh"
ACTIVE_DIAGNOSTICS_DIR = SCRIPT_DIR / "diagnostics"
LIVE_MQ5_PATH = SCRIPT_DIR / "live.mq5"
LIVE_EX5_PATH = SCRIPT_DIR / "live.ex5"
LIVE_COMPILE_LOG_PATH = SCRIPT_DIR / "live.compile.log"
DEFAULT_METAEDITOR_PATH = Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe")
DEFAULT_MODEL_STAMP_FORMAT = "%d__%m__%Y-%H_%M__%S"
DEFAULT_TEST_CONFIG_NAME = "backtest_config.json"
MODEL_STAMP_PATTERN = re.compile(r"^\d{2}__\d{2}__\d{4}-\d{2}_\d{2}__\d{2}$")
MODEL_STAMP_FORMATS = (
    DEFAULT_MODEL_STAMP_FORMAT,
    "%d__%m__%Y-%H_%M_%S",
    "%Y%m%d_%H%M%S",
)
COMPILE_RESULT_PATTERN = re.compile(r"Result:\s+(\d+)\s+errors?,\s+(\d+)\s+warnings?")


def format_model_stamp(value: datetime | None = None) -> str:
    return (value or datetime.now()).strftime(DEFAULT_MODEL_STAMP_FORMAT)


def parse_model_stamp(value: str) -> datetime:
    for stamp_format in MODEL_STAMP_FORMATS:
        try:
            return datetime.strptime(value, stamp_format)
        except ValueError:
            continue
    raise ValueError(
        "Model date must match the model folder name, for example 03__04__2026-06_45__00."
    )


def resolve_model_dir(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        model_dir = candidate
    else:
        model_dir = MODELS_DIR / value
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")
    parse_model_stamp(model_dir.name)
    return model_dir


def default_test_config() -> dict[str, int | float | str]:
    return {
        "month": "",
        "from_date": "",
        "to_date": "",
        "symbol": "XAUUSD",
        "deposit": 10000.0,
        "currency": "USD",
        "leverage": "1:2000",
        "timeout_seconds": 600,
        "retries": 1,
    }


def load_test_config(path: Path) -> dict[str, int | float | str]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_test_config(path: Path, data: dict[str, int | float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_default_test_config(config_dir: Path) -> Path:
    config_path = config_dir / DEFAULT_TEST_CONFIG_NAME
    if not config_path.exists():
        write_test_config(config_path, default_test_config())
    return config_path


def read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def resolve_metaeditor_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    which_path = shutil.which(path_str)
    if which_path:
        return Path(which_path)

    raise FileNotFoundError(
        f"MetaEditor executable not found at '{path_str}'. Pass --metaeditor-path with the correct location."
    )


def compile_live_expert(metaeditor_path: Path) -> Path:
    command = [
        str(metaeditor_path),
        f"/compile:{LIVE_MQ5_PATH}",
        f"/log:{LIVE_COMPILE_LOG_PATH}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_text = read_text_best_effort(LIVE_COMPILE_LOG_PATH) if LIVE_COMPILE_LOG_PATH.exists() else ""
    if not log_text and completed.stdout:
        log_text = completed.stdout

    result_match = COMPILE_RESULT_PATTERN.search(log_text)
    if result_match:
        errors = int(result_match.group(1))
        warnings = int(result_match.group(2))
        if errors > 0:
            raise RuntimeError(
                f"live.mq5 compile failed with {errors} errors and {warnings} warnings. "
                f"Check log at {LIVE_COMPILE_LOG_PATH}."
            )
        return LIVE_COMPILE_LOG_PATH

    if completed.returncode != 0:
        raise RuntimeError(
            f"MetaEditor returned exit code {completed.returncode} while compiling {LIVE_MQ5_PATH}.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    raise RuntimeError(f"Could not confirm live.mq5 compile status. Check log at {LIVE_COMPILE_LOG_PATH}.")
