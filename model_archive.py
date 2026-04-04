from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from mt5_runtime import (
    DEFAULT_WINDOWS_METAEDITOR_PATH,
    Mt5RuntimePaths,
    build_metaeditor_compile_command,
    ensure_runtime_dirs,
    resolve_metaeditor_path,
    runtime_env,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
ACTIVE_ONNX_PATH = SCRIPT_DIR / "model.onnx"
ACTIVE_MODEL_CONFIG_PATH = SCRIPT_DIR / "model_config.mqh"
ACTIVE_SHARED_CONFIG_PATH = SCRIPT_DIR / "shared_config.mqh"
ACTIVE_DIAGNOSTICS_DIR = SCRIPT_DIR / "diagnostics"
LIVE_MQ5_PATH = SCRIPT_DIR / "live.mq5"
LIVE_EX5_PATH = SCRIPT_DIR / "live.ex5"
LIVE_COMPILE_LOG_PATH = SCRIPT_DIR / "live.compile.log"
DEFAULT_METAEDITOR_PATH = DEFAULT_WINDOWS_METAEDITOR_PATH
DEFAULT_MODEL_STAMP_FORMAT = "%d_%m_%Y-%H_%M__%S"
MODEL_STAMP_PATTERN = re.compile(r"^\d{2}_\d{2}_\d{4}-\d{2}_\d{2}__\d{2}$")
MODEL_STAMP_FORMATS = (
    DEFAULT_MODEL_STAMP_FORMAT,
    "%d__%m__%Y-%H_%M__%S",
    "%d__%m__%Y-%H_%M_%S",
    "%Y%m%d_%H%M%S",
)
COMPILE_RESULT_PATTERN = re.compile(r"Result:\s+(\d+)\s+errors?,\s+(\d+)\s+warnings?")
DEFINE_PATTERN = re.compile(r"^\s*#define\s+([A-Z0-9_]+)\s+(.+?)\s*$")
SAFE_SYMBOL_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
MODEL_FILE_NAME = "model.onnx"
MODEL_CONFIG_SNAPSHOT_NAME = "model_config_snapshot.mqh"
SHARED_CONFIG_SNAPSHOT_NAME = "shared_config_snapshot.mqh"
DEFAULT_TEST_CONFIG_NAME = "backtest_config.json"


def format_model_stamp(value: datetime | None = None) -> str:
    return (value or datetime.now()).strftime(DEFAULT_MODEL_STAMP_FORMAT)


def parse_model_stamp(value: str) -> datetime:
    for stamp_format in MODEL_STAMP_FORMATS:
        try:
            return datetime.strptime(value, stamp_format)
        except ValueError:
            continue
    raise ValueError(
        "Model date must match the model folder name, for example 03_04_2026-06_45__00."
    )


def parse_define_value(raw_value: str, known_values: dict[str, int | float | str]) -> int | float | str:
    value = raw_value.split("//", 1)[0].strip()
    if value.endswith("f"):
        value = value[:-1]
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

    return eval(value, {"__builtins__": {}}, dict(known_values))


def load_define_file(path: Path) -> dict[str, int | float | str]:
    values: dict[str, int | float | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = DEFINE_PATTERN.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        values[name] = parse_define_value(raw_value, values)
    return values


def sanitize_symbol(symbol: str) -> str:
    cleaned = SAFE_SYMBOL_PATTERN.sub("_", symbol.strip())
    return cleaned or "UNKNOWN"


def configured_symbol(config_path: Path = ACTIVE_SHARED_CONFIG_PATH) -> str:
    shared = load_define_file(config_path)
    return str(shared.get("SYMBOL", "")).strip() or "XAUUSD"


def symbol_models_dir(symbol: str) -> Path:
    return MODELS_DIR / sanitize_symbol(symbol)


def iter_model_dirs(symbol: str) -> list[Path]:
    root = symbol_models_dir(symbol)
    if not root.exists():
        return []

    model_dirs: list[tuple[datetime, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            model_dirs.append((parse_model_stamp(candidate.name), candidate))
        except ValueError:
            continue
    return [path for _, path in sorted(model_dirs, key=lambda item: item[0])]


def latest_model_dir(symbol: str) -> Path:
    candidates = iter_model_dirs(symbol)
    if not candidates:
        raise FileNotFoundError(f"No archived models found for symbol '{symbol}'.")
    return candidates[-1]


def resolve_model_dir(symbol: str, value: str = "") -> Path:
    if not value:
        return latest_model_dir(symbol)

    candidate = Path(value)
    if candidate.is_absolute():
        model_dir = candidate
    else:
        model_dir = symbol_models_dir(symbol) / value

    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")
    parse_model_stamp(model_dir.name)
    return model_dir


def model_onnx_path(model_dir: Path) -> Path:
    return model_dir / MODEL_FILE_NAME


def model_diagnostics_dir(model_dir: Path) -> Path:
    return model_dir / "diagnostics"


def model_tests_dir(model_dir: Path) -> Path:
    return model_dir / "tests"


def model_config_snapshot_path(model_dir: Path) -> Path:
    return model_diagnostics_dir(model_dir) / MODEL_CONFIG_SNAPSHOT_NAME


def shared_config_snapshot_path(model_dir: Path) -> Path:
    return model_diagnostics_dir(model_dir) / SHARED_CONFIG_SNAPSHOT_NAME


def default_test_config(symbol: str) -> dict[str, int | float | str]:
    return {
        "month": "",
        "from_date": "",
        "to_date": "",
        "symbol": symbol,
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


def ensure_default_test_config(tests_dir: Path, symbol: str) -> Path:
    config_path = tests_dir / DEFAULT_TEST_CONFIG_NAME
    if not config_path.exists():
        write_test_config(config_path, default_test_config(symbol))
    return config_path


def read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def sync_directory_contents(source_dir: Path, destination_dir: Path) -> None:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def activate_model(model_dir: Path) -> None:
    onnx_path = model_onnx_path(model_dir)
    diagnostics_dir = model_diagnostics_dir(model_dir)
    model_config_path = model_config_snapshot_path(model_dir)
    shared_config_path = shared_config_snapshot_path(model_dir)

    if not onnx_path.exists():
        raise FileNotFoundError(f"Archived ONNX file not found: {onnx_path}")
    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"Archived diagnostics folder not found: {diagnostics_dir}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Archived model config snapshot not found: {model_config_path}")
    if not shared_config_path.exists():
        raise FileNotFoundError(f"Archived shared config snapshot not found: {shared_config_path}")

    ACTIVE_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, ACTIVE_ONNX_PATH)
    shutil.copy2(model_config_path, ACTIVE_MODEL_CONFIG_PATH)
    shutil.copy2(shared_config_path, ACTIVE_SHARED_CONFIG_PATH)
    sync_directory_contents(diagnostics_dir, ACTIVE_DIAGNOSTICS_DIR)


def deploy_active_model(runtime: Mt5RuntimePaths) -> None:
    ensure_runtime_dirs(runtime)
    copies = (
        (LIVE_MQ5_PATH, runtime.deployed_live_mq5),
        (ACTIVE_ONNX_PATH, runtime.deployed_model_path),
        (ACTIVE_MODEL_CONFIG_PATH, runtime.deployed_model_config_path),
        (ACTIVE_SHARED_CONFIG_PATH, runtime.deployed_shared_config_path),
    )
    for source_path, destination_path in copies:
        if not source_path.exists():
            raise FileNotFoundError(f"Required runtime file not found: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def compile_live_expert(runtime: Mt5RuntimePaths) -> Path:
    deploy_active_model(runtime)
    command = build_metaeditor_compile_command(
        runtime=runtime,
        source_path=runtime.deployed_live_mq5,
        log_path=runtime.deployed_compile_log,
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=runtime_env(runtime),
    )
    log_text = read_text_best_effort(runtime.deployed_compile_log) if runtime.deployed_compile_log.exists() else ""
    if not log_text and completed.stdout:
        log_text = completed.stdout

    result_match = COMPILE_RESULT_PATTERN.search(log_text)
    if result_match:
        errors = int(result_match.group(1))
        warnings = int(result_match.group(2))
        if errors > 0:
            raise RuntimeError(
                f"live.mq5 compile failed with {errors} errors and {warnings} warnings. "
                f"Check log at {runtime.deployed_compile_log}."
            )
        return runtime.deployed_compile_log

    if completed.returncode != 0:
        raise RuntimeError(
            f"MetaEditor returned exit code {completed.returncode} while compiling {runtime.deployed_live_mq5}.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    raise RuntimeError(f"Could not confirm live.mq5 compile status. Check log at {runtime.deployed_compile_log}.")
