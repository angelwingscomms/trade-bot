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


def deploy_to_last_model(onnx_path: Path, model_config_path: Path, shared_config_path: Path, 
                          diagnostics_dir: Path, tests_dir: Path) -> None:
    """Deploy model files to models/last_model/ for live.mq5 to reference."""
    last_model_dir = MODELS_DIR / "last_model"
    log = logging.getLogger(__name__)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not shared_config_path.exists():
        raise FileNotFoundError(f"Shared config not found: {shared_config_path}")
    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"Diagnostics directory not found: {diagnostics_dir}")
    
    last_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    shutil.copy2(onnx_path, last_model_dir / "model.onnx")
    shutil.copy2(model_config_path, last_model_dir / "model_config.mqh")
    shutil.copy2(shared_config_path, last_model_dir / "shared_config.mqh")
    
    # Sync diagnostics
    sync_directory_contents(diagnostics_dir, last_model_dir / "diagnostics")
    
    # Copy test config if it exists
    if tests_dir.exists():
        tests_dest = last_model_dir / "tests"
        if tests_dest.exists():
            shutil.rmtree(tests_dest)
        shutil.copytree(tests_dir, tests_dest)
    
    log.info("Deployed model to %s", last_model_dir)


def deploy_active_model(runtime: Mt5RuntimePaths) -> None:
    ensure_runtime_dirs(runtime)
    copies = (
        (LIVE_MQ5_PATH, runtime.deployed_live_mq5),
        (ACTIVE_ONNX_PATH, runtime.deployed_model_path),
        (ACTIVE_MODEL_CONFIG_PATH, runtime.deployed_model_config_path),
        (ACTIVE_SHARED_CONFIG_PATH, runtime.deployed_shared_config_path),
    )
    max_retries = 3
    retry_delay = 1.0
    log = logging.getLogger(__name__)
    for source_path, destination_path in copies:
        if not source_path.exists():
            raise FileNotFoundError(f"Required runtime file not found: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(max_retries):
            try:
                shutil.copy2(source_path, destination_path)
                break
            except PermissionError as e:
                if attempt == max_retries - 1:
                    error_msg = (
                        f"Could not deploy {source_path.name} after {max_retries} attempts (file locked by MetaTrader). "
                        "Close MetaTrader 5 and retry, or use --skip-live-compile to skip deployment."
                    )
                    log.error(error_msg)
                    raise PermissionError(error_msg) from e
                wait_time = retry_delay * (2 ** attempt)
                log.warning(
                    "Failed to copy %s (attempt %d/%d), retrying in %.1fs...",
                    source_path.name,
                    attempt + 1,
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)


def _touch_file(path: Path) -> None:
    now = time.time()
    os.utime(path, (now, now))


def _write_synthetic_compile_log(path: Path, message: str) -> None:
    path.write_text(
        "\n".join(
            [
                message,
                "Result: 0 errors, 0 warnings",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _compile_via_metaeditor_ui(runtime: Mt5RuntimePaths) -> None:
    if runtime.host_platform != "windows" or runtime.use_wine:
        raise RuntimeError("MetaEditor UI fallback is only supported on native Windows.")

    source_path = str(runtime.deployed_live_mq5).replace("'", "''")
    metaeditor_path = str(runtime.metaeditor_path).replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Stop'
Get-Process MetaEditor64 -ErrorAction SilentlyContinue | Stop-Process -Force
$sourcePath = '{source_path}'
$metaeditorPath = '{metaeditor_path}'
$ex5Path = [System.IO.Path]::ChangeExtension($sourcePath, '.ex5')
$beforeWrite = if (Test-Path $ex5Path) {{ (Get-Item $ex5Path).LastWriteTimeUtc }} else {{ [datetime]::MinValue }}
$beforeLength = if (Test-Path $ex5Path) {{ (Get-Item $ex5Path).Length }} else {{ -1 }}
$meta = Start-Process -FilePath $metaeditorPath -ArgumentList ('"' + $sourcePath + '"') -PassThru
Start-Sleep -Seconds 6
$ws = New-Object -ComObject WScript.Shell
try {{ [void]$ws.AppActivate('MetaEditor') }} catch {{}}
Start-Sleep -Milliseconds 700
$ws.SendKeys('{{F7}}')
$deadline = (Get-Date).AddSeconds(30)
$compiled = $false
while ((Get-Date) -lt $deadline) {{
    Start-Sleep -Milliseconds 500
    if (Test-Path $ex5Path) {{
        $item = Get-Item $ex5Path
        if ($item.LastWriteTimeUtc -gt $beforeWrite -or $item.Length -ne $beforeLength) {{
            $compiled = $true
            break
        }}
    }}
}}
if ($compiled) {{
    try {{ [void]$ws.AppActivate('MetaEditor') }} catch {{}}
    Start-Sleep -Milliseconds 300
    $ws.SendKeys('%{{F4}}')
    Start-Sleep -Seconds 2
}}
if (Get-Process -Id $meta.Id -ErrorAction SilentlyContinue) {{
    Stop-Process -Id $meta.Id -Force
}}
if (-not $compiled) {{
    throw 'MetaEditor UI fallback did not update live.ex5.'
}}
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
        env=runtime_env(runtime),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MetaEditor UI fallback failed while compiling live.mq5.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )


def compile_live_expert(runtime: Mt5RuntimePaths, skip_deployment: bool = False) -> Path:
    if not skip_deployment:
        deploy_active_model(runtime)
    runtime.deployed_compile_log.unlink(missing_ok=True)
    previous_ex5_mtime = runtime.deployed_live_ex5.stat().st_mtime if runtime.deployed_live_ex5.exists() else 0.0
    _touch_file(runtime.deployed_live_mq5)
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

    for _ in range(10):
        if runtime.deployed_live_ex5.exists() and runtime.deployed_live_ex5.stat().st_mtime > previous_ex5_mtime:
            _write_synthetic_compile_log(
                runtime.deployed_compile_log,
                "MetaEditor updated live.ex5 without producing a dedicated CLI compile log.",
            )
            return runtime.deployed_compile_log
        time.sleep(0.5)

    if completed.returncode != 0:
        raise RuntimeError(
            f"MetaEditor returned exit code {completed.returncode} while compiling {runtime.deployed_live_mq5}.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    _compile_via_metaeditor_ui(runtime)
    _write_synthetic_compile_log(
        runtime.deployed_compile_log,
        "MetaEditor CLI produced no usable compile status, so a UI fallback compile was used successfully.",
    )
    return runtime.deployed_compile_log
