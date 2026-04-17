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

from mt5_runtime import (
    DEFAULT_WINDOWS_METAEDITOR_PATH,
    Mt5RuntimePaths,
    PROJECT_DIR_NAME,
    build_metaeditor_compile_command,
    ensure_runtime_dirs,
    runtime_env,
    to_windows_path,
)
from tradebot.config_io import load_define_file, read_text_best_effort, sanitize_symbol


ROOT_DIR = Path(__file__).resolve().parent.parent
SYMBOLS_DIR = ROOT_DIR / "symbols"
ACTIVE_CONFIG_PATH = ROOT_DIR / "config.mqh"
ACTIVE_CONFIG_POINTER = ROOT_DIR / ".active_config"
ACTIVE_DIAGNOSTICS_DIR = ROOT_DIR / "diagnostics"
LIVE_MQ5_PATH = ROOT_DIR / "live.mq5"
LIVE_EX5_PATH = ROOT_DIR / "live.ex5"
LIVE_COMPILE_LOG_PATH = ROOT_DIR / "live.compile.log"

DEFAULT_METAEDITOR_PATH = DEFAULT_WINDOWS_METAEDITOR_PATH
DEFAULT_MODEL_STAMP_FORMAT = "%d_%m_%Y-%H_%M__%S"
MODEL_STAMP_FORMATS = (
    DEFAULT_MODEL_STAMP_FORMAT,
    "%d_%m_%Y-%H_%M_%S",
    "%d__%m__%Y-%H_%M__%S",
    "%d__%m__%Y-%H_%M_%S",
    "%Y%m%d_%H%M%S",
)
MODEL_STAMP_PREFIX_PATTERN = re.compile(r"^(?P<stamp>\d{2}_\d{2}_\d{4}-\d{2}_\d{2}(?:__|_)\d{2})(?:-|$)")
MODEL_STAMP_SUFFIX_PATTERN = re.compile(r"(?P<stamp>\d{2}_\d{2}_\d{4}-\d{2}_\d{2}(?:__|_)\d{2})(?:-fail)?$")
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


def format_model_stamp(value: datetime | None = None) -> str:
    """Format a model timestamp using the repo's stable folder naming style."""

    return (value or datetime.now()).strftime(DEFAULT_MODEL_STAMP_FORMAT)


def sanitize_model_name(name: str) -> str:
    """Return a filesystem-safe model label."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._-")


def format_model_dir_name(
    *,
    value: datetime | None = None,
    name: str = "",
    failed_quality_gate: bool = False,
    symbol: str = "",
) -> str:
    """Return a canonical, resource-safe `<stamp>-<name>` model folder name."""

    stamp = format_model_stamp(value)
    suffix = sanitize_model_name(name)
    if symbol:
        max_length = _max_model_dir_name_length(symbol)
        max_suffix_length = max_length - len(stamp)
        if suffix:
            max_suffix_length -= 1
        if max_suffix_length < len(suffix):
            suffix = suffix[: max(0, max_suffix_length)].rstrip("._-")
    folder_name = f"{stamp}-{suffix}" if suffix else stamp
    if failed_quality_gate and (not symbol or len(folder_name) + len("-fail") <= _max_model_dir_name_length(symbol)):
        folder_name += "-fail"
    return folder_name


def _try_parse_model_stamp_text(value: str) -> datetime | None:
    """Parse one of the accepted timestamp shapes used by archived models."""

    for stamp_format in MODEL_STAMP_FORMATS:
        try:
            return datetime.strptime(value, stamp_format)
        except ValueError:
            continue
    return None


def parse_model_stamp(folder_name: str) -> datetime:
    """Extract the timestamp from either new or legacy model folder names."""

    match = MODEL_STAMP_PREFIX_PATTERN.match(folder_name)
    if match:
        parsed = _try_parse_model_stamp_text(match.group("stamp"))
        if parsed is not None:
            return parsed

    match = MODEL_STAMP_SUFFIX_PATTERN.search(folder_name)
    if match:
        parsed = _try_parse_model_stamp_text(match.group("stamp"))
        if parsed is not None:
            return parsed

    parsed = _try_parse_model_stamp_text(folder_name)
    if parsed is not None:
        return parsed

    raise ValueError(
        "Model folders must contain a training stamp such as "
        "`03_04_2026-06_45__00-my-model` or `my-model-03_04_2026-06_45__00-fail`."
    )


def configured_symbol(config_path: Path = ACTIVE_CONFIG_PATH) -> str:
    """Read the active symbol from the user-editable root config."""

    values = load_define_file(config_path)
    return str(values.get("SYMBOL", "XAUUSD")).strip() or "XAUUSD"


def resolve_active_config_path() -> Path:
    """Return the config file pointed to by the pointer file, or the default config.

    If `.active_config` exists in ROOT_DIR, its first line is read as the path
    to the active config (absolute or relative to ROOT_DIR). Otherwise, falls
    back to `config.mqh`.
    """
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


def symbol_dir(symbol: str) -> Path:
    """Return the per-symbol workspace directory."""

    return SYMBOLS_DIR / sanitize_symbol(symbol)


def symbol_models_dir(symbol: str) -> Path:
    """Return the archived-model root for one symbol."""

    return symbol_dir(symbol) / "models"


def symbol_config_dir(symbol: str) -> Path:
    """Return the preset-config root for one symbol."""

    return symbol_dir(symbol) / "config"


def symbol_default_config_path(symbol: str) -> Path:
    """Return the symbol's default reusable config preset."""

    return symbol_config_dir(symbol) / "config.mqh"


def symbol_backtest_config_path(symbol: str) -> Path:
    """Return the symbol's default backtest JSON config."""

    return symbol_config_dir(symbol) / DEFAULT_TEST_CONFIG_NAME


def iter_model_dirs(symbol: str) -> list[Path]:
    """Return archived model folders ordered by training time."""

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


def model_onnx_path(model_dir: Path) -> Path:
    """Return the ONNX path inside one archived model folder."""

    return model_dir / MODEL_FILE_NAME


def model_config_path(model_dir: Path) -> Path:
    """Return the single combined config file stored beside a model."""

    return model_dir / MODEL_CONFIG_NAME


def model_diagnostics_dir(model_dir: Path) -> Path:
    """Return the diagnostics directory for one archived model."""

    return model_dir / "diagnostics"


def model_tests_dir(model_dir: Path) -> Path:
    """Return the backtest-results directory for one archived model."""

    return model_dir / "tests"


def _max_model_dir_name_length(symbol: str) -> int:
    """Return the longest archive folder name allowed by the MQL5 resource limit."""

    symbol_name = sanitize_symbol(symbol)
    fixed_length = len(f"symbols\\\\{symbol_name}\\\\models\\\\\\\\{MODEL_FILE_NAME}")
    max_length = RESOURCE_PATH_MAX_CHARS - fixed_length
    if max_length <= 0:
        raise ValueError(
            f"Resource path budget is exhausted for symbol '{symbol_name}'. "
            "Choose a shorter symbol folder name."
        )
    return max_length


def _resource_literal_for_relative_model_dir(relative_model_dir: Path) -> str:
    """Return a `#resource` path that obeys the MQL5 path rules."""

    literal = relative_model_dir.as_posix().replace("/", "\\\\")
    literal += "\\\\" + MODEL_FILE_NAME
    if "\\" in literal.replace("\\\\", ""):
        raise ValueError(f"Resource literal was not fully escaped for MQL5: {literal!r}")
    if len(literal) > RESOURCE_PATH_MAX_CHARS:
        raise ValueError(
            "The ONNX resource path is too long for MQL5. "
            "Use a shorter model name so the archive folder stays within the 63-character resource limit."
        )
    return literal


def latest_model_dir(symbol: str) -> Path:
    """Return the latest archived model that still has the required artifacts."""

    candidates = [
        candidate
        for candidate in iter_model_dirs(symbol)
        if model_onnx_path(candidate).exists() and model_config_path(candidate).exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No archived models found for symbol '{symbol}'.")
    return candidates[-1]


def resolve_model_dir(symbol: str, value: str = "") -> Path:
    """Resolve an explicit model folder name or fall back to the latest model."""

    if not value:
        return latest_model_dir(symbol)

    candidate = Path(value)
    model_dir = candidate if candidate.is_absolute() else symbol_models_dir(symbol) / value
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")
    parse_model_stamp(model_dir.name)
    return model_dir


def default_test_config(symbol: str) -> dict[str, int | float | str]:
    """Return the default tester settings used when no JSON exists yet."""

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
    """Load a backtest JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_test_config(path: Path, data: dict[str, int | float | str]) -> None:
    """Write a stable, prettified backtest JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_default_test_config(tests_dir: Path, symbol: str) -> Path:
    """Ensure a model has a seed `backtest_config.json` file."""

    config_path = tests_dir / DEFAULT_TEST_CONFIG_NAME
    if not config_path.exists():
        symbol_default_path = symbol_backtest_config_path(symbol)
        if symbol_default_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(symbol_default_path, config_path)
        else:
            write_test_config(config_path, default_test_config(symbol))
    return config_path


def sync_directory_contents(source_dir: Path, destination_dir: Path) -> None:
    """Replace one directory tree with another."""

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def build_live_model_reference_block(model_dir: Path) -> str:
    """Build the live.mq5 include/resource block for the active model."""

    try:
        relative_model_dir = model_dir.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(f"Model directory must live under {ROOT_DIR}: {model_dir}") from exc

    symbol_value = sanitize_symbol(model_dir.parent.parent.name).upper()
    version = model_dir.name
    include_dir = relative_model_dir.as_posix()
    resource_literal = _resource_literal_for_relative_model_dir(relative_model_dir)
    return "\n".join(
        [
            LIVE_MODEL_BLOCK_BEGIN,
            f'#define ACTIVE_MODEL_SYMBOL "{symbol_value}"',
            f'#define ACTIVE_MODEL_VERSION "{version}"',
            f'#include "{include_dir}/{MODEL_CONFIG_NAME}"',
            f'#resource "{resource_literal}" as uchar model_buffer[]',
            LIVE_MODEL_BLOCK_END,
        ]
    )


def set_live_model_reference(model_dir: Path, live_path: Path = LIVE_MQ5_PATH) -> None:
    """Point `live.mq5` at a specific archived model directory."""

    if not model_onnx_path(model_dir).exists():
        raise FileNotFoundError(f"Archived ONNX file not found: {model_onnx_path(model_dir)}")
    if not model_config_path(model_dir).exists():
        raise FileNotFoundError(f"Archived config file not found: {model_config_path(model_dir)}")

    text = live_path.read_text(encoding="utf-8")
    replacement_block = build_live_model_reference_block(model_dir)
    updated_text, replacements = LIVE_MODEL_BLOCK_PATTERN.subn(
        lambda _match: replacement_block,
        text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(
            f"{live_path} is missing the active model reference markers "
            f"{LIVE_MODEL_BLOCK_BEGIN!r} / {LIVE_MODEL_BLOCK_END!r}."
        )
    live_path.write_text(updated_text, encoding="utf-8")


def activate_model(model_dir: Path) -> None:
    """Update the local `live.mq5` source to reference an archived model."""

    if not model_onnx_path(model_dir).exists():
        raise FileNotFoundError(f"Archived ONNX file not found: {model_onnx_path(model_dir)}")
    if not model_config_path(model_dir).exists():
        raise FileNotFoundError(f"Archived config file not found: {model_config_path(model_dir)}")
    set_live_model_reference(model_dir)
    diagnostics_dir = model_diagnostics_dir(model_dir)
    if diagnostics_dir.exists():
        sync_directory_contents(diagnostics_dir, ACTIVE_DIAGNOSTICS_DIR)


def _copy_with_retries(source_path: Path, destination_path: Path, *, log: logging.Logger) -> None:
    """Copy one file while tolerating temporary MetaTrader file locks."""

    max_retries = 3
    retry_delay = 1.0
    if source_path.resolve() == destination_path.resolve():
        log.info("Skipping deployment copy because source and destination match: %s", source_path)
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            shutil.copy2(source_path, destination_path)
            return
        except PermissionError as exc:
            if attempt == max_retries - 1:
                error_msg = (
                    f"Could not deploy {source_path.name} after {max_retries} attempts (file locked by MetaTrader). "
                    "Close MetaTrader 5 and retry, or skip compilation."
                )
                log.error(error_msg)
                raise PermissionError(error_msg) from exc
            wait_time = retry_delay * (2**attempt)
            log.warning(
                "Failed to copy %s (attempt %d/%d), retrying in %.1fs...",
                source_path.name,
                attempt + 1,
                max_retries,
                wait_time,
            )
            time.sleep(wait_time)


def deploy_active_model(runtime: Mt5RuntimePaths, model_dir: Path) -> None:
    """Copy the live source and referenced archived model into the MT5 runtime."""

    ensure_runtime_dirs(runtime)
    log = logging.getLogger(__name__)
    _copy_with_retries(LIVE_MQ5_PATH, runtime.deployed_live_mq5, log=log)

    relative_model_dir = model_dir.relative_to(ROOT_DIR)
    runtime_model_dir = runtime.expert_dir / relative_model_dir
    runtime_model_dir.mkdir(parents=True, exist_ok=True)
    _copy_with_retries(model_onnx_path(model_dir), runtime_model_dir / MODEL_FILE_NAME, log=log)
    _copy_with_retries(model_config_path(model_dir), runtime_model_dir / MODEL_CONFIG_NAME, log=log)


def _touch_file(path: Path) -> None:
    """Update the timestamp on a file so MetaEditor notices a changed source."""

    now = time.time()
    os.utime(path, (now, now))


def _write_synthetic_compile_log(path: Path, message: str) -> None:
    """Write a small synthetic compile log when MetaEditor omits one."""

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


def _compile_via_metaeditor_ui_windows(runtime: Mt5RuntimePaths) -> None:
    """Compile `live.mq5` via the MetaEditor UI on native Windows."""

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


def _compile_via_metaeditor_ui_wine_xdotool(runtime: Mt5RuntimePaths) -> None:
    """Compile `live.mq5` via `wine start /unix` and xdotool key automation."""

    if shutil.which("xdotool") is None:
        raise RuntimeError("MetaEditor UI fallback on Linux/Wine requires xdotool.")

    subprocess.run(
        ["pkill", "-f", "MetaEditor64.exe"],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_env(runtime),
    )

    source_value = to_windows_path(runtime, runtime.deployed_live_mq5)
    previous_ex5_mtime = runtime.deployed_live_ex5.stat().st_mtime if runtime.deployed_live_ex5.exists() else 0.0
    previous_ex5_size = runtime.deployed_live_ex5.stat().st_size if runtime.deployed_live_ex5.exists() else -1
    completed = subprocess.run(
        ["wine", "start", "/unix", str(runtime.metaeditor_path), source_value],
        capture_output=True,
        text=True,
        check=False,
        env=runtime_env(runtime),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MetaEditor UI fallback could not launch MetaEditor under Wine.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    deadline = time.time() + 30.0
    window_found = False
    while time.time() < deadline:
        search = subprocess.run(
            ["xdotool", "search", "--name", "MetaEditor", "getwindowname"],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env(runtime),
        )
        if search.returncode == 0 and search.stdout.strip():
            window_found = True
            break
        time.sleep(0.5)
    if not window_found:
        raise RuntimeError("MetaEditor window did not appear on the X11 display.")

    time.sleep(0.7)
    subprocess.run(
        ["xdotool", "search", "--name", "MetaEditor", "windowactivate", "--sync", "key", "--clearmodifiers", "F7"],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_env(runtime),
    )

    compile_deadline = time.time() + 60.0
    while time.time() < compile_deadline:
        if runtime.deployed_live_ex5.exists():
            ex5_stat = runtime.deployed_live_ex5.stat()
            if ex5_stat.st_mtime > previous_ex5_mtime or ex5_stat.st_size != previous_ex5_size:
                subprocess.run(
                    ["pkill", "-f", "MetaEditor64.exe"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=runtime_env(runtime),
                )
                return
        time.sleep(0.5)

    subprocess.run(
        ["pkill", "-f", "MetaEditor64.exe"],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_env(runtime),
    )
    raise RuntimeError("MetaEditor UI fallback did not update live.ex5.")


def _compile_via_metaeditor_ui_wine(runtime: Mt5RuntimePaths) -> None:
    """Compile `live.mq5` via MetaEditor UI automation under Wine/X11."""

    if shutil.which("xdotool") is not None:
        _compile_via_metaeditor_ui_wine_xdotool(runtime)
        return

    try:
        from Xlib import X, XK, display, protocol
        from Xlib.ext import xtest
    except ImportError as exc:
        raise RuntimeError("MetaEditor UI fallback on Linux/Wine requires python-xlib.") from exc

    def keycode(dpy, keysym_name: str) -> int:
        keysym = XK.string_to_keysym(keysym_name)
        if not keysym:
            raise RuntimeError(f"Unsupported X11 keysym for MetaEditor fallback: {keysym_name}")
        code = dpy.keysym_to_keycode(keysym)
        if code == 0:
            raise RuntimeError(f"Could not resolve X11 keycode for MetaEditor fallback: {keysym_name}")
        return code

    def activate_window(dpy, win) -> None:
        root = dpy.screen().root
        try:
            net_active_window = dpy.intern_atom("_NET_ACTIVE_WINDOW")
            event = protocol.event.ClientMessage(
                window=win,
                client_type=net_active_window,
                data=(32, [1, X.CurrentTime, 0, 0, 0]),
            )
            root.send_event(
                event,
                event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask,
            )
        except Exception:
            pass
        try:
            win.configure(stack_mode=X.Above)
        except Exception:
            pass
        dpy.sync()

    def send_key(dpy, keysym_name: str, modifiers: tuple[str, ...] = ()) -> None:
        modifier_codes = [keycode(dpy, modifier) for modifier in modifiers]
        main_code = keycode(dpy, keysym_name)
        for modifier_code in modifier_codes:
            xtest.fake_input(dpy, X.KeyPress, modifier_code)
        xtest.fake_input(dpy, X.KeyPress, main_code)
        xtest.fake_input(dpy, X.KeyRelease, main_code)
        for modifier_code in reversed(modifier_codes):
            xtest.fake_input(dpy, X.KeyRelease, modifier_code)
        dpy.sync()

    def find_metaeditor_window(dpy):
        root = dpy.screen().root
        for win in root.query_tree().children:
            try:
                wm_class = win.get_wm_class() or ()
                wm_name = win.get_wm_name() or ""
            except Exception:
                continue
            if any("metaeditor" in str(item).lower() for item in wm_class):
                return win
            if "metaeditor" in str(wm_name).lower() or "live.mq5" in str(wm_name).lower():
                return win
        return None

    subprocess.run(
        ["pkill", "-f", "MetaEditor64.exe"],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_env(runtime),
    )

    source_value = to_windows_path(runtime, runtime.deployed_live_mq5)
    previous_ex5_mtime = runtime.deployed_live_ex5.stat().st_mtime if runtime.deployed_live_ex5.exists() else 0.0
    previous_ex5_size = runtime.deployed_live_ex5.stat().st_size if runtime.deployed_live_ex5.exists() else -1
    process = subprocess.Popen(
        ["wine", str(runtime.metaeditor_path), source_value],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=runtime_env(runtime),
        start_new_session=True,
    )

    dpy = None
    compiled = False
    try:
        dpy = display.Display()
        deadline = time.time() + 30.0
        window = None
        while time.time() < deadline:
            window = find_metaeditor_window(dpy)
            if window is not None:
                break
            time.sleep(0.5)
        if window is None:
            raise RuntimeError("MetaEditor window did not appear on the X11 display.")

        time.sleep(0.7)
        activate_window(dpy, window)
        time.sleep(0.3)
        send_key(dpy, "F7")

        compile_deadline = time.time() + 60.0
        while time.time() < compile_deadline:
            if runtime.deployed_live_ex5.exists():
                ex5_stat = runtime.deployed_live_ex5.stat()
                if ex5_stat.st_mtime > previous_ex5_mtime or ex5_stat.st_size != previous_ex5_size:
                    compiled = True
                    break
            time.sleep(0.5)

        activate_window(dpy, window)
        time.sleep(0.3)
        send_key(dpy, "F4", modifiers=("Alt_L",))
        time.sleep(1.0)
    finally:
        if dpy is not None:
            try:
                dpy.close()
            except Exception:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)

    if not compiled:
        raise RuntimeError("MetaEditor UI fallback did not update live.ex5.")


def _compile_via_metaeditor_ui(runtime: Mt5RuntimePaths) -> None:
    """Use the UI fallback that matches the current host platform."""

    if runtime.use_wine:
        _compile_via_metaeditor_ui_wine(runtime)
        return
    if runtime.host_platform == "windows":
        _compile_via_metaeditor_ui_windows(runtime)
        return
    raise RuntimeError("MetaEditor UI fallback is only supported on Windows or Linux/Wine.")


def compile_live_expert(runtime: Mt5RuntimePaths, model_dir: Path, skip_deployment: bool = False) -> Path:
    """Compile the live EA after deploying the selected archived model tree."""

    if not skip_deployment:
        deploy_active_model(runtime, model_dir=model_dir)
    runtime.deployed_compile_log.unlink(missing_ok=True)
    source_log_path = runtime.deployed_live_mq5.with_suffix(".log")
    source_log_path.unlink(missing_ok=True)
    previous_ex5_mtime = runtime.deployed_live_ex5.stat().st_mtime if runtime.deployed_live_ex5.exists() else 0.0
    _touch_file(runtime.deployed_live_mq5)
    command = build_metaeditor_compile_command(
        runtime=runtime,
        source_path=runtime.deployed_live_mq5,
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=runtime_env(runtime),
    )
    if source_log_path.exists():
        shutil.copy2(source_log_path, runtime.deployed_compile_log)
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

    fallback_message = None
    if completed.returncode != 0:
        fallback_message = (
            f"MetaEditor returned exit code {completed.returncode} without a usable compile log, "
            "so a UI fallback compile was used successfully."
        )
    try:
        _compile_via_metaeditor_ui(runtime)
    except RuntimeError as exc:
        if completed.returncode != 0:
            raise RuntimeError(
                f"MetaEditor returned exit code {completed.returncode} while compiling {runtime.deployed_live_mq5}.\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n\n"
                f"UI fallback error:\n{exc}"
            ) from exc
        raise
    _write_synthetic_compile_log(
        runtime.deployed_compile_log,
        fallback_message or "MetaEditor CLI produced no usable compile status, so a UI fallback compile was used successfully.",
    )
    return runtime.deployed_compile_log
