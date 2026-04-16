"""Compile and run the MT5 data-export scripts, then collect the resulting CSV."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from mt5_runtime import (
    PROJECT_DIR_NAME,
    build_metaeditor_compile_command,
    build_terminal_command,
    read_text_best_effort,
    resolve_mt5_runtime,
    runtime_env,
    stop_terminal_best_effort,
)
from tradebot.config_io import load_define_file
from tradebot.workspace import ACTIVE_CONFIG_PATH, configured_symbol, symbol_default_config_path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_FILE = "market_ticks.csv"
COMPILE_LOG_PATH = SCRIPT_DIR / "data.compile.log"
STARTUP_CONFIG_DIR = Path(tempfile.gettempdir()) / "mt5_export_configs"
DATA_PROFILE_SCRIPTS = {
    "default": SCRIPT_DIR / "data.mq5",
    "gold": SCRIPT_DIR / "data_gold.mq5",
}


def runtime_script_dir(runtime) -> Path:
    return runtime.instance_root / "MQL5" / "Scripts" / PROJECT_DIR_NAME


def profile_script_name(profile: str) -> str:
    return "data_gold" if profile == "gold" else "data"


def deploy_script_files(runtime, shared_config_path: Path, profile: str) -> tuple[Path, Path]:
    script_dir = runtime_script_dir(runtime)
    script_dir.mkdir(parents=True, exist_ok=True)
    deployed_source = script_dir / f"{profile_script_name(profile)}.mq5"
    deployed_shared_config = script_dir / "config.mqh"
    source_path = DATA_PROFILE_SCRIPTS.get(profile, SCRIPT_DIR / "data.mq5")
    shutil.copy2(source_path, deployed_source)
    shutil.copy2(shared_config_path, deployed_shared_config)
    return deployed_source, deployed_shared_config


def resolve_symbol_config(requested_symbol: str) -> tuple[str, Path]:
    requested = requested_symbol.strip()
    if requested:
        config_path = symbol_default_config_path(requested)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found for symbol '{requested}': {config_path}")
        shared = load_define_file(config_path)
        symbol = str(shared.get("SYMBOL", requested)).strip() or requested
        return symbol, config_path

    active_symbol = configured_symbol()
    config_path = symbol_default_config_path(active_symbol)
    if config_path.exists():
        shared = load_define_file(config_path)
        symbol = str(shared.get("SYMBOL", active_symbol)).strip() or active_symbol
        return symbol, config_path
    return active_symbol, ACTIVE_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MT5 tick data into ./data/<SYMBOL>/ticks.csv.")
    parser.add_argument("--symbol", type=str, default="", help="Optional symbol preset to load from symbols/<symbol>/config.")
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=sorted(DATA_PROFILE_SCRIPTS.keys()),
        help="Select the data export script profile (default or gold).",
    )
    parser.add_argument("--instance-root", type=str, default="", help="Optional explicit MT5 data root.")
    parser.add_argument("--terminal-path", type=str, default="", help="Optional explicit terminal64.exe path.")
    parser.add_argument("--metaeditor-path", type=str, default="", help="Optional explicit MetaEditor path.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Maximum time to wait for the exported CSV to appear in MQL5/Files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="CSV filename written by data.mq5 inside MQL5/Files.",
    )
    return parser.parse_args()


def compile_data_script(runtime, shared_config_path: Path, profile: str) -> Path:
    source_path, deployed_shared_config = deploy_script_files(runtime, shared_config_path, profile=profile)
    target_path = source_path.with_suffix(".ex5")
    source_log_path = source_path.with_suffix(".log")
    COMPILE_LOG_PATH.unlink(missing_ok=True)
    source_log_path.unlink(missing_ok=True)

    newest_input_mtime = max(source_path.stat().st_mtime, deployed_shared_config.stat().st_mtime)
    if target_path.exists() and target_path.stat().st_mtime >= newest_input_mtime:
        return target_path

    command = build_metaeditor_compile_command(
        runtime=runtime,
        source_path=source_path,
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=runtime_env(runtime),
    )
    if source_log_path.exists():
        shutil.copy2(source_log_path, COMPILE_LOG_PATH)

    if completed.returncode != 0 and not target_path.exists():
        log_text = read_text_best_effort(source_log_path) if source_log_path.exists() else ""
        raise RuntimeError(
            "MetaEditor failed to compile data.mq5.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n\nlog:\n{log_text}"
        )
    deadline = time.time() + (60.0 if runtime.use_wine else 10.0)
    while time.time() < deadline:
        if target_path.exists() and target_path.stat().st_mtime >= newest_input_mtime:
            if source_log_path.exists() and not COMPILE_LOG_PATH.exists():
                shutil.copy2(source_log_path, COMPILE_LOG_PATH)
            return target_path
        time.sleep(0.5)
    log_text = read_text_best_effort(source_log_path) if source_log_path.exists() else ""
    if not target_path.exists():
        raise FileNotFoundError(
            f"Compiled script not found after MetaEditor run: {target_path}\n\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n\nlog:\n{log_text}"
        )
    return target_path


def write_startup_config(symbol: str, profile: str) -> Path:
    STARTUP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    script_name = profile_script_name(profile)
    startup_config_path = STARTUP_CONFIG_DIR / f"{symbol.lower()}_{script_name}_export.ini"
    startup_config_path.write_text(
        "\n".join(
            [
                "[StartUp]",
                f"Script={PROJECT_DIR_NAME}\\{script_name}",
                f"Symbol={symbol}",
                "Period=M1",
                "ShutdownTerminal=1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return startup_config_path


def run_script(runtime, symbol: str, profile: str) -> None:
    if not runtime.terminal_path.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {runtime.terminal_path}")

    stop_terminal_best_effort(runtime)
    config_path = write_startup_config(symbol, profile=profile)
    command = build_terminal_command(runtime, config_path)
    print(f"[INFO] Launching MT5 with {profile_script_name(profile)}.mq5 for {symbol}...")
    print(f"[INFO] Command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=runtime_env(runtime),
        text=True,
        start_new_session=(runtime.host_platform != "windows"),
    )
    print(f"[INFO] Terminal started (PID: {process.pid})")


def find_csv_file(files_dir: Path, output_file: str, max_wait: float) -> Path | None:
    start_time = time.time()
    csv_path = files_dir / output_file
    print(f"[INFO] Waiting for {csv_path}...")

    while time.time() - start_time < max_wait:
        if csv_path.exists():
            time.sleep(0.5)
            if csv_path.stat().st_size > 100:
                print(f"[INFO] Found: {csv_path}")
                return csv_path
        time.sleep(0.5)
    return None


def move_to_data_dir(symbol: str, csv_path: Path) -> Path:
    symbol_dir = OUTPUT_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    dest_path = symbol_dir / "ticks.csv"

    print(f"[INFO] Moving {csv_path.name} to {dest_path}...")
    shutil.move(str(csv_path), str(dest_path))
    file_size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Successfully moved file ({file_size_mb:.2f} MB)")
    print(f"[INFO] Output: {dest_path}")
    return dest_path


def main() -> bool:
    args = parse_args()
    print("=" * 60)
    print("MT5 Data Export Tool")
    print("=" * 60)
    print()

    try:
        print("[STEP 1] Resolving symbol configuration...")
        symbol, config_path = resolve_symbol_config(args.symbol)
        print(f"[STEP 1] SUCCESS - Symbol: {symbol}")
        print(f"[STEP 1] Config: {config_path}")
        print()

        print("[STEP 2] Resolving MT5 runtime...")
        runtime = resolve_mt5_runtime(
            instance_root_override=args.instance_root,
            terminal_path_override=args.terminal_path,
            metaeditor_path_override=args.metaeditor_path,
        )
        print(f"[STEP 2] SUCCESS - Terminal: {runtime.terminal_path}")
        print(f"[STEP 2] SUCCESS - MetaEditor: {runtime.metaeditor_path}")
        print()

        if config_path != ACTIVE_CONFIG_PATH:
            shutil.copy2(config_path, ACTIVE_CONFIG_PATH)

        print("[STEP 3] Compiling data.mq5...")
        compiled_path = compile_data_script(runtime, config_path, profile=args.profile)
        print(f"[STEP 3] SUCCESS - Compiled: {compiled_path}")
        print()

        print(f"[STEP 4] Launching MT5 terminal with {profile_script_name(args.profile)}.mq5...")
        stale_output = runtime.files_dir / args.output_file
        stale_output.unlink(missing_ok=True)
        run_script(runtime, symbol, profile=args.profile)
        print("[STEP 4] SUCCESS")
        print()

        print(f"[STEP 5] Waiting for script output (up to {args.timeout_seconds} seconds)...")
        csv_path = find_csv_file(runtime.files_dir, args.output_file, max_wait=float(args.timeout_seconds))
        if csv_path is None:
            print("[STEP 5] FAILED - Timeout waiting for CSV file")
            print("[INFO] Possible issues:")
            print(f"  - MT5 Files folder: {runtime.files_dir}")
            print("  - Check MT5 Experts and Tester logs for errors")
            print("  - Symbol may not have tick history downloaded")
            return False
        print("[STEP 5] SUCCESS")
        print()

        print("[STEP 6] Moving file to data directory...")
        move_to_data_dir(symbol, csv_path)
        print("[STEP 6] SUCCESS")
        print()

        print("=" * 60)
        print("Export completed successfully!")
        print("=" * 60)
        return True
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
