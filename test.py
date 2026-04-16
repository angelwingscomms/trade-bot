"""Backtest one archived model folder through the MT5 daily tester flow."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from mt5_runtime import PROJECT_DIR_NAME, iter_agent_log_paths, launch_terminal as launch_mt5_terminal, resolve_mt5_runtime
from tradebot.config_io import load_define_file, read_text_best_effort, sanitize_symbol
from tradebot.workspace import (
    ACTIVE_DIAGNOSTICS_DIR,
    ACTIVE_CONFIG_PATH,
    activate_model,
    compile_live_expert,
    configured_symbol,
    ensure_default_test_config,
    format_model_stamp,
    load_test_config,
    model_config_path,
    model_tests_dir,
    resolve_model_dir,
)


SUMMARY_PATTERN = re.compile(r"\[SUMMARY\]\s+(.*)")
NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
BOOL_PATTERN = re.compile(r"^(true|false)$", re.IGNORECASE)
INITIAL_DEPOSIT_PATTERN = re.compile(r"initial deposit\s+(-?\d+(?:\.\d+)?)\s+([A-Z]+), leverage\s+(.+)$")
FINAL_BALANCE_PATTERN = re.compile(r"final balance\s+(-?\d+(?:\.\d+)?)\s+([A-Z]+)$")
GENERATED_PATTERN = re.compile(
    r":\s+(\d+)\s+ticks,\s+(\d+)\s+bars generated\..*?Test passed in\s+([0-9:.]+)",
    re.IGNORECASE,
)
STATUS_LINE_PATTERN = re.compile(r"testing of .* from (\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}) to (\d{4}\.\d{2}\.\d{2} \d{2}:\d{2})")

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ACTIVE_CONFIG_PATH


@dataclass
class BacktestResult:
    day: str
    status: str
    from_date: str
    to_date: str
    initial_deposit: float
    final_balance: float
    profit: float
    currency: str
    leverage: str
    ticks: int
    bars: int
    duration: str
    predictions: int
    hold_skips: int
    confidence_skips: int
    position_skips: int
    stops_too_close: int
    open_failures: int
    trades_opened: int
    trades_closed: int
    wins: int
    losses: int
    realized_pnl: float
    risk_mode: str
    fixed_move: float
    stop_out: int
    tester_finished: int
    error: str


def error_result(day_value: date, error: str) -> BacktestResult:
    return BacktestResult(
        day=day_value.isoformat(),
        status="error",
        from_date=day_value.strftime("%Y.%m.%d"),
        to_date=(day_value + timedelta(days=1)).strftime("%Y.%m.%d"),
        initial_deposit=0.0,
        final_balance=0.0,
        profit=0.0,
        currency="",
        leverage="",
        ticks=0,
        bars=0,
        duration="",
        predictions=0,
        hold_skips=0,
        confidence_skips=0,
        position_skips=0,
        stops_too_close=0,
        open_failures=0,
        trades_opened=0,
        trades_closed=0,
        wins=0,
        losses=0,
        realized_pnl=0.0,
        risk_mode="",
        fixed_move=0.0,
        stop_out=0,
        tester_finished=0,
        error=error,
    )


def parse_month(value: str | None) -> tuple[date, date]:
    if value:
        month_start = datetime.strptime(value, "%Y-%m").date().replace(day=1)
    else:
        today = date.today().replace(day=1)
        previous_month_last_day = today - timedelta(days=1)
        month_start = previous_month_last_day.replace(day=1)
    if month_start.month == 12:
        month_end = date(month_start.year + 1, 1, 1)
    else:
        month_end = date(month_start.year, month_start.month + 1, 1)
    return month_start, month_end


def iter_days(month_start: date, month_end: date) -> list[date]:
    days: list[date] = []
    current = month_start
    while current < month_end:
        days.append(current)
        current += timedelta(days=1)
    return days


def filter_days(days: list[date], from_date: str, to_date: str) -> list[date]:
    start = date.fromisoformat(from_date) if from_date else None
    end = date.fromisoformat(to_date) if to_date else None
    filtered: list[date] = []
    for day_value in days:
        if start and day_value < start:
            continue
        if end and day_value > end:
            continue
        filtered.append(day_value)
    return filtered


def parse_single_day(value: str) -> date:
    text = value.strip()
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError("Daily test dates must use DDMMYY format, for example 050426.")
    return datetime.strptime(text, "%d%m%y").date()


def bool_literal(value: bool) -> str:
    return "true" if value else "false"


def ini_leverage_value(value: str) -> str:
    text = str(value).strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    return text or "200"


def set_line(name: str, value: str) -> str:
    return f"{name}={value}||{value}||0||{value}||N"


def current_log_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


def log_offsets(paths: list[Path]) -> dict[Path, int]:
    offsets: dict[Path, int] = {}
    for path in paths:
        if path.exists():
            offsets[path] = path.stat().st_size
        else:
            offsets[path] = 0
    return offsets


def read_appended_text(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    for encoding in ("utf-16", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def parse_summary_text(text: str) -> dict[str, int | float | str]:
    match = SUMMARY_PATTERN.search(text)
    if not match:
        return {}
    values: dict[str, int | float | str] = {}
    for token in match.group(1).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if NUMBER_PATTERN.match(value):
            numeric = float(value)
            values[key] = int(numeric) if numeric.is_integer() else numeric
        elif BOOL_PATTERN.match(value):
            values[key] = value.lower() == "true"
        else:
            values[key] = value
    return values


def parse_result(day_value: date, tester_text: str, agent_text: str) -> BacktestResult:
    initial_deposit = 0.0
    final_balance = 0.0
    currency = ""
    leverage = ""
    ticks = 0
    bars = 0
    duration = ""
    from_date = day_value.strftime("%Y.%m.%d")
    to_date = (day_value + timedelta(days=1)).strftime("%Y.%m.%d")
    error = ""

    for line in agent_text.splitlines():
        initial_match = INITIAL_DEPOSIT_PATTERN.search(line)
        if initial_match:
            initial_deposit = float(initial_match.group(1))
            currency = initial_match.group(2)
            leverage = initial_match.group(3)

        final_match = FINAL_BALANCE_PATTERN.search(line)
        if final_match:
            final_balance = float(final_match.group(1))
            if not currency:
                currency = final_match.group(2)

        generated_match = GENERATED_PATTERN.search(line)
        if generated_match:
            ticks = int(generated_match.group(1))
            bars = int(generated_match.group(2))
            duration = generated_match.group(3)

        status_match = STATUS_LINE_PATTERN.search(line)
        if status_match:
            from_date = status_match.group(1)
            to_date = status_match.group(2)

    summary_values = parse_summary_text(agent_text)
    tester_finished = int("automatical testing finished" in tester_text.lower())
    stop_out = int("stop out" in agent_text.lower())

    if not agent_text.strip():
        error = "agent_log_missing"
    elif final_balance == 0.0 and initial_deposit == 0.0 and ticks == 0 and bars == 0:
        error = "summary_not_found"

    status = "ok"
    if error:
        status = "error"
    elif ticks == 0 and bars == 0:
        status = "no_data"
    elif stop_out:
        status = "stop_out"

    return BacktestResult(
        day=day_value.isoformat(),
        status=status,
        from_date=from_date,
        to_date=to_date,
        initial_deposit=initial_deposit,
        final_balance=final_balance,
        profit=final_balance - initial_deposit,
        currency=currency,
        leverage=leverage,
        ticks=ticks,
        bars=bars,
        duration=duration,
        predictions=int(summary_values.get("predictions", 0)),
        hold_skips=int(summary_values.get("hold_skips", 0)),
        confidence_skips=int(summary_values.get("confidence_skips", 0)),
        position_skips=int(summary_values.get("position_skips", 0)),
        stops_too_close=int(summary_values.get("stops_too_close", 0)),
        open_failures=int(summary_values.get("open_failures", 0)),
        trades_opened=int(summary_values.get("trades_opened", 0)),
        trades_closed=int(summary_values.get("trades_closed", 0)),
        wins=int(summary_values.get("wins", 0)),
        losses=int(summary_values.get("losses", 0)),
        realized_pnl=float(summary_values.get("realized_pnl", 0.0)),
        risk_mode=str(summary_values.get("risk_mode", "")),
        fixed_move=float(summary_values.get("fixed_move", 0.0)),
        stop_out=stop_out,
        tester_finished=tester_finished,
        error=error,
    )


def write_csv(path: Path, rows: list[BacktestResult]) -> None:
    fieldnames = list(BacktestResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_report(path: Path, scope_label: str, rows: list[BacktestResult], daily_mode: bool) -> None:
    total_profit = sum(row.profit for row in rows)
    profitable_days = sum(1 for row in rows if row.profit > 0.0)
    losing_days = sum(1 for row in rows if row.profit < 0.0)
    no_data_days = sum(1 for row in rows if row.status == "no_data")
    stop_out_days = sum(1 for row in rows if row.stop_out)
    total_trades = sum(row.trades_opened for row in rows)
    total_closed = sum(row.trades_closed for row in rows)
    total_predictions = sum(row.predictions for row in rows)
    total_realized = sum(row.realized_pnl for row in rows)

    lines = [
        "# Daily Backtest Report",
        "",
        f"- {'day' if daily_mode else 'month'}: {scope_label}",
        f"- days_tested: {len(rows)}",
        f"- profitable_days: {profitable_days}",
        f"- losing_days: {losing_days}",
        f"- no_data_days: {no_data_days}",
        f"- stop_out_days: {stop_out_days}",
        f"- total_profit: {total_profit:.2f}",
        f"- total_realized_pnl: {total_realized:.2f}",
        f"- total_predictions: {total_predictions}",
        f"- total_trades_opened: {total_trades}",
        f"- total_trades_closed: {total_closed}",
        "",
        "| day | status | profit | final_balance | trades_opened | trades_closed | wins | losses | ticks | bars | risk_mode | error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f"| {row.day} | {row.status} | {row.profit:.2f} | {row.final_balance:.2f} | "
            f"{row.trades_opened} | {row.trades_closed} | {row.wins} | {row.losses} | "
            f"{row.ticks} | {row.bars} | {row.risk_mode or '-'} | {row.error or '-'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_set_file(path: Path, config: dict[str, int | float | str | bool]) -> None:
    model_uses_atr = bool(int(config.get("MODEL_USE_ATR_RISK", 1)))
    contents = "\n".join(
        [
            f"; auto-generated by {Path(__file__).name} on {datetime.now().strftime('%Y.%m.%d %H:%M:%S')}",
            set_line("R", bool_literal(not model_uses_atr)),
            set_line("FIXED_MOVE", f"{float(config['DEFAULT_FIXED_MOVE']):.2f}"),
            set_line("SL_MULTIPLIER", f"{float(config['DEFAULT_SL_MULTIPLIER']):.2f}"),
            set_line("TP_MULTIPLIER", f"{float(config['DEFAULT_TP_MULTIPLIER']):.2f}"),
            set_line("LOT_SIZE", f"{float(config['DEFAULT_LOT_SIZE']):.2f}"),
            set_line("LOT_SIZE_CAP", f"{float(config['DEFAULT_LOT_SIZE_CAP']):.2f}"),
            set_line("RISK_PERCENT", f"{float(config['DEFAULT_RISK_PERCENT']):.2f}"),
            set_line("BROKER_MIN_LOT_SIZE", f"{float(config['DEFAULT_BROKER_MIN_LOT_SIZE']):.2f}"),
            set_line("USE_BROKER_MIN_LOT", bool_literal(bool(config.get("USE_BROKER_MIN_LOT_SIZE", False)))),
            set_line("USE_LOT_SIZE_CAP_INPUT", bool_literal(bool(config.get("USE_LOT_SIZE_CAP", False)))),
            set_line("USE_RISK_PERCENT_INPUT", bool_literal(bool(config.get("USE_RISK_PERCENT", False)))),
            set_line("MAGIC_NUMBER", "777777"),
            set_line("DEBUG_LOG", "false"),
            "",
        ]
    )
    path.write_text(contents, encoding="ascii")


def build_ini_file(
    path: Path,
    set_name: str,
    day_value: date,
    deposit: float,
    currency: str,
    leverage: str,
    symbol: str,
) -> None:
    contents = "\n".join(
        [
            "[Tester]",
            f"Expert={PROJECT_DIR_NAME}\\live.ex5",
            f"ExpertParameters={set_name}",
            f"Symbol={symbol}",
            "Period=H1",
            "Model=4",
            f"FromDate={day_value.strftime('%Y.%m.%d')}",
            f"ToDate={(day_value + timedelta(days=1)).strftime('%Y.%m.%d')}",
            "Optimization=0",
            "ExecutionMode=0",
            "Visual=0",
            "ReplaceReport=1",
            "ShutdownTerminal=1",
            f"Deposit={deposit:.2f}",
            f"Currency={currency}",
            f"Leverage={ini_leverage_value(leverage)}",
            "UseLocal=1",
            "UseRemote=0",
            "UseCloud=0",
            "",
            "[Experts]",
            "AllowLiveTrading=1",
            "",
        ]
    )
    path.write_text(contents, encoding="ascii")


def wait_for_tester_completion(main_log_path: Path, offset: int, timeout_seconds: int) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        tester_text = read_appended_text(main_log_path, offset)
        tester_text_lower = tester_text.lower()
        if (
            "automatical testing finished" in tester_text_lower
            or "thread finished" in tester_text_lower
            or "stop testing" in tester_text_lower
        ):
            return tester_text
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for tester completion in {main_log_path}.")


def merged_test_config(args: argparse.Namespace, default_symbol: str, model_dir: Path) -> dict[str, int | float | str]:
    tests_dir = model_tests_dir(model_dir)
    config_path = ensure_default_test_config(tests_dir, symbol=default_symbol)
    base = load_test_config(config_path)
    symbol = args.symbol or str(base.get("symbol", default_symbol)) or default_symbol
    return {
        "month": args.month or str(base.get("month", "")),
        "from_date": args.from_date or str(base.get("from_date", "")),
        "to_date": args.to_date or str(base.get("to_date", "")),
        "symbol": symbol,
        "deposit": float(args.deposit if args.deposit is not None else base.get("deposit", 10000.0)),
        "currency": str(args.currency or base.get("currency", "USD")),
        "leverage": str(args.leverage or base.get("leverage", "1:2000")),
        "timeout_seconds": int(
            args.timeout_seconds if args.timeout_seconds is not None else base.get("timeout_seconds", 600)
        ),
        "retries": int(args.retries if args.retries is not None else base.get("retries", 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test an archived model with the MT5 daily backtest pipeline.")
    parser.add_argument(
        "-i",
        "--symbol",
        type=str,
        default="",
        help="Symbol/model folder to test. Defaults to SYMBOL from config.mqh.",
    )
    parser.add_argument(
        "-r",
        "--revision",
        type=str,
        default="",
        help="Model training timestamp in DD_MM_YYYY-HH_MM__SS. If blank, test the latest model for the symbol.",
    )
    parser.add_argument("--month", type=str, default="", help="Month in YYYY-MM format. Defaults to config or last month.")
    parser.add_argument("--from-date", type=str, default="", help="Optional first day to run, in YYYY-MM-DD.")
    parser.add_argument("--to-date", type=str, default="", help="Optional last day to run, in YYYY-MM-DD.")
    parser.add_argument(
        "-d",
        "--day",
        nargs="?",
        const="",
        default=None,
        metavar="DDMMYY",
        help="Run a single-day backtest. Optionally pass DDMMYY; if omitted, it defaults to the previous day.",
    )
    parser.add_argument("--deposit", type=float, default=None, help="Initial deposit for each daily test.")
    parser.add_argument("--currency", type=str, default="", help="Deposit currency.")
    parser.add_argument("--leverage", type=str, default="", help="Tester leverage, for example 1:2000.")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Maximum wait time per daily run.")
    parser.add_argument("--retries", type=int, default=None, help="Retries per day after a launch/logging failure.")
    parser.add_argument(
        "--metaeditor-path",
        type=str,
        default="",
        help="Optional explicit MetaEditor path. Leave blank to auto-detect on Windows or Linux/Wine.",
    )
    parser.add_argument(
        "--skip-live-compile",
        action="store_true",
        help="Skip compiling live.mq5 after activating the chosen archived model.",
    )
    parser.add_argument(
        "--instance-root",
        type=str,
        default="",
        help="Override the MT5 terminal data root that contains MQL5/ and Tester/.",
    )
    parser.add_argument(
        "--terminal-path",
        type=str,
        default="",
        help="Optional explicit path to terminal64.exe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol or configured_symbol()
    model_dir = resolve_model_dir(symbol, args.revision)
    test_config = merged_test_config(args, default_symbol=symbol, model_dir=model_dir)
    runtime = resolve_mt5_runtime(
        instance_root_override=args.instance_root,
        terminal_path_override=args.terminal_path,
        metaeditor_path_override=args.metaeditor_path,
    )

    activate_model(model_dir)

    daily_mode = args.day is not None
    single_day = None
    if daily_mode:
        single_day = parse_single_day(args.day) if args.day else (date.today() - timedelta(days=1))

    run_stamp = format_model_stamp()
    run_dir_name = f"{run_stamp} d" if daily_mode else run_stamp
    run_dir = model_tests_dir(model_dir) / run_dir_name
    config_dir = run_dir / "configs"
    raw_log_dir = run_dir / "raw_logs"
    compile_dir = run_dir / "compile"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_live_compile:
        compile_log_path = compile_live_expert(runtime, model_dir=model_dir)
        (compile_dir / compile_log_path.name).write_text(
            read_text_best_effort(compile_log_path),
            encoding="utf-8",
        )

    config = load_define_file(model_config_path(model_dir))
    if daily_mode:
        if single_day is None:
            raise RuntimeError("Daily mode failed to resolve a test date.")
        report_scope = single_day.isoformat()
        day_values = [single_day]
    else:
        month_value = str(test_config["month"]) or None
        month_start, month_end = parse_month(month_value)
        report_scope = month_start.strftime("%Y-%m")
        day_values = filter_days(
            iter_days(month_start, month_end),
            str(test_config["from_date"]),
            str(test_config["to_date"]),
        )

    set_name = f"{sanitize_symbol(str(test_config['symbol']))}_daily_backtest_{run_stamp}.set"
    set_path = runtime.tester_profile_dir / set_name
    build_set_file(set_path, config=config)
    (run_dir / "tester_inputs.set").write_text(set_path.read_text(encoding="ascii"), encoding="ascii")
    (run_dir / "config.mqh").write_text(model_config_path(model_dir).read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "backtest_config_snapshot.json").write_text(
        json.dumps(test_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    today_stamp = current_log_stamp()
    main_log_path = runtime.terminal_log_dir / f"{today_stamp}.log"

    rows: list[BacktestResult] = []
    for day_value in day_values:
        if not runtime.terminal_path.exists():
            raise FileNotFoundError(f"terminal64.exe not found at {runtime.terminal_path}")

        config_path = config_dir / f"{day_value.isoformat()}.ini"
        launch_config_dir = Path(tempfile.gettempdir()) / "mt5_tester_configs"
        launch_config_dir.mkdir(parents=True, exist_ok=True)
        launch_config_path = launch_config_dir / f"{sanitize_symbol(str(test_config['symbol']))}_{day_value.isoformat()}.ini"
        build_ini_file(
            path=config_path,
            set_name=set_name,
            day_value=day_value,
            deposit=float(test_config["deposit"]),
            currency=str(test_config["currency"]),
            leverage=str(test_config["leverage"]),
            symbol=str(test_config["symbol"]),
        )
        launch_config_path.write_text(config_path.read_text(encoding="ascii"), encoding="ascii")

        last_error = ""
        day_result: BacktestResult | None = None
        tester_log_output = raw_log_dir / f"{day_value.isoformat()}_tester.log"
        agent_log_output = raw_log_dir / f"{day_value.isoformat()}_agent.log"
        for _attempt in range(int(test_config["retries"]) + 1):
            tester_offsets = log_offsets([main_log_path])
            agent_log_paths = list(iter_agent_log_paths(runtime))
            agent_offsets = log_offsets(agent_log_paths)
            try:
                launch_mt5_terminal(
                    runtime,
                    launch_config_path,
                    timeout_seconds=int(test_config["timeout_seconds"]),
                    detach=True,
                    stop_existing=True,
                )
                tester_text = wait_for_tester_completion(
                    main_log_path=main_log_path,
                    offset=tester_offsets.get(main_log_path, 0),
                    timeout_seconds=int(test_config["timeout_seconds"]),
                )
                time.sleep(1.0)
                agent_chunks: list[str] = []
                for path in iter_agent_log_paths(runtime):
                    agent_chunks.append(read_appended_text(path, agent_offsets.get(path, 0)))
                agent_text = "\n".join(chunk for chunk in agent_chunks if chunk)

                tester_log_output.write_text(tester_text, encoding="utf-8")
                agent_log_output.write_text(agent_text, encoding="utf-8")

                day_result = parse_result(day_value=day_value, tester_text=tester_text, agent_text=agent_text)
                if day_result.error == "agent_log_missing":
                    raise RuntimeError(day_result.error)
                break
            except Exception as exc:
                last_error = str(exc)
                tester_text = read_appended_text(main_log_path, tester_offsets.get(main_log_path, 0))
                tester_log_output.write_text(tester_text, encoding="utf-8")
                agent_chunks = []
                for path in iter_agent_log_paths(runtime):
                    agent_chunks.append(read_appended_text(path, agent_offsets.get(path, 0)))
                agent_text = "\n".join(chunk for chunk in agent_chunks if chunk)
                agent_log_output.write_text(agent_text, encoding="utf-8")
                time.sleep(2.0)
        if day_result is None:
            day_result = error_result(day_value, last_error or "unknown_error")

        rows.append(day_result)
        write_csv(run_dir / "daily_results.csv", rows)
        write_report(run_dir / "report.md", scope_label=report_scope, rows=rows, daily_mode=daily_mode)

    write_csv(run_dir / "daily_results.csv", rows)
    write_report(run_dir / "report.md", scope_label=report_scope, rows=rows, daily_mode=daily_mode)
    print(run_dir)


if __name__ == "__main__":
    main()
