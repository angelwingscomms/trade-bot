from __future__ import annotations

from .shared import *  # noqa: F401,F403


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
            f"Expert={PROJECT_DIR_NAME}\\mt5\\compiled\\live.ex5",
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
