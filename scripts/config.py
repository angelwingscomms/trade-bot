from __future__ import annotations

import os
import sys
import tty
import termios
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"


@dataclass
class ConfigValue:
    name: str
    value: str
    default_value: str
    description: str = ""


@dataclass
class ConfigCategory:
    name: str
    subcategories: dict[str, "ConfigCategory | list[ConfigValue]"] = field(default_factory=dict)
    config_values: list[ConfigValue] = field(default_factory=list)


def load_existing_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("#define "):
            parts = line[8:].split(None, 1)
            if len(parts) == 2:
                values[parts[0]] = parts[1]
    return values


def parse_config_spec() -> ConfigCategory:
    def make_bool(name: str, default: bool, desc: str) -> ConfigValue:
        return ConfigValue(
            name=name,
            value="true" if default else "false",
            default_value="true" if default else "false",
            description=desc,
        )

    def make_int(name: str, default: int, desc: str) -> ConfigValue:
        return ConfigValue(name=name, value=str(default), default_value=str(default), description=desc)

    def make_float(name: str, default: float, desc: str) -> ConfigValue:
        return ConfigValue(name=name, value=str(default), default_value=str(default), description=desc)

    def make_str(name: str, default: str, desc: str) -> ConfigValue:
        return ConfigValue(name=name, value=f'"{default}"', default_value=f'"{default}"', description=desc)

    root = ConfigCategory(name="root")

    root.subcategories["Project"] = ConfigCategory(
        name="Project",
        config_values=[
            make_str("ARCHITECTURE_CONFIG", "symbols/xauusd/config/gold.config", "Architecture config file path"),
            make_str("DATA_FILE", "data/XAUUSD/ticks.csv", "Input data file path"),
            make_str("MODEL_NAME", "xau-default", "Model name identifier"),
            make_str("SYMBOL", "XAUUSD", "Trading symbol"),
        ],
    )

    minimal_features = ConfigCategory(
        name="Minimal Features",
        config_values=[
            make_bool("MINIMAL_FEATURE_ATR_REL", True, "ATR relative feature"),
            make_bool("MINIMAL_FEATURE_CLOSE_IN_RANGE", True, "Close in range feature"),
            make_bool("MINIMAL_FEATURE_HIGH_REL_PREV", True, "High relative to previous"),
            make_bool("MINIMAL_FEATURE_LOW_REL_PREV", True, "Low relative to previous"),
            make_bool("MINIMAL_FEATURE_RET1", True, "Return over 1 bar"),
            make_bool("MINIMAL_FEATURE_RETURN_N", True, "Return over N bars"),
            make_bool("MINIMAL_FEATURE_RV", True, "Realized volatility"),
            make_bool("MINIMAL_FEATURE_SPREAD_REL", True, "Relative spread"),
            make_bool("MINIMAL_FEATURE_TICK_IMBALANCE", True, "Tick imbalance"),
        ],
    )
    minimal_features.subcategories["Periods"] = ConfigCategory(
        name="Periods",
        config_values=[
            make_int("MINIMAL_FEATURE_RET_N_PERIOD", 9, "Return N period"),
        ],
    )
    root.subcategories["Features"] = ConfigCategory(
        name="Features",
        subcategories={
            "Minimal": minimal_features,
            "Main": ConfigCategory(
                name="Main",
                config_values=[
                    make_bool("USE_MAIN_FEATURE_SET", False, "Use main feature set"),
                    make_bool("USE_MINIMAL_FEATURE_SET", False, "Use minimal feature set"),
                ],
            ),
            "Context": ConfigCategory(
                name="Context",
                config_values=[
                    make_bool("USE_GOLD_CONTEXT", True, "Use gold context features"),
                ],
            ),
            "SMA": ConfigCategory(
                name="SMA",
                config_values=[
                    make_bool("FEATURE_SMA_3_9_GAP", True, "SMA 3/9 gap"),
                    make_bool("FEATURE_SMA_5_20_GAP", True, "SMA 5/20 gap"),
                    make_bool("FEATURE_SMA_9_20_GAP", True, "SMA 9/20 gap"),
                    make_bool("FEATURE_SMA_SLOPE_20", True, "SMA slope 20"),
                    make_bool("FEATURE_SMA_SLOPE_9", True, "SMA slope 9"),
                    make_bool("FEATURE_CLOSE_REL_SMA_3", True, "Close rel SMA 3"),
                    make_bool("FEATURE_CLOSE_REL_SMA_9", True, "Close rel SMA 9"),
                    make_bool("FEATURE_CLOSE_REL_SMA_20", True, "Close rel SMA 20"),
                ],
            ),
            "Periods": ConfigCategory(
                name="Periods",
                config_values=[
                    make_int("FEATURE_SMA_FAST_PERIOD", 3, "SMA fast period"),
                    make_int("FEATURE_SMA_MID_PERIOD", 9, "SMA medium period"),
                    make_int("FEATURE_SMA_SLOW_PERIOD", 20, "SMA slow period"),
                    make_int("FEATURE_SMA_TREND_FAST_PERIOD", 5, "SMA trend fast period"),
                    make_int("FEATURE_SMA_SLOPE_SHIFT", 3, "SMA slope shift"),
                    make_int("FEATURE_MAIN_SHORT_PERIOD", 9, "Main short period"),
                    make_int("FEATURE_MAIN_MEDIUM_PERIOD", 18, "Main medium period"),
                    make_int("FEATURE_MAIN_LONG_PERIOD", 27, "Main long period"),
                    make_int("FEATURE_MAIN_XLONG_PERIOD", 54, "Main xlong period"),
                ],
            ),
            "Normalization": ConfigCategory(
                name="Normalization",
                config_values=[
                    make_bool("FEATURE_CLOSE_Z_250", True, "Close Z-score 250"),
                    make_bool("FEATURE_RET_Z_250", True, "Return Z-score 250"),
                    make_int("FEATURE_NORMALIZE_PERIOD", 270, "Normalize period"),
                ],
            ),
            "Bollinger": ConfigCategory(
                name="Bollinger",
                config_values=[
                    make_bool("FEATURE_BOLLINGER_POS_20", True, "Bollinger position 20"),
                    make_bool("FEATURE_BOLLINGER_WIDTH_20", True, "Bollinger width 20"),
                    make_int("FEATURE_BOLLINGER_PERIOD", 20, "Bollinger period"),
                ],
            ),
            "Donchian": ConfigCategory(
                name="Donchian",
                config_values=[
                    make_bool("FEATURE_DONCHIAN_POS_9", True, "Donchian position 9"),
                    make_bool("FEATURE_DONCHIAN_POS_20", True, "Donchian position 20"),
                    make_bool("FEATURE_DONCHIAN_WIDTH_9", True, "Donchian width 9"),
                    make_bool("FEATURE_DONCHIAN_WIDTH_20", True, "Donchian width 20"),
                    make_int("FEATURE_DONCHIAN_FAST_PERIOD", 9, "Donchian fast period"),
                    make_int("FEATURE_DONCHIAN_SLOW_PERIOD", 20, "Donchian slow period"),
                ],
            ),
            "ATR": ConfigCategory(
                name="ATR",
                config_values=[
                    make_bool("FEATURE_ATR_REL", True, "ATR relative"),
                    make_bool("FEATURE_ATR_RATIO_20", True, "ATR ratio 20"),
                    make_int("FEATURE_ATR_PERIOD", 9, "ATR period"),
                    make_int("FEATURE_ATR_RATIO_PERIOD", 20, "ATR ratio period"),
                    make_int("TARGET_ATR_PERIOD", 9, "Target ATR period"),
                ],
            ),
            "Stochastic": ConfigCategory(
                name="Stochastic",
                config_values=[
                    make_bool("FEATURE_STOCH_D_3", True, "Stochastic D 3"),
                    make_bool("FEATURE_STOCH_GAP", True, "Stochastic gap"),
                    make_bool("FEATURE_STOCH_K_9", True, "Stochastic K 9"),
                    make_int("FEATURE_STOCH_PERIOD", 9, "Stochastic period"),
                    make_int("FEATURE_STOCH_SMOOTH_PERIOD", 3, "Stochastic smooth period"),
                ],
            ),
            "RSI": ConfigCategory(
                name="RSI",
                config_values=[
                    make_bool("FEATURE_RSI_14", True, "RSI 14"),
                    make_bool("FEATURE_RSI_6", True, "RSI 6"),
                    make_int("FEATURE_RSI_SLOW_PERIOD", 14, "RSI slow period"),
                    make_int("FEATURE_RSI_FAST_PERIOD", 6, "RSI fast period"),
                ],
            ),
            "Returns": ConfigCategory(
                name="Returns",
                config_values=[
                    make_bool("FEATURE_RET1", True, "Return 1 bar"),
                    make_bool("FEATURE_RET_2", True, "Return 2 bars"),
                    make_bool("FEATURE_RET_3", True, "Return 3 bars"),
                    make_bool("FEATURE_RET_6", True, "Return 6 bars"),
                    make_bool("FEATURE_RET_12", True, "Return 12 bars"),
                    make_bool("FEATURE_RET_20", True, "Return 20 bars"),
                    make_bool("FEATURE_RETURN_N", True, "Return N bars"),
                    make_int("FEATURE_RET_2_PERIOD", 2, "Return 2 period"),
                    make_int("FEATURE_RET_3_PERIOD", 3, "Return 3 period"),
                    make_int("FEATURE_RET_6_PERIOD", 6, "Return 6 period"),
                    make_int("FEATURE_RET_12_PERIOD", 12, "Return 12 period"),
                    make_int("FEATURE_RET_20_PERIOD", 20, "Return 20 period"),
                    make_int("RETURN_PERIOD", 9, "Return period"),
                ],
            ),
            "RV": ConfigCategory(
                name="RV",
                config_values=[
                    make_bool("FEATURE_RV", True, "Realized volatility"),
                    make_bool("FEATURE_RV_18", True, "RV 18"),
                    make_int("FEATURE_RV_LONG_PERIOD", 18, "RV long period"),
                    make_int("RV_PERIOD", 9, "RV period"),
                ],
            ),
            "Tick": ConfigCategory(
                name="Tick",
                config_values=[
                    make_bool("FEATURE_TICK_IMBALANCE", True, "Tick imbalance"),
                    make_bool("FEATURE_TICK_IMBALANCE_SMA_5", True, "Tick imbalance SMA 5"),
                    make_bool("FEATURE_TICK_IMBALANCE_SMA_9", True, "Tick imbalance SMA 9"),
                    make_bool("FEATURE_TICK_COUNT_CHG", True, "Tick count change"),
                    make_bool("FEATURE_TICK_COUNT_REL_9", True, "Tick count relative 9"),
                    make_bool("FEATURE_TICK_COUNT_Z_9", True, "Tick count Z 9"),
                    make_int("FEATURE_TICK_COUNT_PERIOD", 9, "Tick count period"),
                    make_int("FEATURE_TICK_IMBALANCE_FAST_PERIOD", 5, "Tick imbalance fast period"),
                    make_int("FEATURE_TICK_IMBALANCE_SLOW_PERIOD", 9, "Tick imbalance slow period"),
                ],
            ),
            "Spread": ConfigCategory(
                name="Spread",
                config_values=[
                    make_bool("FEATURE_SPREAD_REL", True, "Spread relative"),
                    make_bool("FEATURE_SPREAD_Z_9", True, "Spread Z 9"),
                    make_int("FEATURE_SPREAD_Z_PERIOD", 9, "Spread Z period"),
                ],
            ),
            "Candle": ConfigCategory(
                name="Candle",
                config_values=[
                    make_bool("FEATURE_CLOSE_IN_RANGE", True, "Close in range"),
                    make_bool("FEATURE_HIGH_REL_PREV", True, "High relative prev"),
                    make_bool("FEATURE_LOW_REL_PREV", True, "Low relative prev"),
                    make_bool("FEATURE_OPEN_REL_PREV", True, "Open relative prev"),
                    make_bool("FEATURE_RANGE_REL", True, "Range relative"),
                    make_bool("FEATURE_BODY_REL", True, "Body relative"),
                    make_bool("FEATURE_LOWER_WICK_REL", True, "Lower wick relative"),
                    make_bool("FEATURE_UPPER_WICK_REL", True, "Upper wick relative"),
                ],
            ),
            "MACD": ConfigCategory(
                name="MACD",
                config_values=[
                    make_int("FEATURE_MACD_FAST_PERIOD", 12, "MACD fast period"),
                    make_int("FEATURE_MACD_SLOW_PERIOD", 26, "MACD slow period"),
                    make_int("FEATURE_MACD_SIGNAL_PERIOD", 9, "MACD signal period"),
                ],
            ),
            "External": ConfigCategory(
                name="External",
                config_values=[
                    make_bool("FEATURE_USDJPY_RET1", True, "USDJPY return 1"),
                    make_bool("FEATURE_USDX_RET1", True, "USDX return 1"),
                ],
            ),
        },
    )

    root.subcategories["Trading"] = ConfigCategory(
        name="Trading",
        subcategories={
            "Bars": ConfigCategory(
                name="Bars",
                config_values=[
                    make_int("PRIMARY_BAR_SECONDS", 9, "Primary bar seconds"),
                    make_int("PRIMARY_TICK_DENSITY", 27, "Primary tick density"),
                    make_bool("USE_FIXED_TICK_BARS", True, "Use fixed tick bars"),
                    make_bool("USE_SECOND_BARS", False, "Use second bars"),
                ],
            ),
            "Imbalance": ConfigCategory(
                name="Imbalance",
                config_values=[
                    make_int("IMBALANCE_EMA_SPAN", 3, "Imbalance EMA span"),
                    make_int("IMBALANCE_MIN_TICKS", 3, "Imbalance min ticks"),
                    make_bool("USE_IMBALANCE_EMA_THRESHOLD", True, "Use imbalance EMA threshold"),
                    make_bool("USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD", True, "Use min ticks/3 threshold"),
                ],
            ),
            "Labels": ConfigCategory(
                name="Labels",
                config_values=[
                    make_int("DEFAULT_FIXED_MOVE", 1440, "Default fixed move in points"),
                    make_float("LABEL_SL_MULTIPLIER", 5.4, "Stop loss multiplier"),
                    make_float("LABEL_TP_MULTIPLIER", 5.4, "Take profit multiplier"),
                    make_int("LABEL_TIMEOUT_BARS", 100, "Label timeout bars"),
                    make_bool("USE_FIXED_TARGETS", True, "Use fixed targets"),
                ],
            ),
            "Risk": ConfigCategory(
                name="Risk",
                config_values=[
                    make_float("DEFAULT_LOT_SIZE", 0.54, "Default lot size"),
                    make_float("DEFAULT_LOT_SIZE_CAP", 0.54, "Lot size cap"),
                    make_float("DEFAULT_RISK_PERCENT", 0.0, "Risk percent"),
                    make_float("DEFAULT_SL_MULTIPLIER", 5.4, "SL multiplier"),
                    make_float("DEFAULT_TP_MULTIPLIER", 5.4, "TP multiplier"),
                    make_float("DEFAULT_BROKER_MIN_LOT_SIZE", 0.01, "Broker min lot size"),
                    make_bool("USE_LOT_SIZE_CAP", True, "Use lot size cap"),
                    make_bool("USE_RISK_PERCENT", False, "Use risk percent"),
                    make_bool("USE_BROKER_MIN_LOT_SIZE", False, "Use broker min lot size"),
                ],
            ),
        },
    )

    root.subcategories["Model"] = ConfigCategory(
        name="Model",
        subcategories={
            "Architecture": ConfigCategory(
                name="Architecture",
                config_values=[
                    make_int("ATTENTION_DIM", 128, "Attention dimension"),
                    make_int("ATTENTION_HEADS", 4, "Attention heads"),
                    make_int("ATTENTION_LAYERS", 2, "Attention layers"),
                    make_float("ATTENTION_DROPOUT", 0.1, "Attention dropout"),
                    make_int("SEQUENCE_HIDDEN_SIZE", 64, "Sequence hidden size"),
                    make_int("SEQUENCE_LAYERS", 2, "Sequence layers"),
                    make_float("SEQUENCE_DROPOUT", 0.1, "Sequence dropout"),
                    make_int("TCN_KERNEL_SIZE", 3, "TCN kernel size"),
                    make_int("TCN_LEVELS", 4, "TCN levels"),
                    make_bool("USE_MULTIHEAD_ATTENTION", True, "Use multihead attention"),
                    make_bool("USE_NO_HOLD", False, "Use no-hold mode"),
                    make_bool("FLIP", False, "Flip buy/sell signals"),
                ],
            ),
            "Training": ConfigCategory(
                name="Training",
                config_values=[
                    make_int("DEFAULT_BATCH_SIZE", 54, "Batch size"),
                    make_int("DEFAULT_EPOCHS", 54, "Epochs"),
                    make_int("DEFAULT_PATIENCE", 144, "Patience"),
                    make_int("DEFAULT_MAX_TRAIN_WINDOWS", 14400, "Max train windows"),
                    make_int("DEFAULT_MAX_EVAL_WINDOWS", 1440, "Max eval windows"),
                    make_float("LEARNING_RATE", 0.001, "Learning rate"),
                    make_float("WEIGHT_DECAY", 0.0, "Weight decay"),
                    make_float("FOCAL_GAMMA", 2.0, "Focal gamma"),
                    make_str("DEFAULT_LOSS_MODE", "cross-entropy", "Loss mode"),
                    make_bool("USE_CUSTOM_LEARNING_RATE", False, "Use custom LR"),
                    make_bool("USE_CUSTOM_WEIGHT_DECAY", False, "Use custom weight decay"),
                    make_bool("USE_ALL_WINDOWS", False, "Use all windows"),
                ],
            ),
            "Search": ConfigCategory(
                name="Search",
                config_values=[
                    make_float("CONFIDENCE_SEARCH_MIN", 0.4, "Confidence search min"),
                    make_float("CONFIDENCE_SEARCH_MAX", 0.99, "Confidence search max"),
                    make_int("CONFIDENCE_SEARCH_STEPS", 60, "Confidence search steps"),
                    make_int("MIN_SELECTED_TRADES", 12, "Min selected trades"),
                    make_float("MIN_TRADE_PRECISION", 0.5, "Min trade precision"),
                    make_bool("USE_CONFIDENCE_THRESHOLD", True, "Use confidence threshold"),
                ],
            ),
            "Chronos": ConfigCategory(
                name="Chronos",
                config_values=[
                    make_str("CHRONOS_BOLT_MODEL", "amazon/chronos-bolt-tiny", "Chronos bolt model"),
                    make_bool("USE_CHRONOS_AUTO_CONTEXT", False, "Use chronos auto context"),
                    make_bool("USE_CHRONOS_PATCH_ALIGNED_CONTEXT", False, "Use chronos patch aligned"),
                    make_bool("USE_CHRONOS_ENSEMBLE_CONTEXTS", False, "Use chronos ensemble contexts"),
                ],
            ),
            "MiniRocket": ConfigCategory(
                name="MiniRocket",
                config_values=[
                    make_int("MINIROCKET_FEATURES", 10080, "MiniRocket features count"),
                ],
            ),
            "Device": ConfigCategory(
                name="Device",
                config_values=[
                    make_str("DEVICE", "cpu", "Compute device"),
                    make_bool("USE_MAX_BARS", False, "Use max bars"),
                    make_int("MAX_BARS", 0, "Max bars"),
                    make_int("SEQ_LEN", 144, "Sequence length"),
                ],
            ),
            "Compilation": ConfigCategory(
                name="Compilation",
                config_values=[
                    make_bool("SKIP_LIVE_COMPILE", False, "Skip live compile"),
                    make_str("METAEDITOR_PATH", "", "MetaEditor path"),
                ],
            ),
        },
    )

    return root


def flatten_category(cat: ConfigCategory, path: list[str]) -> list[tuple[list[str], ConfigValue]]:
    result: list[tuple[list[str], ConfigValue]] = []
    for cv in cat.config_values:
        result.append((path + [cv.name], cv))
    for sub_name, sub_cat in cat.subcategories.items():
        if isinstance(sub_cat, ConfigCategory):
            result.extend(flatten_category(sub_cat, path + [sub_name]))
    return result


def get_all_configs(cat: ConfigCategory) -> dict[str, ConfigValue]:
    flat = flatten_category(cat, [])
    return {cv.name: cv for _, cv in flat}


def find_subcategory(cat: ConfigCategory, path: list[str]) -> ConfigCategory:
    if not path:
        return cat
    current = cat
    for part in path:
        if part in current.subcategories:
            sub = current.subcategories[part]
            if isinstance(sub, ConfigCategory):
                current = sub
            else:
                raise ValueError(f"{part} is not a category")
        else:
            raise ValueError(f"Cannot find {part}")
    return current


def get_key() -> Optional[str]:
    import select

    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def get_arrow_key() -> Optional[str]:
    import select

    if not select.select([sys.stdin], [], [], 0)[0]:
        return None

    first = sys.stdin.read(1)
    if first != "\x1b":
        return first

    if not select.select([sys.stdin], [], [], 0)[0]:
        return "\x1b"

    second = sys.stdin.read(1)
    if second != "[":
        return first + second

    if not select.select([sys.stdin], [], [], 0)[0]:
        return "\x1b["

    third = sys.stdin.read(1)
    return f"\x1b[{third}"


def prompt_choice(prompt_text: str, options: list[str]) -> int:
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options):
        print(f"  [{i + 1}] {opt}")
    while True:
        try:
            choice = input("> ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return idx
        except (ValueError, EOFError):
            pass
        print("Invalid. Try again.")


def list_config_files() -> list[Path]:
    files = []
    if CONFIG_DIR.exists():
        for f in CONFIG_DIR.iterdir():
            if f.suffix in (".mqh", ".ini"):
                files.append(f)
    for pattern in ["symbols/*/config/*.mqh", "symbols/*/models/*/config.mqh"]:
        files.extend(ROOT_DIR.glob(pattern))
    return sorted(files, key=lambda p: p.name)


def run_config_editor(create_new: bool, config_path: Path | None = None):
    spec = parse_config_spec()
    existing_values: dict[str, str] = {}

    if not create_new and config_path and config_path.exists():
        existing_values = load_existing_config(config_path)
        all_cvs = get_all_configs(spec)
        for name, cv in all_cvs.items():
            if name not in existing_values:
                existing_values[name] = cv.default_value
    else:
        all_cvs = get_all_configs(spec)
        for name, cv in all_cvs.items():
            existing_values[name] = cv.default_value

    current_path: list[str] = []
    cursor = 0
    typing_buffer = ""
    editing_config: Optional[ConfigValue] = None
    input_buffer = ""

    while True:
        current_cat = find_subcategory(spec, current_path)
        subcat_names = sorted(current_cat.subcategories.keys())
        config_list = current_cat.config_values
        all_items = subcat_names + [cv.name for cv in config_list]
        total = len(all_items)

        if total == 0:
            cursor = 0
        elif cursor >= total:
            cursor = total - 1

        os.system("cls" if os.name == "nt" else "clear")

        breadcrumb = " > ".join(current_path) if current_path else "ROOT"
        mode_label = "EDIT" if not create_new else "NEW"
        print(f"\n=== Config Editor [{mode_label}] ===")
        print(f"=== {breadcrumb} ===")
        print()

        if not all_items:
            print("  (empty - press ← to go back)")
        else:
            display_count = min(9, total)
            start = max(0, min(cursor - 4, total - display_count))

            for i in range(start, start + display_count):
                name = all_items[i]
                prefix = ">" if i == cursor else " "
                if i < len(subcat_names):
                    sub = current_cat.subcategories[name]
                    if isinstance(sub, ConfigCategory) and sub.config_values:
                        count = len(get_all_configs(sub))
                        print(f"{prefix} [{name}] ({count} configs)")
                    else:
                        print(f"{prefix} [{name}]")
                else:
                    cv = config_list[i - len(subcat_names)]
                    val = existing_values.get(cv.name, cv.value)
                    print(f"{prefix} {cv.name} = {val}")

        print()
        help_text = "↑/↓:nav  ←:back  →/Enter:enter  Esc:save&exit"
        if editing_config:
            help_text = f"TYPING: {editing_config.name} = {input_buffer}_"
        print(help_text)

        key = get_arrow_key()

        if key is None:
            continue

        if key == "\x1b":
            break
        elif key == "\x1b[A":
            cursor = max(0, cursor - 1)
        elif key == "\x1b[B":
            cursor = min(total - 1, cursor + 1) if total > 0 else 0
        elif key == "\x1b[D":
            if current_path:
                current_path.pop()
                cursor = 0
        elif key == "\x1b[C" or key == "\r":
            if total > 0 and cursor < len(subcat_names):
                current_path.append(subcat_names[cursor])
                cursor = 0
            elif total > 0 and config_list:
                idx = cursor - len(subcat_names)
                if 0 <= idx < len(config_list):
                    editing_config = config_list[idx]
                    input_buffer = existing_values.get(editing_config.name, editing_config.value)
                    typing_buffer = ""
        elif key == "\x7f":
            if editing_config and input_buffer:
                input_buffer = input_buffer[:-1]
                existing_values[editing_config.name] = input_buffer
        elif key == "\r":
            if editing_config:
                existing_values[editing_config.name] = input_buffer
                editing_config = None
                input_buffer = ""
        elif editing_config:
            if key.isprintable():
                input_buffer += key
                existing_values[editing_config.name] = input_buffer
        else:
            if key.isprintable():
                typing_buffer += key.upper()
                for i, name in enumerate(all_items):
                    if name.startswith(typing_buffer):
                        cursor = i
                        break

    saved = config_path or (CONFIG_DIR / "new_config.mqh")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(saved, "w", encoding="utf-8") as f:
        f.write("// Auto-generated config\n\n")
        all_items_flat = flatten_category(spec, [])
        for _, cv in all_items_flat:
            val = existing_values.get(cv.name, cv.value)
            f.write(f"#define {cv.name} {val}\n")

    print(f"\nSaved to {saved}")


def main():
    print("=== Config Editor ===")
    print("This tool creates/edits config files based on Python config definitions.")

    mode_idx = prompt_choice("Create new or edit existing?", ["Create new config", "Edit existing config"])

    if mode_idx == 1:
        files = list_config_files()
        if files:
            print("\nAvailable config files:")
            for i, f in enumerate(files):
                print(f"  [{i + 1}] {f.relative_to(ROOT_DIR)}")
            file_idx = prompt_choice("Select file to edit:", [f.name for f in files])
            selected_path = files[file_idx]
            run_config_editor(create_new=False, config_path=selected_path)
        else:
            print("No existing configs found, creating new.")
            run_config_editor(create_new=True, config_path=None)
    else:
        run_config_editor(create_new=True, config_path=None)


if __name__ == "__main__":
    main()