# Trade Bot Pipeline Notes

## Layout

```
trade-bot/
├── config/          # Config presets + active config pointer
│   ├── active.mqh   # Active training config (replaces root config.mqh)
│   ├── .active_config   # Points to a config preset by path
│   ├── au.config    # XAUUSD HFT preset
│   ├── bitcoin.config
│   ├── testrun.config
│   └── config.mqh   # Default XAUUSD preset (copy to active.mqh to use)
├── scripts/         # CLI entry-points (run from repo root)
│   ├── nn.py
│   ├── test.py
│   ├── export_data.py
│   ├── inspect_bars.py
│   ├── join_files.py
│   ├── minirocket_search.py
│   ├── move_ticks.py
│   ├── i.sh         # Train + test pipeline (Linux/Mac)
│   └── i.ps1        # Train + test pipeline (Windows)
├── mt5/             # MetaTrader 5 artefacts
│   ├── scripts/     # MQL5 data-export scripts (data.mq5, data_gold.mq5)
│   ├── compiled/    # Compiled .ex5 binaries
│   ├── logs/        # Compile logs
│   └── config/      # MT5 startup ini templates
├── live/            # Live EA source
│   ├── live.mq5     # Main expert file
│   └── functions/   # Included .mqh function files
├── common/          # Shared Python modules (features, bars, config parsing)
├── tradebot/        # Python pipeline (training, export, workspace, CLI impls)
├── symbols/         # Per-symbol model archives and configs
├── data/            # Tick CSV files (gitignored)
├── diagnostics/     # Latest model diagnostics snapshot
├── artifacts/       # Misc build artefacts (model.onnx staging)
├── docs/notes/      # Dev notes, todos, request files
├── meta/            # skills-lock.json and other meta files
├── env/             # Python virtualenv (gitignored)
├── .gitignore
├── .gitattributes
├── .python-version
├── requirements.txt
├── AGENTS.md
└── README.md
```

## Scripts

Run all scripts from the **repo root** directory:

```
python scripts/nn.py            # Train, export ONNX, update live.mq5, compile EA
python scripts/inspect_bars.py  # Show bar count for active config without training
python scripts/test.py          # Run archived-model backtests
python scripts/export_data.py --profile gold   # Export XAUUSD ticks + USDX/USDJPY context
python scripts/move_ticks.py -i market_ticks.csv  # Move tick file from MT5 Files/
```

The shell pipeline helper (`scripts/i.sh` or `scripts/i.ps1`) trains and immediately backtests:

```
./scripts/i.sh [-i days]        # Linux / macOS
.\scripts\i.ps1 [-Days N]       # Windows PowerShell
```

## Active Config

The active config is resolved in this order:

1. `config/.active_config` — if it exists, its first line is read as a path (absolute or relative to repo root) pointing to the config preset to use.
2. `config/active.mqh` — fallback if `.active_config` is absent or empty.

To switch config:

```
echo "config/au.config" > config/.active_config
```

## Features

### Static features
All static feature names (e.g. `ret1`, `rsi_9`, `atr_rel`) are defined in `common/feature_columns.py`
and toggled per-config with `#define FEATURE_<NAME> true`.

### PAST_DIR_* features (dynamic price-direction)

Add these to any config to include price-direction-since-N lookback features:

```cpp
#define PAST_DIR_5400_S true   // price direction over the past 5400 seconds
#define PAST_DIR_60_S   true   // price direction over the past 60 seconds
#define PAST_DIR_27_T   true   // price direction over the past 27 bars
#define PAST_DIR_9_T    true   // price direction over the past 9 bars
```

**Suffix `_S`** = wall-clock seconds lookback (converted to bars at training time using `PRIMARY_BAR_SECONDS`).  
**Suffix `_T`** = bar-count lookback (direct).

**Value encoding:** `tanh(log(close_now / close_then))`

- Near **-1**: price dropped strongly since that point.
- **0**: flat.
- Near **+1**: price rose strongly.

`tanh` of the log-return is the best choice for NNs: symmetric around 0, naturally bounded in (-1, 1), and saturation is gradual (no hard clipping artefacts).

Supported values in the live EA: `_T` — 1, 2, 3, 5, 9, 12, 18, 27, 36, 54, 72, 100, 144, 200, 288, 360;  
`_S` — 60, 120, 300, 600, 900, 1800, 3600, 5400, 7200, 10800, 14400, 21600, 43200, 86400.

## Fixed-Point Target SL/TP

When `USE_FIXED_TARGETS true`, the training labels use a fixed price distance for both the stop-loss and take-profit barrier. You can now define them **separately**:

```cpp
#define DEFAULT_FIXED_MOVE 300    // fallback for both if the below are absent
#define DEFAULT_FIXED_SL   200    // label SL barrier (points); optional
#define DEFAULT_FIXED_TP   400    // label TP barrier (points); optional
```

The live EA exposes separate `FIXED_SL` and `FIXED_TP` inputs (both default to `DEFAULT_FIXED_MOVE` if their dedicated defines are absent). The ATR multipliers (`LABEL_SL_MULTIPLIER`, `LABEL_TP_MULTIPLIER`) are unchanged and still used in ATR mode.

## Model Archive Layout

```
symbols/<SYMBOL>/models/<stamp>-<name>/
├── model.onnx
├── config.mqh      # Full combined config used for this model
└── diagnostics/
    ├── report.md
    ├── bars.csv
    ├── active_features.txt
    ├── validation_predictions.csv
    ├── holdout_predictions.csv
    ├── validation_confusion_matrix.csv
    └── holdout_confusion_matrix.csv
```
