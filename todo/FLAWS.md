# Pipeline Flaws

Findings from a full codebase audit. Ordered roughly: safety → correctness → reliability → ML/data → ops/deploy.

---

## Safety

### F1 — Failed model can still go live
**File:** `tradebot/training/main.py`  
`quality_gate_passed` can be `False`, yet the code still calls `set_live_model_reference(model_dir)` unconditionally, updates `live.mq5`, and compiles/deploys the EA. A model that failed the precision or trade-count gate silently becomes the live model.  
**Fix:** Gate `set_live_model_reference` and `compile_live_expert` behind `quality_gate_passed`.

### F2 — `test.py` quietly re-points live model
**File:** `tradebot/root_modules/test_cli/main.py`  
Running a backtest calls `activate_model(model_dir)` which rewrites `live/live.mq5` and clobbers `diagnostics/`. A read-only backtest command should not mutate live state.  
**Fix:** Separate "activate" from "test"; pass `model_dir` to the tester without touching live references.

---

## Correctness

### F3 — Two competing active-config resolution paths
**Files:** `tradebot/workspace_parts/shared.py`, `tradebot/workspace_parts/resolve_active_config_path.py`, `tradebot/root_modules/export_data/resolve_symbol_config.py`, `tradebot/root_modules/test_cli/main.py`  
`resolve_active_config_path()` reads `config/.active_config` then falls back to `config/active.mqh`.  
`resolve_symbol_config()` (used by `export_data`) reads the active symbol from `config/active.mqh` directly and then looks up `symbols/<sym>/config/config.mqh` — completely ignoring `.active_config`.  
Result: training and data-export can run against different configs without any warning.

### F4 — `lookback_requirement` raises `KeyError` for unknown feature names
**File:** `common/lookback_requirement.py` (pre-fix)  
Any feature name not in the hardcoded `requirements` dict raises `KeyError` with no useful message. Dynamic features (now `past_dir_*`) would silently break `max_feature_lookback` during warmup calculation.  
**Fix (applied):** Added a dynamic dispatch for `past_dir_*` and a clear `raise KeyError` fallback.

### F5 — `DEFAULT_FIXED_MOVE` used for both label SL and TP
**File:** `tradebot/pipeline/market_data_parts/get_triple_barrier_labels.py` (pre-fix)  
In fixed-risk mode, both the stop-loss and take-profit barriers were identical (`fixed_move_price`). There was no way to set asymmetric SL/TP without patching source code.  
**Fix (applied):** Added `fixed_sl_price` / `fixed_tp_price` optional args; configs can now define `DEFAULT_FIXED_SL` and `DEFAULT_FIXED_TP`.

### F6 — `build_live_model_reference_block` emits wrong relative paths after `live.mq5` move
**File:** `tradebot/workspace_parts/build_live_model_reference_block.py` (pre-fix)  
When `live.mq5` lived at the project root, `symbols/...` was a valid relative include path. After moving it to `live/live.mq5` the path needed a `../` prefix.  
**Fix (applied):** Block builder now prepends `../` to both the `#include` and `#resource` lines.

### F7 — `data.mq5` / `data_gold.mq5` included wrong config path
**Files:** `mt5/scripts/data.mq5`, `mt5/scripts/data_gold.mq5` (pre-fix)  
Both scripts did `#include "config.mqh"` which resolved relative to where MT5 deployed them (its Scripts folder), not to the project root. After moving the scripts to `mt5/scripts/` the path became `../../config/active.mqh`.  
**Fix (applied).**

### F8 — `move_ticks.py` had wrong `parents[2]` depth
**File:** `scripts/move_ticks.py` (pre-fix)  
`Path(__file__).resolve().parents[2]` resolved two levels above the old root location (grandparent of project root). After moving to `scripts/`, the correct depth is `parents[1]`.  
**Fix (applied).**

### F9 — `i.sh` heredoc expansion broke after quoting
**File:** `scripts/i.sh` (pre-fix)  
The quoted heredoc (`<<'PY'`) prevented `${days}` shell variable expansion inside the Python snippet. The days value was never passed to Python.  
**Fix (applied):** Replaced with `-c "..."` inline, interpolating `$days` safely.

---

## Reliability / Ops

### F10 — MT5 process killing is too broad
**File:** `tradebot/root_modules/mt5_runtime/stop_terminal_best_effort.py`  
Kills every `terminal64.exe` process on the machine. On a machine with multiple MT5 instances (e.g. live + paper trading), this terminates unrelated sessions silently.

### F11 — Compile fallback uses fragile GUI automation
**Files:** `tradebot/workspace_parts/_compile_via_metaeditor_ui_windows.py`, `_compile_via_metaeditor_ui_wine.py`  
Both use `SendKeys` / X11 `xdotool` to drive MetaEditor's UI as a last resort. This breaks in headless environments, remote desktops, and locked-down Windows sessions. There is no timeout on the Wine/X11 window-search loop.

### F12 — `keyboard.add_hotkey` registered at module import time
**File:** `tradebot/training/main.py`  
`import keyboard` and `keyboard.add_hotkey("ctrl+k", ...)` execute at import time, before `main()` is called. This fails silently or raises on headless Linux servers, CI runners, and Wine, where no keyboard device is present.

### F13 — Shared repo state mutated by transient commands
`export_data.py` copies a preset into `config/active.mqh`.  
`test.py` rewrites `live/live.mq5` and `diagnostics/`.  
Training overwrites `diagnostics/`.  
Interrupted or parallel runs leave the repo in an inconsistent state with no rollback mechanism.

### F14 — Backtest success detection is heuristic
**Files:** `tradebot/root_modules/test_cli/wait_for_tester_completion.py`, `parse_result.py`, `find_csv_file.py`  
Success/failure is determined by scanning log substrings and checking file size > 100 bytes. Stale logs or partial writes can produce false positives or false negatives.

### F15 — Deployment is not self-contained
**File:** `tradebot/workspace_parts/deploy_active_model.py` (pre-fix)  
Previously only copied `live.mq5`, `model.onnx`, and `config.mqh`. It did not sync `live/functions/*.mqh`, so any machine where the repo wasn't already present would fail to compile.  
**Fix (applied):** Now uses `sync_directory_contents` to mirror the entire `live/` source tree.

---

## ML / Data

### F16 — No dataset hash in model archive
**Files:** `tradebot/training/main.py`, `tradebot/pipeline/diagnostics_parts/write_diagnostics.py`  
The archived model bundle stores the config and diagnostics but not a hash or timestamp of the training CSV. If `data/<symbol>/ticks.csv` is replaced later, the archived model can no longer be reproducibly retrained.

### F17 — Training data is overwritten in place
**File:** `tradebot/root_modules/export_data/move_to_data_dir.py`  
Always writes to `data/<symbol>/ticks.csv` with no versioning or backup. A failed or partial export silently corrupts the training dataset.

### F18 — IQR normalisation computed on full train+val slice
**File:** `tradebot/training/main.py`  
`median` and `iqr` are computed over `x[:train_range[1]]` which includes the validation window. Strictly, normalisation statistics should be computed only on `x[:train_range[0]]` (the training portion) to prevent any lookahead.

### F19 — No embargo between train and val for normalisation
Related to F18: the embargo gap between train and val is enforced for window indices but the normalisation step uses a flat slice `x[:val_end]`, so a few bars of val data inform the scaler.

### F20 — `ALL_FEATURE_COLUMNS` is a static compile-time tuple
**File:** `common/feature_columns.py`  
`ALL_FEATURE_COLUMNS` does not include dynamic `past_dir_*` features. Any code that iterates `ALL_FEATURE_COLUMNS` to validate or enumerate features will miss them. The `lookback_requirement` dict lookup was the first casualty (F4).

### F21 — `FeatureEngineeringConfig` was missing `primary_bar_seconds`
**File:** `tradebot/pipeline/feature_builder_parts/FeatureEngineeringConfig.py` (pre-fix)  
Without `primary_bar_seconds`, the `_S` (seconds) variant of `past_dir_*` features could not convert wall-clock lookback to bar count at feature-computation time.  
**Fix (applied).**

---

## Config / Source-of-truth

### F22 — `bitcoin.config` references a flat `data/bitcoin.csv` path
**File:** `config/bitcoin.config`  
`#define DATA_FILE "data/bitcoin.csv"` — inconsistent with the canonical `data/<SYMBOL>/ticks.csv` convention used everywhere else. The data exporter writes `data/BTCUSD/ticks.csv` so training with this config will fail with a missing file error.

### F23 — `symbols/xauusd/config/` has no `config.mqh`
**Directory:** `symbols/xauusd/config/`  
Only `backtest_config.json` is present. `resolve_symbol_config` falls back to `config/active.mqh` instead of raising a clear error, silently using whatever the active config is rather than the XAUUSD preset.

### F24 — `README.md` was out of date
**File:** `README.md` (pre-fix)  
Described old root-level file layout and old script invocation paths.  
**Fix (applied):** Rewritten to reflect the new folder structure, `scripts/`, `config/`, `live/`, `mt5/`.

### F25 — `AGENTS.md` referenced wrong active-config filename and path
**File:** `AGENTS.md` (pre-fix)  
Said `.active-config` (hyphen) and `python nn.py` (root path).  
**Fix (applied):** Corrected to `config/.active_config` (underscore) and `python scripts/nn.py`.

---

## Deployment

### F26 — `DEFAULT_WINDOWS_INSTALL_DIR` is hardcoded
**File:** `tradebot/root_modules/mt5_runtime/shared.py`  
`C:\Program Files\MetaTrader 5` is the default on all platforms. There is no automatic discovery; Wine path variants must be set via CLI flags or env vars manually.

### F27 — No CI, no lockfile, loose `requirements.txt`
Most packages are pinned with `>=` rather than exact versions. Combined with no `pyproject.toml` or lockfile, two installs at different times can produce different environments and silently different results.

### F28 — `skills-lock.json` has no documented purpose
**File:** `meta/skills-lock.json`  
No code reads this file, no docs reference it. Either document it or remove it.

---

## Fixed in this session

| ID | Summary |
|----|---------|
| F4 | `lookback_requirement` KeyError on unknown features |
| F5 | Fixed-mode SL == TP forced; now independently configurable |
| F6 | Wrong `#include` paths in live model reference block |
| F7 | `data.mq5` included wrong config path |
| F8 | `move_ticks.py` wrong `parents` depth |
| F9 | `i.sh` heredoc variable expansion broken |
| F16 | Archived models now store `training_dataset.json` with CSV path, size, timestamp, and SHA-256 |
| F17 | Data export now writes immutable versioned snapshots before atomically refreshing `ticks.csv` |
| F18 | Robust scaler now fits on explicit `train_range[0]:train_range[1]` training rows |
| F19 | Same scaler fix preserves embargo safety if train start offset changes later |
| F20 | Added `resolve_all_feature_columns(values)` for correct dynamic `past_dir_*` enumeration |
| F15 | Deployment missed `live/functions/` source tree |
| F21 | `FeatureEngineeringConfig` missing `primary_bar_seconds` |
| F22 | `bitcoin.config` now points at canonical `data/BTCUSD/ticks.csv` |
| F23 | Added missing `symbols/xauusd/config/config.mqh` preset and loadable `gold.config` overlay |
| F24 | README out of date |
| F25 | AGENTS.md wrong active-config path/name |
