be minimalist
always speak in an extremely concise manner to the user and always explain stuff to the user like i'm 9 years old
code like an expert deep learning engineer with decades of experience and an uncompromisable standard of extreme perfection

before ever doing anything or attending to any request or user message, run this:
`git add .; git commit -m"before codex"; git push"
don't worry if push fails, it's the commit that's important

# Agent Guide

This repo is a self-contained MT5 + Python pipeline.

## Architecture Snapshot

- Symbol presets live in `symbols/<symbol>/config/`.
- Archived models live in `symbols/<symbol>/models/<date>-<name>/`.
- Each archived model now stores a single combined `config.mqh` beside `model.onnx`.
- `live.mq5` includes the archived model folder directly.
- Pipeline helpers live in `tradebot/pipeline/`.
- Sequence architectures live in `tradebot/models/sequence/`.
- `sequence_models.py` is now just a small compatibility wrapper.

## Gold Profiles

- `symbols/xauusd/config/gold.config` selects the legacy gold architecture.
- `symbols/xauusd/config/gold-new.config` selects the newer gold architecture.
- Those files are architecture-only presets. They must not override feature, bar, or target settings.
- Gold data export still uses `data_gold.mq5` so USDX/USDJPY ticks can be aligned by timestamp.

## AU Profile

- `symbols/xauusd/config/au.config` selects `MODEL_ARCHITECTURE "au"`.
- The `au` architecture mirrors `LSTM(64) -> MultiHeadAttention(4, 64) -> GlobalAveragePooling -> Dense`.
- `au.config` also enables `USE_MAIN_FEATURE_SET true`.

## Config Rules

- Prefer explicit booleans over magic zero values for enable/disable behavior.
- Full feature mode must always include the minimal feature set.
- Never add `.onnx` files to `.gitignore`.
- `USE_MAIN_FEATURE_SET` enables the 40-feature notebook-style main feature pack.
- Imbalance threshold priority is: EMA threshold first, then optional `IMBALANCE_MIN_TICKS / 3`, then raw `IMBALANCE_MIN_TICKS`.
- Config selection: edit `.active_config` (single line: path to config file, absolute or relative to repo root), then run `python nn.py`. Example: `symbols/xauusd/config/au.config`, `symbols/xauusd/config/gold.config`, or `symbols/xauusd/config/gold-new.config`.

## Maintenance Rules

- Always update `AGENTS.md` when repo changes make this guide stale.
- When changing repo structure, update every path/reference, not just the trainer.

be minimalist
