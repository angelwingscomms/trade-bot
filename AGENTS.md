be minimalist
always explain stuff to me like i'm 9 years old
code like an expert deep learning engineer with decades of experience and an uncompromisable standard of extreme perfection

# Agent Guide

This repo is a self-contained MT5 + Python pipeline.

## Architecture Snapshot

- Symbol presets live in `symbols/<symbol>/config/`.
- Archived models live in `symbols/<symbol>/models/<date>-<name>/`.
- Each archived model now stores a single combined `config.mqh` beside `model.onnx`.
- `live.mq5` includes the archived model folder directly.

## Gold Profiles

- `symbols/xauusd/config/gold.config` selects the legacy gold architecture.
- `symbols/xauusd/config/gold-new.config` selects the newer gold architecture.
- Those files are architecture-only presets. They must not override feature, bar, or target settings.
- Gold data export still uses `data_gold.mq5` so USDX/USDJPY ticks can be aligned by timestamp.

## Config Rules

- Prefer explicit booleans over magic zero values for enable/disable behavior.
- Full feature mode must always include the minimal feature set.
- Never add `.onnx` files to `.gitignore`.

## Maintenance Rules

- Always update `AGENTS.md` when repo changes make this guide stale.
- When changing repo structure, update every path/reference, not just the trainer.

be minimalist
