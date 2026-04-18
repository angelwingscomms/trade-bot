# Trade Bot Pipeline Notes

This repo trains MT5-compatible ONNX models from `config.mqh` and archives them under `symbols/<symbol>/models/`.

## Layout

- `config.mqh` is the active config used by `nn.py`, `data.mq5`, `data_gold.mq5`, and `inspect_bars.py`.
- `symbols/<symbol>/config/` stores reusable presets.
- `symbols/<symbol>/models/<date>-<name>/` stores `model.onnx`, the combined `config.mqh`, diagnostics, and tests.
- `live.mq5` points directly at one archived model folder.
- Only CLI helper scripts remain in the repo root. Importable Python modules now live under `tradebot/`, with CLI implementations under `tradebot/root_modules/`.

## Scripts

- `python nn.py` trains, exports ONNX, updates `live.mq5`, compiles the EA, and prints the new model folder name.
- `python inspect_bars.py` shows how many bars the active config would build without training.
- `python test.py` runs archived-model backtests and saves results under the model's `tests/` folder.
- `python export_data.py --profile gold` exports XAUUSD ticks plus USDX/USDJPY context.
