# GOLD pipeline

This folder now uses a reduced 31-feature layout:

- GOLD block: 15 features
  - `f0, f1, f2, f3, f4, f5, f6, f7, f8, f10, f11, f12, f13, f14, f15`
- USDX block: 8 features
  - `f0, f1, f5, f6, f7, f8, f10, f15`
- USDJPY block: 8 features
  - `f0, f1, f5, f6, f7, f8, f10, f15`

Removed features:

- `f9` for all symbols
  - MACD histogram is linearly redundant with `f7 - f8`
- `f11-f14` from USDX and USDJPY
  - time cyclicals are shared once through the GOLD block
- `f2` from USDX and USDJPY
  - bar duration is GOLD-aligned for those symbols
- `f3` and `f4` from USDX and USDJPY
  - upper/lower wicks are partially covered by range and close-in-range

Source of truth:

- trainer: [nn.py](/C:/Users/edhog/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Experts/nn/gold/nn.py)
- live EA: [live.mq5](/C:/Users/edhog/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Experts/nn/gold/live.mq5)
- tick exporter: [data.mq5](/C:/Users/edhog/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Experts/nn/gold/data.mq5)
- untouched Colab copy: [colab.py](/C:/Users/edhog/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Experts/nn/gold/colab.py)
  - left as-is per request and does not reflect the reduced 31-feature layout

Refresh flow:

1. Export fresh ticks with `data.mq5`.
2. Run `nn.py` to train a new `gold_mamba.onnx`.
3. Paste the emitted `medians[...]` and `iqrs[...]` arrays into `live.mq5`.
4. Compile `live.mq5` to regenerate `live.ex5`.

Important note:

- Existing generated artifacts such as `gold_mamba.onnx` and `live.ex5` may still reflect the old feature layout until they are rebuilt from the updated source files above.
