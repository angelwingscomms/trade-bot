data.mq5
```cpp
#property script_show_inputs

input int days_to_export = 60;
input string output_file = "gold_market_ticks.csv";
input string gold_symbol = "XAUUSD";

void OnStart() {
   int file_handle = FileOpen(output_file, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
   if(file_handle == INVALID_HANDLE) {
      PrintFormat("Cannot open %s. err=%d", output_file, GetLastError());
      return;
   }

   FileWrite(file_handle, "time_msc", "bid", "ask");

   long end_time = TimeCurrent() * 1000LL;
   long start_time = end_time - (long)days_to_export * 24LL * 3600LL * 1000LL;

   MqlTick ticks[];
   int copied = CopyTicksRange(gold_symbol, ticks, COPY_TICKS_ALL, start_time, end_time);
   if(copied <= 0) {
      PrintFormat("CopyTicksRange failed for %s. err=%d", gold_symbol, GetLastError());
      FileClose(file_handle);
      return;
   }

   for(int i = 0; i < copied; i++) {
      if(ticks[i].bid > 0.0) {
         FileWrite(file_handle, ticks[i].time_msc, ticks[i].bid, ticks[i].ask);
      }
   }

   FileClose(file_handle);
}
```

nn.py
```python
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gold_mamba_lite import GoldMambaLiteClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gold.nn")

EPS = 1e-10
SEQ_LEN = 120
TARGET_HORIZON = 30
DEFAULT_DATA_FILE = "gold_market_ticks.csv"
DEFAULT_OUTPUT_FILE = "gold_mamba.onnx"
FEATURE_COLUMNS = (
    "ret1",
    "high_rel_prev",
    "low_rel_prev",
    "spread_rel",
    "close_in_range",
    "atr14_rel",
    "rv4",
    "ret8",
    "tick_imbalance",
)
MODEL_FEATURE_COUNT = len(FEATURE_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stripped-down GOLD Mamba pipeline.")
    parser.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE, help="CSV with symbol,time_msc,bid,ask.")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="ONNX output file.")
    parser.add_argument("--epochs", type=int, default=16, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--max-train-windows", type=int, default=1536, help="Training window cap.")
    parser.add_argument("--max-eval-windows", type=int, default=384, help="Validation/test window cap.")
    parser.add_argument(
        "--imbalance-min-ticks",
        type=int,
        default=180,
        help="Minimum XAUUSD ticks before an imbalance bar may close.",
    )
    parser.add_argument(
        "--imbalance-ema-span",
        type=int,
        default=20,
        help="EWMA span for the imbalance threshold.",
    )
    parser.add_argument("--device", type=str, default="", help="Optional torch device override.")
    return parser.parse_args()


def resolve_local_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def compute_tick_signs(prices: np.ndarray) -> np.ndarray:
    signs = np.empty(len(prices), dtype=np.int8)
    last_sign = 1
    prev_price = float(prices[0]) if len(prices) else 0.0
    for i, price in enumerate(prices):
        if i > 0:
            diff = float(price) - prev_price
            if diff > 0.0:
                last_sign = 1
            elif diff < 0.0:
                last_sign = -1
        signs[i] = last_sign
        prev_price = float(price)
    return signs


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_values = high.to_numpy(dtype=np.float64, copy=False)
    low_values = low.to_numpy(dtype=np.float64, copy=False)
    close_values = close.to_numpy(dtype=np.float64, copy=False)

    tr = np.empty(len(close_values), dtype=np.float64)
    if len(tr) == 0:
        return pd.Series(dtype=np.float64, index=close.index)

    tr[0] = high_values[0] - low_values[0]
    for i in range(1, len(tr)):
        tr[i] = max(
            high_values[i] - low_values[i],
            abs(high_values[i] - close_values[i - 1]),
            abs(low_values[i] - close_values[i - 1]),
        )

    atr = np.full(len(tr), np.nan, dtype=np.float64)
    if len(tr) >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, len(tr)):
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / period

    return pd.Series(atr, index=close.index, dtype=np.float64)


def build_primary_bar_ids(
    df_ticks: pd.DataFrame,
    imbalance_min_ticks: int,
    imbalance_ema_span: int,
) -> np.ndarray:
    prices = df_ticks["bid"].to_numpy(dtype=np.float64, copy=False)
    tick_signs = compute_tick_signs(prices)
    alpha = 2.0 / (max(1, imbalance_ema_span) + 1.0)
    expected_abs_theta = max(2.0, float(max(2, imbalance_min_ticks // 3)))
    bar_ids = np.empty(len(prices), dtype=np.int64)
    current_bar = 0
    ticks_in_bar = 0
    theta = 0.0

    for i, sign in enumerate(tick_signs):
        bar_ids[i] = current_bar
        ticks_in_bar += 1
        theta += float(sign)
        if ticks_in_bar >= imbalance_min_ticks and abs(theta) >= expected_abs_theta:
            observed = max(2.0, abs(theta))
            expected_abs_theta = (1.0 - alpha) * expected_abs_theta + alpha * observed
            current_bar += 1
            ticks_in_bar = 0
            theta = 0.0

    return bar_ids


def build_gold_bars(csv_path: Path, imbalance_min_ticks: int, imbalance_ema_span: int) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(
        csv_path,
        usecols=["time_msc", "bid", "ask"],
        dtype={"time_msc": np.int64, "bid": np.float64, "ask": np.float64},
    ).sort_values("time_msc").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No ticks found in {csv_path}")

    df["bar_id"] = build_primary_bar_ids(df, imbalance_min_ticks, imbalance_ema_span)
    df["tick_sign"] = compute_tick_signs(df["bid"].to_numpy(dtype=np.float64, copy=False))
    df["spread"] = df["ask"] - df["bid"]

    grouped = df.groupby("bar_id").agg(
        {
            "bid": ["first", "max", "min", "last"],
            "time_msc": "first",
            "tick_sign": "mean",
            "ask": ["max", "min"],
            "spread": "last",
        }
    )
    grouped.columns = ["open", "high", "low", "close", "time_open", "tick_imbalance", "ask_high", "ask_low", "spread"]
    grouped = grouped.reset_index(drop=True)
    log.info("Built %d gold bars in %.2fs", len(grouped), time.time() - t0)
    return grouped


def compute_features(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    ret1 = np.log(close / (prev_close + EPS))
    atr14 = wilder_atr(df["high"], df["low"], close, period=14)

    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = ret1
    feat["high_rel_prev"] = np.log(df["high"] / (prev_close + EPS))
    feat["low_rel_prev"] = np.log(df["low"] / (prev_close + EPS))
    feat["spread_rel"] = df["spread"] / (close + EPS)
    feat["close_in_range"] = (close - df["low"]) / (df["high"] - df["low"] + 1e-8)
    feat["atr14_rel"] = atr14 / (close + EPS)
    feat["rv4"] = ret1.rolling(4, min_periods=4).std(ddof=0)
    feat["ret8"] = np.log(close / (close.shift(8) + EPS))
    feat["tick_imbalance"] = df["tick_imbalance"].astype(float)
    return feat.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)


def get_triple_barrier_labels(
    df_gold: pd.DataFrame,
    tp_mult: float = 9.0,
    sl_mult: float = 5.4,
    horizon: int = TARGET_HORIZON,
) -> np.ndarray:
    close = df_gold["close"].to_numpy(dtype=np.float64, copy=False)
    high = df_gold["high"].to_numpy(dtype=np.float64, copy=False)
    low = df_gold["low"].to_numpy(dtype=np.float64, copy=False)
    spread = df_gold["spread"].to_numpy(dtype=np.float64, copy=False)
    ask_high = df_gold["ask_high"].to_numpy(dtype=np.float64, copy=False)
    ask_low = df_gold["ask_low"].to_numpy(dtype=np.float64, copy=False)
    atr = wilder_atr(df_gold["high"], df_gold["low"], df_gold["close"], period=14).to_numpy(dtype=np.float64, copy=False)

    labels = np.zeros(len(df_gold), dtype=np.int64)
    for i in range(len(df_gold) - horizon):
        vol = atr[i]
        if not np.isfinite(vol) or vol <= 0.0:
            continue

        long_entry = close[i] + spread[i]
        short_entry = close[i]
        long_tp = long_entry + tp_mult * vol
        long_sl = long_entry - sl_mult * vol
        short_tp = short_entry - tp_mult * vol
        short_sl = short_entry + sl_mult * vol

        long_result = 0
        short_result = 0
        for j in range(i + 1, i + horizon + 1):
            if long_result == 0:
                hit_tp = high[j] >= long_tp
                hit_sl = low[j] <= long_sl
                if hit_tp and not hit_sl:
                    long_result = 1
                elif hit_sl:
                    long_result = -1

            if short_result == 0:
                hit_tp = ask_low[j] <= short_tp
                hit_sl = ask_high[j] >= short_sl
                if hit_tp and not hit_sl:
                    short_result = 1
                elif hit_sl:
                    short_result = -1

            if long_result != 0 and short_result != 0:
                break

        if long_result == 1 and short_result != 1:
            labels[i] = 1
        elif short_result == 1 and long_result != 1:
            labels[i] = 2

    return labels


def choose_evenly_spaced(indices: np.ndarray, max_count: int) -> np.ndarray:
    if len(indices) <= max_count:
        return indices.astype(np.int64, copy=False)
    positions = np.linspace(0, len(indices) - 1, max_count)
    return indices[np.unique(np.round(positions).astype(np.int64))]


def build_segment_end_indices(
    valid_mask: np.ndarray,
    start_bar: int,
    end_bar: int,
    seq_len: int,
    horizon: int,
) -> np.ndarray:
    first_end = start_bar + seq_len - 1
    last_end = end_bar - horizon - 1
    if last_end < first_end:
        return np.empty(0, dtype=np.int64)

    ends = []
    for end_idx in range(first_end, last_end + 1):
        start_idx = end_idx - seq_len + 1
        if valid_mask[start_idx : end_idx + 1].all():
            ends.append(end_idx)
    return np.asarray(ends, dtype=np.int64)


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    end_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.empty((len(end_indices), seq_len, features.shape[1]), dtype=np.float32)
    ys = np.empty(len(end_indices), dtype=np.int64)
    for i, end_idx in enumerate(end_indices):
        start_idx = end_idx - seq_len + 1
        xs[i] = features[start_idx : end_idx + 1]
        ys[i] = labels[end_idx]
    return xs, ys


def make_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = np.ones(3, dtype=np.float32)
    total = counts.sum()
    for cls in range(3):
        if counts[cls] > 0:
            weights[cls] = total / (3.0 * counts[cls])
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits_list.append(model(xb.to(device)).cpu().numpy())
            labels_list.append(yb.numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def choose_confidence_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    candidate_mask = preds > 0
    if not candidate_mask.any():
        return 0.60

    min_selected = max(12, int(0.03 * len(labels)))
    best_threshold = 0.60
    best_precision = -1.0
    best_coverage = -1.0
    confidences = probs.max(axis=1)

    for threshold in np.linspace(0.40, 0.85, 19):
        selected = candidate_mask & (confidences >= threshold)
        if selected.sum() < min_selected:
            continue

        precision = float((preds[selected] == labels[selected]).mean())
        coverage = float(selected.mean())
        if precision > best_precision + 1e-12 or (
            abs(precision - best_precision) <= 1e-12 and coverage > best_coverage
        ):
            best_threshold = float(threshold)
            best_precision = precision
            best_coverage = coverage

    return best_threshold


def summarize_gate(name: str, probs: np.ndarray, labels: np.ndarray, threshold: float) -> None:
    preds = probs.argmax(axis=1)
    selected = (preds > 0) & (probs.max(axis=1) >= threshold)
    coverage = float(selected.mean())
    if selected.any():
        precision = float((preds[selected] == labels[selected]).mean())
        log.info("%s: threshold=%.2f precision=%.4f coverage=%.4f trades=%d", name, threshold, precision, coverage, int(selected.sum()))
    else:
        log.warning("%s: threshold=%.2f produced no trades.", name, threshold)


def format_float_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(v):.8f}f" for v in values)


def build_mql_config(
    median: np.ndarray,
    iqr: np.ndarray,
    imbalance_min_ticks: int,
    imbalance_ema_span: int,
    primary_confidence: float,
) -> str:
    return "\n".join(
        [
            "// Auto-generated by gold/nn.py. Re-run training to refresh these values.",
            f"#define MODEL_FEATURE_COUNT {MODEL_FEATURE_COUNT}",
            f"#define IMBALANCE_MIN_TICKS {imbalance_min_ticks}",
            f"#define IMBALANCE_EMA_SPAN {imbalance_ema_span}",
            f"#define PRIMARY_CONFIDENCE {primary_confidence:.8f}",
            f"float medians[MODEL_FEATURE_COUNT] = {{{format_float_array(median)}}};",
            f"float iqrs[MODEL_FEATURE_COUNT] = {{{format_float_array(iqr)}}};",
        ]
    )


def main() -> None:
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    data_path = resolve_local_path(args.data_file)
    output_path = resolve_local_path(args.output_file)
    config_path = Path(__file__).resolve().with_name("gold_model_config.mqh")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using device: %s", device)

    df_gold = build_gold_bars(data_path, args.imbalance_min_ticks, args.imbalance_ema_span)
    x = compute_features(df_gold)
    y = get_triple_barrier_labels(df_gold)

    warmup = 16
    x = x[warmup:]
    y = y[warmup:]
    n_rows = len(x)
    embargo = max(SEQ_LEN, TARGET_HORIZON)

    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    train_range = (0, train_end)
    val_range = (train_end + embargo, val_end)
    test_range = (val_end + embargo, n_rows)
    if test_range[0] >= test_range[1]:
        raise ValueError("Dataset is too small for leakage-safe train/val/test splits.")

    median = np.nanmedian(x[: train_range[1]], axis=0)
    median = np.nan_to_num(median, nan=0.0)
    iqr = np.nanpercentile(x[: train_range[1]], 75, axis=0) - np.nanpercentile(x[: train_range[1]], 25, axis=0)
    iqr = np.nan_to_num(iqr, nan=1.0)
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    x_scaled = np.clip((x - median) / iqr, -10.0, 10.0).astype(np.float32)
    valid_mask = ~np.isnan(x_scaled).any(axis=1)

    train_end_idx = choose_evenly_spaced(
        build_segment_end_indices(valid_mask, *train_range, SEQ_LEN, TARGET_HORIZON),
        args.max_train_windows,
    )
    val_end_idx = choose_evenly_spaced(
        build_segment_end_indices(valid_mask, *val_range, SEQ_LEN, TARGET_HORIZON),
        args.max_eval_windows,
    )
    test_end_idx = choose_evenly_spaced(
        build_segment_end_indices(valid_mask, *test_range, SEQ_LEN, TARGET_HORIZON),
        args.max_eval_windows,
    )
    if min(len(train_end_idx), len(val_end_idx), len(test_end_idx)) == 0:
        raise ValueError("One or more leakage-safe splits ended up empty.")

    x_train, y_train = build_windows(x_scaled, y, train_end_idx, SEQ_LEN)
    x_val, y_val = build_windows(x_scaled, y, val_end_idx, SEQ_LEN)
    x_test, y_test = build_windows(x_scaled, y, test_end_idx, SEQ_LEN)
    log.info("Window counts | train=%d val=%d test=%d", len(x_train), len(x_val), len(x_test))

    class_weights = make_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = GoldMambaLiteClassifier(n_features=MODEL_FEATURE_COUNT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, max(args.batch_size, 64), shuffle=False)
    test_loader = make_loader(x_test, y_test, max(args.batch_size, 64), shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    patience = 4
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_logits, val_labels = evaluate_model(model, val_loader, device)
        val_loss = float(
            criterion(
                torch.tensor(val_logits, dtype=torch.float32, device=device),
                torch.tensor(val_labels, dtype=torch.long, device=device),
            ).item()
        )
        log.info(
            "Epoch %02d | train_loss=%.4f val_loss=%.4f wait=%d/%d",
            epoch,
            float(np.mean(train_losses)),
            val_loss,
            wait,
            patience,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    val_logits, val_labels = evaluate_model(model, val_loader, device)
    val_probs = softmax(val_logits)
    primary_confidence = choose_confidence_threshold(val_probs, val_labels)
    summarize_gate("validation", val_probs, val_labels, primary_confidence)

    test_logits, test_labels = evaluate_model(model, test_loader, device)
    test_probs = softmax(test_logits)
    summarize_gate("holdout", test_probs, test_labels, primary_confidence)

    model.eval()
    model.to("cpu")
    dummy = torch.randn(1, SEQ_LEN, MODEL_FEATURE_COUNT)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )

    config_path.write_text(
        build_mql_config(
            median=median,
            iqr=iqr,
            imbalance_min_ticks=args.imbalance_min_ticks,
            imbalance_ema_span=args.imbalance_ema_span,
            primary_confidence=primary_confidence,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Saved ONNX to %s", output_path)
    log.info("Saved config to %s", config_path)
    log.info("Total runtime: %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
```

live.mq5
```cpp
#include <Trade\Trade.mqh>
#include "gold_model_config.mqh"

#resource "\\Experts\\nn\\gold\\gold_mamba.onnx" as uchar model_buffer[]

#define SEQ_LEN 120
#define FEATURE_COUNT 9
#define REQUIRED_HISTORY_INDEX 127
#define HISTORY_SIZE (REQUIRED_HISTORY_INDEX + 1)
#define INPUT_BUFFER_SIZE (SEQ_LEN * MODEL_FEATURE_COUNT)

input double SL_MULTIPLIER = 5.4;
input double TP_MULTIPLIER = 9.0;
input double LOT_SIZE = 0.01;
input int MAGIC_NUMBER = 777777;

long onnx_handle = INVALID_HANDLE;
CTrade trade;

struct Bar {
   double o;
   double h;
   double l;
   double c;
   double spread;
   double tick_imbalance;
   double atr14;
   ulong time_msc;
   bool valid;
};

Bar history[HISTORY_SIZE];
Bar current_bar;
int ticks_in_bar = 0;
bool bar_started = false;
ulong last_tick_time = 0;
double tick_imbalance_sum = 0.0;
double last_bid = 0.0;
int last_sign = 1;
double primary_expected_abs_theta = 60.0;
int warmup_count = 0;
double warmup_sum = 0.0;
float input_data[INPUT_BUFFER_SIZE];
float output_data[3];

int UpdateTickSign(double bid);
void ProcessTick(MqlTick &tick);
void UpdateIndicators(Bar &bar);
bool ShouldClosePrimaryBar(double &observed_abs_theta);
void UpdatePrimaryImbalanceThreshold(double observed_abs_theta);
void CloseBar();
void LoadHistory();
float ScaleAndClip(float value, int feature_index);
double SafeLogRatio(double num, double den);
double LogReturnAt(int h);
double ReturnOverBars(int h, int bars);
double RollingStdReturn(int h, int window);
void ExtractFeatures(int h, float &features[]);
void Softmax(const float &logits[], float &probs[]);
void Predict();
void Execute(int signal);

int OnInit() {
   onnx_handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) {
      Print("[FATAL] OnnxCreateFromBuffer failed: ", GetLastError());
      return INIT_FAILED;
   }

   long input_shape[3];
   long output_shape[2];
   input_shape[0] = 1;
   input_shape[1] = SEQ_LEN;
   input_shape[2] = MODEL_FEATURE_COUNT;
   output_shape[0] = 1;
   output_shape[1] = 3;
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape) || !OnnxSetOutputShape(onnx_handle, 0, output_shape)) {
      Print("[FATAL] OnnxSetShape failed: ", GetLastError());
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return INIT_FAILED;
   }

   for(int i = 0; i < HISTORY_SIZE; i++) {
      history[i].valid = false;
   }

   ArrayInitialize(input_data, 0.0f);
   trade.SetExpertMagicNumber(MAGIC_NUMBER);
   primary_expected_abs_theta = MathMax(2.0, (double)MathMax(2, IMBALANCE_MIN_TICKS / 3));

   MqlTick tick;
   if(SymbolInfoTick(_Symbol, tick)) {
      last_tick_time = tick.time_msc;
   } else {
      last_tick_time = TimeCurrent() * 1000ULL;
   }

   LoadHistory();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if(onnx_handle != INVALID_HANDLE) {
      OnnxRelease(onnx_handle);
   }
}

int UpdateTickSign(double bid) {
   int sign = last_sign;
   if(last_bid <= 0.0) {
      sign = 1;
   } else {
      double diff = bid - last_bid;
      if(diff > 0.0) {
         sign = 1;
      } else if(diff < 0.0) {
         sign = -1;
      }
   }

   last_bid = bid;
   last_sign = sign;
   return sign;
}

void ProcessTick(MqlTick &tick) {
   if(tick.bid <= 0.0) {
      return;
   }

   if(!bar_started) {
      current_bar.o = tick.bid;
      current_bar.h = tick.bid;
      current_bar.l = tick.bid;
      current_bar.c = tick.bid;
      current_bar.spread = tick.ask - tick.bid;
      current_bar.tick_imbalance = 0.0;
      current_bar.time_msc = tick.time_msc;
      ticks_in_bar = 0;
      tick_imbalance_sum = 0.0;
      bar_started = true;
   }

   int tick_sign = UpdateTickSign(tick.bid);
   current_bar.h = MathMax(current_bar.h, tick.bid);
   current_bar.l = MathMin(current_bar.l, tick.bid);
   current_bar.c = tick.bid;
   current_bar.spread = tick.ask - tick.bid;
   ticks_in_bar++;
   tick_imbalance_sum += tick_sign;
}

void UpdateIndicators(Bar &bar) {
   Bar prev = history[0];
   double tr = (warmup_count == 0)
      ? (bar.h - bar.l)
      : MathMax(bar.h - bar.l, MathMax(MathAbs(bar.h - prev.c), MathAbs(bar.l - prev.c)));

   if(warmup_count < 14) {
      warmup_sum += tr;
      warmup_count++;
      bar.atr14 = warmup_sum / warmup_count;
   } else {
      double prev_atr = (prev.atr14 > 0.0 ? prev.atr14 : tr);
      bar.atr14 = prev_atr + (tr - prev_atr) / 14.0;
      warmup_count++;
   }

   bar.valid = (warmup_count >= 16);
}

bool ShouldClosePrimaryBar(double &observed_abs_theta) {
   if(ticks_in_bar < IMBALANCE_MIN_TICKS) {
      observed_abs_theta = 0.0;
      return false;
   }

   observed_abs_theta = MathAbs(tick_imbalance_sum);
   return (observed_abs_theta >= primary_expected_abs_theta);
}

void UpdatePrimaryImbalanceThreshold(double observed_abs_theta) {
   if(observed_abs_theta <= 0.0) {
      return;
   }

   double alpha = 2.0 / (MathMax(1, IMBALANCE_EMA_SPAN) + 1.0);
   double observed = MathMax(2.0, observed_abs_theta);
   primary_expected_abs_theta = (1.0 - alpha) * primary_expected_abs_theta + alpha * observed;
}

void CloseBar() {
   current_bar.tick_imbalance = tick_imbalance_sum / MathMax(1, ticks_in_bar);
   UpdateIndicators(current_bar);

   for(int i = HISTORY_SIZE - 1; i > 0; i--) {
      history[i] = history[i - 1];
   }
   history[0] = current_bar;

   ticks_in_bar = 0;
   tick_imbalance_sum = 0.0;
   bar_started = false;
}

void OnTick() {
   MqlTick ticks[];
   int count = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, last_tick_time + 1, 100000);
   if(count <= 0) {
      return;
   }

   for(int i = 0; i < count; i++) {
      if(ticks[i].bid <= 0.0) {
         continue;
      }

      ProcessTick(ticks[i]);
      last_tick_time = ticks[i].time_msc;

      double observed_abs_theta = 0.0;
      if(ShouldClosePrimaryBar(observed_abs_theta)) {
         CloseBar();
         UpdatePrimaryImbalanceThreshold(observed_abs_theta);
         if(history[REQUIRED_HISTORY_INDEX].valid) {
            Predict();
         }
      }
   }
}

void LoadHistory() {
   ulong start_time_msc = (TimeCurrent() - 86400 * 3) * 1000ULL;
   MqlTick ticks[];
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, start_time_msc, 250000);
   if(copied <= 0) {
      start_time_msc = (TimeCurrent() - 86400) * 1000ULL;
      copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, start_time_msc, 250000);
   }
   if(copied <= 0) {
      return;
   }

   last_tick_time = ticks[0].time_msc - 1;
   for(int i = 0; i < copied; i++) {
      if(ticks[i].bid <= 0.0) {
         continue;
      }

      ProcessTick(ticks[i]);
      last_tick_time = ticks[i].time_msc;

      double observed_abs_theta = 0.0;
      if(ShouldClosePrimaryBar(observed_abs_theta)) {
         CloseBar();
         UpdatePrimaryImbalanceThreshold(observed_abs_theta);
      }
   }
}

float ScaleAndClip(float value, int feature_index) {
   float iqr = (iqrs[feature_index] > 1e-6f ? iqrs[feature_index] : 1.0f);
   float scaled = (value - medians[feature_index]) / iqr;
   return MathMax(-10.0f, MathMin(10.0f, scaled));
}

double SafeLogRatio(double num, double den) {
   return MathLog((num + 1e-10) / (den + 1e-10));
}

double LogReturnAt(int h) {
   return SafeLogRatio(history[h].c, history[h + 1].c);
}

double ReturnOverBars(int h, int bars) {
   return SafeLogRatio(history[h].c, history[h + bars].c);
}

double RollingStdReturn(int h, int window) {
   double values[8];
   double mean = 0.0;
   for(int i = 0; i < window; i++) {
      values[i] = LogReturnAt(h + i);
      mean += values[i];
   }
   mean /= window;

   double var = 0.0;
   for(int i = 0; i < window; i++) {
      double diff = values[i] - mean;
      var += diff * diff;
   }
   return MathSqrt(var / window);
}

void ExtractFeatures(int h, float &features[]) {
   Bar bar = history[h];
   Bar prev = history[h + 1];
   double close = bar.c;

   features[0] = ScaleAndClip((float)LogReturnAt(h), 0);
   features[1] = ScaleAndClip((float)SafeLogRatio(bar.h, prev.c), 1);
   features[2] = ScaleAndClip((float)SafeLogRatio(bar.l, prev.c), 2);
   features[3] = ScaleAndClip((float)(bar.spread / (close + 1e-10)), 3);
   features[4] = ScaleAndClip((float)((close - bar.l) / (bar.h - bar.l + 1e-8)), 4);
   features[5] = ScaleAndClip((float)(bar.atr14 / (close + 1e-10)), 5);
   features[6] = ScaleAndClip((float)RollingStdReturn(h, 4), 6);
   features[7] = ScaleAndClip((float)ReturnOverBars(h, 8), 7);
   features[8] = ScaleAndClip((float)bar.tick_imbalance, 8);
}

void Softmax(const float &logits[], float &probs[]) {
   double max_logit = MathMax(logits[0], MathMax(logits[1], logits[2]));
   double e0 = MathExp(logits[0] - max_logit);
   double e1 = MathExp(logits[1] - max_logit);
   double e2 = MathExp(logits[2] - max_logit);
   double sum = e0 + e1 + e2;
   probs[0] = (float)(e0 / sum);
   probs[1] = (float)(e1 / sum);
   probs[2] = (float)(e2 / sum);
}

void Predict() {
   for(int i = 0; i < SEQ_LEN; i++) {
      int h = SEQ_LEN - 1 - i;
      int offset = i * MODEL_FEATURE_COUNT;
      float features[FEATURE_COUNT];
      ExtractFeatures(h, features);
      for(int k = 0; k < FEATURE_COUNT; k++) {
         input_data[offset + k] = features[k];
      }
   }

   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
      return;
   }

   float probs[3];
   Softmax(output_data, probs);
   int signal = ArrayMaximum(probs);
   if(signal <= 0) {
      return;
   }
   if(probs[signal] < PRIMARY_CONFIDENCE) {
      return;
   }

   Execute(signal);
}

void Execute(int signal) {
   if(PositionSelect(_Symbol)) {
      return;
   }

   double price = (signal == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = (signal == 1) ? (price - history[0].atr14 * SL_MULTIPLIER) : (price + history[0].atr14 * SL_MULTIPLIER);
   double tp = (signal == 1) ? (price + history[0].atr14 * TP_MULTIPLIER) : (price - history[0].atr14 * TP_MULTIPLIER);

   double min_dist = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(MathAbs(price - sl) < min_dist || MathAbs(tp - price) < min_dist) {
      return;
   }

   trade.PositionOpen(_Symbol, (signal == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL), LOT_SIZE, price, sl, tp);
}
```
