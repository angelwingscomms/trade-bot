from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn as torch_nn

import nn
from minirocket_classifier import fit_minirocket, transform_sequences


def parse_float_list(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick MiniRocket sweep over ATR stop/target multipliers without touching archived/live model files."
    )
    parser.add_argument("--sl-values", type=str, default="0.54,0.72,1.0,1.5")
    parser.add_argument("--tp-values", type=str, default="0.12,0.18,0.27,0.36,0.54")
    parser.add_argument("--train-windows", type=int, default=12000)
    parser.add_argument("--eval-windows", type=int, default=2048)
    parser.add_argument("--minirocket-features", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-selected-trades", type=int, default=12)
    parser.add_argument("--use-fixed-time-bars", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path("diagnostics") / "minirocket_search_results.csv"),
    )
    return parser.parse_args()


def train_once(
    x_scaled: np.ndarray,
    y: np.ndarray,
    train_range: tuple[int, int],
    val_range: tuple[int, int],
    test_range: tuple[int, int],
    valid_mask: np.ndarray,
    train_windows: int,
    eval_windows: int,
    minirocket_features: int,
    epochs: int,
    batch_size: int,
    min_selected_trades: int,
) -> dict[str, float | int]:
    device = torch.device("cpu")
    train_end_idx_all = nn.build_segment_end_indices(valid_mask, *train_range, nn.SEQ_LEN, nn.TARGET_HORIZON)
    val_end_idx_all = nn.build_segment_end_indices(valid_mask, *val_range, nn.SEQ_LEN, nn.TARGET_HORIZON)
    test_end_idx_all = nn.build_segment_end_indices(valid_mask, *test_range, nn.SEQ_LEN, nn.TARGET_HORIZON)
    train_end_idx = nn.choose_evenly_spaced(train_end_idx_all, train_windows)
    val_end_idx = nn.choose_evenly_spaced(val_end_idx_all, eval_windows)
    test_end_idx = nn.choose_evenly_spaced(test_end_idx_all, eval_windows)

    x_train, y_train = nn.build_windows(x_scaled, y, train_end_idx, nn.SEQ_LEN)
    x_val, y_val = nn.build_windows(x_scaled, y, val_end_idx, nn.SEQ_LEN)
    x_test, y_test = nn.build_windows(x_scaled, y, test_end_idx, nn.SEQ_LEN)

    parameters = fit_minirocket(
        x_train.transpose(0, 2, 1),
        num_features=minirocket_features,
        seed=42,
    )
    train_features = transform_sequences(parameters, x_train, batch_size=batch_size, device=device)
    val_features = transform_sequences(parameters, x_val, batch_size=batch_size, device=device)
    test_features = transform_sequences(parameters, x_test, batch_size=batch_size, device=device)

    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = np.where(train_features.std(axis=0) < 1e-6, 1.0, train_features.std(axis=0)).astype(np.float32)
    train_features -= feature_mean
    train_features /= feature_std
    val_features -= feature_mean
    val_features /= feature_std
    test_features -= feature_mean
    test_features /= feature_std

    model = torch_nn.Linear(train_features.shape[1], len(nn.LABEL_NAMES)).to(device)
    torch_nn.init.zeros_(model.weight)
    torch_nn.init.zeros_(model.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-8,
        patience=5,
    )
    criterion = torch_nn.CrossEntropyLoss().to(device)
    train_loader = nn.make_loader(train_features, y_train, batch_size, shuffle=True)
    val_loader = nn.make_loader(val_features, y_val, batch_size, shuffle=False)
    test_loader = nn.make_loader(test_features, y_test, batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    wait = 0
    for _epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_logits, val_labels = nn.evaluate_model(model, val_loader, device)
        val_loss = float(
            criterion(
                torch.tensor(val_logits, dtype=torch.float32, device=device),
                torch.tensor(val_labels, dtype=torch.long, device=device),
            ).item()
        )
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 8:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")
    model.load_state_dict(best_state)

    val_logits, val_labels = nn.evaluate_model(model, val_loader, device)
    val_probs = nn.softmax(val_logits)
    threshold = nn.choose_confidence_threshold(
        val_probs,
        val_labels,
        min_selected=max(1, min_selected_trades),
        threshold_min=0.40,
        threshold_max=0.999,
        threshold_steps=120,
    )
    val_metrics = nn.gate_metrics(val_labels, val_probs, threshold)

    test_logits, test_labels = nn.evaluate_model(model, test_loader, device)
    test_probs = nn.softmax(test_logits)
    test_metrics = nn.gate_metrics(test_labels, test_probs, threshold)

    return {
        "threshold": float(threshold),
        "train_selected_rate": float((y_train > 0).mean()),
        "val_selected_trades": int(val_metrics["selected_trades"]),
        "val_precision": float(val_metrics["precision"]) if np.isfinite(val_metrics["precision"]) else float("nan"),
        "holdout_selected_trades": int(test_metrics["selected_trades"]),
        "holdout_precision": float(test_metrics["precision"]) if np.isfinite(test_metrics["precision"]) else float("nan"),
        "holdout_coverage": float(test_metrics["trade_coverage"]),
    }


def main() -> None:
    args = parse_args()
    sl_values = parse_float_list(args.sl_values)
    tp_values = parse_float_list(args.tp_values)

    bars = nn.build_market_bars(Path("market_ticks.csv"), use_fixed_time_bars=bool(args.use_fixed_time_bars))
    x_all = nn.compute_features(bars)
    x = x_all[nn.WARMUP_BARS :]
    n_rows = len(x)
    embargo = max(nn.SEQ_LEN, nn.TARGET_HORIZON)

    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    train_range = (0, train_end)
    val_range = (train_end + embargo, val_end)
    test_range = (val_end + embargo, n_rows)

    median = np.nanmedian(x[: train_range[1]], axis=0)
    median = np.nan_to_num(median, nan=0.0)
    iqr = np.nanpercentile(x[: train_range[1]], 75, axis=0) - np.nanpercentile(x[: train_range[1]], 25, axis=0)
    iqr = np.nan_to_num(iqr, nan=1.0)
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    x_scaled = np.clip((x - median) / iqr, -10.0, 10.0).astype(np.float32)
    valid_mask = ~np.isnan(x_scaled).any(axis=1)

    original_sl = nn.LABEL_SL_MULTIPLIER
    original_tp = nn.LABEL_TP_MULTIPLIER
    rows: list[dict[str, float | int | str]] = []
    try:
        for sl_mult in sl_values:
            for tp_mult in tp_values:
                nn.LABEL_SL_MULTIPLIER = sl_mult
                nn.LABEL_TP_MULTIPLIER = tp_mult
                y_all = nn.get_triple_barrier_labels(bars, use_atr_risk=True)
                y = y_all[nn.WARMUP_BARS :]
                metrics = train_once(
                    x_scaled=x_scaled,
                    y=y,
                    train_range=train_range,
                    val_range=val_range,
                    test_range=test_range,
                    valid_mask=valid_mask,
                    train_windows=args.train_windows,
                    eval_windows=args.eval_windows,
                    minirocket_features=args.minirocket_features,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    min_selected_trades=args.min_selected_trades,
                )
                row = {
                    "bar_mode": "FIXED_TIME" if args.use_fixed_time_bars else "IMBALANCE",
                    "sl_multiplier": sl_mult,
                    "tp_multiplier": tp_mult,
                    **metrics,
                }
                rows.append(row)
                print(
                    "RESULT"
                    f" bar_mode={row['bar_mode']}"
                    f" sl={sl_mult:.2f}"
                    f" tp={tp_mult:.2f}"
                    f" threshold={float(row['threshold']):.4f}"
                    f" val_precision={row['val_precision']}"
                    f" val_trades={row['val_selected_trades']}"
                    f" holdout_precision={row['holdout_precision']}"
                    f" holdout_trades={row['holdout_selected_trades']}"
                )
    finally:
        nn.LABEL_SL_MULTIPLIER = original_sl
        nn.LABEL_TP_MULTIPLIER = original_tp

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bar_mode",
                "sl_multiplier",
                "tp_multiplier",
                "threshold",
                "train_selected_rate",
                "val_selected_trades",
                "val_precision",
                "holdout_selected_trades",
                "holdout_precision",
                "holdout_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
