from __future__ import annotations

from .shared import *  # noqa: F401,F403

def main() -> None:
    args = parse_args()
    sl_values = parse_float_list(args.sl_values)
    tp_values = parse_float_list(args.tp_values)

    bars = nn.build_market_bars(
        Path("market_ticks.csv"),
        use_fixed_time_bars=bool(args.use_fixed_time_bars),
        use_fixed_tick_bars=False,
        tick_density=nn.DEFAULT_PRIMARY_TICK_DENSITY,
        max_bars=0,
    )
    x_all = nn.compute_features(bars)
    x = x_all[nn.WARMUP_BARS :]
    n_rows = len(x)
    embargo = max(nn.SEQ_LEN, nn.LABEL_TIMEOUT_BARS)

    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    train_range = (0, train_end)
    val_range = (train_end + embargo, val_end)
    test_range = (val_end + embargo, n_rows)

    median, iqr = fit_robust_scaler(x[train_range[0] : train_range[1]])
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
