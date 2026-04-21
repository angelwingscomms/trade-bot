from __future__ import annotations

from .shared import *  # noqa: F401,F403

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
    train_end_idx_all = nn.build_segment_end_indices(valid_mask, *train_range, nn.SEQ_LEN, nn.LABEL_TIMEOUT_BARS)
    val_end_idx_all = nn.build_segment_end_indices(valid_mask, *val_range, nn.SEQ_LEN, nn.LABEL_TIMEOUT_BARS)
    test_end_idx_all = nn.build_segment_end_indices(valid_mask, *test_range, nn.SEQ_LEN, nn.LABEL_TIMEOUT_BARS)
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
