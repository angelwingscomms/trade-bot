from __future__ import annotations

from .shared import *  # noqa: F401,F403
import tradebot.training.shared as _shared

def main() -> None:
    t0 = time.time()
    args = parse_args()
    project = args.config_project
    apply_shared_settings(project.values, project=project, shared_config_path=project.config_path)
    # Update module globals from shared module after apply_shared_settings ran
    _keys = [
        "SHARED", "CURRENT_CONFIG_PATH", "SYMBOL", "SEQ_LEN", "TARGET_HORIZON", "FEATURE_ATR_PERIOD",
        "FEATURE_ATR_RATIO_PERIOD", "FEATURE_BOLLINGER_PERIOD", "FEATURE_DONCHIAN_FAST_PERIOD",
        "FEATURE_DONCHIAN_SLOW_PERIOD", "FEATURE_RET_2_PERIOD", "FEATURE_RET_3_PERIOD",
        "FEATURE_RET_6_PERIOD", "FEATURE_RET_12_PERIOD", "FEATURE_RET_20_PERIOD",
        "FEATURE_RSI_FAST_PERIOD", "FEATURE_RSI_SLOW_PERIOD", "FEATURE_RV_LONG_PERIOD",
        "FEATURE_SMA_FAST_PERIOD", "FEATURE_SMA_MID_PERIOD", "FEATURE_SMA_SLOW_PERIOD",
        "FEATURE_SMA_SLOPE_SHIFT", "FEATURE_SMA_TREND_FAST_PERIOD",
        "FEATURE_SPREAD_Z_PERIOD", "FEATURE_STOCH_PERIOD", "FEATURE_STOCH_SMOOTH_PERIOD",
        "FEATURE_TICK_COUNT_PERIOD", "FEATURE_TICK_IMBALANCE_FAST_PERIOD",
        "FEATURE_TICK_IMBALANCE_SLOW_PERIOD", "TARGET_ATR_PERIOD", "RV_PERIOD",
        "RETURN_PERIOD", "MAX_FEATURE_LOOKBACK", "WARMUP_BARS",
        "IMBALANCE_MIN_TICKS", "IMBALANCE_EMA_SPAN", "USE_IMBALANCE_EMA_THRESHOLD",
        "USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD", "PRIMARY_BAR_SECONDS",
        "PRIMARY_TICK_DENSITY", "BAR_DURATION_MS", "DEFAULT_FIXED_MOVE",
        "LABEL_SL_MULTIPLIER", "LABEL_TP_MULTIPLIER", "EXECUTION_SL_MULTIPLIER",
        "EXECUTION_TP_MULTIPLIER", "USE_ALL_WINDOWS", "DEFAULT_EPOCHS",
        "DEFAULT_BATCH_SIZE", "DEFAULT_MAX_TRAIN_WINDOWS", "DEFAULT_MAX_EVAL_WINDOWS",
        "DEFAULT_PATIENCE", "DEFAULT_LOSS_MODE", "ACTIVE_PROJECT",
    ]
    for _k in _keys:
        globals()[_k] = getattr(_shared, _k)
    torch.manual_seed(42)
    np.random.seed(42)
    architecture = resolve_architecture(args)
    requested_model_name = args.name.strip()
    model_name = sanitize_model_name(requested_model_name)
    if requested_model_name and not model_name:
        log.warning("Model name %r sanitized to empty; using a timestamp-only archive folder.", requested_model_name)
    elif requested_model_name and model_name != requested_model_name:
        log.info("Model folder prefix sanitized from %r to %r.", requested_model_name, model_name)
    use_extended_features = bool(args.use_extended_features)
    use_atr_risk = not bool(args.use_fixed_risk)
    use_fixed_time_bars = bool(args.use_fixed_time_bars)
    use_fixed_tick_bars = bool(args.use_fixed_tick_bars)
    use_no_hold = bool(args.no_hold)
    active_label_names = LABEL_NAMES_BINARY if use_no_hold else LABEL_NAMES
    if architecture == "chronos_bolt" and use_extended_features:
        log.warning("Chronos-Bolt backend uses the minimal live feature pack; ignoring extra feature switches.")
        use_extended_features = False
    feature_columns = project.feature_columns
    feature_profile = project.feature_profile
    feature_count = len(feature_columns)
    use_multihead_attention = bool(args.use_multihead_attention)
    if use_fixed_tick_bars and use_fixed_time_bars:
        raise ValueError("Choose only one bar mode: fixed-time or fixed-tick.")
    if architecture == "legacy_lstm_attention":
        if use_multihead_attention:
            log.info("Legacy LSTM attention architecture has built-in self-attention; the -a flag is redundant.")
        else:
            log.info("Legacy LSTM attention architecture includes self-attention by design.")
        use_multihead_attention = True
        if args.sequence_hidden_size != DEFAULT_SEQUENCE_HIDDEN_SIZE:
            log.warning(
                "Legacy LSTM attention fixes its recurrent width to the active feature count; ignoring --sequence-hidden-size=%d.",
                args.sequence_hidden_size,
            )
        if args.sequence_layers != DEFAULT_SEQUENCE_LAYERS:
            log.warning(
                "Legacy LSTM attention uses a single recurrent layer; ignoring --sequence-layers=%d.",
                args.sequence_layers,
            )
        if abs(args.sequence_dropout - DEFAULT_SEQUENCE_DROPOUT) > 1e-12:
            log.warning(
                "Legacy LSTM attention does not use the recurrent dropout path; ignoring --sequence-dropout=%.3f.",
                args.sequence_dropout,
            )
        if args.attention_layers != DEFAULT_ATTENTION_LAYERS:
            log.warning(
                "Legacy LSTM attention uses a single self-attention block; ignoring --attention-layers=%d.",
                args.attention_layers,
            )
    if architecture == "ela":
        if not use_multihead_attention:
            log.info("ELA uses the multihead attention head by design; enabling attention automatically.")
        use_multihead_attention = True
    if architecture == "fusion_lstm":
        if not use_multihead_attention:
            log.info("Fusion-LSTM includes its residual self-attention block by design; enabling attention automatically.")
        use_multihead_attention = True
        if args.sequence_hidden_size != DEFAULT_SEQUENCE_HIDDEN_SIZE:
            log.warning(
                "Fusion-LSTM fixes its recurrent width to the active feature count; ignoring --sequence-hidden-size=%d.",
                args.sequence_hidden_size,
            )
        if args.sequence_layers != DEFAULT_SEQUENCE_LAYERS:
            log.warning("Fusion-LSTM uses a single Mish-LSTM layer; ignoring --sequence-layers=%d.", args.sequence_layers)
        if abs(args.sequence_dropout - DEFAULT_SEQUENCE_DROPOUT) > 1e-12:
            log.warning(
                "Fusion-LSTM does not use the recurrent dropout path; ignoring --sequence-dropout=%.3f.",
                args.sequence_dropout,
            )
        if args.attention_layers != DEFAULT_ATTENTION_LAYERS:
            log.warning(
                "Fusion-LSTM uses a single self-attention block; ignoring --attention-layers=%d.",
                args.attention_layers,
            )
    if architecture == "au":
        au_hidden_size = 64
        au_sequence_layers = 1
        au_attention_heads = 4
        au_attention_layers = 1
        au_sequence_dropout = 0.0
        au_attention_dropout = 0.0
        if not use_multihead_attention:
            log.info("AU includes its attention block by design; enabling attention automatically.")
        use_multihead_attention = True
        if args.sequence_hidden_size != au_hidden_size:
            log.warning(
                "AU fixes its LSTM width to 64; ignoring --sequence-hidden-size=%d.",
                args.sequence_hidden_size,
            )
        if args.sequence_layers != au_sequence_layers:
            log.warning("AU uses a single LSTM layer; ignoring --sequence-layers=%d.", args.sequence_layers)
        if abs(args.sequence_dropout - au_sequence_dropout) > 1e-12:
            log.warning("AU does not use recurrent dropout; ignoring --sequence-dropout=%.3f.", args.sequence_dropout)
        if args.attention_heads != au_attention_heads:
            log.warning("AU fixes its attention heads to 4; ignoring --attention-heads=%d.", args.attention_heads)
        if args.attention_layers != au_attention_layers:
            log.warning("AU uses one attention block; ignoring --attention-layers=%d.", args.attention_layers)
        if abs(args.attention_dropout - au_attention_dropout) > 1e-12:
            log.warning("AU uses zero attention dropout; ignoring --attention-dropout=%.3f.", args.attention_dropout)
    if architecture == "chronos_bolt" and use_multihead_attention:
        log.warning("Chronos-Bolt backend ignores multihead-attention settings.")
        use_multihead_attention = False
    if architecture != "chronos_bolt" and (
        args.chronos_patch_aligned_context or args.chronos_auto_context or args.chronos_ensemble_contexts
    ):
        log.warning("Chronos context switches are ignored unless MODEL_ARCHITECTURE is chronos_bolt.")
    if use_fixed_time_bars and _shared.PRIMARY_BAR_SECONDS <= 0:
        raise ValueError("PRIMARY_BAR_SECONDS must be positive.")
    if use_fixed_tick_bars and args.primary_tick_density <= 0:
        raise ValueError("PRIMARY_TICK_DENSITY must be positive.")
    if _shared.DEFAULT_FIXED_MOVE <= 0.0:
        raise ValueError("DEFAULT_FIXED_MOVE must be positive in points.")

    data_path = resolve_local_path(args.data_file)
    archive_only = bool(args.archive_only)
    if archive_only and not args.skip_live_compile:
        log.info("archive-only mode implies skip-live-compile; local live references will still be updated.")

    requested_device = str(args.device).strip() or "cpu"
    device = torch.device(requested_device)
    log.info("Using device: %s", device)
    log.info(
        "Shared config | path=%s seq_len=%d horizon=%d atr_feature=%d atr_target=%d rv=%d ret=%d "
        "bar_mode=%s imbalance_min_ticks=%d imbalance_ema_span=%d bar_seconds=%d tick_density=%d risk_mode=%s fixed_move_points=%.2f "
        "label_sl=%.2f label_tp=%.2f exec_sl=%.2f exec_tp=%.2f use_all_windows=%d",
        CURRENT_CONFIG_PATH,
        SEQ_LEN,
        TARGET_HORIZON,
        FEATURE_ATR_PERIOD,
        TARGET_ATR_PERIOD,
        RV_PERIOD,
        RETURN_PERIOD,
        "FIXED_TICK" if use_fixed_tick_bars else ("FIXED_TIME" if use_fixed_time_bars else "IMBALANCE"),
        IMBALANCE_MIN_TICKS,
        IMBALANCE_EMA_SPAN,
        PRIMARY_BAR_SECONDS,
        args.primary_tick_density,
        "ATR" if use_atr_risk else "FIXED",
        DEFAULT_FIXED_MOVE,
        LABEL_SL_MULTIPLIER,
        LABEL_TP_MULTIPLIER,
        EXECUTION_SL_MULTIPLIER,
        EXECUTION_TP_MULTIPLIER,
        int(USE_ALL_WINDOWS),
    )
    log.info(
        "Run config | symbol=%s architecture=%s attention=%d focal_gamma=%.2f feature_profile=%s feature_count=%d",
        SYMBOL,
        architecture.upper(),
        int(use_multihead_attention),
        args.focal_gamma,
        feature_profile,
        feature_count,
    )

    bars, point_size = build_market_bars_frame(
        data_path,
        use_fixed_time_bars=use_fixed_time_bars,
        use_fixed_tick_bars=use_fixed_tick_bars,
        tick_density=args.primary_tick_density,
        max_bars=int(args.max_bars),
        bar_duration_ms=BAR_DURATION_MS,
        imbalance_min_ticks=IMBALANCE_MIN_TICKS,
        imbalance_ema_span=IMBALANCE_EMA_SPAN,
        use_imbalance_ema_threshold=USE_IMBALANCE_EMA_THRESHOLD,
        use_imbalance_min_ticks_div3_threshold=USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD,
        require_gold_context=bool(args.gold_context) and not bool(SHARED.get("USE_MINIMAL_FEATURE_SET", False)),
    )
    fixed_move_price = fixed_move_distance(DEFAULT_FIXED_MOVE, point_size)
    log.info(
        "Fixed risk config | fixed_move_points=%.2f point_size=%.8f fixed_move_price=%.8f",
        DEFAULT_FIXED_MOVE,
        point_size,
        fixed_move_price,
    )
    x_all = build_feature_array(
        bars,
        feature_columns=feature_columns,
        config=FeatureEngineeringConfig.from_values(SHARED),
    )
    y_all = build_triple_barrier_labels(
        bars,
        use_atr_risk=use_atr_risk,
        fixed_move_price=fixed_move_price,
        target_horizon=TARGET_HORIZON,
        target_atr_period=TARGET_ATR_PERIOD,
        label_tp_multiplier=LABEL_TP_MULTIPLIER,
        label_sl_multiplier=LABEL_SL_MULTIPLIER,
    )

    x = x_all[WARMUP_BARS:]
    y = y_all[WARMUP_BARS:]
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

    train_end_idx_all = build_segment_end_indices(valid_mask, *train_range, SEQ_LEN, TARGET_HORIZON)
    val_end_idx_all = build_segment_end_indices(valid_mask, *val_range, SEQ_LEN, TARGET_HORIZON)
    test_end_idx_all = build_segment_end_indices(valid_mask, *test_range, SEQ_LEN, TARGET_HORIZON)
    train_end_idx = maybe_cap_windows(train_end_idx_all, args.max_train_windows, USE_ALL_WINDOWS)
    val_end_idx = maybe_cap_windows(val_end_idx_all, args.max_eval_windows, USE_ALL_WINDOWS)
    test_end_idx = maybe_cap_windows(test_end_idx_all, args.max_eval_windows, USE_ALL_WINDOWS)
    if min(len(train_end_idx), len(val_end_idx), len(test_end_idx)) == 0:
        raise ValueError("One or more leakage-safe splits ended up empty.")
    log.info(
        "Window usage | train=%d/%d val=%d/%d test=%d/%d",
        len(train_end_idx),
        len(train_end_idx_all),
        len(val_end_idx),
        len(val_end_idx_all),
        len(test_end_idx),
        len(test_end_idx_all),
    )

    x_train, y_train = build_windows(x_scaled, y, train_end_idx, SEQ_LEN)
    x_val, y_val = build_windows(x_scaled, y, val_end_idx, SEQ_LEN)
    x_test, y_test = build_windows(x_scaled, y, test_end_idx, SEQ_LEN)
    log.info("Window counts | train=%d val=%d test=%d", len(x_train), len(x_val), len(x_test))

    if use_no_hold:
        train_mask = y_train > 0
        val_mask = y_val > 0
        test_mask = y_test > 0
        x_train = x_train[train_mask]
        y_train = y_train[train_mask] - 1
        x_val = x_val[val_mask]
        y_val = y_val[val_mask] - 1
        x_test = x_test[test_mask]
        y_test = y_test[test_mask] - 1
        log.info(
            "No-hold mode | filtered train=%d val=%d test=%d",
            len(x_train),
            len(x_val),
            len(x_test),
        )

    loss_mode = resolve_loss_mode(architecture, args.loss_mode)
    export_model: nn.Module | None = None

    if architecture == "chronos_bolt":
        if args.loss_mode != "auto":
            log.warning("Chronos-Bolt backend is zero-shot; ignoring --loss-mode=%s.", args.loss_mode)
        if args.lr > 0.0 or args.weight_decay >= 0.0:
            log.warning("Chronos-Bolt backend is zero-shot; ignoring optimizer overrides.")
        log.info(
            "Chronos-Bolt backend | model_id=%s feature_profile=%s prediction_length=%d",
            args.chronos_bolt_model,
            feature_profile,
            TARGET_HORIZON,
        )
        chronos_bolt_batch_size = max(1, min(args.batch_size, 16))
        if chronos_bolt_batch_size != args.batch_size:
            log.info(
                "Chronos-Bolt batch size capped to %d tasks to keep CPU memory use predictable.",
                chronos_bolt_batch_size,
            )
        export_model = load_chronos_bolt_barrier_model(
            device=device,
            model_id=args.chronos_bolt_model,
            median=median,
            iqr=iqr,
            feature_columns=feature_columns,
            prediction_length=TARGET_HORIZON,
            use_atr_risk=use_atr_risk,
            label_tp_multiplier=LABEL_TP_MULTIPLIER,
            label_sl_multiplier=LABEL_SL_MULTIPLIER,
            context_tail_lengths=(0,),
        ).to(device)
        val_loader = build_loader(
            x_val,
            y_val,
            chronos_bolt_batch_size,
            shuffle=False,
        )
        test_loader = build_loader(
            x_test,
            y_test,
            chronos_bolt_batch_size,
            shuffle=False,
        )
        context_variants = chronos_context_variants(
            args,
            sequence_length=SEQ_LEN,
            patch_size=int(getattr(export_model, "patch_size", 0)),
        )
        selected_context_variant = context_variants[0]

        if args.chronos_auto_context and len(context_variants) > 1:
            best_score: tuple[float, float, int, float] | None = None
            best_val_logits: np.ndarray | None = None
            best_val_labels: np.ndarray | None = None

            for candidate_context_variant in context_variants:
                export_model.set_context_tail_lengths(candidate_context_variant)
                candidate_val_logits, candidate_val_labels = run_model_evaluation(export_model, val_loader, device)
                candidate_val_probs = compute_softmax(candidate_val_logits)
                candidate_threshold = select_confidence_threshold(
                    candidate_val_probs,
                    candidate_val_labels,
                    min_selected=max(1, args.min_selected_trades),
                    threshold_min=args.confidence_search_min,
                    threshold_max=args.confidence_search_max,
                    threshold_steps=args.confidence_search_steps,
                )
                candidate_validation_gate = compute_gate_metrics(
                    candidate_val_labels,
                    candidate_val_probs,
                    candidate_threshold,
                )
                candidate_score = chronos_context_score(
                    candidate_validation_gate,
                    min_selected=max(1, args.min_selected_trades),
                )
                log.info(
                    "Chronos auto-context | candidate=%s threshold=%.2f precision=%s coverage=%.4f trades=%d",
                    chronos_context_label(candidate_context_variant),
                    candidate_threshold,
                    format_metric(float(candidate_validation_gate["precision"])),
                    float(candidate_validation_gate["trade_coverage"]),
                    int(candidate_validation_gate["selected_trades"]),
                )
                if best_score is None or candidate_score > best_score:
                    best_score = candidate_score
                    selected_context_variant = candidate_context_variant
                    best_val_logits = candidate_val_logits
                    best_val_labels = candidate_val_labels

            if best_val_logits is None or best_val_labels is None:
                raise RuntimeError("Chronos auto-context search did not evaluate any candidates.")
            export_model.set_context_tail_lengths(selected_context_variant)
            log.info(
                "Chronos auto-context selected %s.",
                chronos_context_label(selected_context_variant),
            )
            val_logits, val_labels = best_val_logits, best_val_labels
        else:
            export_model.set_context_tail_lengths(selected_context_variant)
            if architecture == "chronos_bolt":
                log.info(
                    "Chronos context mode | selected=%s",
                    chronos_context_label(selected_context_variant),
                )
            val_logits, val_labels = run_model_evaluation(export_model, val_loader, device)

        model_backend = getattr(export_model, "backend_name", "chronos-bolt-zero-shot-close-barrier")
        test_logits, test_labels = run_model_evaluation(export_model, test_loader, device)
    else:
        scheduler = None
        feature_mean: np.ndarray | None = None
        feature_std: np.ndarray | None = None
        token_mean: np.ndarray | None = None
        token_std: np.ndarray | None = None
        minirocket_parameters = None
        train_sample_weights = build_sample_weights(y_train, class_count=len(active_label_names))

        if architecture == "minirocket":
            transform_batch_size = max(args.batch_size, DEFAULT_BATCH_SIZE)
            minirocket_parameters = fit_minirocket(
                x_train.transpose(0, 2, 1),
                num_features=args.minirocket_features,
                seed=42,
            )
            learning_rate = args.lr if args.lr > 0.0 else DEFAULT_MINIROCKET_LR
            weight_decay = args.weight_decay if args.weight_decay >= 0.0 else DEFAULT_MINIROCKET_WEIGHT_DECAY

            if use_multihead_attention:
                train_inputs = transform_sequence_tokens(
                    minirocket_parameters,
                    x_train,
                    batch_size=transform_batch_size,
                    device=device,
                )
                val_inputs = transform_sequence_tokens(
                    minirocket_parameters,
                    x_val,
                    batch_size=transform_batch_size,
                    device=device,
                )
                test_inputs = transform_sequence_tokens(
                    minirocket_parameters,
                    x_test,
                    batch_size=transform_batch_size,
                    device=device,
                )

                token_mean = train_inputs.mean(axis=0).astype(np.float32)
                token_std = np.where(train_inputs.std(axis=0) < 1e-6, 1.0, train_inputs.std(axis=0)).astype(np.float32)
                train_inputs = ((train_inputs - token_mean) / token_std).astype(np.float32, copy=False)
                val_inputs = ((val_inputs - token_mean) / token_std).astype(np.float32, copy=False)
                test_inputs = ((test_inputs - token_mean) / token_std).astype(np.float32, copy=False)

                training_model = MiniRocketMultiAttentionHead(
                    num_tokens=train_inputs.shape[1],
                    token_dim=train_inputs.shape[2],
                    n_classes=len(active_label_names),
                    model_dim=args.attention_dim,
                    num_heads=args.attention_heads,
                    num_layers=args.attention_layers,
                    dropout=args.attention_dropout,
                ).to(device)
                model_backend = "minirocket-multivariate-attention"
            else:
                train_inputs = transform_sequences(
                    minirocket_parameters,
                    x_train,
                    batch_size=transform_batch_size,
                    device=device,
                )
                val_inputs = transform_sequences(
                    minirocket_parameters,
                    x_val,
                    batch_size=transform_batch_size,
                    device=device,
                )
                test_inputs = transform_sequences(
                    minirocket_parameters,
                    x_test,
                    batch_size=transform_batch_size,
                    device=device,
                )

                feature_mean = train_inputs.mean(axis=0).astype(np.float32)
                feature_std = np.where(train_inputs.std(axis=0) < 1e-6, 1.0, train_inputs.std(axis=0)).astype(np.float32)
                train_inputs = ((train_inputs - feature_mean) / feature_std).astype(np.float32, copy=False)
                val_inputs = ((val_inputs - feature_mean) / feature_std).astype(np.float32, copy=False)
                test_inputs = ((test_inputs - feature_mean) / feature_std).astype(np.float32, copy=False)

                training_model = nn.Linear(train_inputs.shape[1], len(active_label_names)).to(device)
                model_backend = "minirocket-multivariate"

            optimizer = torch.optim.AdamW(training_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                min_lr=1e-8,
                patience=max(1, args.patience // 2),
            )
            train_loader = build_loader(
                train_inputs,
                y_train,
                args.batch_size,
                shuffle=True,
                sample_weights=train_sample_weights,
            )
            val_loader = build_loader(
                val_inputs,
                y_val,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
            test_loader = build_loader(
                test_inputs,
                y_test,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
        else:
            if architecture in {
                "ela",
                "fusion_lstm",
                "bilstm",
                "gru",
                "tcn",
                "tla",
                "au",
                "legacy_lstm_attention",
                "gold_legacy",
                "gold_new",
            }:
                learning_rate = args.lr if args.lr > 0.0 else DEFAULT_SEQUENCE_LR
                weight_decay = args.weight_decay if args.weight_decay >= 0.0 else DEFAULT_SEQUENCE_WEIGHT_DECAY
                if architecture == "legacy_lstm_attention":
                    training_model = LegacyLSTMAttentionClassifier(
                        n_features=feature_count,
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "gold_legacy":
                    training_model = GoldLegacyLSTMAttentionClassifier(
                        n_features=feature_count,
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "gold_new":
                    training_model = GoldNewTemporalClassifier(
                        n_features=feature_count,
                        channels=max(16, min(64, args.sequence_hidden_size)),
                        hidden=max(32, min(64, args.sequence_hidden_size)),
                        dense_hidden=max(64, feature_count * 2),
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                        dropout=args.sequence_dropout,
                        n_classes=len(active_label_names),
                    ).to(device)
                elif architecture == "au":
                    training_model = AuLSTMMultiheadAttentionClassifier(
                        n_features=feature_count,
                        n_classes=len(active_label_names),
                    ).to(device)
                elif architecture == "fusion_lstm":
                    training_model = FusionLSTMClassifier(
                        n_features=feature_count,
                        hidden=20,
                        attention_heads=args.attention_heads,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "tcn":
                    training_model = TCNClassifier(
                        n_features=feature_count,
                        channels=args.sequence_hidden_size,
                        hidden=max(args.sequence_hidden_size, feature_count * 4),
                        dropout=args.sequence_dropout,
                        n_layers=args.tcn_levels,
                        kernel_size=args.tcn_kernel_size,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                elif architecture == "tla":
                    training_model = TemporalLSTMAttentionClassifier(
                        n_features=feature_count,
                        conv_channels=128,
                        lstm_hidden=args.sequence_hidden_size,
                        hidden=max(args.sequence_hidden_size, feature_count * 4),
                        dropout=args.sequence_dropout,
                        conv_kernel_size=5,
                        lstm_layers=args.sequence_layers,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                else:
                    recurrent_cell_type = "gru" if architecture == "gru" else "lstm"
                    recurrent_bidirectional = architecture == "bilstm"
                    backend_name = {
                        "ela": "ela-lstm-attention",
                        "bilstm": "bilstm-attention" if use_multihead_attention else "bilstm",
                        "gru": "gru-attention" if use_multihead_attention else "gru",
                    }[architecture]
                    training_model = RecurrentSequenceClassifier(
                        n_features=feature_count,
                        cell_type=recurrent_cell_type,
                        hidden_size=args.sequence_hidden_size,
                        hidden=max(args.sequence_hidden_size, feature_count * 4),
                        dropout=args.sequence_dropout,
                        num_layers=args.sequence_layers,
                        bidirectional=recurrent_bidirectional,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                        backend_name=backend_name,
                    ).to(device)
            else:
                learning_rate = args.lr if args.lr > 0.0 else DEFAULT_MAMBA_LR
                weight_decay = args.weight_decay if args.weight_decay >= 0.0 else DEFAULT_MAMBA_WEIGHT_DECAY
                if architecture == "castor":
                    training_model = CastorClassifier(
                        n_features=feature_count,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
                else:
                    training_model = MambaLiteClassifier(
                        n_features=feature_count,
                        use_multihead_attention=use_multihead_attention,
                        attention_heads=args.attention_heads,
                        attention_layers=args.attention_layers,
                        attention_dropout=args.attention_dropout,
                    ).to(device)
            optimizer = torch.optim.AdamW(training_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            train_loader = build_loader(
                x_train,
                y_train,
                args.batch_size,
                shuffle=True,
                sample_weights=train_sample_weights,
            )
            val_loader = build_loader(
                x_val,
                y_val,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
            test_loader = build_loader(
                x_test,
                y_test,
                max(args.batch_size, DEFAULT_BATCH_SIZE),
                shuffle=False,
            )
            model_backend = getattr(training_model, "backend_name", "portable-mamba-lite")

        class_weights = build_class_weights(y_train, class_count=len(active_label_names)).to(device)
        if loss_mode == "cross-entropy":
            criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights).to(device)
        else:
            criterion = PipelineFocalLoss(alpha=class_weights, gamma=args.focal_gamma).to(device)
        log.info(
            "Optimization | loss=%s lr=%.6g weight_decay=%.6g balanced_sampling=1 confidence_search=[%.2f, %.2f]x%d",
            loss_mode,
            learning_rate,
            weight_decay,
            args.confidence_search_min,
            args.confidence_search_max,
            args.confidence_search_steps,
        )

        best_state = None
        best_val_loss = float("inf")
        wait = 0

        for epoch in tqdm(range(args.epochs), desc="Training"):
            training_model.train()
            train_losses = []
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
                logits = training_model(xb.to(device))
                loss = criterion(logits, yb.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            val_logits, val_labels = run_model_evaluation(training_model, val_loader, device)
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
                args.patience,
            )
            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in training_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a valid checkpoint.")

        training_model.load_state_dict(best_state)
        training_model.to(device)

        val_logits, val_labels = run_model_evaluation(training_model, val_loader, device)
        test_logits, test_labels = run_model_evaluation(training_model, test_loader, device)

        if architecture == "minirocket":
            if minirocket_parameters is None:
                raise RuntimeError("MiniRocket export requested without fitted parameters.")
            if use_multihead_attention:
                export_model = MiniRocketClassifier(
                    parameters=minirocket_parameters,
                    n_classes=len(active_label_names),
                    token_mean=token_mean,
                    token_std=token_std,
                    head_type="multiattention",
                    attention_dim=args.attention_dim,
                    attention_heads=args.attention_heads,
                    attention_layers=args.attention_layers,
                    attention_dropout=args.attention_dropout,
                )
            else:
                export_model = MiniRocketClassifier(
                    parameters=minirocket_parameters,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    n_classes=len(active_label_names),
                    head_type="linear",
                )
            export_model.head.load_state_dict(training_model.state_dict())
        else:
            export_model = training_model
    val_probs = compute_softmax(val_logits)
    selected_primary_confidence = select_confidence_threshold(
        val_probs,
        val_labels,
        min_selected=max(1, args.min_selected_trades),
        threshold_min=args.confidence_search_min,
        threshold_max=args.confidence_search_max,
        threshold_steps=args.confidence_search_steps,
    )
    validation_gate = log_gate_summary("validation", val_probs, val_labels, selected_primary_confidence)

    test_probs = compute_softmax(test_logits)
    holdout_gate = log_gate_summary("holdout", test_probs, test_labels, selected_primary_confidence)

    quality_gate_reasons: list[str] = []
    if int(validation_gate["selected_trades"]) < args.min_selected_trades:
        quality_gate_reasons.append(
            "validation selected trades "
            f"{int(validation_gate['selected_trades'])} < required {args.min_selected_trades}"
        )
    validation_precision = float(validation_gate["precision"])
    if not np.isfinite(validation_precision):
        quality_gate_reasons.append("validation selected-trade precision unavailable")
    elif validation_precision < args.min_trade_precision:
        quality_gate_reasons.append(
            f"validation selected-trade precision {validation_precision:.4f} < required {args.min_trade_precision:.4f}"
        )
    quality_gate_passed = len(quality_gate_reasons) == 0
    quality_gate_reason = "; ".join(quality_gate_reasons)
    deployed_primary_confidence = selected_primary_confidence
    if not quality_gate_passed:
        log.warning(
            "Model failed the live quality gate (%s). Keeping PRIMARY_CONFIDENCE=%.2f.",
            quality_gate_reason,
            deployed_primary_confidence,
        )

    if export_model is None:
        raise RuntimeError("Model export path was not initialized.")

    completed_at = datetime.now()
    model_dir_name = format_model_dir_name(
        value=completed_at,
        name=model_name,
        failed_quality_gate=not quality_gate_passed,
        symbol=SYMBOL,
    )
    model_dir = symbol_models_dir(SYMBOL) / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_dir = model_diagnostics_dir(model_dir)
    model_test_dir = model_dir / "tests"
    model_test_dir.mkdir(parents=True, exist_ok=True)
    archive_output_path = model_dir / "model.onnx"
    combined_model_config_path = model_dir / "config.mqh"

    export_model.eval()
    export_model.to("cpu")
    dummy = torch.randn(1, SEQ_LEN, feature_count)
    export_onnx_model(export_model, dummy, archive_output_path)

    model_config_text = (
        render_mql_config(
            project=project,
            active_config_path=CURRENT_CONFIG_PATH,
            median=median,
            iqr=iqr,
            primary_confidence=deployed_primary_confidence,
            use_atr_risk=use_atr_risk,
            use_fixed_time_bars=use_fixed_time_bars,
            architecture=architecture,
            use_multihead_attention=use_multihead_attention,
            feature_columns=feature_columns,
            feature_profile=feature_profile,
            use_extended_features=use_extended_features,
            use_fixed_tick_bars=use_fixed_tick_bars,
            max_feature_lookback=MAX_FEATURE_LOOKBACK,
            warmup_bars=WARMUP_BARS,
        )
        + "\n"
    )
    combined_model_config_path.write_text(model_config_text, encoding="utf-8")
    write_diagnostics_report(
        diagnostics_dir=diagnostics_dir,
        config=DiagnosticsConfig(
            current_config_name=CURRENT_CONFIG_PATH.name,
            seq_len=SEQ_LEN,
            target_horizon=TARGET_HORIZON,
            primary_bar_seconds=PRIMARY_BAR_SECONDS,
            imbalance_min_ticks=IMBALANCE_MIN_TICKS,
            imbalance_ema_span=IMBALANCE_EMA_SPAN,
            feature_atr_period=FEATURE_ATR_PERIOD,
            target_atr_period=TARGET_ATR_PERIOD,
            rv_period=RV_PERIOD,
            return_period=RETURN_PERIOD,
            warmup_bars=WARMUP_BARS,
            default_fixed_move=DEFAULT_FIXED_MOVE,
            label_sl_multiplier=LABEL_SL_MULTIPLIER,
            label_tp_multiplier=LABEL_TP_MULTIPLIER,
            execution_sl_multiplier=EXECUTION_SL_MULTIPLIER,
            execution_tp_multiplier=EXECUTION_TP_MULTIPLIER,
            use_all_windows=USE_ALL_WINDOWS,
        ),
        bars=bars,
        y_full=y,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        val_probs=val_probs,
        test_probs=test_probs,
        label_names=active_label_names,
        selected_primary_confidence=selected_primary_confidence,
        deployed_primary_confidence=deployed_primary_confidence,
        validation_gate=validation_gate,
        holdout_gate=holdout_gate,
        quality_gate_passed=quality_gate_passed,
        quality_gate_reason=quality_gate_reason,
        available_window_counts={
            "train": len(train_end_idx_all),
            "validation": len(val_end_idx_all),
            "holdout": len(test_end_idx_all),
        },
        used_window_counts={
            "train": len(train_end_idx),
            "validation": len(val_end_idx),
            "holdout": len(test_end_idx),
        },
        use_atr_risk=use_atr_risk,
        use_fixed_time_bars=use_fixed_time_bars,
        symbol=SYMBOL,
        model_backend=model_backend,
        loss_mode=loss_mode,
        focal_gamma=args.focal_gamma,
        model_config_text=model_config_text,
        feature_columns=feature_columns,
        feature_profile=feature_profile,
        point_size=point_size,
        fixed_move_price=fixed_move_price,
        use_fixed_tick_bars=use_fixed_tick_bars,
        tick_density=args.primary_tick_density,
    )
    if not archive_only:
        shutil.rmtree(ACTIVE_DIAGNOSTICS_DIR, ignore_errors=True)
        shutil.copytree(diagnostics_dir, ACTIVE_DIAGNOSTICS_DIR)
    ensure_default_test_config(model_test_dir, symbol=SYMBOL)

    set_live_model_reference(model_dir)

    if not archive_only and not args.skip_live_compile:
        runtime_paths = resolve_mt5_runtime(metaeditor_path_override=args.metaeditor_path)
        compile_log_path = compile_live_expert(runtime_paths, model_dir=model_dir)
        warnings_match = re.search(
            r"Result:\s+(\d+)\s+errors?,\s+(\d+)\s+warnings?",
            read_text_best_effort(compile_log_path),
        )
        warnings = int(warnings_match.group(2)) if warnings_match else 0
        log.info(
            "Compiled live EA successfully with %d warnings. Log: %s",
            warnings,
            compile_log_path,
        )
        shutil.copy2(compile_log_path, diagnostics_dir / compile_log_path.name)
        log.info("Saved live compile log to %s", compile_log_path)
    log.info("Archived ONNX to %s", archive_output_path)
    log.info("Saved combined model config to %s", combined_model_config_path)
    log.info("Saved diagnostics to %s", diagnostics_dir)
    log.info("Archived model artifacts to %s", model_dir)
    log.info("Total runtime: %.2fs", time.time() - t0)
    print(model_dir.name)
