int OnInit() {
   DebugPrint("Model reference: " + ACTIVE_MODEL_SYMBOL + "/" + ACTIVE_MODEL_VERSION);

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
   primary_expected_abs_theta = ResolveImbalanceThresholdBase();
   if(MODEL_USE_FIXED_TICK_BARS != 0 && PRIMARY_TICK_DENSITY <= 0) {
      Print("[FATAL] PRIMARY_TICK_DENSITY must be positive for fixed-tick bars.");
      return INIT_FAILED;
   }
   DebugPrint(
      StringFormat(
         "init seq=%d horizon=%d history=%d bar_mode=%s imbalance_min_ticks=%d imbalance_span=%d bar_seconds=%d tick_density=%d risk_mode=%s fixed_move=%.2f sl=%.2f tp=%.2f lot=%.2f risk_pct=%.3f primary_conf=%.2f",
         SEQ_LEN,
         LABEL_TIMEOUT_BARS,
         REQUIRED_HISTORY_INDEX,
         (MODEL_USE_FIXED_TICK_BARS != 0 ? "FIXED_TICK" : (MODEL_USE_FIXED_TIME_BARS != 0 ? "FIXED_TIME" : "IMBALANCE")),
         IMBALANCE_MIN_TICKS,
         IMBALANCE_EMA_SPAN,
         PRIMARY_BAR_SECONDS,
         PRIMARY_TICK_DENSITY,
         (R ? "FIXED" : "ATR"),
         FIXED_MOVE,
         SL_MULTIPLIER,
         TP_MULTIPLIER,
         LOT_SIZE,
         RISK_PERCENT,
         PRIMARY_CONFIDENCE
      )
   );
   if(PRIMARY_CONFIDENCE > 1.0) {
      Print("[INFO] Live trading disabled because the active model failed the trainer quality gate.");
   }

   MqlTick tick;
   if(SymbolInfoTick(_Symbol, tick)) {
      last_tick_time = tick.time_msc;
   } else {
      last_tick_time = TimeCurrent() * 1000ULL;
   }
   usdx_available = SymbolSelect(USDX_SYMBOL, true);
   usdjpy_available = SymbolSelect(USDJPY_SYMBOL, true);

   LoadHistory();
   return INIT_SUCCEEDED;
}
