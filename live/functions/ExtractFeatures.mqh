void ExtractFeatures(int h, float &features[]) {
   Bar bar = history[h];
   Bar prev = history[h + 1];
   double close = bar.c;

#ifdef FEATURE_IDX_RET1
   features[FEATURE_IDX_RET1] = ScaleAndClip((float)LogReturnAt(h), FEATURE_IDX_RET1);
#endif
#ifdef FEATURE_IDX_HIGH_REL_PREV
   features[FEATURE_IDX_HIGH_REL_PREV] = ScaleAndClip((float)SafeLogRatio(bar.h, prev.c), FEATURE_IDX_HIGH_REL_PREV);
#endif
#ifdef FEATURE_IDX_LOW_REL_PREV
   features[FEATURE_IDX_LOW_REL_PREV] = ScaleAndClip((float)SafeLogRatio(bar.l, prev.c), FEATURE_IDX_LOW_REL_PREV);
#endif
#ifdef FEATURE_IDX_SPREAD_REL
   features[FEATURE_IDX_SPREAD_REL] = ScaleAndClip((float)(bar.spread / (close + 1e-10)), FEATURE_IDX_SPREAD_REL);
#endif
#ifdef FEATURE_IDX_CLOSE_IN_RANGE
   features[FEATURE_IDX_CLOSE_IN_RANGE] = ScaleAndClip(
      (float)((close - bar.l) / (bar.h - bar.l + 1e-8)),
      FEATURE_IDX_CLOSE_IN_RANGE
   );
#endif
#ifdef FEATURE_IDX_ATR_REL
   features[FEATURE_IDX_ATR_REL] = ScaleAndClip((float)(bar.atr_feature / (close + 1e-10)), FEATURE_IDX_ATR_REL);
#endif
#ifdef FEATURE_IDX_RV
   features[FEATURE_IDX_RV] = ScaleAndClip((float)RollingStdReturn(h, RV_PERIOD), FEATURE_IDX_RV);
#endif
#ifdef FEATURE_IDX_RETURN_N
   features[FEATURE_IDX_RETURN_N] = ScaleAndClip((float)ReturnOverBars(h, RETURN_PERIOD), FEATURE_IDX_RETURN_N);
#endif
#ifdef FEATURE_IDX_TICK_IMBALANCE
   features[FEATURE_IDX_TICK_IMBALANCE] = ScaleAndClip((float)bar.tick_imbalance, FEATURE_IDX_TICK_IMBALANCE);
#endif
#ifdef FEATURE_IDX_RET_2
   features[FEATURE_IDX_RET_2] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_2_PERIOD), FEATURE_IDX_RET_2);
#endif
#ifdef FEATURE_IDX_RET_3
   features[FEATURE_IDX_RET_3] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_3_PERIOD), FEATURE_IDX_RET_3);
#endif
#ifdef FEATURE_IDX_RET_6
   features[FEATURE_IDX_RET_6] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_6_PERIOD), FEATURE_IDX_RET_6);
#endif
#ifdef FEATURE_IDX_RET_12
   features[FEATURE_IDX_RET_12] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_12_PERIOD), FEATURE_IDX_RET_12);
#endif
#ifdef FEATURE_IDX_RET_20
   features[FEATURE_IDX_RET_20] = ScaleAndClip((float)ReturnOverBars(h, FEATURE_RET_20_PERIOD), FEATURE_IDX_RET_20);
#endif
#ifdef FEATURE_IDX_RANGE_REL
   features[FEATURE_IDX_RANGE_REL] = ScaleAndClip((float)((bar.h - bar.l) / (close + 1e-10)), FEATURE_IDX_RANGE_REL);
#endif
#ifdef FEATURE_IDX_BODY_REL
   features[FEATURE_IDX_BODY_REL] = ScaleAndClip((float)((bar.c - bar.o) / (close + 1e-10)), FEATURE_IDX_BODY_REL);
#endif
#ifdef FEATURE_IDX_UPPER_WICK_REL
   features[FEATURE_IDX_UPPER_WICK_REL] = ScaleAndClip(
      (float)((bar.h - MathMax(bar.o, bar.c)) / (close + 1e-10)),
      FEATURE_IDX_UPPER_WICK_REL
   );
#endif
#ifdef FEATURE_IDX_LOWER_WICK_REL
   features[FEATURE_IDX_LOWER_WICK_REL] = ScaleAndClip(
      (float)((MathMin(bar.o, bar.c) - bar.l) / (close + 1e-10)),
      FEATURE_IDX_LOWER_WICK_REL
   );
#endif
#ifdef FEATURE_IDX_CLOSE_REL_SMA_9
   features[FEATURE_IDX_CLOSE_REL_SMA_9] = ScaleAndClip(
      (float)SafeLogRatio(close, MeanClose(h, FEATURE_SMA_MID_PERIOD)),
      FEATURE_IDX_CLOSE_REL_SMA_9
   );
#endif
#ifdef FEATURE_IDX_CLOSE_REL_SMA_20
   features[FEATURE_IDX_CLOSE_REL_SMA_20] = ScaleAndClip(
      (float)SafeLogRatio(close, MeanClose(h, FEATURE_SMA_SLOW_PERIOD)),
      FEATURE_IDX_CLOSE_REL_SMA_20
   );
#endif
#ifdef FEATURE_IDX_CLOSE_REL_SMA_3
   features[FEATURE_IDX_CLOSE_REL_SMA_3] = ScaleAndClip(
      (float)SafeLogRatio(close, MeanClose(h, FEATURE_SMA_FAST_PERIOD)),
      FEATURE_IDX_CLOSE_REL_SMA_3
   );
#endif
#ifdef FEATURE_IDX_SMA_3_9_GAP
   features[FEATURE_IDX_SMA_3_9_GAP] = ScaleAndClip(
      (float)SafeLogRatio(MeanClose(h, FEATURE_SMA_FAST_PERIOD), MeanClose(h, FEATURE_SMA_MID_PERIOD)),
      FEATURE_IDX_SMA_3_9_GAP
   );
#endif
#ifdef FEATURE_IDX_SMA_5_20_GAP
   features[FEATURE_IDX_SMA_5_20_GAP] = ScaleAndClip(
      (float)SafeLogRatio(MeanClose(h, FEATURE_SMA_TREND_FAST_PERIOD), MeanClose(h, FEATURE_SMA_SLOW_PERIOD)),
      FEATURE_IDX_SMA_5_20_GAP
   );
#endif
#ifdef FEATURE_IDX_SMA_9_20_GAP
   features[FEATURE_IDX_SMA_9_20_GAP] = ScaleAndClip(
      (float)SafeLogRatio(MeanClose(h, FEATURE_SMA_MID_PERIOD), MeanClose(h, FEATURE_SMA_SLOW_PERIOD)),
      FEATURE_IDX_SMA_9_20_GAP
   );
#endif
#ifdef FEATURE_IDX_SMA_SLOPE_9
   features[FEATURE_IDX_SMA_SLOPE_9] = ScaleAndClip(
      (float)SafeLogRatio(
         MeanClose(h, FEATURE_SMA_MID_PERIOD),
         MeanClose(h + FEATURE_SMA_SLOPE_SHIFT, FEATURE_SMA_MID_PERIOD)
      ),
      FEATURE_IDX_SMA_SLOPE_9
   );
#endif
#ifdef FEATURE_IDX_SMA_SLOPE_20
   features[FEATURE_IDX_SMA_SLOPE_20] = ScaleAndClip(
      (float)SafeLogRatio(
         MeanClose(h, FEATURE_SMA_SLOW_PERIOD),
         MeanClose(h + FEATURE_SMA_SLOPE_SHIFT, FEATURE_SMA_SLOW_PERIOD)
      ),
      FEATURE_IDX_SMA_SLOPE_20
   );
#endif
#ifdef FEATURE_IDX_RSI_6
   features[FEATURE_IDX_RSI_6] = ScaleAndClip((float)SimpleRsi(h, FEATURE_RSI_FAST_PERIOD), FEATURE_IDX_RSI_6);
#endif
#ifdef FEATURE_IDX_RSI_14
   features[FEATURE_IDX_RSI_14] = ScaleAndClip((float)SimpleRsi(h, FEATURE_RSI_SLOW_PERIOD), FEATURE_IDX_RSI_14);
#endif
#ifdef FEATURE_IDX_STOCH_K_9
   features[FEATURE_IDX_STOCH_K_9] = ScaleAndClip((float)StochK(h, FEATURE_STOCH_PERIOD), FEATURE_IDX_STOCH_K_9);
#endif
#ifdef FEATURE_IDX_STOCH_D_3
   features[FEATURE_IDX_STOCH_D_3] = ScaleAndClip((float)StochD(h, FEATURE_STOCH_SMOOTH_PERIOD), FEATURE_IDX_STOCH_D_3);
#endif
#ifdef FEATURE_IDX_STOCH_GAP
   features[FEATURE_IDX_STOCH_GAP] = ScaleAndClip(
      (float)(StochK(h, FEATURE_STOCH_PERIOD) - StochD(h, FEATURE_STOCH_SMOOTH_PERIOD)),
      FEATURE_IDX_STOCH_GAP
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_POS_20
   {
      double sma20 = MeanClose(h, FEATURE_BOLLINGER_PERIOD);
      double std20 = StdClose(h, FEATURE_BOLLINGER_PERIOD);
      features[FEATURE_IDX_BOLLINGER_POS_20] = ScaleAndClip(
         (float)((close - sma20) / (2.0 * std20 + 1e-10)),
         FEATURE_IDX_BOLLINGER_POS_20
      );
   }
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_20
   {
      double sma20 = MeanClose(h, FEATURE_BOLLINGER_PERIOD);
      double std20 = StdClose(h, FEATURE_BOLLINGER_PERIOD);
      features[FEATURE_IDX_BOLLINGER_WIDTH_20] = ScaleAndClip(
         (float)((4.0 * std20) / (sma20 + 1e-10)),
         FEATURE_IDX_BOLLINGER_WIDTH_20
      );
   }
#endif
#ifdef FEATURE_IDX_ATR_RATIO_20
   {
      double mean_atr = MeanAtrFeature(h, FEATURE_ATR_RATIO_PERIOD);
      features[FEATURE_IDX_ATR_RATIO_20] = ScaleAndClip(
         (float)SafeLogRatio(bar.atr_feature, mean_atr),
         FEATURE_IDX_ATR_RATIO_20
      );
   }
#endif
#ifdef FEATURE_IDX_RV_18
   features[FEATURE_IDX_RV_18] = ScaleAndClip((float)RollingStdReturn(h, FEATURE_RV_LONG_PERIOD), FEATURE_IDX_RV_18);
#endif
#ifdef FEATURE_IDX_DONCHIAN_POS_20
   {
      double high20 = MaxHigh(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      double low20 = MinLow(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      features[FEATURE_IDX_DONCHIAN_POS_20] = ScaleAndClip(
         (float)((close - low20) / (high20 - low20 + 1e-10)),
         FEATURE_IDX_DONCHIAN_POS_20
      );
   }
#endif
#ifdef FEATURE_IDX_DONCHIAN_POS_9
   {
      double high9 = MaxHigh(h, FEATURE_DONCHIAN_FAST_PERIOD);
      double low9 = MinLow(h, FEATURE_DONCHIAN_FAST_PERIOD);
      features[FEATURE_IDX_DONCHIAN_POS_9] = ScaleAndClip(
         (float)((close - low9) / (high9 - low9 + 1e-10)),
         FEATURE_IDX_DONCHIAN_POS_9
      );
   }
#endif
#ifdef FEATURE_IDX_DONCHIAN_WIDTH_9
   {
      double high9 = MaxHigh(h, FEATURE_DONCHIAN_FAST_PERIOD);
      double low9 = MinLow(h, FEATURE_DONCHIAN_FAST_PERIOD);
      features[FEATURE_IDX_DONCHIAN_WIDTH_9] = ScaleAndClip(
         (float)((high9 - low9) / (close + 1e-10)),
         FEATURE_IDX_DONCHIAN_WIDTH_9
      );
   }
#endif
#ifdef FEATURE_IDX_DONCHIAN_WIDTH_20
   {
      double high20 = MaxHigh(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      double low20 = MinLow(h, FEATURE_DONCHIAN_SLOW_PERIOD);
      features[FEATURE_IDX_DONCHIAN_WIDTH_20] = ScaleAndClip(
         (float)((high20 - low20) / (close + 1e-10)),
         FEATURE_IDX_DONCHIAN_WIDTH_20
      );
   }
#endif
#ifdef FEATURE_IDX_TICK_COUNT_Z_9
   {
      double mean_tc = MeanTickCount(h, FEATURE_TICK_COUNT_PERIOD);
      double std_tc = StdTickCount(h, FEATURE_TICK_COUNT_PERIOD);
      double z = (std_tc > 1e-10) ? ((bar.tick_count - mean_tc) / std_tc) : 0.0;
      features[FEATURE_IDX_TICK_COUNT_Z_9] = ScaleAndClip((float)z, FEATURE_IDX_TICK_COUNT_Z_9);
   }
#endif
#ifdef FEATURE_IDX_TICK_COUNT_REL_9
   {
      double mean_tc = MeanTickCount(h, FEATURE_TICK_COUNT_PERIOD);
      features[FEATURE_IDX_TICK_COUNT_REL_9] = ScaleAndClip(
         (float)(bar.tick_count / (mean_tc + 1e-10) - 1.0),
         FEATURE_IDX_TICK_COUNT_REL_9
      );
   }
#endif
#ifdef FEATURE_IDX_TICK_COUNT_CHG
   features[FEATURE_IDX_TICK_COUNT_CHG] = ScaleAndClip(
      (float)MathLog((bar.tick_count + 1.0) / (history[h + 1].tick_count + 1.0)),
      FEATURE_IDX_TICK_COUNT_CHG
   );
#endif
#ifdef FEATURE_IDX_TICK_IMBALANCE_SMA_9
   features[FEATURE_IDX_TICK_IMBALANCE_SMA_9] = ScaleAndClip(
      (float)MeanTickImbalance(h, FEATURE_TICK_IMBALANCE_SLOW_PERIOD),
      FEATURE_IDX_TICK_IMBALANCE_SMA_9
   );
#endif
#ifdef FEATURE_IDX_TICK_IMBALANCE_SMA_5
   features[FEATURE_IDX_TICK_IMBALANCE_SMA_5] = ScaleAndClip(
      (float)MeanTickImbalance(h, FEATURE_TICK_IMBALANCE_FAST_PERIOD),
      FEATURE_IDX_TICK_IMBALANCE_SMA_5
   );
#endif
#ifdef FEATURE_IDX_SPREAD_Z_9
   {
      double mean_spread = MeanSpreadRel(h, FEATURE_SPREAD_Z_PERIOD);
      double std_spread = StdSpreadRel(h, FEATURE_SPREAD_Z_PERIOD);
      double spread_rel = bar.spread / (close + 1e-10);
      double z = (std_spread > 1e-10) ? ((spread_rel - mean_spread) / std_spread) : 0.0;
      features[FEATURE_IDX_SPREAD_Z_9] = ScaleAndClip((float)z, FEATURE_IDX_SPREAD_Z_9);
   }
#endif
#ifdef FEATURE_IDX_USDX_RET1
   features[FEATURE_IDX_USDX_RET1] = ScaleAndClip(
      (float)SafeLogRatio(bar.usdx_bid, prev.usdx_bid),
      FEATURE_IDX_USDX_RET1
   );
#endif
#ifdef FEATURE_IDX_USDJPY_RET1
   features[FEATURE_IDX_USDJPY_RET1] = ScaleAndClip(
      (float)SafeLogRatio(bar.usdjpy_bid, prev.usdjpy_bid),
      FEATURE_IDX_USDJPY_RET1
   );
#endif
#ifdef FEATURE_IDX_SPREAD_ABS
   features[FEATURE_IDX_SPREAD_ABS] = ScaleAndClip((float)bar.spread_mean, FEATURE_IDX_SPREAD_ABS);
#endif
#ifdef FEATURE_IDX_BAR_DURATION_MS
   features[FEATURE_IDX_BAR_DURATION_MS] = ScaleAndClip(
      (float)(bar.time_close_msc - bar.time_open_msc),
      FEATURE_IDX_BAR_DURATION_MS
   );
#endif
#ifdef FEATURE_IDX_RSI_9
   features[FEATURE_IDX_RSI_9] = ScaleAndClip((float)SimpleRsi(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_RSI_9);
#endif
#ifdef FEATURE_IDX_RSI_18
   features[FEATURE_IDX_RSI_18] = ScaleAndClip((float)SimpleRsi(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_RSI_18);
#endif
#ifdef FEATURE_IDX_RSI_27
   features[FEATURE_IDX_RSI_27] = ScaleAndClip((float)SimpleRsi(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_RSI_27);
#endif
#ifdef FEATURE_IDX_ATR_9
   features[FEATURE_IDX_ATR_9] = ScaleAndClip((float)SimpleAtr(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_ATR_9);
#endif
#ifdef FEATURE_IDX_ATR_18
   features[FEATURE_IDX_ATR_18] = ScaleAndClip((float)SimpleAtr(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_ATR_18);
#endif
#ifdef FEATURE_IDX_ATR_27
   features[FEATURE_IDX_ATR_27] = ScaleAndClip((float)SimpleAtr(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_ATR_27);
#endif
   {
      double macd_line = 0.0;
      double macd_signal = 0.0;
      double macd_hist = 0.0;
      MacdAt(h, macd_line, macd_signal, macd_hist);
      #ifdef FEATURE_IDX_MACD_LINE
         features[FEATURE_IDX_MACD_LINE] = ScaleAndClip((float)macd_line, FEATURE_IDX_MACD_LINE);
      #endif
      #ifdef FEATURE_IDX_MACD_SIGNAL
         features[FEATURE_IDX_MACD_SIGNAL] = ScaleAndClip((float)macd_signal, FEATURE_IDX_MACD_SIGNAL);
      #endif
      #ifdef FEATURE_IDX_MACD_HIST
         features[FEATURE_IDX_MACD_HIST] = ScaleAndClip((float)macd_hist, FEATURE_IDX_MACD_HIST);
      #endif
   }
#ifdef FEATURE_IDX_EMA_GAP_9
   features[FEATURE_IDX_EMA_GAP_9] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_SHORT_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_9
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_18
   features[FEATURE_IDX_EMA_GAP_18] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_MEDIUM_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_18
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_27
   features[FEATURE_IDX_EMA_GAP_27] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_LONG_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_27
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_54
   features[FEATURE_IDX_EMA_GAP_54] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_XLONG_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_54
   );
#endif
#ifdef FEATURE_IDX_EMA_GAP_144
   features[FEATURE_IDX_EMA_GAP_144] = ScaleAndClip(
      (float)(EmaClose(h, FEATURE_MAIN_XXLONG_PERIOD) - close),
      FEATURE_IDX_EMA_GAP_144
   );
#endif
#ifdef FEATURE_IDX_CCI_9
   features[FEATURE_IDX_CCI_9] = ScaleAndClip((float)SimpleCci(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_CCI_9);
#endif
#ifdef FEATURE_IDX_CCI_18
   features[FEATURE_IDX_CCI_18] = ScaleAndClip((float)SimpleCci(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_CCI_18);
#endif
#ifdef FEATURE_IDX_CCI_27
   features[FEATURE_IDX_CCI_27] = ScaleAndClip((float)SimpleCci(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_CCI_27);
#endif
#ifdef FEATURE_IDX_WILLR_9
   features[FEATURE_IDX_WILLR_9] = ScaleAndClip((float)WilliamsR(h, FEATURE_MAIN_SHORT_PERIOD), FEATURE_IDX_WILLR_9);
#endif
#ifdef FEATURE_IDX_WILLR_18
   features[FEATURE_IDX_WILLR_18] = ScaleAndClip((float)WilliamsR(h, FEATURE_MAIN_MEDIUM_PERIOD), FEATURE_IDX_WILLR_18);
#endif
#ifdef FEATURE_IDX_WILLR_27
   features[FEATURE_IDX_WILLR_27] = ScaleAndClip((float)WilliamsR(h, FEATURE_MAIN_LONG_PERIOD), FEATURE_IDX_WILLR_27);
#endif
#ifdef FEATURE_IDX_MOM_9
   features[FEATURE_IDX_MOM_9] = ScaleAndClip(
      (float)(bar.c - history[h + FEATURE_MAIN_SHORT_PERIOD].c),
      FEATURE_IDX_MOM_9
   );
#endif
#ifdef FEATURE_IDX_MOM_18
   features[FEATURE_IDX_MOM_18] = ScaleAndClip(
      (float)(bar.c - history[h + FEATURE_MAIN_MEDIUM_PERIOD].c),
      FEATURE_IDX_MOM_18
   );
#endif
#ifdef FEATURE_IDX_MOM_27
   features[FEATURE_IDX_MOM_27] = ScaleAndClip(
      (float)(bar.c - history[h + FEATURE_MAIN_LONG_PERIOD].c),
      FEATURE_IDX_MOM_27
   );
#endif
#ifdef FEATURE_IDX_USDX_PCT_CHANGE
   features[FEATURE_IDX_USDX_PCT_CHANGE] = ScaleAndClip(
      (float)((bar.usdx_bid / (prev.usdx_bid + 1e-10)) - 1.0),
      FEATURE_IDX_USDX_PCT_CHANGE
   );
#endif
#ifdef FEATURE_IDX_USDJPY_PCT_CHANGE
   features[FEATURE_IDX_USDJPY_PCT_CHANGE] = ScaleAndClip(
      (float)((bar.usdjpy_bid / (prev.usdjpy_bid + 1e-10)) - 1.0),
      FEATURE_IDX_USDJPY_PCT_CHANGE
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_9
   features[FEATURE_IDX_BOLLINGER_WIDTH_9] = ScaleAndClip(
      (float)((4.0 * StdClose(h, FEATURE_MAIN_SHORT_PERIOD)) / (MeanClose(h, FEATURE_MAIN_SHORT_PERIOD) + 1e-10)),
      FEATURE_IDX_BOLLINGER_WIDTH_9
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_18
   features[FEATURE_IDX_BOLLINGER_WIDTH_18] = ScaleAndClip(
      (float)((4.0 * StdClose(h, FEATURE_MAIN_MEDIUM_PERIOD)) / (MeanClose(h, FEATURE_MAIN_MEDIUM_PERIOD) + 1e-10)),
      FEATURE_IDX_BOLLINGER_WIDTH_18
   );
#endif
#ifdef FEATURE_IDX_BOLLINGER_WIDTH_27
   features[FEATURE_IDX_BOLLINGER_WIDTH_27] = ScaleAndClip(
      (float)((4.0 * StdClose(h, FEATURE_MAIN_LONG_PERIOD)) / (MeanClose(h, FEATURE_MAIN_LONG_PERIOD) + 1e-10)),
      FEATURE_IDX_BOLLINGER_WIDTH_27
   );
#endif
   {
      MqlDateTime parts;
      TimeToStruct((datetime)(bar.time_open_msc / 1000ULL), parts);
      double hour_angle = 2.0 * M_PI * parts.hour / 24.0;
      double minute_angle = 2.0 * M_PI * parts.min / 60.0;
      #ifdef FEATURE_IDX_HOUR_SIN
         features[FEATURE_IDX_HOUR_SIN] = ScaleAndClip((float)MathSin(hour_angle), FEATURE_IDX_HOUR_SIN);
      #endif
      #ifdef FEATURE_IDX_HOUR_COS
         features[FEATURE_IDX_HOUR_COS] = ScaleAndClip((float)MathCos(hour_angle), FEATURE_IDX_HOUR_COS);
      #endif
      #ifdef FEATURE_IDX_MINUTE_SIN
         features[FEATURE_IDX_MINUTE_SIN] = ScaleAndClip((float)MathSin(minute_angle), FEATURE_IDX_MINUTE_SIN);
      #endif
      #ifdef FEATURE_IDX_MINUTE_COS
         features[FEATURE_IDX_MINUTE_COS] = ScaleAndClip((float)MathCos(minute_angle), FEATURE_IDX_MINUTE_COS);
      #endif
      #ifdef FEATURE_IDX_DAY_OF_WEEK_SCALED
         features[FEATURE_IDX_DAY_OF_WEEK_SCALED] = ScaleAndClip(
            (float)(((parts.day_of_week + 1) % 7) / 6.0),
            FEATURE_IDX_DAY_OF_WEEK_SCALED
         );
      #endif
   }

   // ---------------------------------------------------------------------------
   // PAST_DIR_<N>_T  (bar-count lookback) and PAST_DIR_<N>_S (second lookback)
   // ---------------------------------------------------------------------------
   // Training emits  #define FEATURE_IDX_PAST_DIR_<N>_<U> <idx>  for each
   // enabled feature.  The handler computes tanh(log(close_now/close_then))
   // which maps any log-return to (-1, 1): near -1 = strong drop, 0 = flat,
   // near +1 = strong rise.  Blocks are ordered: _T first, then _S.
   //
   // ---- _T (bars) -----------------------------------------------------------
#ifdef FEATURE_IDX_PAST_DIR_1_T
   features[FEATURE_IDX_PAST_DIR_1_T] = ScaleAndClip((float)PastDirBarAt(h, 1), FEATURE_IDX_PAST_DIR_1_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_2_T
   features[FEATURE_IDX_PAST_DIR_2_T] = ScaleAndClip((float)PastDirBarAt(h, 2), FEATURE_IDX_PAST_DIR_2_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_3_T
   features[FEATURE_IDX_PAST_DIR_3_T] = ScaleAndClip((float)PastDirBarAt(h, 3), FEATURE_IDX_PAST_DIR_3_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_5_T
   features[FEATURE_IDX_PAST_DIR_5_T] = ScaleAndClip((float)PastDirBarAt(h, 5), FEATURE_IDX_PAST_DIR_5_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_9_T
   features[FEATURE_IDX_PAST_DIR_9_T] = ScaleAndClip((float)PastDirBarAt(h, 9), FEATURE_IDX_PAST_DIR_9_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_12_T
   features[FEATURE_IDX_PAST_DIR_12_T] = ScaleAndClip((float)PastDirBarAt(h, 12), FEATURE_IDX_PAST_DIR_12_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_18_T
   features[FEATURE_IDX_PAST_DIR_18_T] = ScaleAndClip((float)PastDirBarAt(h, 18), FEATURE_IDX_PAST_DIR_18_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_27_T
   features[FEATURE_IDX_PAST_DIR_27_T] = ScaleAndClip((float)PastDirBarAt(h, 27), FEATURE_IDX_PAST_DIR_27_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_36_T
   features[FEATURE_IDX_PAST_DIR_36_T] = ScaleAndClip((float)PastDirBarAt(h, 36), FEATURE_IDX_PAST_DIR_36_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_54_T
   features[FEATURE_IDX_PAST_DIR_54_T] = ScaleAndClip((float)PastDirBarAt(h, 54), FEATURE_IDX_PAST_DIR_54_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_72_T
   features[FEATURE_IDX_PAST_DIR_72_T] = ScaleAndClip((float)PastDirBarAt(h, 72), FEATURE_IDX_PAST_DIR_72_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_100_T
   features[FEATURE_IDX_PAST_DIR_100_T] = ScaleAndClip((float)PastDirBarAt(h, 100), FEATURE_IDX_PAST_DIR_100_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_144_T
   features[FEATURE_IDX_PAST_DIR_144_T] = ScaleAndClip((float)PastDirBarAt(h, 144), FEATURE_IDX_PAST_DIR_144_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_200_T
   features[FEATURE_IDX_PAST_DIR_200_T] = ScaleAndClip((float)PastDirBarAt(h, 200), FEATURE_IDX_PAST_DIR_200_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_288_T
   features[FEATURE_IDX_PAST_DIR_288_T] = ScaleAndClip((float)PastDirBarAt(h, 288), FEATURE_IDX_PAST_DIR_288_T);
#endif
#ifdef FEATURE_IDX_PAST_DIR_360_T
   features[FEATURE_IDX_PAST_DIR_360_T] = ScaleAndClip((float)PastDirBarAt(h, 360), FEATURE_IDX_PAST_DIR_360_T);
#endif
   //
   // ---- _S (seconds) --------------------------------------------------------
#ifdef FEATURE_IDX_PAST_DIR_60_S
   features[FEATURE_IDX_PAST_DIR_60_S] = ScaleAndClip((float)PastDirSecondsAt(h, 60), FEATURE_IDX_PAST_DIR_60_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_120_S
   features[FEATURE_IDX_PAST_DIR_120_S] = ScaleAndClip((float)PastDirSecondsAt(h, 120), FEATURE_IDX_PAST_DIR_120_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_300_S
   features[FEATURE_IDX_PAST_DIR_300_S] = ScaleAndClip((float)PastDirSecondsAt(h, 300), FEATURE_IDX_PAST_DIR_300_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_600_S
   features[FEATURE_IDX_PAST_DIR_600_S] = ScaleAndClip((float)PastDirSecondsAt(h, 600), FEATURE_IDX_PAST_DIR_600_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_900_S
   features[FEATURE_IDX_PAST_DIR_900_S] = ScaleAndClip((float)PastDirSecondsAt(h, 900), FEATURE_IDX_PAST_DIR_900_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_1800_S
   features[FEATURE_IDX_PAST_DIR_1800_S] = ScaleAndClip((float)PastDirSecondsAt(h, 1800), FEATURE_IDX_PAST_DIR_1800_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_3600_S
   features[FEATURE_IDX_PAST_DIR_3600_S] = ScaleAndClip((float)PastDirSecondsAt(h, 3600), FEATURE_IDX_PAST_DIR_3600_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_5400_S
   features[FEATURE_IDX_PAST_DIR_5400_S] = ScaleAndClip((float)PastDirSecondsAt(h, 5400), FEATURE_IDX_PAST_DIR_5400_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_7200_S
   features[FEATURE_IDX_PAST_DIR_7200_S] = ScaleAndClip((float)PastDirSecondsAt(h, 7200), FEATURE_IDX_PAST_DIR_7200_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_10800_S
   features[FEATURE_IDX_PAST_DIR_10800_S] = ScaleAndClip((float)PastDirSecondsAt(h, 10800), FEATURE_IDX_PAST_DIR_10800_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_14400_S
   features[FEATURE_IDX_PAST_DIR_14400_S] = ScaleAndClip((float)PastDirSecondsAt(h, 14400), FEATURE_IDX_PAST_DIR_14400_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_21600_S
   features[FEATURE_IDX_PAST_DIR_21600_S] = ScaleAndClip((float)PastDirSecondsAt(h, 21600), FEATURE_IDX_PAST_DIR_21600_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_43200_S
   features[FEATURE_IDX_PAST_DIR_43200_S] = ScaleAndClip((float)PastDirSecondsAt(h, 43200), FEATURE_IDX_PAST_DIR_43200_S);
#endif
#ifdef FEATURE_IDX_PAST_DIR_86400_S
   features[FEATURE_IDX_PAST_DIR_86400_S] = ScaleAndClip((float)PastDirSecondsAt(h, 86400), FEATURE_IDX_PAST_DIR_86400_S);
#endif
}
