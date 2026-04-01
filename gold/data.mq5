#property script_show_inputs
input int days_to_export = 60; // Export last N days
input string output_file = "gold_market_ticks.csv";
input string gold_symbol = "XAUUSD";
input string usdx_symbol = "$USDX";
input string usdjpy_symbol = "USDJPY";

void ExportSymbol(string symbol, long start_time, long end_time, int file_handle) {
   long op_start_time = (long)GetTickCount64();
   PrintFormat("[INFO] Exporting %s → %s", symbol, output_file);

   MqlTick ticks[];
   int copied = CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, start_time, end_time);

   if(copied <= 0) {
      PrintFormat("❌ %s: CopyTicksRange failed. Code: %d", symbol, GetLastError());
      return;
   }

   int progress_interval = (copied > 10) ? (copied / 10) : 1000;
   long process_start = (long)GetTickCount64();

   for(int i = 0; i < copied; i++) {
      if(ticks[i].bid <= 0.0) continue;
      FileWrite(file_handle, symbol, ticks[i].time_msc, ticks[i].bid, ticks[i].ask);
      if((i + 1) % progress_interval == 0) {
         int percent = (int)(((long)(i + 1) * 100) / copied);
         long elapsed_ms = (long)GetTickCount64() - process_start;
         long eta = (i > 0) ? (long)((double)elapsed_ms / (i + 1) * (copied - i - 1) / 1000.0) : 0;
         PrintFormat("📊 %s %d%% | Elapsed: %llds | ETA: %llds", symbol, percent, elapsed_ms / 1000, eta);
      }
   }

   long total_time = (long)GetTickCount64() - op_start_time;
   PrintFormat("✅ %s: Exported %d ticks in %.2f sec", symbol, copied, total_time / 1000.0);
}

void OnStart() {
   Print("[INFO] Multi-Symbol Tick Exporter Starting...");

   int h = FileOpen(output_file, FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) {
      PrintFormat("❌ Cannot open file: %s", output_file);
      return;
   }

   FileWrite(h, "symbol", "time_msc", "bid", "ask");

   long end_time = TimeCurrent() * 1000LL;
   long start_time = end_time - (long)days_to_export * 24LL * 3600LL * 1000LL;

   ExportSymbol(gold_symbol, start_time, end_time, h);
   ExportSymbol(usdx_symbol, start_time, end_time, h);
   ExportSymbol(usdjpy_symbol, start_time, end_time, h);

   FileClose(h);

   Print("[INFO] All exports complete.");
}
