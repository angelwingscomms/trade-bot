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
