#property script_show_inputs

#include "../../config/active.mqh"

input int days_to_export = 60;
input string output_file = "market_ticks.csv";
input string symbol_to_export = SYMBOL;
input string USDX_Symbol = "$USDX";
input string USDJPY_Symbol = "USDJPY";

void OnStart() {
   Print("===== GOLD DATA EXPORT SCRIPT STARTED =====");
   PrintFormat("[STEP 1] days_to_export=%d output_file=%s symbol=%s", days_to_export, output_file, symbol_to_export);

   int file_handle = FileOpen(output_file, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
   if(file_handle == INVALID_HANDLE) {
      PrintFormat("[ERROR] Cannot open %s. Error code: %d", output_file, GetLastError());
      Print("===== DATA EXPORT TERMINATED =====");
      return;
   }
   FileWrite(file_handle, "time_msc", "bid", "ask", "usdx_bid", "usdjpy_bid");

   long current_time_sec = TimeCurrent();
   long end_time = current_time_sec * 1000LL;
   long time_delta_ms = (long)days_to_export * 24LL * 3600LL * 1000LL;
   long start_time = end_time - time_delta_ms;

   MqlTick ticks[], usdx_ticks[], usdjpy_ticks[];
   int copied = CopyTicksRange(symbol_to_export, ticks, COPY_TICKS_ALL, start_time, end_time);
   if(copied <= 0) {
      PrintFormat("[ERROR] CopyTicksRange failed for %s. Error code: %d", symbol_to_export, GetLastError());
      FileClose(file_handle);
      return;
   }

   bool usdx_available = SymbolSelect(USDX_Symbol, true);
   bool usdjpy_available = SymbolSelect(USDJPY_Symbol, true);

   int usdx_copied = 0;
   int usdjpy_copied = 0;
   if(usdx_available) {
      usdx_copied = CopyTicksRange(USDX_Symbol, usdx_ticks, COPY_TICKS_ALL, start_time, end_time);
      if(usdx_copied <= 0) { usdx_available = false; }
   }
   if(usdjpy_available) {
      usdjpy_copied = CopyTicksRange(USDJPY_Symbol, usdjpy_ticks, COPY_TICKS_ALL, start_time, end_time);
      if(usdjpy_copied <= 0) { usdjpy_available = false; }
   }

   int usdx_idx = 0;
   int usdjpy_idx = 0;
   double last_usdx = 0.0;
   double last_usdjpy = 0.0;

   for(int i = 0; i < copied; i++) {
      ulong t = ticks[i].time_msc;
      if(ticks[i].bid <= 0.0) {
         continue;
      }

      double usdx_bid = last_usdx;
      if(usdx_available && usdx_copied > 0) {
         while(usdx_idx < usdx_copied - 1 && usdx_ticks[usdx_idx].time_msc <= t) usdx_idx++;
         if(usdx_idx > 0 && usdx_ticks[usdx_idx].time_msc > t) usdx_idx--;
         usdx_bid = usdx_ticks[usdx_idx].bid;
         last_usdx = usdx_bid;
      }

      double usdjpy_bid = last_usdjpy;
      if(usdjpy_available && usdjpy_copied > 0) {
         while(usdjpy_idx < usdjpy_copied - 1 && usdjpy_ticks[usdjpy_idx].time_msc <= t) usdjpy_idx++;
         if(usdjpy_idx > 0 && usdjpy_ticks[usdjpy_idx].time_msc > t) usdjpy_idx--;
         usdjpy_bid = usdjpy_ticks[usdjpy_idx].bid;
         last_usdjpy = usdjpy_bid;
      }

      FileWrite(
         file_handle,
         ticks[i].time_msc,
         ticks[i].bid,
         ticks[i].ask,
         usdx_bid,
         usdjpy_bid
      );
   }

   FileClose(file_handle);
   PrintFormat("===== EXPORT COMPLETE (%d ticks) =====", copied);
   if(usdx_available) PrintFormat("   USDX ticks: %d", usdx_copied);
   if(usdjpy_available) PrintFormat("   USDJPY ticks: %d", usdjpy_copied);
}
