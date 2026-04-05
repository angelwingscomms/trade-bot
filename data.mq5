#property script_show_inputs

#include "shared_config.mqh"

input int days_to_export = 60;
input string output_file = "market_ticks.csv";
input string symbol_to_export = SYMBOL;

void OnStart() {
   Print("===== DATA EXPORT SCRIPT STARTED =====");
   PrintFormat("[STEP 1] Script parameters configured:");
   PrintFormat("  - days_to_export = %d", days_to_export);
   PrintFormat("  - output_file = %s", output_file);
   PrintFormat("  - symbol_to_export = %s", symbol_to_export);
   Print("");
   
   Print("[STEP 2] Attempting to open output file for writing...");
   int file_handle = FileOpen(output_file, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
   if(file_handle == INVALID_HANDLE) {
      int error_code = GetLastError();
      PrintFormat("[ERROR] Cannot open %s. Error code: %d", output_file, error_code);
      Print("[STEP 2] FAILED - File could not be opened");
      Print("===== DATA EXPORT SCRIPT TERMINATED =====");
      return;
   }
   PrintFormat("[STEP 2] SUCCESS - File handle created: %d", file_handle);
   Print("");

   Print("[STEP 3] Writing CSV header row...");
   FileWrite(file_handle, "time_msc", "bid", "ask");
   Print("[STEP 3] SUCCESS - Header written: time_msc, bid, ask");
   Print("");

   Print("[STEP 4] Calculating time range for tick export...");
   long current_time_sec = TimeCurrent();
   PrintFormat("  - Current time (seconds): %lld", current_time_sec);
   long end_time = current_time_sec * 1000LL;
   PrintFormat("  - End time (milliseconds): %lld", end_time);
   long time_delta_ms = (long)days_to_export * 24LL * 3600LL * 1000LL;
   PrintFormat("  - Time delta (%d days in milliseconds): %lld", days_to_export, time_delta_ms);
   long start_time = end_time - time_delta_ms;
   PrintFormat("  - Start time (milliseconds): %lld", start_time);
   Print("[STEP 4] SUCCESS - Time range calculated");
   Print("");

   Print("[STEP 5] Preparing array for CopyTicksRange...");
   MqlTick ticks[];
   Print("[STEP 5] SUCCESS - MqlTick array declared");
   Print("");
   
   Print("[STEP 6] Calling CopyTicksRange to fetch tick data...");
   PrintFormat("  - Symbol: %s", symbol_to_export);
   PrintFormat("  - Mode: COPY_TICKS_ALL (value=0)");
   PrintFormat("  - Start time: %lld ms", start_time);
   PrintFormat("  - End time: %lld ms", end_time);
   int copied = CopyTicksRange(symbol_to_export, ticks, COPY_TICKS_ALL, start_time, end_time);
   PrintFormat("[STEP 6] CopyTicksRange returned: %d ticks", copied);
   
   if(copied <= 0) {
      int error_code = GetLastError();
      PrintFormat("[ERROR] CopyTicksRange failed for %s. Error code: %d", symbol_to_export, error_code);
      PrintFormat("[STEP 6] FAILED - No ticks were copied (returned %d)", copied);
      Print("[STEP 7] SKIPPED - Cleaning up due to error");
      FileClose(file_handle);
      PrintFormat("[STEP 7] File closed (handle %d)", file_handle);
      Print("===== DATA EXPORT SCRIPT TERMINATED WITH ERROR =====");
      return;
   }
   PrintFormat("[STEP 6] SUCCESS - %d ticks retrieved", copied);
   Print("");

   Print("[STEP 7] Writing tick data to CSV file...");
   int valid_ticks_written = 0;
   int ticks_skipped = 0;
   for(int i = 0; i < copied; i++) {
      if(i % 1000 == 0) {
         PrintFormat("  - Processing tick %d/%d...", i, copied);
      }
      if(ticks[i].bid > 0.0) {
         FileWrite(file_handle, ticks[i].time_msc, ticks[i].bid, ticks[i].ask);
         valid_ticks_written++;
      } else {
         ticks_skipped++;
      }
   }
   PrintFormat("[STEP 7] SUCCESS - Wrote %d valid ticks to file", valid_ticks_written);
   PrintFormat("  - Ticks skipped (bid <= 0): %d", ticks_skipped);
   Print("");

   Print("[STEP 8] Closing output file...");
   FileClose(file_handle);
   PrintFormat("[STEP 8] SUCCESS - File closed (handle %d)", file_handle);
   Print("");
   
   Print("===== DATA EXPORT SCRIPT COMPLETED SUCCESSFULLY =====");
   PrintFormat("Total ticks processed: %d", copied);
   PrintFormat("Valid ticks written: %d", valid_ticks_written);
   PrintFormat("Output file: %s", output_file);
}
