#property script_show_inputs // Show settings window
input int ticks_to_export = 2160000; // Total ticks (~5 days of Gold)
input string USDX_Symbol = "$USDX"; // Name of USD Index
input string USDJPY_Symbol = "USDJPY"; // Name of USDJPY

// FLAW 4.4 FIX: Optimized tick data exporting with StringFormat and Two-Pointer Merge

//+------------------------------------------------------------------+
//| Logging helper functions                                          |
//+------------------------------------------------------------------+
void LogInfo(string message) {
   Print("[INFO] ", message);
}

void LogSuccess(string message) {
   Print("✅ [SUCCESS] ", message);
}

void LogWarning(string message) {
   Print("⚠️ [WARNING] ", message);
}

void LogError(string message) {
   Print("❌ [ERROR] ", message);
}

void LogProgress(string stage, int current, int total, string extra = "") {
   int percent = (int)((double)current / total * 100);
   Print("📊 [PROGRESS] ", stage, ": ", current, "/", total, " (", percent, "%)", extra);
}

void LogSeparator() {
   Print("═══════════════════════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| Main script function                                              |
//+------------------------------------------------------------------+
void OnStart() {
   ulong start_time = GetTickCount64(); // Script start timestamp
   LogSeparator();
   LogInfo("ACHILLES TICK DATA EXPORTER - Starting execution");
   LogSeparator();
   LogInfo(StringFormat("Parameters: ticks_to_export=%d, USDX='%s', USDJPY='%s'", 
                        ticks_to_export, USDX_Symbol, USDJPY_Symbol));
   LogInfo(StringFormat("Main symbol: %s", _Symbol));
   
   MqlTick ticks[], usdx_ticks[], usdjpy_ticks[]; // Arrays to hold tick data
   
   // === SYMBOL SELECTION PHASE ===
   LogInfo("Phase 1: Symbol Selection");
   LogInfo(StringFormat("  Attempting to select USDX symbol: '%s'", USDX_Symbol));
   bool usdx_available = SymbolSelect(USDX_Symbol, true);
   if(usdx_available) {
      LogSuccess(StringFormat("  USDX symbol '%s' selected successfully", USDX_Symbol));
   } else {
      LogWarning(StringFormat("  USDX symbol '%s' NOT available - will use placeholder (0.0)", USDX_Symbol));
   }
   
   LogInfo(StringFormat("  Attempting to select USDJPY symbol: '%s'", USDJPY_Symbol));
   bool usdjpy_available = SymbolSelect(USDJPY_Symbol, true);
   if(usdjpy_available) {
      LogSuccess(StringFormat("  USDJPY symbol '%s' selected successfully", USDJPY_Symbol));
   } else {
      LogWarning(StringFormat("  USDJPY symbol '%s' NOT available - will use placeholder (0.0)", USDJPY_Symbol));
   }
   
   // === TICK DATA COPYING PHASE ===
   LogSeparator();
   LogInfo("Phase 2: Tick Data Acquisition");
   LogInfo(StringFormat("  Copying %d ticks for main symbol '%s'...", ticks_to_export, _Symbol));
   
   ulong copy_start = GetTickCount64();
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, ticks_to_export);
   ulong copy_time = GetTickCount64() - copy_start;
   
   if(copied <= 0) {
      LogError(StringFormat("  Failed to copy ticks for '%s'! Error code: %d", _Symbol, GetLastError()));
      LogError("  Script terminated - no data to export");
      return;
   }
   LogSuccess(StringFormat("  Copied %d ticks for '%s' in %llu ms", copied, _Symbol, copy_time));
   
   // Log tick data time range
   if(copied > 0) {
      datetime first_time = (datetime)(ticks[0].time_msc / 1000);
      datetime last_time = (datetime)(ticks[copied-1].time_msc / 1000);
      LogInfo(StringFormat("  Tick time range: %s to %s", 
                           TimeToString(first_time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
                           TimeToString(last_time, TIME_DATE|TIME_MINUTES|TIME_SECONDS)));
   }
   
   // Get tick data for auxiliary symbols if available
   int usdx_copied = 0, usdjpy_copied = 0;
   
   if(usdx_available) {
      LogInfo(StringFormat("  Copying %d ticks for USDX '%s'...", ticks_to_export, USDX_Symbol));
      copy_start = GetTickCount64();
      usdx_copied = CopyTicks(USDX_Symbol, usdx_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      copy_time = GetTickCount64() - copy_start;
      
      if(usdx_copied <= 0) {
         LogWarning(StringFormat("  USDX ticks not available (error: %d), using placeholder", GetLastError()));
         usdx_available = false;
      } else {
         LogSuccess(StringFormat("  Copied %d USDX ticks in %llu ms", usdx_copied, copy_time));
      }
   }
   
   if(usdjpy_available) {
      LogInfo(StringFormat("  Copying %d ticks for USDJPY '%s'...", ticks_to_export, USDJPY_Symbol));
      copy_start = GetTickCount64();
      usdjpy_copied = CopyTicks(USDJPY_Symbol, usdjpy_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      copy_time = GetTickCount64() - copy_start;
      
      if(usdjpy_copied <= 0) {
         LogWarning(StringFormat("  USDJPY ticks not available (error: %d), using placeholder", GetLastError()));
         usdjpy_available = false;
      } else {
         LogSuccess(StringFormat("  Copied %d USDJPY ticks in %llu ms", usdjpy_copied, copy_time));
      }
   }
   
    // === FILE CREATION PHASE ===
    LogSeparator();
    LogInfo("Phase 3: File Creation");
    LogInfo("  Creating output file: fast/achilles_ticks.csv");
    LogInfo("  (MQL5 sandbox restricts to MQL5\\Files, run move_ticks.py after export)");
    
    int h = FileOpen("fast/achilles_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) {
      LogError(StringFormat("  Failed to create file! Error code: %d", GetLastError()));
      LogError("  Script terminated - cannot write data");
      return;
   }
   LogSuccess("  File opened successfully");
   
   FileWrite(h, "time_msc,bid,ask,usdx,usdjpy"); // Write CSV header
   LogInfo("  CSV header written: time_msc,bid,ask,usdx,usdjpy");
   
   // === DATA PROCESSING PHASE ===
   LogSeparator();
   LogInfo("Phase 4: Data Processing & Export");
   LogInfo(StringFormat("  Processing %d ticks with Two-Pointer Merge algorithm...", copied));
   LogInfo("  Algorithm complexity: O(N) - linear time");
   
   // FLAW 4.4 FIX: Two-Pointer Merge algorithm for O(N) timestamp alignment
   int usdx_idx = 0, usdjpy_idx = 0; // Indices for auxiliary tick arrays
   double usdx_bid = 0.0, usdjpy_bid = 0.0; // Current matched prices
   
   int usdx_matches = 0, usdjpy_matches = 0; // Count of successful matches
   int progress_interval = copied / 10; // Report progress every 10%
   if(progress_interval < 1000) progress_interval = 1000; // Minimum 1000 ticks between reports
   
   ulong process_start = GetTickCount64();
   
   for(int i = 0; i < copied; i++) {
      ulong t = ticks[i].time_msc; // Current tick timestamp
      
      // FLAW 4.4 FIX: Two-Pointer Merge for USDX
      if(usdx_available && usdx_copied > 0) {
         int prev_idx = usdx_idx;
         while(usdx_idx < usdx_copied - 1 && usdx_ticks[usdx_idx + 1].time_msc <= t) {
            usdx_idx++;
         }
         if(usdx_idx != prev_idx) usdx_matches++;
         usdx_bid = usdx_ticks[usdx_idx].bid;
      }
      
      // FLAW 4.4 FIX: Two-Pointer Merge for USDJPY
      if(usdjpy_available && usdjpy_copied > 0) {
         int prev_idx = usdjpy_idx;
         while(usdjpy_idx < usdjpy_copied - 1 && usdjpy_ticks[usdjpy_idx + 1].time_msc <= t) {
            usdjpy_idx++;
         }
         if(usdjpy_idx != prev_idx) usdjpy_matches++;
         usdjpy_bid = usdjpy_ticks[usdjpy_idx].bid;
      }
      
      // Use StringFormat for efficient string building
      string row = StringFormat("%lld,%.5f,%.5f,%.5f,%.5f",
                                ticks[i].time_msc,
                                ticks[i].bid,
                                ticks[i].ask,
                                usdx_bid,
                                usdjpy_bid);
      FileWrite(h, row);
      
      // Progress reporting
      if(progress_interval > 0 && (i + 1) % progress_interval == 0) {
         int percent = (int)((double)(i + 1) / copied * 100);
         ulong elapsed = GetTickCount64() - process_start;
         int estimated_total = (int)((double)elapsed / (i + 1) * copied / 1000);
         int estimated_remaining = (int)((double)elapsed / (i + 1) * (copied - i - 1) / 1000);
         LogProgress("Export", i + 1, copied, 
                     StringFormat(" | Elapsed: %ds | ETA: %ds", 
                                  (int)(elapsed / 1000), estimated_remaining));
      }
   }
   
   ulong process_time = GetTickCount64() - process_start;
   
   // === FILE FINALIZATION PHASE ===
   LogSeparator();
   LogInfo("Phase 5: File Finalization");
   FileClose(h);
   LogSuccess("  File closed successfully");
   
   // Calculate file size estimate (approximate)
   long file_size_estimate = copied * 60L; // ~60 bytes per row estimate
   LogInfo(StringFormat("  Estimated file size: ~%.2f MB", (double)file_size_estimate / 1024 / 1024));
   
   // === FINAL SUMMARY ===
   LogSeparator();
   LogInfo("EXECUTION SUMMARY");
   LogSeparator();
   
   ulong total_time = GetTickCount64() - start_time;
   
    LogSuccess(StringFormat("Exported %d ticks to fast/achilles_ticks.csv", copied));
    LogInfo("  Run 'python fast/move_ticks.py' to move file to project directory");
    LogInfo(StringFormat("  Main symbol (%s): %d ticks", _Symbol, copied));
   
   if(usdx_available) {
      LogInfo(StringFormat("  USDX (%s): %d ticks loaded, %d timestamp matches", 
                           USDX_Symbol, usdx_copied, usdx_matches));
   } else {
      LogInfo("  USDX: Not available (used placeholder 0.0)");
   }
   
   if(usdjpy_available) {
      LogInfo(StringFormat("  USDJPY (%s): %d ticks loaded, %d timestamp matches", 
                           USDJPY_Symbol, usdjpy_copied, usdjpy_matches));
   } else {
      LogInfo("  USDJPY: Not available (used placeholder 0.0)");
   }
   
   LogInfo(StringFormat("Processing time: %llu ms (%.2f seconds)", process_time, (double)process_time / 1000));
   LogInfo(StringFormat("Throughput: %.0f ticks/second", (double)copied / (process_time / 1000.0)));
   LogInfo(StringFormat("Total script execution time: %llu ms (%.2f seconds)", total_time, (double)total_time / 1000));
   
   LogSeparator();
   LogSuccess("SCRIPT COMPLETED SUCCESSFULLY");
   LogSeparator();
}
