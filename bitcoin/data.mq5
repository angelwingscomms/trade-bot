#property script_show_inputs
input int ticks_to_export = 2160000;
input int chunk_size      = 100000;

void OnStart() {
   Print("[INFO] Initializing High-Throughput Tick Export Engine...");
   
   // Clear errors before starting
   ResetLastError(); 
   
   int h = FileOpen("fast/bitcoin_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) { 
      PrintFormat("❌ FATAL I/O ERROR: Cannot open file. MQL5 Error Code: %d", GetLastError()); 
      return; 
   }
   
   FileWrite(h, "time_msc", "bid", "ask", "vol"); 
   
   MqlTick ticks[];
   int total_copied = 0;
   ulong last_time  = 0;
   ulong start_time = GetTickCount64();
   
   Print("[INFO] Handshake complete. Commencing synchronous sequence. MT5 may download missing history. Stand by...");
   
   while(total_copied < ticks_to_export) {
      int to_copy = MathMin(chunk_size, ticks_to_export - total_copied);
      
      // Measure precise network/disk retrieval time
      ulong fetch_start = GetTickCount64();
      int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, last_time, to_copy);
      ulong fetch_time = GetTickCount64() - fetch_start;
      
      if(copied <= 0) {
         int err = GetLastError();
         PrintFormat("⚠️ WARNING: Tick stream exhausted or network failure. Copied: %d | Error Code: %d", copied, err);
         if (err == 4401) Print("💡 Hint: Error 4401 means History Not Found. Broker server lacks requested depth.");
         break;
      }
      
      // Advance the temporal pointer to prevent infinite loops on the exact same millisecond
      last_time = ticks[copied-1].time_msc + 1; 
      
      int valid_ticks = 0;
      
      // SERIALIZATION PHASE: Direct OS-Level buffering. Zero O(N^2) heap allocations.
      for(int i = 0; i < copied; i++) {
         // Sanitize structural anomalies (negative spreads / zeroed bids)
         if(ticks[i].bid <= 0.0 || ticks[i].ask < ticks[i].bid) continue;
         
         double v = (ticks[i].volume > 0) ? (double)ticks[i].volume : 1.0;
         
         // FileWrite automatically maps array arguments to CSV columns based on FileOpen delimiter
         FileWrite(h, ticks[i].time_msc, ticks[i].bid, ticks[i].ask, v);
         valid_ticks++;
      }
      
      total_copied += copied;
      
      // Telemetry Output
      double progress = ((double)total_copied / ticks_to_export) * 100.0;
      PrintFormat("[STREAM] %.2f%% | Fetched: %d ticks in %llu ms | Validated: %d | Total: %d / %d", 
                  progress, copied, fetch_time, valid_ticks, total_copied, ticks_to_export);
                  
      // CRITICAL: Yield thread to MT5 GUI to prevent "Not Responding" freeze lock
      Sleep(10); 
   }
   
   FileClose(h);
   ulong elapsed = GetTickCount64() - start_time;
   PrintFormat("✅ TELEMETRY COMPLETE. Serialized %d ticks. Total Execution Time: %.2f seconds.", total_copied, elapsed / 1000.0);
}