#property script_show_inputs
input int ticks_to_export = 2160000; 

void OnStart() {
   long start_time = (long)GetTickCount64(); // Cast to long for signed arithmetic
   Print("[INFO] BITCOIN TICK DATA EXPORTER - Starting");

   MqlTick ticks[];
   // CopyTicks returns int, so we stay in signed integer space for the count
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, (uint)ticks_to_export);
   
   if(copied <= 0) {
      PrintFormat("❌ Error: CopyTicks failed. Code: %d", GetLastError());
      return;
   }

   int h = FileOpen("achilles_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) return;

   FileWrite(h, "time_msc", "bid", "ask");

   // Define progress intervals as int to match the loop index 'i'
   int progress_interval = (copied > 10) ? (copied / 10) : 1000;
   long process_start = (long)GetTickCount64();

   for(int i = 0; i < copied; i++) {
      // 1. Data Integrity Check
      if(ticks[i].bid <= 0.0) continue;

      // 2. High-Performance Write (No StringFormat overhead)
      FileWrite(h, ticks[i].time_msc, ticks[i].bid, ticks[i].ask);

      // 3. Warning-Free Progress Logic
      // We use 'i' (int) and progress_interval (int) - No sign mismatch
      if((i + 1) % progress_interval == 0) {
         int percent = (int)(((long)(i + 1) * 100) / copied);
         long elapsed_ms = (long)GetTickCount64() - process_start;
         
         // Calculate ETA using long to prevent overflow
         long eta = (i > 0) ? (long)((double)elapsed_ms / (i + 1) * (copied - i - 1) / 1000.0) : 0;
         
         PrintFormat("📊 Progress: %d%% | Elapsed: %llds | ETA: %llds", 
                     percent, elapsed_ms / 1000, eta);
      }
   }

   FileClose(h);
   long total_time = (long)GetTickCount64() - start_time;
   PrintFormat("✅ Export Complete. Total Time: %.2f sec", total_time / 1000.0);
}