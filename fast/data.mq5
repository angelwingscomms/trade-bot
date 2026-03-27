#property script_show_inputs // Show settings window
input int ticks_to_export = 2160000; // Total ticks (~5 days of Gold)
input string USDX_Symbol = "$USDX"; // Name of USD Index
input string USDJPY_Symbol = "USDJPY"; // Name of USDJPY

// FLAW 4.4 FIX: Optimized tick data exporting with StringFormat and Two-Pointer Merge

void OnStart() { // Main script function
   MqlTick ticks[], usdx_ticks[], usdjpy_ticks[]; // Arrays to hold tick data
   
   // Enable symbols and check if they're available
   bool usdx_available = SymbolSelect(USDX_Symbol, true);
   bool usdjpy_available = SymbolSelect(USDJPY_Symbol, true);
   
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, ticks_to_export); // Get main symbol ticks
   if(copied <= 0) { Print("❌ Failed to copy ticks"); return; } // Error check
   
   // Get tick data for auxiliary symbols if available
   int usdx_copied = 0, usdjpy_copied = 0;
   if(usdx_available) {
      usdx_copied = CopyTicks(USDX_Symbol, usdx_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      if(usdx_copied <= 0) { Print("⚠️ USDX ticks not available, using placeholder"); usdx_available = false; }
   }
   if(usdjpy_available) {
      usdjpy_copied = CopyTicks(USDJPY_Symbol, usdjpy_ticks, COPY_TICKS_ALL, 0, ticks_to_export);
      if(usdjpy_copied <= 0) { Print("⚠️ USDJPY ticks not available, using placeholder"); usdjpy_available = false; }
   }
   
   int h = FileOpen("achilles_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ","); // Create file
   FileWrite(h, "time_msc,bid,ask,usdx,usdjpy"); // Write CSV header
   
   // FLAW 4.4 FIX: Two-Pointer Merge algorithm for O(N) timestamp alignment
   // Each pointer only moves forward, eliminating nested while loops
   int usdx_idx = 0, usdjpy_idx = 0; // Indices for auxiliary tick arrays
   double usdx_bid = 0.0, usdjpy_bid = 0.0; // Current matched prices
   
   for(int i = 0; i < copied; i++) { // Loop through every tick
      ulong t = ticks[i].time_msc; // Current tick timestamp
      
      // FLAW 4.4 FIX: Two-Pointer Merge for USDX - advance pointer only when timestamp <= current
      // This is O(N) instead of O(N^2) - each array is traversed exactly once
      if(usdx_available && usdx_copied > 0) {
         // Advance usdx_idx while the next tick's time is <= current tick time
         while(usdx_idx < usdx_copied - 1 && usdx_ticks[usdx_idx + 1].time_msc <= t) {
            usdx_idx++;
         }
         usdx_bid = usdx_ticks[usdx_idx].bid;
      }
      
      // FLAW 4.4 FIX: Two-Pointer Merge for USDJPY - same O(N) algorithm
      if(usdjpy_available && usdjpy_copied > 0) {
         // Advance usdjpy_idx while the next tick's time is <= current tick time
         while(usdjpy_idx < usdjpy_copied - 1 && usdjpy_ticks[usdjpy_idx + 1].time_msc <= t) {
            usdjpy_idx++;
         }
         usdjpy_bid = usdjpy_ticks[usdjpy_idx].bid;
      }
      
      // FLAW 4.4 FIX: Use StringFormat() instead of string concatenation with +
      // StringFormat is significantly faster and avoids memory fragmentation
      string row = StringFormat("%lld,%.5f,%.5f,%.5f,%.5f",
                                ticks[i].time_msc,
                                ticks[i].bid,
                                ticks[i].ask,
                                usdx_bid,
                                usdjpy_bid);
      FileWrite(h, row);
   }
   FileClose(h); // Close file
   Print("✅ Exported ", copied, " ticks to MQL5\\Files\\achilles_ticks.csv"); // Success message
   if(usdx_available) Print("   USDX: ", usdx_copied, " ticks matched");
   if(usdjpy_available) Print("   USDJPY: ", usdjpy_copied, " ticks matched");
}
