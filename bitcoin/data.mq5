#property script_show_inputs
input int ticks_to_export = 2160000;
input int days_lookback   = 180;     
input int chunk_size      = 100000;

void OnStart() {
   Print("[INFO] Initializing Absolute-Chronological Tick Export...");
   int h = FileOpen("bitcoin_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ",");
   if(h == INVALID_HANDLE) return;
   
   FileWrite(h, "time_msc", "bid", "ask", "vol"); 
   MqlTick ticks[];
   int total_copied = 0;
   
   ulong anchor_msc = ((ulong)TimeCurrent() - ((ulong)days_lookback * 86400ull)) * 1000ull;
   ulong last_time  = anchor_msc;
   
   while(total_copied < ticks_to_export) {
      int to_copy = MathMin(chunk_size, ticks_to_export - total_copied);
      int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, last_time, to_copy);
      
      if(copied <= 0) break;
      last_time = ticks[copied-1].time_msc + 1; 
      
      int valid_in_chunk = 0;
      for(int i = 0; i < copied; i++) {
         if(ticks[i].bid <= 0.0 || ticks[i].ask < ticks[i].bid) continue;
         double v = (ticks[i].volume > 0) ? (double)ticks[i].volume : 0.01;
         FileWrite(h, ticks[i].time_msc, ticks[i].bid, ticks[i].ask, v);
         valid_in_chunk++;
      }
      total_copied += valid_in_chunk;
      if(last_time >= (ulong)TimeCurrent() * 1000ull) break;
   }
   FileClose(h);
   PrintFormat("✅ Exported %d ticks.", total_copied);
}