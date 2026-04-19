// PastDirAt.mqh
// Returns tanh(log(close_now / close_n_bars_ago)), clipped to (-1, 1).
// Used by PAST_DIR_*_T (bar-count lookback) and PAST_DIR_*_S (second-count lookback).
// history[0] is the most recent closed bar; history[n] is n bars ago.

double PastDirBarAt(int h, int n_bars) {
   int idx = h + n_bars;
   if(idx >= HISTORY_SIZE || !history[idx].valid) {
      return 0.0;
   }
   double prev_close = history[idx].c;
   if(prev_close <= 0.0) return 0.0;
   double log_ret = MathLog(history[h].c / prev_close);
   // tanh via (e^2x - 1) / (e^2x + 1) for numerical safety
   double e2x = MathExp(2.0 * log_ret);
   return (e2x - 1.0) / (e2x + 1.0);
}

double PastDirSecondsAt(int h, int n_seconds) {
   if(n_seconds <= 0) return 0.0;
   // Walk back from bar h until accumulated bar time >= n_seconds.
   // Each bar's duration is (time_close_msc - time_open_msc) / 1000.0 seconds.
   double elapsed = 0.0;
   for(int i = h; i < HISTORY_SIZE - 1; i++) {
      if(!history[i].valid) return 0.0;
      double bar_dur = (history[i].time_close_msc - history[i].time_open_msc) / 1000.0;
      if(bar_dur < 0.0) bar_dur = 0.0;
      elapsed += bar_dur;
      if(elapsed >= (double)n_seconds) {
         return PastDirBarAt(h, i - h + 1);
      }
   }
   // Reached end of history: use oldest valid bar
   return PastDirBarAt(h, HISTORY_SIZE - 1 - h);
}
