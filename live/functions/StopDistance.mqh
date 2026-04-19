double StopDistance() {
   if(R) {
      return FIXED_SL;
   }
   return history[0].atr_trade * SL_MULTIPLIER;
}
