double TargetDistance() {
   if(R) {
      return FIXED_TP;
   }
   return history[0].atr_trade * TP_MULTIPLIER;
}
