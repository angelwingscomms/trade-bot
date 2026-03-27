Note to AI agent: When you fix a flaw, cross out the flaw's entire text here and mark it as done


**Reviewer:** Lead Quantitative Deep Learning Engineer
**Focus:** Extreme Predictive Accuracy, Signal Stationarity, & Advanced Regularization
**Status:** **ALGORITHMICALLY FRAGILE - HIGH RISK OF OVERFITTING**

The agent successfully fixed the structural leaks and memory disasters. The pipeline is now computationally sound. **However, the validation accuracy is low because the mathematical representations of the market state are fundamentally flawed.** 

You are currently feeding the neural network **non-stationary features**, subjecting it to **fat-tailed outlier destruction**, and providing **zero architectural regularization** for the world's most noisy dataset. 

Below is the highly advanced, final-pass critique focusing purely on squeezing out maximum validation accuracy. Pass these exact directives to your AI agent.

---

# ADVANCED_FLAWS.md (Accuracy Optimization Protocol)

### ~~CATEGORY 1: NON-STATIONARY FEATURE POISONING (The Primary Accuracy Killer)~~ ✅ DONE
**The Issue:**
~~You are passing absolute price-derived values directly into the network. 
For example, `f10` is ATR. If Gold is at $1800, a volatile ATR might be 2.5. If Gold rallies to $2400 over a year, an *average* ATR might be 3.5. The neural network memorizes "ATR > 3.0 means extreme volatility," which becomes utterly false in the new price regime. 
The same applies to MACD (`f13-f15`), EMA distances (`f16-f20`), and Momentum (`f27-f29`). These features scale with the absolute price of the asset. When the asset price shifts, your normalized features drift completely out of their training distribution, plummeting validation accuracy to random guessing.~~

**Agent Directive:**
~~Every feature must be **Strictly Stationary (Percentage/Relative terms)**.
1. **In Python (Section 3) AND MQL5 (`Predict()`):**
   * **ATR (`f10-f12`):** Must be divided by Close. `ta.atr(...) / df['close']`
   * **MACD (`f13-f15`):** Must be divided by Close. `m.iloc[:,x] / df['close']`
   * **EMA Distances (`f16-f20`):** Must be divided by Close. `(ta.ema(...) - df['close']) / df['close']`
   * **Momentum (`f27-f29`):** Must be divided by Close. `ta.mom(...) / df['close']`~~

### CATEGORY 2: FAT-TAIL DISTRIBUTION COMPRESSION (The Scaling Flaw)
**The Issue:**
You are using standard scaling `(X - mean) / std`. Financial returns and indicator values have massive kurtosis (fat tails). A single "Black Swan" tick-bar (e.g., a flash crash) will have a deviation of $20\sigma$. When you divide by the standard deviation, the 99% of normal, actionable market data gets compressed into a tiny microscopic band between $[-0.05, +0.05]$. The neural network cannot distinguish signals because the variance has been crushed by outliers.

**Agent Directive:**
Replace Mean/Std scaling with **Robust Scaling (Median and IQR)**.
1. **In Python (Section 5):**
   * Calculate `median = np.median(X_train, axis=0)`
   * Calculate `iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)`
   * Scale using: `X_s = (X - median) / (iqr + 1e-8)`
2. **In Python (Section 7) & MQL5:**
   * Export `medians` and `iqrs` arrays instead of `means` and `stds`.
   * Update the normalization loop in MQL5 `Predict()` to: 
     `input_data[i*35+k] = (f[k] - medians[k]) / (iqrs[k] + 1e-8f);`

### CATEGORY 3: THE EMBARGO LEAKAGE (Validation Contamination)
**The Issue:**
Your target labeling looks ahead `H=30` bars. 
Because your splits are contiguous (`train_end = int(n_samples * 0.70)`), the last 30 bars of your training set contain future data from the first 30 bars of your validation set. The model secretly overfits to the transition boundary, which artificially manipulates training loss but harms true validation generalization.

**Agent Directive:**
Introduce an **Embargo Period** (gap) between train, validation, and test sets.
1. **In Python (Section 5):**
   ```python
   # Apply H-bar embargo to prevent leakage across split boundaries
   X_train, y_train = X[:train_end], y[:train_end]
   X_val, y_val = X[train_end + H : val_end], y[train_end + H : val_end]
   X_test, y_test = X[val_end + H :], y[val_end + H :]
   ```

### CATEGORY 4: ARCHITECTURAL OVERFITTING (Missing Regularization)
**The Issue:**
You are feeding a high-dimensional, noisy time-series into an LSTM and Multi-Head Attention block with zero regularization. The network will perfectly memorize the training noise instead of extracting generalized features. Furthermore, Attention layers without Layer Normalization suffer from exploding/vanishing gradients during optimization.

**Agent Directive:**
Inject `LayerNormalization` and `Dropout` into the Neural Network architecture.
1. **In Python (Section 6), rewrite the architecture exactly as follows:**
   ```python
   in_lay = tf.keras.Input(shape=(120, 35))
   
   # LSTM Layer
   ls = tf.keras.layers.LSTM(35, return_sequences=True, activation='mish')(in_lay)
   ls_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ls)
   
   # Self-Attention with Dropout
   at = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=35, dropout=0.2)(ls_norm, ls_norm)
   
   # Residual Connection & Normalization
   res_add = tf.keras.layers.Add(name="Residual_Add")([ls_norm, at])
   res_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res_add)
   
   # Context Extraction (Last Step)
   pl = tf.keras.layers.Lambda(lambda x: x[:, -1, :], name="Extract_Last_Step")(res_norm)
   
   # Dense Network with Dropout Regularization
   do = tf.keras.layers.Dropout(0.3)(pl)
   d1 = tf.keras.layers.Dense(20, activation='mish', kernel_regularizer=tf.keras.regularizers.L2(1e-4))(do)
   ou = tf.keras.layers.Dense(3, activation='softmax')(d1)
   ```

### CATEGORY 5: THE CLASS 0 DOMINANCE
**The Issue:**
You are operating on 144-tick bars. Depending on the asset, a 144-tick bar might only move 0.1 points. Asking it to hit `TP=1.44` or `SL=0.50` within `H=30` bars might only be possible during the New York open, causing 90%+ of your dataset to be Class 0 (Do Nothing). Even with class weights, the network struggles.
**Agent Directive (Optional but Highly Recommended):**
*   Check the class distribution printed during training. If Class 0 is > 85%, your `TP` and `SL` values are too wide for a 144-tick timeframe. 
*   **Action:** Reduce `TP_POINTS` to 0.72 and `SL_POINTS` to 0.35 in both Python and MQL5 to increase signal frequency, providing the network with more balanced positive/negative examples to learn from.

---

### MQL5 `Predict()` Implementation Guide for the Agent (Strict Mathematics):

When implementing Category 1 in MQL5, the equations in `Predict()` must change to:
```cpp
// ATR Normalized by Close
f[10]=(float)(CATR(x,9)/c_a[x]); f[11]=(float)(CATR(x,18)/c_a[x]); f[12]=(float)(CATR(x,27)/c_a[x]);

// MACD Normalized by Close
f[13]=(float)(macd_a[x]/c_a[x]); f[14]=(float)(macd_signal_a[x]/c_a[x]); f[15]=(float)(macd_hist_a[x]/c_a[x]);

// EMA Distance Normalized by Close
f[16]=(float)((e9-c_a[x])/c_a[x]); f[17]=(float)((e18-c_a[x])/c_a[x]); f[18]=(float)((e27-c_a[x])/c_a[x]); 
f[19]=(float)((e54-c_a[x])/c_a[x]); f[20]=(float)((e144-c_a[x])/c_a[x]);

// Momentum Normalized by Close
f[27]=(float)((c_a[x]-c_a[RingIdx(119-i+9)])/c_a[x]); 
f[28]=(float)((c_a[x]-c_a[RingIdx(119-i+18)])/c_a[x]); 
f[29]=(float)((c_a[x]-c_a[RingIdx(119-i+27)])/c_a[x]);
```

Implementing these mathematical alignments will fundamentally transform the signal-to-noise ratio, allowing the Attention mechanism to find persistent, regime-agnostic patterns, drastically raising `val_accuracy`.

#### FLAW 6: INCOMPLETE SCALE-INVARIANCE (ATR Left Behind)
**The Issue:** The agent successfully divided MACD, Momentum, and EMA deviation by `Close`, but left ATR as an absolute dollar value in both Python and MQL5. ATR scales linearly with asset price.
**Agent Directive:**
*   **In Python:** Change `df[f'f{f_idx}'] = tr.rolling(p).mean()` to `df[f'f{f_idx}'] = tr.rolling(p).mean() / df['close']`
*   **In MQL5 (`Predict()`):** Change `f[10]=CATR(x,9);` to `f[10]=(float)(CATR(x,9)/c_a[x]);` (apply to 9, 18, and 27 periods).

#### FLAW 7: CONTIGUOUS SPLIT LEAKAGE (Missing Embargo)
**The Issue:** The dataset is split sequentially: `train_end = 0.70`, `val_start = 0.70`. Because the labeling function `label()` looks ahead `H=30` bars, the last 30 bars of the training set contain labels derived from the first 30 bars of the validation set. The model memorizes the validation data, resulting in false confidence during training.
**Agent Directive:**
*   **In Python:** You must enforce an Embargo Gap equal to the horizon `H`.
    ```python
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end + H : val_end], y[train_end + H : val_end]
    X_test, y_test = X[val_end + H :], y[val_end + H :]
    ```

#### FLAW 8: FAT-TAIL DISTRIBUTION DESTRUCTION (Standard vs Robust Scaling)
**The Issue:** Financial data is non-Gaussian (fat-tailed). Using standard `mean` and `std` scaling compresses 99% of normal market movements into a tiny range to accommodate massive outliers (e.g., flash crashes). The neural network cannot differentiate signals in a compressed range.
**Agent Directive:**
*   **In Python:** Replace `mean` and `std` with `median` and `iqr` (Interquartile Range).
    ```python
    median = np.median(X_train, axis=0)
    iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)
    X_train_seq, y_train_seq = win((X_train - median) / (iqr + 1e-8), y_train, H)
    ```
*   **In Python (Export):** Export `medians` and `iqrs` arrays to MQL5 instead of `means` and `stds`.
*   **In MQL5 (`Predict()`):** Replace `means[k]` and `stds[k]` with `medians[k]` and `iqrs[k]`.

---

### FINAL COPY-PASTE DIRECTIVES FOR THE AI AGENT

Tell the agent to apply these **exact** code replacements to guarantee extreme accuracy and absolute synchronization between Python and MQL5.

**1. PYTHON: Fix ATR & Embargo & Robust Scaling**
```python
# FIX: Scale-invariant SMA-based ATR
tr_df = pd.DataFrame({
    'hl': df['high'] - df['low'],
    'hc': (df['high'] - df['close'].shift(1)).abs(),
    'lc': (df['low'] - df['close'].shift(1)).abs()
})
tr = tr_df.max(axis=1)
for p, f_idx in zip([9, 18, 27],[10, 11, 12]):
    df[f'f{f_idx}'] = tr.rolling(p).mean() / df['close'] # Added division by close

# ... later down in the file ...

n_samples = len(X)
train_end, val_end = int(n_samples * 0.70), int(n_samples * 0.85)

# FIX: Add H-bar embargo to prevent leakage
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end + H : val_end], y[train_end + H : val_end]

# FIX: Robust Scaling (Median/IQR) to handle financial fat tails
median = np.median(X_train, axis=0)
iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)

X_train_seq, y_train_seq = win((X_train - median) / (iqr + 1e-8), y_train, H)
X_val_seq, y_val_seq = win((X_val - median) / (iqr + 1e-8), y_val, H)

# ... at the bottom ...
means_str = f"float medians[35]={{{','.join([f'{m:.6f}f' for m in median])}}};"
stds_str = f"float iqrs[35]={{{','.join([f'{s:.6f}f' for s in iqr])}}};"
```

**2. MQL5: Fix ATR array division & Normalization Variables**
```cpp
// Rename global arrays
float medians[35] = {0.0f}; // ⚠️ PASTE FROM PYTHON
float iqrs[35]  = {1.0f}; // ⚠️ PASTE FROM PYTHON

// Inside Predict()
void Predict() {
   for(int i=0; i<120; i++) {
      int x = RingIdx(119-i);     
      int x1 = RingIdx(119-i+1);  
      float f[35];
      f[0]=(float)MathLog(c_a[x]/(c_a[x1]+1e-8)); 
      f[1]=(float)(s_a[x]/c_a[x]); 
      f[2]=(float)d_a[x];
      f[3]=(float)((h_a[x]-MathMax(o_a[x],c_a[x]))/(c_a[x]+1e-8)); 
      f[4]=(float)((MathMin(o_a[x],c_a[x])-l_a[x])/(c_a[x]+1e-8));
      f[5]=(float)((h_a[x]-l_a[x])/(c_a[x]+1e-8)); 
      f[6]=(float)((c_a[x]-l_a[x])/(h_a[x]-l_a[x]+1e-8));
      f[7]=CRSI(x,9); f[8]=CRSI(x,18); f[9]=CRSI(x,27);
      
      // FIX: ATR is now scale-invariant
      f[10]=(float)(CATR(x,9)/c_a[x]); f[11]=(float)(CATR(x,18)/c_a[x]); f[12]=(float)(CATR(x,27)/c_a[x]);
      
      double e9=ema9_a[x], e18=ema18_a[x], e27=ema27_a[x], e54=ema54_a[x], e144=ema144_a[x];
      
      f[13]=(float)(macd_a[x]/c_a[x]); f[14]=(float)(macd_signal_a[x]/c_a[x]); f[15]=(float)(macd_hist_a[x]/c_a[x]);
      f[16]=(float)((e9-c_a[x])/c_a[x]); f[17]=(float)((e18-c_a[x])/c_a[x]); f[18]=(float)((e27-c_a[x])/c_a[x]); 
      f[19]=(float)((e54-c_a[x])/c_a[x]); f[20]=(float)((e144-c_a[x])/c_a[x]);
      
      f[21]=CalcCCI(x,9); f[22]=CalcCCI(x,18); f[23]=CalcCCI(x,27);
      f[24]=CWPR(x,9); f[25]=CWPR(x,18); f[26]=CWPR(x,27);
      
      f[27]=(float)((c_a[x]-c_a[RingIdx(119-i+9)])/c_a[x]); 
      f[28]=(float)((c_a[x]-c_a[RingIdx(119-i+18)])/c_a[x]); 
      f[29]=(float)((c_a[x]-c_a[RingIdx(119-i+27)])/c_a[x]);
      
      f[30]=(float)((dx_a[x]-dx_a[x1])/(dx_a[x1]+1e-8)); 
      f[31]=(float)((jp_a[x]-jp_a[x1])/(jp_a[x1]+1e-8));
      f[32]=CBBW(x,9); f[33]=CBBW(x,18); f[34]=CBBW(x,27);
      
      // FIX: Robust Scaling Implementation
      for(int k=0; k<35; k++) input_data[i*35+k]=(f[k]-medians[k])/(iqrs[k]+1e-8f);
   }
   // ... [Rest of inference block remains the same] ...
