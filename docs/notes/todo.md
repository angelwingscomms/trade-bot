`norm_w (250): The features are Z-score normalized using a rolling 250-bar window to keep input values stable for the neural network.` does the pipeline do this?

`Input: A window of the last 144 bars (ticks aggregated) with 40 features each.
LSTM Layer: Captures the sequential 'memory' and trends in the price action.
Multi-Head Attention: Allows the model to 'focus' on specific past events within that 144-bar window that are most relevant to the current prediction.
Global Pooling & Dense: Flattens the temporal data into a final probability distribution across the 3 classes (Buy/Sell/Hold).
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')` which architecture does this?