import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load processed data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Safety check
assert not np.isnan(X).any(), "❌ X contains NaNs"
assert not np.isnan(y).any(), "❌ y contains NaNs"

# Model
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(1, activation='sigmoid')  # Because target is normalized between 0-1
])

model.compile(optimizer='adam', loss='mse')  # Use 'mae' if you prefer

# Train
history = model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# Save model
model.save("data/lstm_model.keras")

# Plot training loss
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()
