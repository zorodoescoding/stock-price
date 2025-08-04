import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# --------------------
# Load preprocessed data from data/
# --------------------
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# --------------------
# Sanity checks
# --------------------
assert not np.isnan(X_train).any(), "❌ X_train contains NaNs"
assert not np.isnan(y_train).any(), "❌ y_train contains NaNs"

# --------------------
# Model definition
# --------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# --------------------
# Train the model
# --------------------
os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint("models/lstm_model.keras", save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

print("✅ Model training complete.")
