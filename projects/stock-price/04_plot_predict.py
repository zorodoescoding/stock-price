import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Load data and model
X = np.load("data/X.npy")
y = np.load("data/y.npy")
model = load_model("data/lstm_model.keras")

# Predict
y_pred = model.predict(X).flatten()

# Load scaler
scaler = joblib.load("data/scaler.pkl")
data_min = scaler['min']
data_max = scaler['max']

# Inverse transform predictions and actual values
y_pred_inv = y_pred * (data_max - data_min) + data_min
y_actual_inv = y * (data_max - data_min) + data_min

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_actual_inv, label="Actual Price", linewidth=2)
plt.plot(y_pred_inv, label="Predicted Price", linestyle="--")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
