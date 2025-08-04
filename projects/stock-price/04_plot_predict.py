import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

# -------------------------------
# Load the saved model and data
# -------------------------------
model = load_model("models/lstm_model.keras")

# Load test data from data/
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Load the scaler to inverse transform
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# -------------------------------
# Make predictions
# -------------------------------
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
