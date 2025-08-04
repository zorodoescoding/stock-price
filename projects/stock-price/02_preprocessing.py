import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# --------------------
# Parameters
# --------------------
SEQ_LEN = 60
INPUT_COL = ('Close', 'AAPL')
DATA_PATH = "data/stock_data.csv"
SCALER_PATH = "models/scaler.pkl"

# Output paths
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

X_train_path = "data/X_train.npy"
y_train_path = "data/y_train.npy"
X_test_path = "data/X_test.npy"
y_test_path = "data/y_test.npy"

# --------------------
# Load Data
# --------------------
df = pd.read_csv(DATA_PATH, header=[0, 1])
print("✅ Data loaded")
print("Columns:", df.columns.tolist())

# Drop rows with NaNs (just in case)
df.dropna(inplace=True)

# Extract Close prices
prices = df[INPUT_COL].values.reshape(-1, 1)

# --------------------
# Train-Test Split BEFORE scaling
# --------------------
split_index = int(len(prices) * 0.8)
train_prices = prices[:split_index]
test_prices = prices[split_index - SEQ_LEN:]  # include previous SEQ_LEN for test sequences

# --------------------
# Fit scaler ONLY on training data
# --------------------
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_prices)
test_scaled = scaler.transform(test_prices)

# Save the scaler
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# --------------------
# Create Sequences
# --------------------
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

# --------------------
# Save to .npy files
# --------------------
np.save(X_train_path, X_train)
np.save(y_train_path, y_train)
np.save(X_test_path, X_test)
np.save(y_test_path, y_test)

print("✅ Preprocessing complete.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
