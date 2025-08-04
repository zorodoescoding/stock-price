import pandas as pd
import numpy as np
import os
import joblib

# Load MultiIndex CSV
df_raw = pd.read_csv("data/stock_data.csv", header=[0, 1])

# Extract 'Close' price for AAPL
df = df_raw[('Close', 'AAPL')].copy()
df.name = 'Close'

# Drop NaNs
df.dropna(inplace=True)

# Normalize
df_min = df.min()
df_max = df.max()
df_norm = (df - df_min) / (df_max - df_min)

# Save scaler
scaler = {'min': df_min, 'max': df_max}
joblib.dump(scaler, 'data/scaler.pkl')

# Windowing
window_size = 60
X, y = [], []

for i in range(len(df_norm) - window_size):
    window = df_norm.iloc[i: i + window_size].values
    target = df_norm.iloc[i + window_size]
    
    if np.isnan(window).any() or np.isnan(target):
        continue

    X.append(window.reshape(-1, 1))  # Reshape for LSTM
    y.append(target)

X = np.array(X)
y = np.array(y)

# Save data
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("✅ Preprocessing complete.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("📦 Saved scaler.pkl with min and max values")
