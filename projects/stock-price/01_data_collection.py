# 01_download.py
import pandas as pd
import yfinance as yf
import os

# Download historical data
ticker = 'AAPL'
data = yf.download(ticker, start='2018-01-01', end='2024-01-01')

# Save the full DataFrame as CSV
os.makedirs('data', exist_ok=True)
data.to_csv('data/stock_data.csv')
print("✅ Stock data saved to data/stock_data.csv")
