📈 LSTM Stock Price Forecasting
This project implements a time series forecasting pipeline using an LSTM (Long Short-Term Memory) neural network to predict future stock closing prices based on historical stock data.

🧠 Project Goal
To build a robust LSTM-based model that forecasts future stock prices using past data only, without data leakage, and following core machine learning principles such as proper data splitting, normalization, and model evaluation.

📊 Dataset
The dataset used is a historical stock dataset (stock_data.csv) containing:
Date
Open
High
Low
Close
Volume
Only the Close column is used for modeling.

⚙️ Workflow
1. Data Loading (01_load_data.py)
Loads stock_data.csv and stores it locally. You can replace it with your own stock CSV as long as the column format is the same.

2. Preprocessing (02_preprocessing.py)
Converts Date to datetime
Sorts chronologically
Uses a sliding window approach (default: window_size=60)
Splits data into 80% Train, 20% Test (chronologically, no leakage)
Scales features using MinMaxScaler
Saves:
X_train, y_train, X_test, y_test
scaler.pkl (for inverse transforms during evaluation)

3. Model Training (03_model_training.py)
Loads preprocessed data
Defines a stacked LSTM model with:
2 LSTM layers
1 Dense output layer
Uses MSE loss and Adam optimizer
Trains for 50 epochs (can be changed)
Saves trained model as models/lstm_model.keras

4. Prediction & Plotting (04_plot_predict.py)
Loads test data and trained model
Makes predictions
Inverses scaling for real-world interpretation

Plots 
loss vs. number of epochs
actual vs. predicted prices

📉 Example Plot
After running 04_plot_predict.py, you will see a matplotlib chart of predicted vs actual closing prices on the test set.

🧪 Requirements
Python 3.8+
tensorflow
pandas
numpy
matplotlib
scikit-learn

