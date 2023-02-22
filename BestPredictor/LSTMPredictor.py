# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:10:18 2023

@author: marca
"""

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

ticker_name = 'AMZN'

# Download stock price data for a single stock
stock_data = yf.download(ticker_name, period='max')

# Select the 'Close' and 'Volume' columns
selected_data = stock_data[['Close', 'Volume']]

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(selected_data)

# Create training and test data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences and labels for training and test data
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate the model
model.evaluate(X_test, y_test)

# Predict future stock prices and volumes
future_periods = 60
future_data = scaled_data[-seq_length:]
future_predictions = []
for i in range(future_periods):
    prediction = model.predict(future_data.reshape(1, seq_length, 2))[0, 0]
    volume = scaled_data[-future_periods+i, 1]
    future_predictions.append((prediction, volume))
    future_data = np.append(future_data[1:], [[prediction, volume]], axis=0)

# Inverse transform the predicted future stock prices and volumes to get the actual values
future_predictions = scaler.inverse_transform(np.array(future_predictions))

# Get the future dates for the predicted stock prices
future_dates = pd.date_range(start=stock_data.index[-1], periods=future_periods+1, freq='B')[1:]

# Only keep the predictions for the next future_periods days
future_predictions = future_predictions[-future_periods:]

# Plot the predicted future stock prices and volumes
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color=color)
ax1.plot(future_dates, future_predictions[:, 0], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Volume', color=color)
ax2.bar(future_dates, future_predictions[:, 1], alpha=0.3, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

















