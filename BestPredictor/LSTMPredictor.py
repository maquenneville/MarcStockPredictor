# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:10:18 2023

@author: marca
"""

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping


def download_stock_data(ticker_name):
    stock_data = yf.download(ticker_name, period="max")
    return stock_data


def select_columns(data, columns):
    selected_data = data[columns]
    return selected_data


def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data


def create_sequences(data, dates, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y), dates[seq_length:]


def split_data(X, y, test_size=0.2):
    num_train_samples = int((1 - test_size) * X.shape[0])
    X_train = X[:num_train_samples]
    y_train = y[:num_train_samples]
    X_test = X[num_train_samples:]
    y_test = y[num_train_samples:]
    return X_train, y_train, X_test, y_test


def build_model(seq_length):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                128, return_sequences=True, input_shape=(seq_length, 2)
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(monitor="val_loss", patience=5)
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
    )
    return model, history


def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    return loss


def predict_future_stock_prices(
    model, scaler, selected_data, seq_length, future_periods
):
    scaled_data = scaler.transform(selected_data)
    future_data = scaled_data[-seq_length:]
    future_predictions = []
    future_dates = pd.date_range(
        start=selected_data.index[-1], periods=future_periods + 1, freq="B"
    )[1:]
    for i in range(future_periods):
        prediction = model.predict(future_data.reshape(1, seq_length, 2))[0, 0]
        volume = scaled_data[-future_periods + i, 1]
        future_predictions.append((prediction, volume))
        future_data = np.append(future_data[1:], [[prediction, volume]], axis=0)
    future_predictions = scaler.inverse_transform(np.array(future_predictions))
    future_predictions = future_predictions[-future_periods:]
    return future_dates, future_predictions


def plot_predicted_stock_prices_and_volumes(future_dates, future_predictions):
    price_data = future_predictions[:, 0]
    volume_data = future_predictions[:, 1]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.plot(future_dates, price_data, color="tab:red")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volume")
    ax2.bar(future_dates, volume_data, alpha=0.3, color="tab:blue")
    fig.tight_layout()
    plt.show()


def predict_stock_prices(ticker_name, seq_length, future_periods):
    # Download stock price data for a single stock
    stock_data = download_stock_data(ticker_name)

    # Select the 'Close' and 'Volume' columns
    selected_data = select_columns(stock_data, ["Close", "Volume"])

    # Convert selected_data to a pandas object
    if selected_data.shape[1] == 2:
        selected_data = pd.DataFrame(selected_data, columns=["Close", "Volume"])
    else:
        selected_data = pd.Series(selected_data, name="Close")

    # Normalize the data
    scaler, scaled_data = normalize_data(selected_data)

    # Create sequences and labels for training and test data
    X, y, index = create_sequences(scaled_data, selected_data.index, seq_length)

    # Build the model
    model = build_model(seq_length)

    # Train the model
    X_train, y_train, X_test, y_test = split_data(X, y)
    train_model(model, X_train, y_train, X_test, y_test)

    # Predict future stock prices and volumes
    future_dates, future_predictions = predict_future_stock_prices(
        model, scaler, selected_data[-seq_length:], seq_length, future_periods
    )

    return future_dates, future_predictions


def main():
    ticker_name = "AAPL"
    seq_length = 60
    future_periods = 60

    # Predict future stock prices and volumes
    future_dates, future_predictions = predict_stock_prices(
        ticker_name, seq_length, future_periods
    )

    # Plot the predicted future stock prices and volumes
    plot_predicted_stock_prices_and_volumes(future_dates, future_predictions)


if __name__ == "__main__":
    main()


















