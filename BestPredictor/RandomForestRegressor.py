# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 00:16:26 2023

@author: marca
"""

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
import ta


def load_data(ticker_name):
    """
    Downloads historical data for a given stock ticker from yfinance.
    Returns a pandas DataFrame with the historical data.
    """
    stock_data = yf.download(ticker_name, period="max")
    stock_data = stock_data[["Close", "Volume"]]  # select only the Close column
    stock_data.index.name = "Date"  # set the index name
    return stock_data


def preprocess_data(stock_data):
    """
    Adds technical analysis indicators to the stock_data DataFrame,
    forward-fills any 0 or NaN values, and normalizes the features.
    Returns the modified DataFrame.
    """
    # Add the 5SMA, 20SMA, 50SMA, and 200SMA.
    stock_data.loc[:, "SMA5"] = ta.trend.sma_indicator(stock_data["Close"], window=5)
    stock_data.loc[:, "SMA20"] = ta.trend.sma_indicator(stock_data["Close"], window=20)
    stock_data.loc[:, "SMA50"] = ta.trend.sma_indicator(stock_data["Close"], window=50)
    stock_data.loc[:, "SMA200"] = ta.trend.sma_indicator(
        stock_data["Close"], window=200
    )

    # Add the RSI.
    stock_data.loc[:, "RSI"] = ta.momentum.RSIIndicator(stock_data["Close"]).rsi()

    # Add the MACD.
    macd = ta.trend.MACD(stock_data["Close"])
    stock_data.loc[:, "MACD"] = macd.macd()
    stock_data.loc[:, "MACD Signal"] = macd.macd_signal()

    # Forward-fill any missing or 0 values in the DataFrame.
    stock_data = stock_data.replace(0, pd.np.nan).ffill()

    # Normalize the features.
    scaler = StandardScaler()
    stock_data = pd.DataFrame(
        scaler.fit_transform(stock_data),
        index=stock_data.index,
        columns=stock_data.columns,
    )
    stock_data.fillna(0, inplace=True)

    inv_scaler = StandardScaler().fit(np.array(stock_data["Close"]).reshape(-1, 1))

    return stock_data, inv_scaler


def split_data(data, test_size=0.2):
    """
    Splits data into training and testing sets.
    Returns a tuple with the training and testing sets.
    """
    n_train = int(len(data) * (1 - test_size))

    train_data = data[:n_train]
    test_data = data[n_train:]

    train_inputs = train_data.drop(columns=["Close"])
    print(train_inputs.columns)
    train_outputs = train_data["Close"]
    test_inputs = test_data.drop(columns=["Close"])
    test_outputs = test_data["Close"]
    print(test_inputs.columns)

    return (train_inputs, train_outputs), (test_inputs, test_outputs)


def build_model(train_data, test_data):
    """
    Builds and trains a Random Forest model using the given training data.
    Returns the trained model and the test set.
    """
    best_model, best_params = optimize_hyperparameters(train_data, test_data)
    
    train_inputs, train_outputs = train_data
    test_inputs, test_outputs = test_data

    model = RandomForestRegressor(**best_params)
    model.fit(train_inputs, train_outputs)

    return model, (test_inputs, test_outputs)



def optimize_hyperparameters(train_data, test_data):
    """
    Optimizes the hyperparameters of a Random Forest model using Grid Search.
    Returns the best model and its parameters.
    """
    train_inputs, train_outputs = train_data
    test_inputs, test_outputs = test_data
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    model = RandomForestRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(train_inputs, train_outputs)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return best_model, best_params



def predict_future_prices(model, test_data, inv_scaler, days=30):
    """
    Predicts future stock prices using the given model and test data.
    Returns a DataFrame with the predicted prices and the corresponding dates.
    """
    test_inputs, test_outputs = test_data
    last_date = test_inputs.index[-1].date()

    predictions = []
    for i in range(days):
        next_date = last_date + pd.DateOffset(1)
        next_input = test_inputs.iloc[[-1]]
        next_input["Close"] = (
            test_outputs[-1] if test_outputs.size > 0 else next_input["Close"]
        )
        next_output = model.predict(next_input.drop(columns=["Close"]))
        next_output = inv_scaler.inverse_transform(next_output.reshape(1, -1))[0][0]
        predictions.append([next_date, next_output])
        last_date = next_date
        test_outputs = np.append(test_outputs, next_output)

    return pd.DataFrame(predictions, columns=["Date", "Predicted Close"])


def plot_predicted_prices(predictions):
    """
    Plots the predicted prices over time.
    """
    plt.plot(predictions["Date"], predictions["Predicted Close"])
    plt.xlabel("Date")
    plt.ylabel("Predicted Close Price")
    plt.title("Predicted Prices Over Time")
    plt.show()


if __name__ == "__main__":

    load_data = load_data("AAPL")

    data, scaler = preprocess_data(load_data)

    train_data, test_data = split_data(data)

    model, td = build_model(train_data, test_data)

    predictions = predict_future_prices(model, test_data, scaler)

    plot_predicted_prices(predictions)
