# MarcStockPredictor


This script is a simple implementation of a Long Short-Term Memory (LSTM) model for stock price prediction using Tensorflow and Keras. The script downloads stock data from Yahoo Finance, selects the Close and Volume columns, normalizes the data, creates training and test data, creates sequences and labels for training and test data, defines the model architecture, compiles the model, trains the model, evaluates the model, predicts future stock prices and volumes, and plots the predicted future stock prices and volumes.

To use this script, you need to have Tensorflow, Keras, and yfinance installed. You can install them using pip or conda. You also need to change the ticker_name variable to the name of the stock you want to predict. By default, the script predicts the future stock prices and volumes for Amazon (AMZN).

The model used in this script is a simple LSTM model that consists of three LSTM layers, each followed by a dropout layer, and a dense layer. This model is not guaranteed to give high accuracy stock price predictions and is just a starting point for further experimentation and optimization.
