# MarcStockPredictor


The first script is a simple implementation of a Long Short-Term Memory (LSTM) model for stock price prediction using Tensorflow and Keras. The script downloads stock data from Yahoo Finance, selects the Close and Volume columns, normalizes the data, creates training and test data, creates sequences and labels for training and test data, defines the model architecture, compiles the model, trains the model, evaluates the model, predicts future stock prices and volumes, and plots the predicted future stock prices and volumes.

To use this script, you need to have Tensorflow, Keras, and yfinance installed. You can install them using pip or conda. You also need to change the ticker_name variable to the name of the stock you want to predict. By default, the script predicts the future stock prices and volumes for Amazon (AMZN).

The model used in this script is a simple LSTM model that consists of three LSTM layers, each followed by a dropout layer, and a dense layer. 


The second script uses historical stock price data, along with various technical analysis indicators, to train a machine learning model that can predict future prices. The script first downloads historical stock price data from Yahoo Finance using the yfinance library. It then adds technical analysis indicators to the data, forward-fills any missing or 0 values, and normalizes the features. The technical indicators used are:

5-day, 20-day, 50-day, and 200-day Simple Moving Averages (SMAs)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
The script then splits the data into training and testing sets, trains a Random Forest machine learning model using the training data, and evaluates the model's performance on the testing data. Finally, the script uses the trained model to predict future stock prices.


The models in this project are not guaranteed to give high accuracy stock price predictions and are just a starting point for further experimentation and optimization.
