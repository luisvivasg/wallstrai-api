# wallstrai
Forecast shares price using LSTM and news.


# Wallstrai – Stock Price Forecast with LSTM

**Wallstrai** is a simple, interactive web app that lets users get quick forecasts of stock prices using historical data and a basic LSTM neural network model. Enter a stock ticker (e.g., AAPL), click "Get forecast", and view:

- Historical price charts (last 6 months view with high/low)
- LSTM-predicted prices for the next 15 business days
- Model evaluation on the last 30 days of data
- Training loss curves

The app pulls real-time historical data from Yahoo Finance and trains a small LSTM model on-the-fly for each query.


## Project Overview

This Streamlit application provides an easy-to-use interface for visualizing stock price history and generating short-term forecasts using deep learning. Key features include:

- User input for any valid stock ticker
- 10 years of historical daily data download
- Min-Max scaling and sequence preparation (60-day lookback)
- LSTM model training with dropout and early stopping
- Rolling-window forecasting for the next 15 business days
- Interactive Plotly charts for historical data, forecast, backtesting (last 30 days), and training loss
- Basic company info display (name and website)

The goal is to demonstrate a full end-to-end ML workflow in a clean, shareable web app.

## Tech Stack

- **Frontend / Web Framework**: [Streamlit](https://streamlit.io/)
- **Data Fetching**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance unofficial client)
- **Data Processing & Visualization**:
  - pandas
  - numpy
  - plotly.graph_objects (interactive charts)
- **Machine Learning**:
  - scikit-learn (MinMaxScaler)
  - tensorflow / keras (LSTM model, Dense layers, Dropout, EarlyStopping)
- **Date Handling**: datetime, pandas.tseries.offsets
- **Deployment**: Streamlit Community Cloud (or local / other hosts)

## How AI is Used

The core AI component is a **recurrent neural network (LSTM)** trained specifically for each user-requested stock:

1. **Data Preparation**:
   - Fetch ~10 years of daily closing prices via yfinance
   - Normalize prices to [0, 1] using MinMaxScaler
   - Create sequences: 60-day input windows → predict the next day's close

2. **Model Architecture**:
   ```python
   Sequential([
       LSTM(50, return_sequences=True, input_shape=(60, 1)),
       Dropout(0.2),
       LSTM(50, return_sequences=False),
       Dropout(0.2),
       Dense(1)
   ])

## Improvements
News information can be taken to generate a comprenhensive forcast that also includes the fundamental analysis and merge it with the technical analysis.

Hyperparameter tuning — Use Optuna or Keras Tuner to find better layer sizes / learning rates per stock.

Ensemble or advanced models — Try Prophet, XGBoost, Transformer-based models (e.g. Temporal Fusion Transformer), or even pre-trained time-series models from Hugging Face.

Multi-step improvements — Add support for custom forecast periods, multi-ticker comparison, or portfolio-level forecasts.

Reliable data source — Migrate to Polygon.io (official APIs with keys) to avoid rate limits entirely.

Create Token for API.

## Instructions
Knowing the US stock ticket, fill the filed and click enter.

