from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging

# Set up logging (visible in Render logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wallstrai Forecast API")

# Allow CORS (tighten origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    ticker: str
    forecast_days: int = 15
    history_years: int = 5          # Reduced default to help free tier
    seq_length: int = 60

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        logger.info(f"Starting forecast for ticker: {req.ticker}")

        # Date range
        hoy = datetime.today()
        start_date = hoy - timedelta(days=req.history_years * 365 + 100)  # buffer
        end_date = hoy

        # Fetch data
        logger.info("Downloading data...")
        data = yf.download(req.ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data returned from yfinance for {req.ticker}")

        logger.info(f"Downloaded data shape: {data.shape}")

        # Select needed columns and standardize names
        data = data[['Close', 'Low', 'High']].copy()
        data.columns = ['close', 'low', 'high']

        # Rename index to 'date' (lowercase)
        data.index.name = 'date'

        # Preprocessing
        close_prices = data['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(close_prices)

        seq_length = req.seq_length
        X, y = [], []
        for i in range(len(scaled_prices) - seq_length):
            X.append(scaled_prices[i:i + seq_length])
            y.append(scaled_prices[i + seq_length])

        if len(X) == 0:
            raise ValueError(
                f"Not enough data points for {req.ticker}. "
                f"Need at least {seq_length + 1} trading days, got {len(scaled_prices)}"
            )

        X, y = np.array(X), np.array(y)
        logger.info(f"Created {len(X)} sequences")

        # Model - smaller to reduce memory/time
        model = Sequential([
            LSTM(32, input_shape=(seq_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)

        # Train
        logger.info("Starting model training...")
        model.fit(
            X, y,
            batch_size=32,
            epochs=30,                # Reduced from 50
            callbacks=[early_stopping],
            verbose=0
        )
        logger.info("Training finished")

        # Forecast
        logger.info("Starting forecasting...")
        num_forecast = req.forecast_days
        forecast_scaled = []
        current_seq = scaled_prices[-seq_length:].copy()

        for _ in range(num_forecast):
            pred = model.predict(current_seq.reshape(1, seq_length, 1), verbose=0)[0, 0]
            forecast_scaled.append(pred)
            current_seq = np.append(current_seq[1:], [[pred]], axis=0)

        forecast_prices = scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).flatten().tolist()

        # Forecast dates (business days)
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=num_forecast,
            freq='B'
        ).strftime('%Y-%m-%d').tolist()

        # Prepare data table
        data_table = data.reset_index().to_dict(orient='records')

        # Convert datetime to string
        for row in data_table:
            row['date'] = row['date'].strftime('%Y-%m-%d')

        logger.info("Forecast completed successfully")

        return {
            "data_table": data_table,
            "forecast_dates": forecast_dates,
            "forecast_prices": forecast_prices
        }

    except Exception as e:
        logger.error(f"Error processing {req.ticker}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))