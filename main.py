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

app = FastAPI(title="Wallstrai Forecast API")

# Allow CORS so Streamlit (different domain) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Tighten this in production (e.g. your Streamlit URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    ticker: str
    forecast_days: int = 15
    history_years: int = 10
    seq_length: int = 60

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        # Dates
        hoy = datetime.today()
        start_date = hoy - timedelta(days=req.history_years * 365 + 100)  # buffer
        end_date = hoy

        # Fetch data
        data = yf.download(req.ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError("No data returned from yfinance")

        # Robust column handling
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Keep only needed columns and standardize names
        data = data[['Close', 'Low', 'High']].copy()
        data.columns = ['close', 'low', 'high']

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
            raise ValueError("Not enough data for sequences")

        X, y = np.array(X), np.array(y)

        # Model
        model = Sequential([
            LSTM(50, input_shape=(seq_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Train (no validation split here to keep it simple/fast)
        model.fit(X, y, batch_size=32, epochs=50, callbacks=[early_stopping], verbose=0)

        # Forecast
        num_forecast = req.forecast_days
        forecast_scaled = []
        current_seq = scaled_prices[-seq_length:].copy()

        for _ in range(num_forecast):
            pred = model.predict(current_seq.reshape(1, seq_length, 1), verbose=0)[0, 0]
            forecast_scaled.append(pred)
            current_seq = np.append(current_seq[1:], [[pred]], axis=0)

        forecast_prices = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten().tolist()

        # Forecast dates (business days)
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_forecast, freq='B')
        forecast_dates = forecast_dates.strftime('%Y-%m-%d').tolist()

        # Historical data table (as list of dicts for easy JSON)
        data_table = data.reset_index().to_dict(orient='records')
        # Convert Timestamp to str
        for row in data_table:
            row['date'] = row['date'].strftime('%Y-%m-%d')

        return {
            "data_table": data_table,           # list of dicts: [{"date": "...", "close": ..., "low": ..., "high": ...}, ...]
            "forecast_dates": forecast_dates,   # ["2026-02-06", ...]
            "forecast_prices": forecast_prices  # [152.3, 153.1, ...]
        }

    except Exception as e:
        logging.error(f"Error processing {req.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))