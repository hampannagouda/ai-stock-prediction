# app.py
# --------------------------------------------------
# Streamlit user interface for next-day stock prediction
# --------------------------------------------------

import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from tensorflow.keras.models import load_model
from data_pipeline import add_technical_indicators

st.set_page_config(page_title="AI Stock Predictor", layout="centered")

st.title("📈 AI Stock Prediction — Next Day Close (LSTM)")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS):", "AAPL")
start = st.date_input("Start Date", value=date(2018, 1, 1))
predict_button = st.button("🔮 Predict Next Day Price")

if predict_button:
    st.info("Fetching latest stock data...")
    df = yf.download(ticker, start=start, progress=False).dropna()
    df = add_technical_indicators(df)

    try:
        model = load_model("models/lstm_model.h5")
        scaler_x = joblib.load("models/scaler_x.save")
        scaler_y = joblib.load("models/scaler_y.save")
        st.success("Loaded trained LSTM model.")
    except Exception as e:
        st.error("Model or scalers not found. Please train the model using train_lstm.py first.")
        st.stop()

    feature_cols = [
        "Adj Close", "Open", "High", "Low", "Volume",
        "SMA_5", "SMA_10", "EMA_12", "RSI_14",
        "MACD", "MACD_Signal", "Return_1d",
        "lag_1", "lag_2", "lag_3"
    ]

    window = 30
    df = df.dropna()
    last_window = df[feature_cols].values[-window:]
    last_scaled = scaler_x.transform(last_window)
    last_scaled = last_scaled.reshape(1, last_scaled.shape[0], last_scaled.shape[1])

    pred_scaled = model.predict(last_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)[0][0]

    st.metric(label=f"Predicted Next Day Close for {ticker}", value=f"${pred:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    df["Adj Close"].iloc[-100:].plot(ax=ax, label="Actual Close")
    ax.scatter(df.index[-1] + timedelta(days=1), pred, color="red", label="Predicted Next Day")
    ax.legend()
    ax.set_title(f"{ticker} — Predicted vs Actual")
    st.pyplot(fig)
