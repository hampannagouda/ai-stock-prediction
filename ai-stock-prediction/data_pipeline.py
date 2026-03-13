# data_pipeline.py
# --------------------------------------------------
# Handles data download, feature engineering,
# scaling, and sequence generation for LSTM model
# --------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta

def download_data(ticker="AAPL", start="2015-01-01", end=None):
    """
    Download historical OHLCV data using yfinance.
    Ensures 'Adj Close' column exists even if auto_adjust=True.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df = df.dropna()

    # Ensure consistent column naming
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    return df

def add_technical_indicators(df):
    """Add key technical indicators safely and ensure 1D input for all TA functions."""
    import pandas as pd
    import ta

    # --- Ensure Adj Close exists ---
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    # --- Convert Adj Close to 1D Series (important!) ---
    if isinstance(df["Adj Close"], pd.DataFrame):
        df["Adj Close"] = df["Adj Close"].squeeze()
    else:
        df["Adj Close"] = pd.Series(df["Adj Close"], index=df.index)

    # Now it's guaranteed to be 1D
    close_series = df["Adj Close"].astype(float).values.flatten()  # Flatten to 1D np array
    close_series = pd.Series(close_series, index=df.index, name="Adj Close")

    # --- Technical indicators ---
    df["SMA_5"] = close_series.rolling(5).mean()
    df["SMA_10"] = close_series.rolling(10).mean()
    df["EMA_12"] = ta.trend.EMAIndicator(close=close_series, window=12).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

    macd_calc = ta.trend.MACD(close_series)
    df["MACD"] = macd_calc.macd()
    df["MACD_Signal"] = macd_calc.macd_signal()

    df["Return_1d"] = close_series.pct_change(1)

    # Lag features (use flattened values)
    for lag in range(1, 6):
        df[f"lag_{lag}"] = close_series.shift(lag)

    df["target"] = close_series.shift(-1)

    df = df.dropna()
    print("✅ Indicators added. Final shape:", df.shape)
    return df



def prepare_features(df):
    """Split features, scale, and return train/test data."""
    feature_cols = [
        "Adj Close", "Open", "High", "Low", "Volume",
        "SMA_5", "SMA_10", "EMA_12", "RSI_14",
        "MACD", "MACD_Signal", "Return_1d",
        "lag_1", "lag_2", "lag_3"
    ]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = scaler_x.fit_transform(train_df[feature_cols])
    X_test = scaler_x.transform(test_df[feature_cols])
    y_train = scaler_y.fit_transform(train_df[["target"]])
    y_test = scaler_y.transform(test_df[["target"]])

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y, train_df, test_df, feature_cols


def create_sequences(X, y, window=30):
    """Create rolling window sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)
