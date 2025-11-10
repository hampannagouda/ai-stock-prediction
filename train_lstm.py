# train_lstm.py
# --------------------------------------------------
# Builds, trains, evaluates, and saves the LSTM model
# --------------------------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from data_pipeline import download_data, add_technical_indicators, prepare_features, create_sequences


def build_lstm(n_features, units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(None, n_features)))
    model.add(Dropout(dropout))
    model.add(LSTM(units // 2))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(ticker="AAPL"):
    df = download_data(ticker)
    df = add_technical_indicators(df)
    X_train, X_test, y_train, y_test, scaler_x, scaler_y, train_df, test_df, feature_cols = prepare_features(df)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)

    model = build_lstm(n_features=X_train_seq.shape[2])

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=80,
        batch_size=32,
        callbacks=[es, lr],
        verbose=2,
    )

    # Predict and evaluate
    preds = model.predict(X_test_seq)
    y_true = y_test_seq.reshape(-1, 1)
    y_pred_inv = scaler_y.inverse_transform(preds)
    y_true_inv = scaler_y.inverse_transform(y_true)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = mean_squared_error(y_true_inv, y_pred_inv, squared=False)
    print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save model and scalers
    model.save("models/lstm_model.h5")
    joblib.dump(scaler_x, "models/scaler_x.save")
    joblib.dump(scaler_y, "models/scaler_y.save")

    print("✅ Model and scalers saved successfully.")
    return model, scaler_x, scaler_y, feature_cols


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_model("AAPL")
