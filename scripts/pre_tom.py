# scripts/fetch_and_predict_nifty50_tomorrow.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import os

# ===============================
# NIFTY 50 stocks
# ===============================
STOCKS = [
    "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS",
    "HDFC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS",
    "WIPRO.NS"
]

MODEL_PATH = "models-NIFTY-240-1-LSTM/final_lstm_nifty.h5"
DATA_DIR = "data/stock_data_2025"
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# Load LSTM model
# ===============================
model = load_model(MODEL_PATH)

# ===============================
# Fetch and save stock data
# ===============================
def fetch_and_save(stock):
    print(f"Fetching {stock}...")
    df = yf.download(stock, start="2023-09-17", end="2025-09-18", interval="1d")
    if df.empty:
        print(f"⚠️ No data for {stock}")
        return None
    df.to_csv(f"{DATA_DIR}/{stock.replace('.NS','')}_data.csv")
    return df

# ===============================
# Predict tomorrow
# ===============================
def predict_stock(df, stock):
    if len(df) < 240:
        print(f"⚠️ Not enough data for {stock}")
        return None

    df['Return'] = df['Close'].pct_change()
    df = df.dropna()

    returns = df['Return'].values[-240:].reshape(-1, 1)

    scaler = RobustScaler()
    returns_scaled = scaler.fit_transform(returns)

    X_input = returns_scaled.reshape(1, 240, 1).astype(np.float32)

    prediction = model.predict(X_input)
    predicted_class = np.argmax(prediction, axis=1)[0]
    movement = "Up" if predicted_class == 1 else "Down"
    return movement

# ===============================
# Main
# ===============================
results = []

for stock in STOCKS:
    try:
        df = fetch_and_save(stock)
        if df is not None:
            movement = predict_stock(df, stock)
            if movement:
                results.append({'Stock': stock, 'Predicted': movement})
    except Exception as e:
        print(f"❌ Error with {stock}: {e}")

# Save predictions
if results:
    result_df = pd.DataFrame(results)
    result_df.to_csv("results-NIFTY-240-1-LSTM/predictions_tomorrow.csv", index=False)
    print("✅ Predictions saved for all stocks:")
    print(result_df)
else:
    print("⚠️ No valid predictions made!")
