# scripts/download_nifty_ohlc.py
import yfinance as yf
import pandas as pd
import os

# NIFTY 50 symbols with NSE suffix
nifty50 = [
    "RELIANCE.NS", "TCS.NS", "HDFC.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "AXISBANK.NS", "HCLTECH.NS", "BAJFINANCE.NS", "MARUTI.NS", "ASIANPAINT.NS",
    "NESTLEIND.NS", "TECHM.NS", "SUNPHARMA.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "TITAN.NS", "POWERGRID.NS", "DIVISLAB.NS", "BRITANNIA.NS",
    "M&M.NS", "HDFCBANK.NS", "ONGC.NS", "JSWSTEEL.NS", "NTPC.NS",
    "GRASIM.NS", "HDFCLIFE.NS", "TATASTEEL.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS",
    "SBILIFE.NS", "INDUSINDBK.NS", "BPCL.NS", "TATAMOTORS.NS", "CIPLA.NS",
    "EICHERMOT.NS", "DRREDDY.NS", "SHREECEM.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "ICICIGI.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS"
]

# Download historical data
print("📥 Downloading historical OHLC data for NIFTY 50...")
data = yf.download(nifty50, start="1993-01-01", end="2025-01-01")  # daily OHLC

# Create data folder
os.makedirs("data", exist_ok=True)

# Save Open and Close separately
df_open = data['Open']
df_close = data['Close']

df_open.to_csv("data/NIFTY_open.csv")
df_close.to_csv("data/NIFTY_close.csv")

print("✅ NIFTY_open.csv and NIFTY_close.csv created in data/ folder.")
