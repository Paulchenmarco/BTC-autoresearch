"""
Fetch BTC daily OHLC data and save as parquet.

Usage:
    python scripts/build_dataset.py

Saves to data/btc_features.parquet with columns:
    date, open, high, low, close, volume
"""

import os
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "btc_features.parquet")

# BTC genesis date for power law features
BTC_GENESIS = pd.Timestamp("2009-01-03")


def fetch_btc_ohlc(start="2017-01-01"):
    """Fetch BTC-USD daily OHLC from Yahoo Finance."""
    print(f"Fetching BTC-USD daily data from {start}...")
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start=start, interval="1d")

    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance")

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # Normalize date to date-only (no timezone)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)

    # Drop rows with missing close prices
    df = df.dropna(subset=["close"])

    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = fetch_btc_ohlc()
    print(f"Fetched {len(df)} daily rows: {df['date'].min().date()} to {df['date'].max().date()}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
