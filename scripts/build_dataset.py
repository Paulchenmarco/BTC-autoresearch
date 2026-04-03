"""
Merge raw data files into a single parquet dataset.
Run download_data.py first to fetch the raw files.

Usage:
    python scripts/build_dataset.py

Reads from data/raw/*.csv, outputs data/btc_features.parquet.
"""

import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "btc_features.parquet")


def _load_csv(filename, date_col="date"):
    """Load a CSV from raw dir, parse dates, return DataFrame."""
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    return df


def load_ohlc():
    """Load BTC OHLC from Yahoo Finance CSV."""
    df = _load_csv("btc_ohlc.csv")
    if df is None:
        raise FileNotFoundError("btc_ohlc.csv not found. Run download_data.py first.")
    return df


def load_coinmetrics():
    """Load CoinMetrics data and extract relevant columns."""
    path = os.path.join(RAW_DIR, "coinmetrics_btc.csv")
    if not os.path.exists(path):
        print("  [warn] coinmetrics_btc.csv not found, skipping")
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()

    # Extract and rename relevant columns
    col_map = {
        "CapMVRVCur": "mvrv_ratio",
        "HashRate": "hash_rate",
        "IssTotNtv": "issuance_ntv",
        "IssTotUSD": "issuance_usd",
        "AdrActCnt": "active_addresses",
        "SplyCur": "supply_current",
        "CapMrktCurUSD": "market_cap_usd",
        "FlowInExNtv": "exchange_inflow_ntv",
        "FlowOutExNtv": "exchange_outflow_ntv",
    }

    out = df[["date"]].copy()
    for src, dst in col_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    return out


def load_bgeometrics():
    """Load all BGeometrics indicator CSVs."""
    frames = {}
    for filename in glob.glob(os.path.join(RAW_DIR, "bgeometrics_*.csv")):
        if "_debug" in filename:
            continue
        basename = os.path.basename(filename)
        # Extract metric name: bgeometrics_nupl.csv -> nupl
        metric = basename.replace("bgeometrics_", "").replace(".csv", "")

        df = pd.read_csv(filename)
        # Standardize date column
        date_col = [c for c in df.columns if c.lower() in ("date", "time", "t", "x")]
        if date_col:
            df = df.rename(columns={date_col[0]: "date"})
        elif df.columns[0] != "date":
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])

        # The second column is the metric value
        value_col = [c for c in df.columns if c != "date"][0]
        df[metric] = pd.to_numeric(df[value_col], errors="coerce")
        frames[metric] = df[["date", metric]]

    if not frames:
        print("  [warn] No BGeometrics files found")
        return None

    # Merge all BGeometrics metrics on date
    result = None
    for metric, df in frames.items():
        if result is None:
            result = df
        else:
            result = result.merge(df, on="date", how="outer")

    return result


def load_dvol():
    """Load Deribit DVOL data."""
    df = _load_csv("deribit_dvol.csv")
    if df is None:
        print("  [warn] deribit_dvol.csv not found, skipping")
        return None

    # Keep close as the primary DVOL value
    out = df[["date"]].copy()
    out["dvol"] = pd.to_numeric(df["dvol_close"], errors="coerce")
    return out


def load_funding():
    """Load Binance funding rate data."""
    df = _load_csv("binance_funding.csv")
    if df is None:
        print("  [warn] binance_funding.csv not found, skipping")
        return None

    out = df[["date"]].copy()
    out["funding_rate"] = pd.to_numeric(df["funding_rate_avg"], errors="coerce")
    return out


def compute_derived(df):
    """Compute indicators that can be derived from raw data."""
    # Puell Multiple: daily issuance USD / 365-day MA of daily issuance USD
    if "issuance_usd" in df.columns:
        iss = df["issuance_usd"]
        ma365 = iss.rolling(365, min_periods=365).mean()
        df["puell_multiple_computed"] = iss / ma365

    # Hash Ribbons: 30-day MA vs 60-day MA of hash rate
    if "hash_rate" in df.columns:
        hr = df["hash_rate"]
        ma30 = hr.rolling(30, min_periods=30).mean()
        ma60 = hr.rolling(60, min_periods=60).mean()
        # Signal: 1 when 30d MA crosses above 60d MA (recovery), -1 when below (capitulation)
        df["hash_ribbon_signal"] = (ma30 > ma60).astype(float)
        df["hash_ribbon_ratio"] = ma30 / ma60

    # Net exchange flow
    if "exchange_inflow_ntv" in df.columns and "exchange_outflow_ntv" in df.columns:
        df["net_exchange_flow"] = df["exchange_inflow_ntv"] - df["exchange_outflow_ntv"]

    # Realized price from MVRV: realized_price = market_price / mvrv_ratio
    if "mvrv_ratio" in df.columns and "close" in df.columns:
        mvrv = df["mvrv_ratio"]
        df["realized_price_computed"] = df["close"] / mvrv.replace(0, np.nan)

    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading raw data sources...")

    # Base: OHLC
    print("  Loading OHLC...")
    df = load_ohlc()
    print(f"    {len(df)} rows")

    # CoinMetrics
    print("  Loading CoinMetrics...")
    cm = load_coinmetrics()
    if cm is not None:
        df = df.merge(cm, on="date", how="left")
        print(f"    Merged: {list(cm.columns[1:])}")

    # BGeometrics
    print("  Loading BGeometrics...")
    bg = load_bgeometrics()
    if bg is not None:
        df = df.merge(bg, on="date", how="left")
        print(f"    Merged: {list(bg.columns[1:])}")

    # DVOL
    print("  Loading DVOL...")
    dvol = load_dvol()
    if dvol is not None:
        df = df.merge(dvol, on="date", how="left")
        print(f"    Merged: dvol (from {dvol['date'].min().date()})")

    # Funding rates
    print("  Loading funding rates...")
    funding = load_funding()
    if funding is not None:
        df = df.merge(funding, on="date", how="left")
        print(f"    Merged: funding_rate (from {funding['date'].min().date()})")

    # Compute derived indicators
    print("  Computing derived indicators...")
    df = compute_derived(df)

    # Sort and save
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows, {len(df.columns)} columns to {OUTPUT_PATH}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Columns: {list(df.columns)}")

    # Show data availability summary
    print(f"\nData availability:")
    for col in df.columns:
        if col == "date":
            continue
        non_null = df[col].notna().sum()
        first_valid = df[df[col].notna()]["date"].min()
        print(f"  {col:30s}  {non_null:5d} rows  from {first_valid.date() if pd.notna(first_valid) else 'N/A'}")


if __name__ == "__main__":
    main()
