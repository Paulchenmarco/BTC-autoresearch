"""
One-time data download script. Fetches all raw data and saves to data/raw/.
Run this ONCE, then use build_dataset.py to merge into the parquet.

Sources:
    - Yahoo Finance: BTC-USD daily OHLC
    - CoinMetrics GitHub CSV: MVRV, hash rate, issuance, active addresses
    - BGeometrics API: NUPL, SOPR, Puell Multiple, Realized Price, STH Realized Price
    - Deribit API: DVOL (from 2021-03-24)
    - Binance API: Funding rates (from 2019-09-10)

Usage:
    python scripts/download_data.py              # fetch all sources
    python scripts/download_data.py --skip-bgeometrics  # skip rate-limited source
    python scripts/download_data.py --only bgeometrics   # fetch only BGeometrics (if rate limit reset)
"""

import os
import sys
import time
import json
import argparse
import datetime
import urllib.request
import urllib.error
import csv

import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

# ---------------------------------------------------------------------------
# Yahoo Finance: BTC OHLC
# ---------------------------------------------------------------------------

def fetch_ohlc():
    path = os.path.join(RAW_DIR, "btc_ohlc.csv")
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return

    print("  Fetching BTC-USD daily OHLC from Yahoo Finance...")
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(start="2017-01-01", interval="1d")
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows to {path}")

# ---------------------------------------------------------------------------
# CoinMetrics GitHub CSV
# ---------------------------------------------------------------------------

def fetch_coinmetrics():
    path = os.path.join(RAW_DIR, "coinmetrics_btc.csv")
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return

    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
    print(f"  Fetching CoinMetrics BTC data...")
    urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path)
    print(f"  Saved {len(df)} rows to {path}")
    print(f"  Columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# BGeometrics API (8 req/hour free tier — be patient)
# ---------------------------------------------------------------------------

BGEOMETRICS_METRICS = {
    "nupl": "nupl",
    "sopr": "sopr",
    "puell-multiple": "puell_multiple",
    "realized-price": "realized_price",
    "sth-realized-price": "sth_realized_price",
    "mvrv-z-score": "mvrv_z_score",
}

def _fetch_bgeometrics_metric(metric_slug, output_name):
    """Fetch a single metric from BGeometrics. Returns True if successful."""
    path = os.path.join(RAW_DIR, f"bgeometrics_{output_name}.csv")
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return True

    url = f"https://bitcoin-data.com/api/v1/{metric_slug}"
    print(f"  Fetching BGeometrics {metric_slug}...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "btc-autoresearch/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        if "error" in data:
            print(f"  [error] {data['error'].get('message', data['error'])}")
            return False

        # BGeometrics returns various formats — handle common patterns
        rows = []
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "data" in data:
            rows = data["data"]
        elif isinstance(data, dict) and "chart" in data:
            rows = data["chart"]

        if not rows:
            print(f"  [error] No data returned for {metric_slug}")
            print(f"  Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            # Save raw response for debugging
            debug_path = os.path.join(RAW_DIR, f"bgeometrics_{output_name}_debug.json")
            with open(debug_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved raw response to {debug_path}")
            return False

        # Write CSV
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", output_name])
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    writer.writerow([row[0], row[1]])
                elif isinstance(row, dict):
                    # Try common key patterns
                    date_val = row.get("date") or row.get("t") or row.get("time") or row.get("x")
                    metric_val = row.get("value") or row.get("v") or row.get("y") or row.get(output_name)
                    if date_val and metric_val is not None:
                        writer.writerow([date_val, metric_val])

        df = pd.read_csv(path)
        print(f"  Saved {len(df)} rows to {path}")
        return True

    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"  [rate limited] BGeometrics hourly limit hit. Wait and retry with:")
            print(f"    python scripts/download_data.py --only bgeometrics")
        else:
            print(f"  [error] HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"  [error] {e}")
        return False


def fetch_bgeometrics():
    """Fetch all BGeometrics metrics with delays between requests."""
    remaining = []
    for slug, name in BGEOMETRICS_METRICS.items():
        path = os.path.join(RAW_DIR, f"bgeometrics_{name}.csv")
        if not os.path.exists(path):
            remaining.append((slug, name))

    if not remaining:
        print("  [skip] All BGeometrics files already exist")
        return

    print(f"  Need to fetch {len(remaining)} BGeometrics metrics")
    print(f"  Free tier: 8 requests/hour — this may take pauses")

    for i, (slug, name) in enumerate(remaining):
        success = _fetch_bgeometrics_metric(slug, name)
        if not success:
            print(f"  Stopping BGeometrics fetches. Run again later for remaining metrics.")
            return
        # Wait between requests to avoid rate limiting
        if i < len(remaining) - 1:
            print(f"  Waiting 60s before next request ({len(remaining) - i - 1} remaining)...")
            time.sleep(60)

# ---------------------------------------------------------------------------
# Deribit: DVOL (Bitcoin Volatility Index)
# ---------------------------------------------------------------------------

def fetch_dvol():
    path = os.path.join(RAW_DIR, "deribit_dvol.csv")
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return

    print("  Fetching Deribit DVOL...")
    all_points = []
    # DVOL starts 2021-03-24. Fetch in chunks (API returns max 1000 per request)
    end_ts = int(datetime.datetime.now().timestamp() * 1000)
    start_ts = int(datetime.datetime(2021, 3, 1).timestamp() * 1000)

    while True:
        url = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data"
               f"?currency=BTC&resolution=86400"
               f"&start_timestamp={start_ts}&end_timestamp={end_ts}")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            points = data.get("result", {}).get("data", [])
            continuation = data.get("result", {}).get("continuation")

            if not points:
                break

            all_points.extend(points)
            print(f"    Fetched {len(points)} points (total: {len(all_points)})")

            if continuation and continuation > start_ts:
                end_ts = continuation
                time.sleep(0.5)
            else:
                break
        except Exception as e:
            print(f"  [error] {e}")
            break

    if all_points:
        # Sort by timestamp and deduplicate
        all_points.sort(key=lambda x: x[0])
        seen = set()
        unique = []
        for p in all_points:
            if p[0] not in seen:
                seen.add(p[0])
                unique.append(p)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "dvol_open", "dvol_high", "dvol_low", "dvol_close"])
            for p in unique:
                dt = datetime.datetime.utcfromtimestamp(p[0] / 1000).strftime("%Y-%m-%d")
                writer.writerow([dt, p[1], p[2], p[3], p[4]])

        print(f"  Saved {len(unique)} rows to {path}")

# ---------------------------------------------------------------------------
# Binance: Funding Rates
# ---------------------------------------------------------------------------

def fetch_funding_rates():
    path = os.path.join(RAW_DIR, "binance_funding.csv")
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return

    print("  Fetching Binance BTCUSDT funding rates...")
    all_records = []
    start_ts = int(datetime.datetime(2019, 9, 1).timestamp() * 1000)

    while True:
        url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
               f"?symbol=BTCUSDT&startTime={start_ts}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())

            if not data:
                break

            all_records.extend(data)
            last_ts = data[-1]["fundingTime"]
            print(f"    Fetched {len(data)} records (total: {len(all_records)})")

            if len(data) < 1000:
                break

            start_ts = last_ts + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"  [error] {e}")
            break

    if all_records:
        # Aggregate to daily average funding rate
        daily = {}
        for r in all_records:
            dt = datetime.datetime.utcfromtimestamp(r["fundingTime"] / 1000).strftime("%Y-%m-%d")
            rate = float(r["fundingRate"])
            if dt not in daily:
                daily[dt] = []
            daily[dt].append(rate)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "funding_rate_avg", "funding_rate_count"])
            for dt in sorted(daily.keys()):
                rates = daily[dt]
                writer.writerow([dt, sum(rates) / len(rates), len(rates)])

        print(f"  Saved {len(daily)} daily rows to {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download raw BTC data")
    parser.add_argument("--skip-bgeometrics", action="store_true",
                        help="Skip BGeometrics (rate-limited, fetch separately)")
    parser.add_argument("--only", type=str, default=None,
                        help="Fetch only this source: ohlc, coinmetrics, bgeometrics, dvol, funding")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    sources = {
        "ohlc": ("Yahoo Finance OHLC", fetch_ohlc),
        "coinmetrics": ("CoinMetrics GitHub CSV", fetch_coinmetrics),
        "bgeometrics": ("BGeometrics on-chain", fetch_bgeometrics),
        "dvol": ("Deribit DVOL", fetch_dvol),
        "funding": ("Binance funding rates", fetch_funding_rates),
    }

    if args.only:
        if args.only not in sources:
            print(f"Unknown source: {args.only}. Available: {list(sources.keys())}")
            sys.exit(1)
        name, fn = sources[args.only]
        print(f"[{args.only}] {name}")
        fn()
        return

    for key, (name, fn) in sources.items():
        if key == "bgeometrics" and args.skip_bgeometrics:
            print(f"[{key}] {name} — SKIPPED (use --only bgeometrics later)")
            continue
        print(f"[{key}] {name}")
        fn()
        print()

    print("Done. Now run: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
