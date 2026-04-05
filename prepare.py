"""
Fixed evaluation harness for BTC bear-market deployment research.
DO NOT MODIFY — this file is read-only for the autoresearch loop.

Provides:
    - Data loading and feature construction
    - Backtest engine (day-by-day spot buy simulation)
    - Scoring function
    - Naive baselines for reference

Usage:
    Imported by train.py. Not run directly.
"""

import os
import math
import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

INITIAL_CASH = 50_000.0
DEFAULT_SPOT_FEE = 0.001       # 10 bps taker fee
MIN_TRADE_USD = 10.0
BTC_GENESIS = pd.Timestamp("2009-01-03")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "btc_features.parquet")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    date: datetime.date
    cash_available: float
    btc_held: float
    day_index: int = 0


@dataclass
class Action:
    spot_buy_usd: float = 0.0


@dataclass
class Scenario:
    name: str
    start_date: datetime.date
    end_date: datetime.date
    description: str


@dataclass
class BacktestResult:
    scenario_name: str
    terminal_btc: float
    total_usd_deployed: float
    btc_per_usd: float
    num_spot_buys: int


@dataclass
class ScoreSummary:
    composite_score: float
    per_scenario: dict

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def build_scenarios():
    """Fixed bear-to-recovery scenarios."""
    return [
        Scenario(
            name="bear_2018",
            start_date=datetime.date(2018, 1, 1),
            end_date=datetime.date(2019, 6, 26),
            description="2018 bear: ATH crash ($17k->$3.1k) through recovery to $13.9k",
        ),
        Scenario(
            name="bear_2022",
            start_date=datetime.date(2021, 11, 11),
            end_date=datetime.date(2023, 9, 30),
            description="2022 bear: ATH ($69k) -> LUNA -> FTX ($15.5k) -> recovery to $27k",
        ),
    ]

# ---------------------------------------------------------------------------
# Data loading and feature construction
# ---------------------------------------------------------------------------

def load_features(path=None):
    """Load raw OHLC data from parquet."""
    if path is None:
        path = DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at {path}. Run: python scripts/build_dataset.py"
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _rolling_slope(series, window):
    """Compute rolling linear regression slope (backward-looking)."""
    def _slope(arr):
        n = len(arr)
        if n < 2 or np.isnan(arr).any():
            return np.nan
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = arr.mean()
        num = ((x - x_mean) * (arr - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        if den == 0:
            return 0.0
        return num / den
    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def construct_features(df):
    """
    Add derived features to raw OHLC data.
    All operations are backward-looking only — no lookahead.
    """
    df = df.copy()
    close = df["close"].astype(float)

    # Moving averages
    df["sma_50"] = close.rolling(50, min_periods=50).mean()
    df["sma_200"] = close.rolling(200, min_periods=200).mean()
    df["ma_200w"] = close.rolling(1400, min_periods=1400).mean()

    # Mayer Multiple
    df["mayer_multiple"] = close / df["sma_200"]

    # Returns
    df["return_1d"] = close.pct_change(1)
    df["return_7d"] = close.pct_change(7)
    df["return_30d"] = close.pct_change(30)

    # Realized volatility (annualized)
    log_ret = np.log(close / close.shift(1))
    df["realized_vol_7d"] = log_ret.rolling(7, min_periods=7).std() * np.sqrt(365)
    df["realized_vol_30d"] = log_ret.rolling(30, min_periods=30).std() * np.sqrt(365)

    # Distance from ATH (expanding max — no lookahead)
    expanding_max = close.expanding(min_periods=1).max()
    df["distance_from_ath"] = (expanding_max - close) / expanding_max

    # Power Law features
    dates_pd = pd.to_datetime(df["date"])
    days_since_genesis = (dates_pd - BTC_GENESIS).dt.days.astype(float)
    df["log_days_since_genesis"] = np.log10(days_since_genesis.clip(lower=1))
    df["log_price"] = np.log10(close.clip(lower=0.01))
    log_d = df["log_days_since_genesis"]
    log_p = df["log_price"]
    cum_n = np.arange(1, len(df) + 1, dtype=float)
    cum_x = log_d.cumsum()
    cum_y = log_p.cumsum()
    cum_xy = (log_d * log_p).cumsum()
    cum_xx = (log_d * log_d).cumsum()
    denom = cum_n * cum_xx - cum_x * cum_x
    slope = np.where(denom != 0, (cum_n * cum_xy - cum_x * cum_y) / denom, 0)
    intercept = np.where(denom != 0, (cum_y - slope * cum_x) / cum_n, 0)
    predicted_log_p = slope * log_d + intercept
    df["power_law_residual"] = log_p - predicted_log_p

    # --- On-chain derived features (if available) ---

    if "nupl" not in df.columns and "nupl_computed" in df.columns:
        df["nupl"] = df["nupl_computed"]
    if "mvrv_z_score" not in df.columns and "mvrv_z_score_computed" in df.columns:
        df["mvrv_z_score"] = df["mvrv_z_score_computed"]

    if "realized_price" in df.columns:
        rp = df["realized_price"].astype(float)
        df["realized_price_ratio"] = close / rp.replace(0, np.nan)
    elif "realized_price_computed" in df.columns:
        rp = df["realized_price_computed"].astype(float)
        df["realized_price_ratio"] = close / rp.replace(0, np.nan)

    if "puell_multiple" not in df.columns and "puell_multiple_computed" in df.columns:
        df["puell_multiple"] = df["puell_multiple_computed"]

    # Percentile ranks (backward-looking)
    pct_cols = ["mayer_multiple", "realized_vol_30d", "distance_from_ath",
                "power_law_residual", "mvrv_ratio", "nupl", "puell_multiple"]
    for col in pct_cols:
        if col in df.columns:
            df[f"{col}_pct_365d"] = df[col].rolling(365, min_periods=90).rank(pct=True)

    # Slopes
    slope_cols = ["mayer_multiple", "distance_from_ath", "mvrv_ratio", "nupl"]
    for col in slope_cols:
        if col in df.columns:
            df[f"{col}_slope_7d"] = _rolling_slope(df[col], 7)
            df[f"{col}_slope_30d"] = _rolling_slope(df[col], 30)

    return df

# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def _validate_action(action, portfolio):
    """Validate and clamp action to enforce buy-only and cash constraints."""
    spot = max(0.0, action.spot_buy_usd)
    spot = min(spot, portfolio.cash_available)
    if spot < MIN_TRADE_USD:
        spot = 0.0
    return Action(spot_buy_usd=spot)


def run_backtest(strategy_fn, scenario, features_df):
    """Run day-by-day backtest of a strategy over a scenario."""
    mask = (features_df["date"] >= scenario.start_date) & (features_df["date"] <= scenario.end_date)
    scenario_df = features_df[mask].copy()

    if scenario_df.empty:
        raise ValueError(f"No data for scenario {scenario.name}")

    all_dates = scenario_df["date"].tolist()
    feature_lookup = {r["date"]: r for r in scenario_df.to_dict("records")}

    portfolio = PortfolioState(
        date=scenario.start_date,
        cash_available=INITIAL_CASH,
        btc_held=0.0,
        day_index=0,
    )

    stats = {"spot_buys": 0, "total_usd_deployed": 0.0}

    for i, date in enumerate(all_dates):
        portfolio.date = date
        portfolio.day_index = i

        features = feature_lookup.get(date)
        if features is None:
            continue

        try:
            action = strategy_fn(features, portfolio)
        except Exception:
            action = Action()

        action = _validate_action(action, portfolio)

        if action.spot_buy_usd > 0:
            btc_price = features["close"]
            if btc_price > 0:
                btc_bought = (action.spot_buy_usd * (1 - DEFAULT_SPOT_FEE)) / btc_price
                portfolio.btc_held += btc_bought
                portfolio.cash_available -= action.spot_buy_usd
                stats["spot_buys"] += 1
                stats["total_usd_deployed"] += action.spot_buy_usd

    terminal_btc = portfolio.btc_held
    total_deployed = stats["total_usd_deployed"]
    btc_per_usd = terminal_btc / total_deployed if total_deployed > 0 else 0.0

    return BacktestResult(
        scenario_name=scenario.name,
        terminal_btc=terminal_btc,
        total_usd_deployed=total_deployed,
        btc_per_usd=btc_per_usd,
        num_spot_buys=stats["spot_buys"],
    )

# ---------------------------------------------------------------------------
# Naive baselines (for reference only, not part of scoring)
# ---------------------------------------------------------------------------

def _run_naive_lump_sum(scenario, features_df):
    """Buy all BTC on day 1."""
    mask = (features_df["date"] >= scenario.start_date) & (features_df["date"] <= scenario.end_date)
    sdf = features_df[mask]
    if sdf.empty:
        return 0.0
    first_close = sdf.iloc[0]["close"]
    if first_close <= 0:
        return 0.0
    return (INITIAL_CASH * (1 - DEFAULT_SPOT_FEE)) / first_close


def _run_naive_dca(scenario, features_df):
    """Buy equal weekly amounts over the period."""
    mask = (features_df["date"] >= scenario.start_date) & (features_df["date"] <= scenario.end_date)
    sdf = features_df[mask]
    if sdf.empty:
        return 0.0
    weekly_rows = sdf.iloc[::7]
    n_weeks = len(weekly_rows)
    if n_weeks == 0:
        return 0.0
    weekly_amount = INITIAL_CASH / n_weeks
    total_btc = 0.0
    for _, row in weekly_rows.iterrows():
        price = row["close"]
        if price > 0:
            total_btc += (weekly_amount * (1 - DEFAULT_SPOT_FEE)) / price
    return total_btc

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results):
    """Score backtest results. Composite = arithmetic mean of terminal BTC."""
    terminal_btcs = [r.terminal_btc for r in results]
    composite = sum(terminal_btcs) / len(terminal_btcs) if terminal_btcs else 0.0
    per_scenario = {r.scenario_name: r for r in results}
    return ScoreSummary(composite_score=composite, per_scenario=per_scenario)


def print_results(summary, scenarios, features_df):
    """Print results in the format expected by program.md."""
    print("---")
    print(f"composite_score:      {summary.composite_score:.6f}")

    for scenario in scenarios:
        result = summary.per_scenario.get(scenario.name)
        if result:
            lump = _run_naive_lump_sum(scenario, features_df)
            dca = _run_naive_dca(scenario, features_df)
            print(f"scenario: {result.scenario_name:20s}  "
                  f"terminal_btc: {result.terminal_btc:.6f}  "
                  f"spot_buys: {result.num_spot_buys}  "
                  f"| baseline lump: {lump:.6f}  dca: {dca:.6f}")
