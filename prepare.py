"""
Fixed evaluation harness for BTC bear-market deployment research.
DO NOT MODIFY — this file is read-only for the autoresearch loop.

Provides:
    - Data loading and feature construction
    - Backtest engine (day-by-day simulation)
    - CSP (cash-secured put) settlement simulation
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
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

INITIAL_CASH = 50_000.0
DEFAULT_SPOT_FEE = 0.001       # 10 bps taker fee
DEFAULT_CSP_FEE = 0.0003       # Deribit maker fee
MIN_TRADE_USD = 10.0
BTC_GENESIS = pd.Timestamp("2009-01-03")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "btc_features.parquet")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpenCSP:
    sell_date: datetime.date
    expiry_date: datetime.date
    strike: float
    notional: float
    premium_usd: float
    delta_at_sale: float


@dataclass
class OpenPutSpread:
    """Bull put spread: sell put at short_strike, buy put at long_strike (lower)."""
    sell_date: datetime.date
    expiry_date: datetime.date
    short_strike: float     # higher strike (sold)
    long_strike: float      # lower strike (bought)
    max_loss: float         # (short_strike - long_strike) * contracts - net premium
    collateral: float       # cash reserved = max_loss
    net_premium_usd: float  # premium received - premium paid
    short_delta: float


@dataclass
class OpenCallBuy:
    """Long call option position."""
    buy_date: datetime.date
    expiry_date: datetime.date
    strike: float
    premium_paid: float
    delta_at_buy: float
    num_contracts_btc: float  # number of BTC-denominated contracts


@dataclass
class PortfolioState:
    date: datetime.date
    cash_available: float
    cash_reserved: float
    btc_held: float
    total_premium_earned: float
    open_csps: list
    open_spreads: list = field(default_factory=list)
    open_calls: list = field(default_factory=list)
    day_index: int = 0


@dataclass
class CSPOrder:
    delta: float
    dte: int
    notional_usd: float


@dataclass
class PutSpreadOrder:
    """Bull put spread order."""
    short_delta: float      # delta of the put we sell (e.g. 0.20)
    long_delta: float       # delta of the put we buy (e.g. 0.10, further OTM)
    dte: int
    num_spreads_usd: float  # dollar amount controlling spread sizing


@dataclass
class CallBuyOrder:
    """Buy a call option."""
    delta: float            # target delta (e.g. 0.30 for OTM call)
    dte: int
    premium_budget: float   # max USD to spend on call premium


@dataclass
class Action:
    spot_buy_usd: float = 0.0
    csp_sells: list = field(default_factory=list)
    spread_sells: list = field(default_factory=list)
    call_buys: list = field(default_factory=list)
    close_csp_indices: list = field(default_factory=list)
    close_spread_indices: list = field(default_factory=list)


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
    total_premium_earned: float
    num_spot_buys: int
    num_csps_sold: int
    num_csps_assigned: int
    num_csps_expired_otm: int
    num_csps_closed_early: int
    num_spreads_sold: int = 0
    num_spreads_assigned: int = 0
    num_spreads_expired_otm: int = 0
    num_calls_bought: int = 0
    num_calls_itm: int = 0
    num_calls_expired_otm: int = 0
    call_profit_usd: float = 0.0


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
    df["ma_200w"] = close.rolling(1400, min_periods=1400).mean()  # ~200 weeks

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
    # Linear regression on log-log gives power law; residual = deviation from trend
    # Use expanding regression to avoid lookahead
    log_d = df["log_days_since_genesis"]
    log_p = df["log_price"]
    # Rolling power law residual (expanding OLS)
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

    # --- On-chain derived features (if available from dataset) ---

    # Use best available NUPL (BGeometrics if fetched, otherwise computed)
    if "nupl" not in df.columns and "nupl_computed" in df.columns:
        df["nupl"] = df["nupl_computed"]

    # Use best available MVRV Z-Score
    if "mvrv_z_score" not in df.columns and "mvrv_z_score_computed" in df.columns:
        df["mvrv_z_score"] = df["mvrv_z_score_computed"]

    # Realized price ratio: close / realized_price
    if "realized_price" in df.columns:
        rp = df["realized_price"].astype(float)
        df["realized_price_ratio"] = close / rp.replace(0, np.nan)
    elif "realized_price_computed" in df.columns:
        rp = df["realized_price_computed"].astype(float)
        df["realized_price_ratio"] = close / rp.replace(0, np.nan)

    # STH realized price ratio
    if "sth_realized_price" in df.columns:
        sth_rp = df["sth_realized_price"].astype(float)
        df["sth_rp_ratio"] = close / sth_rp.replace(0, np.nan)

    # Use best available Puell Multiple
    if "puell_multiple" not in df.columns and "puell_multiple_computed" in df.columns:
        df["puell_multiple"] = df["puell_multiple_computed"]

    # Funding rate features (if available)
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        df["funding_rate_7d_avg"] = fr.rolling(7, min_periods=1).mean()
        df["funding_rate_30d_avg"] = fr.rolling(30, min_periods=7).mean()

    # DVOL features (if available)
    if "dvol" in df.columns:
        dvol = df["dvol"].astype(float)
        df["dvol_pct_90d"] = dvol.rolling(90, min_periods=30).rank(pct=True)

    # --- Percentile ranks (backward-looking) ---
    pct_cols = ["mayer_multiple", "realized_vol_30d", "distance_from_ath",
                "power_law_residual", "mvrv_ratio", "nupl", "sopr", "puell_multiple"]
    for col in pct_cols:
        if col in df.columns:
            df[f"{col}_pct_365d"] = df[col].rolling(365, min_periods=90).rank(pct=True)

    # --- Slopes ---
    slope_cols = ["mayer_multiple", "distance_from_ath", "mvrv_ratio", "nupl"]
    for col in slope_cols:
        if col in df.columns:
            df[f"{col}_slope_7d"] = _rolling_slope(df[col], 7)
            df[f"{col}_slope_30d"] = _rolling_slope(df[col], 30)

    return df

# ---------------------------------------------------------------------------
# CSP premium model (fixed — agent cannot modify)
# ---------------------------------------------------------------------------

def _delta_to_strike(delta, spot, iv, dte):
    """Convert target put delta (absolute value, e.g. 0.15) to OTM strike price.

    For a BS put with r=0: delta_put = N(d1) - 1 = -N(-d1)
    So |delta_put| = N(-d1), meaning -d1 = norm.ppf(|delta|)
    Solving for K: K = S * exp(d1 * sigma * sqrt(T) + 0.5 * sigma^2 * T)
    where d1 = -norm.ppf(delta).
    """
    if dte <= 0 or iv <= 0:
        return spot * (1 - delta)
    T = dte / 365.0
    sqrt_T = math.sqrt(T)
    # |put delta| = N(-d1), so -d1 = norm.ppf(delta), d1 = -norm.ppf(delta)
    d1 = -norm.ppf(delta)
    # From d1 = [ln(S/K) + 0.5*iv^2*T] / (iv*sqrt(T))
    # => ln(K/S) = -d1 * iv * sqrt(T) + 0.5 * iv^2 * T
    # But we want K < S for OTM put, so:
    # K = S * exp(-d1 * iv * sqrt(T) + 0.5 * iv^2 * T)
    # Wait, let me derive carefully:
    # d1 = [ln(S/K) + 0.5*σ²T] / (σ√T)
    # d1 * σ√T = ln(S/K) + 0.5*σ²T
    # ln(S/K) = d1*σ√T - 0.5*σ²T
    # ln(K/S) = -d1*σ√T + 0.5*σ²T
    # K = S * exp(-d1*σ√T + 0.5*σ²T)
    log_moneyness = -d1 * iv * sqrt_T + 0.5 * iv * iv * T
    strike = spot * math.exp(log_moneyness)
    # Ensure strike is OTM (below spot) and not absurdly far
    strike = min(strike, spot * 0.99)  # cap at 1% OTM minimum
    strike = max(strike, spot * 0.30)  # floor at 70% OTM
    return strike


def estimate_iv(features, delta, dte):
    """
    Estimate implied volatility for a BTC put option.
    Uses DVOL (Deribit vol index) when available, otherwise realized vol with premium.
    Applies term structure and OTM skew adjustments.
    """
    # Base ATM IV: prefer DVOL (actual market IV), fall back to realized vol
    dvol = features.get("dvol")
    rv30 = features.get("realized_vol_30d", 0.60)
    if rv30 is None or (isinstance(rv30, float) and math.isnan(rv30)):
        rv30 = 0.60

    if dvol is not None and isinstance(dvol, (int, float)) and not math.isnan(dvol) and dvol > 0:
        # DVOL is annualized percentage, convert to decimal
        atm_iv = dvol / 100.0
    else:
        # Fall back to realized vol with IV premium
        atm_iv = rv30 * 1.15

    # Term structure: DVOL is calibrated to 30-day.
    # Short-dated (<30d): IV is typically higher (more gamma, event risk)
    # Long-dated (>30d): IV is typically lower (mean reversion of vol)
    if dte < 14:
        term_adj = 1.10
    elif dte < 30:
        term_adj = 1.05
    elif dte < 60:
        term_adj = 1.00
    else:
        term_adj = 0.95  # long-dated vol slightly lower

    # OTM put skew: BTC puts have significant skew
    # 0.50 delta (ATM) = no adjustment
    # 0.25 delta = ~5-10% higher IV
    # 0.15 delta = ~10-20% higher IV
    # 0.10 delta = ~15-25% higher IV
    skew_adj = 1.0 + 0.4 * max(0.50 - delta, 0)

    iv = atm_iv * term_adj * skew_adj
    return max(iv, 0.15)  # floor at 15%


def estimate_premium(csp_order, features):
    """
    Estimate put premium using Black-Scholes.
    Returns premium in USD after fees.
    """
    spot = features.get("close", 0)
    if spot <= 0:
        return 0.0

    delta = csp_order.delta
    dte = csp_order.dte
    notional = csp_order.notional_usd

    if dte <= 0 or notional <= 0:
        return 0.0

    iv = estimate_iv(features, delta, dte)
    T = dte / 365.0
    sqrt_T = math.sqrt(T)

    strike = _delta_to_strike(delta, spot, iv, dte)

    # Black-Scholes put price (r=0 for crypto)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    put_price = strike * norm.cdf(-d2) - spot * norm.cdf(-d1)

    # Scale to notional
    num_contracts = notional / strike
    premium_usd = put_price * num_contracts

    # Subtract fees
    premium_usd *= (1 - DEFAULT_CSP_FEE)

    return max(premium_usd, 0.0), strike


def _bs_put_price(spot, strike, iv, T):
    """Black-Scholes put price with r=0."""
    if T <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return max(strike - spot, 0) if spot > 0 and strike > 0 else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    return strike * norm.cdf(-d2) - spot * norm.cdf(-d1)


def _bs_call_price(spot, strike, iv, T):
    """Black-Scholes call price with r=0."""
    if T <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0) if spot > 0 and strike > 0 else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)


def _delta_to_call_strike(delta, spot, iv, dte):
    """Convert target call delta (e.g. 0.30) to OTM call strike (above spot)."""
    if dte <= 0 or iv <= 0:
        return spot * (1 + delta)
    T = dte / 365.0
    sqrt_T = math.sqrt(T)
    # For a call: delta = N(d1), so d1 = norm.ppf(delta)
    d1 = norm.ppf(delta)
    log_moneyness = -d1 * iv * sqrt_T + 0.5 * iv * iv * T
    strike = spot * math.exp(log_moneyness)
    strike = max(strike, spot * 1.01)  # at least 1% OTM
    strike = min(strike, spot * 3.0)   # cap at 3x spot
    return strike


def estimate_call_premium(order, features):
    """Estimate call option premium. Returns (premium_usd, strike)."""
    spot = features.get("close", 0)
    if spot <= 0 or order.dte <= 0:
        return 0.0, spot

    # Use call delta for IV (calls have less skew than puts)
    iv = estimate_iv(features, 0.50, order.dte)  # ATM IV as base
    # Calls have less skew — slight reduction for OTM calls
    call_skew = 1.0 - 0.1 * max(0.50 - order.delta, 0)
    iv *= call_skew

    T = order.dte / 365.0
    strike = _delta_to_call_strike(order.delta, spot, iv, order.dte)
    call_price = _bs_call_price(spot, strike, iv, T)

    # For BTC options on Deribit, 1 contract = 1 BTC
    # premium_budget buys premium_budget / (call_price * spot_in_usd) contracts
    # But call_price is in USD per 1 BTC notional
    premium_per_contract = call_price  # USD per 1 BTC contract
    if premium_per_contract <= 0:
        return 0.0, strike

    premium_per_contract *= (1 + DEFAULT_CSP_FEE)  # buyer pays fee
    return premium_per_contract, strike


def estimate_spread_premium(order, features):
    """Estimate bull put spread net premium. Returns (net_premium, short_strike, long_strike)."""
    spot = features.get("close", 0)
    if spot <= 0 or order.dte <= 0:
        return 0.0, spot, spot

    T = order.dte / 365.0

    # Price the short put (higher strike, we sell)
    iv_short = estimate_iv(features, order.short_delta, order.dte)
    short_strike = _delta_to_strike(order.short_delta, spot, iv_short, order.dte)
    short_put_price = _bs_put_price(spot, short_strike, iv_short, T)

    # Price the long put (lower strike, we buy)
    iv_long = estimate_iv(features, order.long_delta, order.dte)
    long_strike = _delta_to_strike(order.long_delta, spot, iv_long, order.dte)
    long_put_price = _bs_put_price(spot, long_strike, iv_long, T)

    # Ensure short strike > long strike
    if short_strike <= long_strike:
        return 0.0, short_strike, long_strike

    # Net premium = premium received - premium paid
    net_premium = short_put_price - long_put_price
    # Scale to notional
    num_contracts = order.num_spreads_usd / (short_strike - long_strike)
    net_premium_usd = net_premium * num_contracts
    net_premium_usd *= (1 - DEFAULT_CSP_FEE)  # net of fees

    return max(net_premium_usd, 0.0), short_strike, long_strike


def reprice_csp(csp, features):
    """
    Reprice an open CSP at current market conditions.
    Returns the current cost to buy back the put (in USD).
    Used by the strategy to decide whether to take profit.
    """
    spot = features.get("close", 0)
    if spot <= 0:
        return csp.premium_usd  # can't reprice, assume no change

    # Calculate remaining DTE
    current_date = features.get("date")
    if current_date is None:
        return csp.premium_usd

    if isinstance(current_date, datetime.date):
        remaining_dte = (csp.expiry_date - current_date).days
    else:
        return csp.premium_usd

    if remaining_dte <= 0:
        # At or past expiry — value is intrinsic only
        intrinsic = max(csp.strike - spot, 0)
        return intrinsic * (csp.notional / csp.strike)

    iv = estimate_iv(features, csp.delta_at_sale, remaining_dte)
    T = remaining_dte / 365.0
    sqrt_T = math.sqrt(T)

    d1 = (math.log(spot / csp.strike) + 0.5 * iv * iv * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T
    put_price = csp.strike * norm.cdf(-d2) - spot * norm.cdf(-d1)

    num_contracts = csp.notional / csp.strike
    current_value = put_price * num_contracts
    return max(current_value, 0.0)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def _settle_expired_csps(portfolio, date, features_df, stats):
    """Settle CSPs that have expired. Mutates portfolio in place."""
    still_open = []
    for csp in portfolio.open_csps:
        if date >= csp.expiry_date:
            # Find close price at expiry
            expiry_rows = [r for r in features_df if r["date"] == csp.expiry_date]
            if not expiry_rows:
                # Use closest prior date
                prior = [r for r in features_df if r["date"] <= csp.expiry_date]
                if prior:
                    expiry_rows = [prior[-1]]
                else:
                    # Can't settle, free collateral
                    portfolio.cash_available += csp.notional
                    portfolio.cash_reserved -= csp.notional
                    stats["csps_expired_otm"] += 1
                    continue

            spot_at_expiry = expiry_rows[0]["close"]

            portfolio.cash_reserved -= csp.notional

            if spot_at_expiry < csp.strike:
                # ITM: cash-settled loss, then immediately buy spot BTC
                loss = (csp.strike - spot_at_expiry) * (csp.notional / csp.strike)
                net_cash = csp.notional - loss
                if net_cash > 0:
                    btc_bought = (net_cash * (1 - DEFAULT_SPOT_FEE)) / spot_at_expiry
                    portfolio.btc_held += btc_bought
                stats["csps_assigned"] += 1
            else:
                # OTM: collateral freed, premium already credited
                portfolio.cash_available += csp.notional
                stats["csps_expired_otm"] += 1
        else:
            still_open.append(csp)
    portfolio.open_csps = still_open


def _settle_expired_spreads(portfolio, date, features_df, stats):
    """Settle expired bull put spreads."""
    still_open = []
    for spread in portfolio.open_spreads:
        if date >= spread.expiry_date:
            expiry_rows = [r for r in features_df if r["date"] == spread.expiry_date]
            if not expiry_rows:
                prior = [r for r in features_df if r["date"] <= spread.expiry_date]
                if prior:
                    expiry_rows = [prior[-1]]
                else:
                    portfolio.cash_available += spread.collateral
                    portfolio.cash_reserved -= spread.collateral
                    stats["spreads_expired_otm"] += 1
                    continue

            spot = expiry_rows[0]["close"]
            portfolio.cash_reserved -= spread.collateral

            if spot >= spread.short_strike:
                # Both OTM: keep full collateral + premium already credited
                portfolio.cash_available += spread.collateral
                stats["spreads_expired_otm"] += 1
            elif spot <= spread.long_strike:
                # Both ITM: max loss. Collateral covers it. Buy BTC with whatever remains.
                # Loss = (short_strike - long_strike) * contracts
                # But collateral = max_loss, so no cash returned from spread itself
                # Buy BTC with the net (collateral was already reserved from max_loss)
                stats["spreads_assigned"] += 1
            else:
                # Partial loss: spot between strikes
                loss_per_btc = spread.short_strike - spot
                spread_width = spread.short_strike - spread.long_strike
                loss_frac = loss_per_btc / spread_width
                actual_loss = spread.collateral * loss_frac
                returned = spread.collateral - actual_loss
                portfolio.cash_available += returned
                stats["spreads_assigned"] += 1
        else:
            still_open.append(spread)
    portfolio.open_spreads = still_open


def _settle_expired_calls(portfolio, date, features_df, stats):
    """Settle expired call options. If ITM, convert profit to BTC."""
    still_open = []
    for call in portfolio.open_calls:
        if date >= call.expiry_date:
            expiry_rows = [r for r in features_df if r["date"] == call.expiry_date]
            if not expiry_rows:
                prior = [r for r in features_df if r["date"] <= call.expiry_date]
                if prior:
                    expiry_rows = [prior[-1]]
                else:
                    stats["calls_expired_otm"] += 1
                    continue

            spot = expiry_rows[0]["close"]

            if spot > call.strike:
                # ITM: profit = (spot - strike) * num_contracts
                profit_usd = (spot - call.strike) * call.num_contracts_btc
                # Convert profit to BTC (buy-only philosophy)
                btc_from_calls = (profit_usd * (1 - DEFAULT_SPOT_FEE)) / spot
                portfolio.btc_held += btc_from_calls
                stats["calls_itm"] += 1
                stats["call_profit_usd"] += profit_usd
            else:
                # OTM: call expires worthless, premium already paid
                stats["calls_expired_otm"] += 1
        else:
            still_open.append(call)
    portfolio.open_calls = still_open


def _validate_action(action, portfolio):
    """Validate and clamp action to enforce buy-only and cash constraints."""
    available = portfolio.cash_available

    # Clamp spot buy
    spot = max(0.0, action.spot_buy_usd)
    spot = min(spot, available)
    if spot < MIN_TRADE_USD:
        spot = 0.0
    remaining = available - spot

    # Clamp CSP orders
    valid_csps = []
    for csp_order in action.csp_sells:
        notional = max(0.0, csp_order.notional_usd)
        notional = min(notional, remaining)
        if notional < MIN_TRADE_USD:
            continue
        delta = max(0.05, min(0.50, csp_order.delta))
        dte = max(1, min(90, csp_order.dte))
        valid_csps.append(CSPOrder(delta=delta, dte=dte, notional_usd=notional))
        remaining -= notional

    return Action(spot_buy_usd=spot, csp_sells=valid_csps,
                  spread_sells=action.spread_sells, call_buys=action.call_buys,
                  close_csp_indices=action.close_csp_indices,
                  close_spread_indices=action.close_spread_indices)


def run_backtest(strategy_fn, scenario, features_df):
    """
    Run day-by-day backtest of a strategy over a scenario.

    Args:
        strategy_fn: callable(features_dict, PortfolioState) -> Action
        scenario: Scenario
        features_df: DataFrame with columns including 'date', 'close', and derived features

    Returns:
        BacktestResult
    """
    # Pre-filter and convert features to list of dicts for fast lookup
    mask = (features_df["date"] >= scenario.start_date) & (features_df["date"] <= scenario.end_date)
    scenario_df = features_df[mask].copy()

    if scenario_df.empty:
        raise ValueError(f"No data for scenario {scenario.name} ({scenario.start_date} to {scenario.end_date})")

    # Also get prior data for features that might need history
    all_dates = scenario_df["date"].tolist()
    feature_records = scenario_df.to_dict("records")

    # Build lookup: date -> feature dict
    feature_lookup = {r["date"]: r for r in feature_records}

    # For CSP settlement, we need access to features by date
    all_feature_records = features_df.to_dict("records")

    portfolio = PortfolioState(
        date=scenario.start_date,
        cash_available=INITIAL_CASH,
        cash_reserved=0.0,
        btc_held=0.0,
        total_premium_earned=0.0,
        open_csps=[],
        day_index=0,
    )

    stats = {
        "spot_buys": 0,
        "csps_sold": 0,
        "csps_assigned": 0,
        "csps_expired_otm": 0,
        "csps_closed_early": 0,
        "spreads_sold": 0,
        "spreads_assigned": 0,
        "spreads_expired_otm": 0,
        "calls_bought": 0,
        "calls_itm": 0,
        "calls_expired_otm": 0,
        "call_profit_usd": 0.0,
        "total_usd_deployed": 0.0,
    }

    for i, date in enumerate(all_dates):
        portfolio.date = date
        portfolio.day_index = i

        # 1. Settle expired positions
        _settle_expired_csps(portfolio, date, all_feature_records, stats)
        _settle_expired_spreads(portfolio, date, all_feature_records, stats)
        _settle_expired_calls(portfolio, date, all_feature_records, stats)

        # 2. Get features (no lookahead — only data up to this date)
        features = feature_lookup.get(date)
        if features is None:
            continue

        # 3. Call strategy
        try:
            action = strategy_fn(features, portfolio)
        except Exception:
            action = Action()

        # 4. Validate
        action = _validate_action(action, portfolio)

        # 5. Close CSPs early (take profit / cut loss — strategy decision)
        if action.close_csp_indices:
            # Sort descending so removal doesn't shift indices
            for idx in sorted(action.close_csp_indices, reverse=True):
                if 0 <= idx < len(portfolio.open_csps):
                    csp = portfolio.open_csps[idx]
                    # Cost to buy back = current market value of the put
                    buyback_cost = reprice_csp(csp, features)
                    # Pay buyback cost + fee
                    buyback_cost *= (1 + DEFAULT_CSP_FEE)
                    # Free collateral, deduct buyback cost
                    portfolio.cash_reserved -= csp.notional
                    portfolio.cash_available += csp.notional - buyback_cost
                    portfolio.open_csps.pop(idx)
                    stats["csps_closed_early"] += 1

        if action.spot_buy_usd > 0:
            btc_price = features["close"]
            if btc_price > 0:
                btc_bought = (action.spot_buy_usd * (1 - DEFAULT_SPOT_FEE)) / btc_price
                portfolio.btc_held += btc_bought
                portfolio.cash_available -= action.spot_buy_usd
                stats["spot_buys"] += 1
                stats["total_usd_deployed"] += action.spot_buy_usd

        # 7. Execute CSP sells
        for csp_order in action.csp_sells:
            result = estimate_premium(csp_order, features)
            if isinstance(result, tuple):
                premium, strike = result
            else:
                continue

            if premium <= 0:
                continue

            expiry = date + datetime.timedelta(days=csp_order.dte)

            open_csp = OpenCSP(
                sell_date=date,
                expiry_date=expiry,
                strike=strike,
                notional=csp_order.notional_usd,
                premium_usd=premium,
                delta_at_sale=csp_order.delta,
            )
            portfolio.open_csps.append(open_csp)
            portfolio.cash_reserved += csp_order.notional_usd
            portfolio.cash_available -= csp_order.notional_usd
            portfolio.cash_available += premium
            portfolio.total_premium_earned += premium
            stats["csps_sold"] += 1

        # 8. Execute put spread sells
        for spread_order in action.spread_sells:
            result = estimate_spread_premium(spread_order, features)
            if not isinstance(result, tuple) or len(result) != 3:
                continue
            net_premium, short_strike, long_strike = result
            if net_premium <= 0 or short_strike <= long_strike:
                continue

            spread_width = short_strike - long_strike
            num_contracts = spread_order.num_spreads_usd / spread_width
            max_loss = spread_width * num_contracts - net_premium
            max_loss = max(max_loss, 0)

            if max_loss > portfolio.cash_available:
                continue

            expiry = date + datetime.timedelta(days=spread_order.dte)
            portfolio.open_spreads.append(OpenPutSpread(
                sell_date=date, expiry_date=expiry,
                short_strike=short_strike, long_strike=long_strike,
                max_loss=max_loss, collateral=max_loss,
                net_premium_usd=net_premium, short_delta=spread_order.short_delta,
            ))
            portfolio.cash_reserved += max_loss
            portfolio.cash_available -= max_loss
            portfolio.cash_available += net_premium
            portfolio.total_premium_earned += net_premium
            stats["spreads_sold"] += 1

        # 9. Execute call buys
        for call_order in action.call_buys:
            result = estimate_call_premium(call_order, features)
            if not isinstance(result, tuple):
                continue
            premium_per_contract, strike = result
            if premium_per_contract <= 0 or call_order.premium_budget <= 0:
                continue

            budget = min(call_order.premium_budget, portfolio.cash_available)
            if budget < MIN_TRADE_USD:
                continue

            num_contracts = budget / premium_per_contract
            portfolio.open_calls.append(OpenCallBuy(
                buy_date=date, expiry_date=date + datetime.timedelta(days=call_order.dte),
                strike=strike, premium_paid=budget,
                delta_at_buy=call_order.delta, num_contracts_btc=num_contracts,
            ))
            portfolio.cash_available -= budget
            stats["calls_bought"] += 1

    # Force-settle remaining positions at scenario end
    far_future = scenario.end_date + datetime.timedelta(days=365)
    _settle_expired_csps(portfolio, far_future, all_feature_records, stats)
    _settle_expired_spreads(portfolio, far_future, all_feature_records, stats)
    _settle_expired_calls(portfolio, far_future, all_feature_records, stats)

    # Any remaining reserved cash becomes available (safety)
    portfolio.cash_available += portfolio.cash_reserved
    portfolio.cash_reserved = 0.0

    terminal_btc = portfolio.btc_held
    total_deployed = stats["total_usd_deployed"]
    btc_per_usd = terminal_btc / total_deployed if total_deployed > 0 else 0.0

    return BacktestResult(
        scenario_name=scenario.name,
        terminal_btc=terminal_btc,
        total_usd_deployed=total_deployed,
        btc_per_usd=btc_per_usd,
        total_premium_earned=portfolio.total_premium_earned,
        num_spot_buys=stats["spot_buys"],
        num_csps_sold=stats["csps_sold"],
        num_csps_assigned=stats["csps_assigned"],
        num_csps_expired_otm=stats["csps_expired_otm"],
        num_csps_closed_early=stats["csps_closed_early"],
        num_spreads_sold=stats["spreads_sold"],
        num_spreads_assigned=stats["spreads_assigned"],
        num_spreads_expired_otm=stats["spreads_expired_otm"],
        num_calls_bought=stats["calls_bought"],
        num_calls_itm=stats["calls_itm"],
        num_calls_expired_otm=stats["calls_expired_otm"],
        call_profit_usd=stats["call_profit_usd"],
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

    # Weekly buys (every 7th trading day)
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
    """
    Score backtest results across scenarios.
    Composite score = arithmetic mean of terminal BTC.
    """
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
            opts = (f"csps: {result.num_csps_sold} "
                   f"(a:{result.num_csps_assigned} o:{result.num_csps_expired_otm} c:{result.num_csps_closed_early})")
            if result.num_spreads_sold > 0:
                opts += (f"  spreads: {result.num_spreads_sold} "
                        f"(a:{result.num_spreads_assigned} o:{result.num_spreads_expired_otm})")
            if result.num_calls_bought > 0:
                opts += (f"  calls: {result.num_calls_bought} "
                        f"(itm:{result.num_calls_itm} otm:{result.num_calls_expired_otm} "
                        f"profit:${result.call_profit_usd:.0f})")
            print(f"scenario: {result.scenario_name:20s}  "
                  f"terminal_btc: {result.terminal_btc:.6f}  "
                  f"premium: ${result.total_premium_earned:.2f}  "
                  f"spot_buys: {result.num_spot_buys}  "
                  f"{opts}  "
                  f"| baseline lump: {lump:.6f}  dca: {dca:.6f}")
