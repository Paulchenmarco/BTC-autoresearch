"""
BTC bear-market deployment strategy. Single-file, agent-editable.
This is the ONLY file the autoresearch loop modifies.

Usage: python train.py
"""

import math

from prepare import (
    load_features, construct_features, build_scenarios,
    run_backtest, score_results, print_results,
    PortfolioState, Action,
)

# ---------------------------------------------------------------------------
# Strategy parameters (edit these)
# ---------------------------------------------------------------------------

# MVRV gate and depth scaling for spot buys
MVRV_THRESHOLD = 0.80
DEPTH_BASE = 0.10
DEPTH_MULT = 11.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return float(val)

# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def decide_action(features, portfolio):
    cash = portfolio.cash_available
    if cash < 10:
        return Action()

    mvrv = _safe(features.get("mvrv_ratio"), 1.0)

    spot_buy = 0.0

    if mvrv < MVRV_THRESHOLD:
        depth = (MVRV_THRESHOLD - mvrv) / MVRV_THRESHOLD
        frac = DEPTH_BASE + depth * DEPTH_MULT
        frac = min(frac, 1.0)
        spot_buy = cash * frac
        if portfolio.btc_held > 0 and cash < 5000 and spot_buy > 0:
            spot_buy = cash

    return Action(spot_buy_usd=spot_buy)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features_df = load_features()
    features_df = construct_features(features_df)
    scenarios = build_scenarios()

    results = []
    for scenario in scenarios:
        print(f"Running {scenario.name}...")
        result = run_backtest(decide_action, scenario, features_df)
        results.append(result)

    summary = score_results(results)
    print()
    print_results(summary, scenarios, features_df)
