"""
BTC bear-market deployment strategy. Single-file, agent-editable.
This is the ONLY file the autoresearch loop modifies.

Usage: python train.py
"""

import math

from prepare import (
    load_features, construct_features, build_scenarios,
    run_backtest, score_results, print_results,
    PortfolioState, Action, CSPOrder,
)

# ---------------------------------------------------------------------------
# Strategy parameters (edit these)
# ---------------------------------------------------------------------------

# Gate: MVRV must be below this to buy
MVRV_THRESHOLD = 0.80

# Sizing: deploy more when volatility is elevated (panic selling = better prices)
RV30_THRESHOLD = 0.70
DEPLOY_HIGH_CONVICTION = 0.50   # MVRV < threshold AND rv30 > threshold
DEPLOY_LOW_CONVICTION = 0.30    # MVRV < threshold AND rv30 <= threshold

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
    rv30 = _safe(features.get("realized_vol_30d"), 0.40)

    spot_buy = 0.0
    if mvrv < MVRV_THRESHOLD:
        # Scale deployment with MVRV depth: lower MVRV = more conviction
        # mvrv=0.80 → 30%, mvrv=0.70 → 50%, mvrv=0.60 → 70%
        depth = (MVRV_THRESHOLD - mvrv) / MVRV_THRESHOLD  # 0 to ~0.25
        frac = 0.10 + depth * 14.0  # 0.10 to ~1.0
        frac = min(frac, 1.0)
        spot_buy = cash * frac

        # Deploy all remainder when small
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
