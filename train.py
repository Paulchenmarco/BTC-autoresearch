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

# Conviction thresholds — when these align, deploy heavily
MVRV_THRESHOLD = 0.80
DIST_ATH_THRESHOLD = 0.75
PL_RESID_THRESHOLD = -0.40
RV30_THRESHOLD = 0.70

# Deployment sizing by conviction level
DEPLOY_4_SIGNALS = 0.50     # 4/4 signals: deploy 50% of cash
DEPLOY_3_SIGNALS = 0.30     # 3/4 signals: deploy 30%
DEPLOY_2_SIGNALS = 0.15     # 2/4 signals: deploy 15%

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
    dist_ath = _safe(features.get("distance_from_ath"), 0.0)
    pl_resid = _safe(features.get("power_law_residual"), 0.0)
    rv30 = _safe(features.get("realized_vol_30d"), 0.40)

    # Count conviction signals
    signals = 0
    if mvrv < MVRV_THRESHOLD:
        signals += 1
    if dist_ath > DIST_ATH_THRESHOLD:
        signals += 1
    if pl_resid < PL_RESID_THRESHOLD:
        signals += 1
    if rv30 > RV30_THRESHOLD:
        signals += 1

    # Deploy based on conviction
    spot_buy = 0.0
    if signals >= 4:
        spot_buy = cash * DEPLOY_4_SIGNALS
    elif signals >= 3:
        spot_buy = cash * DEPLOY_3_SIGNALS
    elif signals >= 2:
        spot_buy = cash * DEPLOY_2_SIGNALS

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
