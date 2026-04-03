"""
BTC bear-market deployment strategy. Single-file, agent-editable.
This is the ONLY file the autoresearch loop modifies.

Usage: python train.py
"""

import math

from prepare import (
    load_features, construct_features, build_scenarios,
    run_backtest, score_results, print_results,
    PortfolioState, Action, CSPOrder, reprice_csp,
)

# ---------------------------------------------------------------------------
# Strategy parameters (edit these)
# ---------------------------------------------------------------------------

# MVRV gate and depth scaling for spot buys
MVRV_THRESHOLD = 0.82
DEPTH_BASE = 0.10
DEPTH_MULT = 13.0

# Put selling phase: earn premium while waiting for MVRV gate
CSP_SELL_ABOVE = 1.10         # Only sell puts when MVRV > this (far from danger)
CSP_CLOSE_BELOW = 1.00        # Force-close ALL puts when MVRV drops below this
CSP_DELTA = 0.10              # Far OTM
CSP_DTE = 30                  # Short DTE — fast cycling
CSP_ALLOC = 0.20              # 20% of cash as collateral
CSP_TAKE_PROFIT = 0.50        # Close when put loses 50% of value (take profit)

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
    csps = []
    close_indices = []

    # === PHASE 1: MVRV above gate — sell puts to earn premium ===
    if mvrv >= MVRV_THRESHOLD:

        # Force-close all puts when approaching danger zone
        if mvrv < CSP_CLOSE_BELOW and len(portfolio.open_csps) > 0:
            close_indices = list(range(len(portfolio.open_csps)))

        # Take profit on puts that have decayed > 50%
        elif len(portfolio.open_csps) > 0:
            for idx, csp in enumerate(portfolio.open_csps):
                current_val = reprice_csp(csp, features)
                if current_val < csp.premium_usd * CSP_TAKE_PROFIT:
                    close_indices.append(idx)

        # Sell new puts only when far from gate and no open positions
        if mvrv > CSP_SELL_ABOVE and len(portfolio.open_csps) == 0 and not close_indices:
            collateral = cash * CSP_ALLOC
            if collateral > 100:
                csps.append(CSPOrder(
                    delta=CSP_DELTA,
                    dte=CSP_DTE,
                    notional_usd=collateral,
                ))

    # === PHASE 2: MVRV below gate — close puts AND deploy spot same day ===
    else:
        # Force-close any remaining puts
        if len(portfolio.open_csps) > 0:
            close_indices = list(range(len(portfolio.open_csps)))

        # Deploy spot with depth scaling (validator accounts for freed cash)
        depth = (MVRV_THRESHOLD - mvrv) / MVRV_THRESHOLD
        frac = DEPTH_BASE + depth * DEPTH_MULT
        frac = min(frac, 1.0)
        spot_buy = cash * frac
        if portfolio.btc_held > 0 and cash < 5000 and spot_buy > 0:
            spot_buy = cash

    return Action(spot_buy_usd=spot_buy, csp_sells=csps, close_csp_indices=close_indices)

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
