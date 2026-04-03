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
MVRV_THRESHOLD = 0.80
DEPTH_BASE = 0.10
DEPTH_MULT = 12.0

# Put selling
CSP_SELL_ABOVE = 1.65         # Sell puts when MVRV > this
CSP_DELTA = 0.14
CSP_DTE = 14
CSP_ALLOC = 0.95

# Roll parameters: when a put doubles in value (moving against us), roll down and out
CSP_ROLL_TRIGGER = 2.0        # Roll when put value reaches 2x premium received
CSP_ROLL_DELTA = 0.10         # Roll to a lower delta (further OTM)
CSP_ROLL_DTE = 30             # Roll to longer DTE (more time value to offset cost)

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
    mvrv = _safe(features.get("mvrv_ratio"), 1.0)

    spot_buy = 0.0
    csps = []
    close_indices = []

    # === ROLL defense: always check open puts regardless of MVRV ===
    if len(portfolio.open_csps) > 0 and mvrv >= MVRV_THRESHOLD:
        for idx, csp in enumerate(portfolio.open_csps):
            current_val = reprice_csp(csp, features)
            if current_val > csp.premium_usd * CSP_ROLL_TRIGGER:
                # Roll down and out: close losing put, open lower strike + longer DTE
                close_indices.append(idx)
                csps.append(CSPOrder(
                    delta=CSP_ROLL_DELTA,
                    dte=CSP_ROLL_DTE,
                    notional_usd=csp.notional,
                ))

    if mvrv < MVRV_THRESHOLD:
        # === SPOT: deploy with depth scaling ===
        if cash < 10:
            return Action()
        depth = (MVRV_THRESHOLD - mvrv) / MVRV_THRESHOLD
        frac = DEPTH_BASE + depth * DEPTH_MULT
        frac = min(frac, 1.0)
        spot_buy = cash * frac
        if portfolio.btc_held > 0 and cash < 5000 and spot_buy > 0:
            spot_buy = cash

    elif mvrv > CSP_SELL_ABOVE and len(portfolio.open_csps) - len(close_indices) == 0:
        # === PUTS: sell new when no open positions ===
        total_cash = cash
        for idx in close_indices:
            total_cash += portfolio.open_csps[idx].notional
        collateral = total_cash * CSP_ALLOC
        if collateral > 100:
            csps.append(CSPOrder(delta=CSP_DELTA, dte=CSP_DTE, notional_usd=collateral))

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
