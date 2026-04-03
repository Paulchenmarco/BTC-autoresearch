"""
BTC bear-market deployment strategy. Single-file, agent-editable.
This is the ONLY file the autoresearch loop modifies.

Usage: python train.py
"""

from prepare import (
    load_features, construct_features, build_scenarios,
    run_backtest, score_results, print_results,
    PortfolioState, Action, CSPOrder,
)

# ---------------------------------------------------------------------------
# Strategy parameters (edit these)
# ---------------------------------------------------------------------------

# Spot buy triggers
MAYER_BUY_THRESHOLD = 0.8       # Buy spot when Mayer Multiple < this
DISTANCE_ATH_THRESHOLD = 0.60   # Buy spot when >60% below ATH
SPOT_DEPLOY_FRACTION = 0.10     # Fraction of available cash per spot buy

# CSP parameters
CSP_DEPLOY_FRACTION = 0.50      # Fraction of idle cash to deploy as CSP collateral
CSP_DEFAULT_DELTA = 0.20        # Default delta for put sells (lower = fewer assignments)
CSP_DEFAULT_DTE = 30            # Default DTE
CSP_AGGRESSIVE_DELTA = 0.35     # Higher delta when deep in bear
CSP_CONSERVATIVE_DELTA = 0.15   # Lower delta in high vol

# Volatility thresholds
VOL_HIGH_THRESHOLD = 0.90       # RV30 above this = high vol
VOL_LOW_THRESHOLD = 0.50        # RV30 below this = low vol

# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def decide_action(features, portfolio):
    """
    Core strategy function. Called once per day by the backtest engine.

    Args:
        features: dict of feature values for current date
        portfolio: PortfolioState with current holdings

    Returns:
        Action with spot_buy_usd and csp_sells
    """
    cash = portfolio.cash_available
    if cash < 10:
        return Action()

    mayer = features.get("mayer_multiple")
    rv30 = features.get("realized_vol_30d")
    dist_ath = features.get("distance_from_ath")

    # Handle NaN features (early dates before rolling windows fill)
    if mayer is None or (isinstance(mayer, float) and mayer != mayer):
        mayer = 1.0
    if rv30 is None or (isinstance(rv30, float) and rv30 != rv30):
        rv30 = 0.60
    if dist_ath is None or (isinstance(dist_ath, float) and dist_ath != dist_ath):
        dist_ath = 0.0

    spot_buy = 0.0
    csps = []

    # --- Spot buy logic: use power law residual for scaling ---
    pl_resid = features.get("power_law_residual")
    if pl_resid is None or (isinstance(pl_resid, float) and pl_resid != pl_resid):
        pl_resid = 0.0

    buy_signal = False
    if mayer < MAYER_BUY_THRESHOLD:
        buy_signal = True
    if dist_ath > DISTANCE_ATH_THRESHOLD:
        buy_signal = True

    if buy_signal:
        # Scale up when price is below power law trend
        scale = 1.0
        if pl_resid < -0.3:
            scale = 2.0
        elif pl_resid < -0.1:
            scale = 1.5
        spot_buy = cash * SPOT_DEPLOY_FRACTION * scale

    # --- CSP logic: keep idle cash productive ---
    remaining = cash - spot_buy
    if remaining > 100 and len(portfolio.open_csps) == 0:
        # Select delta based on vol regime
        delta = CSP_DEFAULT_DELTA
        if rv30 > VOL_HIGH_THRESHOLD:
            delta = CSP_CONSERVATIVE_DELTA
        elif dist_ath > 0.70:
            delta = CSP_AGGRESSIVE_DELTA

        csp_notional = remaining * CSP_DEPLOY_FRACTION
        if csp_notional > 100:
            csps.append(CSPOrder(
                delta=delta,
                dte=CSP_DEFAULT_DTE,
                notional_usd=csp_notional,
            ))

    return Action(spot_buy_usd=spot_buy, csp_sells=csps)

# ---------------------------------------------------------------------------
# Main: run all scenarios and print results
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
