"""
BTC bear-market deployment strategy. Single-file, agent-editable.
This is the ONLY file the autoresearch loop modifies.

Usage: python train.py
"""

import math
import numpy as np

from prepare import (
    load_features, construct_features, build_scenarios,
    run_backtest, score_results, print_results,
    PortfolioState, Action,
)

# ---------------------------------------------------------------------------
# MVRV bottom regression (walk-forward: only use prior cycles)
# ---------------------------------------------------------------------------

# Historical cycle bottoms
CYCLE_BOTTOMS = {1: 0.387, 2: 0.564, 3: 0.690, 4: 0.754}

def predict_mvrv_bottom_walk_forward(target_cycle):
    """Predict MVRV bottom using only cycles BEFORE the target (no look-ahead)."""
    prior = {c: v for c, v in CYCLE_BOTTOMS.items() if c < target_cycle}
    if len(prior) < 2:
        return 0.80  # fallback
    cn = np.array(list(prior.keys()), dtype=float)
    mb = np.array(list(prior.values()))
    if len(cn) == 2:
        # Linear extrapolation
        slope, intercept = np.polyfit(cn, mb, 1)
        return slope * target_cycle + intercept
    else:
        # Exponential decay: bottom = 1.0 - a * exp(-b * cycle)
        gaps = 1.0 - mb
        log_gaps = np.log(gaps)
        slope, intercept = np.polyfit(cn, log_gaps, 1)
        a, b = np.exp(intercept), -slope
        return 1.0 - a * np.exp(-b * target_cycle)

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

CURRENT_CYCLE = 5
GATE_MARGIN = 0.02

# Depth scaling via MVRV-Z
Z_BASE = 0.05
Z_MULT = 4.0
Z_REF = 0.40

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

def make_strategy(cycle_num):
    """Create a strategy function for a specific cycle (walk-forward gate)."""
    gate = predict_mvrv_bottom_walk_forward(cycle_num) + GATE_MARGIN

    def decide_action(features, portfolio):
        cash = portfolio.cash_available
        if cash < 10:
            return Action()

        mvrv = _safe(features.get("mvrv_ratio"), 1.0)
        mvrv_z = _safe(features.get("mvrv_z_score"), 0.0)

        spot_buy = 0.0

        if mvrv < gate:
            z_depth = max(0, (-mvrv_z - Z_REF)) / Z_REF
            frac = Z_BASE + z_depth * Z_MULT
            frac = min(frac, 1.0)
            spot_buy = cash * frac
            if portfolio.btc_held > 0 and cash < 5000 and spot_buy > 0:
                spot_buy = cash

        return Action(spot_buy_usd=spot_buy)

    return decide_action, gate

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features_df = load_features()
    features_df = construct_features(features_df)
    scenarios = build_scenarios()
    cycle_map = {"bear_2018": 3, "bear_2022": 4}

    print("Walk-forward MVRV regression (no look-ahead):")
    for c in [3, 4, 5]:
        pred = predict_mvrv_bottom_walk_forward(c)
        actual = CYCLE_BOTTOMS.get(c)
        actual_str = f"{actual:.3f}" if actual else "???"
        print(f"  Cycle {c}: predicted={pred:.3f}  actual={actual_str}  gate={pred + GATE_MARGIN:.3f}")
    print()

    results = []
    for scenario in scenarios:
        cycle = cycle_map.get(scenario.name, CURRENT_CYCLE)
        strategy, gate = make_strategy(cycle)
        print(f"Running {scenario.name} (cycle {cycle}, gate={gate:.3f})...")
        result = run_backtest(strategy, scenario, features_df)
        results.append(result)

    summary = score_results(results)
    print()
    print_results(summary, scenarios, features_df)
