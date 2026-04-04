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
# MVRV bottom regression
# ---------------------------------------------------------------------------

# Historical cycle bottoms: {cycle: mvrv_bottom}
CYCLE_BOTTOMS = {1: 0.387, 2: 0.564, 3: 0.690, 4: 0.754}

# Fit: MVRV_bottom = 1.0 - a * exp(-b * cycle)
_cn = np.array(list(CYCLE_BOTTOMS.keys()), dtype=float)
_mb = np.array(list(CYCLE_BOTTOMS.values()))
_slope, _intercept = np.polyfit(_cn, np.log(1.0 - _mb), 1)
_A, _B = np.exp(_intercept), -_slope

def predicted_mvrv_bottom(cycle_num):
    return 1.0 - _A * np.exp(-_B * cycle_num)

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

# Current cycle number (update each cycle)
CURRENT_CYCLE = 5

# Gate: predicted bottom + margin
GATE_MARGIN = 0.02
MVRV_GATE = predicted_mvrv_bottom(CURRENT_CYCLE) + GATE_MARGIN

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

def decide_action(features, portfolio):
    cash = portfolio.cash_available
    if cash < 10:
        return Action()

    mvrv = _safe(features.get("mvrv_ratio"), 1.0)
    mvrv_z = _safe(features.get("mvrv_z_score"), 0.0)

    spot_buy = 0.0

    if mvrv < MVRV_GATE:
        z_depth = max(0, (-mvrv_z - Z_REF)) / Z_REF
        frac = Z_BASE + z_depth * Z_MULT
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

    print(f"Cycle {CURRENT_CYCLE}: predicted bottom MVRV = {predicted_mvrv_bottom(CURRENT_CYCLE):.3f}")
    print(f"Gate (+ {GATE_MARGIN} margin): MVRV < {MVRV_GATE:.3f}")
    print()

    # Backtest using cycle-specific gates for historical scenarios
    scenarios = build_scenarios()
    cycle_map = {"bear_2018": 3, "bear_2022": 4}

    results = []
    for scenario in scenarios:
        cycle = cycle_map.get(scenario.name, CURRENT_CYCLE)
        gate = predicted_mvrv_bottom(cycle) + GATE_MARGIN

        def make_strategy(g=gate):
            def strategy(features, portfolio):
                cash = portfolio.cash_available
                if cash < 10: return Action()
                mvrv = _safe(features.get("mvrv_ratio"), 1.0)
                mvrv_z = _safe(features.get("mvrv_z_score"), 0.0)
                if mvrv < g:
                    z_depth = max(0, (-mvrv_z - Z_REF)) / Z_REF
                    frac = Z_BASE + z_depth * Z_MULT
                    frac = min(frac, 1.0)
                    spot_buy = cash * frac
                    if portfolio.btc_held > 0 and cash < 5000: spot_buy = cash
                    return Action(spot_buy_usd=spot_buy)
                return Action()
            return strategy

        print(f"Running {scenario.name} (cycle {cycle}, gate={gate:.3f})...")
        result = run_backtest(make_strategy(), scenario, features_df)
        results.append(result)

    summary = score_results(results)
    print()
    print_results(summary, scenarios, features_df)
