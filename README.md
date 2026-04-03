# BTC Bear-Market Deployment — Autoresearch

Autonomous strategy discovery for deploying cash into BTC during bear markets.

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the same pattern of fixed evaluation harness + mutable strategy file + git ratchet, applied to BTC accumulation instead of LLM training.

## The Problem

You have $50,000 in cash during a Bitcoin bear market. You want to deploy all of it into BTC before the next bull run starts. The question is: **when to buy, how much, and what to do with idle cash while waiting.**

## How It Works

**Two deployment mechanisms:**

1. **Spot buy** — buy BTC immediately with some/all available cash
2. **Cash-secured put (CSP) sleeve** — sell Deribit-style European cash-settled puts on idle cash. If the put expires OTM, you keep the premium. If ITM, the cash-settled loss is taken and remaining collateral immediately buys spot BTC (synthetic assignment).

**The autoresearch loop** iterates on `train.py` to find the best rules for combining these mechanisms. Each iteration:
1. Edit `train.py` (the strategy)
2. Run backtests across historical bear-to-recovery periods
3. If terminal BTC improved → keep the commit
4. If not → revert

## Philosophy

- **BTC is the unit of account.** Only terminal BTC matters.
- **Buy-only.** Never sell BTC. Never reduce BTC allocation.
- **USD drawdowns don't matter.** A strategy that buys at $60k and BTC drops to $30k is fine — the BTC is still there.
- **Idle cash should be productive.** The CSP sleeve earns premium while waiting for better entry points.

## Repo Structure

| File | Role | Editable? |
|------|------|-----------|
| `prepare.py` | Fixed evaluation harness: data loading, features, backtest engine, CSP simulation, scoring | NO |
| `train.py` | Strategy logic: buy signals, spot sizing, CSP parameters | YES (only this file) |
| `program.md` | Instructions for the autoresearch agent | Human-edited |
| `scripts/build_dataset.py` | Fetch BTC OHLC data | Run once |
| `data/btc_features.parquet` | Daily BTC price data | Generated |

## Quick Start

```bash
# Install dependencies
pip install pandas pyarrow numpy scipy yfinance

# Fetch data
python scripts/build_dataset.py

# Run baseline
python train.py

# Start the autoresearch loop (see program.md)
```

## Scenarios

The strategy is tested across two completed bear-to-recovery cycles:

| Scenario | Period | Shape |
|----------|--------|-------|
| bear_2018 | 2018-01 → 2019-06 | Gradual bleed → V-bottom → sharp recovery |
| bear_2022 | 2021-11 → 2023-09 | Stair-step → double capitulation (LUNA+FTX) → grind up |

Each scenario starts with $50k in cash. The composite score is the average terminal BTC across both scenarios.
