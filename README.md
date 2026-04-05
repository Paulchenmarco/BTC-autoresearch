# BTC Bear-Market Deployment — Autoresearch

Autonomous strategy discovery for deploying cash into BTC during bear markets.

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## The Problem

You have $50,000 during a Bitcoin bear market. Deploy it all into BTC before the bull run. When to buy, and how much?

## Strategy (after ~150 experiments)

**MVRV regression gate + MVRV-Z depth scaling.**

1. **Gate**: predict the cycle's MVRV bottom using walk-forward regression on prior cycles, add 0.02 margin
2. **Sizing**: scale deployment by how negative MVRV-Z is (more extreme undervaluation = deploy more)

| Cycle | Predicted Bottom | Gate | Result |
|-------|-----------------|------|--------|
| 3 (2018) | 0.741 | 0.761 | 14.29 BTC (+73% vs DCA) |
| 4 (2022) | 0.780 | 0.800 | 3.14 BTC (+66% vs DCA) |
| 5 (current) | 0.825 | 0.845 | Waiting... |

**Key finding**: MVRV ratio is the only indicator that matters for timing bear market bottoms. Every other indicator tested (Mayer Multiple, distance from ATH, power law residual, NUPL, Puell Multiple, hash ribbons, realized vol, funding rates) was either redundant or noise.

## Repo Structure

| File | Role | Editable? |
|------|------|-----------|
| `prepare.py` | Fixed backtest engine, features, scoring | NO |
| `train.py` | Strategy logic | YES (only this) |
| `program.md` | Agent instructions | Human-edited |
| `scripts/build_dataset.py` | Fetch BTC data | Run once |

## Quick Start

```bash
pip install pandas pyarrow numpy yfinance
python scripts/download_data.py
python scripts/build_dataset.py
python train.py
```
