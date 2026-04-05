# btc-autoresearch

This is an experiment to have the LLM discover optimal BTC bear-market deployment strategies.

## Context

You are optimizing a **buy-only BTC accumulation strategy**. The setup:
- You have **$50,000 in cash** at the start of a bear market
- You must deploy it into BTC before the next bull run
- You **never sell BTC**. BTC only goes up in your portfolio.
- USD drawdowns do not matter. Only BTC accumulated matters.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr3`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `prepare.py` — fixed evaluation harness: data loading, features, backtest engine, scoring. **Do not modify.**
   - `train.py` — the file you modify. Strategy logic, thresholds, sizing rules.
4. **Verify data exists**: Check that `data/btc_features.parquet` exists. If not, tell the human to run `python scripts/build_dataset.py`.
5. **Initialize results.tsv** and confirm setup.

## Experimentation

Each experiment runs instantly. Launch: `python train.py`.

**What you CAN do:**
- Modify `train.py` — indicator thresholds, buy logic, sizing, multi-indicator combinations, timing.

**What you CANNOT do:**
- Modify `prepare.py`.
- Install new packages.
- Add sell logic. BTC held must never decrease.

**The goal: get the highest composite_score** — average terminal BTC across two bear scenarios (2018, 2022).

**Simplicity criterion**: simpler is better. Removing complexity for equal results is a win.

## Output format

```
---
composite_score:      8.711448
scenario: bear_2018             terminal_btc: 14.286443  spot_buys: 2  | baseline lump: 3.657411  dca: 8.284021
scenario: bear_2022             terminal_btc: 3.136453  spot_buys: 2  | baseline lump: 0.769054  dca: 1.892548
```

## The experiment loop

LOOP FOREVER:

1. Tune `train.py` with an experimental idea
2. git commit
3. Run: `python train.py > run.log 2>&1`
4. Check: `grep "^composite_score:" run.log`
5. Log to `results.tsv`
6. If improved → keep. If worse → `git reset`

**NEVER STOP**. The loop runs until the human interrupts you.
