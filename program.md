# btc-autoresearch

This is an experiment to have the LLM discover optimal BTC bear-market deployment strategies.

## Context

You are optimizing a **buy-only BTC accumulation strategy**. The setup:
- You have **$50,000 in cash** at the start of a bear market
- You must deploy it into BTC before the next bull run
- You can **buy spot BTC** or **sell cash-secured puts (CSPs)** to keep idle cash productive
- You **never sell BTC**. BTC only goes up in your portfolio.
- USD drawdowns do not matter. Only BTC accumulated matters.
- The CSP sleeve simulates Deribit-style European cash-settled puts

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr3`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — project context.
   - `prepare.py` — fixed evaluation harness: data loading, features, backtest engine, CSP simulation, scoring. **Do not modify.**
   - `train.py` — the file you modify. Strategy logic, thresholds, sizing rules.
4. **Verify data exists**: Check that `data/btc_features.parquet` exists. If not, tell the human to run `python scripts/build_dataset.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs instantly (backtests complete in seconds). You launch it as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: indicator thresholds, buy logic, spot sizing, CSP delta selection, CSP DTE, CSP sleeve sizing, regime classification, multi-indicator combinations, timing logic.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed backtest engine, CSP simulation, feature construction, and scoring.
- Install new packages or add dependencies.
- Add sell logic. This is a buy-only strategy. BTC held must never decrease.
- Modify the scoring function or scenarios.

**The goal is simple: get the highest composite_score.** This is the average terminal BTC accumulated across two historical bear-to-recovery scenarios (2018 and 2022). A higher score means your strategy deployed $50k into more BTC.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so run the script as is.

## Output format

The script prints a summary like this:

```
---
composite_score:      5.123456
scenario: bear_2018             terminal_btc: 8.234567  ...  | baseline lump: 2.940000  dca: 5.100000
scenario: bear_2022             terminal_btc: 2.012345  ...  | baseline lump: 0.720000  dca: 1.800000
```

Extract the key metric:

```
grep "^composite_score:" run.log
```

## Logging results

Log experiments to `results.tsv` (tab-separated).

Header and columns:

```
commit	composite_score	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score achieved (e.g. 5.123456) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	composite_score	status	description
a1b2c3d	5.123456	keep	baseline
b2c3d4e	5.456789	keep	lower mayer threshold to 0.7
c3d4e5f	4.890123	discard	aggressive spot sizing 30%
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr3`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1`
5. Read out the results: `grep "^composite_score:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (do not commit results.tsv)
8. If composite_score improved (higher), keep the git commit
9. If composite_score is equal or worse, git reset back to where you started

**Crashes**: If a run crashes, fix obvious bugs and re-run. If the idea is fundamentally broken, skip it and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the feature set in `prepare.py`, try combining indicators, try different CSP strategies, try radical changes to deployment cadence. The loop runs until the human interrupts you.
