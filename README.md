# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/marugamiii/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/marugamiii/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

**GTA VI trophy rarity forecaster** — forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Program

`main.py` (Python 3.13, standard library only). Run by the grader as
`python3 /bot/main.py`; it reads every `*.in` case in `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR`
(default `/output`). Deterministic (two runs are byte-identical) and fast
(~60 ms for the two practice cases, far under the 8 s/case limit).

## Model

An earn rate is **earners / players-tracked**, so the forecast decomposes
into two moving quantities:

- **N(t), players tracked.** Exophase only changes a game's counts when it
  refreshes them, so the history is a step function; between refreshes every
  read is identical. N grows through launch week as more players link PSN.
  The program collapses reads to distinct page states, then forecasts N at
  the target by blending a power-law fit (ln N ~ ln(1 + hours since release))
  with an exponential fit (ln N ~ hours), blended in log space and weighted
  by fit quality — a linear-space blend would let the exponential fit's wild
  long-horizon extrapolation dominate.
- **E_i(t), earners of trophy i.** It grows as the player base progresses.
  Per trophy, the program regresses ln E_i on ln N over the recent states,
  shrinking each slope toward the pooled slope. The slope is the trophy's
  elasticity: early/easy trophies gain earners almost as fast as N grows
  (slope near or above 1, rate holds or rises), the platinum and grind
  trophies barely gain any (slope near 0, rate falls as N grows). The median
  rate is 100 · E_i(target) / N(target).

The spread combines a (shrunk) random-walk term, N-forecast uncertainty,
slope uncertainty, and a refresh-staleness term — the answer is the page as
of Exophase's last refresh before the target, so the forecast must survive
one more refresh.

**Flat history (the practice cases).** Every read is the same page, so no
trend is estimable. The forecast is then a two-component mixture on the
logit scale, using the one refresh the competition documents (launch night:
638 → 3,868 players tracked, every rate but the 100% one lower, mean logit
shift ≈ 1.7): with probability p a refresh lands inside the horizon and the
rate's logit drops by ~1.6 (with spread for per-trophy variation), with
probability 1 − p nothing moves. p grows with how long the page has already
sat unchanged. The 100% trophy is handled separately in both paths: it can
only fall as new players are tracked, so its quantiles are asymmetric with
the median near the top and a long lower tail.

## Results (local evaluation)

Practice answers are not published by the server, so the practice loss below
is scored against a **reconstructed** answer: the live Exophase page fetched
2026-11-19 11:33 UTC (the 04:30 refresh state — 3,868 players tracked — as
of the last refresh before that fetch). Sanity check for the reconstruction:
the house baseline scores **1.543** on it, vs the **~1.51** the rules state
for the baseline on the real practice answers.

| | practice loss (recon.) | typical miss | 90% coverage |
|---|---|---|---|
| House baseline (starter) | 1.543 | 0.68 pp | 0.00 |
| This entry | **0.236** | **0.06 pp** | 0.96 |

Robustness: shifting the reconstructed answers by ±0.15 / ±0.30 in logit
(a later or smaller refresh) moves this entry's loss to 0.22–0.34, always
far below the baseline's 1.25–1.84. In the no-refresh counterfactual the
entry scores 0.85 (the price of hedging for a refresh) vs the baseline's
0.02 — the refresh hedge is the right bet whenever a refresh is likelier
than not inside the horizon, which it is after the page sat unchanged for
3+ hours.

Final-case behaviour was validated on synthetic final-like cases (history
Nov 19 05:00 → Nov 21 11:30, target Nov 26): a power-law generator and a
cohort generator (players join over time, each cohort progresses at its own
pace), across 10 scenarios each (slow/fast/saturating player growth, fast/
slow progression, frequent/rare refreshes, dilution regimes). This entry
beats the baseline in 19 of 20 scenarios (typical loss ≈ 0.13–0.23 vs
baseline ≈ 0.28–0.58). The one exception is an artificial "progression
stops dead at the deadline" scenario (loss 1.48 vs baseline 0.55) — a
structural break the trend extrapolation cannot foresee; it is judged
implausible for a launch week that includes a full weekend of play, and is
not hedged for at the cost of the normal scenarios.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

## Files

- `main.py` — the entry (single file, no dependencies).
- `README.md` — this file.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
