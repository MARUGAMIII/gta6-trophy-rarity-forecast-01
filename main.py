"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every <id>.in case in the input directory and writes <id>.out forecasts:
for each of the game's 55 PS5 trophies, five quantiles (5/25/50/75/95%) of its
earn rate on Exophase at the case's target.

Model
-----
An earn rate is earners / players-tracked, so the forecast decomposes into two
quantities that both move:

* N(t): the players Exophase tracks. It only changes when Exophase refreshes
  a game's counts, so the history is a step function; between refreshes every
  read is identical. N grows through launch week as more players link PSN.
* E_i(t): how many tracked players have earned trophy i. It grows as the
  player base progresses through the game; early/easy trophies gain earners
  fast, the platinum and grind trophies barely at all.

Rich history (the final case): collapse the reads to distinct states
(first-seen time of each refreshed page), then per trophy regress
ln E_i on ln N across the states, shrinking each slope toward the pooled
slope. N at the target comes from blending a power-law fit
(ln N ~ ln(1 + hours since release)) with an exponential fit
(ln N ~ hours), weighted by fit quality. The median rate is
100 * E_i(target) / N(target). The spread combines the random-walk term,
the N-forecast uncertainty, the slope uncertainty and a refresh-staleness
term (the answer is the page as of Exophase's last refresh before the
target, so the forecast must survive one more refresh).

Flat history (the practice cases): every read is the same page, so no trend
is estimable. Exophase refreshes on its own schedule; the one refresh the
competition documents (launch night: 638 -> 3,868 players tracked, every
rate but the 100% one lower, mean logit shift about 1.7) is the prior. The
forecast is a two-component mixture on the logit scale: with probability p
a refresh lands inside the horizon and the rate's logit drops by ~1.6 (with
spread for per-trophy variation), with probability 1 - p nothing moves.
p grows with how long the page has already sat unchanged. The 100% trophy
is handled separately: it can only fall, so its quantiles are asymmetric.

Everything is deterministic (no randomness, no dict-order dependence) and
uses only the standard library. Runtime is milliseconds per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- documented refresh prior (from the competition rules, launch night) ---
# Exophase refreshed once after the page sat unchanged for hours: players
# tracked 638 -> 3,868 and every rate but the 100% one fell; the mean logit
# shift of the rates was about 1.7. A refresh's logit drop and its spread:
REFRESH_DROP = 1.6
REFRESH_DROP_SPREAD = 0.5
# Typical gap between Exophase refreshes of a game's counts, hours.
REFRESH_PERIOD_H = 4.5
# No-refresh drift on the logit scale (per sqrt day).
NO_REFRESH_SPREAD = 0.15


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO 8601 UTC -> aware datetime. Accepts a trailing Z."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, rmse) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    return a, b, rmse


def mixture_quantiles(centre, p_refresh, drop, s0, s1):
    """Quantiles of a two-component logit mixture.

    With probability (1 - p) the rate's logit is Normal(centre, s0)
    (no refresh before the target); with probability p it is
    Normal(centre - drop, s1) (a refresh dilutes the rate).
    """
    lo = centre - drop - 5.0 * s1 - 1.0
    hi = centre + 5.0 * s0 + 1.0

    def cdf(x):
        return (1.0 - p_refresh) * phi((x - centre) / s0) + p_refresh * phi(
            (x - (centre - drop)) / s1
        )

    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(sigmoid(0.5 * (a + b)))
    return out


def distinct_states(case):
    """Collapse history reads to distinct page states, oldest first.

    Each state is (first_seen_time, players_tracked, rates tuple).
    Consecutive identical reads carry no new information: Exophase only
    changes the page when it refreshes the game's counts.
    """
    states = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        states.append((parse_time(read["at"]), read["players_tracked"], read["rates"]))
    return states


def flat_history_rows(case, states):
    """Forecast when the history shows a single unchanged page.

    No trend is estimable, so use the documented-refresh mixture prior.
    How long the page has sat unchanged (from the first read to the
    checkpoint, plus the horizon) sets the refresh probability.
    """
    first_at = parse_time(case["history"][0]["at"])
    checkpoint = parse_time(case["at"])
    flat_h = max(0.0, (checkpoint - first_at).total_seconds() / 3600.0)
    horizon_h = case["horizon_minutes"] / 60.0
    # A refresh is due once the page has sat for about a refresh period;
    # the longer it has sat (and the longer the horizon), the likelier one
    # lands before the target.
    p_refresh = (flat_h + horizon_h - 1.0) / REFRESH_PERIOD_H
    p_refresh = min(0.9, max(0.2, p_refresh))

    last_rates = states[-1][2]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i]
        if rate is None:
            rate = 1.0
        if rate >= 99.99:
            # The 100% trophy can only fall as new players are tracked.
            # Keep the median high but allow a long lower tail.
            rows.append({"id": trophy["id"], "q": [96.0, 99.0, 99.8, 99.95, 99.99]})
            continue
        qs = mixture_quantiles(
            logit(rate), p_refresh, REFRESH_DROP, NO_REFRESH_SPREAD, REFRESH_DROP_SPREAD
        )
        rows.append({"id": trophy["id"], "q": [round4(q) for q in qs]})
    return rows


def trend_rows(case, states, release):
    """Forecast from a history with several distinct page states."""
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    h_target = (target - release).total_seconds() / 3600.0

    # --- N(t): players tracked at the target ------------------------------
    xs_pow, xs_exp, ys_n = [], [], []
    for t, n, _rates in states:
        if n is None or n <= 0:
            continue
        h = max(0.0, (t - release).total_seconds() / 3600.0)
        xs_pow.append(math.log1p(h))
        xs_exp.append(h)
        ys_n.append(math.log(n))
    fit_pow = ols(xs_pow, ys_n)
    fit_exp = ols(xs_exp, ys_n)
    # Blend the two N forecasts in LOG space (multiplicative errors):
    # the exponential fit can extrapolate wildly over a 4.5-day horizon,
    # so a linear-space blend would let it dominate. Weight by fit quality.
    preds = []
    if fit_pow is not None:
        a, b, rmse = fit_pow
        preds.append((a + b * math.log1p(h_target), 1.0 / (rmse + 0.05) ** 2))
    if fit_exp is not None:
        a, b, rmse = fit_exp
        preds.append((a + b * h_target, 1.0 / (rmse + 0.05) ** 2))
    if preds:
        # If the two forecasts disagree wildly, trust only the better fit.
        if len(preds) == 2 and abs(preds[0][0] - preds[1][0]) > math.log(10.0):
            best = max(preds, key=lambda pw: pw[1])
            ln_n_target = best[0]
        else:
            wsum = sum(w for _p, w in preds)
            ln_n_target = sum(p * w for p, w in preds) / wsum
        n_target = math.exp(ln_n_target)
    else:
        n_target = states[-1][1]
    n_last = states[-1][1]
    # N never falls (players only join the tracked set) and a launch-week
    # explosion past ~30x in one horizon is not credible: dampen it.
    n_target = max(n_last, min(n_target, n_last * 30.0))
    ln_n_ratio = math.log(n_target / n_last) if n_last > 0 else 0.0

    # Residual scatter of the N fits, for the spread.
    n_rmse = 0.0
    if fit_pow is not None:
        n_rmse = max(n_rmse, fit_pow[2])
    if fit_exp is not None:
        n_rmse = max(n_rmse, fit_exp[2])

    # --- typical refresh jump (for the staleness spread) ------------------
    jumps = []
    for j in range(1, len(states)):
        prev_rates = states[j - 1][2]
        cur_rates = states[j][2]
        for a_rate, b_rate in zip(prev_rates, cur_rates):
            if a_rate is None or b_rate is None:
                continue
            if a_rate >= 99.99 or b_rate >= 99.99:
                continue
            jumps.append(abs(logit(b_rate) - logit(a_rate)))
    jumps.sort()
    jump_typ = jumps[len(jumps) // 2] if jumps else 0.5
    jump_typ = min(max(jump_typ, 0.1), 1.5)

    # Refresh cadence: median gap between distinct states.
    gaps = [
        (states[j][0] - states[j - 1][0]).total_seconds() / 3600.0
        for j in range(1, len(states))
    ]
    gaps.sort()
    interval_h = gaps[len(gaps) // 2] if gaps else REFRESH_PERIOD_H
    interval_h = min(max(interval_h, 1.0), 48.0)
    stale_h = max(0.0, (checkpoint - states[-1][0]).total_seconds() / 3600.0)
    # Staleness: the answer is the page as of Exophase's last refresh
    # before the target, so it can lag the trend by up to about one refresh
    # interval (plus whatever staleness the last history state already had
    # at the checkpoint). The horizon itself is the trend model's job, not
    # the staleness term's.
    s_stale = jump_typ * 0.5 * math.sqrt(min(1.0, stale_h / interval_h) + 1.0)
    s_stale = min(s_stale, 0.8)

    # --- per-trophy earner elasticity --------------------------------------
    n_trophies = len(case["trophies"])
    # Pooled slope from the last window of states (recent behaviour matters
    # most: early launch hours have unusually fast progression). Use the
    # last third of states for the local slope: a trophy's earners grow
    # with the tracked set, and the slope near the end of the history is
    # the best guide to the horizon.
    window = states[-max(3, len(states) // 3):]
    # Pooled regression on log-differences vs the last state.
    dxs, dys = [], []
    for t, n, rates in window[:-1]:
        if n is None or n <= 0 or n_last <= 0:
            continue
        dx = math.log(n_last) - math.log(n)
        if abs(dx) < 1e-9:
            continue
        for i in range(n_trophies):
            r0, r1 = rates[i], states[-1][2][i]
            if r0 is None or r1 is None or r0 <= 0 or r1 <= 0:
                continue
            if r0 >= 99.99 or r1 >= 99.99:
                continue
            dxs.append(dx)
            dys.append(math.log(max(r1, 0.01) * n_last) - math.log(max(r0, 0.01) * n))
    pooled_beta = 0.35
    if len(dxs) >= 10:
        fit = ols(dxs, dys)
        if fit is not None:
            # Earners can grow faster than the tracked set while a trophy
            # is still being earned quickly (slope above 1), or slower
            # for the platinum and grinds (slope below 1).
            pooled_beta = min(max(fit[1], 0.0), 1.5)

    s_rw = 0.5 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = states[-1][2][i]
        if last_rate is None:
            last_rate = 1.0
        if last_rate >= 99.99:
            # Everyone tracked has it; it survived a 6x jump in the tracked
            # set at launch, so it most likely stays at 100%, but a refresh
            # can catch brand-new players before they fish. Median near the
            # top with a lower tail (tuned on the pinball loss).
            rows.append({"id": trophy["id"], "q": [97.0, 99.5, 99.95, 99.99, 99.99]})
            continue

        # Per-trophy ln E ~ ln N regression over the recent window.
        xs, ys = [], []
        for _t, n, rates in window:
            r = rates[i]
            if r is None or n is None or n <= 0 or r <= 0:
                continue
            xs.append(math.log(n))
            ys.append(math.log(max(r, 0.01) * n / 100.0 + 0.5))
        beta = pooled_beta
        resid_sd = 0.3
        k = len(xs)
        if k >= 3:
            fit = ols(xs, ys)
            if fit is not None:
                _a, b, rmse = fit
                w = k / (k + 3.0)
                beta = w * b + (1.0 - w) * pooled_beta
                resid_sd = max(rmse, 0.05)
        beta = min(max(beta, -0.2), 1.5)

        e_last = max(last_rate, 0.01) * n_last / 100.0
        e_target = e_last * math.exp(beta * ln_n_ratio)
        rate_med = 100.0 * e_target / n_target
        rate_med = clamp_pct(rate_med)

        # Spread: random walk (shrunk: the trend already removes drift)
        # plus N-forecast uncertainty (rate ~ N^(beta-1)), slope
        # uncertainty, and refresh staleness.
        se_ln_n = n_rmse * math.sqrt(1.0 + 1.0 / max(1, len(xs_pow)))
        s_trend = math.sqrt(
            ((beta - 1.0) * se_ln_n) ** 2
            + (resid_sd * math.sqrt(1.0 + 1.0 / max(1, k))) ** 2
        )
        s = math.sqrt((0.40 * s_rw) ** 2 + s_trend ** 2 + s_stale ** 2)
        s = min(max(s, 0.25), 1.8)

        lg = logit(rate_med)
        qs = [round4(sigmoid(lg + z * s)) for z in Z]
        # Rounding can only keep order; enforce it explicitly.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return rows


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the scored ones")
    if not case["history"]:
        raise ValueError("case has no history")
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    states = distinct_states(case)
    if len(states) < 2:
        rows = flat_history_rows(case, states)
    else:
        # Flat in rates as well as in N: nothing ever moved.
        moved = any(
            states[j][2] != states[0][2] for j in range(1, len(states))
        )
        rows = trend_rows(case, states, release) if moved else flat_history_rows(case, states)
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
