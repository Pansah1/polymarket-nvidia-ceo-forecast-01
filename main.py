#!/usr/bin/env python3
"""Forecast Polymarket's Next CEO of NVIDIA prices, 4 hours ahead.

Reads one case at a time (a JSON file with every market's Yes price per
minute for the 72 hours up to a checkpoint) and writes, for each market and
each of the 240 minutes after the checkpoint, five quantiles (5/25/50/75/95%)
of that minute's price.

Method: a persistence forecast -- the price stays where it was, with a
spread widening like a random walk's -- but the spread is NOT the normal
one. The practice data showed the moves are far from normal: most minutes
nothing happens, then a jump, and the jumps are lopsided. So instead of
z_tau * SIGMA, each level gets its own coefficient C_TAU, fitted from the
practice cases' own move distribution: for every horizon h, take the
empirical quantiles of (price[t+h] - price[t]) across all practice
markets, and fit each level's coefficient as the median of
quantile(h) / sqrt(h) over h. Three things the practice data showed:

- The median belongs at the last price. Pulling it toward a longer-term
  mean (mean reversion) made the loss much worse at every weight tried:
  over four hours these prices behave like a random walk, not a spring.
  Adding a drift (recent trend) was far worse still (0.72-1.41).
- Per-market volatility estimates hurt too: the recent wiggles of one
  market do not predict how far it will move next, and adapting the
  spread per market only added noise. One set of coefficients, fitted on
  all the practice cases together, beats the baseline and every adaptive
  variant.
- The normal spread is the wrong shape: the 5% tail is much wider
  (-0.0225 vs the normal -0.0025) and the 75% level much narrower
  (0.0002 vs 0.0010), because prices sit still for long stretches and
  then jump. The empirical coefficients capture that shape.

So the forecast is q_tau(h) = clamp(x + C_TAU[tau] * sqrt(h), 0, 1),
with x the market's last price. Backtested on the practice cases with
leave-one-case-out (fit on 5 cases, score on the 6th): 0.4207, against
0.4268 for the best normal-spread persistence model and 0.4375 for the
house baseline. The fitted coefficients below come from all six cases.
"""

import json
import math
import os
import sys


def _fail(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(1)

Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]  # normal quantiles (reference only)
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]  # matching levels, in order
# Empirical per-level coefficients, fitted from the practice cases' own
# move distribution (median of empirical-quantile(h)/sqrt(h) over h=1..240).
# The shape is lopsided and fat-tailed, unlike the normal spread: the 5%
# tail is far wider and the 75% level far narrower than a normal walk.
# Fitted on all six practice cases; leave-one-case-out backtest 0.4207
# vs 0.4268 for the best normal-spread model and 0.4375 for the baseline.
C_TAU = [-0.022524, -0.000994, 0.0, 0.00021, 0.006637]


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (the output precision).

    The + 0.0 normalises -0.0 to 0.0 so no output row ever reads -0.0.
    """
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon):
    """One market's quantiles: median = last price, spread ~ sqrt(h).

    The spread grows with the square root of the horizon, as a random
    walk's would: after h minutes the price has had h chances to move,
    so its typical displacement scales like sqrt(h). Each quantile level
    uses its own empirical coefficient C_TAU, so the forecast's shape
    follows the practice data's own move distribution rather than a
    normal curve. The coefficients are already ordered, and clamping to
    [0, 1] preserves that order, so every row is non-decreasing.
    """
    if not prices:
        return [[0.5] * len(C_TAU) for _ in range(horizon)]
    x = prices[-1]
    rows = []
    for h in range(1, horizon + 1):
        s = math.sqrt(h)
        rows.append([price6(x + c * s) for c in C_TAU])
    return rows


def main():
    """Read every *.in case, write a *.out forecast per case."""
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        cid = name[:-3]
        with open(os.path.join(in_dir, name)) as f:
            case = json.load(f)
        horizon = case.get("horizon_minutes", 240)
        forecasts = []
        for m in case["markets"]:
            forecasts.append(
                {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            )
        out_path = os.path.join(out_dir, cid + ".out")
        with open(out_path, "w") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    # Entry point: the grader runs `python3 main.py` with the input and
    # output directories in the environment. Nothing else to configure.
    main()
