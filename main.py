#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01 (program/v1).
Python 3.13, standard library only, no network, deterministic.

Why this model
--------------
An earn rate is a ratio: rate = earners / players_tracked. In week one
both parts move fast and they do not move together. The tracked player
base grows in jumps, because Exophase refreshes a game's page on its own
clock: the house recorder can watch an unchanged page for hours and then
see the player count multiply at one refresh, every rate dropping even
though nobody un-earned a trophy. A trend fitted to the rates alone
reads that dilution as players going backwards. So this entry splits
the ratio:

  * it collapses the read history to the distinct page states (one per
    refresh), throwing away the repeated reads of an unchanged page;
  * it forecasts the player count on its own, in log space against
    log-time, which is how launch-week growth bends;
  * it forecasts each trophy's implied earner count, earners =
    rate * players / 100, the same way, with each trophy's growth slope
    shrunk toward the cross-trophy median slope (a trophy with four
    earners has a noisy slope; the field's typical slope is evidence);
  * the median rate is forecast earners / forecast players, blended
    halfway toward a direct logit-of-rate trend fit on the same recent
    states, which hedges the split when the two disagree. Both fits use
    only the recent states and damp their extrapolation, because
    launch-hour growth rates do not survive the week;
  * the spread comes from the earner fit's residuals plus a horizon
    term, so quiet trophies get narrow bands and jumpy ones wide ones.

When the history holds a single page state (the practice cases: one
stale launch-night page), no trend is observable and the page is known
to be stale: a refresh before the target is overdue, the player count
will jump, and earners lag the influx. That case uses a dilution prior
instead of a fit: players multiply, earners multiply by a power of that
factor (the power rises with the trophy's rate, because a trophy almost
everyone has is earned by newcomers almost at once), and the band is
wide, because neither the refresh's size nor its timing is known.

Quantiles are Gaussian on the logit scale around the median, the scale
the contest scores on. Output rows are rounded to 4 decimals, half up.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

TIME_SCALE_H = 8.0       # log-time scale: u = ln(1 + hours / TIME_SCALE_H)
RECENCY_HALF_LIFE_H = 30.0
FIT_WINDOW_H = 48.0      # slopes are fitted on the recent states only:
                         # launch-hour growth rates do not survive the week
SLOPE_SHRINK_STATES = 2.0
EXTRAP_DAMP = 0.65       # growth bends: apply only part of the fitted
                         # log-time distance to the target
LOGIT_BLEND = 0.50
MEDIAN_SHIFT_CAP = 1.6
SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.50
STALE_PLAYERS_JUMP = 4.2
STALE_SIGMA = 0.95


def clamp_pct(x):
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0.0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_slot(text):
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def log_time(hours):
    return math.log(1.0 + max(hours, 0.0) / TIME_SCALE_H)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, residual sd)."""
    sw = sum(ws)
    if sw <= 0.0:
        return (ys[0] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    if len(xs) >= 3:
        var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = 0.0
    return a, b, sd


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_states(history):
    """Collapse repeated reads of an unchanged page to one state each."""
    states = []
    for read in history:
        key = (read.get("players_tracked"), tuple(
            -1.0 if r is None else r for r in read["rates"]))
        if states and states[-1][0] == key:
            continue
        states.append((key, read))
    return [read for _, read in states]


def stale_forecast(rate, horizon_minutes):
    """One page state only: an overdue refresh dilutes every rate."""
    if rate is None:
        rate = 0.0
    beta = 0.55 + 0.40 * (clamp_pct(rate) / 100.0)
    ratio = STALE_PLAYERS_JUMP ** (beta - 1.0)
    med = clamp_pct(rate * ratio) if rate > 0.0 else 0.01
    if rate >= 99.99:
        med = 93.0
    return logit_pct(med), STALE_SIGMA


def forecast_case(case, name):
    if list(case["levels"]) != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    states = distinct_states(history)
    t0 = parse_slot(states[0]["slot"])
    times_h = [(parse_slot(s["slot"]) - t0) / 3600.0 for s in states]
    last = states[-1]
    last_rates = last["rates"]
    t_last = times_h[-1]
    t_target = t_last + (parse_slot(case["at"]) - parse_slot(last["slot"])) / 3600.0 \
        + case["horizon_minutes"] / 60.0
    u_last = log_time(t_last)
    u_target = log_time(t_target)
    du = max(u_target - u_last, 0.0)

    if len(states) < 2:
        rows = []
        for i, trophy in enumerate(trophies):
            med_logit, sigma = stale_forecast(last_rates[i], case["horizon_minutes"])
            rows.append({"id": trophy["id"], "q": [
                round4(inv_logit_pct(med_logit + z * sigma)) for z in Z]})
        return {"trophies": rows}

    weights = [0.5 ** (max(t_last - t, 0.0) / RECENCY_HALF_LIFE_H) for t in times_h]
    us = [log_time(t) for t in times_h]
    recent = [k for k, t in enumerate(times_h) if t_last - t <= FIT_WINDOW_H]
    if len(recent) < 3:
        recent = list(range(len(states)))

    # Players: log count on log-time over the recent states, extrapolated
    # with damping and sanity-capped.
    players = [max(float(s["players_tracked"] or 1), 1.0) for s in states]
    a_p, b_p, _ = wls([us[k] for k in recent],
                      [math.log(players[k]) for k in recent],
                      [weights[k] for k in recent])
    anchor_p = 0.5 * (a_p + b_p * u_last) + 0.5 * math.log(players[-1])
    ln_p_target = anchor_p + b_p * du * EXTRAP_DAMP
    ln_p_target = min(max(ln_p_target, math.log(players[-1] * 0.98)),
                      math.log(players[-1] * 12.0))
    p_target = math.exp(ln_p_target)

    # Earners per trophy: fit each on the recent states, shrink slopes
    # to the field's median.
    n_tr = len(trophies)
    fits = []
    slopes = []
    for i in range(n_tr):
        xs, ys, ws = [], [], []
        for k in recent:
            r = states[k]["rates"][i]
            if r is None:
                continue
            xs.append(us[k])
            ys.append(math.log(r / 100.0 * players[k] + 0.5))
            ws.append(weights[k])
        if len(xs) >= 2:
            a, b, sd = wls(xs, ys, ws)
        else:
            a, b, sd = (ys[0] if ys else math.log(0.5)), 0.0, 0.0
        fits.append((a, b, sd, xs, ys))
        slopes.append(b)
    slope_med = median(slopes)
    shrink = len(recent) / (len(recent) + SLOPE_SHRINK_STATES)

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = last_rates[i]
        base_logit = logit_pct(rate_last if rate_last is not None else 0.0)
        a, b, sd, xs, ys = fits[i]
        b_shrunk = shrink * b + (1.0 - shrink) * slope_med
        if ys:
            anchor = 0.5 * (a + b * u_last) + 0.5 * ys[-1]
            ln_e_target = anchor + b_shrunk * du * EXTRAP_DAMP
            ln_e_target = max(ln_e_target, math.log(max(
                (rate_last or 0.0) / 100.0 * players[-1] * 0.85, 0.0) + 0.4))
            med_rate = clamp_pct(100.0 * math.exp(ln_e_target) / p_target)
        else:
            med_rate = clamp_pct(rate_last if rate_last is not None else 0.0)
        med_logit = logit_pct(med_rate)

        # Direct logit trend on the same recent states, slopes shrunk
        # toward the field's median logit slope, blended with the
        # earner/players split above.
        if len(recent) >= 3:
            lx, ly, lw = [], [], []
            for k in recent:
                r = states[k]["rates"][i]
                if r is None:
                    continue
                lx.append(us[k])
                ly.append(logit_pct(r))
                lw.append(weights[k])
            if len(lx) >= 3:
                la, lb, _ = wls(lx, ly, lw)
                anchor_l = 0.5 * (la + lb * u_last) + 0.5 * ly[-1]
                trend_logit = anchor_l + lb * du * EXTRAP_DAMP * shrink
                med_logit = (1.0 - LOGIT_BLEND) * med_logit + LOGIT_BLEND * trend_logit

        shift = med_logit - base_logit
        if shift > MEDIAN_SHIFT_CAP:
            med_logit = base_logit + MEDIAN_SHIFT_CAP
        elif shift < -MEDIAN_SHIFT_CAP:
            med_logit = base_logit - MEDIAN_SHIFT_CAP

        sigma = math.sqrt(0.22 ** 2 + min(sd, 1.0) ** 2 + (0.30 * math.sqrt(horizon_days)) ** 2)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        qs = [round4(inv_logit_pct(med_logit + z * sigma)) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
