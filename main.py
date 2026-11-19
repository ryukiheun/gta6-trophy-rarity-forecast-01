#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase.

Reads every ``*.in`` case in the input directory (default ``/input``) and
writes one ``<id>.out`` forecast per case to the output directory (default
``/output``).  Python 3.13 standard library only, deterministic, no network.

Model (per trophy, in counts -- not in rates)
---------------------------------------------
A rate is ``earners / players_tracked``.  Exophase does not refresh its
counts continuously: the page can sit unchanged for hours and then jump
(launch night: 638 tracked players for ~4.5 h, then 3,868 at the next
refresh, with every rate but the 100% one falling).  Extrapolating the rate
itself therefore misses the refresh dilution.  Instead we forecast the two
counts separately and divide:

* ``players_target`` -- the tracked-player count is a step function of a
  smoothly growing true population.  Distinct player levels in the history
  give a log-linear growth rate ``g`` (per hour): least squares of ln P
  on hours since the first level appeared, cross-checked against the
  endpoint slope (the smaller is taken when they disagree by more than
  50%, guarding against one outlier level), clamped to [0, 0.12]/h.
  The current level is extrapolated from the moment it first appeared,
  so a stale level (long since its last refresh) already implies growth
  by the target: ``span_h`` is target minus level-start, and
  ``mult = exp(g * span_h)``, clamped to [1, 12] times ``P_last``.
  The 12x cap is on the multiplier of ``P_last`` (not of the first
  level); on the 4.52-day final horizon the OLS rate usually implies
  more than 12x, so the cap binds and the forecast sits at 12x.
  When the history shows only one level (no refresh ever seen, as on
  launch night), ``g`` falls back to a launch prior: 0.38/h within 36 h
  of release, 0.05/h within 120 h, 0.02/h later.  The prior is only
  valid for hours after release, so in the prior branch ``span_h`` is
  capped at 24 h (the practice cases' spans are 4.65 h and 5.15 h and are unaffected).
  The exponent is also capped at ln(12) before ``math.exp``: ``exp``
  overflows at ~709.78, and a level first seen far before the target
  would otherwise crash the run; the cap is behaviour-preserving
  because ``mult`` is clamped to 12 anyway.
* ``earners_target`` -- ``E_last + dP * a + (P_last - E_last) * p`` where
  ``dP`` is the forecast growth in tracked players, ``a`` is the earn
  probability of a newly tracked player, and ``p`` is the share of
  current non-earners who earn the trophy by the target.  The exact
  combination is::

      E_target = clamp(E_last + dP * a_i + (P_last - E_last) * p, 0, P_target)
      p = 1 - exp(-lam * horizon_days)

  The hazard applies only to current non-earners (``P_last - E_last``),
  so a player is never counted twice, and ``E_target`` can never exceed
  ``P_target``.  ``a_i`` is an absolute probability (a fraction 0..1),
  estimated from the history's own refresh jumps as the pooled
  ``sum(dE) / sum(dP)`` across level transitions, shrunk toward a prior
  with a pseudo-count ``shrink = 0.15 * P_last`` (new players)::

      a_i = (sum_de[i] + shrink * a_prior) / (sum_dp + shrink)

  The prior weight is ``shrink / (sum_dp + shrink)``: about 15% when
  one launch-night-sized refresh (``sum_dp`` ~ 3,230, ``P_last`` ~
  3,868) has been seen, and it stays near 13-17% as both ``sum_dp``
  and ``P_last`` grow together -- it does NOT shrink toward zero as
  more refreshes accumulate.  The prior itself is a ratio times the
  incumbent rate: ``a_prior = kappa * r_frac`` with
  ``kappa = 0.067 + 0.933 * r_frac**2``, so ``kappa`` is ~0.067 for
  ordinary trophies and rises to 1.0 for a trophy everybody holds.
  The prior comes from the disclosed launch-night refresh totals: the
  tracked count jumped 638 -> 3,868 and trophies earned 943 -> 4,276,
  while the one trophy at 100% stayed at 100% -- so that trophy alone
  accounts for 3,230 of the 3,333 new earners, and the other 54
  trophies gained only 103 earner-increases between them (103 is the
  sum over the 54 trophies of their per-trophy earner increases, not
  103 distinct people).  Here 0.478 is the sum of the 54 ordinary
  trophies' rates as fractions at the last pre-refresh read
  (2026-11-19 02:30Z: rates 0.63-2.04%, sum 0.4778), so
  103 / (3230 * 0.478) = 0.067.  Consistency check (not independent
  proof -- the totals are the input): rebuilding the totals with
  kappa 0.067 gives 943 + 3,230 + 3,230 * 0.067 * 0.478 = 4,276.4,
  i.e. the published 4,276 to the nearest whole earner.
  Dilution, not progression, is what moves launch-night rates: the
  other trophies' rates fell to ~22% of their launch values, not ~75%
  (the 75% figure is the artefact of averaging in the 100% trophy).
* The median rate is ``100 * earners_target / players_target``.
* Progression hazard: ``lam = 0.20 * tier_f * sqrt(r_frac)`` per day,
  tier_f platinum 0.3, gold 0.6, silver 0.8, bronze 1.0, ``r_frac`` the
  last rate as a fraction (not percent).  E.g. a bronze trophy at
  0.85% has lam = 0.018/day: 90-minute hazard 0.001 (negligible),
  4.52-day hazard 0.08.  lam = 0.20 is a prior constant, not
  identified from the history.
* Dilution guard: the median is clamped to [0.12 * r, max(5 * r, r + 2)]
  where r is the last rate in percent (tier prior when unknown).  One
  shared ``P_target`` error otherwise moves all 55 medians the same
  way; the guard bounds that shared error.  It does not bind on the
  practice cases (median/last ratios ~0.22-0.25).

Edge-case rules (all verified by tests, see the design document):
empty history -> tier-prior medians (platinum 0.5, gold 1.0, silver
1.5, bronze 2.0%) with s = 1.2; one read -> the launch-prior branch;
null rates are skipped (the last known rate is kept); reads with
``players_tracked`` <= 0 or non-finite are skipped; transitions with
dP <= 0 are skipped; non-finite rates inside a transition are skipped
(a NaN rate would otherwise poison the pooled estimate and the output
would contain NaN, which ``json.dump`` writes as the non-standard
literal ``NaN``); ``E_target`` is clamped to [0, ``P_target``]; the
median is clamped to [0.005, 99.995]; quantiles are clamped to
[0, 100] and forced non-decreasing after rounding to 4 decimals.
The 100% trophy therefore comes out at ~99.98-99.995, never 100; a
0% trophy has r = 0, hence lam = 0 and a_prior = 0, so its median
stays at the 0.005 floor unless a transition shows earners.

Spread (logit scale, ``q_tau = inv_logit(logit(m) + z_tau * s)``)
------------------------------------------------------------------
``s`` combines a random-walk term ``0.5 * sqrt(d)`` (the house
persistence baseline's width; ``d`` is the horizon in days,
``horizon_minutes / 1440``), a refresh term ``s_ref`` (0 transitions
-> 0.45, 1-2 transitions -> 0.28, 3 or more -> 0.20) and a count-noise
term ``0.35 / sqrt(E + 2)`` where ``E`` is the trophy's current earner
count ``e_last`` and ``m`` is the median rate in percent.  ``s`` is
clamped to [0.12, 1.5].  A flat history with a refresh ahead is exactly
what made the persistence baseline's 90% coverage 0 on the practice
cases, so the spread is deliberately wide enough to cover a refresh
dilution.  Calibration note: if the quantiles were exactly correct for
a logit-normal with sd ``s``, the mean pinball loss (with the factor
2) would be ``2 * s * mean(phi(z_tau))`` = 0.496 * s -- about 0.55 at
s = 1.1 (final horizon) and about 0.12 at s = 0.24 (practice horizon),
verified analytically and by simulation; the loss floor is set by the
spread itself, not by median error.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS_Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Dilution guard: bound on the median's move from the last rate (see
# module docstring).  Tunable for backtests; 0.12 / 5.0 / 2.0 do not
# bind on the practice cases.
GUARD_FLOOR = 0.12
GUARD_CAP_MULT = 5.0
GUARD_CAP_ADD = 2.0
TIER_FACTOR = {"platinum": 0.3, "gold": 0.6, "silver": 0.8, "bronze": 1.0}
TIER_PRIOR_RATE = {"platinum": 0.5, "gold": 1.0, "silver": 1.5, "bronze": 2.0}


def parse_time(s):
    """Parse an ISO-8601 timestamp; return an aware datetime or None."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x)) if x < 700 else 100.0
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = max(horizon_min, 0.0) / 1440.0
    horizon_h = max(horizon_min, 0.0) / 60.0

    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if target is None and at is not None:
        from datetime import timedelta

        target = at + timedelta(minutes=horizon_min)

    # Release start, for the flat-history growth prior.
    release = None
    game = case.get("game") or {}
    if game.get("release_date"):
        release = parse_time(str(game["release_date"]) + "T00:00:00Z")

    n = len(trophies)
    # Last non-null rate per trophy, and the last usable read.
    last_rate = [None] * n
    last_players = None
    reads = []  # (time, players, rates list) in history order
    for h in history:
        if not isinstance(h, dict):
            continue
        t = parse_time(h.get("slot") or h.get("at"))
        p = h.get("players_tracked")
        rates = h.get("rates") or []
        if p is None or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
            continue
        p = float(p)
        last_players = p
        reads.append((t, p, rates))
        for i in range(min(n, len(rates))):
            r = rates[i]
            if isinstance(r, (int, float)) and math.isfinite(r):
                last_rate[i] = float(r)

    if not reads or last_players is None:
        # No usable history: tier prior medians, very wide spread.
        out = []
        for tr in trophies:
            m = TIER_PRIOR_RATE.get(str(tr.get("tier", "")).lower(), 1.0)
            out.append((tr.get("id"), _quantiles(m, 1.2)))
        return out

    p_last = last_players
    # Distinct player levels: (first-seen time, players, rates snapshot).
    levels = []
    for t, p, rates in reads:
        if not levels or p != levels[-1][1]:
            levels.append((t, p, rates))
    n_transitions = max(0, len(levels) - 1)

    # --- players growth rate g (per hour, log scale) -------------------
    prior_branch = False
    if len(levels) >= 2 and levels[0][0] is not None:
        t0 = levels[0][0]
        xs = [(lv[0] - t0).total_seconds() / 3600.0 for lv in levels]
        ys = [math.log(lv[1]) for lv in levels]
        g = ols_slope(xs, ys)
        # Endpoint slope as a sanity cross-check; take the smaller when
        # they disagree strongly (guards one outlier level).
        if xs[-1] > 0:
            g_end = (ys[-1] - ys[0]) / xs[-1]
            if g > 0 and g_end > 0:
                g = min(g, g_end) if abs(g - g_end) > 0.5 * abs(g) else g
        g = clamp(g, 0.0, 0.12)
    else:
        # No refresh ever observed: launch prior.  Growth is explosive
        # in the first day after release and settles afterwards.
        prior_branch = True
        age_h = None
        if release is not None and at is not None:
            age_h = (at - release).total_seconds() / 3600.0
        if age_h is None:
            age_h = 3.0
        if age_h < 36.0:
            g = 0.38
        elif age_h < 120.0:
            g = 0.05
        else:
            g = 0.02

    # Extrapolate the current level from when it first appeared: the
    # level is as old as (at - first seen), and the target is a further
    # horizon beyond the checkpoint.
    level_start = levels[-1][0]
    if level_start is not None and target is not None:
        span_h = (target - level_start).total_seconds() / 3600.0
    else:
        span_h = horizon_h
    span_h = max(span_h, 0.0)
    if prior_branch:
        # The launch prior (0.38/h) is only valid for hours after
        # release: a level first seen long before the target must not
        # be extrapolated at 0.38/h over its whole age.  The practice
        # cases' spans (4.65 h and 5.15 h) are unaffected by the 24 h cap.
        span_h = min(span_h, 24.0)
    # Cap the exponent before math.exp: exp() overflows at ~709.78 and
    # a level first seen far before the target would crash the run.
    # Behaviour-preserving: mult is clamped to 12 below anyway.
    mult = math.exp(min(g * span_h, math.log(12.0)))
    mult = clamp(mult, 1.0, 12.0)
    p_target = p_last * mult
    dp = max(p_target - p_last, 0.0)

    # --- new-cohort earn probability a_i -------------------------------
    # Pooled dE/dP over refresh transitions, shrunk toward the prior.
    sum_dp = 0.0
    sum_de = [0.0] * n
    for k in range(1, len(levels)):
        _, p_prev, r_prev = levels[k - 1]
        _, p_cur, r_cur = levels[k]
        d = p_cur - p_prev
        if d <= 0:
            continue
        sum_dp += d
        for i in range(n):
            a = r_prev[i] if i < len(r_prev) else None
            b = r_cur[i] if i < len(r_cur) else None
            if (isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and math.isfinite(a) and math.isfinite(b)):
                sum_de[i] += (b * p_cur - a * p_prev) / 100.0
    shrink = 0.15 * p_last  # pseudo-count in units of new players

    # --- spread ---------------------------------------------------------
    s_rw = 0.5 * math.sqrt(horizon_days)
    if n_transitions == 0:
        s_ref = 0.45
    elif n_transitions < 3:
        s_ref = 0.28
    else:
        s_ref = 0.20

    out = []
    for i, tr in enumerate(trophies):
        r = last_rate[i]
        if r is None:
            r = TIER_PRIOR_RATE.get(str(tr.get("tier", "")).lower(), 1.0)
        r_frac = clamp(r, 0.0, 100.0) / 100.0
        e_last = r_frac * p_last

        # Prior for a newly tracked player's earn probability: ~7% of
        # the incumbents' rate for ordinary trophies (launch-night
        # refresh arithmetic, see module docstring), rising to 100%
        # for near-universal trophies (the launch-night 100% trophy
        # stayed 100% through the refresh).
        kappa = 0.067 + 0.933 * r_frac * r_frac
        a_prior = clamp(kappa * r_frac, 0.0, 1.0)
        if sum_dp > 0:
            a_i = (sum_de[i] + shrink * a_prior) / (sum_dp + shrink)
        else:
            a_i = a_prior
        a_i = clamp(a_i, 0.0005, 1.0)

        # Existing-player progression: saturating hazard, faster for
        # easy/early trophies, slower for platinum/gold/silver.
        tier_f = TIER_FACTOR.get(str(tr.get("tier", "")).lower(), 1.0)
        lam = 0.20 * tier_f * math.sqrt(r_frac)  # per day
        prog = 1.0 - math.exp(-lam * horizon_days)
        e_prog = (p_last - e_last) * prog

        e_target = e_last + dp * a_i + e_prog
        e_target = clamp(e_target, 0.0, p_target)
        m = 100.0 * e_target / p_target if p_target > 0 else r
        # Dilution guard: one shared P_target error must not drag all
        # medians arbitrarily far from the last observed rates.
        m = clamp(m, GUARD_FLOOR * r, max(GUARD_CAP_MULT * r, r + GUARD_CAP_ADD))
        if not math.isfinite(m):
            m = clamp(r, 0.005, 99.995)
        m = clamp(m, 0.005, 99.995)

        s_cnt = 0.35 / math.sqrt(e_last + 2.0)
        s = math.sqrt(s_rw * s_rw + s_ref * s_ref + s_cnt * s_cnt)
        s = clamp(s, 0.12, 1.5)
        out.append((tr.get("id"), _quantiles(m, s)))
    return out


def _quantiles(median_rate, s):
    base = logit(median_rate / 100.0)
    qs = []
    prev = None
    for z in LEVELS_Z:
        q = inv_logit(base + z * s)
        q = round(clamp(q, 0.0, 100.0), 4)
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def main(argv):
    in_dir = None
    out_dir = None
    if len(argv) > 1:
        in_dir = argv[1]
    if len(argv) > 2:
        out_dir = argv[2]
    in_dir = in_dir or os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = out_dir or os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        payload = {
            "trophies": [
                {"id": tid, "q": qs} for tid, qs in rows
            ]
        }
        out_path = os.path.join(out_dir, fn[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
