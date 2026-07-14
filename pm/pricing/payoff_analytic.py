"""Analytic at-expiry statistics over the piecewise-linear payoff.

The at-expiry NET P&L of an option structure is exactly piecewise-linear in the
terminal spot, with kinks only at strikes. This module computes the strategy
statistics — breakevens, bounded-vs-unbounded max profit and loss with their
attainment regions, profit intervals, and the closed-form lognormal probability
of profit — directly from the kink set ``{0} ∪ strikes ∪ tail slopes``. There is
no spot grid anywhere: nothing is window-clipped, so a breakeven or plateau at
any distance from spot is found exactly, and an empty breakeven list truthfully
means the curve never crosses zero.

It supersedes the grid-sampled statistics in :mod:`pm.pricing.payoff_risk`
(``strategy_breakevens`` / ``strategy_max_profit_loss`` and the interval
location inside ``pop_lognormal``), which stay frozen, behavior-identical, for
the byte-identical regression gate. Leg contract is the toolkit's: option legs
``{opt_type: 'Call'|'Put', K, qty (signed contracts), mid (entry premium/share,
positive magnitude)}``; stock legs ``{opt_type: 'Stock', qty (signed shares),
cost_basis (per share)}``. Option intrinsic and premium carry the 100-share
contract multiplier; stock legs are per-share.

Conventions (deliberate, pinned by tests):

* An exact-zero at an isolated kink counts as ONE breakeven iff the flanking
  signs differ; a maximal identically-zero plateau whose flanking signs differ
  emits ONE breakeven at its positive-side boundary (where P&L first becomes
  strictly positive). A zero-touch with same-sign flanks is not a crossing.
* Profit means strictly positive P&L (a zero plateau contributes no PoP mass).
* ``always_profitable`` / ``always_loss``: the curve has no zero-crossing and a
  single sign (a zero-touch counts as its flanking sign).

Import surface: stdlib ``math`` + the shared normal CDF — no scipy, no pricing
engines, no numpy, no pandas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pm.pricing.conventions import norm_cdf

_KINK_TOL = 1e-9          # merge strikes closer than this (relative to size)
_CONTRACT_MULT = 100.0


@dataclass(frozen=True)
class PWLPayoff:
    """The exact piecewise-linear at-expiry net P&L: values at the kink set
    ``{0} ∪ strikes`` plus the two tail slopes. ``kinks[0]`` is always 0.0."""
    kinks: tuple          # sorted spots, kinks[0] == 0.0
    values: tuple         # net P&L at each kink (exact)
    slope_low: float      # d(P&L)/dS on [0, first strike)
    slope_high: float     # d(P&L)/dS on (last strike, +inf)


# ---------------------------------------------------------------------------
# Construction + evaluation
# ---------------------------------------------------------------------------

def _leg_pnl_at(legs, spot: float) -> float:
    """Exact net P&L of the combined legs at one terminal spot."""
    total = 0.0
    for leg in legs:
        qty = float(leg.get("qty", 0.0))
        opt_type = leg.get("opt_type")
        if opt_type == "Call":
            k = float(leg["K"])
            mid = float(leg.get("mid") or 0.0)
            total += qty * max(spot - k, 0.0) * _CONTRACT_MULT
            total -= qty * mid * _CONTRACT_MULT
        elif opt_type == "Put":
            k = float(leg["K"])
            mid = float(leg.get("mid") or 0.0)
            total += qty * max(k - spot, 0.0) * _CONTRACT_MULT
            total -= qty * mid * _CONTRACT_MULT
        elif opt_type == "Stock":
            total += qty * (spot - float(leg["cost_basis"]))
        else:
            raise ValueError(
                f"payoff_analytic: unknown opt_type {opt_type!r} "
                f"(expected 'Call', 'Put', or 'Stock')"
            )
    return total


def pwl_from_legs(legs) -> PWLPayoff:
    """Build the exact PWL description from combined leg dicts.

    Raises ValueError on an option leg without a strike (parity with the grid
    toolkit, which cannot price it either)."""
    strikes = []
    stock_qty = 0.0
    net_call_qty = 0.0
    net_put_qty = 0.0
    for leg in legs:
        qty = float(leg.get("qty", 0.0))
        opt_type = leg.get("opt_type")
        if opt_type in ("Call", "Put"):
            k = leg.get("K")
            if k is None:
                raise ValueError("payoff_analytic: option leg without a strike")
            strikes.append(float(k))
            if opt_type == "Call":
                net_call_qty += qty
            else:
                net_put_qty += qty
        elif opt_type == "Stock":
            stock_qty += qty
        else:
            raise ValueError(
                f"payoff_analytic: unknown opt_type {opt_type!r} "
                f"(expected 'Call', 'Put', or 'Stock')"
            )

    kinks = [0.0]
    for k in sorted(strikes):
        if k > 0.0 and k - kinks[-1] > _KINK_TOL * max(1.0, k):
            kinks.append(k)
    values = tuple(_leg_pnl_at(legs, x) for x in kinks)
    slope_high = stock_qty + _CONTRACT_MULT * net_call_qty
    slope_low = stock_qty - _CONTRACT_MULT * net_put_qty
    return PWLPayoff(kinks=tuple(kinks), values=values,
                     slope_low=slope_low, slope_high=slope_high)


def pwl_value(pwl: PWLPayoff, spot: float) -> float:
    """Exact P&L at any spot ≥ 0 (linear interpolation between kinks; affine
    extension beyond the last kink)."""
    xs, vs = pwl.kinks, pwl.values
    if spot >= xs[-1]:
        return vs[-1] + pwl.slope_high * (spot - xs[-1])
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:                      # bisect the bracketing segment
        mid = (lo + hi) // 2
        if xs[mid] <= spot:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    t = 0.0 if x1 == x0 else (spot - x0) / (x1 - x0)
    return vs[lo] + t * (vs[hi] - vs[lo])


# ---------------------------------------------------------------------------
# Breakevens
# ---------------------------------------------------------------------------

def _sign(v: float) -> int:
    return 0 if v == 0.0 else (1 if v > 0.0 else -1)


def pwl_breakevens(pwl: PWLPayoff) -> list:
    """Every spot where the P&L crosses zero, exactly — segment crossings, the
    unbounded right tail, and the documented exact-zero / zero-plateau
    convention. Complete by construction: an empty list means no crossing."""
    xs, vs = pwl.kinks, pwl.values
    n = len(xs)
    tail_sign = _sign(pwl.slope_high) or _sign(vs[-1])

    bes = []
    i = 0
    while i < n:
        v = vs[i]
        if v == 0.0:
            j = i                                   # maximal zero run [i, j]
            while j + 1 < n and vs[j + 1] == 0.0:
                j += 1
            prev_sign = _sign(vs[i - 1]) if i > 0 else 0
            if j == n - 1:
                next_sign = _sign(pwl.slope_high)   # run reaches the tail
            else:
                next_sign = _sign(vs[j + 1])
            if prev_sign * next_sign < 0:
                bes.append(xs[j] if next_sign > 0 else xs[i])
            i = j + 1
            continue
        if i + 1 < n and v * vs[i + 1] < 0.0:       # strict segment crossing
            t = -v / (vs[i + 1] - v)
            bes.append(xs[i] + t * (xs[i + 1] - xs[i]))
        i += 1

    # Right tail: affine beyond the last kink.
    if vs[-1] != 0.0 and vs[-1] * tail_sign < 0 and pwl.slope_high != 0.0:
        bes.append(xs[-1] - vs[-1] / pwl.slope_high)

    bes.sort()
    deduped = []
    for b in bes:
        if not deduped or b - deduped[-1] > _KINK_TOL * max(1.0, b):
            deduped.append(b)
    return deduped


# ---------------------------------------------------------------------------
# Max profit / max loss (kink evaluation + tail-slope sign)
# ---------------------------------------------------------------------------

def _attain_region(xs, vs, target, slope_high):
    """Contiguous attainment region (lo, hi) for an extremum over the kink set;
    hi is math.inf when the flat right tail holds the extremum. Falls back to
    the first attainment point when the attainment set is not contiguous."""
    idx = [i for i, v in enumerate(vs) if v == target]
    lo, hi = xs[idx[0]], xs[idx[-1]]
    contiguous = idx == list(range(idx[0], idx[-1] + 1))
    if not contiguous:
        lo = hi = xs[idx[0]]
    if slope_high == 0.0 and vs[-1] == target and (contiguous or lo == xs[-1]):
        hi = math.inf
    return (lo, hi)


def pwl_max_profit_loss(pwl: PWLPayoff) -> dict:
    """Exact max profit / max loss with unbounded flags, attainment regions,
    and the constant-sign flags for a curve that never crosses zero."""
    xs, vs = pwl.kinks, pwl.values
    unbounded_gain = pwl.slope_high > 0.0
    unbounded_loss = pwl.slope_high < 0.0

    vmax, vmin = max(vs), min(vs)
    has_neg = vmin < 0.0 or unbounded_loss
    has_pos = vmax > 0.0 or unbounded_gain

    return {
        "max_profit": None if unbounded_gain else vmax,
        "max_loss": None if unbounded_loss else vmin,
        "max_profit_region": (None if unbounded_gain
                              else _attain_region(xs, vs, vmax, pwl.slope_high)),
        "max_loss_region": (None if unbounded_loss
                            else _attain_region(xs, vs, vmin, pwl.slope_high)),
        "unbounded_gain": unbounded_gain,
        "unbounded_loss": unbounded_loss,
        "always_profitable": (not has_neg) and has_pos,
        "always_loss": (not has_pos) and has_neg,
    }


# ---------------------------------------------------------------------------
# Profit intervals + closed-form lognormal PoP
# ---------------------------------------------------------------------------

def profit_intervals(pwl: PWLPayoff) -> list:
    """Maximal open intervals of [0, ∞) where P&L is strictly positive, as
    (lo, hi) with hi possibly math.inf. Interval boundaries are exact (kinks,
    breakevens, zero-plateau edges); intervals separated by a single zero-touch
    point are merged (the touch has zero probability mass)."""
    xs, vs = pwl.kinks, pwl.values
    bes = pwl_breakevens(pwl)

    critical = sorted(set(list(xs) + bes))
    intervals = []
    for a, b in zip(critical, critical[1:]):
        if pwl_value(pwl, 0.5 * (a + b)) > 0.0:
            intervals.append([a, b])
    # Final piece: beyond the last critical point the sign is constant.
    last = critical[-1]
    tail_positive = (pwl.slope_high > 0.0
                     or (pwl.slope_high == 0.0 and vs[-1] > 0.0))
    if tail_positive:
        intervals.append([last, math.inf])

    merged = []
    for iv in intervals:
        if merged and iv[0] <= merged[-1][1]:
            merged[-1][1] = iv[1]
        else:
            merged.append(iv)
    return [(a, b) for a, b in merged]


def pop_lognormal_intervals(spot, sigma, T, r, q, intervals) -> float:
    """Probability of profit: lognormal risk-neutral mass over the given profit
    intervals (the same distribution and CDF as the grid toolkit's
    ``pop_lognormal`` — only the interval location differs, being analytic).
    Returns NaN when T ≤ 0 / sigma ≤ 0 / spot ≤ 0."""
    if not (T > 0.0) or not (sigma > 0.0) or not (spot > 0.0):
        return float("nan")
    sigma_sqt = sigma * math.sqrt(T)
    drift = (r - q - 0.5 * sigma * sigma) * T

    def _ln_cdf(x: float) -> float:
        if x <= 0.0:
            return 0.0
        if math.isinf(x):
            return 1.0
        return float(norm_cdf((math.log(x / spot) - drift) / sigma_sqt))

    pop = 0.0
    for a, b in intervals:
        pop += _ln_cdf(b) - _ln_cdf(a)
    return max(0.0, min(1.0, pop))
