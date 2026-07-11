"""Objective + client-fit ranking for the priced scanner candidates.

Ranks the roll / overlay candidates the generation layer already priced, so the
best fit for the desk surfaces first WITH a transparent reason — the list is
ordered, never reduced to a single recommendation. Two ingredients combine:

* **Objective-fit** — for the active objective (roll-for-credit, max-premium,
  extend-duration, defend-cut-delta, add-hedge) one driver metric, converted to a
  within-set percentile so metrics on different scales (dollars, days, delta)
  become comparable. Roll-for-credit and max-premium additionally earn a small
  IV-richness bonus (selling a rich short leg is the point of the trade). IV-rank
  is name-level context, shown upstream, never a per-candidate differentiator.

* **Client-fit** — a read of the account's ``ClientProfile`` (tenor preference,
  strategy posture) that NUDGES, never filters. It degrades to neutral on a thin
  or absent profile rather than inventing a fit, and its weight scales with the
  profile's coverage confidence, so a shallow history can never re-order the book.

Combination: ``final = 0.7·objective_fit + w_client·client_fit`` with
``w_client = 0.3 × {low:0, medium:0.5, high:1}[coverage.band]``. A neutral client-fit
(0.5) is a constant across the set, so it is order-preserving by construction — a
thin profile falls back to pure objective order automatically.

Pure and read-only: no Bloomberg, no recompute, no state writes. The caller hands
in the priced candidates, the account profile, the slice IV+pp rows, and the
held-leg context; ranking returns the ordered list with a plain-English reason
per row and a flag for the over-extends / degraded cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from pm.candidates.generate import (
    ADD_HEDGE,
    COSTLESS,
    DEFEND_CUT_DELTA,
    EXTEND_DURATION,
    MAX_PREMIUM,
    ROLL_FOR_CREDIT,
    ROLL_UP_OUT,
)

# Combination weights + the coverage-confidence damping on the client nudge.
_W_OBJ = 0.7
_W_CLIENT = 0.3
_BAND_MULT = {"low": 0.0, "medium": 0.5, "high": 1.0}

# IV-richness bonus: proportional to the short leg's within-set richness percentile,
# capped so it only re-orders near-ties (never inverts an obviously-better driver).
_IV_BONUS_MAX = 0.15
_IV_BONUS_OBJECTIVES = (ROLL_FOR_CREDIT, MAX_PREMIUM)

# Tenor fit by bucket distance (same / adjacent / farther); over-extends is flagged,
# never excluded, when the candidate runs past 1.5× the client's median tenor.
_TENOR_FIT = {0: 1.0, 1: 0.5}
_TENOR_FIT_FAR = 0.2
_BUCKET_IDX = {"short": 0, "swing": 1, "leaps": 2}
_OVEREXTEND_MULT = 1.5


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if (f == f and math.isfinite(f)) else None   # drop NaN / inf


def _money(x) -> str:
    if x is None:
        return "n/a"
    s = f"${abs(x):,.0f}"
    return ("+" + s) if x >= 0 else ("-" + s)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pctl(p) -> str:
    return f"{_ordinal(round(p * 100))} pctl" if p is not None else "n/a"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class RankedCandidate:
    """One priced candidate placed in the order, with its score decomposed and a
    reason a trader can read. ``rank == 1`` is the recommended default; the rest are
    the alternatives. ``score`` is None only when the objective's driver is
    unavailable (that candidate sorts last, its reason states why)."""
    candidate: object
    rank: int = 0
    score: Optional[float] = None
    objective_fit: Optional[float] = None
    client_fit: float = 0.5
    iv_richness_pct: Optional[float] = None
    over_extends: bool = False
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalization — tie-aware average-rank percentile
# ---------------------------------------------------------------------------

def _avg_rank_percentile(values) -> list:
    """Map each finite value to a within-set percentile in [0, 1] via tie-aware
    average rank: ``p(x) = (#{v < x} + 0.5·#{v == x}) / m`` over the m finite values.

    All-equal -> 0.5 for every member; a lone finite value -> 1.0 ("best available");
    a None / non-finite value -> None (unavailable, the caller sorts it last)."""
    out: list = [None] * len(values)
    idx = [i for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    m = len(idx)
    if m == 0:
        return out
    if m == 1:
        out[idx[0]] = 1.0
        return out
    finite = [values[i] for i in idx]
    for i in idx:
        x = values[i]
        less = sum(1 for v in finite if v < x)
        equal = sum(1 for v in finite if v == x)
        out[i] = (less + 0.5 * equal) / m
    return out


# ---------------------------------------------------------------------------
# Objective driver + reason
# ---------------------------------------------------------------------------

# Tenor weight for the up-and-out relief driver: added days scaled into strike-$ units.
_OUT_W = 0.05


def _new_strike(cand) -> Optional[float]:
    for lg in (getattr(cand, "legs", None) or []):
        if lg.get("opt_type") in ("Call", "Put") and lg.get("K") is not None:
            return _num(lg.get("K"))
    return None


def _relief(cand, held) -> Optional[float]:
    """Up-and-out relief — strike increase plus scaled added tenor. The driver for the
    Roll up & out and Costless objectives (higher = more room bought)."""
    nk = _new_strike(cand)
    hk = _num((held or {}).get("strike"))
    dte = _num((getattr(cand, "economics", None) or {}).get("dte"))
    hdte = _num((held or {}).get("dte"))
    strike_relief = (nk - hk) if (nk is not None and hk is not None) else 0.0
    tenor = _OUT_W * ((dte - hdte) if (dte is not None and hdte is not None) else 0.0)
    return strike_relief + tenor


def _driver(cand, objective, held) -> Optional[float]:
    """The single objective driver, oriented so higher is always better."""
    if objective == MAX_PREMIUM:
        # Credit per dollar of cap surrendered (the new strike) — not raw credit.
        nc = _num(getattr(cand, "net_credit", None))
        k = _new_strike(cand)
        return (nc / k) if (nc is not None and k) else nc
    if objective in (ROLL_FOR_CREDIT, ADD_HEDGE):
        # roll-for-credit = raw credit collected; add-hedge = cheaper/financed protection.
        return _num(getattr(cand, "net_credit", None))
    if objective in (ROLL_UP_OUT, COSTLESS):
        return _relief(cand, held)
    if objective == EXTEND_DURATION:
        return _num((getattr(cand, "economics", None) or {}).get("dte"))   # more tenor
    if objective == DEFEND_CUT_DELTA:
        nd = _num(getattr(cand, "new_leg_delta", None))
        if nd is None:
            return None
        hd = _num((held or {}).get("delta"))
        if hd is None:
            return -abs(nd)                          # no held Δ: lower |Δ| is more defensive
        return abs(hd) - abs(nd)                     # delta reduction
    return _num(getattr(cand, "net_credit", None))


def _objective_reason(cand, objective, driver, pct, held) -> Optional[str]:
    if driver is None:
        return None
    if objective in (ROLL_FOR_CREDIT, MAX_PREMIUM):
        return f"{_money(driver)} net credit ({_pctl(pct)})"
    if objective == ADD_HEDGE:
        return f"{_money(driver)} to establish ({_pctl(pct)})"
    if objective == EXTEND_DURATION:
        dte = int(round(driver))
        held_dte = (held or {}).get("dte")
        added = f", +{dte - int(held_dte)}d added" if held_dte is not None else ""
        return f"{dte}d to expiry{added} ({_pctl(pct)})"
    if objective in (ROLL_UP_OUT, COSTLESS):
        nk = _new_strike(cand)
        hk = _num((held or {}).get("strike"))
        relief = f" +{nk - hk:g} strike" if (nk is not None and hk is not None) else ""
        lead = "costless roll" if objective == COSTLESS else "up & out"
        return f"{lead}{relief} ({_pctl(pct)})"
    if objective == DEFEND_CUT_DELTA:
        nd = _num(getattr(cand, "new_leg_delta", None))
        hd = _num((held or {}).get("delta"))
        if nd is not None and hd is not None:
            return f"cuts |Δ| by {abs(hd) - abs(nd):.2f} (new Δ {nd:+.2f} vs held {hd:+.2f})"
        if nd is not None:
            return f"new Δ {nd:+.2f} ({_pctl(pct)})"
    return None


# ---------------------------------------------------------------------------
# IV-richness (short-leg IV+pp)
# ---------------------------------------------------------------------------

def _short_leg_excess(cand, excess_by_ticker):
    """(iv_excess, status) for the candidate's short option leg — the leg being sold.
    status: 'ok' (found), 'no_short' (no premium-selling leg), 'not_in_slice' (the
    short leg's contract fell outside the pulled slice, so no IV+pp)."""
    shorts = [lg for lg in (getattr(cand, "legs", None) or [])
              if lg.get("opt_type") in ("Call", "Put") and (lg.get("qty") or 0) < 0]
    if not shorts:
        return None, "no_short"
    tk = shorts[0].get("position_id")
    exc = excess_by_ticker.get(tk)
    if exc is not None:
        return exc, "ok"
    return None, "not_in_slice"


def _iv_reason(status, excess, pct) -> Optional[str]:
    if status == "ok":
        return f"short leg {excess:+.1f}pp rich ({_pctl(pct)})"
    if status == "no_short":
        return "no premium leg — no IV+pp bonus"
    return "IV+pp n/a (short leg outside slice)"


# ---------------------------------------------------------------------------
# Client-fit — tenor + posture, guarded on the fragile profile
# ---------------------------------------------------------------------------

def _candidate_posture(cand) -> Optional[str]:
    """The candidate's resulting posture from its single option leg's role
    (short_call / long_call / short_put / long_put). A collar / no-option candidate
    has no single posture -> None (posture dimension skipped)."""
    opt = [lg for lg in (getattr(cand, "legs", None) or [])
           if lg.get("opt_type") in ("Call", "Put")]
    if len(opt) != 1:
        return None
    return opt[0].get("role")


def _tenor_fit(cand_dte, tenor_pref):
    """(fit in [0,1] or None, reason or None, over-extends flag or None) for the
    candidate tenor vs the client's revealed tenor preference. Bucket match is the
    robust signal; a numeric distance-decay is the fallback when only the median
    is known."""
    if cand_dte is None or tenor_pref is None:
        return None, None, None
    median = _num(getattr(tenor_pref, "median_dte_at_open", None))
    bucket = getattr(tenor_pref, "bucket", None)
    over = (f"over-extends: {int(round(cand_dte))}d vs client median {int(round(median))}d"
            if (median is not None and cand_dte > _OVEREXTEND_MULT * median) else None)
    if bucket is not None:
        from pm.insight.client_profile import _dte_bucket   # reuse the profile's own thresholds
        cand_bucket = _dte_bucket(cand_dte)
        diff = abs(_BUCKET_IDX.get(cand_bucket, 1) - _BUCKET_IDX.get(bucket, 1))
        fit = _TENOR_FIT.get(diff, _TENOR_FIT_FAR)
        reason = (f"matches {bucket} tenor" if diff == 0
                  else f"{cand_bucket} tenor vs client {bucket} ({int(round(cand_dte))}d)")
        return fit, reason, over
    if median is not None:
        fit = 1.0 / (1.0 + abs(cand_dte - median) / median)
        return fit, f"{int(round(cand_dte))}d vs client median {int(round(median))}d", over
    return None, None, over


def _posture_fit(posture, strategy_bias):
    """(fit in [0,1] or None, reason or None) from the account's own weight on the
    candidate's posture. None (dimension skipped) when there is no opening flow or
    no single posture to match."""
    if posture is None or strategy_bias is None:
        return None, None
    if getattr(strategy_bias, "n_opening", 0) == 0 or not getattr(strategy_bias, "weights", None):
        return None, None
    w = _num(strategy_bias.weights.get(posture, 0.0)) or 0.0
    label = posture.replace("_", " ")
    if w > 0:
        return w, f"matches {label} posture ({round(w * 100)}% of opens)"
    return 0.0, f"off client's posture ({label} unseen in opens)"


def _client_fit(cand, profile):
    """(client_fit in [0,1], reasons, flags, over_extends). Neutral 0.5 — the
    order-preserving fallback — when the profile is absent, thin (low band), or has
    no dimension this candidate can be scored on. Never fabricated."""
    if profile is None:
        return 0.5, ["no client profile — objective-fit only"], [], False
    band = getattr(getattr(profile, "coverage", None), "band", "low")
    if band == "low":
        return 0.5, ["thin history (low coverage) — objective-fit only"], [], False

    reasons: list = []
    flags: list = []
    dims: list = []

    cand_dte = _num((getattr(cand, "economics", None) or {}).get("dte"))
    tfit, treason, over_msg = _tenor_fit(cand_dte, getattr(profile, "tenor_pref", None))
    if tfit is not None:
        dims.append(tfit)
    if treason:
        reasons.append(treason)
    over = over_msg is not None
    if over_msg:
        flags.append(over_msg)

    pfit, preason = _posture_fit(_candidate_posture(cand), getattr(profile, "strategy_bias", None))
    if pfit is not None:
        dims.append(pfit)
    if preason:
        reasons.append(preason)

    if not dims:
        return 0.5, (reasons or ["profile too thin for this candidate — objective-fit only"]), flags, over
    return sum(dims) / len(dims), reasons, flags, over


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def _sort_key(item):
    rc, gen_index = item
    scored = rc.score is not None
    ivp = rc.iv_richness_pct
    return (
        0 if scored else 1,                                   # unavailable-driver rows last
        -(rc.score if scored else 0.0),                       # score desc
        0 if ivp is not None else 1,                          # IV-richness present first on ties
        -(ivp if ivp is not None else 0.0),
        -(abs(rc.candidate.net_credit) if getattr(rc.candidate, "net_credit", None) is not None else 0.0),
        gen_index,                                            # else preserve generation order
    )


def rank_candidates(candidates, *, objective, client_profile=None, iv_pp=None,
                    held=None) -> list:
    """Rank the priced candidates for one objective. Returns ``[RankedCandidate, ...]``
    ordered best-first (rank 1 = recommended), each carrying its score decomposition
    and a readable reason.

    ``candidates`` may contain other objectives — only those tagged ``objective`` are
    ranked. ``iv_pp`` is the slice's IV+pp rows (``[{ticker, iv_excess, ...}]``);
    ``held`` is ``{delta, dte}`` for the held leg (rolls) or None (stock overlays).
    Pure — no Bloomberg, no state writes."""
    cands = [c for c in (candidates or []) if getattr(c, "objective", None) == objective]
    if not cands:
        return []

    excess_by_ticker = {}
    for row in (iv_pp or []):
        tk = row.get("ticker")
        if tk is not None:
            excess_by_ticker[tk] = _num(row.get("iv_excess"))

    drivers = [_driver(c, objective, held) for c in cands]
    driver_pct = _avg_rank_percentile(drivers)

    is_premium = objective in _IV_BONUS_OBJECTIVES
    excess_status = [_short_leg_excess(c, excess_by_ticker) for c in cands]
    excesses = [e if is_premium else None for (e, _s) in excess_status]
    iv_pct = _avg_rank_percentile(excesses)

    band = getattr(getattr(client_profile, "coverage", None), "band", "low")
    w_client = _W_CLIENT * _BAND_MULT.get(band, 0.0)

    ranked: list = []
    for i, cand in enumerate(cands):
        reasons: list = []
        flags: list = list(getattr(cand, "warnings", None) or [])
        if getattr(cand, "economics", None) is None:
            flags.append("economics unavailable (pricing degraded)")

        # Objective-fit (+ IV bonus for premium objectives).
        primary = driver_pct[i]
        oreason = _objective_reason(cand, objective, drivers[i], primary, held)
        if oreason:
            reasons.append(oreason)

        iv_richness_pct = iv_pct[i] if is_premium else None
        if primary is None:
            objective_fit = None
            flags.append("objective driver unavailable — sorted last")
        else:
            bonus = _IV_BONUS_MAX * iv_richness_pct if iv_richness_pct is not None else 0.0
            objective_fit = min(1.0, primary + bonus)
        if is_premium:
            reasons.append(_iv_reason(excess_status[i][1], excess_status[i][0], iv_richness_pct))

        # Client-fit (nudge, band-scaled).
        cfit, creasons, cflags, over = _client_fit(cand, client_profile)
        reasons.extend(creasons)
        flags.extend(cflags)

        score = None if objective_fit is None else _W_OBJ * objective_fit + w_client * cfit

        ranked.append(RankedCandidate(
            candidate=cand, score=score, objective_fit=objective_fit, client_fit=cfit,
            iv_richness_pct=iv_richness_pct, over_extends=over,
            reasons=[r for r in reasons if r], flags=[f for f in flags if f],
        ))

    order = sorted(enumerate(ranked), key=lambda t: _sort_key((t[1], t[0])))
    n = len(order)
    out: list = []
    for rank, (_gen, rc) in enumerate(order, start=1):
        rc.rank = rank
        if rank == 1:
            rc.reasons.insert(0, "only candidate" if n == 1 else f"recommended — rank 1 of {n}")
        out.append(rc)
    return out
