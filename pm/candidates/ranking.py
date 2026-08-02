"""Objective + client-fit ranking for the priced scanner candidates.

Ranks the roll / overlay candidates the generation layer already priced, so the
best fit for the desk surfaces first WITH a transparent reason — the list is
ordered, never reduced to a single recommendation. Two ingredients combine:

* **Objective-fit** — for the active objective (harvest, extend-duration,
  defend-cut-delta, add-hedge, …) one driver metric, converted to a
  within-set percentile so metrics on different scales (dollars, days, delta)
  become comparable. Harvest's driver is the execution-adjusted credit per day
  per standard contract, graded by the soft DTE/Δ band kernels (see
  ``pm.candidates.objectives``), and additionally earns a small IV-richness
  bonus GATED on the short leg being inside the fitted smile (selling a rich —
  and real — quote is the point of the trade). IV-rank is name-level context,
  shown upstream, never a per-candidate differentiator.

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
    DEFEND,
    DEFEND_CUT_DELTA,
    EXTEND_DURATION,
    HARVEST,
)
from pm.candidates.objectives import (
    DEFAULT_HARVEST_PARAMS,
    OI_THIN,
    SPREAD_FLAG_PCT,
    band_kernel,
    defend_tie,
)

# Combination weights + the coverage-confidence damping on the client nudge.
_W_OBJ = 0.7
_W_CLIENT = 0.3
_BAND_MULT = {"low": 0.0, "medium": 0.5, "high": 1.0}

# IV-richness bonus: proportional to the short leg's within-set richness percentile,
# capped so it only re-orders near-ties (never inverts an obviously-better driver).
# Harvest-only, and GATED on the short leg being inside the fitted smile — a
# contract the surface fit excluded (wide/stale market) earns no bonus.
_IV_BONUS_MAX = 0.15
_IV_BONUS_OBJECTIVES = (HARVEST,)

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

# PER-CONTRACT dollar scale of the costless solve's cap tie-break. The priced
# max_profit is normalized by the opened contracts BEFORE the map — a fixed
# position-total scale saturated on large lines (a 3,190-lot's caps all
# compressed to ~0.899, the tie-break's designed resolution gone), and the
# per-contract cap is size-invariant, so the same book ranks identically at
# any lot count. The per-contract value maps through 0.5 + 0.4·x/(|x|+scale)
# into (0.1, 0.9) strictly; unbounded gain sits at 0.95, missing economics at
# 0.0. Total spread ≤ 0.95 — STRICTLY below the 1-day tenor step — so
# nearest-term ordering is exactly lexicographic even when caps straddle sign
# (an underwater covered call's caps are negative), and within a day:
# unpriced < every real cap < unbounded.
_CAP_SCALE = 1_000.0


def _cap_term(cand) -> float:
    e = getattr(cand, "economics", None)
    if not e:
        return 0.0
    if e.get("unbounded_gain"):
        return 0.95
    cap = _num(e.get("max_profit"))
    if cap is None:
        return 0.0
    n = sum(abs(_num(lg.get("qty")) or 0.0) for lg in _opt_legs(cand)) or 1.0
    per_ct = cap / n
    return 0.5 + 0.4 * per_ct / (abs(per_ct) + _CAP_SCALE)


def _opt_legs(cand) -> list:
    """The candidate's option legs, narrowed to the legs the transaction OPENS when
    the marker is present — a structure-anchored roll carries the enclosing
    structure's kept legs too, and those must never key a driver, the IV+pp lookup
    or the posture read. Candidates without the marker keep every option leg."""
    opts = [lg for lg in (getattr(cand, "legs", None) or [])
            if lg.get("opt_type") in ("Call", "Put")]
    opened = [lg for lg in opts if lg.get("opened")]
    return opened or opts


def _new_strike(cand) -> Optional[float]:
    for lg in _opt_legs(cand):
        if lg.get("K") is not None:
            return _num(lg.get("K"))
    return None


def _cand_dte(cand) -> Optional[float]:
    """The candidate's OWN tenor — the OPENED leg's days-to-expiry. The economics
    ``dte`` is the resulting position's nearest expiry, which on a structure-anchored
    roll is often a KEPT sibling's (a collar's put outlives no roll); every tenor
    read keys here so drivers, client fit and reasons describe the rolled leg."""
    nd = _num(getattr(cand, "new_leg_dte", None))
    if nd is not None:
        return nd
    return _num((getattr(cand, "economics", None) or {}).get("dte"))


def _away(cand, held) -> float:
    """The away-from-the-money direction for the rolled leg: calls +1 (higher
    strikes), puts −1 (lower) — for a short leg the assignment-risk-reducing
    move, for a long leg the premium-at-risk-reducing one. Keyed on the OPENED
    leg's right (same as the held leg's on a roll), so overlays need no held."""
    opts = _opt_legs(cand)
    right = (opts[0].get("opt_type") if opts else None) or (held or {}).get("right")
    return -1.0 if str(right).upper().startswith("P") else 1.0


def _defend_rel(cand, held) -> Optional[float]:
    """Defend's recapture: the directional strike distance as a fraction of the
    held strike — assignment-risk relief bought back, never a priced cap.
    None when the strikes are unknowable (the candidate sorts last)."""
    nk = _new_strike(cand)
    hk = _num((held or {}).get("strike"))
    if nk is None or hk is None or not hk:
        return None
    return (nk - hk) * _away(cand, held) / abs(hk)


def _driver(cand, objective, held) -> Optional[float]:
    """The single objective driver, oriented so higher is always better."""
    # A joint roll carries its objective's own selection metric (joint net cash,
    # total delta cut, total directional move) — the single-leg drivers below
    # don't generalise to a multi-leg roll. Joint costless deliberately leaves it
    # unset so the lexicographic dte+cap driver applies unchanged.
    jd = _num(getattr(cand, "joint_driver", None))
    if jd is not None:
        return jd
    if objective == ADD_HEDGE:
        # add-hedge = cheaper/financed protection.
        return _num(getattr(cand, "net_credit", None))
    if objective == DEFEND:
        # Strictly lexicographic (the costless mechanism, a recapture tie
        # term): NEAREST expiry first; within a day, maximum directional
        # strike recapture. Credit is admission-only, never the driver.
        dte = _cand_dte(cand)
        rel = _defend_rel(cand, held)
        if dte is None or rel is None:
            return None
        return -dte + defend_tie(rel)
    if objective == COSTLESS:
        # The costless solve: NEAREST expiry first; the priced upside cap
        # (economics.max_profit — never a raw strike) breaks ties within a day.
        # Strictly lexicographic — see _cap_term's bounds.
        dte = _cand_dte(cand)
        if dte is None:
            return None
        return -dte + _cap_term(cand)
    if objective == EXTEND_DURATION:
        return _cand_dte(cand)                       # more tenor on the rolled leg
    if objective == DEFEND_CUT_DELTA:
        nd = _num(getattr(cand, "new_leg_delta", None))
        if nd is None:
            return None
        hd = _num((held or {}).get("delta"))
        if hd is None:
            return -abs(nd)                          # no held Δ: lower |Δ| is more defensive
        return abs(hd) - abs(nd)                     # delta reduction
    return _num(getattr(cand, "net_credit", None))


# ---------------------------------------------------------------------------
# Harvest — execution-adjusted credit per day per contract, band-kerneled
# ---------------------------------------------------------------------------

_HARVEST_MULT = 100.0    # per-share half-spread -> per-contract dollars


def _harvest_parts(cand, liquidity, params) -> Optional[dict]:
    """The harvest driver's decomposition for one candidate: per-contract
    credit after the crossing haircut, per day of the OPENED leg's tenor,
    graded by the DTE and |Δ| band kernels — plus the liquidity readings the
    flags and reasons quote. None when the driver's inputs are missing (the
    candidate sorts last with the standard unavailable-driver flag). Joint
    candidates never reach here (their ``joint_driver`` short-circuits)."""
    p = params or DEFAULT_HARVEST_PARAMS
    nc = _num(getattr(cand, "net_credit", None))
    dte = _cand_dte(cand)
    opts = _opt_legs(cand)
    n = abs(_num(opts[0].get("qty")) or 0.0) if opts else 0.0
    if nc is None or dte is None or not n:
        return None
    lq = (liquidity or {}).get(opts[0].get("position_id")) or {}
    bid, ask = _num(lq.get("bid")), _num(lq.get("ask"))
    spread = (ask - bid) if (bid is not None and ask is not None and ask >= bid) else None
    haircut = (p.crossing_k * spread / 2.0 * _HARVEST_MULT) if spread is not None else 0.0
    per_ct = nc / n - haircut
    per_day = per_ct / max(dte, 1.0)
    nd = _num(getattr(cand, "new_leg_delta", None))
    kd = band_kernel(dte, p.dte_lo, p.dte_hi, p.dte_hard_lo, p.dte_hard_hi)
    kdelta = band_kernel(abs(nd) if nd is not None else None,
                         p.delta_lo, p.delta_hi, p.delta_hard_lo, p.delta_hard_hi)
    mid = _num(lq.get("mid"))
    if mid is None and spread is not None:
        mid = (bid + ask) / 2.0
    return {"driver": per_day * kd * kdelta, "per_day": per_day,
            "kd": kd, "kdelta": kdelta, "spread_known": spread is not None,
            "spread_pct": (spread / mid) if (spread is not None and mid and mid > 0)
            else None,
            "oi": _num(lq.get("oi"))}


def _liquidity_flags(cand, liquidity) -> list:
    """Wide-market / thin-strike demotion flags for the candidate's OPENED
    legs — execution context every objective's rows carry (a contract is wide
    or thin regardless of why you're rolling to it). Display-only: flags never
    change a driver. A multi-leg candidate tags each offending leg by its
    contract so the reader knows which side is thin."""
    flags: list = []
    opts = _opt_legs(cand)
    multi = len(opts) > 1
    for lg in opts:
        lq = (liquidity or {}).get(lg.get("position_id")) or {}
        bid, ask = _num(lq.get("bid")), _num(lq.get("ask"))
        spread = (ask - bid) if (bid is not None and ask is not None and ask >= bid) else None
        mid = _num(lq.get("mid"))
        if mid is None and spread is not None:
            mid = (bid + ask) / 2.0
        k = lg.get("K")
        tag = (f" ({lg.get('opt_type', '?')[0]}{k:g})"
               if (multi and k is not None) else "")
        if spread is not None and mid and mid > 0 and spread / mid > SPREAD_FLAG_PCT:
            flags.append(f"wide market — {spread / mid:.0%} spread{tag}")
        oi = _num(lq.get("oi"))
        if oi is not None and oi < OI_THIN:
            flags.append(f"thin strike — OI {int(oi)}{tag}")
    return flags


def _harvest_reason(cand, parts, pct) -> Optional[str]:
    nc_txt = _money(_num(getattr(cand, "net_credit", None)))
    if parts is None:
        if _num(getattr(cand, "joint_driver", None)) is not None:
            return f"{nc_txt} joint net credit ({_pctl(pct)})"
        return None
    txt = (f"{nc_txt} net credit — ${parts['per_day']:,.2f}/day·contract"
           + (" after crossing" if parts["spread_known"] else "")
           + f" ({_pctl(pct)})")
    if parts["kd"] != 1.0 or parts["kdelta"] != 1.0:
        txt += f" · band ×{parts['kd']:.2f}, Δ ×{parts['kdelta']:.2f}"
    return txt


def _objective_reason(cand, objective, driver, pct, held) -> Optional[str]:
    if driver is None:
        return None
    if objective == ADD_HEDGE:
        return f"{_money(driver)} to establish ({_pctl(pct)})"
    if objective == EXTEND_DURATION:
        dte = int(round(driver))
        held_dte = (held or {}).get("dte")
        added = f", +{dte - int(held_dte)}d added" if held_dte is not None else ""
        return f"{dte}d to expiry{added} ({_pctl(pct)})"
    if objective == DEFEND:
        dte = _cand_dte(cand)
        nk = _new_strike(cand)
        hk = _num((held or {}).get("strike"))
        nc = _num(getattr(cand, "net_credit", None))
        dte_txt = f"{int(round(dte))}d nearest" if dte is not None else "—"
        rec = (f", {nk - hk:+g} strikes recaptured"
               if (nk is not None and hk is not None) else "")
        net = f" · net {_money(nc)}" if nc is not None else ""
        return f"defends — {dte_txt}{rec} ({_pctl(pct)}){net}"
    if objective == COSTLESS:
        dte = _cand_dte(cand)
        e = getattr(cand, "economics", None) or {}
        cap_txt = ("∞" if e.get("unbounded_gain")
                   else _money(_num(e.get("max_profit"))))
        dte_txt = f"{int(round(dte))}d" if dte is not None else "—"
        return f"costless — {dte_txt} to expiry, cap {cap_txt} ({_pctl(pct)})"
    if objective == DEFEND_CUT_DELTA:
        if _num(getattr(cand, "joint_driver", None)) is not None:
            # A joint roll's metric is the total cut across every rolled leg.
            return f"cuts total |Δ|·contracts by {driver:.2f} ({_pctl(pct)})"
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
    """(iv_excess, status) for the candidate's short option leg — the leg being SOLD
    by this transaction (kept sibling shorts of an enclosing structure never key it).
    status: 'ok' (found, inside the fitted smile), 'not_in_fit' (measured but the
    surface fit EXCLUDED the contract — wide/stale market or a degraded fit; no
    bonus, the gate that closes the stale-quote magnet), 'no_short' (no
    premium-selling leg), 'not_in_slice' (the short leg's contract fell outside
    the pulled slice, so no IV+pp)."""
    shorts = [lg for lg in _opt_legs(cand) if (lg.get("qty") or 0) < 0]
    if not shorts:
        return None, "no_short"
    entry = excess_by_ticker.get(shorts[0].get("position_id"))
    if entry is None:
        return None, "not_in_slice"
    exc, in_fit = entry
    if exc is None:
        return None, "not_in_slice"
    if not in_fit:
        return None, "not_in_fit"
    return exc, "ok"


def _iv_reason(status, excess, pct) -> Optional[str]:
    if status == "ok":
        return f"short leg {excess:+.1f}pp rich ({_pctl(pct)}, in fit)"
    if status == "not_in_fit":
        return "no IV+pp bonus — outside the fitted smile"
    if status == "no_short":
        return "no premium leg — no IV+pp bonus"
    return "IV+pp n/a (short leg outside slice)"


# ---------------------------------------------------------------------------
# Client-fit — tenor + posture, guarded on the fragile profile
# ---------------------------------------------------------------------------

def _candidate_posture(cand) -> Optional[str]:
    """The candidate's posture from the single option leg the transaction OPENS
    (short_call / long_call / short_put / long_put) — a structure-anchored roll
    reads the rolled leg, not its kept siblings. A collar / no-option candidate
    has no single posture -> None (posture dimension skipped)."""
    opt = _opt_legs(cand)
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

    # Per-dimension confidence gate: a dimension whose OWN read is
    # low-confidence (thin sample) must not move ranking even when the global
    # band is medium+ — it is skipped with a reason, never half-trusted.
    def _dim_ok(dim, label):
        if getattr(dim, "confidence", None) == "low":
            reasons.append(f"{label} read low-confidence — not scored")
            return False
        return True

    tenor_pref = getattr(profile, "tenor_pref", None)
    cand_dte = _cand_dte(cand)
    tfit, treason, over_msg = _tenor_fit(cand_dte, tenor_pref)
    if tfit is not None and _dim_ok(tenor_pref, "tenor"):
        dims.append(tfit)
        if treason:
            reasons.append(treason)
    over = over_msg is not None
    if over_msg:
        flags.append(over_msg)

    strategy_bias = getattr(profile, "strategy_bias", None)
    pfit, preason = _posture_fit(_candidate_posture(cand), strategy_bias)
    if pfit is not None and _dim_ok(strategy_bias, "posture"):
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
                    held=None, liquidity=None, params=None) -> list:
    """Rank the priced candidates for one objective. Returns ``[RankedCandidate, ...]``
    ordered best-first (rank 1 = recommended), each carrying its score decomposition
    and a readable reason.

    ``candidates`` may contain other objectives — only those tagged ``objective`` are
    ranked. ``iv_pp`` is the slice's IV+pp rows (``[{ticker, iv_excess, in_fit, ...}]``);
    ``held`` is ``{delta, dte, strike, right}`` for the held leg (rolls) or None
    (stock overlays); ``liquidity`` is the per-ticker ``{bid, ask, mid, oi}`` map and
    ``params`` the ``HarvestParams`` — both consumed by HARVEST only and both
    defaulting to None, so every other objective's path is byte-identical without
    them. Pure — no Bloomberg, no state writes."""
    cands = [c for c in (candidates or []) if getattr(c, "objective", None) == objective]
    if not cands:
        return []

    excess_by_ticker = {}
    for row in (iv_pp or []):
        tk = row.get("ticker")
        if tk is not None:
            excess_by_ticker[tk] = (_num(row.get("iv_excess")), bool(row.get("in_fit")))

    is_harvest = objective in _IV_BONUS_OBJECTIVES
    if is_harvest:
        hparts = [None if _num(getattr(c, "joint_driver", None)) is not None
                  else _harvest_parts(c, liquidity, params) for c in cands]
        drivers = [
            _num(getattr(c, "joint_driver", None)) if
            _num(getattr(c, "joint_driver", None)) is not None
            else (p["driver"] if p is not None else None)
            for c, p in zip(cands, hparts)]
    else:
        hparts = [None] * len(cands)
        drivers = [_driver(c, objective, held) for c in cands]
    driver_pct = _avg_rank_percentile(drivers)

    excess_status = [_short_leg_excess(c, excess_by_ticker) for c in cands]
    excesses = [e if is_harvest else None for (e, _s) in excess_status]
    iv_pct = _avg_rank_percentile(excesses)

    band = getattr(getattr(client_profile, "coverage", None), "band", "low")
    w_client = _W_CLIENT * _BAND_MULT.get(band, 0.0)

    ranked: list = []
    for i, cand in enumerate(cands):
        reasons: list = []
        flags: list = list(getattr(cand, "warnings", None) or [])
        if getattr(cand, "economics", None) is None:
            flags.append("economics unavailable (pricing degraded)")

        # Objective-fit (+ the gated IV bonus for harvest).
        primary = driver_pct[i]
        oreason = (_harvest_reason(cand, hparts[i], primary) if is_harvest
                   else _objective_reason(cand, objective, drivers[i], primary, held))
        if oreason:
            reasons.append(oreason)

        iv_richness_pct = iv_pct[i] if is_harvest else None
        if primary is None:
            objective_fit = None
            flags.append("objective driver unavailable — sorted last")
        else:
            bonus = _IV_BONUS_MAX * iv_richness_pct if iv_richness_pct is not None else 0.0
            objective_fit = min(1.0, primary + bonus)
        if is_harvest:
            reasons.append(_iv_reason(excess_status[i][1], excess_status[i][0], iv_richness_pct))
            hp = hparts[i]
            if hp is not None and not hp["spread_known"]:
                # An unknown spread is said out loud (no haircut was applied).
                reasons.append("spread unknown — no execution adjustment")

        # Liquidity: demote by FLAG on every objective, never a silent drop.
        flags.extend(_liquidity_flags(cand, liquidity))

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
