"""Per-position assignment risk + entry-cost breakevens (pure, read-path).

Computes, per account in the load path:

* **P(assignment)** for every SHORT option leg — the European risk-neutral
  probability the leg finishes in the money at expiry (``N(d2)``), evaluated
  with the same lognormal distribution the payoff statistics use
  (:func:`pm.pricing.payoff_analytic.pop_lognormal_intervals` over the leg's
  ITM region — ``[(K, inf)]`` for a short call, ``[(0, K)]`` for a short put)
  at the leg's own IV. A leg with no usable IV reports ``None`` with a reason,
  never a fabricated number.
* the **structure-level union** over a structure's short legs: each side is
  anchored on its extreme strike (lowest short-call strike, highest short-put
  strike) and priced at that anchor leg's IV; the union is
  ``min(1.0, call_side + put_side)``. Same-side sibling legs are never
  summed — their ITM regions are nested, so the extreme-strike side IS that
  side's union. When the two regions overlap (a short-call strike below a
  short-put strike) they cover the whole half-line: one side finishes in the
  money with certainty, so the union is exactly 1.0 — flagged, and needing no
  vol input.
* an **early-exercise qualifier** per short American leg — ``div_call`` (in
  the money with a discrete ex-dividend date before expiry) or
  ``deep_itm_put`` (time value below the carry on the strike, the same
  economics as the deep-ITM short-put management fire). The flag qualifies
  the at-expiry probability; it never changes it. An American call on a name
  with no dividend before expiry never flags — without the dividend there is
  no early-exercise premium to capture.
* the **entry-cost breakeven(s)** per position — the at-expiry zero-crossings
  of the position alone against its opening-fill basis, assembled through the
  same leg builder the payoff panel uses (contract multiplier threaded to
  standard-contract equivalents, entry premium per real share) and read off
  the analytic kernel (:func:`pm.pricing.payoff_analytic.pwl_breakevens`).
  Inputs are extract-only, so breakevens fill with Bloomberg off. A position
  whose option cost basis is missing reports NO breakeven with a reason — an
  unknown entry price is not a zero entry price. An empty breakeven list is
  complete information (the curve never crosses zero) and travels with the
  ``always_profitable`` / ``always_loss`` flags.

Stored on ``AccountState.assignment`` as ``{"positions": {position_id:
record}, "structures": {structure_id: record}}``. Pure read of already-loaded
state — no Bloomberg call, no recompute of any upstream product. The UI reads
the records and never recomputes.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from pm.pricing import payoff_analytic
from pm.risk.payoff import _num, build_structure_payoff_legs
from pm.risk.pricing_adapter import EngineLeg, build_engine_legs

# Calendar-day carry basis for the deep-ITM short-put check, matching the
# management fire's convention (strike x rate x dte/365).
_CARRY_DAYS = 365.0

# Asset classes with a payoff against an entry basis.
_BREAKEVEN_CLASSES = ("option", "equity", "fund_etf")


# ---------------------------------------------------------------------------
# Per-leg P(assignment)
# ---------------------------------------------------------------------------

def leg_p_assign(eleg: EngineLeg) -> tuple[Optional[float], Optional[str]]:
    """Risk-neutral P(ITM at expiry) for one SHORT option leg, at the leg's
    own IV. Returns ``(p, None)`` or ``(None, reason)`` when the leg cannot be
    priced — a missing vol is reported, never substituted."""
    qty = _num(eleg.qty)
    if qty is None or qty >= 0:
        return None, "not a short leg"
    K = _num(eleg.K)
    if K is None or K <= 0:
        return None, "no strike"
    T = _num(eleg.T)
    if T is None or T <= 0:
        return None, "expired or zero tenor"
    spot = _num(eleg.spot)
    sigma = _num(eleg.sigma)
    if not eleg.priceable or sigma is None or sigma <= 0 or spot is None or spot <= 0:
        return None, f"no usable IV/spot (sigma: {eleg.sigma_source})"
    itm = [(K, math.inf)] if eleg.opt_type == "Call" else [(0.0, K)]
    p = payoff_analytic.pop_lognormal_intervals(
        spot, sigma, T, _num(eleg.r) or 0.0, _num(eleg.q) or 0.0, itm)
    if p != p:  # NaN guard; unreachable behind the checks above, kept honest
        return None, "degenerate pricing inputs"
    return float(p), None


# ---------------------------------------------------------------------------
# Early-exercise qualifier (flags the number; never changes it)
# ---------------------------------------------------------------------------

def assignment_flag(eleg: EngineLeg) -> tuple[Optional[str], Optional[str]]:
    """Early-exercise qualifier for one SHORT American leg: ``('div_call',
    reason)`` for an ITM short call with a discrete ex-dividend before expiry,
    ``('deep_itm_put', reason)`` when a short put's time value sits below the
    carry on its strike, else ``(None, None)``. European legs never flag —
    they cannot be exercised early."""
    qty = _num(eleg.qty)
    if qty is None or qty >= 0 or eleg.style != "American":
        return None, None
    spot, K = _num(eleg.spot), _num(eleg.K)
    if spot is None or K is None or K <= 0:
        return None, None

    if eleg.opt_type == "Call":
        divs = eleg.divs_df
        if spot > K and divs is not None and len(divs) > 0:
            try:
                nearest = divs.loc[divs["EX_DATE"].idxmin()]
                ex = pd.Timestamp(nearest["EX_DATE"]).date().isoformat()
                dps = float(nearest["DIVIDENDS"])
                reason = (f"ITM short call with ex-dividend {ex} "
                          f"(${dps:,.2f}/sh) before expiry — early assignment "
                          f"risk into the ex-date")
            except Exception:
                reason = ("ITM short call with an ex-dividend before expiry — "
                          "early assignment risk into the ex-date")
            return "div_call", reason
        return None, None

    # Short put: exercise-now economics when the remaining time value is worth
    # less than the interest on the strike over the option's remaining life.
    if spot >= K:
        return None, None
    mid = _num(eleg.mid)
    if mid is None:
        return None, None
    try:
        dte = int((pd.Timestamp(eleg.expiry)
                   - pd.Timestamp(eleg.today).normalize()).days)
    except Exception:
        return None, None
    if dte <= 0:
        return None, None
    extrinsic = max(0.0, mid - (K - spot))
    carry = K * (_num(eleg.r) or 0.0) * (dte / _CARRY_DAYS)
    if extrinsic < carry:
        return ("deep_itm_put",
                f"deep-ITM short put — time value ${extrinsic:,.2f}/sh is "
                f"below the carry on the strike (~${carry:,.2f}/sh over "
                f"{dte}d); early-assignment economics")
    return None, None


# ---------------------------------------------------------------------------
# Structure-level union
# ---------------------------------------------------------------------------

def structure_assignment(structure, elegs: dict) -> Optional[dict]:
    """The assignment union over a structure's SHORT option legs. Returns
    ``None`` when the structure has no short option leg (no record, mirroring
    the Tier-2 pass). The union prices each side at its extreme-strike anchor
    leg only; a present side whose anchor cannot be priced makes the union
    ``None`` with a coverage reason — a dash, never a fabricated 0."""
    shorts: list[EngineLeg] = []
    for leg in getattr(structure, "legs", None) or []:
        alloc = _num(getattr(leg, "allocated_qty", None))
        if alloc is None or alloc >= 0:
            continue
        eleg = elegs.get(getattr(leg, "position_id", None))
        if eleg is None:
            continue  # not an option leg (e.g. short stock) or unresolved
        shorts.append(eleg)
    if not shorts:
        return None

    per_leg = {e.position_id: leg_p_assign(e) for e in shorts}
    n_unpriced = sum(1 for p, _ in per_leg.values() if p is None)

    def _side(legs_, pick):
        if not legs_:
            return None
        anchor = pick(legs_, key=lambda e: e.K)
        p, why = per_leg[anchor.position_id]
        return {"K": float(anchor.K), "position_id": anchor.position_id,
                "p": p, "reason": why}

    call_side = _side([e for e in shorts if e.opt_type == "Call"], min)
    put_side = _side([e for e in shorts if e.opt_type == "Put"], max)

    overlap = bool(call_side and put_side and put_side["K"] > call_side["K"])
    reason = None
    if overlap:
        # The two ITM regions cover the whole half-line: whatever the vol, one
        # side finishes in the money.
        union: Optional[float] = 1.0
        reason = ("short-call strike below short-put strike — the ITM regions "
                  "overlap, so one side finishes in the money")
    else:
        sides = [s for s in (call_side, put_side) if s is not None]
        if any(s["p"] is None for s in sides):
            union = None
            reason = (f"{n_unpriced} of {len(shorts)} short legs unpriced — "
                      f"union unavailable")
        else:
            union = min(1.0, sum(s["p"] for s in sides))

    expiries = {e.expiry for e in shorts if e.expiry is not None}
    return {
        "p_assign": union,
        "call_side": call_side,
        "put_side": put_side,
        "n_short_legs": len(shorts),
        "n_unpriced": n_unpriced,
        "overlap": overlap,
        "multi_expiry": len(expiries) > 1,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Per-position record (P(assignment) + entry-cost breakevens)
# ---------------------------------------------------------------------------

def _no_breakeven(reason: str) -> dict:
    return {"breakevens": None, "always_profitable": False,
            "always_loss": False, "basis_source": None,
            "breakeven_reason": reason}


def _position_record(state, account_state, position, elegs: dict,
                     today=None) -> Optional[dict]:
    """One position's assignment/breakeven record, or ``None`` when the
    position carries neither (cash / other)."""
    ac = getattr(position, "asset_class", None)
    qty = _num(getattr(position, "quantity", None))
    rec: dict = {}
    warnings: list = []

    # ---- P(assignment) + the early-exercise qualifier (short options only) --
    if ac == "option" and qty is not None and qty < 0:
        eleg = elegs.get(position.position_id)
        if eleg is not None:
            p, why = leg_p_assign(eleg)
            flag, flag_reason = assignment_flag(eleg)
            rec.update({"p_assign": p, "p_assign_reason": why,
                        "flag": flag, "flag_reason": flag_reason,
                        "sigma_source": eleg.sigma_source})
            if (eleg.opt_type == "Put" and eleg.style == "American"
                    and _num(eleg.mid) is None
                    and _num(eleg.spot) is not None
                    and _num(eleg.K) is not None
                    and _num(eleg.spot) < _num(eleg.K)):
                warnings.append("early-exercise check skipped — no market mid "
                                "for the leg")
        else:
            rec.update({"p_assign": None,
                        "p_assign_reason": "option leg not resolved to "
                                           "pricing inputs",
                        "flag": None, "flag_reason": None,
                        "sigma_source": "missing"})

    # ---- entry-cost breakevens (options + stock-like positions) ------------
    if ac in _BREAKEVEN_CLASSES:
        cb = _num(getattr(position, "cost_basis", None))
        mv = _num(getattr(position, "market_value", None))
        if not qty:
            rec.update(_no_breakeven("quantity unavailable — entry breakeven "
                                     "not computed"))
        elif ac == "option" and cb is None:
            # The leg assembler would fall back to a silent zero premium here,
            # which reads as a breakeven at the strike. An unknown entry price
            # is not a zero entry price — report nothing, with the reason.
            rec.update(_no_breakeven("cost basis unavailable — entry "
                                     "breakeven not computed"))
        elif ac != "option" and cb is None and mv is None:
            rec.update(_no_breakeven("no cost basis or mark — entry breakeven "
                                     "not computed"))
        else:
            asm = build_structure_payoff_legs(state, account_state, position,
                                              today=today, elegs=elegs)
            legs = asm.get("leg_dicts") or []
            if legs:
                pwl = payoff_analytic.pwl_from_legs(legs)
                mpl = payoff_analytic.pwl_max_profit_loss(pwl)
                rec.update({
                    "breakevens": [float(b) for b in
                                   payoff_analytic.pwl_breakevens(pwl)],
                    "always_profitable": bool(mpl["always_profitable"]),
                    "always_loss": bool(mpl["always_loss"]),
                    "basis_source": "cost_basis" if cb is not None else "mark",
                    "breakeven_reason": None,
                })
                warnings.extend(asm.get("warnings") or [])

    if not rec:
        return None
    rec["warnings"] = warnings
    return rec


# ---------------------------------------------------------------------------
# Load-path pass
# ---------------------------------------------------------------------------

def run_account_assignment(state, today=None) -> None:
    """Load-path pass: per account, resolve the option engine legs ONCE, then
    store the per-position assignment/breakeven records and the per-structure
    short-leg unions on ``account_state.assignment``. Pure / read-only; runs
    after the structure Tier-2 pass. A position or structure whose record
    cannot be built degrades to no entry, not a crash. ``today`` (default:
    wall clock, as the live load intends) threads through the engine legs and
    the leg assembly so the pass is deterministic under test."""
    for acc in state.accounts.values():
        positions_map: dict = {}
        structures_map: dict = {}
        try:
            elegs = {e.position_id: e
                     for e in build_engine_legs(state, acc, today=today)}
        except Exception:
            elegs = {}
        for p in getattr(acc, "positions", None) or []:
            try:
                rec = _position_record(state, acc, p, elegs, today=today)
            except Exception:
                rec = None
            if rec:
                positions_map[p.position_id] = rec
        for s in getattr(acc, "structures", None) or []:
            if getattr(s, "status", None) == "rejected":
                continue
            try:
                srec = structure_assignment(s, elegs)
            except Exception:
                srec = None
            if srec:
                structures_map[s.structure_id] = srec
        acc.assignment = {"positions": positions_map,
                          "structures": structures_map}
