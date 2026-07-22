"""Candidate generation + per-candidate economics for the options scanner.

For a held position, generate the certain-core adjustment candidates — rolls of a
held option and single-leg overlays on held stock — drawing each candidate's strikes
and expiries from the cached slice, assembling it as a leg set, and pricing it through
the validated payoff engine in ONE ``compute_payoff`` call. A held option that is a
leg of a detected structure rolls WITHIN it: the caller threads the structure's kept
legs through (entry-basis dicts from the payoff assembly) and each candidate prices
as the resulting structure, not as an isolated leg. No new pricing math: the
economics (max P/L, capital-at-risk, PoP, breakevens, net greeks) come straight from
the engine; the only added arithmetic is the transaction's net credit/debit, computed
from contemporaneous mids.

Leg fields follow the payoff engine's contract exactly: option legs use the capitalized
``opt_type`` ('Call'/'Put'), decimal sigma/r/q, a signed integer ``qty`` (long +, short
-), and a positive per-share entry ``mid`` (sign carried by qty); stock legs use a
per-share ``cost_basis`` in place of ``mid``. Conventions mirror the pricing adapter:
sigma = iv/100, T = year_frac(today, expiry) busday/252, r via pick_rate_for_dte, q =
the name's continuous dividend yield.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

_MULT = 100
_CAP_DEFAULT = 15
# |net debit/credit| within this per-share $ reads "costless" (× 100 × contracts).
_COSTLESS_PER_SHARE = 0.05

ROLL_FOR_CREDIT = "roll-for-credit"
EXTEND_DURATION = "extend-duration"
DEFEND_CUT_DELTA = "defend-cut-delta"
MAX_PREMIUM = "max-premium"
ADD_HEDGE = "add-hedge"
ROLL_UP_OUT = "roll-up-out"
COSTLESS = "costless"
_DEFAULT_ROLL_OBJECTIVES = (ROLL_FOR_CREDIT, EXTEND_DURATION, DEFEND_CUT_DELTA,
                            ROLL_UP_OUT, COSTLESS, MAX_PREMIUM)


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _qty(v):
    """Leg quantity for the engine: integral values stay int; a fractional
    standard-contract equivalent (non-100 multiplier) is preserved exactly,
    mirroring the payoff assembly's qty scaling."""
    f = float(v)
    return int(f) if f.is_integer() else f


@dataclass
class SliceContract:
    strike: float
    expiry: date
    right: str                    # 'CALL' | 'PUT'
    iv: Optional[float] = None    # percent
    mid: Optional[float] = None   # per-share
    delta: Optional[float] = None
    ticker: Optional[str] = None


@dataclass
class Candidate:
    objective: str
    kind: str
    description: str
    legs: list
    net_credit: Optional[float]           # $, positive = credit received
    economics: Optional[dict] = None
    greeks: Optional[dict] = None
    breakevens: Optional[list] = None
    warnings: list = field(default_factory=list)
    new_leg_delta: Optional[float] = None  # the new option leg's own per-contract delta
    # The OPENED leg's own days-to-expiry. The economics dte is the RESULTING
    # position's nearest expiry — on a structure-anchored roll that is often a KEPT
    # sibling's — so every tenor read (drivers, client fit, reasons) keys on this.
    new_leg_dte: Optional[int] = None


# ---------------------------------------------------------------------------
# Leg + tier1 assembly (to the payoff engine's exact contract)
# ---------------------------------------------------------------------------

def _rate(curve, dte, r_scalar) -> float:
    if curve:
        from pm.core.bloomberg_client import pick_rate_for_dte
        pick = pick_rate_for_dte(curve, dte)
        if pick and pick.get("rate") is not None:
            return float(pick["rate"])
    return float(r_scalar)


def _sigma(iv, mid, spot, strike, T, r, q, right) -> Optional[float]:
    """Decimal sigma from the slice IV (percent), else solved from the mid, else None."""
    d = _num(iv)
    if d is not None and d > 0:
        return d / 100.0
    m = _num(mid)
    if m and m > 0 and T and T > 0:
        from pm.pricing.implied_vol import implied_vol
        return implied_vol(m, spot, strike, T, r, q, "Call" if right == "CALL" else "Put",
                           model="American")   # decimal or None, never NaN
    return None


def _option_leg(sc: SliceContract, qty, spot, *, curve, r_scalar, q, today, role) -> dict:
    from pm.pricing.conventions import year_frac
    T = float(year_frac(today, sc.expiry))
    dte = max((sc.expiry - today).days, 0)
    r = _rate(curve, dte, r_scalar)
    sigma = _sigma(sc.iv, sc.mid, spot, sc.strike, T, r, q, sc.right)
    mid = _num(sc.mid)
    return {
        "opt_type": "Call" if sc.right == "CALL" else "Put",
        "K": float(sc.strike), "expiry": sc.expiry, "T": T, "sigma": sigma,
        "style": "American", "qty": _qty(qty), "mid": mid, "r": float(r), "q": float(q),
        "priceable": bool(sigma is not None and T > 0), "position_id": sc.ticker,
        "role": role, "delta": _num(sc.delta),
        # Every option leg this module builds is OPENED by the transaction; held
        # sibling legs (an enclosing structure's kept legs) enter from the payoff
        # assembly without the marker, so consumers can tell the two apart.
        "opened": True,
    }


def _stock_leg(qty, cost_basis_per_share, position_id="held_stock", role="long_stock") -> dict:
    return {"opt_type": "Stock", "K": None, "expiry": None, "T": None, "sigma": None,
            "style": None, "qty": int(qty), "mid": None,
            "cost_basis": float(cost_basis_per_share), "r": None, "q": None,
            "priceable": True, "position_id": position_id, "role": role}


def _build_tier1(legs, today) -> dict:
    ndc = 0.0
    opt_premium = 0.0
    strikes, expiries = [], []
    for lg in legs:
        if lg.get("opt_type") == "Stock":
            ndc += lg["qty"] * float(lg["cost_basis"])
        else:
            m = _num(lg.get("mid"))
            if m is not None:
                ndc += lg["qty"] * m * _MULT
                opt_premium += lg["qty"] * m * _MULT
            if lg.get("K") is not None:
                strikes.append(float(lg["K"]))
            if lg.get("expiry") is not None:
                expiries.append(lg["expiry"])
    return {"net_debit_credit": ndc, "net_premium": opt_premium, "net_pnl": None,
            "strikes": sorted(set(strikes)), "expiries": sorted(set(expiries))}


def _price(legs, spot, today) -> dict:
    from pm.risk.payoff import compute_payoff
    try:
        return compute_payoff(legs, float(spot), _build_tier1(legs, today), today=today)
    except Exception as exc:
        logger.warning("compute_payoff failed for a candidate: %s", exc)
        return {}


def _finish(objective, kind, description, legs, net_credit, spot, today,
            new_leg=None, extra_warnings=None) -> Candidate:
    res = _price(legs, spot, today)
    # The rolled/new option leg's own delta (assignment proxy) and tenor — carried
    # through for the ranking drivers, client tenor fit and the "new Δ" decision
    # column. Keyed on the leg the transaction OPENS (kept sibling legs from an
    # enclosing structure may ride along in ``legs``); a two-option overlay (collar)
    # passes none, having no single new leg to name.
    new_leg_delta = _num(new_leg.get("delta")) if new_leg is not None else None
    new_leg_dte = ((new_leg["expiry"] - today).days
                   if new_leg is not None and new_leg.get("expiry") is not None else None)
    return Candidate(objective=objective, kind=kind, description=description, legs=legs,
                     net_credit=net_credit, economics=res.get("economics"),
                     greeks=res.get("greeks_now"), breakevens=res.get("breakevens"),
                     warnings=list(extra_warnings or []) + list(res.get("warnings") or []),
                     new_leg_delta=new_leg_delta, new_leg_dte=new_leg_dte)


# ---------------------------------------------------------------------------
# Slice parsing + shared helpers
# ---------------------------------------------------------------------------

def _parse_slice(slice_df) -> list:
    from pm.core.ticker_utils import parse_option_description
    out = []
    if slice_df is None or getattr(slice_df, "empty", True):
        return out
    for tk, row in slice_df.iterrows():
        p = parse_option_description(str(tk))
        if not p:
            continue
        out.append(SliceContract(strike=p["strike"], expiry=p["expiry"], right=p["right"],
                                 iv=_num(row.get("iv_mid")), mid=_num(row.get("PX_MID")),
                                 delta=_num(row.get("delta_mid")), ticker=str(tk)))
    return out


def _roll_kind(held: SliceContract, new: SliceContract) -> str:
    if abs(new.strike - held.strike) < 1e-6:
        return "roll_out"
    return "roll_up_out" if new.strike > held.strike else "roll_down_out"


def _role_for(qty, right) -> str:
    side = "short" if qty < 0 else "long"
    return f"{side}_{'call' if right == 'CALL' else 'put'}"


# ---------------------------------------------------------------------------
# Rolls (held is an option) — the depth-first core
# ---------------------------------------------------------------------------

def _select_roll(objective, held, held_qty, held_mid, held_delta, later, cap) -> list:
    """(SliceContract, kind) picks for one objective, over the later-expiry same-right
    contracts, capped."""
    hm = _num(held_mid)
    if objective == ROLL_FOR_CREDIT:
        scored = []
        for c in later:
            nm = _num(c.mid)
            if nm is None or hm is None:
                continue
            nc = held_qty * (hm - nm) * _MULT
            if nc > 0:
                scored.append((nc, c))
        scored.sort(key=lambda x: -x[0])
        return [(c, _roll_kind(held, c)) for _, c in scored[:cap]]

    if objective == EXTEND_DURATION:
        strikes = {c.strike for c in later}
        near = min(strikes, key=lambda k: abs(k - held.strike)) if strikes else None
        picks = [c for c in later if near is not None and abs(c.strike - near) < 1e-6]
        picks.sort(key=lambda c: c.expiry, reverse=True)
        return [(c, "roll_out") for c in picks[:cap]]

    if objective == DEFEND_CUT_DELTA:
        if held_delta is None:
            return []
        scored = [(abs(held_delta) - abs(c.delta), c) for c in later
                  if c.delta is not None and abs(c.delta) < abs(held_delta)]
        scored.sort(key=lambda x: -x[0])
        return [(c, _roll_kind(held, c)) for _, c in scored[:cap]]

    if objective == MAX_PREMIUM:
        # Roll to collect premium — the credit rolls; the ranker orders by premium per
        # dollar of cap (a distinct lens from raw credit).
        scored = []
        for c in later:
            nm = _num(c.mid)
            if nm is None or hm is None:
                continue
            nc = held_qty * (hm - nm) * _MULT
            if nc > 0:
                scored.append((nc, c))
        scored.sort(key=lambda x: -x[0])
        return [(c, _roll_kind(held, c)) for _, c in scored[:cap]]

    if objective == ROLL_UP_OUT:
        # Raise the strike AND extend — the ITM short-call workflow. The cap keeps the
        # cheapest (near-costless) up-and-out rolls; the ranker orders by strike relief.
        picks = []
        for c in later:
            if c.strike <= held.strike:
                continue
            nm = _num(c.mid)
            cost = (abs(held_qty * (hm - nm) * _MULT)
                    if (hm is not None and nm is not None) else float("inf"))
            picks.append((cost, c))
        picks.sort(key=lambda x: x[0])
        return [(c, _roll_kind(held, c)) for _, c in picks[:cap]]

    if objective == COSTLESS:
        band = _COSTLESS_PER_SHARE * _MULT * max(abs(held_qty), 1)
        picks = []
        for c in later:
            nm = _num(c.mid)
            if nm is None or hm is None:
                continue
            if abs(held_qty * (hm - nm) * _MULT) <= band:
                relief = (c.strike - held.strike) + 0.05 * (c.expiry - held.expiry).days
                picks.append((relief, c))
        picks.sort(key=lambda x: -x[0])
        return [(c, _roll_kind(held, c)) for _, c in picks[:cap]]

    return []


def candidates_from_slice(slice_df, held, held_mid, spot, *, held_stock=None,
                          sibling_legs=None, rolled_qty=None, context_warnings=None,
                          risk_free_curve=None, risk_free_rate=0.045, div_yield=0.0,
                          today=None, objectives=None, cap=_CAP_DEFAULT) -> list:
    """Roll candidates for a held OPTION. ``held`` is a dict
    ``{strike, expiry, right, quantity, delta[, multiplier]}``; ``held_mid`` its
    contemporaneous buy-to-close mid; ``held_stock`` an optional
    ``(shares, cost_basis_per_share)`` when the option is covered.

    When the held option is a leg of a detected structure, the caller passes
    ``sibling_legs`` — the structure's KEPT legs as entry-basis payoff leg dicts
    (the same assembly the payoff panel prices) — and ``rolled_qty``, the rolled
    leg's signed standard-contract slice. Each candidate then prices as the
    RESULTING STRUCTURE: kept legs at entry basis, the new leg at the current
    slice mid, sized to the rolled slice. Without a structure the resulting
    position falls back to the covered-call special case (covering stock enters
    only for a short call + long stock). Each candidate is priced through
    compute_payoff, plus the roll transaction's own net credit/debit."""
    today = today or date.today()
    objectives = list(objectives) if objectives else list(_DEFAULT_ROLL_OBJECTIVES)
    contracts = _parse_slice(slice_df)
    held_qty = int(held.get("quantity") or -1)
    held_delta = _num(held.get("delta"))
    held_sc = SliceContract(strike=float(held["strike"]), expiry=held["expiry"],
                            right=held["right"], mid=held_mid, delta=held_delta)
    # The transaction's size in standard-contract equivalents: the enclosing
    # structure's allocated slice when threaded through, else the full position
    # scaled by its own contract multiplier (100 -> unchanged).
    mult = _num(held.get("multiplier")) or 100.0
    qty_std = _qty(rolled_qty) if rolled_qty is not None else _qty(held_qty * (mult / 100.0))
    # A structure holding only a slice of the position rolls the SLICE — say so on
    # every candidate (the remainder sits outside this structure's economics).
    warns = list(context_warnings or [])
    full_std = abs(_num(held.get("quantity")) or 0.0) * (mult / 100.0)
    if rolled_qty is not None and full_std and abs(qty_std) + 1e-9 < full_std:
        warns.insert(0, (f"rolls the structure's {abs(qty_std):g}-contract slice of a "
                         f"{full_std:g}-contract position — the remainder sits outside "
                         "this structure"))
    # The resulting position's held side: the structure's kept legs when provided;
    # else the covering stock enters only for a covered-call roll (short call +
    # long stock) — a long-option roll is just the new option leg.
    stock_leg = None
    if sibling_legs is None and held_stock and held_qty < 0 and held_sc.right == "CALL":
        shares, basis = held_stock
        stock_leg = _stock_leg(shares, basis)
    base_legs = list(sibling_legs) if sibling_legs is not None \
        else ([stock_leg] if stock_leg else [])

    later = [c for c in contracts if c.right == held_sc.right
             and c.expiry > held_sc.expiry and c.mid is not None]

    out = []
    hm = _num(held_mid)
    for obj in objectives:
        picks = _select_roll(obj, held_sc, qty_std, held_mid, held_delta, later, cap)
        if not picks:
            continue
        for sc, kind in picks:
            new_leg = _option_leg(sc, qty_std, spot, curve=risk_free_curve,
                                  r_scalar=risk_free_rate, q=div_yield, today=today,
                                  role=_role_for(qty_std, sc.right))
            legs = base_legs + [new_leg]
            nm = _num(sc.mid)
            net_credit = qty_std * (hm - nm) * _MULT if (hm is not None and nm is not None) else None
            desc = (f"{kind.replace('_', ' ')} {held_sc.right.lower()} "
                    f"{held_sc.strike:g}->{sc.strike:g} @ {sc.expiry:%Y-%m-%d}")
            out.append(_finish(obj, kind, desc, legs, net_credit, spot, today,
                               new_leg=new_leg, extra_warnings=warns))
    return out


# ---------------------------------------------------------------------------
# Single-leg overlays (held is stock) — built after the rolls are verified
# ---------------------------------------------------------------------------

def overlays_from_slice(slice_df, spot, stock_shares, stock_basis, *,
                        risk_free_curve=None, risk_free_rate=0.045, div_yield=0.0,
                        today=None, cap=_CAP_DEFAULT) -> list:
    """Single-leg (and collar) overlays on a held stock position: covered call
    (max-premium), protective put and collar (add-hedge).

    Overlays are sized to the POSITION: ``floor(shares / 100)`` contracts on
    every option leg, with ``net_credit`` scaled to match — one contract against
    a 1,000-share holding is not a covered call, and its economics/greeks/PoP
    would describe mostly-uncovered stock. The stock leg keeps the FULL share
    count, so a non-round holding (e.g. 1,050 shares, 10 contracts) prices
    honestly with its residual uncovered shares. Under 100 shares there is
    nothing writable — returns no candidates."""
    today = today or date.today()
    n_contracts = int(_num(stock_shares) or 0) // 100
    if n_contracts < 1:
        return []
    contracts = _parse_slice(slice_df)
    stock_leg = _stock_leg(stock_shares, stock_basis)
    kw = dict(spot=spot, curve=risk_free_curve, r_scalar=risk_free_rate, q=div_yield, today=today)
    calls = [c for c in contracts if c.right == "CALL" and c.strike >= spot and c.mid]
    puts = [c for c in contracts if c.right == "PUT" and c.strike <= spot and c.mid]
    out = []

    # Covered call (sell an OTM call) — most premium first.
    for c in sorted(calls, key=lambda c: -(c.mid or 0))[:cap]:
        leg = _option_leg(c, -n_contracts, role="short_call", **kw)
        out.append(_finish(MAX_PREMIUM, "covered_call",
                           f"covered call {c.strike:g} @ {c.expiry:%Y-%m-%d}",
                           [stock_leg, leg], (c.mid or 0) * _MULT * n_contracts, spot, today,
                           new_leg=leg))

    # Protective put (buy an OTM put) — closest to spot first.
    for c in sorted(puts, key=lambda c: abs(c.strike - spot))[:cap]:
        leg = _option_leg(c, n_contracts, role="long_put", **kw)
        out.append(_finish(ADD_HEDGE, "protective_put",
                           f"protective put {c.strike:g} @ {c.expiry:%Y-%m-%d}",
                           [stock_leg, leg], -(c.mid or 0) * _MULT * n_contracts, spot, today,
                           new_leg=leg))

    # Collar (buy put + sell call at the same expiry) — pair the nearest of each.
    by_exp: dict = {}
    for c in contracts:
        if c.mid:
            by_exp.setdefault(c.expiry, {"C": [], "P": []})[c.right[0]].append(c)
    made = 0
    for exp in sorted(by_exp):
        cc = [c for c in by_exp[exp]["C"] if c.strike >= spot]
        pp = [c for c in by_exp[exp]["P"] if c.strike <= spot]
        if not cc or not pp or made >= cap:
            continue
        call = min(cc, key=lambda c: abs(c.strike - spot))
        put = min(pp, key=lambda c: abs(c.strike - spot))
        legs = [stock_leg, _option_leg(put, n_contracts, role="long_put", **kw),
                _option_leg(call, -n_contracts, role="short_call", **kw)]
        nc = ((call.mid or 0) - (put.mid or 0)) * _MULT * n_contracts
        out.append(_finish(ADD_HEDGE, "collar",
                           f"collar {put.strike:g}/{call.strike:g} @ {exp:%Y-%m-%d}",
                           legs, nc, spot, today))
        made += 1
    return out
