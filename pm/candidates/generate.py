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
    # Joint rolls only: the objective's own selection metric, computed at generation
    # over the WHOLE rolled set (joint net cash, total delta cut, total directional
    # move). The ranker's single-leg drivers don't generalise to a multi-leg roll,
    # so when present this IS the driver. None on every single-leg/overlay candidate;
    # deliberately None on joint costless too (the lexicographic dte+cap driver holds).
    joint_driver: Optional[float] = None


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


def _stock_leg(qty, cost_basis_per_share, position_id="held_stock", role=None) -> dict:
    role = role or ("long_stock" if (_num(qty) or 0) >= 0 else "short_stock")
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
        # Roll AWAY from the money AND extend — direction keyed on the rolled leg's
        # right (calls: higher strikes; puts: lower), which for a short leg is
        # exactly the assignment-risk-reducing move (the ITM-short workflow) and
        # for a long leg the premium-at-risk-reducing one. The cap keeps the
        # cheapest (near-costless) rolls; the ranker orders by directional relief.
        away = 1.0 if held.right == "CALL" else -1.0
        picks = []
        for c in later:
            if (c.strike - held.strike) * away <= 0:
                continue
            nm = _num(c.mid)
            cost = (abs(held_qty * (hm - nm) * _MULT)
                    if (hm is not None and nm is not None) else float("inf"))
            picks.append((cost, c))
        picks.sort(key=lambda x: x[0])
        return [(c, _roll_kind(held, c)) for _, c in picks[:cap]]

    if objective == COSTLESS:
        # The costless solve: inside the band, NEAREST expiry first — the ranker
        # then breaks ties within a day by the priced upside cap (max_profit).
        # Nearest-term picks fill the cap slice before any farther expiry, so no
        # NEARER-expiry candidate is ever discarded in favour of a farther one.
        # (Within one overfull expiry the slice keeps the furthest-from-money
        # strikes — a pre-pricing proxy; the band bounds what it can mis-keep.)
        band = _COSTLESS_PER_SHARE * _MULT * max(abs(held_qty), 1)
        picks = []
        for c in later:
            nm = _num(c.mid)
            if nm is None or hm is None:
                continue
            if abs(held_qty * (hm - nm) * _MULT) <= band:
                picks.append(c)
        away = 1.0 if held.right == "CALL" else -1.0
        picks.sort(key=lambda c: (c.expiry, -(c.strike - held.strike) * away))
        return [(c, _roll_kind(held, c)) for c in picks[:cap]]

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
    """Overlays on a held stock position. Long stock: covered call (max-premium),
    protective put and collar (add-hedge). SHORT stock: the mirror income write —
    the covered put (sell an OTM put against the short, max-premium); a hedge
    (protective call) is deliberately not generated.

    Overlays are sized to the POSITION: ``floor(|shares| / 100)`` contracts on
    every option leg, with ``net_credit`` scaled to match — one contract against
    a 1,000-share holding is not a covered call, and its economics/greeks/PoP
    would describe mostly-uncovered stock. The stock leg keeps the FULL share
    count, so a non-round holding (e.g. 1,050 shares, 10 contracts) prices
    honestly with its residual uncovered shares. Under 100 shares there is
    nothing writable — returns no candidates."""
    today = today or date.today()
    shares = _num(stock_shares) or 0
    n_contracts = int(abs(shares)) // 100
    if n_contracts < 1:
        return []
    contracts = _parse_slice(slice_df)
    stock_leg = _stock_leg(stock_shares, stock_basis)
    kw = dict(spot=spot, curve=risk_free_curve, r_scalar=risk_free_rate, q=div_yield, today=today)

    if shares < 0:
        # Covered put (sell an at/below-spot put against short stock) — most
        # premium first. The long-stock overlays don't apply to a short.
        out = []
        for c in sorted([c for c in contracts
                         if c.right == "PUT" and c.strike <= spot and c.mid],
                        key=lambda c: -(c.mid or 0))[:cap]:
            leg = _option_leg(c, -n_contracts, role="short_put", **kw)
            out.append(_finish(MAX_PREMIUM, "covered_put",
                               f"covered put {c.strike:g} @ {c.expiry:%Y-%m-%d}",
                               [stock_leg, leg], (c.mid or 0) * _MULT * n_contracts,
                               spot, today, new_leg=leg))
        return out

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


# ---------------------------------------------------------------------------
# Joint rolls — a SET of one structure's legs rolled together to ONE common new
# expiry, priced as the resulting structure (kept siblings + all new legs).
# ---------------------------------------------------------------------------

# Ceiling on the enumerated strike assignments per expiry. When the cross-group
# product would exceed it, every group's choice list is trimmed to an equal
# closest-move-first budget (the G-th root of the ceiling), so each rolled leg
# loses its most-distant strikes first — never one leg's whole range while
# another keeps junk combinations. Truncation is disclosed on the candidates of
# the affected expiry.
_JOINT_ENUM_CAP = 2000

JOINT_ROLL = "joint_roll"


def _away_dir(right) -> float:
    return -1.0 if str(right).upper().startswith("P") else 1.0


def _partition_rolled_groups(rolled) -> list:
    """[(leader, [(follower, strike_offset), ...]), ...] over the rolled entries.

    A SPREAD group is a same-right pair with equal |qty| and opposite signs —
    strikes and quantities come straight off the structure's allocation ledger;
    the PAIRING itself is chosen here, first match in the ledger's leg order
    (deterministic; only a hand-edited structure can offer more than one same-right
    pairing). The SHORT leg leads — it carries the assignment risk — and the long
    follows at its signed strike offset, so the spread's width is maintained
    exactly. Every other rolled leg is its own single-leg group (one strike DOF);
    an UNEQUAL-size same-right opposite-sign pair (a ratio) rolls as independents,
    which the caller discloses — its geometry is not maintained."""
    rolled = list(rolled)
    used: set = set()
    groups = []
    for i, a in enumerate(rolled):
        if i in used:
            continue
        partner = None
        for j in range(i + 1, len(rolled)):
            if j in used:
                continue
            b = rolled[j]
            if (b["right"] == a["right"]
                    and abs(abs(_num(b["qty"]) or 0) - abs(_num(a["qty"]) or 0)) < 1e-9
                    and ((_num(b["qty"]) or 0) > 0) != ((_num(a["qty"]) or 0) > 0)):
                partner = j
                break
        if partner is None:
            groups.append((a, []))
            used.add(i)
        else:
            b = rolled[partner]
            leader, follower = (a, b) if (_num(a["qty"]) or 0) < 0 else (b, a)
            groups.append((leader, [(follower, float(follower["strike"]) - float(leader["strike"]))]))
            used.update({i, partner})
    return groups


def _passes_band(c: SliceContract, delta_band) -> bool:
    if delta_band is None:
        return True
    d = _num(c.delta)
    if d is None:
        return False                     # a band means the delta must be known
    lo, hi = delta_band
    return lo <= abs(d) <= hi


def _joint_assignments(groups, contracts, expiry, delta_band):
    """(assignments, truncated) for one common expiry: every assignment covers every
    rolled leg — leaders enumerate their own right's strikes, spread followers sit
    at the maintained offset (the pair exists only if the follower contract is
    listed). Over the ceiling, every group trims to an equal closest-move budget;
    assignments return in closest-total-move-first order."""
    from itertools import product

    at_exp = [c for c in contracts if c.expiry == expiry and c.mid is not None
              and _passes_band(c, delta_band)]
    by_right: dict = {}
    for c in at_exp:
        by_right.setdefault(c.right, []).append(c)

    def _find(right, strike):
        for c in by_right.get(right, []):
            if abs(c.strike - strike) < 1e-6:
                return c
        return None

    per_group: list = []
    for leader, followers in groups:
        choices = []
        for c in sorted(by_right.get(leader["right"], []),
                        key=lambda c: abs(c.strike - float(leader["strike"]))):
            legs = [(leader, c)]
            ok = True
            for f, off in followers:
                fc = _find(f["right"], c.strike + off)
                if fc is None:
                    ok = False
                    break
                legs.append((f, fc))
            if ok:
                choices.append(legs)
        if not choices:
            # A rolled leg with no admissible target at this expiry — nothing
            # listed at the maintained offset, or nothing inside the pulled
            # slice window. No candidates rather than a guess.
            return [], False
        per_group.append(choices)

    total = 1
    for ch in per_group:
        total *= len(ch)
    truncated = total > _JOINT_ENUM_CAP
    if truncated:
        # Equal closest-move-first budget per group: each leg loses its most
        # distant strikes first, no leg keeps its whole range at another's cost.
        budget = max(1, int(_JOINT_ENUM_CAP ** (1.0 / len(per_group))))
        per_group = [ch[:budget] for ch in per_group]
    assignments = [[pair for grp in combo for pair in grp]
                   for combo in product(*per_group)]
    # Deterministic closest-total-move-first order for every downstream slice.
    assignments.sort(key=lambda a: sum(abs(sc.strike - float(r["strike"]))
                                       for r, sc in a))
    return assignments, truncated


def _joint_nc(assignment) -> Optional[float]:
    total = 0.0
    for r, sc in assignment:
        hm, nm = _num(r.get("mid")), _num(sc.mid)
        if hm is None or nm is None:
            return None
        total += (_num(r["qty"]) or 0.0) * (hm - nm) * _MULT
    return total


def _joint_away(assignment) -> float:
    return sum((sc.strike - float(r["strike"])) * _away_dir(r["right"])
               for r, sc in assignment)


def joint_candidates_from_slice(slice_df, rolled, spot, *, sibling_legs=None,
                                context_warnings=None, risk_free_curve=None,
                                risk_free_rate=0.045, div_yield=0.0, today=None,
                                objectives=None, cap=_CAP_DEFAULT,
                                delta_band=None) -> list:
    """Joint-roll candidates: every entry of ``rolled`` (dicts ``{position_id,
    strike, expiry, right, qty, mid, delta}`` — the structure's allocated slices
    with contemporaneous buy-to-close mids) rolls to ONE common new expiry, each
    leg to its own strike, priced as the resulting structure (``sibling_legs`` =
    the kept legs, entry basis) through the same engine path as single-leg rolls.
    Spread pairs inside the rolled set keep their width (one strike DOF; the short
    leads). ``delta_band`` optionally bounds every new leg's |delta|. The
    direction-aware objectives and the costless solve apply to the JOINT
    transaction's cash and the resulting structure's priced economics."""
    today = today or date.today()
    objectives = list(objectives) if objectives else list(_DEFAULT_ROLL_OBJECTIVES)
    rolled = list(rolled or [])
    if len(rolled) < 2:
        raise ValueError("joint_candidates_from_slice needs >= 2 rolled legs; "
                         "a single leg is the single-leg path")
    contracts = _parse_slice(slice_df)
    base = list(sibling_legs or [])
    warns = list(context_warnings or [])
    groups = _partition_rolled_groups(rolled)
    # A same-right opposite-sign pair that did NOT pair (unequal sizes — a ratio)
    # rolls as independents: say so, its geometry is not being maintained.
    solos = [g[0] for g in groups if not g[1]]
    if any(a["right"] == b["right"]
           and ((_num(a["qty"]) or 0) > 0) != ((_num(b["qty"]) or 0) > 0)
           for i, a in enumerate(solos) for b in solos[i + 1:]):
        warns.append("unequal-size same-right legs roll independently — their "
                     "spread geometry is not maintained")
    latest_old = max(r["expiry"] for r in rolled)
    expiries = sorted({c.expiry for c in contracts if c.expiry > latest_old})
    kw = dict(spot=spot, curve=risk_free_curve, r_scalar=risk_free_rate,
              q=div_yield, today=today)
    total_contracts = sum(abs(_num(r["qty"]) or 0.0) for r in rolled)
    old_dte = (latest_old - today).days

    per_exp: list = []                    # [(expiry, assignments, truncated)]
    for exp in expiries:
        assigns, cut = _joint_assignments(groups, contracts, exp, delta_band)
        if assigns:
            per_exp.append((exp, assigns, cut))
    if not per_exp:
        # Nothing strictly beyond the latest rolled expiry inside the pulled
        # window, or a rolled leg with no admissible target anywhere — the
        # anchor's slice window bounds what a joint roll can see.
        logger.info("joint roll: no admissible common expiry in the slice window "
                    "(latest rolled expiry %s)", latest_old)
        return []
    trunc_msg = ("joint enumeration truncated — each rolled leg's most-distant "
                 "strikes at this expiry were not scored")
    exp_warns = {exp: (warns + [trunc_msg] if cut else warns)
                 for exp, _a, cut in per_exp}
    per_exp = [(exp, assigns) for exp, assigns, _c in per_exp]

    def _emit(objective, exp, assignment, nc, joint_driver):
        new_legs = [_option_leg(sc, r["qty"], role=_role_for(_num(r["qty"]) or 0, sc.right), **kw)
                    for r, sc in assignment]
        legs = base + new_legs
        moves = " · ".join(f"{r['right'][0].lower()}{float(r['strike']):g}->{sc.strike:g}"
                           for r, sc in assignment)
        desc = f"joint roll {moves} @ {exp:%Y-%m-%d}"
        c = _finish(objective, JOINT_ROLL, desc, legs, nc, spot, today,
                    new_leg=None, extra_warnings=exp_warns[exp])
        c.new_leg_dte = (exp - today).days
        c.joint_driver = joint_driver
        return c

    out = []
    for obj in objectives:
        picks = []                        # (sort_key, expiry, assignment, nc, driver)
        if obj in (ROLL_FOR_CREDIT, MAX_PREMIUM):
            for exp, assigns in per_exp:
                for a in assigns:
                    nc = _joint_nc(a)
                    if nc is not None and nc > 0:
                        picks.append((-nc, exp, a, nc, nc))
        elif obj == EXTEND_DURATION:
            # One candidate per expiry: every leader at its nearest strike (the
            # choices are pre-ordered closest-move-first, so assignment 0 is it).
            for exp, assigns in per_exp:
                a = assigns[0]
                picks.append((-(exp - today).days, exp, a, _joint_nc(a),
                              float((exp - today).days)))
        elif obj == DEFEND_CUT_DELTA:
            for exp, assigns in per_exp:
                for a in assigns:
                    cut_total = 0.0
                    ok = True
                    for r, sc in a:
                        od, nd = _num(r.get("delta")), _num(sc.delta)
                        if od is None or nd is None or abs(nd) >= abs(od):
                            ok = False
                            break
                        cut_total += (abs(od) - abs(nd)) * abs(_num(r["qty"]) or 0.0)
                    if ok:
                        picks.append((-cut_total, exp, a, _joint_nc(a), cut_total))
        elif obj == ROLL_UP_OUT:
            for exp, assigns in per_exp:
                for a in assigns:
                    if any((sc.strike - float(r["strike"])) * _away_dir(r["right"]) <= 0
                           for r, sc in a):
                        continue
                    nc = _joint_nc(a)
                    cost = abs(nc) if nc is not None else float("inf")
                    driver = _joint_away(a) + 0.05 * ((exp - today).days - old_dte)
                    picks.append((cost, exp, a, nc, driver))
        elif obj == COSTLESS:
            band = _COSTLESS_PER_SHARE * _MULT * max(total_contracts, 1.0)
            for exp, assigns in per_exp:                 # expiries already ascend
                for a in sorted(assigns, key=lambda a: -_joint_away(a)):
                    nc = _joint_nc(a)
                    if nc is not None and abs(nc) <= band:
                        # key preserves (expiry asc, away desc) arrival order
                        picks.append((len(picks), exp, a, nc, None))
        else:
            continue
        picks.sort(key=lambda t: t[0])
        out.extend(_emit(obj, exp, a, nc, drv) for _, exp, a, nc, drv in picks[:cap])
    return out
