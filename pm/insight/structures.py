"""Structure detection — recognise the core multi-leg structures a book holds.

Pure and deterministic. Reads only the already-built ``Position`` list on each
``AccountState`` (no Bloomberg, no UI). Runs in the load path after the insight
engine; the UI reads ``AccountState.structures`` and never recomputes.

Holdings-anchored and **account-scoped**: legs group on the same underlying via
``option.underlying_symbol == stock.symbol`` (bare tickers). Trades only
*corroborate* — two option legs opened on the same day with opening actions
raise a grouping's confidence; a missing trade never lowers it (a long-held
stock leg has no trade in the few-week window).

Every structure is a claim on a **signed-quantity slice** of each leg, never the
whole leg: a partial cover emits a covered slice + a residual slice; an
over-write emits a covered slice + a naked-excess slice. Per leg, the allocated
quantities never exceed the leg and never flip its sign.

Two suites are detected. The **stock-anchored / residual suite** is imperative
(covered call full / partial / over-write, collar, married put, covered /
cash-secured put, vertical, straddle / strangle, the naked sweep). The
**combination suite** is declarative: each shape is a spec (legs, signs, qty
ratio, strike/expiry relations) matched by one generic engine — box, iron
butterfly / iron condor / condor, double diagonal, jelly roll, conversion /
reversal, butterfly (incl. broken-wing), jade lizard, ladder, ratio spread /
backspread, synthetic long / short, risk reversal, calendar, diagonal.

**Stock coverage is senior to any composite claim.** Before the exact-composite
tier runs, the long stock's covering capacity is reserved against short calls
(slice-aware, per the leg's own multiplier); composites match only the
unreserved remainder. Mislabelling a spread is cosmetic; reporting a client as
naked when they hold deliverable stock is a risk-reporting error. The accepted
tradeoff: a genuine box on a name where stock is also held reads as covered
call + partial box. (Conversions are exempt — their own stock leg IS the cover.)

**A poor-man's covered call is a qualifier, not a type.** The stable type is
``diagonal``; "PMCC" renders as a live qualifier at display time (long leg
trading like stock). A type that depends on a market price is an unstable
identity — a PMCC whose underlying fell would silently re-type on a fixed leg
set, invalidating stored confirms and re-labelling the grid day to day. Type
describes what was put on.

**Uncovered short slices carry the naked-excess role in any type**: within a
claimed structure, a short option is covered by in-structure long options of
the same right with same-or-later expiry (shares-weighted, whole contracts;
an earlier-dated long dies first and covers nothing beyond its own life) or by
the structure's own stock; the uncovered remainder is sliced out under
``naked_excess_short_call`` / ``naked_excess_short_put`` so coverage surfaces
and the coverage-breach alert read the role, not the type.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import pandas as pd

from pm.ingest.position_builder import Position

# ---------------------------------------------------------------------------
# Confidence bands + structure types
# ---------------------------------------------------------------------------
HIGH = "high"
MEDIUM = "medium"
LOW_AMBIGUOUS = "low_ambiguous"

COVERED_CALL = "covered_call"
COLLAR = "collar"
VERTICAL = "vertical"
COVERED_PUT = "covered_put"
CASH_SECURED_PUT = "cash_secured_put"
STRADDLE = "straddle"
STRANGLE = "strangle"
RESIDUAL_LONG = "residual_long"
NAKED_EXCESS_SHORT_CALL = "naked_excess_short_call"

# The combination suite (each a self-contained spec behind the generic matcher).
BOX = "box"
IRON_BUTTERFLY = "iron_butterfly"
IRON_CONDOR = "iron_condor"
CONDOR = "condor"
DOUBLE_DIAGONAL = "double_diagonal"
JELLY_ROLL = "jelly_roll"
CONVERSION = "conversion"
REVERSAL = "reversal"
BUTTERFLY = "butterfly"          # incl. broken-wing (named in the rationale)
JADE_LIZARD = "jade_lizard"
LADDER = "ladder"                # incl. Christmas tree
MARRIED_PUT = "married_put"
RATIO_SPREAD = "ratio_spread"
BACKSPREAD = "backspread"
SYNTHETIC_LONG = "synthetic_long"
SYNTHETIC_SHORT = "synthetic_short"
RISK_REVERSAL = "risk_reversal"
CALENDAR = "calendar"
DIAGONAL = "diagonal"            # PMCC is a render-time qualifier on this type

# Leg role (not a type): the uncovered short-put analog of naked_excess_short_call.
NAKED_EXCESS_SHORT_PUT = "naked_excess_short_put"

# Types whose structural HIGH band demotes to MEDIUM while sibling option legs
# on the same underlying remain unexplained (never over-claim a structure on a
# partial read; co-opened legs keep their trade-corroborated band). The pass-5
# put sweep is included: a leftover short put beside an unexplained sibling leg
# (e.g. the long call of a cross-expiry risk reversal no pair spec can claim)
# may be that combination's financing side, not an income put — its HIGH is
# structural only, so it demotes like an uncorroborated combo.
_SIBLING_DEMOTABLE = {BOX, IRON_BUTTERFLY, IRON_CONDOR, CONDOR, DOUBLE_DIAGONAL,
                      JELLY_ROLL, CONVERSION, REVERSAL, BUTTERFLY, JADE_LIZARD,
                      LADDER, MARRIED_PUT, CASH_SECURED_PUT, COVERED_PUT}

_MULT = 100  # standard option contract multiplier (the fallback)
_OPEN_ACTIONS = {"Buy to Open", "Sell to Open"}


def _opt_mult(o) -> int | float:
    """The leg's own shares-per-contract, read from ``Position.multiplier``
    (100 fallback). Every contracts<->shares conversion in detection goes
    through this so an adjusted/mini contract covers exactly the shares it
    delivers — a hardcoded 100 turned a fully-covered mini write into a false
    naked-excess (and its false tier-1 coverage breach). Integral multipliers
    stay ints so share counts (and their rationale strings) stay integers."""
    m = _num(getattr(o, "multiplier", None))
    if m is None or m <= 0:
        return _MULT
    return int(m) if float(m).is_integer() else m


@dataclass
class StructureLeg:
    position_id: str
    allocated_qty: float   # signed slice; abs(allocated) <= abs(leg.quantity), same sign
    role: str              # e.g. long_stock, short_call, long_put, residual_long, naked_excess_short_call


@dataclass
class Structure:
    structure_id: str
    account: str
    underlying: str
    type: str
    confidence_band: str
    status: str                 # "proposed" until a user confirm/override flips it
    legs: list[StructureLeg]
    rationale_trace: dict
    source: str
    contention_group: Optional[str] = None  # set when ranked alternatives compete
    resolved_at: Optional[str] = None        # timestamp when the user confirmed/edited it


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def _sid(account: str, underlying: str, typ: str, leg_pids: list[str], suffix: str = "") -> str:
    """Deterministic id, stable for the same leg-set so confirm/override persistence can key on it."""
    base = f"{account}|{underlying}|{typ}|" + ",".join(sorted(leg_pids))
    return base + (f"|{suffix}" if suffix else "")


def _co_opened(leg_keys: list[str], trades_df) -> bool:
    """True if every option leg has an *opening* trade on a shared trade_date —
    co-opening corroborates that the legs were intended as one structure."""
    if trades_df is None or getattr(trades_df, "empty", True) or not leg_keys:
        return False
    if "option_contract_key" not in trades_df.columns or "trade_date" not in trades_df.columns:
        return False
    action_col = "option_lifecycle_action"
    date_sets: list[set] = []
    for key in leg_keys:
        sub = trades_df[trades_df["option_contract_key"] == key]
        if action_col in trades_df.columns:
            sub = sub[sub[action_col].isin(_OPEN_ACTIONS)]
        if sub.empty:
            return False
        date_sets.append(set(pd.to_datetime(sub["trade_date"]).dt.date))
    return bool(set.intersection(*date_sets))


def _trace(inputs: dict, computation: str, result: str, thresholds: Optional[dict] = None) -> dict:
    """Canonical trace dict (matches the engine's signal/fire trace shape)."""
    return {
        "inputs": inputs,
        "computation": computation,
        "thresholds": thresholds or {},
        "result": result,
    }


# ---------------------------------------------------------------------------
# The combination suite — declarative specs + one generic matcher
# ---------------------------------------------------------------------------
# Each spec is a shape over aggregated remaining option slices: legs carry a
# right ('C' / 'P', or 'X' bound to one right per match), a sign, a per-unit
# contract ratio, and strike / expiry slot names; strike slots bind strictly
# ascending distinct strikes, expiry slots strictly ascending distinct
# expiries. ``mirror`` also matches the all-signs-flipped orientation. One
# matcher walks the table in priority order, claims signed slices through the
# same remaining-quantity ledger the imperative passes use, and repeats a spec
# until it no longer matches. Deterministic tie-breaks: earliest expiries, then
# tightest strike span, then lowest strikes, base orientation before mirror.
@dataclass(frozen=True)
class _Spec:
    typ: str
    legs: tuple          # ((right, sign, ratio, k_slot, t_slot), ...)
    k_slots: tuple       # ascending distinct strike slots
    t_slots: tuple       # ascending distinct expiry slots
    mirror: bool
    word: str            # base-orientation phrase (trace result)
    flip_word: str       # mirrored-orientation phrase


_TIER_A_PRE = (
    _Spec(BOX,
          (("C", +1, 1, "K1", "T1"), ("C", -1, 1, "K2", "T1"),
           ("P", +1, 1, "K2", "T1"), ("P", -1, 1, "K1", "T1")),
          ("K1", "K2"), ("T1",), True, "box (long)", "box (short)"),
    _Spec(IRON_BUTTERFLY,
          (("P", +1, 1, "K1", "T1"), ("P", -1, 1, "K2", "T1"),
           ("C", -1, 1, "K2", "T1"), ("C", +1, 1, "K3", "T1")),
          ("K1", "K2", "K3"), ("T1",), True, "iron butterfly", "iron butterfly (reverse)"),
    _Spec(IRON_CONDOR,
          (("P", +1, 1, "K1", "T1"), ("P", -1, 1, "K2", "T1"),
           ("C", -1, 1, "K3", "T1"), ("C", +1, 1, "K4", "T1")),
          ("K1", "K2", "K3", "K4"), ("T1",), True, "iron condor", "iron condor (reverse)"),
    _Spec(CONDOR,
          (("X", +1, 1, "K1", "T1"), ("X", -1, 1, "K2", "T1"),
           ("X", -1, 1, "K3", "T1"), ("X", +1, 1, "K4", "T1")),
          ("K1", "K2", "K3", "K4"), ("T1",), True, "condor (long)", "condor (short)"),
    _Spec(DOUBLE_DIAGONAL,
          (("P", +1, 1, "K1", "T2"), ("P", -1, 1, "K2", "T1"),
           ("C", -1, 1, "K3", "T1"), ("C", +1, 1, "K4", "T2")),
          ("K1", "K2", "K3", "K4"), ("T1", "T2"), True, "double diagonal", "double diagonal (reverse)"),
    _Spec(JELLY_ROLL,
          (("C", +1, 1, "K1", "T1"), ("P", -1, 1, "K1", "T1"),
           ("C", -1, 1, "K1", "T2"), ("P", +1, 1, "K1", "T2")),
          ("K1",), ("T1", "T2"), True, "jelly roll", "jelly roll (reverse)"),
)
_TIER_A_POST = (
    _Spec(BUTTERFLY,
          (("X", +1, 1, "K1", "T1"), ("X", -1, 2, "K2", "T1"), ("X", +1, 1, "K3", "T1")),
          ("K1", "K2", "K3"), ("T1",), True, "butterfly (long)", "butterfly (short)"),
    _Spec(JADE_LIZARD,
          (("P", -1, 1, "K1", "T1"), ("C", -1, 1, "K2", "T1"), ("C", +1, 1, "K3", "T1")),
          ("K1", "K2", "K3"), ("T1",), False, "jade lizard", ""),
    _Spec(LADDER,
          (("X", +1, 1, "K1", "T1"), ("X", -1, 1, "K2", "T1"), ("X", -1, 1, "K3", "T1")),
          ("K1", "K2", "K3"), ("T1",), True, "ladder", "ladder (inverted)"),
    _Spec(LADDER,
          (("X", -1, 1, "K1", "T1"), ("X", -1, 1, "K2", "T1"), ("X", +1, 1, "K3", "T1")),
          ("K1", "K2", "K3"), ("T1",), True, "ladder", "ladder (inverted)"),
)
_PAIR_SPECS = (
    _Spec(SYNTHETIC_LONG,
          (("C", +1, 1, "K1", "T1"), ("P", -1, 1, "K1", "T1")),
          ("K1",), ("T1",), False, "synthetic long stock", ""),
    _Spec(SYNTHETIC_SHORT,
          (("C", -1, 1, "K1", "T1"), ("P", +1, 1, "K1", "T1")),
          ("K1",), ("T1",), False, "synthetic short stock", ""),
    _Spec(RISK_REVERSAL,
          (("P", -1, 1, "K1", "T1"), ("C", +1, 1, "K2", "T1")),
          ("K1", "K2"), ("T1",), True, "risk reversal (bullish)", "risk reversal (bearish)"),
    _Spec(RISK_REVERSAL,
          (("C", +1, 1, "K1", "T1"), ("P", -1, 1, "K2", "T1")),
          ("K1", "K2"), ("T1",), True, "risk reversal (bullish)", "risk reversal (bearish)"),
)
_XEXP_SPECS = (
    _Spec(CALENDAR,
          (("X", -1, 1, "K1", "T1"), ("X", +1, 1, "K1", "T2")),
          ("K1",), ("T1", "T2"), True, "calendar (long)", "calendar (reverse)"),
    _Spec(DIAGONAL,
          (("X", -1, 1, "K1", "T1"), ("X", +1, 1, "K2", "T2")),
          ("K1", "K2"), ("T1", "T2"), True, "diagonal", "diagonal (reverse)"),
    _Spec(DIAGONAL,
          (("X", -1, 1, "K2", "T1"), ("X", +1, 1, "K1", "T2")),
          ("K1", "K2"), ("T1", "T2"), True, "diagonal", "diagonal (reverse)"),
)

_RIGHT = {"C": "CALL", "P": "PUT"}


def _reserve_stock_cover(stock_rem: float, options, opt_rem) -> dict[str, int]:
    """Contracts per short-call position that the available long stock will cover
    — the same greedy (expiry, strike) walk the covered-call pass takes, each
    contract claiming its own ``_opt_mult`` shares. Computed BEFORE the exact-
    composite tier so a composite can only claim the uncovered remainder:
    stock coverage is senior to any composite claim."""
    res: dict[str, int] = {}
    avail = stock_rem if stock_rem and stock_rem > 0 else 0.0
    if avail <= 0:
        return res
    shorts = [o for o in options if o.right == "CALL" and opt_rem.get(o.position_id, 0) < 0]
    for o in sorted(shorts, key=lambda o: (str(o.expiry), o.strike or 0)):
        mult = _opt_mult(o)
        have = int(abs(opt_rem[o.position_id]))
        take = min(have, int(avail // mult))
        if take > 0:
            res[o.position_id] = take
            avail -= take * mult
    return res


def _avail_qty(o, opt_rem, reserve) -> float:
    """A position's claimable remaining quantity — the raw remainder less any
    stock-cover reservation on a short call (magnitude shrinks toward zero)."""
    rem = opt_rem.get(o.position_id, 0.0)
    if reserve and rem < 0:
        held = reserve.get(o.position_id, 0)
        if held:
            rem = min(0.0, rem + held)
    return rem


def _slice_groups(options, opt_rem, reserve) -> dict:
    """Aggregated remaining slices keyed ``(right, expiry, strike, mult)`` —
    the matching substrate. Mixed contract sizes never share a group (a mini
    must not pair as a standard contract's equal)."""
    groups: dict = defaultdict(list)
    for o in options:
        if o.expiry is None or o.strike is None:
            continue
        if _avail_qty(o, opt_rem, reserve) == 0:
            continue
        groups[(o.right, o.expiry, o.strike, _opt_mult(o))].append(o)
    return groups


def _group_avail(groups, key, sign, opt_rem, reserve) -> int:
    total = 0.0
    for o in groups.get(key, []):
        a = _avail_qty(o, opt_rem, reserve)
        if (a > 0) == (sign > 0) and a != 0:
            total += abs(a)
    return int(total)


def _claim_group(groups, key, sign, contracts, opt_rem, reserve, claimed: list) -> None:
    """Take ``contracts`` (unsigned) from the group's positions in stable pid
    order, appending one claimed-slice record per position touched."""
    right, expiry, strike, mult = key
    need = contracts
    for o in sorted(groups.get(key, []), key=lambda o: o.position_id):
        if need <= 0:
            break
        a = _avail_qty(o, opt_rem, reserve)
        if (a > 0) != (sign > 0) or a == 0:
            continue
        take = min(int(abs(a)), need)
        opt_rem[o.position_id] -= sign * take
        need -= take
        claimed.append({"pid": o.position_id, "qty": sign * take, "right": right,
                        "strike": strike, "expiry": expiry, "mult": mult,
                        "key": o.option_contract_key})


def _cover_split(claimed) -> list[StructureLeg]:
    """Claimed slices -> StructureLegs with the in-structure cover rule: a short
    is covered by same-right longs of same-or-later expiry (shares-weighted,
    whole contracts — a partially-covered contract cannot deliver, so the
    remainder rounds UP to naked); the uncovered remainder is sliced out under
    the naked-excess role. Cover order leaves the furthest-OTM short naked:
    calls cover ascending strike, puts descending."""
    legs: list[StructureLeg] = []
    for right, naked_role in (("CALL", NAKED_EXCESS_SHORT_CALL), ("PUT", NAKED_EXCESS_SHORT_PUT)):
        side = [c for c in claimed if c["right"] == right]
        if not side:
            continue
        longs = [c for c in side if c["qty"] > 0]
        shorts = [c for c in side if c["qty"] < 0]
        for c in longs:
            legs.append(StructureLeg(c["pid"], c["qty"], "long_" + right.lower()))
        pool = [[c["expiry"], abs(c["qty"]) * c["mult"]] for c in longs]
        for c in sorted(shorts, key=lambda c: (c["strike"] or 0), reverse=(right == "PUT")):
            mult = c["mult"]
            have = int(abs(c["qty"]))
            need = have * mult
            for slot in pool:
                if need <= 0:
                    break
                if c["expiry"] is not None and slot[0] is not None and slot[0] < c["expiry"]:
                    continue  # an earlier-dated long covers nothing beyond its own life
                take = min(need, slot[1])
                slot[1] -= take
                need -= take
            naked = min(have, -(-int(need) // mult)) if need > 0 else 0
            if have - naked > 0:
                legs.append(StructureLeg(c["pid"], -(have - naked), "short_" + right.lower()))
            if naked > 0:
                legs.append(StructureLeg(c["pid"], -naked, naked_role))
    return legs


def _slices_phrase(claimed) -> str:
    parts = []
    for c in sorted(claimed, key=lambda c: (str(c["expiry"]), c["strike"] or 0, c["right"])):
        side = "long" if c["qty"] > 0 else "short"
        r = "C" if c["right"] == "CALL" else "P"
        parts.append(f"{side} {int(abs(c['qty']))} {c['strike']:g}{r} exp {c['expiry']}")
    return " + ".join(parts)


def _match_spec_once(spec: _Spec, options, opt_rem, reserve, exact=False):
    """Find the best binding for one spec and claim it. Returns
    ``(claimed_slices, meta)`` or ``(None, None)`` when nothing matches.

    ``exact`` (the exact-composite tier): the binding must consume every
    participating strike group WHOLE — a fully-determined match, never a
    proportion carved out of a lopsided book. A partial carve fabricates
    structure (and can mark contracts naked that the name's remaining longs
    cover); the lopsided book falls through to the lower tiers instead."""
    groups = _slice_groups(options, opt_rem, reserve)
    if not groups:
        return None, None
    wants_x = any(l[0] == "X" for l in spec.legs)
    candidates = []
    for xr in (("CALL", "PUT") if wants_x else (None,)):
        rights_used = {_RIGHT.get(l[0], xr) for l in spec.legs}
        exps = sorted({e for (r, e, k, m) in groups if r in rights_used})
        if not exps or len(exps) > 8:
            continue
        for tvals in combinations(exps, len(spec.t_slots)):
            tmap = dict(zip(spec.t_slots, tvals))
            ks = sorted({k for (r, e, k, m) in groups
                         if r in rights_used and e in tvals})
            if len(ks) > 16:
                continue
            mults = sorted({m for (r, e, k, m) in groups if r in rights_used})
            for kvals in combinations(ks, len(spec.k_slots)):
                kmap = dict(zip(spec.k_slots, kvals))
                for flip in ((1, -1) if spec.mirror else (1,)):
                    for mult in mults:
                        n = None
                        avails = []
                        for right, sign, ratio, kslot, tslot in spec.legs:
                            key = (_RIGHT.get(right, xr), tmap[tslot], kmap[kslot], mult)
                            avail = _group_avail(groups, key, sign * flip, opt_rem, reserve)
                            avails.append((avail, ratio))
                            cap = avail // ratio
                            n = cap if n is None else min(n, cap)
                            if n < 1:
                                break
                        if not (n and n >= 1):
                            continue
                        if exact and any(a != n * r for a, r in avails):
                            continue
                        candidates.append((tvals, kvals[-1] - kvals[0], kvals,
                                           0 if flip == 1 else 1, mult, xr, n))
    if not candidates:
        return None, None
    tvals, _span, kvals, flipped, mult, xr, n = min(
        candidates, key=lambda c: (tuple(str(t) for t in c[0]), c[1], c[2], c[3], c[4]))
    flip = -1 if flipped else 1
    tmap = dict(zip(spec.t_slots, tvals))
    kmap = dict(zip(spec.k_slots, kvals))
    claimed: list = []
    for right, sign, ratio, kslot, tslot in spec.legs:
        key = (_RIGHT.get(right, xr), tmap[tslot], kmap[kslot], mult)
        _claim_group(groups, key, sign * flip, n * ratio, opt_rem, reserve, claimed)
    meta = {"n": n, "flip": flip, "right": xr,
            "strikes": list(kvals), "expiries": [str(t) for t in tvals]}
    return claimed, meta


def _emit_combo(spec: _Spec, meta, claimed, account, underlying, trades_df,
                out: list, demotable: list, tier_a: bool) -> None:
    word = spec.flip_word if (meta["flip"] < 0 and spec.flip_word) else spec.word
    if spec.typ == BUTTERFLY:
        k1, k2, k3 = meta["strikes"]
        if abs((k2 - k1) - (k3 - k2)) > 1e-9:
            word = "broken-wing " + word
    keys = sorted({c["key"] for c in claimed if c.get("key")})
    co = _co_opened(keys, trades_df)
    legs = _cover_split(claimed)
    phrase = _slices_phrase(claimed)
    if tier_a:
        band = HIGH
    else:
        band = HIGH if co else MEDIUM
    st = Structure(
        structure_id=_sid(account, underlying, spec.typ, sorted({c["pid"] for c in claimed})),
        account=account, underlying=underlying, type=spec.typ,
        confidence_band=band, status="proposed", legs=legs,
        rationale_trace=_trace(
            {"strikes": {"value": meta["strikes"], "source": "EXTRACT:option_strike"},
             "expiries": {"value": meta["expiries"], "source": "EXTRACT:option_expiration"},
             "qty": {"value": meta["n"], "source": "EXTRACT:quantity"},
             "co_opened": {"value": co, "source": "computed:trade corroboration"}},
            f"{meta['n']}x {underlying} {word}: {phrase}",
            word),
        source=f"detector:{spec.typ}")
    out.append(st)
    if tier_a and not co and spec.typ in _SIBLING_DEMOTABLE:
        demotable.append(st)


def _run_specs(specs, account, underlying, options, opt_rem, trades_df,
               out: list, demotable: list, *, tier_a: bool, reserve_fn=None) -> None:
    for spec in specs:
        while True:
            reserve = reserve_fn() if reserve_fn else None
            claimed, meta = _match_spec_once(spec, options, opt_rem, reserve,
                                             exact=tier_a)
            if claimed is None:
                break
            _emit_combo(spec, meta, claimed, account, underlying, trades_df,
                        out, demotable, tier_a)


def _detect_conversion_reversal(account, underlying, stock_pid, stock_rem,
                                options, opt_rem, trades_df,
                                out: list, demotable: list):
    """Conversion (long stock + short call + long put, same strike AND expiry)
    / reversal (the short-stock mirror). The exact-strike specialization of the
    collar / covered-put reads, so it runs before them; the structure's own
    stock IS the short side's cover (exempt from the composite reservation).
    Returns the updated stock remainder."""
    while stock_pid is not None and stock_rem:
        direction = 1 if stock_rem > 0 else -1
        call_sign = -direction          # conversion shorts the call; reversal longs it
        groups = _slice_groups(options, opt_rem, None)
        best = None
        for (right, expiry, strike, mult) in sorted(
                groups, key=lambda k: (str(k[1]), k[2], k[3])):
            if right != "CALL":
                continue
            ckey = ("CALL", expiry, strike, mult)
            pkey = ("PUT", expiry, strike, mult)
            n = min(_group_avail(groups, ckey, call_sign, opt_rem, None),
                    _group_avail(groups, pkey, -call_sign, opt_rem, None),
                    int(abs(stock_rem) // mult))
            if n >= 1:
                best = (ckey, pkey, mult, n)
                break
        if best is None:
            return stock_rem
        ckey, pkey, mult, n = best
        claimed: list = []
        _claim_group(groups, ckey, call_sign, n, opt_rem, None, claimed)
        _claim_group(groups, pkey, -call_sign, n, opt_rem, None, claimed)
        shares = n * mult * direction
        typ = CONVERSION if direction > 0 else REVERSAL
        legs = [StructureLeg(stock_pid, shares,
                             "long_stock" if direction > 0 else "short_stock")]
        for c in claimed:   # the stock is the cover — plain option roles
            side = "long" if c["qty"] > 0 else "short"
            legs.append(StructureLeg(c["pid"], c["qty"], f"{side}_{c['right'].lower()}"))
        keys = sorted({c["key"] for c in claimed if c.get("key")})
        co = _co_opened(keys, trades_df)
        strike, expiry = ckey[2], ckey[1]
        st = Structure(
            structure_id=_sid(account, underlying, typ,
                              sorted({stock_pid} | {c["pid"] for c in claimed})),
            account=account, underlying=underlying, type=typ,
            confidence_band=HIGH, status="proposed", legs=legs,
            rationale_trace=_trace(
                {"strike": {"value": strike, "source": "EXTRACT:option_strike"},
                 "expiry": {"value": str(expiry), "source": "EXTRACT:option_expiration"},
                 "shares": {"value": abs(shares), "source": "EXTRACT:quantity"},
                 "co_opened": {"value": co, "source": "computed:trade corroboration"}},
                f"{abs(shares):g} sh {'long' if direction > 0 else 'short'} + "
                f"{n}x {underlying} {strike:g} call/put pair, matched strike + expiry {expiry}",
                typ),
            source=f"detector:{typ}")
        out.append(st)
        if not co:
            demotable.append(st)
        stock_rem -= shares
    return stock_rem


def _detect_married_put(account, underlying, stock_pid, stock_rem,
                        options, opt_rem, trades_df, out: list, demotable: list):
    """Married / protective put: stock the covered-call pass left + long puts,
    claimed greedily in (expiry, strike) order, each contract protecting its
    own ``_opt_mult`` shares. Runs AFTER the covered-call pass so stock
    coverage of short calls stays senior. Returns the updated stock remainder."""
    if stock_pid is None or not stock_rem or stock_rem <= 0:
        return stock_rem
    puts = [o for o in options if o.right == "PUT"
            and opt_rem.get(o.position_id, 0) > 0
            and o.strike is not None and o.expiry is not None]
    claimed: list = []
    for o in sorted(puts, key=lambda o: (str(o.expiry), o.strike or 0)):
        mult = _opt_mult(o)
        lots = min(int(opt_rem[o.position_id]), int(stock_rem // mult))
        if lots <= 0:
            continue
        opt_rem[o.position_id] -= lots
        stock_rem -= lots * mult
        claimed.append({"pid": o.position_id, "qty": lots, "right": "PUT",
                        "strike": o.strike, "expiry": o.expiry, "mult": mult,
                        "key": o.option_contract_key})
    if not claimed:
        return stock_rem
    shares = sum(c["qty"] * c["mult"] for c in claimed)
    legs = [StructureLeg(stock_pid, shares, "long_stock")]
    legs += [StructureLeg(c["pid"], c["qty"], "long_put") for c in claimed]
    keys = sorted({c["key"] for c in claimed if c.get("key")})
    co = _co_opened(keys, trades_df)
    st = Structure(
        structure_id=_sid(account, underlying, MARRIED_PUT,
                          sorted({stock_pid} | {c["pid"] for c in claimed})),
        account=account, underlying=underlying, type=MARRIED_PUT,
        confidence_band=HIGH, status="proposed", legs=legs,
        rationale_trace=_trace(
            {"protected_shares": {"value": shares, "source": "EXTRACT:quantity"},
             "puts": {"value": _slices_phrase(claimed), "source": "EXTRACT:quantity"},
             "co_opened": {"value": co, "source": "computed:trade corroboration"}},
            f"{int(shares)} {underlying} shares floored by {_slices_phrase(claimed)}",
            "married put (stock floored by the long put)"),
        source=f"detector:{MARRIED_PUT}")
    out.append(st)
    if not co:
        demotable.append(st)
    return stock_rem


def _detect_ratio_backspread(account, underlying, options, opt_rem, trades_df,
                             out: list) -> None:
    """Ratio spread / backspread: same right, expiry and contract size, exactly
    two strikes, opposite-sign totals, UNEQUAL quantities — the guard that lets
    a true 1:1 fall through to the vertical pass. The fused reading claims BOTH
    legs whole; the unpaired short contracts slice out under the naked-excess
    role (coverage surfaces and the coverage-breach alert stay live)."""
    while True:
        groups = _slice_groups(options, opt_rem, None)
        match = None
        for right in ("CALL", "PUT"):
            per_et: dict = defaultdict(set)
            for (r, e, k, m) in groups:
                if r == right:
                    per_et[(e, m)].add(k)
            for (e, m), ks in sorted(per_et.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
                if len(ks) != 2:
                    continue
                k1, k2 = sorted(ks)
                a1 = sum(_avail_qty(o, opt_rem, None) for o in groups[(right, e, k1, m)])
                a2 = sum(_avail_qty(o, opt_rem, None) for o in groups[(right, e, k2, m)])
                if not a1 or not a2 or (a1 > 0) == (a2 > 0) or abs(a1) == abs(a2):
                    continue
                match = (right, e, m, k1, k2, a1, a2)
                break
            if match:
                break
        if match is None:
            return
        right, e, m, k1, k2, a1, a2 = match
        claimed: list = []
        for k, a in ((k1, a1), (k2, a2)):
            _claim_group(groups, (right, e, k, m), 1 if a > 0 else -1,
                         int(abs(a)), opt_rem, None, claimed)
        n_long = int(abs(a1 if a1 > 0 else a2))
        n_short = int(abs(a1 if a1 < 0 else a2))
        typ = RATIO_SPREAD if n_short > n_long else BACKSPREAD
        word = (f"{right.lower()} {'ratio spread' if typ == RATIO_SPREAD else 'backspread'}"
                f" ({n_long} vs {n_short})")
        keys = sorted({c["key"] for c in claimed if c.get("key")})
        co = _co_opened(keys, trades_df)
        out.append(Structure(
            structure_id=_sid(account, underlying, typ,
                              sorted({c["pid"] for c in claimed})),
            account=account, underlying=underlying, type=typ,
            confidence_band=HIGH if co else MEDIUM, status="proposed",
            legs=_cover_split(claimed),
            rationale_trace=_trace(
                {"strikes": {"value": [k1, k2], "source": "EXTRACT:option_strike"},
                 "expiry": {"value": str(e), "source": "EXTRACT:option_expiration"},
                 "long_contracts": {"value": n_long, "source": "EXTRACT:quantity"},
                 "short_contracts": {"value": n_short, "source": "EXTRACT:quantity"},
                 "co_opened": {"value": co, "source": "computed:trade corroboration"}},
                f"{underlying} {word} {k1:g}/{k2:g} exp {e}: {_slices_phrase(claimed)}",
                word),
            source=f"detector:{typ}"))


def _reconcile_naked_roles(structures, stock_shares, options) -> None:
    """Cap every naked-excess mark to the NAME-LEVEL truth. First the plain
    short marks draw down the name's cover capacity — same-right long-option
    shares (same-or-later expiry) for spread-covered shorts, the stock slot for
    stock-anchored structures — since a plain mark represents an exclusive
    in-structure cover claim. Whatever capacity remains then covers the
    naked-marked slices, which downgrade to the plain role: no decomposition
    may report a short as naked while cover for it sits elsewhere on the name
    (unclaimed, or spare inside another structure). Capping only — plain marks
    are never raised to naked here (the per-structure splits and the sweep
    already mark every genuinely uncovered short)."""
    by_pid = {o.position_id: o for o in options}

    def _draw(pool, expiry, need):
        for slot in pool:
            if need <= 0:
                break
            if expiry is not None and slot[0] is not None and slot[0] < expiry:
                continue
            take = min(need, slot[1])
            slot[1] -= take
            need -= take
        return need

    for right, naked_role, short_role in (
            ("CALL", NAKED_EXCESS_SHORT_CALL, "short_call"),
            ("PUT", NAKED_EXCESS_SHORT_PUT, "short_put")):
        marked = any(l.role == naked_role
                     for st in structures if not st.contention_group
                     for l in st.legs)
        if not marked:
            continue
        pool = sorted(([o.expiry, (_num(o.quantity) or 0.0) * _opt_mult(o)]
                       for o in options
                       if o.right == right and (_num(o.quantity) or 0) > 0),
                      key=lambda s: (s[0] is None, str(s[0])))
        stock_avail = float(stock_shares) if (right == "CALL" and stock_shares
                                              and stock_shares > 0) else 0.0
        # 1) plain shorts consume the capacity their structures claimed
        for st in structures:
            if st.contention_group:
                continue
            has_stock = any(l.role == "long_stock" for l in st.legs)
            for leg in st.legs:
                if leg.role != short_role or (leg.allocated_qty or 0) >= 0:
                    continue
                o = by_pid.get(leg.position_id)
                if o is None:
                    continue
                need = int(abs(leg.allocated_qty)) * _opt_mult(o)
                need = _draw(pool, o.expiry, need)
                if need > 0 and has_stock:
                    take = min(need, stock_avail)
                    stock_avail -= take
        # 2) the remaining capacity covers naked marks (whole contracts)
        for st in structures:
            if st.contention_group:
                continue
            legs: list[StructureLeg] = []
            for leg in st.legs:
                if leg.role != naked_role or (leg.allocated_qty or 0) >= 0:
                    legs.append(leg)
                    continue
                o = by_pid.get(leg.position_id)
                mult = _opt_mult(o) if o is not None else _MULT
                have = int(abs(leg.allocated_qty))
                need = have * mult
                need = _draw(pool, getattr(o, "expiry", None), need)
                if need > 0 and stock_avail > 0:
                    take = min(need, stock_avail)
                    stock_avail -= take
                    need -= take
                keep = min(have, -(-int(need) // mult)) if need > 0 else 0
                if keep > 0:
                    legs.append(StructureLeg(leg.position_id, -keep, naked_role))
                if have - keep > 0:
                    legs.append(StructureLeg(leg.position_id, -(have - keep), short_role))
            st.legs = legs


# ---------------------------------------------------------------------------
# Per-underlying detection (one account-scoped allocation pass)
# ---------------------------------------------------------------------------
def _detect_for_underlying(
    account: str,
    underlying: str,
    stock_legs: list[Position],
    options: list[Position],
    trades_df,
) -> list[Structure]:
    """Allocate signed-quantity slices in priority order: the exact-composite
    tier (box → iron butterfly → iron condor → condor → double diagonal →
    jelly roll → conversion / reversal → butterfly → jade lizard → ladder,
    matching only what the stock-cover reservation leaves) → the covered-call-
    vs-vertical contention carve-out → collar (matched expiry) → covered call
    (+ naked-excess slices) → married put → ratio spread / backspread (unequal
    only) → vertical (1:1, same expiry) → straddle / strangle → synthetic
    long / short → risk reversal → calendar → diagonal → covered / cash-secured
    put → the naked short-call sweep → the deferred residual-long slice. Legs
    consumed by a higher-priority structure are not re-used, which
    deterministically resolves the containment subsumptions (a box never
    fragments into its verticals); genuinely symmetric contention (a leg that
    fits two equally-valid readings) is reserved for ranked alternatives."""
    out: list[Structure] = []
    demotable: list[Structure] = []
    cc_ran = False   # the covered-call pass gates the deferred residual slice

    # Remaining signed quantities to allocate.
    stock_rem = sum((_num(s.quantity) or 0.0) for s in stock_legs)
    stock_total = stock_rem                     # original holding, for the naked reconcile
    stock_pid = stock_legs[0].position_id if stock_legs else None
    opt_rem: dict[str, float] = {o.position_id: (_num(o.quantity) or 0.0) for o in options}
    by_pid = {o.position_id: o for o in options}

    def calls_short():
        return [o for o in options if o.right == "CALL" and opt_rem[o.position_id] < 0]

    def calls_long():
        return [o for o in options if o.right == "CALL" and opt_rem[o.position_id] > 0]

    def puts_short():
        return [o for o in options if o.right == "PUT" and opt_rem[o.position_id] < 0]

    def puts_long():
        return [o for o in options if o.right == "PUT" and opt_rem[o.position_id] > 0]

    # ---- A) The exact-composite tier — before every stock-anchored pass -------
    # Fully-determined multi-leg matches (equal qty, coherent strike ladder,
    # matched expiries) whose coincidental assembly is implausible; run later
    # they demonstrably fragment (a box into two verticals, a conversion into a
    # collar). Stock coverage stays senior via the reservation: composites match
    # only the short-call remainder the stock cannot cover. The reservation is
    # recomputed per spec because the conversion pass consumes stock.
    def _reserve():
        return _reserve_stock_cover(stock_rem, options, opt_rem)

    _run_specs(_TIER_A_PRE, account, underlying, options, opt_rem, trades_df,
               out, demotable, tier_a=True, reserve_fn=_reserve)
    stock_rem = _detect_conversion_reversal(account, underlying, stock_pid, stock_rem,
                                            options, opt_rem, trades_df, out, demotable)
    _run_specs(_TIER_A_POST, account, underlying, options, opt_rem, trades_df,
               out, demotable, tier_a=True, reserve_fn=_reserve)

    # ---- 0) Contention: covered call vs vertical on a shared short call --------
    # A short call that long stock can fully cover AND that also pairs with a long
    # call into a vertical (same expiry, equal size, different strike) admits two
    # equally-valid readings. With no co-opening trade to favour the vertical and an
    # exact cover for the covered call, no signal breaks the tie: emit BOTH as ranked
    # alternatives sharing a contention group, banded low-ambiguous, nothing
    # auto-selected. The shared short call is the contended leg; the legs are consumed
    # here so the priority pass below does not silently resolve the tie.
    if stock_pid is not None and stock_rem > 0:
        for sc in list(calls_short()):
            for lc in list(calls_long()):
                if sc.expiry is None or sc.expiry != lc.expiry or sc.strike == lc.strike:
                    continue
                if _opt_mult(sc) != _opt_mult(lc):
                    continue  # mixed contract sizes are not an equal-size two-reading tie
                qty = int(abs(opt_rem[sc.position_id]))
                if qty == 0 or qty != int(abs(opt_rem[lc.position_id])):
                    continue
                shares = qty * _opt_mult(sc)
                if stock_rem < shares:
                    continue  # stock can't fully cover → not the clean two-reading case
                if _co_opened([sc.option_contract_key, lc.option_contract_key], trades_df):
                    continue  # a co-opened vertical is corroborated → let the normal pass form it
                group = _sid(account, underlying, "contention", [stock_pid, lc.position_id, sc.position_id])
                # Reading A — covered call: the stock covers the short call (long call is the residual).
                out.append(Structure(
                    structure_id=_sid(account, underlying, COVERED_CALL, [stock_pid, sc.position_id], "altA"),
                    account=account, underlying=underlying, type=COVERED_CALL,
                    confidence_band=LOW_AMBIGUOUS, status="proposed",
                    legs=[StructureLeg(stock_pid, shares, "long_stock"),
                          StructureLeg(sc.position_id, -qty, "short_call")],
                    rationale_trace=_trace(
                        {"long_shares": {"value": shares, "source": "EXTRACT:quantity"},
                         "short_call_strike": {"value": sc.strike, "source": "EXTRACT:option_strike"},
                         "residual": {"value": f"long {qty} {lc.strike:g}C", "source": "computed:unclaimed leg"}},
                        f"{shares} sh cover the short {sc.strike:g}C; the long {lc.strike:g}C is the residual",
                        "covered call (reading A of 2 — long call is the residual)"),
                    source="detector:contention", contention_group=group))
                # Reading B — vertical: the two calls are the spread (the stock is the residual).
                debit = lc.strike < sc.strike
                out.append(Structure(
                    structure_id=_sid(account, underlying, VERTICAL, [lc.position_id, sc.position_id], "altB"),
                    account=account, underlying=underlying, type=VERTICAL,
                    confidence_band=LOW_AMBIGUOUS, status="proposed",
                    legs=[StructureLeg(lc.position_id, qty, "long_call"),
                          StructureLeg(sc.position_id, -qty, "short_call")],
                    rationale_trace=_trace(
                        {"long_strike": {"value": lc.strike, "source": "EXTRACT:option_strike"},
                         "short_strike": {"value": sc.strike, "source": "EXTRACT:option_strike"},
                         "qty": {"value": qty, "source": "EXTRACT:quantity"},
                         "residual": {"value": f"long {int(shares)} sh", "source": "computed:unclaimed leg"}},
                        f"{qty}x {underlying} call vertical {lc.strike:g}/{sc.strike:g}; the {int(shares)} shares are the residual",
                        f"call vertical ({'debit' if debit else 'credit'}) (reading B of 2 — stock is the residual)"),
                    source="detector:contention", contention_group=group))
                stock_rem -= shares
                opt_rem[sc.position_id] = 0.0
                opt_rem[lc.position_id] = 0.0

    # ---- 1) Collar: long stock + short call + long put, MATCHED expiry --------
    if stock_pid is not None and stock_rem > 0:
        for sc in sorted(calls_short(), key=lambda o: (str(o.expiry), o.strike or 0)):
            for lp in sorted(puts_long(), key=lambda o: (str(o.expiry), o.strike or 0)):
                if sc.expiry is None or sc.expiry != lp.expiry:
                    continue  # mismatched expiry is NOT a collar (it's a CC + standalone hedge)
                mult = _opt_mult(sc)
                if mult != _opt_mult(lp):
                    continue  # cap and floor must hedge the same shares per contract
                lots = min(int(stock_rem // mult), int(abs(opt_rem[sc.position_id])), int(opt_rem[lp.position_id]))
                if lots <= 0:
                    continue
                shares = lots * mult
                out.append(Structure(
                    structure_id=_sid(account, underlying, COLLAR, [stock_pid, sc.position_id, lp.position_id]),
                    account=account, underlying=underlying, type=COLLAR, confidence_band=HIGH,
                    status="proposed",
                    legs=[StructureLeg(stock_pid, shares, "long_stock"),
                          StructureLeg(sc.position_id, -lots, "short_call"),
                          StructureLeg(lp.position_id, lots, "long_put")],
                    rationale_trace=_trace(
                        {"long_shares": {"value": shares, "source": "EXTRACT:quantity"},
                         "short_call": {"value": lots, "source": "EXTRACT:quantity"},
                         "long_put": {"value": lots, "source": "EXTRACT:quantity"},
                         "call_expiry": {"value": str(sc.expiry), "source": "EXTRACT:option_expiration"},
                         "put_expiry": {"value": str(lp.expiry), "source": "EXTRACT:option_expiration"}},
                        f"{shares} sh + short {lots} {underlying} {sc.strike:g}C + long {lots} {lp.strike:g}P, matched expiry {sc.expiry}",
                        "collar (stock capped by the short call, floored by the long put)"),
                    source="detector:collar"))
                stock_rem -= shares
                opt_rem[sc.position_id] += lots
                opt_rem[lp.position_id] -= lots

    # ---- 2) Covered call: long stock + short call(s); split partial / over-write
    short_calls = sorted(calls_short(), key=lambda o: (str(o.expiry), o.strike or 0))
    if stock_pid is not None and stock_rem > 0 and short_calls:
        short_call_shares = sum(abs(opt_rem[o.position_id]) * _opt_mult(o) for o in short_calls)

        # Greedy per-leg cover in (expiry, strike) order, each contract claiming
        # its OWN shares-per-contract — for the all-standard book this reproduces
        # the single-multiplier arithmetic take for take.
        covered_legs: list[StructureLeg] = [StructureLeg(stock_pid, 0, "long_stock")]
        excess_legs: list[StructureLeg] = []
        stock_avail = stock_rem
        covered_shares = 0
        covered_contracts = 0
        for o in short_calls:
            mult = _opt_mult(o)
            have = int(abs(opt_rem[o.position_id]))
            take = min(have, int(stock_avail // mult))
            if take > 0:
                covered_legs.append(StructureLeg(o.position_id, -take, "short_call"))
                stock_avail -= take * mult
                covered_shares += take * mult
                covered_contracts += take
            leftover = have - take
            if leftover > 0:
                excess_legs.append(StructureLeg(o.position_id, -leftover, "naked_excess_short_call"))
            opt_rem[o.position_id] = 0.0  # all short calls accounted for (covered or naked-excess)
        covered_legs[0] = StructureLeg(stock_pid, covered_shares, "long_stock")

        full = covered_shares == stock_rem and not excess_legs
        out.append(Structure(
            structure_id=_sid(account, underlying, COVERED_CALL, [l.position_id for l in covered_legs]),
            account=account, underlying=underlying, type=COVERED_CALL, confidence_band=HIGH,
            status="proposed", legs=covered_legs,
            rationale_trace=_trace(
                {"long_shares": {"value": stock_rem, "source": "EXTRACT:quantity"},
                 "short_call_shares": {"value": short_call_shares, "source": "computed:|qty|xmultiplier"},
                 "covered_shares": {"value": covered_shares, "source": "computed:min(long,short)"}},
                f"{covered_contracts} short call(s) cover {covered_shares} of {int(stock_rem)} {underlying} shares",
                f"covered call ({'full' if full else 'partial cover'})"),
            source="detector:covered_call"))

        stock_after = stock_rem - covered_shares
        cc_ran = True   # the residual-long slice is emitted at the end, AFTER
        # the married-put pass has had the chance to claim the leftover stock.
        if excess_legs:
            out.append(Structure(
                structure_id=_sid(account, underlying, NAKED_EXCESS_SHORT_CALL,
                                  [l.position_id for l in excess_legs], suffix="excess"),
                account=account, underlying=underlying, type=NAKED_EXCESS_SHORT_CALL, confidence_band=HIGH,
                status="proposed", legs=excess_legs,
                rationale_trace=_trace(
                    {"naked_excess_contracts": {"value": sum(int(abs(l.allocated_qty)) for l in excess_legs),
                                                "source": "computed:short-covered"}},
                    f"short calls exceed the {int(stock_rem)} shares held → the excess is uncovered",
                    "naked-excess short call (over-write beyond stock held)"),
                source="detector:covered_call"))
        stock_rem = max(0.0, stock_after)

    # ---- 2b) Married put: the stock the covered-call pass left + long puts ----
    stock_rem = _detect_married_put(account, underlying, stock_pid, stock_rem,
                                    options, opt_rem, trades_df, out, demotable)

    # ---- 2c) Ratio spread / backspread — ABOVE vertical, guarded to unequal ---
    # A 1x2 must not fragment into a vertical + a naked remainder; a true 1:1
    # fails the unequal-quantity guard and falls through to the vertical pass.
    _detect_ratio_backspread(account, underlying, options, opt_rem, trades_df, out)

    # ---- 3) Vertical: one long + one short, same right + expiry, EQUAL qty -----
    for right in ("CALL", "PUT"):
        by_expiry: dict = defaultdict(lambda: {"long": [], "short": []})
        for o in options:
            if o.right != right or opt_rem[o.position_id] == 0:
                continue
            by_expiry[o.expiry]["long" if opt_rem[o.position_id] > 0 else "short"].append(o)
        for expiry, side in by_expiry.items():
            # Clean 1:1 only — a single long and single short of equal size, different strikes.
            # Unequal sizes (a ratio) or >2 legs are deliberately left ungrouped.
            if len(side["long"]) == 1 and len(side["short"]) == 1:
                lo, sh = side["long"][0], side["short"][0]
                if abs(opt_rem[lo.position_id]) == abs(opt_rem[sh.position_id]) and lo.strike != sh.strike:
                    qty = int(abs(opt_rem[lo.position_id]))
                    keys = [lo.option_contract_key, sh.option_contract_key]
                    band = HIGH if _co_opened(keys, trades_df) else MEDIUM
                    debit = (right == "CALL" and lo.strike < sh.strike) or (right == "PUT" and lo.strike > sh.strike)
                    out.append(Structure(
                        structure_id=_sid(account, underlying, VERTICAL, [lo.position_id, sh.position_id]),
                        account=account, underlying=underlying, type=VERTICAL, confidence_band=band,
                        status="proposed",
                        legs=[StructureLeg(lo.position_id, opt_rem[lo.position_id], "long_" + right.lower()),
                              StructureLeg(sh.position_id, opt_rem[sh.position_id], "short_" + right.lower())],
                        rationale_trace=_trace(
                            {"long_strike": {"value": lo.strike, "source": "EXTRACT:option_strike"},
                             "short_strike": {"value": sh.strike, "source": "EXTRACT:option_strike"},
                             "qty": {"value": qty, "source": "EXTRACT:quantity"},
                             "expiry": {"value": str(expiry), "source": "EXTRACT:option_expiration"},
                             "co_opened": {"value": band == HIGH, "source": "computed:trade corroboration"}},
                            f"{qty}x {underlying} {right.lower()} spread {lo.strike:g}/{sh.strike:g} exp {expiry}",
                            f"{right.lower()} vertical ({'debit' if debit else 'credit'})"),
                        source="detector:vertical"))
                    opt_rem[lo.position_id] = 0.0
                    opt_rem[sh.position_id] = 0.0

    # ---- 4) Straddle / strangle: call + put, same expiry + same sign ----------
    by_exp_sign: dict = defaultdict(lambda: {"call": None, "put": None})
    for o in options:
        if opt_rem[o.position_id] == 0:
            continue
        sign = "long" if opt_rem[o.position_id] > 0 else "short"
        slot = by_exp_sign[(o.expiry, sign)]
        if o.right == "CALL" and slot["call"] is None:
            slot["call"] = o
        elif o.right == "PUT" and slot["put"] is None:
            slot["put"] = o
    for (expiry, sign), slot in by_exp_sign.items():
        c, p = slot["call"], slot["put"]
        if c is None or p is None:
            continue
        qty = min(int(abs(opt_rem[c.position_id])), int(abs(opt_rem[p.position_id])))
        if qty <= 0:
            continue
        same_strike = c.strike == p.strike
        typ = STRADDLE if same_strike else STRANGLE
        keys = [c.option_contract_key, p.option_contract_key]
        band = HIGH if _co_opened(keys, trades_df) else MEDIUM
        cq = qty if sign == "long" else -qty
        out.append(Structure(
            structure_id=_sid(account, underlying, typ, [c.position_id, p.position_id]),
            account=account, underlying=underlying, type=typ, confidence_band=band,
            status="proposed",
            legs=[StructureLeg(c.position_id, cq, sign + "_call"),
                  StructureLeg(p.position_id, cq, sign + "_put")],
            rationale_trace=_trace(
                {"call_strike": {"value": c.strike, "source": "EXTRACT:option_strike"},
                 "put_strike": {"value": p.strike, "source": "EXTRACT:option_strike"},
                 "qty": {"value": qty, "source": "EXTRACT:quantity"},
                 "expiry": {"value": str(expiry), "source": "EXTRACT:option_expiration"},
                 "co_opened": {"value": band == HIGH, "source": "computed:trade corroboration"}},
                f"{sign} {qty}x {underlying} {c.strike:g}C + {p.strike:g}P exp {expiry}",
                f"{sign} {typ}"),
            source=f"detector:{typ}"))
        opt_rem[c.position_id] -= cq
        opt_rem[p.position_id] -= cq

    # ---- 4b) Synthetic stock, then risk reversal (same expiry, opposite-sign
    # call/put pair) — AFTER straddle/strangle so the existing suite's claims on
    # mixed books are preserved; a clean combo has no same-sign pair to lose.
    _run_specs(_PAIR_SPECS, account, underlying, options, opt_rem, trades_df,
               out, demotable, tier_a=False)

    # ---- 4c) Cross-expiry pairs: calendar (same strike) then diagonal ---------
    # Below every same-expiry pass (the tighter relation wins), above the short-
    # put and naked sweeps (a put calendar's short front leg is not an income
    # put; a diagonal's short call must not reach the sweep). The PMCC reading
    # of a call diagonal is a render-time qualifier, not a type (see module doc).
    _run_specs(_XEXP_SPECS, account, underlying, options, opt_rem, trades_df,
               out, demotable, tier_a=False)

    # ---- 5) Covered / cash-secured put: short put(s) NOT consumed above --------
    # Runs last so a short put that is the leg of a vertical or straddle is
    # claimed there first; only a *lone* short put is an income-overlay put.
    short_puts = sorted(puts_short(), key=lambda o: (str(o.expiry), o.strike or 0))
    if short_puts:
        is_covered = stock_rem < 0  # short stock backing the short put
        typ = COVERED_PUT if is_covered else CASH_SECURED_PUT
        contracts = sum(int(abs(opt_rem[o.position_id])) for o in short_puts)
        legs = [StructureLeg(o.position_id, opt_rem[o.position_id], "short_put") for o in short_puts]
        put_sweep = Structure(
            structure_id=_sid(account, underlying, typ, [l.position_id for l in legs]),
            account=account, underlying=underlying, type=typ, confidence_band=HIGH,
            status="proposed", legs=legs,
            rationale_trace=_trace(
                {"short_put_contracts": {"value": contracts, "source": "EXTRACT:quantity"},
                 "short_stock": {"value": is_covered, "source": "EXTRACT:quantity"}},
                f"short {contracts} {underlying} put(s)"
                + (" against short stock" if is_covered else " (income posture; no stock leg)"),
                typ),
            source=f"detector:{typ}")
        out.append(put_sweep)
        if typ in _SIBLING_DEMOTABLE:
            # A sweep has no co-opening corroboration, so its HIGH is structural
            # only — pass 9 demotes it while unexplained sibling legs remain.
            demotable.append(put_sweep)
        for o in short_puts:
            opt_rem[o.position_id] = 0.0

    # ---- 6) Naked short call: short calls no pass above could claim ------------
    # The uncovered sweep of last resort. No coverable stock reaches here: when
    # long shares exist the covered-call pass consumed every short call (covered
    # or excess), so what arrives is a short call on a stock-less name or the
    # leftover a collar / contention left behind. Unclaimed LONG calls still
    # offset it — a ratio spread's short side has bounded upside, not naked — but
    # only same-or-later expiry counts (an earlier-expiry long call dies first
    # and covers nothing beyond its own life). Only the remainder is naked.
    # Reuses NAKED_EXCESS_SHORT_CALL so every downstream surface (P16/P19, the
    # By-Structure grid, tier-2 pricing, the exposure rollup, resolve/rederive)
    # works unchanged.
    naked_shorts = sorted(calls_short(), key=lambda o: (str(o.expiry), o.strike or 0))
    if naked_shorts:
        lc_pool = [[o.expiry, opt_rem[o.position_id] * _opt_mult(o)] for o in calls_long()]
        naked_legs: list[StructureLeg] = []
        for o in naked_shorts:
            mult = _opt_mult(o)
            have = int(abs(opt_rem[o.position_id]))
            need = have * mult
            for slot in lc_pool:
                if need <= 0:
                    break
                if o.expiry is not None and slot[0] is not None and slot[0] < o.expiry:
                    continue
                take = min(need, slot[1])
                slot[1] -= take
                need -= take
            # A contract counts covered only when fully covered — the remainder
            # rounds UP to whole naked contracts (a partially-covered contract
            # cannot deliver).
            naked = min(have, -(-int(need) // mult)) if need > 0 else 0
            opt_rem[o.position_id] = 0.0
            if naked > 0:
                naked_legs.append(StructureLeg(o.position_id, -naked, "naked_excess_short_call"))
        if naked_legs:
            contracts = sum(int(abs(l.allocated_qty)) for l in naked_legs)
            no_stock = not stock_legs
            out.append(Structure(
                structure_id=_sid(account, underlying, NAKED_EXCESS_SHORT_CALL,
                                  [l.position_id for l in naked_legs], suffix="naked"),
                account=account, underlying=underlying, type=NAKED_EXCESS_SHORT_CALL,
                confidence_band=HIGH, status="proposed", legs=naked_legs,
                rationale_trace=_trace(
                    {"naked_excess_contracts": {"value": contracts,
                                                "source": "computed:short calls − available cover"},
                     "stock_held": {"value": not no_stock, "source": "EXTRACT:quantity"}},
                    f"short {contracts} {underlying} call(s) with "
                    + ("no covering shares held on the name"
                       if no_stock else "no shares left to cover them"),
                    "naked short call (no cover on the name)"),
                source="detector:naked_call"))

    # ---- 7) Residual long stock — deferred from the covered-call pass so the
    # married-put pass could claim the leftover first; same slice + structure_id
    # as the original inline emission when no married put forms.
    if cc_ran and stock_pid is not None and stock_rem > 0:
        out.append(Structure(
            structure_id=_sid(account, underlying, RESIDUAL_LONG, [stock_pid], suffix="residual"),
            account=account, underlying=underlying, type=RESIDUAL_LONG, confidence_band=MEDIUM,
            status="proposed", legs=[StructureLeg(stock_pid, stock_rem, "residual_long")],
            rationale_trace=_trace(
                {"residual_shares": {"value": stock_rem, "source": "computed:long-covered"}},
                f"{int(stock_rem)} {underlying} shares uncovered after the over-write",
                "uncovered long stock (residual of a partial covered call)"),
            source="detector:covered_call"))

    # ---- 8) Naked-role reconcile — no decomposition overstates nakedness ------
    _reconcile_naked_roles(out, stock_total, options)

    # ---- 9) Sibling discipline — never over-claim on a partial read: while
    # unexplained option legs remain on the name, a structural-HIGH combination
    # (not corroborated by co-opening trades) reads MEDIUM instead.
    if any(abs(v) > 1e-9 for v in opt_rem.values()):
        for st in demotable:
            if st.confidence_band == HIGH:
                st.confidence_band = MEDIUM

    return out


# ---------------------------------------------------------------------------
# Account- and portfolio-level entry points
# ---------------------------------------------------------------------------
def detect_account_structures(account_state) -> list[Structure]:
    """Detect the core structures in one account. Account-scoped: never groups
    legs across accounts."""
    account = account_state.account
    positions = list(getattr(account_state, "positions", []) or [])
    trades_by_underlying = getattr(account_state, "trades_by_underlying", {}) or {}

    stocks_by_symbol: dict[str, list[Position]] = defaultdict(list)
    opts_by_underlying: dict[str, list[Position]] = defaultdict(list)
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and _num(p.quantity):
            stocks_by_symbol[p.symbol].append(p)
        elif p.asset_class == "option" and p.underlying_symbol and _num(p.quantity):
            opts_by_underlying[p.underlying_symbol].append(p)

    structures: list[Structure] = []
    for underlying in sorted(set(stocks_by_symbol) | set(opts_by_underlying)):
        structures.extend(_detect_for_underlying(
            account, underlying,
            stocks_by_symbol.get(underlying, []),
            opts_by_underlying.get(underlying, []),
            trades_by_underlying.get(underlying),
        ))
    return structures


def run_structure_detection(state) -> None:
    """Detect structures for every account and attach them to each AccountState,
    applying any stored confirm/override resolutions. Called in the load path after
    the insight engine; mutates state in place. The pure detector
    (``detect_account_structures``) does no I/O; the stored resolutions are read
    here, in the load path."""
    from pm.store.structure_store import all_resolutions, apply_resolutions
    resolutions = all_resolutions()
    for account_state in state.accounts.values():
        structures = detect_account_structures(account_state)
        apply_resolutions(account_state.account, structures, resolutions)
        account_state.structures = structures


# ---------------------------------------------------------------------------
# Allocation reconciliation — the shared signed-slice ledger
# ---------------------------------------------------------------------------
# How much of each position is claimed by the recognised structures, and how much
# is left standalone. This is the one place that knows the conservation rule
# (every position's allocated slices + its unallocated remainder == its full
# quantity), so every surface that splits a book into structured vs standalone —
# the By-Structure holdings view and the portfolio exposure rollup — reads it
# instead of re-deriving the arithmetic. Pure and duck-typed: it reads only
# ``account_state.structures`` and ``account_state.positions``.

def reconcile_allocations(account_state) -> dict:
    """Signed-slice invariant: for every position, the sum of its allocated slices
    across all non-rejected structures plus its unallocated remainder must equal
    the position's full quantity (no position dropped, none double-counted). The
    remainder is what the standalone / unstructured views carry.

    Returns ``{position_id: {"allocated", "quantity", "ok", "remainder"}}``.
    """
    by_id = {p.position_id: p for p in account_state.positions}
    allocated: dict[str, float] = {}

    def _add(pid: str, qty) -> None:
        try:
            allocated[pid] = allocated.get(pid, 0.0) + float(qty)
        except (TypeError, ValueError):
            pass

    # Contention groups are mutually-exclusive readings of the SAME legs — a
    # contended leg appears in every alternative, so it must count once, not
    # once per reading. Collapse each group to a single allocation per position.
    groups: dict[str, list] = {}
    for st in getattr(account_state, "structures", []) or []:
        if st.status == "rejected":
            continue
        if st.contention_group:
            groups.setdefault(st.contention_group, []).append(st)
        else:
            for leg in st.legs:
                _add(leg.position_id, leg.allocated_qty)
    for alts in groups.values():
        per_pos: dict[str, float] = {}
        for alt in alts:
            for leg in alt.legs:
                cur = per_pos.get(leg.position_id)
                try:
                    val = float(leg.allocated_qty)
                except (TypeError, ValueError):
                    continue
                if cur is None or abs(val) > abs(cur):
                    per_pos[leg.position_id] = val
        for pid, qty in per_pos.items():
            _add(pid, qty)

    out: dict[str, dict] = {}
    for pid, pos in by_id.items():
        alloc = allocated.get(pid, 0.0)
        qty = pos.quantity
        try:
            full = float(qty) if qty is not None else None
        except (TypeError, ValueError):
            full = None
        if full is None:
            ok, remainder = True, None          # can't reconcile a None-qty position; not a drop
        else:
            remainder = full - alloc            # carried by a standalone / unstructured row when non-zero
            # The bug this catches: a position allocated beyond its own size, or
            # with the wrong sign (double-counted across structures).
            ok = (abs(alloc) <= abs(full) + 1e-6
                  and (alloc == 0 or (alloc > 0) == (full > 0)))
        out[pid] = {"allocated": alloc, "quantity": full, "ok": ok, "remainder": remainder}
    return out
