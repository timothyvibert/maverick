"""Structure-aware management fires — the alerts the recognised structures unlock.

Runs UPSTREAM in the load path, after structure detection. Reads the already-built
positions, the BBG underlying snapshot (spot), the per-contract option mark from
holdings, an assumed short rate, and an as-of date — the same inputs the engine and
P7 already consume — and appends ``Fire`` objects to each ``AccountState``. The UI
reads them; nothing recomputes downstream.

Only desk-grade triggers fire. Each fire carries structure context. The carry input
is source-aware: a live treasury-curve rate is stated plainly, a fallback scalar rate
is flagged as estimated. Contested mechanical conventions (the 21-DTE window, "50% of
max profit", "75–85% of width") are deliberately NOT triggers here — they belong on a
card as context only.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pm.core.bloomberg_client import pick_rate_for_dte
from pm.insight import structures as S
from pm.insight.patterns import Fire

T1, T2 = 1, 2

# pattern_id -> (display name, tier)
_META = {
    "P16": ("Coverage breach", T1),
    "P17": ("Deep-ITM short put — carry assignment risk", T2),
    "P18": ("Covered structure at its cap", T2),
    "P19": ("Pin risk into expiry", T1),
    "P20": ("Collar put — monetize", T2),
}
_CONFIRMED = ("confirmed", "edited")
# Structure-fire thresholds. Unlike the P1-P15 PatternConfig dials (now editable via
# settings_store + the threshold catalog), these stay module constants for now
# — the editable-threshold feature deliberately scoped its surface to PatternConfig. Future injection
# point: a `StructureFireConfig` threaded through run_structure_fires /
# rederive_structure_fires (and the resolve_structure re-derive in state_access) would
# make these editable too — not built this increment.
_PIN_NEAR = 0.02            # |spot-strike|/spot within 2% = near the money
_PIN_DTE = 1               # expiry-day (or last day)
_PIN_DTE_MIN = 0           # never on an already-expired leg (a stale extract row)
# Every structure type carrying a short leg that can pin: the original covered
# call / collar / vertical plus the canonical pin cases the first cut omitted —
# short straddles/strangles, covered / cash-secured short puts, and the naked
# short call itself (the position P16 declares most dangerous).
_PIN_TYPES = (S.COVERED_CALL, S.COLLAR, S.VERTICAL, S.STRADDLE, S.STRANGLE,
              S.COVERED_PUT, S.CASH_SECURED_PUT, S.NAKED_EXCESS_SHORT_CALL)
_PIN_SHORT_ROLES = ("short_call", "short_put", "naked_excess_short_call")
_MONETIZE_EXTRINSIC_FRAC = 0.05   # put extrinsic <= 5% of intrinsic (owner threshold)


# ---------------------------------------------------------------------------
# Market-state helpers (read the same sources the engine/P7 use)
# ---------------------------------------------------------------------------
def _spot(account_state, position) -> Optional[float]:
    snap = getattr(account_state, "snapshot", None)
    df = getattr(snap, "underlyings", None)
    bbg = getattr(position, "underlying_bbg_ticker", None) or getattr(position, "bbg_ticker", None)
    if df is None or getattr(df, "empty", True) or not bbg:
        return None
    if bbg not in df.index or "PX_LAST" not in df.columns:
        return None
    try:
        v = df.loc[bbg, "PX_LAST"]
        return float(v) if (v is not None and v == v) else None  # NaN guard
    except (TypeError, ValueError, KeyError):
        return None


def _mark(position) -> Optional[float]:
    """Current per-share option mark from holdings (mark = |MV| / (|qty| × 100))."""
    mv, qty = position.market_value, position.quantity
    mult = position.multiplier or 100
    try:
        if not mv or not qty:
            return None
        return abs(float(mv)) / abs(float(qty)) / mult
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dte(position, as_of: date) -> Optional[int]:
    if position.expiry is None:
        return None
    try:
        return (position.expiry - as_of).days
    except Exception:
        return None


def _legs_with_role(structure, role: str):
    return [l for l in structure.legs if l.role == role]


def _trace(inputs: dict, computation: str, result: str, thresholds: Optional[dict] = None) -> dict:
    return {"inputs": inputs, "computation": computation,
            "thresholds": thresholds or {}, "result": result}


def _money(v) -> str:
    return "—" if v is None else f"${v:,.2f}"


def _struct_word(typ: str) -> str:
    return {S.COVERED_CALL: "covered call", S.COLLAR: "collar", S.VERTICAL: "vertical",
            S.COVERED_PUT: "covered put", S.CASH_SECURED_PUT: "cash-secured put",
            S.STRADDLE: "straddle", S.STRANGLE: "strangle",
            S.NAKED_EXCESS_SHORT_CALL: "naked short call"}.get(typ, typ)


def _group_representative(account_state, st) -> bool:
    """One voice per contention group: ranked alternatives share the same short
    leg, so an ungated per-structure fire (P19) would flag the same pin event once
    per reading — inflating the T1 tier counts. The group speaks through ONE
    member: the confirmed one if any, else the lowest-structure_id non-rejected
    member, else the lowest overall. Pure function of the account's current
    structures, so the load pass and the confirm/reject re-derive agree."""
    group = getattr(st, "contention_group", None)
    if not group:
        return True
    members = [s for s in account_state.structures
               if getattr(s, "contention_group", None) == group]
    pool = ([s for s in members if s.status in _CONFIRMED]
            or [s for s in members if s.status != "rejected"]
            or members)
    return st.structure_id == min(pool, key=lambda s: s.structure_id).structure_id


def _fire(pattern_id, account, position, underlying, label, rationale, trace, fired_at) -> Fire:
    name, tier = _META[pattern_id]
    return Fire(pattern_id=pattern_id, pattern_name=name, tier=tier, account=account,
                position_id=position.position_id, underlying=underlying,
                asset_class=position.asset_class, label=label, rationale=rationale,
                trace=trace, fired_at=fired_at)


# ---------------------------------------------------------------------------
# Per-structure fires
# ---------------------------------------------------------------------------
def _fires_for_structure(account_state, st, by_id, rate_curve, rate_fallback, as_of, fired_at) -> list[Fire]:
    account, u = st.account, st.underlying
    confirmed = st.status in _CONFIRMED
    out: list[Fire] = []

    # --- P16 coverage breach (T1, holdings-only) ---------------------------
    # One fire per naked LEG position (blotter rows are per (account, position);
    # a single fire on legs[0] would leave every other naked leg alert-less).
    if st.type == S.NAKED_EXCESS_SHORT_CALL:
        outright_naked = st.source == "detector:naked_call"
        for leg in st.legs:
            pos = by_id.get(leg.position_id)
            if pos is None:
                continue
            n = int(abs(leg.allocated_qty))
            if n <= 0:
                continue
            if outright_naked:
                label = f"{u} naked short call — {n} uncovered contract(s)"
                rationale = (
                    f"Short {u} call(s) with no cover on the name: {n} contract(s) are "
                    f"outright naked — assignment delivers short stock, unbounded upside "
                    f"exposure the client likely did not intend. Cover with stock, pair "
                    f"with a longer-dated long call, or close.")
                computation = "short call with no available cover on the name → naked"
            else:
                label = f"{u} over-write — {n} naked-excess short call(s)"
                rationale = (
                    f"Your {u} over-write writes {n} more call(s) than the shares held cover. "
                    f"The excess behaves like a naked short call — assignment there delivers short "
                    f"stock, exposure the client likely did not intend. Bring coverage back to 1:1 "
                    f"or close the excess.")
                computation = "short-call shares > shares held → the excess is uncovered"
            out.append(_fire("P16", account, pos, u, label, rationale,
                _trace({"naked_excess_contracts": {"value": n, "source": "structure:naked_excess_short_call"}},
                       computation,
                       "coverage breach (naked short call)"),
                fired_at))

    # --- P17 carry-driven early exercise on a deep-ITM short put (T2) ------
    if st.type in (S.COVERED_PUT, S.CASH_SECURED_PUT):
        for leg in _legs_with_role(st, "short_put"):
            pos = by_id.get(leg.position_id)
            if pos is None:
                continue
            spot, mark, dte = _spot(account_state, pos), _mark(pos), _dte(pos, as_of)
            strike = float(pos.strike or 0)
            if spot is None or mark is None or dte is None or strike <= 0 or dte <= 0:
                continue
            intrinsic = max(0.0, strike - spot)
            if intrinsic <= 0:                    # not ITM → no early-exercise pressure
                continue
            extrinsic = max(0.0, mark - intrinsic)
            # Rate nearest this leg's DTE from the live treasury curve; else the
            # BBG-off fallback scalar (flagged as an estimate in the copy).
            tenor = pick_rate_for_dte(rate_curve, dte)
            if tenor is not None:
                rate, rate_live, rate_ticker, rate_label = tenor["rate"], True, tenor["ticker"], tenor["label"]
            else:
                rate, rate_live, rate_ticker, rate_label = rate_fallback, False, None, None
            carry = strike * rate * (dte / 365.0)
            if extrinsic < carry:
                if rate_live:
                    rate_phrase = f"the {rate_label} treasury rate, {rate*100:.1f}%"
                    rate_src = f"BBG:{rate_ticker} (USGG curve)"
                else:
                    rate_phrase = f"an estimated {rate*100:.1f}% rate"
                    rate_src = "config:DEFAULT_RISK_FREE_RATE (estimate)"
                out.append(_fire("P17", account, pos, u,
                    f"{u} short put ${strike:g} — carry assignment risk",
                    (f"Deep-ITM short put behind your {u} {_struct_word(st.type)}: remaining time "
                     f"value {_money(extrinsic)} is below the carry on the strike (~{_money(carry)}, "
                     f"based on {rate_phrase}). Carry favours early assignment — "
                     f"plan to roll down/out or accept the stock rather than rely on time decay."),
                    _trace({"spot": {"value": spot, "source": "BBG:PX_LAST"},
                            "put_mark": {"value": mark, "source": "computed:|MV|/(qty×multiplier)"},
                            "strike": {"value": strike, "source": "EXTRACT:option_strike"},
                            "extrinsic": {"value": round(extrinsic, 2), "source": "computed:mark−intrinsic"},
                            "carry": {"value": round(carry, 2), "source": "computed:strike×r×t"},
                            "rate": {"value": rate, "source": rate_src},
                            "leg_expiry": {"value": pos.expiry.isoformat() if pos.expiry else None,
                                           "source": "EXTRACT:option_expiration"}},
                        "ITM short put AND extrinsic (mark − (strike − spot)) < carry (strike × r × DTE/365)",
                        "carry-driven early-exercise risk",
                        {"rate": rate}),
                    fired_at))

    # --- P18 covered call / collar at its structural cap (T2; confirmed) ---
    if st.type in (S.COVERED_CALL, S.COLLAR) and confirmed:
        for leg in _legs_with_role(st, "short_call"):
            pos = by_id.get(leg.position_id)
            if pos is None:
                continue
            spot, strike = _spot(account_state, pos), float(pos.strike or 0)
            if spot is None or strike <= 0:
                continue
            if spot >= strike:
                out.append(_fire("P18", account, pos, u,
                    f"{u} {_struct_word(st.type)} at its cap (${spot:g} ≥ ${strike:g})",
                    (f"Your {u} {_struct_word(st.type)} is at its cap: spot {_money(spot)} is at or "
                     f"through the short-call strike {_money(strike)}. There is no further upside to "
                     f"harvest — the decision is roll the call up/out for a credit vs. let it be "
                     f"assigned (sell the stock at the strike)."),
                    _trace({"spot": {"value": spot, "source": "BBG:PX_LAST"},
                            "short_call_strike": {"value": strike, "source": "EXTRACT:option_strike"},
                            "leg_expiry": {"value": pos.expiry.isoformat() if pos.expiry else None,
                                           "source": "EXTRACT:option_expiration"}},
                        "confirmed covered call/collar AND spot ≥ short-call strike",
                        "at structural cap (roll vs let-assign)"),
                    fired_at))

    # --- P19 pin / partial-ITM into expiry (T1) ---------------------------
    # Every short-legged type pins (incl. straddles/strangles, short puts and the
    # naked short call). Lower DTE bound: an already-expired leg still sitting in
    # the extract must not fire with negative-DTE copy (P17's dte<=0 guard, made
    # explicit here). Contention alternatives share the short leg — only the
    # group's representative speaks, or one pin event counts twice in the T1 tier
    # chips.
    if st.type in _PIN_TYPES and _group_representative(account_state, st):
        for leg in st.legs:
            if leg.allocated_qty >= 0 or leg.role not in _PIN_SHORT_ROLES:
                continue
            pos = by_id.get(leg.position_id)
            if pos is None or pos.asset_class != "option":
                continue
            spot, strike, dte = _spot(account_state, pos), float(pos.strike or 0), _dte(pos, as_of)
            if spot is None or spot <= 0 or strike <= 0 or dte is None:
                continue
            if _PIN_DTE_MIN <= dte <= _PIN_DTE and abs(spot - strike) / spot <= _PIN_NEAR:
                out.append(_fire("P19", account, pos, u,
                    f"{u} {_struct_word(st.type)} — pin risk on the short ${strike:g}",
                    (f"Pin risk: the short {strike:g} leg of your {u} {_struct_word(st.type)} sits "
                     f"{abs(spot - strike) / spot * 100:.1f}% from spot ({_money(spot)}) into expiry "
                     f"({dte} DTE). Near-the-money assignment is a coin-flip you will not learn until "
                     f"after the close — close or roll the leg before the bell to avoid an unhedged "
                     f"stock position over the weekend."),
                    _trace({"spot": {"value": spot, "source": "BBG:PX_LAST"},
                            "short_strike": {"value": strike, "source": "EXTRACT:option_strike"},
                            "dte": {"value": dte, "source": "computed:expiry−as_of"},
                            "leg_expiry": {"value": pos.expiry.isoformat() if pos.expiry else None,
                                           "source": "EXTRACT:option_expiration"}},
                        "short structure leg AND 0 ≤ DTE ≤ 1 AND |spot−strike|/spot ≤ 2%",
                        "pin / partial-ITM conversion risk into expiry",
                        {"near_pct": _PIN_NEAR, "dte_max": _PIN_DTE, "dte_min": _PIN_DTE_MIN}),
                    fired_at))

    # --- P20 collar protective-put monetize-after-drop (T2; confirmed) ----
    if st.type == S.COLLAR and confirmed:
        for leg in _legs_with_role(st, "long_put"):
            pos = by_id.get(leg.position_id)
            if pos is None:
                continue
            spot, mark, strike = _spot(account_state, pos), _mark(pos), float(pos.strike or 0)
            if spot is None or mark is None or strike <= 0:
                continue
            intrinsic = max(0.0, strike - spot)
            if intrinsic <= 0:                    # OTM put → real protection, no fire
                continue
            extrinsic = max(0.0, mark - intrinsic)
            if extrinsic <= _MONETIZE_EXTRINSIC_FRAC * intrinsic:
                out.append(_fire("P20", account, pos, u,
                    f"{u} collar — monetize the deep-ITM put",
                    (f"Your {u} collar's protective put is deep ITM with little insurance left: "
                     f"time value {_money(extrinsic)} is {extrinsic / intrinsic * 100:.1f}% of its "
                     f"{_money(intrinsic)} intrinsic. Consider monetising the put (sell to close) or "
                     f"rolling it down to lock the hedge gain rather than carrying a near-zero-extrinsic "
                     f"insurance leg to expiry."),
                    _trace({"spot": {"value": spot, "source": "BBG:PX_LAST"},
                            "put_mark": {"value": mark, "source": "computed:|MV|/(qty×multiplier)"},
                            "put_strike": {"value": strike, "source": "EXTRACT:option_strike"},
                            "intrinsic": {"value": round(intrinsic, 2), "source": "computed:strike−spot"},
                            "extrinsic": {"value": round(extrinsic, 2), "source": "computed:mark−intrinsic"},
                            "leg_expiry": {"value": pos.expiry.isoformat() if pos.expiry else None,
                                           "source": "EXTRACT:option_expiration"}},
                        "confirmed collar AND long put ITM AND extrinsic ≤ 5% of intrinsic",
                        "monetize / re-strike the protective put",
                        {"extrinsic_frac_max": _MONETIZE_EXTRINSIC_FRAC}),
                    fired_at))

    # Tag each fire with its originating structure so a later confirm/reject can
    # add or remove exactly this structure's fires (st is the structure in scope).
    for f in out:
        f.structure_id = st.structure_id
    return out


# ---------------------------------------------------------------------------
# Structure context on existing leg fires (confirmed structures only)
# ---------------------------------------------------------------------------
_ROLE_ACTION = {
    "short_call": "roll up/out for a credit or accept assignment",
    "long_put": "your downside floor",
    "short_put": "roll down/out for a credit or accept the stock",
    "long_stock": "the covered shares",
    "long_call": "the long leg",
}


def attach_structure_context(account_state) -> None:
    """Annotate engine leg fires (every non-structure pattern) whose position is a leg of a CONFIRMED structure,
    so the framing isn't wrong (e.g. P1 'close for margin' on a covered-call short call
    gains 'short leg of your covered call — roll or accept assignment'). The ex-div fire
    (P7) carries its own source-aware dividend copy, so it needs no dividend annotation
    here — only the structure-leg context below.

    Idempotent: each leg fire keeps a clean base rationale (captured the first time it is
    seen), and every run rebuilds from that base. Re-running after a confirm/reject never
    doubles the annotation, and a leg that leaves a confirmed structure reverts to its
    base rationale."""
    leg_ctx: dict[str, tuple] = {}
    for st in account_state.structures:
        if st.status not in _CONFIRMED:
            continue
        if st.type not in (S.COVERED_CALL, S.COLLAR, S.COVERED_PUT, S.CASH_SECURED_PUT):
            continue
        for leg in st.legs:
            leg_ctx[leg.position_id] = (st.type, leg.role, st.underlying)
    for fire in account_state.fires:
        if getattr(fire, "pattern_id", "") in _META:
            continue  # the structure fires already carry their own context
        # Rebuild from the clean base: capture it once, then re-annotate from it so
        # repeated runs neither double the text nor strand a stale annotation.
        if fire.rationale_base is None:
            fire.rationale_base = fire.rationale
        base = fire.rationale_base or ""
        ctx = leg_ctx.get(fire.position_id)
        if ctx:
            typ, role, u = ctx
            action = _ROLE_ACTION.get(role, "part of the structure")
            fire.rationale = base + (
                f"\n\nThis is the {role.replace('_', ' ')} of your {_struct_word(typ)} on {u} — {action}.")
        else:
            fire.rationale = base


# ---------------------------------------------------------------------------
# Per-structure re-derivation (shared by the load pass and the confirm/reject path)
# ---------------------------------------------------------------------------
def _market_inputs(state):
    """The already-loaded market inputs the structure fires read: the live treasury
    curve (carry fire, per-leg DTE) with the config scalar as the BBG-off fallback,
    and the load timestamp as each fire's fired_at. No fetch — reads the state only."""
    rate_curve = getattr(state, "risk_free_curve", None) or []
    rate_fallback = getattr(state, "risk_free_rate", None) or 0.04
    fired_at = getattr(state, "loaded_at", None) or datetime.now()
    return rate_curve, rate_fallback, fired_at


def rederive_structure_fires(state, account_state, structure, as_of: Optional[date] = None) -> list[Fire]:
    """Re-derive one structure's management fires from already-loaded market data
    (snapshot spot, holdings mark, treasury curve / fallback rate) — no Bloomberg
    fetch. Used when a confirm/reject changes which fires a structure unlocks, so the
    change can be reflected without a full reload. Returns the structure's fires, each
    tagged with its structure_id; the caller swaps them in by that id."""
    as_of = as_of or date.today()
    rate_curve, rate_fallback, fired_at = _market_inputs(state)
    by_id = {p.position_id: p for p in account_state.positions}
    return _fires_for_structure(
        account_state, structure, by_id, rate_curve, rate_fallback, as_of, fired_at)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_structure_fires(state, as_of: Optional[date] = None) -> None:
    """Append structure-aware management fires to every account and annotate
    confirmed-structure leg fires. Reads spot from the snapshot, the option mark from
    holdings, the treasury curve (carry fire, per-leg DTE) with the scalar rate as the
    BBG-off fallback, and the as-of date (today in the live app). Mutates state in
    place; the UI reads the fires and never recomputes."""
    as_of = as_of or date.today()
    for account_state in state.accounts.values():
        new_fires: list[Fire] = []
        for st in account_state.structures:
            new_fires.extend(rederive_structure_fires(state, account_state, st, as_of))
        account_state.fires.extend(new_fires)
        attach_structure_context(account_state)
