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

Core suite detected here: covered call (full / partial / over-write), collar,
vertical, covered / cash-secured put, straddle / strangle. Calendars, diagonals
(incl. poor-man's covered call), risk reversals and ratio / backspreads are not
grouped — their legs fall through as ungrouped standalone legs — but the engine
is built generically so they can be added later behind this same interface.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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

_MULT = 100  # option contract multiplier
_OPEN_ACTIONS = {"Buy to Open", "Sell to Open"}


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
# Per-underlying detection (one account-scoped allocation pass)
# ---------------------------------------------------------------------------
def _detect_for_underlying(
    account: str,
    underlying: str,
    stock_legs: list[Position],
    options: list[Position],
    trades_df,
) -> list[Structure]:
    """Allocate signed-quantity slices in priority order: collar (matched expiry)
    → covered call (+ residual / naked-excess slices) → covered / cash-secured put
    → vertical (1:1, same expiry) → straddle / strangle. Legs consumed by a
    higher-priority structure are not re-used, which deterministically resolves the
    collar-vs-covered-call subsumption; genuinely symmetric contention (a leg that
    fits two equally-valid structures) is reserved for ranked alternatives."""
    out: list[Structure] = []

    # Remaining signed quantities to allocate.
    stock_rem = sum((_num(s.quantity) or 0.0) for s in stock_legs)
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
                qty = int(abs(opt_rem[sc.position_id]))
                if qty == 0 or qty != int(abs(opt_rem[lc.position_id])):
                    continue
                shares = qty * _MULT
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
                lots = min(int(stock_rem // _MULT), int(abs(opt_rem[sc.position_id])), int(opt_rem[lp.position_id]))
                if lots <= 0:
                    continue
                shares = lots * _MULT
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
        short_call_shares = sum(abs(opt_rem[o.position_id]) for o in short_calls) * _MULT
        covered_shares = min(stock_rem, short_call_shares)
        covered_contracts = int(covered_shares // _MULT)

        covered_legs: list[StructureLeg] = [StructureLeg(stock_pid, covered_contracts * _MULT, "long_stock")]
        excess_legs: list[StructureLeg] = []
        need = covered_contracts
        for o in short_calls:
            have = int(abs(opt_rem[o.position_id]))
            take = min(need, have)
            if take > 0:
                covered_legs.append(StructureLeg(o.position_id, -take, "short_call"))
                need -= take
            leftover = have - take
            if leftover > 0:
                excess_legs.append(StructureLeg(o.position_id, -leftover, "naked_excess_short_call"))
            opt_rem[o.position_id] = 0.0  # all short calls accounted for (covered or naked-excess)

        full = covered_contracts * _MULT == stock_rem and not excess_legs
        out.append(Structure(
            structure_id=_sid(account, underlying, COVERED_CALL, [l.position_id for l in covered_legs]),
            account=account, underlying=underlying, type=COVERED_CALL, confidence_band=HIGH,
            status="proposed", legs=covered_legs,
            rationale_trace=_trace(
                {"long_shares": {"value": stock_rem, "source": "EXTRACT:quantity"},
                 "short_call_shares": {"value": short_call_shares, "source": "computed:|qty|x100"},
                 "covered_shares": {"value": covered_contracts * _MULT, "source": "computed:min(long,short)"}},
                f"{covered_contracts} short call(s) cover {covered_contracts * _MULT} of {int(stock_rem)} {underlying} shares",
                f"covered call ({'full' if full else 'partial cover'})"),
            source="detector:covered_call"))

        stock_after = stock_rem - covered_contracts * _MULT
        if stock_after > 0:
            out.append(Structure(
                structure_id=_sid(account, underlying, RESIDUAL_LONG, [stock_pid], suffix="residual"),
                account=account, underlying=underlying, type=RESIDUAL_LONG, confidence_band=MEDIUM,
                status="proposed", legs=[StructureLeg(stock_pid, stock_after, "residual_long")],
                rationale_trace=_trace(
                    {"residual_shares": {"value": stock_after, "source": "computed:long-covered"}},
                    f"{int(stock_after)} {underlying} shares uncovered after the over-write",
                    "uncovered long stock (residual of a partial covered call)"),
                source="detector:covered_call"))
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

    # ---- 5) Covered / cash-secured put: short put(s) NOT consumed above --------
    # Runs last so a short put that is the leg of a vertical or straddle is
    # claimed there first; only a *lone* short put is an income-overlay put.
    short_puts = sorted(puts_short(), key=lambda o: (str(o.expiry), o.strike or 0))
    if short_puts:
        is_covered = stock_rem < 0  # short stock backing the short put
        typ = COVERED_PUT if is_covered else CASH_SECURED_PUT
        contracts = sum(int(abs(opt_rem[o.position_id])) for o in short_puts)
        legs = [StructureLeg(o.position_id, opt_rem[o.position_id], "short_put") for o in short_puts]
        out.append(Structure(
            structure_id=_sid(account, underlying, typ, [l.position_id for l in legs]),
            account=account, underlying=underlying, type=typ, confidence_band=HIGH,
            status="proposed", legs=legs,
            rationale_trace=_trace(
                {"short_put_contracts": {"value": contracts, "source": "EXTRACT:quantity"},
                 "short_stock": {"value": is_covered, "source": "EXTRACT:quantity"}},
                f"short {contracts} {underlying} put(s)"
                + (" against short stock" if is_covered else " (income posture; no stock leg)"),
                typ),
            source=f"detector:{typ}"))
        for o in short_puts:
            opt_rem[o.position_id] = 0.0

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
