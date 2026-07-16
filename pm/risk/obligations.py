"""Assignment obligations — what the book owes if its short options are assigned.

Two sides, kept separate because they are different promises:

  * **Short puts — cash to fund.** If every short put is assigned, the account
    buys stock at strike: cash obligation = Σ |contracts| × multiplier × strike.
  * **Short calls — delivery at strike.** If every short call is assigned, the
    account delivers shares and receives strike proceeds: shares committed =
    Σ |contracts| × multiplier; proceeds = Σ |contracts| × multiplier × strike.
    The uncovered slice (short calls no covered structure claims) is broken out —
    delivery there means buying the stock first.

Short legs only — a LONG option is a right, not an obligation (the old expiry
ladder's |qty| erased that distinction). Expired contracts are excluded and
counted: a dead obligation must not read as an imminent one.

Pure, render-time aggregation over already-loaded state (positions + snapshot
spot + structures): no Bloomberg, no engine, no recompute. The obligation totals
are extract-only, so they render with Bloomberg off; the in-the-money subtotals
need the snapshot spot and are None (a dash, never $0) without it.

Rows are keyed by ``position_id`` so a per-leg probability-of-assignment column
can join later without reshaping the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

# Days-to-expiry windows — the same windows the retired strike-obligation expiry
# ladder used, so the desk's mental buckets carry over.
OBLIGATION_WINDOWS = [
    ("≤30d", lambda d: d <= 30),
    ("31–60d", lambda d: 31 <= d <= 60),
    ("61–90d", lambda d: 61 <= d <= 90),
    (">90d", lambda d: d > 90),
]


@dataclass
class ObligationRow:
    """One short option position's obligation. Keyed by position_id (the future
    per-leg P(assignment) join key)."""
    position_id: str
    underlying_ticker: Optional[str]
    symbol: str
    right: str                       # 'CALL' | 'PUT'
    contracts: float                 # |quantity|, real contracts
    multiplier: float
    strike: float
    expiry: Optional[date]
    dollars: float                   # contracts × multiplier × strike
    shares: float                    # contracts × multiplier (delivery size)
    spot: Optional[float]
    itm: Optional[bool]              # None when the spot is unknown (BBG off)


@dataclass
class ObligationSide:
    """One side's totals (short puts, or short calls)."""
    n_positions: int = 0
    contracts: float = 0.0
    shares: float = 0.0              # contracts × multiplier
    dollars: float = 0.0             # Σ contracts × multiplier × strike
    itm_contracts: float = 0.0
    itm_dollars: Optional[float] = None   # None until at least one row has a spot
    n_unknown_moneyness: int = 0     # rows with no spot — the ITM subtotal excludes them
    by_window: list = field(default_factory=list)   # [{label, contracts, dollars}]
    rows: list = field(default_factory=list)        # [ObligationRow], largest dollars first


@dataclass
class AssignmentObligations:
    puts: ObligationSide
    calls: ObligationSide
    covered_call_contracts: float    # short-call contracts claimed by covering structures
    uncovered_call_contracts: float  # contracts in uncovered (naked-excess) readings
    n_expired: int
    as_of: date
    warnings: list = field(default_factory=list)


def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None     # NaN-safe


def _spot_lookup(account_state) -> "pd.DataFrame | None":
    snap = getattr(getattr(account_state, "snapshot", None), "underlyings", None)
    if snap is None or "PX_LAST" not in getattr(snap, "columns", []):
        return None
    return snap


def _spot_of(snap, ticker) -> Optional[float]:
    if snap is None or ticker is None:
        return None
    try:
        if ticker not in snap.index:
            return None
        v = snap.loc[ticker, "PX_LAST"]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else None
    return _num(v)


def _empty_windows() -> list[dict]:
    return [{"label": lbl, "contracts": 0.0, "dollars": 0.0}
            for lbl, _ in OBLIGATION_WINDOWS]


def _uncovered_call_contracts(account_state) -> float:
    """Short-call contracts sitting in uncovered (naked-excess) readings, counted
    once per position: contention alternatives are mutually-exclusive readings of
    the same legs, so within a group the position's largest uncovered slice counts,
    never the sum across alternatives (mirrors the exposure rollup's ledger)."""
    ungrouped = 0.0
    per_group: dict[str, dict[str, float]] = {}
    for st in getattr(account_state, "structures", []) or []:
        if getattr(st, "status", None) == "rejected":
            continue
        for leg in getattr(st, "legs", []) or []:
            if getattr(leg, "role", None) != "naked_excess_short_call":
                continue
            try:
                n = abs(float(leg.allocated_qty))
            except (TypeError, ValueError):
                continue
            group = getattr(st, "contention_group", None)
            if group:
                g = per_group.setdefault(group, {})
                pid = getattr(leg, "position_id", None)
                g[pid] = max(g.get(pid, 0.0), n)
            else:
                ungrouped += n
    return ungrouped + sum(n for g in per_group.values() for n in g.values())


def assignment_obligations(account_state, as_of: Optional[date] = None) -> AssignmentObligations:
    """The account's short-option assignment obligations, both sides. Pure read of
    ``positions`` (+ snapshot spot for the ITM subtotals, structures for the
    covered/uncovered call split)."""
    ref = as_of or date.today()
    snap = _spot_lookup(account_state)
    sides = {"PUT": ObligationSide(by_window=_empty_windows()),
             "CALL": ObligationSide(by_window=_empty_windows())}
    n_expired = 0
    warnings: list[str] = []

    for p in getattr(account_state, "positions", []) or []:
        if getattr(p, "asset_class", None) != "option":
            continue
        qty = _num(getattr(p, "quantity", None))
        if qty is None or qty >= 0:                 # long options are rights, not obligations
            continue
        right = (getattr(p, "right", None) or getattr(p, "option_type", None) or "").upper()
        if right not in ("CALL", "PUT"):
            continue
        strike = _num(getattr(p, "strike", None))
        if strike is None or strike <= 0:
            warnings.append(f"short option {getattr(p, 'position_id', '?')} has no usable "
                            "strike — excluded from obligations")
            continue
        expiry = getattr(p, "expiry", None)
        dte: Optional[int] = None
        if expiry is not None:
            try:
                dte = (expiry - ref).days
            except TypeError:
                dte = None
        if dte is not None and dte < 0:
            n_expired += 1
            continue

        mult = _num(getattr(p, "multiplier", None)) or 100.0
        contracts = abs(qty)
        shares = contracts * mult
        dollars = shares * strike
        und = getattr(p, "underlying_bbg_ticker", None)
        spot = _spot_of(snap, und)
        itm: Optional[bool] = None
        if spot is not None:
            itm = (strike > spot) if right == "PUT" else (spot > strike)

        side = sides[right]
        side.n_positions += 1
        side.contracts += contracts
        side.shares += shares
        side.dollars += dollars
        if itm is None:
            side.n_unknown_moneyness += 1
        else:
            if side.itm_dollars is None:
                side.itm_dollars = 0.0
            if itm:
                side.itm_contracts += contracts
                side.itm_dollars += dollars
        if dte is not None:
            for i, (_lbl, pred) in enumerate(OBLIGATION_WINDOWS):
                if pred(dte):
                    side.by_window[i]["contracts"] += contracts
                    side.by_window[i]["dollars"] += dollars
                    break
        else:
            warnings.append(f"short option {getattr(p, 'position_id', '?')} has no expiry "
                            "— counted in the totals, not in any window")
        side.rows.append(ObligationRow(
            position_id=getattr(p, "position_id", ""), underlying_ticker=und,
            symbol=getattr(p, "underlying_symbol", None) or str(und or "—").split(" ")[0],
            right=right, contracts=contracts, multiplier=mult, strike=strike,
            expiry=expiry, dollars=dollars, shares=shares, spot=spot, itm=itm))

    for side in sides.values():
        side.rows.sort(key=lambda r: r.dollars, reverse=True)

    uncovered = _uncovered_call_contracts(account_state)
    covered = max(sides["CALL"].contracts - uncovered, 0.0)
    return AssignmentObligations(
        puts=sides["PUT"], calls=sides["CALL"],
        covered_call_contracts=covered, uncovered_call_contracts=uncovered,
        n_expired=n_expired, as_of=ref, warnings=warnings)
