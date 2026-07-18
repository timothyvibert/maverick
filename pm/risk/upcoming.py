"""Forward calendar of dated risk events (pure, render-time).

Projects the account's ALREADY-LOADED state onto the next N calendar days as
one date-keyed event list:

* **Option expiries** — every short option with an assignment obligation
  inside the window, read from the obligations rows (keyed by ``position_id``)
  and joined against the stored per-position assignment records
  (``AccountState.assignment["positions"]``) for P(assignment) and the
  early-exercise qualifier. The join this module performs is exactly the one
  the obligations rows were keyed for.
* **Ex-dividend dates** — from the per-leg discrete dividend schedules already
  resolved on the engine legs (``EngineLeg.divs_df``, ex-dates in
  ``(today, leg expiry]``), deduplicated per (underlying, ex-date). A row is
  flagged urgent when a short call on the name carries the stored
  ``"div_call"`` early-exercise qualifier — an ITM short call into an ex-date.
* **Expected earnings dates** — the raw snapshot report date per underlying
  (never reconstructed from the business-day countdown signal, which lands on
  the wrong calendar day), shown only for names where the book is net short
  vol, with the name's net dollar vega at stake.

Pure read of loaded state — no Bloomberg call, no recompute of any upstream
product. Missing data degrades per event source: with Bloomberg off the
expiry rows still fill (extract-only inputs), ex-div and earnings rows are
simply absent, and an unpriced P(assignment) travels as ``None`` (a dash,
never a fabricated number).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from pm.core import clock

DEFAULT_HORIZON_DAYS = 60

# Stable ordering for same-day events: the hard deadline first.
_KIND_ORDER = {"expiry": 0, "ex_div": 1, "earnings": 2}


def _as_date(v) -> Optional[date]:
    """Defensive date coercion — the snapshot's report-date cell type is not
    pinned (live BDP date vs fixture ISO string)."""
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v, errors="coerce")
    except Exception:
        return None
    if ts is None or ts is pd.NaT or (isinstance(ts, float) and ts != ts):
        return None
    try:
        return ts.date()
    except Exception:
        return None


def _expiry_events(account_state, ob, window_start: date, window_end: date):
    """One event per short option expiring in the window, carrying the
    obligation dollars and the joined assignment record."""
    assignment = (getattr(account_state, "assignment", None) or {}).get(
        "positions", {}) or {}
    events = []
    n_priced = 0
    n_short = 0
    for side in (ob.puts, ob.calls):
        for row in side.rows:
            if row.expiry is None:
                continue
            if not (window_start <= row.expiry <= window_end):
                continue
            rec = assignment.get(row.position_id) or {}
            p_assign = rec.get("p_assign")
            n_short += 1
            if p_assign is not None:
                n_priced += 1
            events.append({
                "kind": "expiry",
                "date": row.expiry,
                "underlying": row.symbol,
                "position_id": row.position_id,
                "right": row.right,
                "contracts": row.contracts,
                "strike": row.strike,
                "obligation": row.dollars,
                "itm": row.itm,
                "p_assign": p_assign,
                "p_assign_reason": rec.get("p_assign_reason"),
                "flag": rec.get("flag"),
                "flag_reason": rec.get("flag_reason"),
                "urgent": bool(row.itm),
            })
    return events, n_priced, n_short


def _ex_div_events(account_state, elegs, window_start: date, window_end: date):
    """One event per (underlying, ex-date) with a discrete dividend in the
    window, deduplicated across the name's option legs. Urgent only for
    ex-dates a ``div_call``-flagged short call actually spans (ex-date ≤ that
    call's expiry) — a later ex-date on the same name, contributed by a
    longer-dated leg's schedule, is a dated fact, not an urgency."""
    assignment = (getattr(account_state, "assignment", None) or {}).get(
        "positions", {}) or {}
    pos_underlying = {p.position_id: (getattr(p, "underlying_symbol", None)
                                      or getattr(p, "symbol", None) or "")
                      for p in getattr(account_state, "positions", None) or []}
    elegs_by_pid = {getattr(e, "position_id", None): e for e in elegs}
    # underlying symbol -> (flag_reason, latest flagged-call expiry)
    flagged: dict[str, tuple[str, Optional[date]]] = {}
    for pid, rec in assignment.items():
        if rec.get("flag") != "div_call":
            continue
        sym = pos_underlying.get(pid, "")
        exp = _as_date(getattr(elegs_by_pid.get(pid), "expiry", None))
        if not sym or exp is None:
            continue
        prev = flagged.get(sym)
        if prev is None or (prev[1] is not None and exp > prev[1]):
            flagged[sym] = (rec.get("flag_reason") or "", exp)

    seen: dict[tuple, dict] = {}
    for eleg in elegs:
        divs = getattr(eleg, "divs_df", None)
        if divs is None or len(divs) == 0:
            continue
        sym = pos_underlying.get(eleg.position_id, "")
        ticker = eleg.underlying_bbg or sym
        for _, drow in divs.iterrows():
            ex = _as_date(drow.get("EX_DATE"))
            if ex is None or not (window_start <= ex <= window_end):
                continue
            key = (ticker, ex)
            if key in seen:
                continue
            try:
                dps = float(drow.get("DIVIDENDS"))
            except (TypeError, ValueError):
                dps = None
            flag = flagged.get(sym)
            urgent = bool(flag and flag[1] is not None and ex <= flag[1])
            seen[key] = {
                "kind": "ex_div",
                "date": ex,
                "underlying": sym,
                "underlying_bbg": ticker,
                "position_id": None,
                "dps": dps,
                "urgent": urgent,
                "flag": "div_call" if urgent else None,
                "flag_reason": flag[0] if urgent else None,
            }
    return list(seen.values())


def _earnings_events(account_state, lenses, window_start: date, window_end: date):
    """One event per net-short-vol name with an expected report date in the
    window. Reads the RAW snapshot date field — the business-day countdown
    signal is a count, not a calendar date."""
    snap = getattr(account_state, "snapshot", None)
    und = getattr(snap, "underlyings", None) if snap is not None else None
    if und is None or getattr(und, "empty", True):
        return []
    events = []
    for row in (lenses or {}).get("rows", []):
        vega = row.get("dollar_vega")
        if vega is None or vega >= 0:
            continue           # earnings rows only where the book is short vol
        ticker = row.get("underlying_ticker")
        if not ticker or ticker not in und.index:
            continue
        try:
            cell = und.loc[ticker]
            if hasattr(cell, "ndim") and cell.ndim > 1:
                cell = cell.iloc[0]      # duplicated snapshot index yields a frame
            raw = cell.get("EXPECTED_REPORT_DT")
        except Exception:
            continue
        report = _as_date(raw)
        if report is None or not (window_start <= report <= window_end):
            continue
        events.append({
            "kind": "earnings",
            "date": report,
            "underlying": row.get("symbol") or ticker,
            "position_id": None,
            "vega": vega,
            "urgent": False,
        })
    return events


def upcoming_events(state, account_state, ob, lenses=None, elegs=None,
                    as_of: Optional[date] = None,
                    horizon_days: int = DEFAULT_HORIZON_DAYS) -> dict:
    """The account's dated risk events over ``[as_of, as_of + horizon_days]``
    (both ends inclusive — an event landing today is still a live decision).

    ``ob`` is the account's :class:`AssignmentObligations`; ``lenses`` the
    concentration lenses (for per-name net vega); ``elegs`` an optional
    pre-built engine-leg list (built here, pure, when absent). Returns
    ``{"rows", "as_of", "horizon_days", "n_assign_priced",
    "n_assign_total", "warnings"}`` with rows date-sorted.
    """
    ref = as_of or clock.today()
    window_end = ref + timedelta(days=horizon_days)
    warnings: list[str] = []

    if elegs is None:
        try:
            from pm.risk.pricing_adapter import build_engine_legs
            elegs = build_engine_legs(state, account_state, today=ref)
        except Exception:
            elegs = []
            warnings.append("engine legs unavailable — ex-dividend rows "
                            "omitted")

    rows, n_priced, n_short = _expiry_events(account_state, ob, ref, window_end)
    rows += _ex_div_events(account_state, elegs, ref, window_end)
    rows += _earnings_events(account_state, lenses, ref, window_end)

    for r in rows:
        r["days"] = (r["date"] - ref).days
    rows.sort(key=lambda r: (r["date"], _KIND_ORDER.get(r["kind"], 9),
                             -(r.get("obligation") or 0.0)))
    return {
        "rows": rows,
        "as_of": ref,
        "horizon_days": horizon_days,
        "n_assign_priced": n_priced,
        "n_assign_total": n_short,
        "warnings": warnings,
    }
