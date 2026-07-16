"""Concentration on every basis that matters for an options book.

The single net delta-$ lens empties exactly the books where concentration is most
dangerous: a delta-neutral structure (box, straddle, collar) nets to ~0 while its
notional is largest, and a short-vol book's convexity appears nowhere by name. This
module slices the SAME per-position dollar greeks the exposure layer aggregates into
per-name rows on five complementary bases — net delta-$, gross |delta-$|, dollar
gamma (per-1%), dollar vega, and the short-option assignment obligations — plus the
name's market value, with an asset-class split alongside.

Default ordering is GROSS |delta-$|: a hedged name floats to the top by the size of
what it is running, not by what happens to net out today.

Pure, render-time aggregation over already-loaded state (``greeks.by_position`` +
positions + the obligations module): no Bloomberg, no engine, no recompute.
Conserved: Σ net delta-$ over the rows == the account's net dollar delta
(``AccountExposure.economic_exposure``). Missing data is counted and surfaced, never
silently dropped: a row whose delta is missing is excluded from the delta sums and
named; a name whose every gamma/vega is missing carries None (a dash), never $0.
"""
from __future__ import annotations

from typing import Optional

from pm.risk.exposure import _iter_greek_rows, _num, _short_name, _symbol_by_underlying
from pm.risk.obligations import AssignmentObligations, assignment_obligations

# Asset-class display order for the split table.
_CLASS_ORDER = ("equity", "fund_etf", "option", "cash", "other")


def _sum_or_none(acc: dict, key: str):
    """Skipna accumulator read: the summed value when anything accrued, else None
    (all-missing must never read as $0)."""
    return acc[key] if acc.get(f"{key}_has") else None


def concentration_lenses(account_state,
                         obligations: Optional[AssignmentObligations] = None) -> dict:
    """Per-underlying multi-basis concentration rows + account totals + asset-class
    split. ``obligations`` may be passed in (the cockpit computes it once for its
    own panel); computed here when absent."""
    nav = abs(_num(getattr(account_state, "nav", None)) or 0.0) or None
    sym_by_ut = _symbol_by_underlying(account_state)
    ob = obligations if obligations is not None else assignment_obligations(account_state)

    put_obl: dict = {}
    call_obl: dict = {}
    for r in ob.puts.rows:
        put_obl[r.underlying_ticker] = put_obl.get(r.underlying_ticker, 0.0) + r.dollars
    for r in ob.calls.rows:
        call_obl[r.underlying_ticker] = call_obl.get(r.underlying_ticker, 0.0) + r.dollars

    # position_id -> asset class, and per-name / per-class market value
    class_of: dict = {}
    mv_by_name: dict = {}
    split: dict = {}
    for p in getattr(account_state, "positions", []) or []:
        ac = getattr(p, "asset_class", None) or "other"
        pid = getattr(p, "position_id", None)
        if pid:
            class_of[pid] = ac
        mv = _num(getattr(p, "market_value", None))
        s = split.setdefault(ac, {"asset_class": ac, "market_value": 0.0,
                                  "n_positions": 0, "net_dollar_delta": 0.0,
                                  "net_dollar_delta_has": False})
        s["n_positions"] += 1
        if mv is not None:
            s["market_value"] += mv
        if ac == "option":
            name = getattr(p, "underlying_bbg_ticker", None)
        elif ac in ("equity", "fund_etf"):
            name = getattr(p, "bbg_ticker", None)
        else:
            continue                                   # cash/other are not name exposure
        if name and mv is not None:
            mv_by_name[name] = mv_by_name.get(name, 0.0) + mv

    # per-name greek sums (skipna per column, with has-flags and missing counts)
    names: dict = {}
    n_missing_delta_rows = 0
    missing_delta_names: list[str] = []
    for r in _iter_greek_rows(account_state):
        ut = r.get("underlying_ticker")
        n = names.setdefault(ut, {
            "net_dd": 0.0, "net_dd_has": False, "gross_dd": 0.0,
            "dg": 0.0, "dg_has": False, "dv": 0.0, "dv_has": False,
        })
        dd = _num(r.get("dollar_delta"))
        if dd is None:
            n_missing_delta_rows += 1
            sym = sym_by_ut.get(ut) or _short_name(ut)
            if sym not in missing_delta_names:
                missing_delta_names.append(sym)
        else:
            n["net_dd"] += dd
            n["gross_dd"] += abs(dd)
            n["net_dd_has"] = True
        dg = _num(r.get("dollar_gamma"))
        if dg is not None:
            n["dg"] += dg
            n["dg_has"] = True
        dv = _num(r.get("dollar_vega"))
        if dv is not None:
            n["dv"] += dv
            n["dv_has"] = True
        # class split delta (by the position's own asset class)
        pid = r.get("position_id")
        ac = class_of.get(pid)
        if ac in split and dd is not None:
            split[ac]["net_dollar_delta"] += dd
            split[ac]["net_dollar_delta_has"] = True

    all_names = set(names) | set(mv_by_name) | set(put_obl) | set(call_obl)
    rows: list[dict] = []
    for ut in all_names:
        n = names.get(ut, {})
        net = _sum_or_none(n, "net_dd") if n else None
        gross = n.get("gross_dd") if n.get("net_dd_has") else None
        rows.append({
            "underlying_ticker": ut,
            "symbol": sym_by_ut.get(ut) or _short_name(ut),
            "net_dollar_delta": net,
            "gross_dollar_delta": gross,
            "dollar_gamma": _sum_or_none(n, "dg") if n else None,
            "dollar_vega": _sum_or_none(n, "dv") if n else None,
            "put_obligation": put_obl.get(ut, 0.0),
            "call_obligation": call_obl.get(ut, 0.0),
            "market_value": mv_by_name.get(ut),
            "pct_nav": (net / nav) if (net is not None and nav) else None,
        })
    # gross first — the lens that survives a hedge; unknown-gross rows sink to the end
    rows.sort(key=lambda d: (d["gross_dollar_delta"] is None,
                             -(d["gross_dollar_delta"] or 0.0)))

    def _tot(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) if vals else None

    account = {
        "net_dollar_delta": _tot("net_dollar_delta"),
        "gross_dollar_delta": _tot("gross_dollar_delta"),
        "dollar_gamma": _tot("dollar_gamma"),
        "dollar_vega": _tot("dollar_vega"),
        "put_obligation": ob.puts.dollars,
        "call_obligation": ob.calls.dollars,
        "market_value": _tot("market_value"),
        "pct_nav": ((_tot("net_dollar_delta") / nav)
                    if (_tot("net_dollar_delta") is not None and nav) else None),
    }

    split_rows = [
        {"asset_class": ac,
         "market_value": split[ac]["market_value"],
         "n_positions": split[ac]["n_positions"],
         "net_dollar_delta": _sum_or_none(split[ac], "net_dollar_delta")}
        for ac in _CLASS_ORDER if ac in split
    ] + [
        {"asset_class": ac,
         "market_value": s["market_value"], "n_positions": s["n_positions"],
         "net_dollar_delta": _sum_or_none(s, "net_dollar_delta")}
        for ac, s in split.items() if ac not in _CLASS_ORDER
    ]

    return {
        "rows": rows,
        "account": account,
        "asset_class_split": split_rows,
        "missing": {"n_rows": n_missing_delta_rows, "names": missing_delta_names},
        "nav": nav,
    }
