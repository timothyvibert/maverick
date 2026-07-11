"""Section 4 — Recent Trades (full per-account book).

Renders the account's raw Trades sheet rows (``account_state.trades`` — the
unkeyed pd.DataFrame, not ``trades_by_underlying``). No time windowing: the
whole sheet for the account, most-recent first. Row-shaping is a pure function
(``build_trades_rows``) so it's unit-testable without a browser.
"""
from __future__ import annotations

from typing import Any, Optional

import dash_ag_grid as dag
from dash import html

from pm.ui.deepdive.formatters import BUYSELL_STYLE, MONEY_FULL_FMT, QTY_FMT


def _isna(v: Any) -> bool:
    """True for None or float NaN (NaN != NaN) — avoids importing pandas here."""
    return v is None or (isinstance(v, float) and v != v)


def _clean(v: Any) -> Any:
    return None if _isna(v) else v


def _as_date_str(v: Any) -> Optional[str]:
    """ISO-ish date string (sorts correctly as text). Accepts date/datetime/str."""
    if _isna(v):
        return None
    try:
        return v.strftime("%Y-%m-%d")
    except Exception:
        s = str(v)
        return s[:10] if len(s) >= 10 else s


def _fmt_expiry(v: Any) -> str:
    if _isna(v):
        return ""
    try:
        return v.strftime("%b-%y")
    except Exception:
        return str(v)


def format_trade_instrument(row: dict) -> str:
    """One-line instrument descriptor for a trade row.

    Option -> "AAPL Put $285 Jun-26"; equity/other -> ticker or product name.
    """
    asset_class = (row.get("asset_class") or "").lower()
    opt_type = _clean(row.get("option_type"))
    if asset_class == "option" or opt_type:
        und = _clean(row.get("underlying_ticker")) or _clean(row.get("ticker_final")) or ""
        right = (str(opt_type).title() if opt_type else "Option")
        strike = _clean(row.get("option_strike"))
        strike_s = f"${strike:g}" if isinstance(strike, (int, float)) else ""
        expiry_s = _fmt_expiry(row.get("option_expiration"))
        parts = [und, right, strike_s, expiry_s]
        return " ".join(p for p in parts if p).strip() or "Option"
    return (_clean(row.get("ticker_final")) or _clean(row.get("product_name"))
            or _clean(row.get("underlying_ticker")) or "—")


def build_trades_columns() -> list[dict]:
    # Community filters: a true date picker on Date (the cell is an ISO string, so it
    # needs the ISO parse comparator), number filters on the numerics, text elsewhere.
    return [
        {"field": "trade_date", "headerName": "Date", "width": 115,
         "filter": "agDateColumnFilter",
         "filterParams": {"comparator": {"function": "dagfuncs.ISODateComparator"},
                          "browserDatePicker": True},
         "sort": "desc", "sortIndex": 0},
        {"field": "buy_sell", "headerName": "B/S", "width": 80,
         "filter": "agTextColumnFilter", "cellStyle": BUYSELL_STYLE},
        {"field": "action", "headerName": "Lifecycle", "width": 150, "filter": "agTextColumnFilter"},
        {"field": "instrument", "headerName": "Instrument", "flex": 2, "minWidth": 240,
         "filter": "agTextColumnFilter", "tooltipField": "instrument"},
        {"field": "quantity", "headerName": "Quantity", "width": 120,
         "type": "rightAligned", "filter": "agNumberColumnFilter", "valueFormatter": QTY_FMT},
        {"field": "principal", "headerName": "Principal", "width": 140,
         "type": "rightAligned", "filter": "agNumberColumnFilter", "valueFormatter": MONEY_FULL_FMT},
    ]


def build_trades_rows(account_state) -> list[dict]:
    """Display rows from the account's raw trades DataFrame, most-recent first.
    Returns [] when the account has no trades."""
    df = getattr(account_state, "trades", None)
    if df is None or getattr(df, "empty", True):
        return []

    records = df.to_dict("records")
    rows: list[dict] = []
    for r in records:
        qty = _clean(r.get("quantity"))
        principal = _clean(r.get("principal_amount"))
        rows.append({
            "trade_date": _as_date_str(r.get("trade_date")),
            "buy_sell": _clean(r.get("buy_sell")) or "",
            "action": _clean(r.get("option_lifecycle_action")) or "",
            "instrument": format_trade_instrument(r),
            "quantity": float(qty) if isinstance(qty, (int, float)) else None,
            "principal": float(principal) if isinstance(principal, (int, float)) else None,
        })
    # Most-recent first (None dates sort last).
    rows.sort(key=lambda x: x["trade_date"] or "", reverse=True)
    return rows


def render_trades_section(account_state) -> html.Div:
    rows = build_trades_rows(account_state)
    grid = dag.AgGrid(
        id="deepdive-trades-grid",
        columnDefs=build_trades_columns(),
        rowData=rows,
        dashGridOptions={
            "rowHeight": 28,
            "headerHeight": 32,
            "animateRows": False,
            # Highlight-to-copy (Community text selection + Ctrl-C), not range copy.
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
            "defaultColDef": {"sortable": True, "resizable": True, "suppressMovable": False},
        },
        className="ag-theme-balham blotter-grid",
        style={"height": "min(40vh, 420px)", "width": "100%"},
    )
    body = grid if rows else html.Div("No trades for this account.", className="dd-empty")
    return html.Div(className="dd-section", children=[
        html.Div(className="dd-section-head", children=[
            html.H2("Recent Trades", className="dd-section-title"),
            html.Span(f"{len(rows)} trades · most-recent first", className="dd-section-meta"),
        ]),
        body,
    ])
