"""Section 1 — Positions (full book, institutional view).

One AG-Grid row per position in the account (every position, not just fired
ones), with an institutional column set and the account's alerts merged into
an Alerts column (the standalone Actionables table was folded in here). Mirrors
the blotter's column/style idioms, reuses ``format_position_descriptor`` and the
shared formatters, and keeps ``getRowId = account::position_id`` so the shared
modal's click + prev/next plumbing works unchanged. Row-building is a pure
function (``build_positions_rows``) — unit-testable without a browser.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import dash_ag_grid as dag
from dash import html

from pm.store.suppression_store import is_active
from pm.ui import state_access as sa
from pm.ui.blotter.grid import format_position_descriptor
from pm.ui.deepdive.actionables import summary_line
from pm.ui.deepdive.formatters import (
    MONEY_FULL_FMT,
    PCT_ABS_FMT,
    PCT_SIGNED_FMT,
    QTY_FMT,
    SIGNED_COLOR_STYLE,
)
from pm.ui.deepdive.structures_panel import (
    build_structure_columns,
    build_structure_rows,
)

# Alerts cell: tier badge prepended (stays visible if the names truncate),
# then the comma-joined alert names. Amber on T1, charcoal otherwise.
_ALERT_FMT = {
    "function": "!params.value ? '' : "
                "((params.data && params.data._alert_tier ? 'T' + params.data._alert_tier + ' · ' : '') "
                "+ params.value)"
}
_ALERT_STYLE = {
    "styleConditions": [
        {"condition": "params.data && params.data._alert_tier == 1",
         "style": {"color": "var(--pm-urgent)", "fontWeight": "700"}},
        {"condition": "params.data && params.data._alert_tier && params.data._alert_tier > 1",
         "style": {"color": "var(--pm-charcoal)", "fontWeight": "600"}},
    ],
    "defaultStyle": {"color": "var(--pm-grey-300)"},
}

_TYPE_LABEL = {
    "equity": "Equity",
    "fund_etf": "Fund",
    "cash": "Cash",
    "other": "Other",
}


def _type_label(p) -> str:
    """Compact instrument-type label across every asset class."""
    if p.asset_class == "option":
        right = (p.option_type or p.right or "").upper()
        return {"CALL": "Call", "PUT": "Put"}.get(right, "Option")
    return _TYPE_LABEL.get(p.asset_class, "Other")


def _strike_expiry(p) -> str:
    """'$130 / Jun-27' for options; '' for everything else."""
    if p.asset_class != "option":
        return ""
    strike = f"${p.strike:g}" if p.strike is not None else ""
    expiry = ""
    if p.expiry is not None:
        try:
            expiry = p.expiry.strftime("%b-%y")
        except Exception:
            expiry = str(p.expiry)
    return " / ".join(part for part in (strike, expiry) if part)


def _dte_for(position) -> Optional[int]:
    """Days to expiry for an option (else None) — matches blotter grid._dte_for."""
    if position is None or position.asset_class != "option" or position.expiry is None:
        return None
    try:
        return (position.expiry - date.today()).days
    except Exception:
        return None


def build_positions_columns() -> list[dict]:
    """Institutional column defs for the full-book grid. No Account column
    (the page is already scoped to one account)."""
    return [
        {"field": "type", "headerName": "Type", "width": 72, "filter": "agTextColumnFilter"},
        {"field": "underlying", "headerName": "Ticker", "width": 92,
         "filter": "agTextColumnFilter", "cellClass": "blotter-ticker-cell",
         "valueFormatter": {"function": "params.value ? params.value : '—'"}},
        {"field": "name", "headerName": "Name", "width": 180, "filter": "agTextColumnFilter",
         "tooltipField": "name", "cellClass": "dd-cell-ellipsis",
         "valueFormatter": {"function": "params.value ? params.value : '—'"}},
        {"field": "position_label", "headerName": "Position", "width": 180,
         "filter": "agTextColumnFilter", "tooltipField": "position_label", "cellClass": "dd-cell-ellipsis"},
        {"field": "quantity", "headerName": "Qty", "width": 96,
         "type": "rightAligned", "filter": "agNumberColumnFilter", "valueFormatter": QTY_FMT},
        {"field": "cost_basis", "headerName": "Cost Basis", "width": 120,
         "type": "rightAligned", "filter": "agNumberColumnFilter", "valueFormatter": MONEY_FULL_FMT},
        {"field": "market_value", "headerName": "Mkt Value", "width": 120,
         "type": "rightAligned", "filter": "agNumberColumnFilter", "valueFormatter": MONEY_FULL_FMT},
        {"field": "position_size_pct", "headerName": "% NAV", "width": 84,
         "type": "rightAligned", "filter": "agNumberColumnFilter", "valueFormatter": PCT_ABS_FMT},
        {"field": "pnl", "headerName": "Unreal P&L", "width": 116,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": MONEY_FULL_FMT, "cellStyle": SIGNED_COLOR_STYLE},
        {"field": "pnl_pct", "headerName": "P&L %", "width": 84,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_SIGNED_FMT, "cellStyle": SIGNED_COLOR_STYLE},
        {"field": "strike_expiry", "headerName": "Strike / Expiry", "width": 120,
         "filter": "agTextColumnFilter",
         "valueFormatter": {"function": "params.value ? params.value : ''"}},
        {"field": "dte", "headerName": "DTE", "width": 70,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": {"function": "params.value == null ? '' : params.value"}},
        {"field": "alerts", "headerName": "Alerts", "flex": 2, "minWidth": 240,
         "tooltipField": "alerts", "valueFormatter": _ALERT_FMT,
         "cellClass": "dd-cell-ellipsis", "cellStyle": _ALERT_STYLE},
    ]


def build_positions_rows(account_state, state) -> list[dict]:
    """One row per position. Hidden ``_account/_position_id/_underlying`` drive
    the shared modal click + prev/next; ``_has_fires``/``_alert_tier`` route the
    click (Alert vs Tearsheet) and colour the badge. The ``alerts`` cell carries
    the position's distinct alert names (severity order), mirroring the blotter.
    Reads only fields already on Position / the loaded fires — nothing recomputed."""
    account = account_state.account
    nav = abs(account_state.nav) if account_state.nav else 0.0

    # fires-on-position index (severity-ordered, for the merged Alerts cell).
    # Suppressed/snoozed fires are skipped so the Alerts cell, the
    # alert badge (_has_fires/_alert_tier) and the click routing reflect only
    # active alerts. The position itself still renders — By Position is the
    # holdings book, so a fully-muted position stays as a row with no alerts.
    fires_by_pos: dict[str, list] = {}
    for f in account_state.fires:
        if not is_active(f):
            continue
        fires_by_pos.setdefault(f.position_id, []).append(f)

    rows: list[dict] = []
    for p in account_state.positions:
        fires = sorted(fires_by_pos.get(p.position_id, []), key=lambda f: f.tier)
        alert_tier = fires[0].tier if fires else None
        alert_names: list[str] = []
        for f in fires:
            if f.pattern_name not in alert_names:
                alert_names.append(f.pattern_name)

        mv = sa.coerce_float(p.market_value)
        pnl = sa.coerce_float(p.unrealized_pnl)
        pnl_pct = sa.coerce_float(p.unrealized_pnl_pct)
        cost_basis = sa.coerce_float(p.cost_basis)
        qty = sa.coerce_float(p.quantity)
        pct_nav = (abs(mv) / nav) if (mv is not None and nav) else None
        underlying = p.underlying_symbol or p.symbol or ""

        rows.append({
            # Hidden — click / nav resolve via these (same keys as the blotter).
            "_account": account,
            "_position_id": p.position_id,
            "_underlying": underlying,
            "_has_fires": bool(fires),
            "_alert_tier": alert_tier,
            # Displayed.
            "type": _type_label(p),
            "underlying": underlying,
            "name": p.name or "",
            "position_label": format_position_descriptor(p),
            "quantity": qty,
            "cost_basis": cost_basis,
            "market_value": mv,
            "position_size_pct": pct_nav,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "strike_expiry": _strike_expiry(p),
            "dte": _dte_for(p),
            "alerts": ", ".join(alert_names),
        })
    return rows


def _render_pos_toggle(pos_view: str) -> html.Div:
    """By Position | By Structure segmented toggle (server-side re-render, like
    the blotter's grouping toggle — Community has no native row grouping)."""
    def cls(mode: str) -> str:
        return "group-toggle" + (" group-toggle-active" if pos_view == mode else "")
    return html.Div(className="struct-toggle", children=[
        html.Span("View:", className="blotter-control-label"),
        html.Button("By Position", id="pos-byposition-btn", n_clicks=0, className=cls("position")),
        html.Button("By Structure", id="pos-bystructure-btn", n_clicks=0, className=cls("structure")),
    ])


def _positions_grid(account_state, state) -> dag.AgGrid:
    return dag.AgGrid(
        id="deepdive-positions-grid",
        columnDefs=build_positions_columns(),
        rowData=build_positions_rows(account_state, state),
        dashGridOptions={
            "rowHeight": 28,
            "headerHeight": 32,
            "animateRows": False,
            # Highlight-to-copy (Community text selection + Ctrl-C), not range copy.
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
            "rowClassRules": {
                "blotter-row-t1": "params.data && params.data._alert_tier == 1",
            },
            "defaultColDef": {"sortable": True, "resizable": True, "suppressMovable": False},
        },
        className="ag-theme-balham blotter-grid",
        getRowId={"function": "params.data._account + '::' + params.data._position_id"},
        style={"height": "min(52vh, 560px)", "width": "100%"},
    )


def _structures_grid(account_state, state, expanded_sids) -> dag.AgGrid:
    # Same id as the By Position grid: only one is mounted at a time, and reusing
    # the id that exists in the initial layout keeps cellClicked wired (a grid id
    # that only ever appears via a callback never binds its event props).
    return dag.AgGrid(
        id="deepdive-positions-grid",
        columnDefs=build_structure_columns(),
        rowData=build_structure_rows(account_state, state, expanded_sids),
        dashGridOptions={
            "rowHeight": 28,
            "headerHeight": 32,
            "animateRows": False,
            # Highlight-to-copy (Community text selection + Ctrl-C), not range copy.
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
            # Structure→leg ordering is meaningful, so this grid is not sortable.
            "rowClassRules": {
                "struct-grid-subrow": "params.data && (params.data._kind == 'leg' "
                                      "|| params.data._kind == 'substructure')",
                "struct-grid-standalone": "params.data && params.data._kind == 'standalone'",
                "struct-grid-contention": "params.data && params.data._kind == 'contention'",
            },
            "defaultColDef": {"sortable": False, "resizable": True, "suppressMovable": True},
        },
        className="ag-theme-balham blotter-grid struct-grid",
        getRowId={"function": "params.data._row_id"},
        style={"height": "min(52vh, 560px)", "width": "100%"},
    )


def render_positions_section(account_state, state, pos_view: str = "position",
                             expanded_sids=None) -> html.Div:
    """The holdings section with the By Position | By Structure toggle. Only the
    ACTIVE grid is rendered — an AG-Grid initialised inside a ``display:none``
    container never binds its cellClicked handler, so hiding the inactive grid
    would make its rows dead. The position-modal callbacks tolerate the other
    grid being absent (suppress_callback_exceptions + None-safe nav)."""
    n_positions = len(account_state.positions)
    by_position = pos_view != "structure"
    grid = (_positions_grid(account_state, state) if by_position
            else _structures_grid(account_state, state, expanded_sids))
    return html.Div(className="dd-section", children=[
        html.Div(className="dd-section-head", children=[
            html.H2("Holdings", className="dd-section-title"),
            html.Span(f"{n_positions} positions · full book", className="dd-section-meta"),
            html.Span(summary_line(account_state),
                      className="dd-section-meta dd-actionables-summary"),
            _render_pos_toggle(pos_view),
        ]),
        grid,
    ])
