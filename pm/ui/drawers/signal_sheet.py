"""Signal sheet drawer — everything the engine knows about one underlying.

Header summary, six collapsible signal
groups (A–F) with per-signal trace expanders, the composite breakdown
table, and the open fires for this underlying in this account.

Reads only from PortfolioState: groups A–D/F from ``signals[underlying]``,
group E from ``position_signals[position_id]`` (per held position).
"""
from __future__ import annotations

from typing import Any, Optional

from dash import html

from pm.core.composite_score import COMPOSITE_WEIGHTS
from pm.store.portfolio_state import PortfolioState
from pm.insight.signal_library import SignalValue
from pm.store.suppression_store import is_active
from pm.ui import state_access as sa
from pm.ui.drawers.trace_table import render_trace


_TIER_CLS = {1: "drawer-tier-1", 2: "drawer-tier-2", 3: "drawer-tier-3"}


# ---------------------------------------------------------------------------
# Small formatters
# ---------------------------------------------------------------------------

def _f2(v: Any, suffix: str = "", decimals: int = 2) -> str:
    fv = sa.coerce_float(v)
    if fv is None:
        return "—"
    return f"{fv:,.{decimals}f}{suffix}"


def _pct_num(v: Any, decimals: int = 2) -> str:
    """For BBG percent-scale numbers already in percent units (e.g. 1.53)."""
    fv = sa.coerce_float(v)
    if fv is None:
        return "—"
    return f"{fv:+.{decimals}f}%"


def _header_item(label: str, value: str) -> html.Div:
    return html.Div(className="sheet-hdr-item", children=[
        html.Span(label, className="sheet-hdr-label"),
        html.Span(value, className="sheet-hdr-value"),
    ])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header(account: str, underlying: str, state: PortfolioState,
                   sigdict: Optional[dict]) -> html.Div:
    snap = sa.snapshot_row_for_underlying(state, account, underlying) or {}
    name = snap.get("security_name")
    name = "" if sa.is_missing(name) else str(name)
    sector = snap.get("GICS_SECTOR_NAME")
    sector = "—" if sa.is_missing(sector) else str(sector)

    analyst_sig = (sigdict or {}).get("analyst_rating_and_target")
    analyst_disp = analyst_sig.display if (analyst_sig is not None and not analyst_sig.stale) else "—"

    earn = (sigdict or {}).get("days_to_earnings")
    earn_disp = earn.display if (earn is not None and not earn.stale) else "—"

    positions = sa.positions_for_underlying(state, account, underlying)
    leg_descs = []
    for p in positions:
        if p.asset_class == "option":
            right = (p.right or "").title()
            strike = f"${p.strike:g}" if p.strike is not None else ""
            qty = f"×{int(p.quantity)}" if p.quantity is not None else ""
            leg_descs.append(f"{right} {strike} {qty}".strip())
        else:
            qty = f"×{int(p.quantity)}" if p.quantity is not None else ""
            leg_descs.append(f"{p.asset_class} {qty}".strip())
    pos_summary = f"{len(positions)} leg(s): " + (", ".join(leg_descs) if leg_descs else "—")

    title = html.Div(className="sheet-hdr-title", children=[
        html.Span(underlying, className="sheet-hdr-ticker"),
        html.Span(name, className="sheet-hdr-name"),
        html.Span(account, className="sheet-hdr-account"),
    ])
    grid = html.Div(className="sheet-hdr-grid", children=[
        _header_item("Sector", sector),
        _header_item("Spot", _f2(snap.get("PX_LAST"))),
        _header_item("1D", _pct_num(snap.get("CHG_PCT_1D"))),
        _header_item("IV 3M", _f2(snap.get("3MTH_IMPVOL_100.0%MNY_DF"), suffix="%", decimals=1)),
        _header_item("Beta", _f2(snap.get("BETA_ADJ_OVERRIDABLE"))),
        _header_item("Earnings", earn_disp),
        _header_item("Analyst", analyst_disp),
    ])
    return html.Div(className="sheet-header", children=[
        title, grid,
        html.Div(pos_summary, className="sheet-hdr-positions"),
    ])


# ---------------------------------------------------------------------------
# Signal rows / groups
# ---------------------------------------------------------------------------

# Signals whose value is a multi-value string ("1D · 5D · …" / "rating ·
# target · upside") render on a single wide line (name + full-width value,
# no mid-token wrap) rather than cramped in the narrow value column.
_WIDE_VALUE_SIGNALS = {
    "return_horizons", "analyst_rating_and_target", "street_consensus_rating_and_target",
}


def _signal_row(display_name: str, sv: Optional[SignalValue]) -> html.Details:
    if sv is None:
        return html.Details(className="sheet-signal sheet-absent", children=[
            html.Summary(className="sheet-signal-summary", children=[
                html.Span(display_name, className="sheet-signal-name"),
                html.Span("—", className="sheet-signal-value"),
                html.Span("not computed", className="sheet-signal-interp"),
            ]),
        ])
    stale = bool(sv.stale)
    wide = sv.signal_id in _WIDE_VALUE_SIGNALS
    summary_cls = "sheet-signal-summary" + (" sheet-signal-summary-wide" if wide else "")
    summary_children = [
        html.Span(display_name, className="sheet-signal-name"),
        html.Span(sv.display or "—", className="sheet-signal-value"),
    ]
    if not wide:   # wide rows drop the interp column to keep the value on one line
        summary_children.append(
            html.Span(sv.interpretation or "", className="sheet-signal-interp"))
    return html.Details(
        className="sheet-signal" + (" sheet-stale" if stale else ""),
        children=[
            html.Summary(className=summary_cls, children=summary_children),
            html.Div(render_trace(sv.trace), className="sheet-signal-trace"),
        ],
    )


def _group_section(title: str, signal_defs: list, sigdict: dict) -> html.Details:
    rows = [_signal_row(disp, sigdict.get(sid)) for sid, disp in signal_defs]
    return html.Details(open=True, className="sheet-group", children=[
        html.Summary(title, className="sheet-group-summary"),
        html.Div(rows, className="sheet-group-body"),
    ])


def _group_e_section(account: str, underlying: str, state: PortfolioState) -> html.Details:
    """Group E is per-position — one labeled sub-block per held position."""
    positions = sa.positions_for_underlying(state, account, underlying)
    title, defs = sa.POSITION_GROUP
    blocks = []
    for p in positions:
        psig = sa.position_signals_for(state, account, p.position_id) or {}
        if p.asset_class == "option":
            right = (p.right or "").title()
            strike = f"${p.strike:g}" if p.strike is not None else ""
            label = f"{right} {strike}".strip()
        else:
            label = p.asset_class
        rows = [_signal_row(disp, psig.get(sid)) for sid, disp in defs]
        blocks.append(html.Div(className="sheet-e-block", children=[
            html.Div(label, className="sheet-e-position"),
            html.Div(rows, className="sheet-group-body"),
        ]))
    if not blocks:
        blocks = [html.Div("No held positions.", className="trace-muted")]
    return html.Details(open=True, className="sheet-group", children=[
        html.Summary(title, className="sheet-group-summary"),
        html.Div(blocks),
    ])


# ---------------------------------------------------------------------------
# Composite breakdown (group F)
# ---------------------------------------------------------------------------

def _composite_section(sigdict: dict) -> html.Details:
    sv = sigdict.get("composite_score")
    header = html.Summary("F — Composite", className="sheet-group-summary")
    if sv is None or sv.stale or not isinstance(sv.value, dict):
        return html.Details(open=True, className="sheet-group", children=[
            header,
            html.Div("Composite score unavailable.", className="trace-muted"),
        ])
    val = sv.value
    components = val.get("components", {}) or {}
    # A real <table> (table-layout:fixed) so values align under their headers;
    # Raw/Weight/Contribution are right-aligned numeric in both th and td.
    body_rows = []
    for name, c in components.items():
        weight = COMPOSITE_WEIGHTS.get(name)
        body_rows.append(html.Tr([
            html.Td(name.replace("_", " ").title(), className="ct-name"),
            html.Td(_f2(c.get("raw"), decimals=1), className="ct-num"),
            html.Td(_f2(weight, decimals=2), className="ct-num"),
            html.Td(_f2(c.get("weighted"), decimals=2), className="ct-num"),
        ]))
    table = html.Table(className="composite-tbl", children=[
        html.Thead(html.Tr([
            html.Th("Component", className="ct-name"),
            html.Th("Raw", className="ct-num"),
            html.Th("Weight", className="ct-num"),
            html.Th("Contribution", className="ct-num"),
        ])),
        html.Tbody(body_rows),
        html.Tfoot(html.Tr([
            html.Td(f"Total — {val.get('label', '')}", className="ct-name ct-total"),
            html.Td("", className="ct-num"),
            html.Td("", className="ct-num"),
            html.Td(_f2(val.get("total"), decimals=1), className="ct-num ct-total"),
        ])),
    ])
    return html.Details(open=True, className="sheet-group", children=[
        header,
        html.Div(table, className="composite-tbl-wrap"),
        html.Details(className="sheet-signal", children=[
            html.Summary("Trace", className="sheet-signal-summary"),
            html.Div(render_trace(sv.trace), className="sheet-signal-trace"),
        ]),
    ])


# ---------------------------------------------------------------------------
# Open fires
# ---------------------------------------------------------------------------

def _open_fires_section(account: str, underlying: str, state: PortfolioState) -> html.Div:
    # Active alerts only: a muted alert is not an "open fire" — it lives in
    # the Alert view's Muted footer, where it can be restored.
    fires = [f for f in sa.fires_for_underlying(state, account, underlying) if is_active(f)]
    fires = sorted(fires, key=lambda f: f.tier)
    if not fires:
        body = [html.Div("No open fires on this name.", className="trace-muted")]
    else:
        # Static display rows — navigation between alerts is via the modal's
        # prev/next + Alert/Tearsheet toggle, not by clicking these.
        body = [
            html.Div(
                className="sheet-fire-row",
                children=[
                    html.Span(f"T{f.tier}", className=f"sheet-fire-tier {_TIER_CLS.get(f.tier, '')}"),
                    html.Span(f.pattern_id, className="sheet-fire-pid"),
                    html.Span(f.label, className="sheet-fire-label"),
                ],
            )
            for f in fires
        ]
    return html.Div(className="sheet-group", children=[
        html.Div("Open fires", className="sheet-group-summary sheet-static-summary"),
        html.Div(body, className="sheet-fires"),
    ])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_signal_sheet(account: str, underlying: str, state: PortfolioState) -> html.Div:
    sigdict = sa.signals_for_underlying(state, account, underlying) or {}

    sections = [_render_header(account, underlying, state, sigdict)]

    # Groups A, B, C, D (per-underlying).
    for title, defs in (sa.GROUP_A, sa.GROUP_B, sa.GROUP_C, sa.GROUP_D):
        sections.append(_group_section(title, defs, sigdict))
    # Group E (per-position).
    sections.append(_group_e_section(account, underlying, state))
    # Group F (composite breakdown).
    sections.append(_composite_section(sigdict))
    # Open fires.
    sections.append(_open_fires_section(account, underlying, state))

    return html.Div(className="drawer-content signal-sheet", children=sections)
