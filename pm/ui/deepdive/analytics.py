"""Section 3 — Analytics.

Detailed positioning view. Renders already-computed diagnostics/greeks plus the
three new pure aggregations. Sector weights are drawn as utilitarian CSS bars —
no charting dependency (constraint). Everything here READS state; the
aggregations are presentation reframings, not recomputations.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.risk.exposure import (
    economic_exposure_by_sector,
    economic_exposure_by_underlying,
    economic_exposure_missing,
)
from pm.ui.deepdive.aggregations import (
    _fmt_money,
    expiry_ladder,
    long_short_premium_split,
)
from pm.ui.deepdive.bars import bar_row


def _stat(label: str, value: str, sub: Optional[str] = None, cls: str = "") -> html.Div:
    children = [html.Div(label, className="dd-stat-label"),
                html.Div(value, className="dd-stat-value")]
    if sub:
        children.append(html.Div(sub, className="dd-stat-sub"))
    return html.Div(className=f"dd-stat {cls}".strip(), children=children)


# ---- panels ---------------------------------------------------------------

def _premium_panel(account_state) -> html.Div:
    s = long_short_premium_split(account_state)
    short_pct = s["short_share"]
    bar = html.Div(className="dd-split-bar", children=[
        html.Div(className="dd-split-collected",
                 style={"width": f"{(short_pct or 0) * 100:.1f}%"}),
        html.Div(className="dd-split-paid",
                 style={"width": f"{(1 - (short_pct or 0)) * 100:.1f}%"}),
    ]) if s["total"] else None
    return html.Div(className="dd-panel", children=[
        html.H3("Options premium — collected vs paid", className="dd-panel-title"),
        html.Div(className="dd-stat-row", children=[
            _stat("Collected (short)", _fmt_money(s["collected"]),
                  f"{s['n_short']} legs", cls="dd-stat-pos"),
            _stat("Paid (long)", _fmt_money(s["paid"]),
                  f"{s['n_long']} legs", cls="dd-stat-neg"),
            _stat("Net", _fmt_money(s["net"]), s["posture"]),
        ]),
        bar,
        html.Div(s["interpretation"], className="dd-panel-note"),
    ])


def _ladder_panel(account_state) -> html.Div:
    ladder, n_expired = expiry_ladder(account_state)
    header = html.Div(className="dd-ladder-row dd-ladder-head", children=[
        html.Span("Window"),
        html.Span("Contracts"),
        html.Span("Notional (strike)",
                  title="Strike obligation = contracts × 100 × strike"),
    ])
    rows = [header]
    for b in ladder:
        rows.append(html.Div(className="dd-ladder-row", children=[
            html.Span(b["label"], className="dd-ladder-bucket"),
            html.Span(str(b["count"]), className="dd-ladder-count"),
            html.Span(_fmt_money(b["notional"]) if b["notional"] else "—",
                      className="dd-ladder-notional"),
        ]))
    children = [
        html.H3("Expiry ladder", className="dd-panel-title"),
        html.Div("Strike-obligation exposure by expiry window",
                 className="dd-panel-subtitle"),
        html.Div(className="dd-ladder", children=rows),
        html.Div("Notional is the strike obligation (contracts × 100 × strike), "
                 "not market value — driven by position size, shown beside the "
                 "contract count.", className="dd-panel-note"),
    ]
    if n_expired:
        children.append(html.Div(
            f"Expired ({n_expired}) — dead contract(s) still on the book "
            "(stale extract); excluded from every window above.",
            className="dd-panel-note"))
    return html.Div(className="dd-panel", children=children)


def _missing_delta_note(account_state) -> Optional[html.Div]:
    """The economic (delta-$) bars silently skip rows with no delta — when any
    were skipped, say so instead of presenting a partial book as complete."""
    missing = economic_exposure_missing(account_state)
    if not missing["n_rows"]:
        return None
    names = missing["names"]
    shown = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
    return html.Div(
        f"Delta missing on {missing['n_rows']} position(s) across "
        f"{len(names)} name(s) — excluded from these bars: {shown}.",
        className="dd-panel-note")


def _sector_panel(account_state) -> html.Div:
    items = economic_exposure_by_sector(account_state)  # sorted by |delta-$| desc
    max_w = max((abs(r["pct_nav"] or 0) for r in items), default=0)
    bars = [bar_row(r["sector"], r["pct_nav"], max_w) for r in items]
    if not bars:
        bars = [html.Div("No economic exposure to show.", className="dd-empty")]
    diag = getattr(account_state, "diagnostics", None)
    beta = getattr(diag, "weighted_beta", None)
    beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else "—"
    children = [
        html.Div(className="dd-panel-headrow", children=[
            html.H3("Sector breakdown", className="dd-panel-title"),
            html.Span(f"Weighted β {beta_str}", className="dd-beta-chip"),
        ]),
        html.Div("Economic exposure (delta-$) by sector, signed % NAV — options "
                 "included and netted against stock.", className="dd-panel-subtitle"),
        html.Div(className="dd-bars", children=bars),
    ]
    note = _missing_delta_note(account_state)
    if note is not None:
        children.append(note)
    return html.Div(className="dd-panel", children=children)


def _concentration_panel(account_state) -> html.Div:
    top = economic_exposure_by_underlying(account_state)[:5]
    max_w = max((abs(r["pct_nav"] or 0) for r in top), default=0)
    rows = [bar_row(r["symbol"] or "—", r["pct_nav"], max_w) for r in top]
    if not rows:
        rows = [html.Div("No economic exposure.", className="dd-empty")]
    children = [
        html.H3("Top-5 economic concentrations (% NAV)", className="dd-panel-title"),
        html.Div("Largest names by delta-equivalent exposure — options netted "
                 "against stock; cash excluded.", className="dd-panel-subtitle"),
        html.Div(className="dd-bars", children=rows),
    ]
    note = _missing_delta_note(account_state)
    if note is not None:
        children.append(note)
    return html.Div(className="dd-panel", children=children)


def render_analytics_section(account_state) -> html.Div:
    return html.Div(className="dd-section", children=[
        html.Div(className="dd-section-head", children=[
            html.H2("Analytics", className="dd-section-title"),
        ]),
        html.Div(className="dd-analytics-grid", children=[
            _premium_panel(account_state),
            _ladder_panel(account_state),
            _sector_panel(account_state),
            _concentration_panel(account_state),
        ]),
    ])
