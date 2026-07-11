"""Tab 2 header — account picker + one-glance KPI strip.

The picker (static, in the tab layout) scopes the whole page; the KPI strip is
rebuilt per selected account by the populate callback. KPIs read state only:
NAV, cash %, # positions, # options, a net-Greeks one-liner, and per-account
alert tier counts. Detailed Greeks live in Analytics, not here.
"""
from __future__ import annotations

from typing import Optional

from dash import dcc, html

from pm.store.portfolio_state import PortfolioState
from pm.ui.deepdive.actionables import summary_line, tier_counts
from pm.ui.deepdive.aggregations import _fmt_money, book_summary, net_greeks_summary


def account_options(state: Optional[PortfolioState]) -> list[dict]:
    if state is None:
        return []
    return [{"label": a, "value": a} for a in sorted(state.accounts)]


def default_account(state: Optional[PortfolioState]) -> Optional[str]:
    opts = account_options(state)
    return opts[0]["value"] if opts else None


def render_account_picker(state: Optional[PortfolioState],
                          active: Optional[str] = None) -> html.Div:
    """The account selector. Static in the layout; its value scopes the page."""
    opts = account_options(state)
    value = active or default_account(state)
    return html.Div(className="dd-picker-row", children=[
        html.Span("Account", className="dd-picker-label"),
        dcc.Dropdown(
            id="deepdive-account-picker",
            options=opts,
            value=value,
            clearable=False,
            className="dd-picker",
            style={"width": "220px"},
        ),
    ])


def _kpi(label: str, value: str, sub: Optional[str] = None, cls: str = "") -> html.Div:
    children = [html.Div(label, className="dd-kpi-label"),
                html.Div(value, className="dd-kpi-value")]
    if sub:
        children.append(html.Div(sub, className="dd-kpi-sub"))
    return html.Div(className=f"dd-kpi {cls}".strip(), children=children)


def _tier_chips(account_state) -> html.Div:
    counts = tier_counts(account_state)
    chips = []
    for t, cls in ((1, "dd-tier-chip-t1"), (2, "dd-tier-chip-t2"), (3, "dd-tier-chip-t3")):
        chips.append(html.Span(f"{counts.get(t, 0)} T{t}",
                               className=f"dd-tier-chip {cls}"))
    return html.Div(className="dd-kpi dd-kpi-tiers", children=[
        html.Div("Alerts", className="dd-kpi-label"),
        html.Div(className="dd-tier-chip-row", children=chips),
    ])


def render_kpis(account_state) -> html.Div:
    """The one-glance KPI strip for the selected account."""
    bs = book_summary(account_state)
    g = net_greeks_summary(account_state)
    cash_pct = bs["cash_pct"]
    return html.Div(className="dd-kpi-strip", children=[
        _kpi("NAV", _fmt_money(bs["nav"])),
        _kpi("Cash", "—" if cash_pct is None else f"{cash_pct * 100:.1f}%"),
        _kpi("Positions", str(bs["n_positions"]), f"{bs['n_options']} options"),
        _kpi("Net Greeks", g["headline"], g["interpretation"], cls="dd-kpi-greeks"),
        _tier_chips(account_state),
    ])
