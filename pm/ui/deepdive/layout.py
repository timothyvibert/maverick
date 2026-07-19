"""Tab 2 layout — account picker + five stacked sections.

The picker is static; each section lives in an id'd host div that the populate
callback rebuilds per selected account. Hosts are rendered eagerly for the
default account so the first paint is correct no matter how ``dcc.Tabs`` mounts
inline children; the callback then drives picker changes / refresh / tab switch.
"""
from __future__ import annotations

import logging
from typing import Optional

from dash import html

logger = logging.getLogger(__name__)

from pm.store.portfolio_state import PortfolioState
from pm.ui.deepdive.header import (
    default_account,
    render_account_picker,
    render_kpis,
)
from pm.ui.deepdive.positions import render_positions_section
from pm.ui.deepdive.risk_cockpit import render_risk_section
from pm.ui.deepdive.trade_insights import render_trade_insights_section
from pm.ui.deepdive.trades import render_trades_section


def render_deepdive_sections(state: Optional[PortfolioState], account: Optional[str],
                             pos_view: str = "position", expanded_sids=None) -> dict:
    """Build the children for each host, keyed by host id. Shared by the layout
    (eager first paint) and the populate callback (re-paint on change). The
    Holdings host carries both the By Position and By Structure grids (toggled).
    When state/account is missing (pre async-load), each host shows a loading
    placeholder so the populate callback has a target to fill."""
    acc_state = state.accounts.get(account) if (state and account) else None
    if acc_state is None:
        empty = html.Div("Loading…", className="dd-empty")
        return {
            "deepdive-kpi": empty,
            "deepdive-positions": empty,
            "deepdive-risk": empty,
            "deepdive-trade-insights": empty,
            "deepdive-trades": empty,
        }
    return {
        "deepdive-kpi": _guarded(lambda: render_kpis(acc_state), "Account KPIs"),
        "deepdive-positions": _guarded(
            lambda: render_positions_section(acc_state, state, pos_view, expanded_sids),
            "Holdings"),
        "deepdive-risk": _guarded(lambda: render_risk_section(acc_state, state), "Risk"),
        "deepdive-trade-insights": _guarded(
            lambda: render_trade_insights_section(acc_state), "Trade-history insights"),
        "deepdive-trades": _guarded(lambda: render_trades_section(acc_state), "Recent trades"),
    }


def _guarded(builder, label: str):
    """Per-section isolation: the populate callback repaints all five hosts in
    one shot, so one section's raise would otherwise freeze the whole tab
    behind a silent error (debug is off). A failed section renders an honest
    error line; the other four render normally."""
    try:
        return builder()
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s section failed to render", label)
        return html.Div(f"{label} failed to render ({type(exc).__name__}).",
                        className="dd-empty risk-block-error")


def render_deepdive_tab(state: Optional[PortfolioState]) -> html.Div:
    """Always renders the full structure — picker + the id'd host divs — even
    before data loads (state is None). The hosts then exist as callback targets
    so the async load / picker / refresh can fill them in place; the picker
    options + value are set by the load callback once accounts are known."""
    account = default_account(state)  # None when no state yet
    sections = render_deepdive_sections(state, account)
    return html.Div(className="deepdive-tab", children=[
        render_account_picker(state, account),
        html.Div(id="deepdive-kpi", className="dd-kpi-host",
                 children=sections["deepdive-kpi"]),
        html.Div(id="deepdive-positions", children=sections["deepdive-positions"]),
        # Risk — the consolidated scenario / exposures / obligations /
        # concentration section (also the header KPI's anchor target).
        html.Div(id="deepdive-risk", children=sections["deepdive-risk"]),
        # Trade-history insights — the client-profile section.
        html.Div(id="deepdive-trade-insights",
                 children=sections["deepdive-trade-insights"]),
        html.Div(id="deepdive-trades", children=sections["deepdive-trades"]),
    ])
