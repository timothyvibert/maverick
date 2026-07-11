"""Tab 2 layout — account picker + five stacked sections.

The picker is static; each section lives in an id'd host div that the populate
callback rebuilds per selected account. Hosts are rendered eagerly for the
default account so the first paint is correct no matter how ``dcc.Tabs`` mounts
inline children; the callback then drives picker changes / refresh / tab switch.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.store.portfolio_state import PortfolioState
from pm.ui.deepdive.analytics import render_analytics_section
from pm.ui.deepdive.exposure import render_exposure_section
from pm.ui.deepdive.scenario import render_scenario_section
from pm.ui.deepdive.header import (
    default_account,
    render_account_picker,
    render_kpis,
)
from pm.ui.deepdive.positions import render_positions_section
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
            "deepdive-exposure": empty,
            "deepdive-scenario": empty,
            "deepdive-analytics": empty,
            "deepdive-trade-insights": empty,
            "deepdive-trades": empty,
        }
    return {
        "deepdive-kpi": render_kpis(acc_state),
        "deepdive-positions": render_positions_section(acc_state, state, pos_view, expanded_sids),
        "deepdive-exposure": render_exposure_section(acc_state),
        "deepdive-scenario": render_scenario_section(acc_state, state),
        "deepdive-analytics": render_analytics_section(acc_state),
        "deepdive-trade-insights": render_trade_insights_section(acc_state),
        "deepdive-trades": render_trades_section(acc_state),
    }


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
        html.Div(id="deepdive-analytics", children=sections["deepdive-analytics"]),
        html.Div(id="deepdive-exposure", children=sections["deepdive-exposure"]),
        html.Div(id="deepdive-scenario", children=sections["deepdive-scenario"]),
        # Trade-history insights — the client-profile section.
        html.Div(id="deepdive-trade-insights",
                 children=sections["deepdive-trade-insights"]),
        html.Div(id="deepdive-trades", children=sections["deepdive-trades"]),
    ])
