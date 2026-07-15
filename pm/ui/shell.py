"""The two-tab app shell: status bar, tabs, the shared right-side drawer,
and the persistent dcc.Store components.

Both tabs (Morning Blotter, Account Deep Dive) are built here from their
layouts. The shell renders immediately with whatever state exists (None at
cold start); a one-shot ``initial-load`` Interval then triggers the data load
(see ``blotter.callbacks``), with a spinner next to the persistent Refresh BBG
button while it runs.
"""
from __future__ import annotations

from typing import Optional

from dash import dcc, html

from pm.insight.pattern_groups import GROUP_ORDER
from pm.store.portfolio_state import PortfolioState
from pm.ui import state_access as sa
from pm.ui.blotter.grid import consolidate_fires_to_rows
from pm.ui.blotter.layout import render_blotter_tab
from pm.ui.components.status_bar import render_status_bar
from pm.ui.deepdive.layout import render_deepdive_tab


def _drawer_root() -> html.Div:
    """The shared fixed-size centered modal.

    Header bar (pinned): Alert|Tearsheet view toggle + prev/next position nav
    (both visible in both modes) + close. The body scrolls inside the fixed
    box. Backdrop click / X / Escape all dismiss it.
    """
    return html.Div(id="drawer-root", className="drawer-root", children=[
        html.Div(id="drawer-overlay", className="drawer-overlay", n_clicks=0),
        html.Div(className="drawer-panel", children=[
            html.Div(className="drawer-headerbar", children=[
                html.Div(className="drawer-headerbar-left", children=[
                    # Segmented view toggle. The visible buttons depend on which popup
                    # family is open (set by _nav_and_toggle): a position popup shows
                    # Alert | Tearsheet | Scanner; a payoff popup shows Payoff | Scanner.
                    # DOM order keeps each family's visible set contiguous and correctly
                    # ordered once the others are hidden.
                    html.Div(className="view-toggle", children=[
                        html.Button("Alert", id="view-alert", n_clicks=0,
                                    className="view-toggle-btn"),
                        html.Button("Tearsheet", id="view-tearsheet", n_clicks=0,
                                    className="view-toggle-btn"),
                        html.Button("Payoff", id="view-payoff", n_clicks=0,
                                    className="view-toggle-btn view-toggle-btn-hidden"),
                        html.Button("Scanner", id="view-scanner", n_clicks=0,
                                    className="view-toggle-btn view-toggle-btn-hidden"),
                    ]),
                    # Prev/next position nav — visible in both modes when open.
                    html.Div(id="drawer-nav", className="drawer-nav drawer-nav-hidden",
                             children=[
                                 html.Button("‹ Prev", id="drawer-prev", n_clicks=0,
                                             className="drawer-nav-btn", disabled=True),
                                 html.Span("", id="drawer-pos", className="drawer-nav-pos"),
                                 html.Button("Next ›", id="drawer-next", n_clicks=0,
                                             className="drawer-nav-btn", disabled=True),
                             ]),
                ]),
                html.Button("✕", id="drawer-close-btn", n_clicks=0,
                            className="drawer-close-btn"),
            ]),
            html.Div(id="drawer-body"),
        ]),
        # Dummy output target for the one-time Escape keydown listener.
        dcc.Store(id="esc-listener-dummy"),
    ])


def _alert_manager_root() -> html.Div:
    """The Alert Manager modal — a SEPARATE centered modal from the per-alert drawer.

    It reuses the same centered-box / overlay CSS (``drawer-root`` / ``drawer-panel``)
    but has its own id, its own open/close callback and its own Escape listener, so it
    never cross-toggles the drawer. Header = a Suppressed | Thresholds | Load notes
    tab strip + ✕; the body is rendered on open (the status bar's notes cluster opens
    straight onto Load notes).
    """
    return html.Div(id="alert-manager-root", className="drawer-root", children=[
        html.Div(id="alert-manager-overlay", className="drawer-overlay", n_clicks=0),
        html.Div(className="drawer-panel am-panel", children=[
            html.Div(className="drawer-headerbar", children=[
                html.Div(className="view-toggle am-tabs", children=[
                    html.Button("Suppressed", id="am-tab-suppressed", n_clicks=0,
                                className="view-toggle-btn view-toggle-btn-active"),
                    html.Button("Patterns", id="am-tab-patterns", n_clicks=0,
                                className="view-toggle-btn"),
                    html.Button("Thresholds", id="am-tab-thresholds", n_clicks=0,
                                className="view-toggle-btn"),
                    html.Button("Load notes", id="am-tab-loadnotes", n_clicks=0,
                                className="view-toggle-btn"),
                ]),
                html.Button("✕", id="alert-manager-close-btn", n_clicks=0,
                            className="drawer-close-btn"),
            ]),
            html.Div(id="alert-manager-body", className="am-body"),
        ]),
        # Dummy output for the manager's own one-time Escape listener.
        dcc.Store(id="am-esc-dummy"),
    ])


def build_shell(state: Optional[PortfolioState]) -> html.Div:
    initial_rows = consolidate_fires_to_rows(sa.all_fires(state), state) if state else []

    return html.Div(className="pm-shell", children=[
        # Persistent stores (outside the tabs so they survive interactions).
        dcc.Store(id="blotter-all-rows", data=initial_rows),
        dcc.Store(id="tier-filter", data=[1, 2, 3]),
        dcc.Store(id="group-mode", data="account"),
        # Blotter alert-group slice: the groups currently shown (all by
        # default). Account/type slices live on their dropdowns' own value.
        dcc.Store(id="group-filter", data=list(GROUP_ORDER)),
        dcc.Store(id="drawer-state", data={"view": None}),
        # Bumped by the load/refresh callback so Tab 2 repopulates from the
        # freshly-loaded state.
        dcc.Store(id="deepdive-refresh-tick", data=0),
        # One-shot trigger: fires once shortly after the page mounts, kicking
        # off the (post-render) data load so the UI is reachable immediately.
        dcc.Interval(id="initial-load", interval=300, n_intervals=0, max_intervals=1),
        # Tab-2 Holdings table: By Position vs By Structure view, and the set of
        # expanded structure ids (in-grid leg expansion).
        dcc.Store(id="pos-view-mode", data="position"),
        dcc.Store(id="struct-expanded", data=[]),

        # Status row: a replaceable left host (swapped by the load callback) +
        # a persistent right cluster (Refresh BBG button + loading spinner) that
        # survives reloads so the spinner can show during them.
        html.Div(className="status-bar", children=[
            html.Div(id="status-bar-host", className="status-left-host",
                     children=render_status_bar(state)),
            html.Div(className="status-right", children=[
                dcc.Loading(
                    id="bbg-loading", type="circle", color="#6b6b6b",
                    className="bbg-loading", children=html.Div(id="bbg-load-sentinel"),
                ),
                html.Button("Refresh Acct Data", id="refresh-acct-button", n_clicks=0,
                            className="status-refresh-btn",
                            title="Re-read the latest extract file from the data directory "
                                  "and re-run the pipeline (picks up a newer file)."),
                html.Button("Refresh BBG", id="refresh-button", n_clicks=0,
                            className="status-refresh-btn",
                            title="Re-pull market data on the current extract "
                                  "(does not switch to a newer file)."),
                html.Button("Alert Manager", id="alert-manager-open-btn", n_clicks=0,
                            className="status-refresh-btn",
                            title="Review and restore suppressed / snoozed alerts."),
            ]),
        ]),

        dcc.Tabs(id="pm-tabs", value="tab-blotter", className="pm-tabs", children=[
            dcc.Tab(label="Morning Blotter", value="tab-blotter",
                    className="pm-tab", selected_className="pm-tab-selected",
                    children=render_blotter_tab(state)),
            dcc.Tab(label="Account Deep Dive", value="tab-account",
                    className="pm-tab", selected_className="pm-tab-selected",
                    children=render_deepdive_tab(state)),
        ]),

        _drawer_root(),
        _alert_manager_root(),
    ])
