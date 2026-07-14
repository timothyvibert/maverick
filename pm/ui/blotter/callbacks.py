"""All Tab-1 + drawer + refresh callbacks.

Registered once per Dash instance via ``register_callbacks(app)``. The UI
reads state through ``state_access`` and never recomputes.
"""
from __future__ import annotations

import dash
from dash import ALL, MATCH, Input, Output, State, ctx, html, no_update

from pm.insight import threshold_catalog as _cat
from pm.insight.pattern_groups import GROUP_ORDER
from pm.store import settings_store as _settings
from pm.ui import state_access as sa
from pm.ui.blotter.grid import (
    apply_account_filter,
    apply_alert_filters,
    build_blotter_columns,
    cell_click_target,
    consolidate_fires_to_rows,
    default_grid_options,
    nav_display,
    sort_rows,
    step_row,
)
from pm.ui.components.status_bar import render_status_bar
from pm.ui.deepdive.header import account_options
from pm.ui.drawers.alert_manager import render_alert_manager_body
from pm.ui.drawers.evidence import render_alerts, _snooze_until
from pm.ui.drawers.signal_sheet import render_signal_sheet


_ALL_TIERS = [1, 2, 3]
_OPEN_CLS = "drawer-root drawer-open"


def _chip_class(base: str, active: bool) -> str:
    return base + (" tier-chip-active" if active else "")


def _filter_rows(rows: list[dict], tiers: list[int]) -> list[dict]:
    tset = set(tiers or [])
    return [r for r in rows if r.get("tier") in tset]


def _underlying_for(state, account: str, position_id: str) -> str:
    """The underlying ticker for a position (for the Tearsheet view)."""
    fires = sa.fires_for_position(state, account, position_id)
    if fires:
        return fires[0].underlying or ""
    pos = sa.position_by_id(state, account, position_id)
    if pos is not None:
        return pos.underlying_symbol or pos.symbol or ""
    return ""


def _render_body(state, account: str, position_id: str, underlying: str, view: str,
                 structure_id=None):
    """Render the drawer body for the active view: Alert = all alerts on the position,
    stacked; Tearsheet = the per-underlying signal sheet; Scanner = the ranked adjustment
    candidates; Payoff = the structure/position economics. Lazy imports keep the drawer
    modules off this module's import path."""
    if view == "tearsheet":
        return render_signal_sheet(account, underlying, state)
    if view == "scanner":
        from pm.ui.drawers.scanner import render_scanner
        return render_scanner(account, position_id=position_id, structure_id=structure_id)
    if view == "payoff":
        from pm.ui.drawers.payoff import render_payoff
        return render_payoff(account, structure_id=structure_id, position_id=position_id)
    return render_alerts(account, position_id, state)


def _dd_grid_rows(state, dd_account, pos_view, expanded):
    """Rebuild the deep-dive Holdings grid's rowData for the picker's account + active
    view, so a mute/restore is reflected there without a reload. Returns no_update when
    no account is in scope (the output is then ignored — the grid may not be mounted)."""
    acc_state = state.accounts.get(dd_account) if (state and dd_account) else None
    if acc_state is None:
        return no_update
    if pos_view == "structure":
        from pm.ui.deepdive.structures_panel import build_structure_rows
        return build_structure_rows(acc_state, state, expanded)
    from pm.ui.deepdive.positions import build_positions_rows
    return build_positions_rows(acc_state, state)


def _visible_rows_for_source(source, blotter_rows, dd_pos_rows):
    """The ordered consolidated rows the modal's prev/next should step through,
    chosen by which grid opened the modal. Defaults to the blotter so Tab 1
    behaviour is unchanged (Tab 2 stamps an explicit ``source``)."""
    if source == "deepdive-positions":
        return dd_pos_rows
    return blotter_rows


def register_callbacks(app: dash.Dash) -> None:

    # ---- Tier filter chips → tier-filter store + chip classes -------------
    @app.callback(
        Output("tier-filter", "data"),
        Output("tier-chip-all", "className"),
        Output("tier-chip-1", "className"),
        Output("tier-chip-2", "className"),
        Output("tier-chip-3", "className"),
        Input("tier-chip-all", "n_clicks"),
        Input("tier-chip-1", "n_clicks"),
        Input("tier-chip-2", "n_clicks"),
        Input("tier-chip-3", "n_clicks"),
        State("tier-filter", "data"),
        prevent_initial_call=True,
    )
    def _on_tier_chip(_a, _1, _2, _3, current):
        tiers = set(current or _ALL_TIERS)
        trig = ctx.triggered_id
        if trig == "tier-chip-all":
            tiers = set(_ALL_TIERS)
        elif trig == "tier-chip-1":
            tiers ^= {1}
        elif trig == "tier-chip-2":
            tiers ^= {2}
        elif trig == "tier-chip-3":
            tiers ^= {3}
        ordered = sorted(tiers)
        all_on = tiers == set(_ALL_TIERS)
        return (
            ordered,
            _chip_class("tier-chip", all_on),
            _chip_class("tier-chip tier-chip-t1", 1 in tiers),
            _chip_class("tier-chip tier-chip-t2", 2 in tiers),
            _chip_class("tier-chip tier-chip-t3", 3 in tiers),
        )

    # ---- Grouping toggle → group-mode store + toggle classes --------------
    @app.callback(
        Output("group-mode", "data"),
        Output("group-account", "className"),
        Output("group-pattern", "className"),
        Input("group-account", "n_clicks"),
        Input("group-pattern", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_grouping(_acc, _pat):
        mode = "pattern" if ctx.triggered_id == "group-pattern" else "account"
        acc_cls = "group-toggle" + (" group-toggle-active" if mode == "account" else "")
        pat_cls = "group-toggle" + (" group-toggle-active" if mode == "pattern" else "")
        return mode, acc_cls, pat_cls

    # ---- Alert-group chips → group-filter store + chip classes ------------
    # Each chip independently toggles whether that group's alerts are shown
    # (all shown by default); the store carries the set of shown groups.
    @app.callback(
        Output("group-filter", "data"),
        Output("group-chip-position", "className"),
        Output("group-chip-market", "className"),
        Output("group-chip-catalyst", "className"),
        Output("group-chip-informational", "className"),
        Input("group-chip-position", "n_clicks"),
        Input("group-chip-market", "n_clicks"),
        Input("group-chip-catalyst", "n_clicks"),
        Input("group-chip-informational", "n_clicks"),
        State("group-filter", "data"),
        prevent_initial_call=True,
    )
    def _on_group_chip(_p, _m, _c, _i, current):
        shown = set(current if current is not None else GROUP_ORDER)
        trig = ctx.triggered_id
        if isinstance(trig, str) and trig.startswith("group-chip-"):
            shown ^= {trig[len("group-chip-"):]}
        ordered = [g for g in GROUP_ORDER if g in shown]
        return (ordered, *[
            _chip_class("tier-chip group-chip", g in shown) for g in GROUP_ORDER
        ])

    # ---- (tier, grouping, account, group, full rows) → grid rows ----------
    # Community has no row grouping, so the Account|Pattern toggle is a
    # server-side re-sort. Slice order: account (row-level) → group (alert-level
    # removal, rebuild, drop-empty) → tier (row-level, on the REBUILT tier) →
    # sort. The account value comes straight from its dropdown (in-memory, no
    # persistence); the shown groups from the group-filter store.
    @app.callback(
        Output("blotter-grid", "columnDefs"),
        Output("blotter-grid", "rowData"),
        Output("blotter-grid", "dashGridOptions"),
        Input("tier-filter", "data"),
        Input("group-mode", "data"),
        Input("blotter-account-picker", "value"),
        Input("group-filter", "data"),
        Input("blotter-all-rows", "data"),
    )
    def _sync_grid(tiers, group_mode, account_sel, shown_groups, all_rows):
        group_mode = group_mode if group_mode in ("account", "pattern") else "account"
        rows = apply_account_filter(all_rows or [], account_sel)
        rows = apply_alert_filters(rows, shown_groups)
        rows = _filter_rows(rows, tiers if tiers is not None else _ALL_TIERS)
        rows = sort_rows(rows, group_mode)
        return build_blotter_columns(), rows, default_grid_options()

    # =======================================================================
    # Modal: open / view-toggle / prev-next. Rows are consolidated (one per
    # position); drawer-state carries {account, position_id, underlying, view}
    # where view ∈ {"alert","tearsheet"}. All write the drawer outputs via
    # allow_duplicate=True. The grid-click handler has ONLY the concrete
    # cellClicked input (a single concrete input — no zero-match wildcards).
    # =======================================================================

    # 1) GRID cell click → open Alert (any column) or Tearsheet (ticker).
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("blotter-grid", "cellClicked"),
        prevent_initial_call=True,
    )
    def _open_from_grid(cell):
        state = sa.get_state()
        if state is None or not cell:
            return no_update, no_update, no_update
        row_id = cell.get("rowId")
        if not row_id or "::" not in row_id:
            return no_update, no_update, no_update          # blank/odd row
        account, _, position_id = row_id.partition("::")    # position_id may contain '|'
        if not position_id:
            return no_update, no_update, no_update
        underlying = _underlying_for(state, account, position_id)
        view = cell_click_target(cell.get("colId"))          # 'tearsheet' on ticker, else 'alert'
        body = _render_body(state, account, position_id, underlying, view)
        return body, _OPEN_CLS, {"view": view, "account": account,
                                 "position_id": position_id, "underlying": underlying,
                                 "source": "blotter",
                                 "structure_id": sa.structure_for_position(state, account, position_id)}

    # 2) View toggle — swaps the body within the open popup's family, keeping the
    #    anchor. Position family: Alert|Tearsheet|Scanner. Payoff family: Payoff|Scanner.
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("view-alert", "n_clicks"),
        Input("view-tearsheet", "n_clicks"),
        Input("view-payoff", "n_clicks"),
        Input("view-scanner", "n_clicks"),
        State("drawer-state", "data"),
        prevent_initial_call=True,
    )
    def _switch_view(_a, _t, _p, _s, drawer_state):
        state = sa.get_state()
        ds = drawer_state or {}
        target = {"view-alert": "alert", "view-tearsheet": "tearsheet",
                  "view-payoff": "payoff", "view-scanner": "scanner"}.get(ctx.triggered_id)
        if state is None or target is None or target == ds.get("view"):
            return no_update, no_update
        # All four tabs are always present on the unified popup, so any button is a valid
        # target; each tab renders from the shared anchor (position_id + optional structure).
        body = _render_body(state, ds.get("account"), ds.get("position_id"),
                            ds.get("underlying", ""), target, structure_id=ds.get("structure_id"))
        return body, {**ds, "view": target}

    # 3) Prev/Next: step POSITION-to-position in the visible (filter+sort)
    #    order (virtualRowData = consolidated rows). View mode persists.
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("drawer-prev", "n_clicks"),
        Input("drawer-next", "n_clicks"),
        State("drawer-state", "data"),
        State("blotter-grid", "virtualRowData"),
        State("deepdive-positions-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
    def _prev_next(_p, _n, drawer_state, blotter_rows, dd_pos_rows):
        state = sa.get_state()
        ds = drawer_state or {}
        # Prev/Next steps positions; the structure modal view has no position nav.
        if state is None or ds.get("view") not in ("alert", "tearsheet"):
            return no_update, no_update, no_update
        # ``or []`` — the source grid may be absent if the view was toggled
        # while a position modal stayed open (the other grid isn't rendered).
        visible_rows = _visible_rows_for_source(
            ds.get("source"), blotter_rows, dd_pos_rows) or []
        direction = "prev" if ctx.triggered_id == "drawer-prev" else "next"
        target = step_row(visible_rows, ds.get("account"), ds.get("position_id"), direction)
        if target is None:
            return no_update, no_update, no_update          # boundary
        account = target.get("_account")
        position_id = target.get("_position_id")
        underlying = target.get("_underlying") or _underlying_for(state, account, position_id)
        view = ds.get("view")
        body = _render_body(state, account, position_id, underlying, view)
        return body, _OPEN_CLS, {"view": view, "account": account,
                                 "position_id": position_id, "underlying": underlying,
                                 "source": ds.get("source", "blotter"),
                                 "structure_id": sa.structure_for_position(state, account, position_id)}

    # ---- Close drawer (close button or overlay click) ---------------------
    @app.callback(
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("drawer-close-btn", "n_clicks"),
        Input("drawer-overlay", "n_clicks"),
        prevent_initial_call=True,
    )
    def _close_drawer(_c, _o):
        return "drawer-root", {"view": None}

    # ---- Nav + toggle presentation: derived from drawer-state -------------
    # Prev/next is visible in BOTH modes; indicator/disabled come from the
    # visible order; the active view's toggle button is highlighted.
    @app.callback(
        Output("drawer-nav", "className"),
        Output("drawer-prev", "disabled"),
        Output("drawer-next", "disabled"),
        Output("drawer-pos", "children"),
        Output("view-alert", "className"),
        Output("view-tearsheet", "className"),
        Output("view-payoff", "className"),
        Output("view-scanner", "className"),
        Input("drawer-state", "data"),
        State("blotter-grid", "virtualRowData"),
        State("deepdive-positions-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
    def _nav_and_toggle(drawer_state, blotter_rows, dd_pos_rows):
        ds = drawer_state or {}
        view = ds.get("view")
        HID = "view-toggle-btn view-toggle-btn-hidden"
        NAV_HID = "drawer-nav drawer-nav-hidden"

        def on(name):
            return "view-toggle-btn" + (" view-toggle-btn-active" if view == name else "")

        # The unified four-tab popup: Alert | Tearsheet | Payoff | Scanner, all present and
        # anchored to the position. Prev/next steps positions on Alert/Tearsheet only.
        if view in ("alert", "tearsheet", "payoff", "scanner"):
            if view in ("alert", "tearsheet"):
                rows = _visible_rows_for_source(ds.get("source"), blotter_rows, dd_pos_rows) or []
                text, prev_dis, next_dis = nav_display(rows, ds.get("account"), ds.get("position_id"))
                nav_cls = "drawer-nav"
            else:
                text, prev_dis, next_dis, nav_cls = "", True, True, NAV_HID
            return (nav_cls, prev_dis, next_dis, text,
                    on("alert"), on("tearsheet"), on("payoff"), on("scanner"))

        # Structure-detail view (Confirm/Reject) or closed: no toggle, no nav.
        return (NAV_HID, True, True, "", HID, HID, HID, HID)

    # ---- Escape-to-close: one-time clientside keydown listener -------------
    # Attaches a guarded document listener once (drawer-root.id fires once on
    # load); on Escape, if the modal is open, clicks the existing close button.
    app.clientside_callback(
        """
        function(_id) {
            if (!window._pmEscBound) {
                window._pmEscBound = true;
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        var root = document.getElementById('drawer-root');
                        if (root && root.className.indexOf('drawer-open') !== -1) {
                            var btn = document.getElementById('drawer-close-btn');
                            if (btn) { btn.click(); }
                        }
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("esc-listener-dummy", "data"),
        Input("drawer-root", "id"),
    )

    # ---- Alert suppress / snooze / restore -----------------------
    # The Mute / Snooze controls (per active alert) and the Muted-footer Restore
    # all write through the shared accessor (state_access.suppress_alert /
    # restore_alert → the store; re-marks the account in place, no recompute), then
    # re-render the modal body (active alerts vs the Muted footer), refresh the
    # blotter row store, and rebuild the active deep-dive grid — so the change shows
    # on both tabs immediately. Pattern-matching ids carry (account, pid, pat); the
    # name keyed on is the fire's underlying.
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("blotter-all-rows", "data", allow_duplicate=True),
        Output("deepdive-positions-grid", "rowData", allow_duplicate=True),
        Input({"type": "sup-mute", "account": ALL, "pid": ALL, "pat": ALL}, "n_clicks"),
        Input({"type": "sup-restore", "account": ALL, "pid": ALL, "pat": ALL}, "n_clicks"),
        Input({"type": "sup-snooze", "account": ALL, "pid": ALL, "pat": ALL}, "value"),
        Input({"type": "sup-date", "account": ALL, "pid": ALL, "pat": ALL}, "date"),
        State("drawer-state", "data"),
        State("deepdive-account-picker", "value"),
        State("pos-view-mode", "data"),
        State("struct-expanded", "data"),
        prevent_initial_call=True,
    )
    def _on_suppress_action(_m, _r, _snz, _dt, drawer_state, dd_account, pos_view, expanded):
        trig = ctx.triggered_id
        state = sa.get_state()
        if not isinstance(trig, dict) or state is None:
            return no_update, no_update, no_update
        # Ignore spurious fires when controls (re)mount with n_clicks 0/None, an
        # empty dropdown, or a cleared date.
        trigval = (ctx.triggered[0] if ctx.triggered else {}).get("value")
        if trigval in (None, 0, ""):
            return no_update, no_update, no_update
        ttype, account = trig.get("type"), trig.get("account")
        pid, pat = trig.get("pid"), trig.get("pat")
        # The fire (active OR muted — both stay in acc.fires) gives the underlying to
        # key on and the trace/rationale to capture.
        fire = sa.fire_by_id(state, account, pid, pat)
        if fire is None:
            return no_update, no_update, no_update
        name = fire.underlying

        if ttype == "sup-mute":
            sa.suppress_alert(account, name, pat, trace=fire.trace, rationale=fire.rationale)
        elif ttype == "sup-restore":
            sa.restore_alert(account, name, pat)
        elif ttype == "sup-snooze":
            if trigval == "pick":               # reveal handled by _reveal_snooze_date
                return no_update, no_update, no_update
            sa.suppress_alert(account, name, pat, suppressed_until=_snooze_until(int(trigval)),
                              trace=fire.trace, rationale=fire.rationale)
        elif ttype == "sup-date":
            sa.suppress_alert(account, name, pat, suppressed_until=trigval,
                              trace=fire.trace, rationale=fire.rationale)
        else:
            return no_update, no_update, no_update

        ds = drawer_state or {}
        blotter_rows = consolidate_fires_to_rows(sa.all_fires(state), state)
        dd_rows = _dd_grid_rows(state, dd_account, pos_view, expanded)
        # Re-render the drawer body only when the Alert view is showing; a suppress must
        # never overwrite another tab (e.g. the scanner) sharing the drawer body.
        body = (render_alerts(ds.get("account", account), ds.get("position_id", pid), state)
                if ds.get("view") == "alert" else no_update)
        return body, blotter_rows, dd_rows

    # Reveal the inline date picker when its sibling Snooze dropdown selects
    # "Pick a date…" (MATCH pairs each picker to its own dropdown).
    @app.callback(
        Output({"type": "sup-date", "account": MATCH, "pid": MATCH, "pat": MATCH}, "style"),
        Input({"type": "sup-snooze", "account": MATCH, "pid": MATCH, "pat": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def _reveal_snooze_date(value):
        return {"display": "inline-block"} if value == "pick" else {"display": "none"}

    # ---- Alert Manager -----------------------------------
    # A SEPARATE modal from the per-alert drawer: its own root, open/close, tab
    # strip and Escape listener, so it never cross-toggles the drawer. The
    # Suppressed tab restores via the SAME state_access.restore_alert the modal
    # footer uses. Active classes are derived from which tab is shown. Two open
    # routes: the Alert Manager button (Suppressed tab) and the status bar's
    # load-notes cluster (straight onto Load notes — the readable, copyable
    # warnings list the hover tooltip can't provide).
    _AM_OPEN = "drawer-root drawer-open"

    def _am_tab_classes(tab):
        return ("view-toggle-btn" + (" view-toggle-btn-active" if tab == "suppressed" else ""),
                "view-toggle-btn" + (" view-toggle-btn-active" if tab == "thresholds" else ""),
                "view-toggle-btn" + (" view-toggle-btn-active" if tab == "loadnotes" else ""))

    @app.callback(
        Output("alert-manager-root", "className"),
        Output("alert-manager-body", "children"),
        Output("am-tab-suppressed", "className"),
        Output("am-tab-thresholds", "className"),
        Output("am-tab-loadnotes", "className"),
        Input("alert-manager-open-btn", "n_clicks"),
        Input("status-load-notes-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _open_alert_manager(_n, _notes_n):
        # The status-bar host is re-rendered on every load, recreating the notes
        # button with n_clicks=0 — guard the spurious (re)creation fire.
        if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
            return (no_update,) * 5
        tab = "loadnotes" if ctx.triggered_id == "status-load-notes-btn" else "suppressed"
        s_cls, t_cls, n_cls = _am_tab_classes(tab)
        return _AM_OPEN, render_alert_manager_body(tab), s_cls, t_cls, n_cls

    @app.callback(
        Output("alert-manager-root", "className", allow_duplicate=True),
        Input("alert-manager-close-btn", "n_clicks"),
        Input("alert-manager-overlay", "n_clicks"),
        prevent_initial_call=True,
    )
    def _close_alert_manager(_c, _o):
        return "drawer-root"

    @app.callback(
        Output("alert-manager-body", "children", allow_duplicate=True),
        Output("am-tab-suppressed", "className", allow_duplicate=True),
        Output("am-tab-thresholds", "className", allow_duplicate=True),
        Output("am-tab-loadnotes", "className", allow_duplicate=True),
        Input("am-tab-suppressed", "n_clicks"),
        Input("am-tab-thresholds", "n_clicks"),
        Input("am-tab-loadnotes", "n_clicks"),
        prevent_initial_call=True,
    )
    def _switch_am_tab(_s, _t, _n):
        tab = {"am-tab-thresholds": "thresholds",
               "am-tab-loadnotes": "loadnotes"}.get(ctx.triggered_id, "suppressed")
        s_cls, t_cls, n_cls = _am_tab_classes(tab)
        return render_alert_manager_body(tab), s_cls, t_cls, n_cls

    @app.callback(
        Output("alert-manager-body", "children", allow_duplicate=True),
        Output("blotter-all-rows", "data", allow_duplicate=True),
        Output("deepdive-positions-grid", "rowData", allow_duplicate=True),
        Input({"type": "am-restore", "account": ALL, "name": ALL, "pat": ALL}, "n_clicks"),
        State("deepdive-account-picker", "value"),
        State("pos-view-mode", "data"),
        State("struct-expanded", "data"),
        prevent_initial_call=True,
    )
    def _restore_from_manager(_n, dd_account, pos_view, expanded):
        trig = ctx.triggered_id
        state = sa.get_state()
        if not isinstance(trig, dict) or state is None:
            return no_update, no_update, no_update
        if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
            return no_update, no_update, no_update
        sa.restore_alert(trig["account"], trig["name"], trig["pat"])
        body = render_alert_manager_body("suppressed")
        blotter_rows = consolidate_fires_to_rows(sa.all_fires(state), state)
        dd_rows = _dd_grid_rows(state, dd_account, pos_view, expanded)
        return body, blotter_rows, dd_rows

    # ---- Thresholds tab: edit dials -> persist -> recompute -> re-paint --
    # Apply / Reset all / per-row Reset all flow through one callback. Each MUTATES the
    # persisted overrides (settings_store) then re-derives the alert set over the
    # already-loaded book via state_access.recompute_thresholds — the persist-then-
    # recompute apply path. The recompute happens in the single state owner (engine +
    # structure fires + suppression marking re-run over loaded data), NOT in the UI
    # layer, and it makes no Bloomberg call: a dial edit changes which fires the engine
    # PRODUCES, not the market data they read — so the old full-reload route (a complete
    # BBG re-pull per dial change) is reserved for the no-state-loaded fallback.
    # Outputs target only always-present components (the manager body/tabs, the status
    # host, the blotter store, the deep-dive refresh tick, the load sentinel). The
    # deep-dive grid is NOT written directly: dcc.Tabs only mounts the active tab, so on
    # the Blotter tab that grid is absent and a direct write would make Dash reject the
    # whole callback. Bumping deepdive-refresh-tick (a store) drives the Tab-2 repopulate
    # callback instead — the same mechanism the Refresh buttons use.
    @app.callback(
        Output("alert-manager-body", "children", allow_duplicate=True),
        Output("am-tab-suppressed", "className", allow_duplicate=True),
        Output("am-tab-thresholds", "className", allow_duplicate=True),
        Output("am-tab-loadnotes", "className", allow_duplicate=True),
        Output("status-bar-host", "children", allow_duplicate=True),
        Output("blotter-all-rows", "data", allow_duplicate=True),
        Output("deepdive-refresh-tick", "data", allow_duplicate=True),
        Output("bbg-load-sentinel", "children", allow_duplicate=True),
        Input("am-thr-apply", "n_clicks"),
        Input("am-thr-reset-all", "n_clicks"),
        Input({"type": "thr-reset", "name": ALL}, "n_clicks"),
        State({"type": "thr-input", "name": ALL}, "value"),
        State({"type": "thr-input", "name": ALL}, "id"),
        State("deepdive-refresh-tick", "data"),
        prevent_initial_call=True,
    )
    def _apply_thresholds(_apply, _reset_all, _row_resets, values, ids, tick):
        noop = (no_update,) * 8
        trig = ctx.triggered_id
        # Ignore the spurious fire when the inputs/buttons are (re)created with n_clicks 0.
        if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
            return noop

        if trig == "am-thr-reset-all":
            _settings.clear_all()
        elif isinstance(trig, dict) and trig.get("type") == "thr-reset":
            _settings.clear_override(trig["name"])
        elif trig == "am-thr-apply":
            for val, cid in zip(values or [], ids or []):
                name = (cid or {}).get("name")
                if name is None or not _cat.is_editable(name):
                    continue
                if val is None:                       # blank input -> revert to default
                    _settings.clear_override(name)
                    continue
                try:
                    # Editing a dial back to its default clears the override (keeps the
                    # store minimal and 'set vs default' honest); else persist it.
                    if abs(float(val) - _cat.default_ui(name)) < 1e-9:
                        _settings.clear_override(name)
                    else:
                        _settings.set_override(name, val)   # validates + clamps via catalog
                except (ValueError, TypeError):
                    continue
        else:
            return noop

        # Persist-then-recompute: re-evaluate the current book under the new
        # thresholds over already-loaded state (no Bloomberg). Only when nothing
        # is loaded yet does Apply fall back to the full reload route.
        try:
            new_state = sa.recompute_thresholds()
            if new_state is None:
                new_state = sa.reload_state(reuse_extract=True)
        except Exception as exc:
            return (no_update, no_update, no_update, no_update,
                    html.Div(f"Apply failed: {exc}", className="status-left status-empty"),
                    no_update, no_update, "")
        s_cls, t_cls, n_cls = _am_tab_classes("thresholds")
        body = render_alert_manager_body("thresholds")          # reseed inputs from persisted
        rows = consolidate_fires_to_rows(sa.all_fires(new_state), new_state)
        return (body, s_cls, t_cls, n_cls, render_status_bar(new_state), rows,
                (tick or 0) + 1, "")

    # The manager's own Escape listener, bound to its own id (independent of the
    # drawer's). On Escape it closes only the manager, only when the manager is open.
    app.clientside_callback(
        """
        function(_id) {
            if (!window._pmAmEscBound) {
                window._pmAmEscBound = true;
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        var root = document.getElementById('alert-manager-root');
                        if (root && root.className.indexOf('drawer-open') !== -1) {
                            var btn = document.getElementById('alert-manager-close-btn');
                            if (btn) { btn.click(); }
                        }
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("am-esc-dummy", "data"),
        Input("alert-manager-root", "id"),
    )

    # ---- Initial load OR Refresh BBG → load state + status + rows + Tab 2 --
    # One callback backs both the one-shot post-render load and the manual
    # Refresh BBG button. It owns the (re)load, so it also bumps the deep-dive
    # tick (race-free: the tick changes only after reload_state() completes and
    # has written the singleton, which the Tab-2 populate callback then reads)
    # and refreshes the account-picker options/value. The spinner is driven by
    # ``bbg-load-sentinel`` (a dcc.Loading child): it shows while this callback
    # runs and clears when it returns. Existing data stays visible throughout —
    # only the outputs swap, on completion — so refresh is non-blocking.
    @app.callback(
        Output("status-bar-host", "children"),
        Output("blotter-all-rows", "data"),
        Output("blotter-account-picker", "options"),
        Output("deepdive-refresh-tick", "data"),
        Output("deepdive-account-picker", "options"),
        Output("deepdive-account-picker", "value"),
        Output("bbg-load-sentinel", "children"),
        Input("initial-load", "n_intervals"),
        Input("refresh-button", "n_clicks"),
        Input("refresh-acct-button", "n_clicks"),
        State("deepdive-account-picker", "value"),
        State("deepdive-refresh-tick", "data"),
        prevent_initial_call=True,
    )
    def _load_or_refresh(_n_intervals, _n_clicks_bbg, _n_clicks_acct, picker_value, tick):
        # Refresh BBG re-pulls market data on the current extract; Refresh Acct Data
        # (and the initial load) read the latest extract file from the data dir.
        reuse_extract = ctx.triggered_id == "refresh-button"
        try:
            new_state = sa.reload_state(reuse_extract=reuse_extract)  # handles prev=None (first load)
        except Exception as exc:  # surface failure in the status bar
            return (html.Div(f"Load failed: {exc}", className="status-left status-empty"),
                    no_update, no_update, no_update, no_update, no_update, "")
        rows = consolidate_fires_to_rows(sa.all_fires(new_state), new_state)
        opts = account_options(new_state)
        valid = [o["value"] for o in opts]
        # Preserve the user's selection on a manual refresh; default to the
        # first account on the initial load (picker_value is None then).
        value = picker_value if picker_value in valid else (valid[0] if valid else None)
        # The blotter account filter offers the accounts that actually have rows.
        blotter_accts = [{"label": a, "value": a}
                         for a in sorted({r["account"] for r in rows if r.get("account")})]
        return (render_status_bar(new_state), rows, blotter_accts,
                (tick or 0) + 1, opts, value, "")
