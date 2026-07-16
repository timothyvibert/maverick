"""Alert Manager — the book-wide review/reverse surface for alert governance.

A discrete modal (separate from the per-alert drawer) opened from the top-right
control cluster. Four tabs: **Suppressed** (active suppressions AND per-fire
acknowledgements, restorable), **Patterns** (the persisted per-pattern on/off
toggles — a toggled-off pattern's fires are hidden-but-counted, never dropped),
**Thresholds** (the editable alert-sensitivity dials), and **Load notes** (every
load/ingestion/market-data warning, readable and copyable — the status bar shows
only the headline). All are dense ``html.Table``s — not a second AG-Grid —
matching the signal-sheet / trace design language. Restore here uses the *same*
``state_access`` write paths as the modal's Muted footer; there is no second
mechanism. The Thresholds tab edits the persisted overrides via ``settings_store``
and applies them with a persist-then-recompute (write the override, then
``state_access.recompute_thresholds`` re-derives the alert set over the
already-loaded book — engine + structure fires + suppression marking, no
Bloomberg call, no extract re-read). The recompute lives in the single state
owner, not the UI layer. See ``pm/insight/threshold_catalog.py`` for the dials.
Pattern toggles and acknowledgements are pure marking flips
(``state_access.set_pattern_enabled`` / ``acknowledge_alert``) — instant, no
engine re-run at all.

Days-active (today − created_at) is the deliberate staleness cue; a muted or
acknowledged alert whose condition moves materially is re-surfaced by the
load-path material-change pass and shows that state here (see
``suppression_store`` / ``alert_governance``).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from dash import dcc, html

from pm.insight import threshold_catalog as cat
from pm.insight.pattern_groups import GROUP_LABELS, PATTERN_GROUP, all_pattern_meta
from pm.store import alert_governance, settings_store, suppression_store
from pm.ui import state_access as sa


def _days_active(created_at: Optional[str], today: date) -> int:
    """Whole days since the suppression was set (>= 0). Defensive against a missing or
    unparseable timestamp."""
    if not created_at:
        return 0
    try:
        started = datetime.fromisoformat(created_at).date()
    except ValueError:
        return 0
    return max(0, (today - started).days)


def _live_mark(state, record: dict):
    """The live ``Fire.suppression`` mark for a stored suppression, or None. Lets the tab
    show a re-surfaced row's third state by reading what the load-path pass computed."""
    if state is None:
        return None
    acc = state.accounts.get(record["account"])
    if acc is None:
        return None
    for f in acc.fires:
        if f.underlying == record["name"] and f.pattern_id == record["pattern_id"]:
            return f.suppression
    return None


def _fmt_delta(v) -> str:
    if isinstance(v, str):       # event dates ride as ISO strings
        return v
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return str(v)


def _state_text(record: dict, mark=None) -> str:
    if getattr(mark, "kind", None) == "resurfaced":
        return f"Re-surfaced — moved {_fmt_delta(mark.captured_value)} → {_fmt_delta(mark.current_value)}"
    until = record.get("suppressed_until")
    return f"Snoozed until {until}" if until else "Suppressed"


def _restore_id(record: dict) -> dict:
    return {"type": "am-restore", "account": record["account"],
            "name": record["name"], "pat": record["pattern_id"]}


def _ack_live_fire(state, rec: dict):
    """The live fire an acknowledgement record governs, or None."""
    if state is None:
        return None
    acc = state.accounts.get(rec["account"])
    if acc is None:
        return None
    for f in acc.fires:
        if f.pattern_id == rec["pattern_id"] and alert_governance.fire_key(f) == rec["fire_key"]:
            return f
    return None


def _unack_id(rec: dict) -> dict:
    return {"type": "am-unack", "account": rec["account"],
            "pat": rec["pattern_id"], "key": rec["fire_key"]}


def render_suppressed_tab(today: Optional[date] = None) -> html.Div:
    """The active suppressions AND per-fire acknowledgements, sorted/grouped by
    account. Empty → a neutral note."""
    today = today or date.today()
    meta = all_pattern_meta()
    state = sa.get_state()      # to read each record's live mark (re-surfaced state)
    records = sorted(suppression_store.active_suppressions(today).values(),
                     key=lambda r: (r["account"], r["name"], r["pattern_id"]))
    acks = sorted(alert_governance.acknowledgements().values(),
                  key=lambda r: (r["account"], r["fire_key"], r["pattern_id"]))
    if not records and not acks:
        return html.Div("No suppressed alerts", className="am-empty")

    header = html.Tr([html.Th(h, className="am-th") for h in
                      ("Account", "Name", "Alert type", "State", "Days active", "")])
    body_rows = []
    for r in records:
        pid = r["pattern_id"]
        alert_type = meta.get(pid, (pid, None))[0]
        # captured_rationale surfaced as a row tooltip — what the alert looked like
        # when it was muted, so a long-lived suppression can be eyeballed.
        rationale = (r.get("captured_rationale") or "").strip()
        mark = _live_mark(state, r)
        resurfaced = getattr(mark, "kind", None) == "resurfaced"
        state_cls = "am-state am-state-resurfaced" if resurfaced else "am-state"
        body_rows.append(html.Tr(
            className="am-row",
            title=rationale or None,
            children=[
                html.Td(r["account"], className="am-acct"),
                html.Td(r["name"], className="am-name"),
                html.Td(alert_type, className="am-type"),
                html.Td(_state_text(r, mark), className=state_cls),
                html.Td(str(_days_active(r.get("created_at"), today)), className="am-days"),
                html.Td(html.Button("Restore", id=_restore_id(r), n_clicks=0,
                                    className="alert-action-btn am-restore-btn")),
            ],
        ))
    for r in acks:
        pid = r["pattern_id"]
        alert_type = meta.get(pid, (pid, None))[0]
        rationale = (r.get("captured_rationale") or "").strip()
        fire = _ack_live_fire(state, r)
        mark = getattr(fire, "suppression", None)
        resurfaced = getattr(mark, "kind", None) == "resurfaced"
        state_txt = (_state_text(r, mark) if resurfaced else "Acknowledged")
        state_cls = "am-state am-state-resurfaced" if resurfaced else "am-state"
        display_name = getattr(fire, "underlying", None) or r["fire_key"]
        body_rows.append(html.Tr(
            className="am-row",
            title=rationale or None,
            children=[
                html.Td(r["account"], className="am-acct"),
                html.Td(display_name, className="am-name"),
                html.Td(alert_type, className="am-type"),
                html.Td(state_txt, className=state_cls),
                html.Td(str(_days_active(r.get("acknowledged_at"), today)), className="am-days"),
                html.Td(html.Button("Restore", id=_unack_id(r), n_clicks=0,
                                    className="alert-action-btn am-restore-btn")),
            ],
        ))
    return html.Table(className="am-table", children=[
        html.Thead(header), html.Tbody(body_rows)])


def _pat_toggle_id(pattern_id: str) -> dict:
    return {"type": "am-pat-toggle", "pat": pattern_id}


def render_patterns_tab() -> html.Div:
    """The persisted per-pattern on/off toggles, one per live pattern. A toggled-off
    pattern's fires are COLLAPSED TO MUTED, not dropped: the row shows the live
    count of fires currently hidden by the toggle ('3 muted'), and toggling back
    on restores them in place — a marking flip, no engine re-run, no reload."""
    meta = all_pattern_meta()
    disabled = alert_governance.disabled_patterns()
    state = sa.get_state()
    counts = alert_governance.disabled_fire_counts(state) if state is not None else {}

    def _pat_sort(pid: str) -> int:
        try:
            return int(pid.lstrip("P"))
        except ValueError:
            return 999

    header = html.Tr([html.Th(h, className="am-th") for h in
                      ("Pattern", "Alert type", "Tier", "Group", "Status", "")])
    body_rows = []
    for pid in sorted(meta, key=_pat_sort):
        name, tier = meta[pid]
        off = pid in disabled
        n_muted = counts.get(pid, 0)
        status = (f"Off — {n_muted} muted" if (off and n_muted) else
                  ("Off" if off else "On"))
        body_rows.append(html.Tr(className="am-row", children=[
            html.Td(pid, className="am-acct"),
            html.Td(name, className="am-type"),
            html.Td(f"T{tier}", className="am-days"),
            html.Td(GROUP_LABELS.get(PATTERN_GROUP.get(pid, ""), "—"), className="am-name"),
            html.Td(status, className=("am-state am-state-off" if off else "am-state")),
            html.Td(html.Button("Turn on" if off else "Turn off",
                                id=_pat_toggle_id(pid), n_clicks=0,
                                className="alert-action-btn am-pat-toggle-btn")),
        ]))
    note = html.Div(
        "Turning a pattern off hides its alerts from every active surface but keeps "
        "them counted here — nothing is dropped, and turning it back on restores "
        "them instantly. Off is final until you flip it: a hidden alert does not "
        "re-surface on a material move while its pattern is off.",
        className="am-thr-note")
    table = html.Table(className="am-table am-patterns-table",
                       children=[html.Thead(header), html.Tbody(body_rows)])
    return html.Div([note, table])


def _thr_input_id(name: str) -> dict:
    return {"type": "thr-input", "name": name}


def _thr_reset_id(name: str) -> dict:
    return {"type": "thr-reset", "name": name}


def _fmt_num(ui_value: float, is_int: bool) -> str:
    return str(int(round(ui_value))) if is_int else f"{ui_value:g}"


def render_thresholds_tab(status: Optional[str] = None) -> html.Div:
    """The editable alert-sensitivity dials, grouped by pattern. Each row seeds
    its input from the persisted override (if any) else the PatternConfig default; the
    Default column always shows the default so 'set vs default' is legible. Apply persists
    the dirty rows and re-runs the engine (persist-then-recompute); Reset clears an override.
    ``status`` is the per-apply outcome line (rejections) rendered in the actions row.

    Pure read — ``settings_store.get_overrides`` never materializes the DB when nothing is
    persisted yet, so opening the tab on a clean store is side-effect-free."""
    overrides = settings_store.get_overrides()        # {name: native} — presence == overridden
    header = html.Tr([html.Th(h, className="am-th") for h in
                      ("Threshold", "Value", "Default", "")])
    body_rows = []
    for pid, pname, specs in cat.grouped_by_pattern():
        body_rows.append(html.Tr(className="am-thr-grouprow", children=[
            html.Td(f"{pid} · {pname}", colSpan=4, className="am-thr-group")]))
        for s in specs:
            overridden = s.name in overrides
            eff_ui = cat.to_ui(s.name, overrides[s.name]) if overridden else cat.default_ui(s.name)
            body_rows.append(html.Tr(className="am-row am-thr-row", children=[
                html.Td(s.label, className="am-thr-label"),
                html.Td(className="am-thr-valcell", children=[
                    # No HTML min/max: an out-of-range entry must still commit so the
                    # server-side catalog can REJECT it with a message (a number input
                    # with max silently refuses out-of-range values, which would look
                    # like a no-op). The catalog is the authoritative validator.
                    dcc.Input(
                        id=_thr_input_id(s.name), type="number", value=eff_ui,
                        step=(1 if s.is_int else "any"), debounce=True,
                        className="am-thr-input" + (" am-thr-input-set" if overridden else "")),
                    html.Span(s.unit, className="am-thr-unit"),
                ]),
                html.Td(f"{_fmt_num(cat.default_ui(s.name), s.is_int)} {s.unit}".strip(),
                        className="am-thr-default"),
                html.Td(html.Button("Reset", id=_thr_reset_id(s.name), n_clicks=0,
                                    disabled=not overridden,
                                    className="alert-action-btn am-thr-reset-btn")),
            ]))
    table = html.Table(className="am-table am-thr-table",
                       children=[html.Thead(header), html.Tbody(body_rows)])
    actions = html.Div(className="am-thr-actions", children=[
        html.Div(status, className="am-thr-status") if status else
        html.Div("Applying re-runs the engine on the current book and re-paints the alerts.",
                 className="am-thr-note"),
        html.Div(className="am-thr-buttons", children=[
            html.Button("Reset all", id="am-thr-reset-all", n_clicks=0,
                        className="alert-action-btn am-thr-resetall-btn"),
            html.Button("Apply", id="am-thr-apply", n_clicks=0,
                        className="alert-action-btn am-thr-apply-btn"),
        ]),
    ])
    return html.Div(className="am-thr-wrap", children=[table, actions])


def render_loadnotes_tab() -> html.Div:
    """Every load-time warning — ingestion notes, market-data misses, insight
    skip notes — as a dense, natively-selectable table (the status bar shows
    only the headline; this is the readable, copyable full list). Urgent ⚠
    notes sort first and render amber. Reads the live singleton at open time
    (all_warnings is per-load, not per-interaction)."""
    from pm.ingest.extract_loader import URGENT_FLAG

    state = sa.get_state()
    notes = list(getattr(state, "all_warnings", []) or []) if state else []
    if not notes:
        return html.Div("No load notes — the last load was clean.", className="am-empty")
    urgent = [n for n in notes if n.lstrip().startswith(URGENT_FLAG)]
    rest = [n for n in notes if not n.lstrip().startswith(URGENT_FLAG)]
    header = html.Tr([html.Th(h, className="am-th") for h in ("", "Note")])
    body_rows = []
    for n_ in urgent + rest:
        is_urgent = n_.lstrip().startswith(URGENT_FLAG)
        body_rows.append(html.Tr(className="am-row", children=[
            html.Td("⚠" if is_urgent else "", className="am-note-flag"),
            html.Td(n_, className="am-note-text am-note-urgent" if is_urgent
                    else "am-note-text"),
        ]))
    count_line = html.Div(
        f"{len(notes)} note(s) from the last load"
        + (f" — {len(urgent)} urgent" if urgent else ""),
        className="am-thr-note")
    table = html.Table(className="am-table am-notes-table",
                       children=[html.Thead(header), html.Tbody(body_rows)])
    return html.Div([count_line, table])


def render_alert_manager_body(tab: str = "suppressed",
                              today: Optional[date] = None,
                              thr_status: Optional[str] = None) -> html.Div:
    if tab == "thresholds":
        inner = render_thresholds_tab(status=thr_status)
    elif tab == "loadnotes":
        inner = render_loadnotes_tab()
    elif tab == "patterns":
        inner = render_patterns_tab()
    else:
        inner = render_suppressed_tab(today)
    return html.Div(className="am-body-inner", children=[inner])
