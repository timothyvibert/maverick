"""Alert view — every alert (Fire) on a consolidated position, stacked.

Rows are now one-per-position, so a position can carry multiple alerts. The
Alert view shows ALL of them: each alert's pattern + tier badge + client-ready
rationale + full audit trace, in sequential sections separated by dividers.
The Alert/Tearsheet toggle lives on the modal header (see shell/callbacks), so
there is no in-content "view signal sheet" button anymore.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from dash import dcc, html

from pm.insight.patterns import Fire
from pm.store.portfolio_state import PortfolioState
from pm.store.suppression_store import is_active
from pm.ui import state_access as sa
from pm.ui.blotter.grid import format_position_descriptor
from pm.ui.drawers.trace_table import render_trace
from pm.core import clock


_TIER_WORD = {1: "T1 · Act today", 2: "T2 · Worth raising", 3: "T3 · FYI"}

# Snooze presets — "1 month" = 30 days (snooze precision is immaterial). The final
# option reveals an inline date picker for an exact date.
_SNOOZE_PRESETS = [
    {"label": "Snooze 1 week", "value": "7"},
    {"label": "Snooze 2 weeks", "value": "14"},
    {"label": "Snooze 1 month", "value": "30"},
    {"label": "Pick a date…", "value": "pick"},
]


def _snooze_until(preset_days: int, today: Optional[date] = None) -> str:
    """The ISO date a snooze of ``preset_days`` runs through (inclusive)."""
    return ((today or clock.today()) + timedelta(days=preset_days)).isoformat()


def _ctl_id(kind: str, fire: Fire) -> dict:
    """Pattern-matching id for a per-alert control. Carries the position so the
    callback can fetch the exact fire (for the baseline capture); the suppression
    itself is keyed (account, underlying, pattern_id)."""
    return {"type": kind, "account": fire.account,
            "pid": fire.position_id, "pat": fire.pattern_id}


def _alert_controls(fire: Fire) -> html.Div:
    """The discrete Acknowledge + Mute + Snooze▾ cluster for one active alert,
    right-aligned in the header next to the tier badge. Acknowledge sets aside
    THIS one fire (it comes back by itself if its condition moves materially);
    muting/snoozing is scoped to this pattern on this name — every other alert
    keeps firing."""
    scope = f"{fire.pattern_name} on {fire.underlying}"
    return html.Div(className="alert-actions", children=[
        html.Button("Acknowledge", id=_ctl_id("sup-ack", fire), n_clicks=0,
                    className="alert-action-btn alert-ack-btn",
                    title=f"Set aside this ONE alert (reviewed, no action needed now). "
                          f"Unlike Mute it is per-position and it re-surfaces on its own "
                          f"if the condition moves materially."),
        html.Button("Mute", id=_ctl_id("sup-mute", fire), n_clicks=0,
                    className="alert-action-btn alert-mute-btn",
                    title=f"Mute {scope} — permanently, until you restore it. "
                          f"Every other alert keeps firing."),
        dcc.Dropdown(
            id=_ctl_id("sup-snooze", fire), options=_SNOOZE_PRESETS,
            placeholder="Snooze ▾", clearable=False, searchable=False,
            className="alert-snooze-dd",
        ),
        dcc.DatePickerSingle(
            id=_ctl_id("sup-date", fire), placeholder="date",
            display_format="YYYY-MM-DD", className="alert-snooze-date",
            style={"display": "none"},
        ),
    ])


def _alert_section(fire: Fire, state: PortfolioState) -> html.Div:
    tier_cls = f"drawer-tier-{fire.tier}"
    return html.Div(className="alert-section", children=[
        html.Div(className="drawer-header-main", children=[
            html.Span(fire.pattern_id, className="drawer-pattern-id"),
            html.Span(fire.pattern_name, className="drawer-pattern-name"),
            html.Span(_TIER_WORD.get(fire.tier, f"T{fire.tier}"),
                      className=f"drawer-tier-badge {tier_cls}"),
            _alert_controls(fire),
        ]),
        html.Div(className="drawer-section", children=[
            html.Div("Rationale", className="drawer-section-label"),
            # Markdown render so the templates' **bold** emphasis shows as bold
            # rather than literal asterisks. The .drawer-rationale CSS (incl. its
            # strong/b rule) styles the rendered output.
            dcc.Markdown(fire.rationale, className="drawer-rationale"),
        ]),
        html.Div(className="drawer-section", children=[
            html.Div("Audit trace", className="drawer-section-label"),
            render_trace(fire.trace),
        ]),
    ])


def _muted_state_text(fire: Fire) -> str:
    sup = fire.suppression
    if sup is not None and sup.kind == "snoozed" and sup.until:
        return f"Snoozed until {sup.until}"
    if sup is not None and sup.kind == "acknowledged":
        return "Acknowledged"
    if sup is not None and sup.kind == "disabled":
        return "Pattern off (Alert Manager)"
    return "Suppressed"


def _muted_line(fire: Fire) -> html.Div:
    return html.Div(className="muted-line", children=[
        html.Span(fire.pattern_id, className="muted-line-pid"),
        html.Span(fire.pattern_name, className="muted-line-name"),
        html.Span(_muted_state_text(fire), className="muted-line-state"),
        html.Button("Restore", id=_ctl_id("sup-restore", fire), n_clicks=0,
                    className="alert-action-btn muted-restore-btn",
                    title=f"Restore {fire.pattern_name} on {fire.underlying}"),
    ])


def _muted_footer(muted: list[Fire]) -> Optional[html.Details]:
    """The collapsed 'Muted (N) ▾' footer — this position's muted/snoozed alerts,
    each restorable. Default collapsed; None when nothing is muted."""
    if not muted:
        return None
    return html.Details(className="muted-footer", children=[
        html.Summary(f"Muted ({len(muted)})", className="muted-footer-summary"),
        html.Div([_muted_line(f) for f in muted], className="muted-footer-lines"),
    ])


def render_alerts(account: str, position_id: str, state: PortfolioState) -> html.Div:
    """Render this position's alerts. Active alerts stack as full sections (each with
    its Mute / Snooze controls); muted/snoozed alerts move to a collapsed
    'Muted (N)' footer where they can be restored. Header = position descriptor +
    active-alert count."""
    fires = sa.fires_for_position(state, account, position_id)
    # Dedupe defensively (one section per distinct alert), preserving order.
    seen, distinct = set(), []
    for f in fires:
        if f.pattern_id not in seen:
            seen.add(f.pattern_id)
            distinct.append(f)
    # Split via the shared predicate so a re-surfaced fire (muted but materially moved)
    # shows as ACTIVE here, not in the Muted footer — consistent with every other surface.
    active = [f for f in distinct if is_active(f)]
    muted = [f for f in distinct if not is_active(f)]
    position = sa.position_by_id(state, account, position_id)
    descriptor = (format_position_descriptor(position)
                  if position is not None else position_id)
    n = len(active)

    header = html.Div(className="alerts-header", children=[
        html.Span(account, className="drawer-account"),
        html.Span(" · "),
        html.Span(descriptor, className="drawer-position"),
        html.Span(f"{n} alert{'s' if n != 1 else ''}", className="alerts-count"),
    ])

    sections = []
    for i, fire in enumerate(active):
        if i > 0:
            sections.append(html.Hr(className="alert-divider"))
        sections.append(_alert_section(fire, state))
    if not sections:
        sections = [html.Div("No active alerts on this position.", className="trace-muted")]

    footer = _muted_footer(muted)
    children = [header, *sections]
    if footer is not None:
        children.append(footer)
    return html.Div(className="drawer-content evidence-drawer", children=children)
