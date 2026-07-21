"""The thin top status strip for the app shell — the LEFT content only.

One line tall, maximum density. Reads everything from the already-loaded
PortfolioState — no recompute. The Refresh BBG button + loading spinner are a
persistent cluster owned by the shell (so they survive the async load / refresh
that replaces this content), not part of what this function returns.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.ingest.extract_loader import URGENT_FLAG
from pm.store.portfolio_state import PortfolioState
from pm.store.suppression_store import is_active
from pm.ui import state_access as sa

# Urgent ⚠ notes each get their own visible chip up to this many; the rest fold
# into the "+N more" counter (all of them readable on the Load notes tab).
_MAX_URGENT_CHIPS = 3


def render_status_bar(state: Optional[PortfolioState]) -> html.Div:
    """Render the left side of the status strip. Before data has loaded
    (state is None) this shows a neutral 'Loading…' line; the async load
    swaps in the populated line when it completes."""
    if state is None:
        # Cold start: name what is actually happening and how long it takes, so
        # a multi-second Bloomberg pull reads as progress, not a hung app. The
        # two click-target chips ride along hidden so their callback Inputs
        # resolve even before the first load (see the nonexistent-Input note
        # below).
        return html.Div([
            html.Span(
                "Loading — reading the latest extract and pulling Bloomberg market "
                "data (typically well under a minute)…"),
            html.Button("", id="status-muted-patterns-btn", n_clicks=0,
                        style={"display": "none"}),
            html.Button("", id="status-load-notes-btn", n_clicks=0,
                        style={"display": "none"}),
            html.Button("", id="status-skips-btn", n_clicks=0,
                        style={"display": "none"}),
        ], className="status-left status-empty")

    # Active alerts only: a suppressed/snoozed fire is excluded from every
    # headline count, so muting an alert drops the strip's totals exactly as it drops
    # the blotter row. The Alert Manager (days-active) is the muted-review surface, so
    # no muted count belongs here.
    fires = [f for f in sa.all_fires(state) if is_active(f)]
    n_t1 = sum(1 for f in fires if f.tier == 1)
    n_t2 = sum(1 for f in fires if f.tier == 2)
    n_t3 = sum(1 for f in fires if f.tier == 3)
    n_positions = sum(len(a.positions) for a in state.accounts.values())
    # Flagged positions = consolidated blotter rows (one per position with ≥1 active
    # alert) — explains why the table shows fewer rows than fires.
    n_flagged = len({(f.account, f.position_id) for f in fires})

    bbg_cls = "status-bbg-on" if state.bloomberg_ok else "status-bbg-off"
    bbg_label = "BBG" if state.bloomberg_ok else "BBG offline"

    items = [
        html.Span(f"Extract {state.extract.extract_ts:%Y-%m-%d %H:%M}",
                  className="status-item status-strong"),
        html.Span(f"{len(state.accounts)} accounts", className="status-item"),
        html.Span(f"{n_positions} positions", className="status-item"),
        html.Span(className="status-item", children=[
            html.Span(f"{len(fires)} fires", className="status-strong"),
            html.Span(f" across {n_flagged} positions", className="status-muted"),
            html.Span(f" · {n_t1} T1", className="status-tier-1"),
            html.Span(f" · {n_t2} T2", className="status-tier-2"),
            html.Span(f" · {n_t3} T3", className="status-tier-3"),
        ]),
        html.Span(className="status-item", children=[
            html.Span("●", className=bbg_cls), html.Span(f" {bbg_label}"),
        ]),
        html.Span(f"Refreshed {state.loaded_at:%H:%M:%S}",
                  className="status-item status-muted", id="status-refreshed"),
    ]

    # Pattern-toggle honesty cue: alerts hidden by a persisted per-pattern
    # off-switch are counted here, never silently dropped. The chip is a click
    # target opening the Alert Manager's Patterns tab (where each pattern shows
    # its own muted count and can be turned back on). ALWAYS rendered (hidden
    # when nothing is off): the open callback takes it as an Input, and a Dash
    # Input whose component is absent from the layout kills the whole callback
    # client-side — the component must exist even when it has nothing to say.
    from pm.store.alert_governance import disabled_fire_counts, disabled_patterns
    disabled = disabled_patterns()
    counts = disabled_fire_counts(state) if disabled else {}
    n_hidden = sum(counts.values())
    label = (f"{len(disabled)} pattern{'s' if len(disabled) != 1 else ''} off"
             + (f" · {n_hidden} muted" if n_hidden else "")) if disabled else ""
    detail = ", ".join(f"{counts.get(p, 0)} muted {p}" for p in sorted(disabled))
    items.append(html.Button(
        label, id="status-muted-patterns-btn", n_clicks=0,
        className="status-item status-muted-patterns",
        style=({} if disabled else {"display": "none"}),
        title=(f"{detail}\n\nClick to review or turn patterns back on."
               if disabled else ""),
    ))

    # Alert-coverage honesty cue: patterns the engine could NOT evaluate because
    # a required signal was missing/stale (the stale-skip records). Without it
    # the fire count silently understates under a market-data outage — the blind
    # spot is quantified here, amber when Bloomberg is the reason. Click opens
    # the Alert Manager's Load notes tab (the full [insight] skip lines).
    # ALWAYS rendered, hidden when every pattern evaluated (same
    # nonexistent-Input hazard as the two chips below).
    skips = getattr(state, "insight_skips", {}) or {}
    n_not_eval = sum(v.get("n_not_evaluated", 0) for v in skips.values())
    skip_label = ""
    skip_title = ""
    if n_not_eval:
        noun = "alert" if n_not_eval == 1 else "alerts"
        skip_label = f"{n_not_eval} {noun} not evaluated"
        if not state.bloomberg_ok:
            skip_label += " — market data missing"
        detail_lines = [
            f"{acct}: {g['pattern']} on {g['n_names']} name(s) — {g['signal']} unavailable"
            for acct, v in sorted(skips.items()) for g in v.get("gaps", [])
        ]
        skip_title = ("\n".join(detail_lines)
                      + "\n\nClick to open the full list on the Load notes tab.")
    items.append(html.Button(
        skip_label, id="status-skips-btn", n_clicks=0,
        className="status-item status-skips"
        + (" status-skips-urgent" if (n_not_eval and not state.bloomberg_ok) else ""),
        style=({} if n_not_eval else {"display": "none"}),
        title=skip_title,
    ))

    # Load-time notes (header aliasing, missing/optional columns, skipped rows,
    # market-data + insight warnings). Pure read of state.all_warnings. Urgent
    # notes (the ⚠ prefix — unresolved names with MV, missing load-bearing
    # columns) are each promoted to their own visible amber chip, never hidden
    # behind a truncated lead; the whole cluster is a click target that opens
    # the Alert Manager's Load notes tab, where the full list is readable and
    # copyable (the hover title stays as a convenience, not the only access).
    # ALWAYS rendered, hidden when the load was clean — the open callback takes
    # it as an Input, and an absent Input component kills the callback (the
    # same nonexistent-Input hazard the muted-patterns chip hit).
    notes = list(getattr(state, "all_warnings", []) or [])
    chips: list = []
    if notes:
        urgent = [n for n in notes if n.lstrip().startswith(URGENT_FLAG)]
        shown = 0
        for u in urgent[:_MAX_URGENT_CHIPS]:
            chips.append(html.Span(u, className="status-load-note-chip status-load-urgent"))
            shown += 1
        if not urgent:
            chips.append(html.Span(notes[0], className="status-load-note-chip"))
            shown = 1
        extra = len(notes) - shown
        if extra > 0:
            chips.append(html.Span(f"+{extra} more", className="status-load-more"))
    items.append(html.Button(
        chips, id="status-load-notes-btn", n_clicks=0,
        className="status-item status-load-notes",
        style=({} if notes else {"display": "none"}),
        title=("\n".join(notes) + "\n\nClick to open the full, copyable list.") if notes else "",
    ))

    return html.Div(items, className="status-left")
