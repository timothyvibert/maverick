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


def render_status_bar(state: Optional[PortfolioState]) -> html.Div:
    """Render the left side of the status strip. Before data has loaded
    (state is None) this shows a neutral 'Loading…' line; the async load
    swaps in the populated line when it completes."""
    if state is None:
        return html.Div("Loading portfolio…", className="status-left status-empty")

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

    # Load-time ingestion notes (header aliasing, missing/optional columns,
    # skipped rows). Pure read of state.all_warnings — no recompute. Shown amber
    # when a load-bearing column or high-impact optional is flagged (urgent
    # prefix); the full list is in the hover title.
    notes = list(getattr(state, "all_warnings", []) or [])
    if notes:
        urgent = [n for n in notes if n.lstrip().startswith(URGENT_FLAG)]
        lead = (urgent or notes)[0]
        extra = len(notes) - 1
        label = lead + (f"  (+{extra} more)" if extra > 0 else "")
        cls = "status-item status-load-notes" + (" status-load-urgent" if urgent else "")
        items.append(html.Span(label, className=cls, title="\n".join(notes)))

    return html.Div(items, className="status-left")
