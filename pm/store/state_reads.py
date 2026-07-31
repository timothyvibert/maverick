"""Pure read helpers over the loaded state model.

Shared by the UI's state accessor and the scanner service — value coercion
and position / structure lookups over ``PortfolioState`` / ``AccountState``.
Pure reads: no Bloomberg, no mutation, no singleton. ``pm.ui.state_access``
re-exports these names, so existing callers of that module are untouched.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

if TYPE_CHECKING:
    from pm.ingest.position_builder import Position
    from pm.store.portfolio_state import PortfolioState


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def coerce_float(v: Any) -> Optional[float]:
    if is_missing(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def position_by_id(
    state: PortfolioState, account: str, position_id: str,
) -> Optional[Position]:
    acc = state.accounts.get(account)
    if acc is None:
        return None
    for p in acc.positions:
        if p.position_id == position_id:
            return p
    return None


def structure_for_position(state, account: str, position_id: str):
    """The non-rejected structure whose legs include this position (confirmed preferred
    over proposed), or None. Lets a popup opened on an alert/holding price the enclosing
    covered structure in its Payoff/Scanner tabs, falling back to the standalone leg."""
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    hits = [s for s in (getattr(acc, "structures", None) or [])
            if getattr(s, "status", None) != "rejected"
            and any(getattr(lg, "position_id", None) == position_id for lg in (s.legs or []))]
    if not hits:
        return None
    hits.sort(key=lambda s: 0 if getattr(s, "status", None) == "confirmed" else 1)
    return hits[0].structure_id
