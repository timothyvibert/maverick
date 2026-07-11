"""Tier-count summary for an account's alerts.

The standalone Actionables table was merged into the Positions grid (its fires
now live in that grid's Alerts column, reachable via the shared modal). What
remains here is the pure tier-count summary used by the KPI strip and the
Positions section header — ``tier_counts`` and ``summary_line``.
"""
from __future__ import annotations

from pm.store.suppression_store import is_active

# T1 = act today, T2 = worth raising, T3 = idea/FYI.
_TIER_WORD = {1: "act-today", 2: "worth-raising", 3: "idea"}


def tier_counts(account_state) -> dict:
    """{1: n_t1, 2: n_t2, 3: n_t3} for the account's ACTIVE fires. Suppressed/snoozed
    fires are excluded, so muting an alert reduces the KPI tier chips and the
    Holdings header count in step with the grid below them."""
    counts = {1: 0, 2: 0, 3: 0}
    for f in account_state.fires:
        if not is_active(f):
            continue
        counts[f.tier] = counts.get(f.tier, 0) + 1
    return counts


def summary_line(account_state) -> str:
    """'3 act-today · 5 worth-raising · 1 idea' (omits zero tiers; all-zero →
    'No alerts')."""
    counts = tier_counts(account_state)
    parts = [f"{counts[t]} {_TIER_WORD[t]}" for t in (1, 2, 3) if counts.get(t)]
    return " · ".join(parts) if parts else "No alerts on this account"
