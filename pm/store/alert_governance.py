"""Alert governance: persisted per-pattern toggles + per-fire acknowledgements.

Two durable controls beside the name-wide suppressions, both applied as
SURFACE-TIME MARKING passes over already-computed fires (never skip-at-fire):
a toggled-off pattern's fires and an acknowledged fire stay on ``acc.fires``,
counted and recoverable — they are flagged inactive, exactly like a
suppression, so flipping the control back restores them with no recompute and
no Bloomberg call.

**Pattern toggles** (``pattern_toggles`` table): one row per DISABLED pattern
(P1–P20); enabling deletes the row, so the default state costs nothing and an
empty store means everything is on. Marking sets ``SuppressionMark(kind=
"disabled")``. A disabled mark is applied LAST in the marking order and is
final: material-change re-surfacing never overrides an explicit off-switch.

**Acknowledgements** (``acknowledgements`` table): keyed ``(account,
pattern_id, fire_key)`` where ``fire_key`` is the fire's ``structure_id`` when
set (structure fires) else its ``position_id`` — the per-FIRE grain the
name-wide suppression key cannot express. An ack captures the fire's trace as
its baseline and marks ``kind="acknowledged"``; the same material-change
comparison the suppressions use (via the headline-metric map) flips the mark
to ``"resurfaced"`` when the fire's condition later moves materially — the
safety net that makes acknowledging a disputed tier-1 structure fire safe
where a permanent name-wide mute was not.

Marking order (the load path and ``suppression_store.remark_account`` both
follow it): suppressions → material change → acknowledgements (with their own
material net) → pattern disables. ``is_active`` needs no change: only ``None``
and ``"resurfaced"`` are active, so both new kinds are inactive by
construction.

Storage lives in the shared SQLite app store behind ``pm.store.db`` —
mirroring the sibling stores (open-per-call ``connection()``, keyed upserts,
lazy reads that never materialize the DB).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pm.insight.patterns import SuppressionMark
from pm.store import db


# ---------------------------------------------------------------------------
# Pattern-id inventory (validation)
# ---------------------------------------------------------------------------

def togglable_pattern_ids() -> set[str]:
    """Every pattern the toggle governs — the P1–P15 engine patterns plus the
    P16–P20 structure fires, from the same union the group-map exhaustiveness
    oracle uses."""
    from pm.insight.pattern_groups import all_pattern_meta
    return set(all_pattern_meta())


# ---------------------------------------------------------------------------
# Pattern toggles — persistence
# ---------------------------------------------------------------------------

def disabled_patterns() -> set[str]:
    """The set of pattern ids currently toggled off. Empty when the store has
    never been written (a pure read never creates the DB)."""
    if not db.store_exists():
        return set()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT pattern_id FROM pattern_toggles WHERE enabled = 0").fetchall()
    return {r[0] for r in rows}


def set_pattern_enabled(pattern_id: str, enabled: bool, *,
                        now: Optional[datetime] = None) -> None:
    """Flip one pattern's toggle. Enabling deletes the row (absence == on);
    disabling upserts it. Raises KeyError on an unknown pattern id so a typo
    can never silently disable nothing."""
    if pattern_id not in togglable_pattern_ids():
        raise KeyError(f"unknown pattern id {pattern_id!r}")
    with db.connection() as conn:
        if enabled:
            conn.execute("DELETE FROM pattern_toggles WHERE pattern_id = ?",
                         (pattern_id,))
        else:
            conn.execute(
                "INSERT OR REPLACE INTO pattern_toggles(pattern_id, enabled, updated_at) "
                "VALUES (?, 0, ?)",
                (pattern_id, (now or datetime.now(timezone.utc)).isoformat()))


# ---------------------------------------------------------------------------
# Acknowledgements — persistence
# ---------------------------------------------------------------------------

def fire_key(fire) -> str:
    """The per-fire grain: the originating structure for structure fires, the
    position for everything else."""
    return getattr(fire, "structure_id", None) or fire.position_id


def acknowledge(account: str, pattern_id: str, key: str, *,
                trace: Optional[dict] = None, rationale: Optional[str] = None,
                now: Optional[datetime] = None) -> None:
    """Persist an acknowledgement, capturing the acting fire's trace as the
    material-change baseline (re-acknowledging refreshes it)."""
    with db.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO acknowledgements"
            "(account, pattern_id, fire_key, acknowledged_at, captured_trace,"
            " captured_rationale) VALUES (?, ?, ?, ?, ?, ?)",
            (account, pattern_id, key,
             (now or datetime.now(timezone.utc)).isoformat(),
             json.dumps(trace, default=str) if trace is not None else None,
             rationale))


def unacknowledge(account: str, pattern_id: str, key: str) -> None:
    with db.connection() as conn:
        conn.execute(
            "DELETE FROM acknowledgements WHERE account = ? AND pattern_id = ? "
            "AND fire_key = ?", (account, pattern_id, key))


def acknowledgements(account: Optional[str] = None) -> dict[tuple, dict]:
    """All acknowledgement records keyed ``(account, pattern_id, fire_key)``,
    optionally filtered to one account. Empty when the store has never been
    written."""
    if not db.store_exists():
        return {}
    sql = ("SELECT account, pattern_id, fire_key, acknowledged_at, "
           "captured_trace, captured_rationale FROM acknowledgements")
    params: tuple = ()
    if account is not None:
        sql += " WHERE account = ?"
        params = (account,)
    with db.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {(r[0], r[1], r[2]): {
        "account": r[0], "pattern_id": r[1], "fire_key": r[2],
        "acknowledged_at": r[3], "captured_trace": r[4],
        "captured_rationale": r[5],
    } for r in rows}


# ---------------------------------------------------------------------------
# Marking passes (read-path; no recompute, no Bloomberg, persist nothing)
# ---------------------------------------------------------------------------

def mark_acknowledged(account_state) -> None:
    """Mark this account's acknowledged fires — or re-surface them.

    Only fires that are currently ACTIVE (unmarked or resurfaced) are
    considered: a name-wide suppression is broader than a per-fire ack and
    keeps precedence. Each ack's captured baseline runs through the SAME
    material comparison the suppressions use (``suppression_store._resurfaces``
    over the headline-metric map) scoped to exactly this fire — a materially
    moved condition stays active, marked ``resurfaced`` with its
    baseline→current delta; an unmoved one is marked ``acknowledged``.
    """
    from pm.store.suppression_store import _resurfaces, is_active
    acks = acknowledgements(account_state.account)
    if not acks:
        return
    for f in account_state.fires:
        if not is_active(f):
            continue                     # suppression/snooze/disable keeps precedence
        rec = acks.get((f.account, f.pattern_id, fire_key(f)))
        if rec is None:
            continue
        resurfaced = _resurfaces(rec, [f])
        f.suppression = resurfaced if resurfaced is not None \
            else SuppressionMark(kind="acknowledged")


def mark_disabled(account_state, disabled: Optional[set[str]] = None) -> None:
    """Mark every fire of a toggled-off pattern ``disabled`` — LAST word in the
    marking order: it overwrites even a ``resurfaced`` mark, because an
    explicit off-switch must hold until the user flips it back. Fires of
    enabled patterns are left untouched (their suppression/ack marks stand)."""
    if disabled is None:
        disabled = disabled_patterns()
    if not disabled:
        return
    for f in account_state.fires:
        if f.pattern_id in disabled:
            f.suppression = SuppressionMark(kind="disabled")


def apply_acknowledgements(state) -> None:
    """Load-path pass: run after ``apply_material_change``."""
    for account_state in state.accounts.values():
        mark_acknowledged(account_state)


def apply_pattern_disables(state) -> None:
    """Load-path pass: run LAST, after the acknowledgement pass."""
    disabled = disabled_patterns()
    if not disabled:
        return
    for account_state in state.accounts.values():
        mark_disabled(account_state, disabled)


def disabled_fire_counts(state) -> dict[str, int]:
    """Per-pattern count of fires currently hidden by the toggle — the visible
    'muted, not dropped' number the manager and status bar show."""
    counts: dict[str, int] = {}
    for acc in state.accounts.values():
        for f in acc.fires:
            s = f.suppression
            if s is not None and s.kind == "disabled":
                counts[f.pattern_id] = counts.get(f.pattern_id, 0) + 1
    return counts
