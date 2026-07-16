"""Local persistence for per-alert suppress / snooze decisions.

A suppression silences a *specific* alert — keyed ``(account, name, pattern_id)``
where ``name == fire.underlying`` — either permanently (``suppressed_until`` is
NULL) or until a date (a snooze). It is **selective by design**: suppressing P12
(concentration) on XLK in one account silences only P12 on XLK there; every other
pattern on XLK, and P12 on other names, still fires.

This is the first feature that *writes* user state through the SQLite seam
(``pm/store/db.py``). It mirrors ``structure_store``'s shape — thin I/O over
``db.connection()`` plus an ``apply_suppressions`` pass — and adds no dependency.

**No-recompute contract.** ``apply_suppressions`` runs in the load path *after* the
engine has produced fires. It only *marks* a fire (sets ``Fire.suppression``); it
never removes the fire from ``acc.fires`` and never re-derives anything. So a restore
(or a snooze that has expired) plus the next apply clears the mark and the alert
returns — the fire object was there all along. The active surfaces (blotter
consolidation, deep-dive rows) filter on :func:`is_active`; the modal's Muted footer
and the Alert Manager read the *muted* fires to display and restore them.

Snooze expiry is **lazy treat-as-inactive**: a row whose ``suppressed_until`` is
before ``as_of`` is simply absent from the active set (so the alert fires again), but
the row is retained — auditable, no background cleanup job. ``created_at`` drives the
Alert Manager's "days active" staleness cue.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from typing import Optional

from pm.insight.headline_metrics import (
    EVENT_RECURRENCE,
    HEADLINE_METRICS,
    HIGHER_FIRES,
    LOWER_FIRES,
    PROXY_ONLY,
    SIGNED,
)
from pm.insight.patterns import SuppressionMark
from pm.store import db


# ---------------------------------------------------------------------------
# Row <-> record mapping
# ---------------------------------------------------------------------------
_COLUMNS = (
    "account", "name", "pattern_id", "suppressed_until",
    "created_at", "captured_trace", "captured_rationale",
)
_SELECT = "SELECT " + ", ".join(_COLUMNS) + " FROM suppressions"


def _record(row) -> dict:
    return dict(zip(_COLUMNS, row))


# ---------------------------------------------------------------------------
# Write interface
# ---------------------------------------------------------------------------
def suppress(account: str, name: str, pattern_id: str, *,
             suppressed_until: Optional[str] = None,
             trace: Optional[dict] = None,
             rationale: Optional[str] = None,
             now: Optional[datetime] = None) -> None:
    """Suppress (``suppressed_until=None``) or snooze (a ``'YYYY-MM-DD'`` string) the
    alert ``(account, name, pattern_id)``. Upserts on the composite key, so re-calling
    flips suppress<->snooze and refreshes ``created_at`` (a re-decision restarts the
    days-active clock).

    ``trace`` / ``rationale`` capture the acting fire's baseline at suppress time
    (the fire is in hand where the control fires). ``trace`` is serialized with
    ``default=str`` so a live BBG trace carrying ``date`` objects in ``as_of`` cannot
    crash the write. This is **store-only** — nothing in this increment ever compares
    the captured baseline against the current fire (that is the deferred
    material-change feature)."""
    created_at = (now or datetime.now(timezone.utc)).isoformat()
    captured_trace = json.dumps(trace, default=str) if trace is not None else None
    with db.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO suppressions"
            "(account, name, pattern_id, suppressed_until, created_at,"
            " captured_trace, captured_rationale) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (account, name, pattern_id, suppressed_until, created_at,
             captured_trace, rationale),
        )


def restore(account: str, name: str, pattern_id: str) -> None:
    """Remove the suppression — the alert returns on the next apply. A pure no-op when
    nothing is persisted yet (never materializes the database)."""
    if not db.store_exists():
        return
    with db.connection() as conn:
        conn.execute(
            "DELETE FROM suppressions WHERE account = ? AND name = ? AND pattern_id = ?",
            (account, name, pattern_id),
        )


def clear_all() -> None:
    """Drop every suppression (test/reset helper). No-op on an empty store."""
    if not db.store_exists():
        return
    with db.connection() as conn:
        conn.execute("DELETE FROM suppressions")


# ---------------------------------------------------------------------------
# Read interface — pure, never creates the database (mirrors structure_store)
# ---------------------------------------------------------------------------
def get_suppression(account: str, name: str, pattern_id: str) -> Optional[dict]:
    if not db.store_exists():
        return None
    with db.connection() as conn:
        row = conn.execute(
            _SELECT + " WHERE account = ? AND name = ? AND pattern_id = ?",
            (account, name, pattern_id),
        ).fetchone()
    return _record(row) if row else None


def all_suppressions() -> list[dict]:
    """Every stored suppression as a full record (the Alert Manager renders these and
    computes days-active). Includes expired snoozes — they are retained, not deleted."""
    if not db.store_exists():
        return []
    with db.connection() as conn:
        rows = conn.execute(_SELECT).fetchall()
    return [_record(r) for r in rows]


def _is_active(suppressed_until: Optional[str], as_of_str: str) -> bool:
    # Permanent, or snoozed through a date >= today. ISO 'YYYY-MM-DD' strings compare
    # lexicographically the same as by date, so the boundary day is still suppressed.
    return suppressed_until is None or suppressed_until >= as_of_str


def active_suppressions(as_of: Optional[date] = None) -> dict[tuple[str, str, str], dict]:
    """The currently-active suppressions as ``{(account, name, pattern_id): record}``.
    A snooze with ``suppressed_until < as_of`` is inactive (the alert returns); its row
    persists. ``as_of`` defaults to today."""
    if not db.store_exists():
        return {}
    as_of_str = (as_of or date.today()).isoformat()
    return {
        (r["account"], r["name"], r["pattern_id"]): r
        for r in all_suppressions()
        if _is_active(r["suppressed_until"], as_of_str)
    }


# ---------------------------------------------------------------------------
# Apply onto already-computed fires (the load-path marking pass)
# ---------------------------------------------------------------------------
def _mark_for(record: dict) -> SuppressionMark:
    until = record["suppressed_until"]
    return SuppressionMark(kind="snoozed" if until else "suppressed", until=until)


def _mark_fires(account: str, fires, active: dict) -> None:
    """Set each fire's ``suppression`` from the active set — and CLEAR it where no
    active suppression matches. Clearing is what makes restore / snooze-expiry return
    the alert on a re-apply without recomputing."""
    for f in fires:
        rec = active.get((account, f.underlying, f.pattern_id))
        f.suppression = _mark_for(rec) if rec else None


def apply_suppressions(state, as_of: Optional[date] = None) -> None:
    """Mark every fire across all accounts whose ``(account, underlying, pattern_id)``
    is actively suppressed. Called in ``load_portfolio_state`` after the engine — it
    only flags already-computed fires (no recompute, no Bloomberg)."""
    active = active_suppressions(as_of)
    for acc in state.accounts.values():
        _mark_fires(acc.account, acc.fires, active)


def remark_account(account_state, as_of: Optional[date] = None) -> None:
    """Re-mark one account's fires — used by ``state_access`` after a structure
    confirm/reject re-derive and the suppress/restore/acknowledge/toggle write paths.
    Applies the full marking order the load path uses: active suppressions →
    material-change re-surfacing → per-fire acknowledgements (their own material
    net) → per-pattern disables last (final — never overridden by a material
    move). No reload, no recompute."""
    from pm.store.alert_governance import mark_acknowledged, mark_disabled
    active = active_suppressions(as_of)
    _mark_fires(account_state.account, account_state.fires, active)
    _remark_material(account_state, active)
    mark_acknowledged(account_state)
    mark_disabled(account_state)


def is_active(fire) -> bool:
    """The shared predicate: a fire is an active alert iff it is unmuted (no mark) OR it
    was muted but its condition has moved materially since (kind ``"resurfaced"``). Every
    surface that hides muted alerts filters on this, so a re-surfaced alert returns to all
    of them at once and a snooze-expiry (mark cleared) is never confused with it."""
    s = fire.suppression
    return s is None or s.kind == "resurfaced"


# ---------------------------------------------------------------------------
# Material-change re-surfacing — flip a muted alert back to active when its
# condition has moved materially since mute time. A read-path MARKING pass, exactly like
# apply_suppressions: it reads the stored captured_trace baseline + the current fire's
# already-computed trace and sets kind="resurfaced". It recomputes no fires, calls no
# Bloomberg, and persists nothing — the state is derived fresh on every load (a condition
# that moves material then back reverts to muted next load). It consumes the item-11
# headline-metric map (which value, what type) + the item-9 captured baseline.
# ---------------------------------------------------------------------------
_MATERIAL_REL_MARGIN = 0.25    # re-surface on a >=25% worse move vs the captured baseline
_MATERIAL_ABS_FLOOR = 1e-9     # degenerate-baseline guard (real fires sit well away from 0)

_MISSING = object()


def _dig(trace, key: tuple):
    """Walk a trace dict by a key tuple (a headline-map trace_key); ``_MISSING`` if any
    step is absent. Works on both a live ``fire.trace`` and ``json.loads(captured_trace)``."""
    cur = trace
    for k in key:
        if not isinstance(cur, dict) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


def _to_float(v) -> Optional[float]:
    """A finite float, or None for missing / non-numeric / NaN / inf (bool excluded)."""
    if v is _MISSING or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _to_date(v) -> Optional[date]:
    """Normalize an event date to a ``date`` from either side of the json round-trip: a
    live ``date`` / ``datetime`` / ``pd.Timestamp`` (a datetime subclass), or the
    ``default=str``-stringified form captured at suppress time."""
    if v is None or v is _MISSING:
        return None
    if isinstance(v, datetime):        # also catches pd.Timestamp (a datetime subclass)
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except (ValueError, TypeError):
        return None


def _metric_label(hm) -> str:
    """A short, stable label for the headline metric (the leaf of its key) — e.g.
    ``"nav_pct"`` or ``"analyst_note_date"``; the surfaces prettify it."""
    key = (hm.event_id_key if hm.metric_type == EVENT_RECURRENCE and hm.event_id_key
           else hm.primary.trace_key)
    return key[-1]


def _material_numeric(captured: Optional[float], current: Optional[float], direction: str) -> bool:
    """True when ``current`` has moved at least the relative margin further than
    ``captured`` in the firing direction. SIGNED (P3) takes the firing sign from the
    captured value (a put fired on a negative break, a call on a positive one)."""
    if captured is None or current is None:
        return False
    if abs(captured) < _MATERIAL_ABS_FLOOR:    # can't size a relative move off ~0
        return False
    need = abs(captured) * _MATERIAL_REL_MARGIN
    move = current - captured
    if direction == HIGHER_FIRES:
        return move >= need
    if direction == LOWER_FIRES:
        return -move >= need
    if direction == SIGNED:
        return (-move >= need) if captured < 0 else (move >= need)
    return False


def _more_extreme(a: float, b: float, direction: str, captured: float) -> bool:
    """Is ``a`` further in the firing direction than ``b``? (picks the worst current
    instance when a pattern fires per position — the multiplicity rule)."""
    if direction == HIGHER_FIRES:
        return a > b
    if direction == LOWER_FIRES:
        return a < b
    if direction == SIGNED:
        return (a < b) if captured < 0 else (a > b)
    return False


def _resurfaces(record: dict, fires) -> Optional[SuppressionMark]:
    """A ``resurfaced`` mark (with the headline delta) if this suppression's condition
    moved materially vs its captured baseline, else None. ``fires`` are the currently-marked
    fires under the ``(account, name, pattern_id)`` key. Excludes P8 (proxy headline) and
    any pattern with no map entry (the structure fires P16-P20 HAVE entries — P16
    monotonic on the naked-contract count, P17-P20 event-keyed on the leg expiry);
    guards a missing/garbled baseline or headline (no re-surface)."""
    hm = HEADLINE_METRICS.get(record["pattern_id"])
    if hm is None or hm.metric_type == PROXY_ONLY:
        return None
    raw = record.get("captured_trace")
    if not raw:
        return None
    try:
        captured = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if hm.metric_type == EVENT_RECURRENCE:
        cap_d = _to_date(_dig(captured, hm.event_id_key)) if hm.event_id_key else None
        if cap_d is None:
            return None
        best_d = None                          # the newest current event date
        for f in fires:
            d = _to_date(_dig(getattr(f, "trace", {}) or {}, hm.event_id_key))
            if d is not None and (best_d is None or d > best_d):
                best_d = d
        if best_d is None or not (best_d > cap_d):
            return None
        return SuppressionMark(kind="resurfaced", metric=_metric_label(hm),
                               captured_value=cap_d.isoformat(), current_value=best_d.isoformat())

    # numeric: monotonic, or the multi_axis primary axis
    axis = hm.primary
    cap_v = _to_float(_dig(captured, axis.trace_key))
    if cap_v is None:
        return None
    best_v = None                              # the most-extreme current value
    for f in fires:
        cv = _to_float(_dig(getattr(f, "trace", {}) or {}, axis.trace_key))
        if cv is None:
            continue
        if best_v is None or _more_extreme(cv, best_v, axis.direction, cap_v):
            best_v = cv
    if best_v is None or not _material_numeric(cap_v, best_v, axis.direction):
        return None
    return SuppressionMark(kind="resurfaced", metric=_metric_label(hm),
                           captured_value=cap_v, current_value=best_v)


def _remark_material(account_state, active: dict) -> None:
    """Flip this account's materially-moved suppressions to ``resurfaced`` (active again),
    over the fires ``_mark_fires`` just marked — so it considers permanent suppressions AND
    unexpired snoozes (a material move overrides an active snooze)."""
    acc = account_state.account
    by_key: dict = {}
    for f in account_state.fires:
        if f.suppression is None:              # unmuted / expired-snooze: nothing to re-surface
            continue
        by_key.setdefault((f.underlying, f.pattern_id), []).append(f)
    for (name, pid), fires in by_key.items():
        record = active.get((acc, name, pid))
        if record is None:
            continue
        mark = _resurfaces(record, fires)
        if mark is not None:
            for f in fires:
                f.suppression = mark


def apply_material_change(state, as_of: Optional[date] = None) -> None:
    """Re-surface materially-moved suppressions across all accounts. Called in
    ``load_portfolio_state`` immediately after ``apply_suppressions`` — a read-path mark,
    no recompute, no Bloomberg, no persistence."""
    active = active_suppressions(as_of)
    for account_state in state.accounts.values():
        _remark_material(account_state, active)
