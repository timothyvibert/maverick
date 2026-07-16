"""Local persistence for editable alert thresholds.

A setting is one threshold override — keyed ``(scope, name)`` where ``scope`` is
``"global"`` for v1 (the column is kept so a future per-account scope slots in without
reshaping) and ``name`` is a ``PatternConfig`` field. The stored ``value`` is the
JSON-encoded override in PatternConfig-native units; defaults that have never been
overridden are simply absent (defaults-over-overrides).

This is the second feature to *write* user state through the SQLite seam
(``pm/store/db.py``), after suppressions. It mirrors ``suppression_store`` /
``structure_store``'s shape — thin I/O over ``db.connection()`` plus a small build
helper — and adds no dependency.

**Reload, not recompute.** Unlike a suppression (which only *marks* an already-computed
fire), changing a threshold changes *which fires the engine produces*, so it cannot be
applied by re-marking — it requires re-running the engine. The apply path is therefore a
deliberate persist-then-reload (write the override here, then reload state through the
existing refresh path), which is why this store exposes only persistence +
``build_pattern_config``; the engine read happens in ``load_portfolio_state``.

**Validation lives in the catalog.** Every value is coerced/range-checked through
``threshold_catalog`` before it is persisted and again when it is read into a config, so
a bad or stale value can never reach the detectors. An override for an unknown/renamed
field is ignored, not fatal.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pm.insight import threshold_catalog as cat
from pm.insight.patterns import PatternConfig
from pm.store import db

# v1 scope. The column exists so per-account overrides can be added later without a
# schema change; today every read and write uses this single scope.
GLOBAL = "global"


# ---------------------------------------------------------------------------
# Write interface
# ---------------------------------------------------------------------------
def set_override(name: str, ui_value, *, now: Optional[datetime] = None) -> float | int:
    """Persist an override for one threshold from a desk-typed (UI-unit) value. The
    value is range-checked against the catalog and converted to the
    PatternConfig-native value before storage; the native value is returned. Raises
    ``KeyError`` for a non-editable / unknown ``name`` (a programming error — the UI
    only ever offers catalog dials) and ``ValueError`` for a non-numeric OR
    out-of-range value — in which case NOTHING is persisted (rejected, not clamped;
    an existing override is untouched).
    Upserts on ``(scope, name)``, so re-setting refreshes the value and ``updated_at``."""
    native = cat.to_stored(name, ui_value)             # KeyError / ValueError (reject)
    updated_at = (now or datetime.now(timezone.utc)).isoformat()
    with db.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings(scope, name, value, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (GLOBAL, name, json.dumps(native), updated_at),
        )
    return native


def clear_override(name: str) -> None:
    """Remove one override — the dial falls back to its PatternConfig default on the next
    reload. A pure no-op when nothing is persisted yet (never materializes the DB)."""
    if not db.store_exists():
        return
    with db.connection() as conn:
        conn.execute("DELETE FROM settings WHERE scope = ? AND name = ?", (GLOBAL, name))


def clear_all() -> None:
    """Drop every override (reset-to-defaults / test helper). No-op on an empty store."""
    if not db.store_exists():
        return
    with db.connection() as conn:
        conn.execute("DELETE FROM settings WHERE scope = ?", (GLOBAL,))


# ---------------------------------------------------------------------------
# Read interface — pure, never creates the database (mirrors the sibling stores)
# ---------------------------------------------------------------------------
def get_overrides() -> dict[str, float | int]:
    """The persisted overrides as ``{name: native_value}``, for the active scope. Each
    value is re-validated through the catalog (defense in depth) — a hand-edited
    out-of-range / wrong-sign DB row is DROPPED so that dial reverts to its default,
    never silently adopted at the most-aggressive bound; an override for a name
    that is no longer an editable dial is dropped, so a renamed/removed field can never
    reach the engine."""
    if not db.store_exists():
        return {}
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT name, value FROM settings WHERE scope = ?", (GLOBAL,)
        ).fetchall()
    out: dict[str, float | int] = {}
    for name, value in rows:
        if not cat.is_editable(name):
            continue                                   # unknown/renamed dial -> ignore
        try:
            out[name] = cat.validate_stored(name, json.loads(value))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue                                   # corrupt value -> fall back to default
    return out


def get_override_ui(name: str) -> Optional[float]:
    """The override for one dial in UI units, or ``None`` if unset — what the editor
    seeds an input with (falling back to the default when ``None``)."""
    stored = get_overrides().get(name)
    return None if stored is None else cat.to_ui(name, stored)


# ---------------------------------------------------------------------------
# Compose defaults + overrides into the config the engine reads
# ---------------------------------------------------------------------------
def build_pattern_config() -> PatternConfig:
    """Start from the ``PatternConfig`` defaults and lay the persisted overrides on top.
    An unset dial keeps its default; an override for a field that is not on PatternConfig
    is ignored. This is the object ``load_portfolio_state`` passes to the engine, so the
    reloaded fires reflect the desk's tuning."""
    config = PatternConfig()
    for name, native in get_overrides().items():
        if hasattr(config, name):
            setattr(config, name, native)
    return config
