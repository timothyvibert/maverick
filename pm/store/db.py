"""The local SQLite app store — one connection seam for all persistent state.

A single database file under the data dir, gitignored exactly like the JSON it
replaces, never committed. It is the on-disk *application* store and is named
distinctly from ``portfolio_state.py`` / the in-memory ``PortfolioState`` (which is
transient, rebuilt on every load) so the two are never confused: this file holds the
state that must survive a reload.

Today it persists structure confirm/reject resolutions, alert suppressions, and
editable alert thresholds (settings). It is deliberately shaped so the next persistent
features slot in behind this same seam — a new table plus a thin sibling store module
reusing :func:`connection`, with no reshaping here:

* dated holdings/signal snapshots for day-over-day diffs.

That table is designed-for but intentionally *not* created yet — see the schema notes
below.

``sqlite3`` is in the Python standard library, so this adds no dependency.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pm.config import DATA_DIR

# The single store file. Read at connection time (not cached), so tests can point it
# at a tmp path with ``monkeypatch.setattr(db, "_DB_PATH", ...)`` and every operation
# follows.
_DB_PATH = DATA_DIR / "app_store.db"

SCHEMA_VERSION = 4

# Bookkeeping key marking that the one-time pre-SQLite resolutions import has run.
_LEGACY_RESOLUTIONS_IMPORTED = "structure_resolutions_json_imported"


def _legacy_resolutions_json() -> Path:
    """The pre-SQLite resolutions file, resolved as a *sibling* of the DB. Keying it
    to the DB's own directory means a monkeypatched store (tests) looks beside the tmp
    DB and never reaches the real ``pm/data`` file."""
    return _DB_PATH.parent / "structure_resolutions.json"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Built now:
#   meta(key, value)                  -- schema version + one-time-import bookkeeping;
#                                        also the home for future migration markers.
#   structure_resolutions(key, ...)   -- confirm / reject / choose / edit, keyed by
#                                        account || sorted leg-set (the composite key
#                                        is stored verbatim as the row key so the
#                                        apply-resolutions logic is unchanged).
#   suppressions(account, name, pattern_id, ...) -- per-alert suppress / snooze,
#                                        keyed by (account, name, pattern_id) where
#                                        name == fire.underlying; selective by design,
#                                        so suppressing one pattern on one name leaves
#                                        every other alert (and that pattern on other
#                                        names) firing. See pm/store/suppression_store.py.
#   settings(scope, name, value, ...) -- editable alert thresholds, keyed by
#                                        (scope, name) where scope is 'global' for v1
#                                        (the column is kept so a future per-account
#                                        scope slots in without reshaping) and name is a
#                                        PatternConfig field; value is the JSON-encoded
#                                        override. See pm/store/settings_store.py.
#
# Designed for, added later behind this same seam (a new table + a sibling store
# module reusing connection(); no change to the interface here):
#   state_snapshots(snapshot_date, account, payload, created_at)
#       -- dated holdings/signal snapshots, queried by date for day-over-day diffs.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS structure_resolutions (
    key         TEXT PRIMARY KEY,   -- account || '||' || ','.join(sorted(leg_pids))
    resolution  TEXT NOT NULL,      -- confirmed | rejected | edited
    chosen_type TEXT,               -- contention choice: the picked reading's type
    edited_legs TEXT,               -- JSON array of kept position_ids, or NULL
    timestamp   TEXT                -- ISO-8601 UTC
);
CREATE TABLE IF NOT EXISTS suppressions (
    account            TEXT NOT NULL,   -- the account the alert fired in
    name               TEXT NOT NULL,   -- == fire.underlying; suppression is per-name
    pattern_id         TEXT NOT NULL,   -- the pattern (P1..P20) being silenced on that name
    suppressed_until   TEXT,            -- NULL = permanent suppress; 'YYYY-MM-DD' = snooze through
    created_at         TEXT NOT NULL,   -- ISO-8601 UTC datetime the suppression was set
    captured_trace     TEXT,            -- json.dumps(fire.trace, default=str) at suppress time
    captured_rationale TEXT,            -- fire.rationale (displayed text) at suppress time
    PRIMARY KEY (account, name, pattern_id)   -- the selective key; one row per (acct,name,pattern)
);
CREATE TABLE IF NOT EXISTS settings (
    scope      TEXT NOT NULL,   -- 'global' for v1 (kept so per-account can slot in later)
    name       TEXT NOT NULL,   -- the PatternConfig field being overridden (e.g. p12_underlying_nav_pct_min)
    value      TEXT NOT NULL,   -- JSON-encoded scalar override, in PatternConfig-native units
    updated_at TEXT NOT NULL,   -- ISO-8601 UTC datetime the override was last written
    PRIMARY KEY (scope, name)   -- one row per (scope, threshold)
);
CREATE TABLE IF NOT EXISTS pattern_toggles (
    pattern_id TEXT PRIMARY KEY,   -- P1..P20; a row exists ONLY while the pattern is off
    enabled    INTEGER NOT NULL,   -- always 0 today (enable deletes the row); kept for audit clarity
    updated_at TEXT NOT NULL       -- ISO-8601 UTC datetime the toggle was last flipped
);
CREATE TABLE IF NOT EXISTS acknowledgements (
    account            TEXT NOT NULL,   -- the account the fire fired in
    pattern_id         TEXT NOT NULL,   -- the acknowledged pattern (P1..P20)
    fire_key           TEXT NOT NULL,   -- structure_id for structure fires, else position_id:
                                        -- the per-fire grain a name-wide suppression lacks
    acknowledged_at    TEXT NOT NULL,   -- ISO-8601 UTC datetime
    captured_trace     TEXT,            -- json.dumps(fire.trace, default=str) at ack time —
                                        -- the baseline material-change re-surfacing compares
    captured_rationale TEXT,            -- fire.rationale at ack time (review surface tooltip)
    PRIMARY KEY (account, pattern_id, fire_key)
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # Upsert (not INSERT OR IGNORE) so meta.schema_version tracks the CURRENT schema:
    # a DB created under an earlier version is brought up to SCHEMA_VERSION on the
    # next open, not left stale. The tables themselves are additive (CREATE TABLE IF
    # NOT EXISTS), so this only corrects the recorded version.
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("schema_version", str(SCHEMA_VERSION)),
    )


def _maybe_import_legacy_resolutions(conn: sqlite3.Connection) -> None:
    """Fold a pre-SQLite ``structure_resolutions.json`` into the table once, on first
    use, then never again. After this runs SQLite is authoritative; the JSON file is
    left untouched as a re-importable backup (so a DB reset can recover from it).

    Double-guarded against double-importing or clobbering newer SQLite state:

    * a ``meta`` flag makes the import run at most once per database, and
    * ``INSERT OR IGNORE`` means any key already present in SQLite wins over the JSON,
      so a lingering stale file can never overwrite a newer resolution.
    """
    flag = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (_LEGACY_RESOLUTIONS_IMPORTED,)
    ).fetchone()
    if flag is not None:
        return
    legacy = _legacy_resolutions_json()
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        rows = [
            (
                key,
                rec.get("resolution"),
                rec.get("chosen_type"),
                json.dumps(rec["edited_legs"]) if rec.get("edited_legs") is not None else None,
                rec.get("timestamp"),
            )
            for key, rec in data.items()
        ]
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO structure_resolutions"
                "(key, resolution, chosen_type, edited_legs, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
    # Mark the migration done whether or not a file was present, so it runs exactly
    # once per database and a later-appearing JSON is not silently re-imported.
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (_LEGACY_RESOLUTIONS_IMPORTED, datetime.now(timezone.utc).isoformat()),
    )


def store_exists() -> bool:
    """True if anything is persisted yet — the database itself, or a not-yet-imported
    legacy resolutions file beside it. Lets a pure read return empty without creating
    a database (preserving the old JSON store's read-a-missing-file semantics)."""
    return _DB_PATH.exists() or _legacy_resolutions_json().exists()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Open a connection to the app store, ensuring the schema exists and the one-time
    legacy import has run. Commits on a clean exit, rolls back on error, and always
    closes. Opened per call (no caching) so a monkeypatched ``_DB_PATH`` always takes
    effect and writes never hold a long-lived lock."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    try:
        # Wait briefly on a concurrently-held file lock instead of raising
        # 'database is locked' — the app server is threaded and two write
        # paths can now legitimately overlap at the SQLite layer.
        conn.execute("PRAGMA busy_timeout = 5000")
        _ensure_schema(conn)
        _maybe_import_legacy_resolutions(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
