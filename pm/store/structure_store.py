"""Local persistence for structure confirm/override resolutions.

Resolutions are keyed by ``(account, sorted leg-set)`` and stored in the SQLite app
store (``pm/store/db.py``). The detection pass reads these and applies them to the
freshly-detected proposals: when the leg-set still matches, the user's confirm /
reject / choose / edit is honoured; when the legs have changed, the stale key no
longer matches and the structure demotes to a fresh proposal.

The public interface is unchanged from the original file-backed store — only the
storage backend is SQLite. A pre-SQLite ``structure_resolutions.json``, if present,
is folded in once on first use and then retained as a backup; SQLite is authoritative
thereafter (see ``db._maybe_import_legacy_resolutions``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pm.store import db

CONFIRMED = "confirmed"
REJECTED = "rejected"
EDITED = "edited"


# ---------------------------------------------------------------------------
# Keying + storage I/O
# ---------------------------------------------------------------------------
def _key(account: str, leg_pids) -> str:
    return account + "||" + ",".join(sorted(leg_pids))


def _load() -> dict:
    """Every stored resolution as ``{key: {resolution, chosen_type, edited_legs,
    timestamp}}`` — the same shape the JSON store returned. A pure read never creates
    the database: with nothing persisted yet, return an empty mapping."""
    if not db.store_exists():
        return {}
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT key, resolution, chosen_type, edited_legs, timestamp "
            "FROM structure_resolutions"
        ).fetchall()
    out: dict = {}
    for key, resolution, chosen_type, edited_legs, timestamp in rows:
        out[key] = {
            "resolution": resolution,
            "chosen_type": chosen_type,
            "edited_legs": json.loads(edited_legs) if edited_legs is not None else None,
            "timestamp": timestamp,
        }
    return out


# ---------------------------------------------------------------------------
# Public write/read interface
# ---------------------------------------------------------------------------
def save_resolution(account: str, leg_pids, resolution: str,
                    chosen_type: Optional[str] = None,
                    edited_legs: Optional[list] = None,
                    now: Optional[datetime] = None) -> None:
    """Upsert ONE resolution row, keyed on (account, sorted leg-set).

    A single keyed write, not a load-the-mapping / rewrite-the-table cycle:
    the old wholesale DELETE + reinsert let two overlapping resolutions clobber
    each other (the later writer re-inserted the mapping it loaded before the
    earlier one committed — a lost update). The per-row upsert makes each
    resolution independent, matching the sibling stores' write shape."""
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO structure_resolutions"
            "(key, resolution, chosen_type, edited_legs, timestamp) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET"
            " resolution = excluded.resolution,"
            " chosen_type = excluded.chosen_type,"
            " edited_legs = excluded.edited_legs,"
            " timestamp = excluded.timestamp",
            (
                _key(account, leg_pids),
                resolution,                  # confirmed | rejected | edited
                chosen_type,                 # for a contention choice: the picked reading
                json.dumps(edited_legs) if edited_legs is not None else None,
                (now or datetime.now(timezone.utc)).isoformat(),
            ),
        )


def get_resolution(account: str, leg_pids) -> Optional[dict]:
    return _load().get(_key(account, leg_pids))


def all_resolutions() -> dict:
    return _load()


def clear_resolution(account: str, leg_pids) -> None:
    with db.connection() as conn:
        conn.execute("DELETE FROM structure_resolutions WHERE key = ?",
                     (_key(account, leg_pids),))


def clear_all() -> None:
    with db.connection() as conn:
        conn.execute("DELETE FROM structure_resolutions")


# ---------------------------------------------------------------------------
# Apply stored resolutions onto freshly-detected structures (in place)
# ---------------------------------------------------------------------------
def decision_leg_pids(structures, s) -> list:
    """The leg-set the resolution is keyed on. For a contended structure it is the
    union of all legs across the contention group (the decision spans the
    alternatives); otherwise it is the structure's own legs."""
    if s.contention_group:
        pids = set()
        for other in structures:
            if other.contention_group == s.contention_group:
                pids.update(leg.position_id for leg in other.legs)
        return sorted(pids)
    return sorted(leg.position_id for leg in s.legs)


def apply_resolutions(account: str, structures, resolutions: Optional[dict] = None) -> None:
    """Set each structure's ``status`` from the stored resolution for its leg-set.
    Legs unchanged → honour confirm / reject / choose / edit. Legs changed → no
    stored key matches → the structure stays ``proposed`` (a fresh proposal)."""
    data = resolutions if resolutions is not None else _load()
    for s in structures:
        r = data.get(_key(account, decision_leg_pids(structures, s)))
        if not r:
            s.status = "proposed"
            continue
        resolution = r.get("resolution")
        if s.contention_group:
            # One reading is chosen; the other alternatives are rejected.
            if resolution == CONFIRMED:
                s.status = CONFIRMED if r.get("chosen_type") == s.type else REJECTED
            else:
                s.status = REJECTED if resolution == REJECTED else "proposed"
        else:
            if resolution == EDITED:
                kept = set(r.get("edited_legs") or [])
                s.legs = [leg for leg in s.legs if leg.position_id in kept] or s.legs
                s.status = EDITED
            elif resolution in (CONFIRMED, REJECTED):
                s.status = resolution
            else:
                s.status = "proposed"
        if r.get("timestamp"):
            s.resolved_at = r["timestamp"]
