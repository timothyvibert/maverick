"""The objective-spec registry — Harvest, its first entry.

An objective is data, not code-paths: admission bounds, a driver definition and
modulator settings, plus the desk-editable parameters behind them. This module
seeds the registry with HARVEST — the income re-write (the merged premium
objective: roll an OTM short for more credit at a moderate delta in a preferred
tenor window). The other objectives deliberately stay in their existing
``generate._select_roll`` / ``ranking._driver`` form until each is migrated on
its own increment; nothing here refactors them.

**Parameters** are the scanner's editable dials, persisted through the same
SQLite ``settings`` table as the alert thresholds (scope ``"global"``, names
prefixed ``harvest_``) via the defaults-over-overrides idiom: an unset dial
keeps its default, an unknown or out-of-range row is dropped, validation
REJECTS rather than clamps. The alert-side reader filters to its own catalog
names, so these rows can never reach ``PatternConfig`` — and vice versa.

**Soft bands.** The dials set the PREFERRED window (DTE, |Δ|); admission uses
hard outer bounds derived from them, and the driver multiplies by a band
kernel: 1.0 inside the preferred window, linear falloff to ``KERNEL_FLOOR`` at
the hard bounds — a just-outside candidate survives but the band wins.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

# The merged premium objective's tag. Generation, ranking and the drawer all
# key on this string; the old roll-for-credit / max-premium tags are retired.
HARVEST = "harvest"

# Hard-bound derivation and kernel shape (constants this increment; promotable
# to dials later). The Δ soft band is target ± DELTA_SOFT_HALF; hard bounds are
# target ± DELTA_HARD_HALF. The DTE hard window is [lo/2, 2·hi].
DELTA_SOFT_HALF = 0.10
DELTA_HARD_HALF = 0.20
KERNEL_FLOOR = 0.25

# Liquidity demotion-flag thresholds (flags, never silent drops). The spread
# threshold matches the vol-surface fit's own wide-market exclusion — one
# convention for "this quote is not to be trusted".
SPREAD_FLAG_PCT = 0.25
OI_THIN = 10

_SCOPE = "global"


@dataclass(frozen=True)
class HarvestParams:
    """The Harvest dials, in native units (Δ as a fraction)."""
    dte_lo: int = 30              # preferred window, earliest new-leg DTE
    dte_hi: int = 60              # preferred window, latest new-leg DTE
    delta_target: float = 0.25    # preferred new short-leg |Δ|
    crossing_k: float = 0.35      # execution haircut: fraction of the half-spread

    @property
    def dte_hard_lo(self) -> float:
        return self.dte_lo / 2.0

    @property
    def dte_hard_hi(self) -> float:
        return self.dte_hi * 2.0

    @property
    def delta_lo(self) -> float:
        return max(self.delta_target - DELTA_SOFT_HALF, 0.0)

    @property
    def delta_hi(self) -> float:
        return self.delta_target + DELTA_SOFT_HALF

    @property
    def delta_hard_lo(self) -> float:
        return max(self.delta_target - DELTA_HARD_HALF, 0.0)

    @property
    def delta_hard_hi(self) -> float:
        return self.delta_target + DELTA_HARD_HALF


DEFAULT_HARVEST_PARAMS = HarvestParams()


def band_kernel(x: Optional[float], lo: float, hi: float,
                hard_lo: float, hard_hi: float) -> float:
    """The soft-band preference multiplier: 1.0 inside [lo, hi], linear falloff
    to ``KERNEL_FLOOR`` at the hard bounds, floor outside them (admission is the
    hard gate — the kernel only grades what admission let through). ``x`` None
    (a defensive path; admission requires the value) grades neutral."""
    if x is None:
        return 1.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        span = lo - hard_lo
        frac = (x - hard_lo) / span if span > 0 else 0.0
    else:
        span = hard_hi - hi
        frac = (hard_hi - x) / span if span > 0 else 0.0
    return KERNEL_FLOOR + (1.0 - KERNEL_FLOOR) * max(0.0, min(1.0, frac))


# ---------------------------------------------------------------------------
# The editable-dial spec (the Thresholds tab's SCANNER group) + validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """One scanner dial: UI label/unit/range + the UI↔native scale."""
    name: str
    label: str
    unit: str
    is_int: bool
    min: float                    # UI-unit bounds; out-of-range REJECTS
    max: float
    scale: float = 1.0            # native = ui * scale
    default_native: float | int = 0


_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("harvest_dte_lo", "Harvest window, earliest new-leg expiry",
              "days", True, 7, 365, 1.0, DEFAULT_HARVEST_PARAMS.dte_lo),
    ParamSpec("harvest_dte_hi", "Harvest window, latest new-leg expiry",
              "days", True, 14, 730, 1.0, DEFAULT_HARVEST_PARAMS.dte_hi),
    ParamSpec("harvest_delta_target", "Harvest delta target (new short leg)",
              "Δ·100", False, 5, 50, 0.01, DEFAULT_HARVEST_PARAMS.delta_target),
    ParamSpec("harvest_crossing_k", "Execution crossing (fraction of half-spread)",
              "", False, 0, 1, 1.0, DEFAULT_HARVEST_PARAMS.crossing_k),
)
_BY_NAME = {s.name: s for s in _SPECS}


def spec_rows() -> Iterator[ParamSpec]:
    return iter(_SPECS)


def is_param(name: str) -> bool:
    return name in _BY_NAME


def default_ui(name: str) -> float:
    s = _BY_NAME[name]
    ui = s.default_native / s.scale
    return int(round(ui)) if s.is_int else ui


def _to_native(name: str, ui_value) -> float | int:
    """UI → native with range validation. Raises ``KeyError`` on an unknown
    dial and ``ValueError`` (reject, never clamp) on a bad value."""
    s = _BY_NAME[name]
    try:
        ui = float(ui_value)
    except (TypeError, ValueError):
        raise ValueError(f"{s.label}: not a number")
    if not (s.min <= ui <= s.max):
        raise ValueError(f"{s.label}: {ui:g} outside {s.min:g}–{s.max:g} {s.unit}".strip())
    native = ui * s.scale
    return int(round(native)) if s.is_int else native


def to_ui(name: str, native) -> float:
    s = _BY_NAME[name]
    ui = float(native) / s.scale
    return int(round(ui)) if s.is_int else ui


# ---------------------------------------------------------------------------
# Persistence — thin sibling of settings_store over the same settings table
# ---------------------------------------------------------------------------

def set_param(name: str, ui_value, *, now: Optional[datetime] = None) -> float | int:
    """Validate and persist one scanner dial (UI units in, native stored).
    Rejected values persist NOTHING — an existing override stays."""
    native = _to_native(name, ui_value)
    from pm.store import db
    updated_at = (now or datetime.now(timezone.utc)).isoformat()
    with db.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings(scope, name, value, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (_SCOPE, name, json.dumps(native), updated_at))
    return native


def clear_param(name: str) -> None:
    from pm.store import db
    if not db.store_exists():
        return
    with db.connection() as conn:
        conn.execute("DELETE FROM settings WHERE scope = ? AND name = ?",
                     (_SCOPE, name))


def get_param_overrides() -> dict[str, float | int]:
    """{name: native} for the persisted scanner dials, re-validated on read —
    a hand-edited out-of-range row is dropped so the dial reverts to its
    default, never silently adopted."""
    from pm.store import db
    if not db.store_exists():
        return {}
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT name, value FROM settings WHERE scope = ?", (_SCOPE,)
        ).fetchall()
    out: dict[str, float | int] = {}
    for name, value in rows:
        if not is_param(name):
            continue
        try:
            out[name] = _to_native(name, to_ui(name, json.loads(value)))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return out


def get_param_ui(name: str) -> Optional[float]:
    stored = get_param_overrides().get(name)
    return None if stored is None else to_ui(name, stored)


def build_harvest_params() -> HarvestParams:
    """Defaults with the persisted overrides laid on top — the object the
    scanner service threads into generation and ranking. A stored window that
    validates per-dial but inverts (lo ≥ hi) falls back to the DEFAULT window
    (both dials), never to a silent swap."""
    o = get_param_overrides()
    lo = int(o.get("harvest_dte_lo", DEFAULT_HARVEST_PARAMS.dte_lo))
    hi = int(o.get("harvest_dte_hi", DEFAULT_HARVEST_PARAMS.dte_hi))
    if lo >= hi:
        lo, hi = DEFAULT_HARVEST_PARAMS.dte_lo, DEFAULT_HARVEST_PARAMS.dte_hi
    return HarvestParams(
        dte_lo=lo, dte_hi=hi,
        delta_target=float(o.get("harvest_delta_target",
                                 DEFAULT_HARVEST_PARAMS.delta_target)),
        crossing_k=float(o.get("harvest_crossing_k",
                               DEFAULT_HARVEST_PARAMS.crossing_k)))
