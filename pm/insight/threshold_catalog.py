"""The editable alert-threshold catalog — one source of truth.

The desk tunes how sensitive each alert is ("flag a single name above 35% of NAV, not
30%"). Every editable dial is a ``PatternConfig`` field; this catalog is the
single place that says, per dial: which pattern it belongs to, the plain-English label
the desk reads, its unit, and the sane range it is clamped to. It drives three things
so they can never drift apart:

* validation/coercion in ``settings_store`` (a bad value can never reach the engine),
* the defaults-over-overrides config build, and
* the Thresholds-tab UI (labels, grouping, the seeded input value).

**Units transform.** ``PatternConfig`` stores native values — fractions (0.30 = 30% of
NAV), signed losses (-1.0 = -100% P&L), raw day counts. The desk should never type a
fraction or a minus sign, so each spec carries a ``scale`` (×100 for percent dials) and
a ``sign`` (-1 where the native value is a magnitude-below, e.g. the P3 break and the
P6/P8 loss floors). The UI shows a positive, human number; ``to_stored`` /
``to_ui`` convert between that and the native value the detectors read.

**Scope of the catalog.** The 27 editable dials only. Two ``PatternConfig`` fields are
deliberately absent: ``p9_fresh_window_days`` (a calendar-day approximation of "<=10
business days" — a mechanical convention, not a sensitivity dial, so it stays locked in
code) and the removed-vestigial ``p4_target_change_window_days``. The structure-fire
thresholds (P16-P20) are module constants in ``structure_fires.py`` and out of scope for
this increment (see the note there).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from pm.insight.patterns import PATTERN_META, PatternConfig

# Unit tokens (display only).
PERCENT = "%"
PERCENTILE = "pctl"
DAYS = "days"
BUSINESS_DAYS = "business days"
SIGMA = "σ"          # σ — vol-adjusted move multiplier
VOL_PTS = "vol pts"

_PCT = 0.01               # native fraction <-> UI percent
_RAW = 1.0               # native == UI


@dataclass(frozen=True)
class ThresholdSpec:
    """One editable dial. ``min`` / ``max`` are in UI units (what the desk types)."""
    name: str                 # the PatternConfig field
    pattern_id: str           # the owning engine pattern (P1..P15, P21)
    label: str                # plain-English, unit-explicit
    unit: str
    is_int: bool              # native value is an int (day counts)
    min: float                # UI-unit clamp floor (always a positive magnitude)
    max: float                # UI-unit clamp ceiling
    scale: float = _RAW       # native = ui * scale * sign
    sign: int = 1            # -1 where the native value is a magnitude-below (loss / break)


# Order follows the pattern number so the UI groups read P1 -> P21 top to bottom.
_SPECS: tuple[ThresholdSpec, ...] = (
    # --- P1 ---------------------------------------------------------------
    ThresholdSpec("p1_captured_min", "P1",
                  "Flag a short option once premium captured reaches", PERCENT,
                  False, 0, 100, _PCT),
    # --- P2 ---------------------------------------------------------------
    ThresholdSpec("p2_captured_min", "P2",
                  "Close-and-rewrite: minimum premium captured", PERCENT,
                  False, 0, 100, _PCT),
    ThresholdSpec("p2_iv_pctl_min", "P2",
                  "Close-and-rewrite: minimum 3M IV percentile (1Y)", PERCENTILE,
                  False, 0, 100),
    # --- P3 ---------------------------------------------------------------
    ThresholdSpec("p3_captured_min", "P3",
                  "Adverse-break: minimum premium captured", PERCENT,
                  False, 0, 100, _PCT),
    ThresholdSpec("p3_200d_break_threshold", "P3",
                  "Adverse-break: flag a 200-day MA break of at least", PERCENT,
                  False, 0, 50, _PCT, sign=-1),
    ThresholdSpec("p3_return_5d_threshold", "P3",
                  "Adverse-break: flag a 5-session move of at least", PERCENT,
                  False, 0, 50, _PCT, sign=-1),
    # --- P4 ---------------------------------------------------------------
    ThresholdSpec("p4_captured_min", "P4",
                  "Analyst-note review: minimum premium captured", PERCENT,
                  False, 0, 100, _PCT),
    # --- P5 ---------------------------------------------------------------
    ThresholdSpec("p5_dte_max", "P5",
                  "Roll-due: flag a short option within this many days of expiry", DAYS,
                  True, 1, 365),
    ThresholdSpec("p5_captured_min", "P5",
                  "Roll-due: minimum premium captured", PERCENT,
                  False, 0, 100, _PCT),
    # --- P6 ---------------------------------------------------------------
    ThresholdSpec("p6_pnl_pct_max", "P6",
                  "Stress (with time pressure): flag a loss of at least", PERCENT,
                  False, 0, 1000, _PCT, sign=-1),
    ThresholdSpec("p6_dte_max", "P6",
                  "Stress (with time pressure): within this many days of expiry", DAYS,
                  True, 1, 365),
    ThresholdSpec("p6_extreme_pnl_pct_max", "P6",
                  "Stress (any expiry): flag an extreme loss of at least", PERCENT,
                  False, 0, 1000, _PCT, sign=-1),
    # --- P7 ---------------------------------------------------------------
    ThresholdSpec("p7_exdiv_window_days", "P7",
                  "Ex-div trap: warn within this many days of ex-dividend", DAYS,
                  True, 1, 60),
    # --- P8 ---------------------------------------------------------------
    ThresholdSpec("p8_recent_trade_window_days", "P8",
                  "Roll asymmetry: a leg counts as 'recently traded' within",
                  BUSINESS_DAYS, True, 1, 60),
    ThresholdSpec("p8_residual_pnl_pct_max", "P8",
                  "Roll asymmetry: a residual leg's loss is at least", PERCENT,
                  False, 0, 1000, _PCT, sign=-1),
    # --- P9 ---------------------------------------------------------------
    ThresholdSpec("p9_nav_pct_min", "P9",
                  "Fresh position: flag once it reaches this share of NAV", PERCENT,
                  False, 0, 100, _PCT),
    # --- P10 --------------------------------------------------------------
    ThresholdSpec("p10_pnl_pct_min", "P10",
                  "Big winner: flag a long option gain of at least", PERCENT,
                  False, 0, 5000, _PCT),
    # --- P11 --------------------------------------------------------------
    ThresholdSpec("p11_idle_days_min", "P11",
                  "Idle account: no trades for at least this many days", DAYS,
                  True, 1, 365),
    ThresholdSpec("p11_cash_pct_min", "P11",
                  "Idle account: cash is at least this share of NAV", PERCENT,
                  False, 0, 100, _PCT),
    # --- P12 --------------------------------------------------------------
    ThresholdSpec("p12_single_position_nav_pct_min", "P12",
                  "Concentration: flag a single equity above this share of NAV", PERCENT,
                  False, 0, 100, _PCT),
    ThresholdSpec("p12_underlying_nav_pct_min", "P12",
                  "Concentration: flag a single name (summed) above this share of NAV", PERCENT,
                  False, 0, 100, _PCT),
    # --- P13 --------------------------------------------------------------
    ThresholdSpec("p13_iv_pctl_min", "P13",
                  "Vol-rich covered call: minimum 3M IV percentile (1Y)", PERCENTILE,
                  False, 0, 100),
    # --- P14 --------------------------------------------------------------
    ThresholdSpec("p14_earnings_window_days", "P14",
                  "Earnings setup: flag within this many days of earnings", DAYS,
                  True, 1, 90),
    ThresholdSpec("p14_iv_pctl_min", "P14",
                  "Earnings setup: minimum 3M IV percentile (1Y)", PERCENTILE,
                  False, 0, 100),
    ThresholdSpec("p14_term_structure_min", "P14",
                  "Earnings setup: minimum IV term-structure richness (3M-6M)", VOL_PTS,
                  False, 0, 50),
    # --- P15 --------------------------------------------------------------
    ThresholdSpec("p15_vol_multiplier_min", "P15",
                  "Notable move: flag a daily move of at least", SIGMA,
                  False, 0, 10),
    # --- P21 --------------------------------------------------------------
    # The ONE definition of a "recent" research note: the standalone note
    # pattern fires within this window, and the captured+note compound (P4)
    # reads the same window through the D3 signal.
    ThresholdSpec("p21_note_window_bd", "P21",
                  "Research note: flag a note within this many business days",
                  BUSINESS_DAYS, True, 1, 20),
)

# name -> spec, the lookup every consumer uses.
THRESHOLD_CATALOG: dict[str, ThresholdSpec] = {s.name: s for s in _SPECS}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def editable_names() -> list[str]:
    """The PatternConfig field names that are editable, in display order."""
    return [s.name for s in _SPECS]


def is_editable(name: str) -> bool:
    return name in THRESHOLD_CATALOG


def spec(name: str) -> ThresholdSpec:
    """The spec for an editable dial; raises KeyError for a non-editable / unknown name
    (a programming error — the UI only ever offers catalog names)."""
    return THRESHOLD_CATALOG[name]


def grouped_by_pattern() -> Iterator[tuple[str, str, list[ThresholdSpec]]]:
    """Yield ``(pattern_id, pattern_name, [specs])`` in pattern order — the structure the
    Thresholds tab renders, one block per pattern."""
    seen: list[str] = []
    for s in _SPECS:
        if s.pattern_id not in seen:
            seen.append(s.pattern_id)
    for pid in seen:
        name = PATTERN_META.get(pid, (pid, None))[0]
        yield pid, name, [s for s in _SPECS if s.pattern_id == pid]


# ---------------------------------------------------------------------------
# UI <-> native (stored) transforms + clamping
# ---------------------------------------------------------------------------
def to_stored(name: str, ui_value) -> float | int:
    """Convert a desk-typed UI value to the PatternConfig-native value, validating and
    clamping along the way: cast to float (raises ValueError on garbage), clamp to the
    spec's UI range, apply scale + sign, and round to int for day-count dials. The
    result is guaranteed in-range, so it can never push the engine outside the catalog's
    sane bounds."""
    s = spec(name)
    v = float(ui_value)                       # ValueError on non-numeric
    v = max(s.min, min(s.max, v))             # clamp in UI units
    native = v * s.scale * s.sign
    if s.is_int:
        native = int(round(native))
    return native


def to_ui(name: str, stored_value) -> float:
    """Inverse of :func:`to_stored` — the positive UI magnitude for a native value, used
    to seed the editor's input from a stored override (or a default)."""
    s = spec(name)
    return float(stored_value) / (s.scale * s.sign)


def clamp_stored(name: str, stored_value) -> float | int:
    """Re-clamp a native value through the catalog's range (defense in depth: a
    hand-edited DB value is brought back into bounds before it reaches the engine)."""
    return to_stored(name, to_ui(name, stored_value))


def default_stored(name: str) -> float | int:
    """The PatternConfig default for a dial, in native units."""
    return getattr(PatternConfig(), name)


def default_ui(name: str) -> float:
    """The PatternConfig default for a dial, in UI units (seeds an un-overridden input)."""
    return to_ui(name, default_stored(name))
