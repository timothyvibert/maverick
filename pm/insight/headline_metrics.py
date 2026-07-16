"""The per-pattern headline-metric map — data + types only.

For each pattern — the engine patterns (P1-P15, P21) AND the P16-P20 structure fires — this
names the **single metric whose movement defines "materially changed"**: where the value
already lives in the fire's ``trace``, the ``PatternConfig`` threshold it is measured
against (None for the structure fires, whose thresholds are module constants), and the
direction that fires. It is the static artifact material-change re-surfacing consumes to
decide whether a suppressed or acknowledged alert's condition has moved enough to bring
it back — by comparing the captured baseline trace against the current fire's trace at
this key. The P16-P20 entries exist precisely so a muted tier-1 structure fire has a way
back (they were the one alert family with no re-surface net).

**This module defines the map; it builds no comparison.** Nothing here reads a
suppression or compares captured-vs-current.

Each entry carries a ``metric_type`` describing how the next item should compare it:

* ``monotonic_numeric`` — one number that crosses the threshold one way (P1, P6, P9,
  P10, P12, P15). The clean case: compare the headline value against the threshold.
* ``event_recurrence`` — the real trigger is a calendar event, not a drifting metric
  (P4 a fresh analyst note, P7 the ex-div date, P14 the earnings countdown). Re-surfacing
  is better keyed to the event recurring than to the numeric drifting.
* ``multi_axis`` — two or more conditions, or a categorical gate (P2, P3, P5, P11, P13).
  The primary axis is the economic spine; the secondaries are recorded so the
  comparison can decide whether one axis suffices.
* ``proxy_only`` — no clean single metric; the trigger is event/structure-shaped (P8 a
  roll-timing asymmetry). The primary is the best available numeric proxy, flagged.

``trace_key`` is a path into ``fire.trace`` (e.g. ``("result", "captured_pct")`` or
``("inputs", "spot_vs_200d_ma", "value")``). ``threshold_field`` is a ``PatternConfig``
attribute, or ``None`` where the gate is a computed/event condition with no single dial.
The test suite asserts every ``trace_key`` resolves on a real fire and every
``threshold_field`` is a real ``PatternConfig`` field, so the map cannot silently drift
from the detectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# metric_type vocabulary
MONOTONIC_NUMERIC = "monotonic_numeric"
EVENT_RECURRENCE = "event_recurrence"
MULTI_AXIS = "multi_axis"
PROXY_ONLY = "proxy_only"
METRIC_TYPES = frozenset({MONOTONIC_NUMERIC, EVENT_RECURRENCE, MULTI_AXIS, PROXY_ONLY})

# direction vocabulary — how movement of the axis relates to firing.
HIGHER_FIRES = "higher_fires"   # value rising past the threshold fires (captured, NAV%, gains)
LOWER_FIRES = "lower_fires"     # value falling past the threshold fires (losses, DTE, idle)
SIGNED = "signed"               # threshold sign depends on the leg (P3 put vs call break)
EVENT = "event"                 # a calendar event / boolean, not a drifting number
NOT_APPLICABLE = "n/a"          # categorical or contextual axis with no numeric threshold
DIRECTIONS = frozenset({HIGHER_FIRES, LOWER_FIRES, SIGNED, EVENT, NOT_APPLICABLE})


@dataclass(frozen=True)
class Axis:
    """One condition behind a fire: where its value lives in the trace, the dial it is
    measured against (or None for a computed/event gate), the firing direction, a
    short human label for the secondary/sub-case axes, and — for structurally
    BOUNDED metrics (captured_pct can never exceed 1.0) — the ceiling, so the
    material-change margin shrinks to the remaining headroom near it instead of
    demanding a relative move the metric cannot make."""
    trace_key: tuple
    threshold_field: Optional[str]
    direction: str
    label: str = ""
    bound: Optional[float] = None


@dataclass(frozen=True)
class HeadlineMetric:
    pattern_id: str
    metric_type: str
    primary: Axis                       # the single headline axis
    secondary: tuple = ()               # additional axes: gates, sub-cases, event econ
    note: str = ""
    # For event_recurrence patterns only: a trace-key path to the EVENT DATE (which cycle),
    # since the primary/secondary axes point at countdown numbers, not event identity. The
    # material-change comparison re-surfaces when the current event date is newer than the
    # captured one. Lives in template_variables (raw values), so it round-trips through the
    # default=str-serialized captured_trace as an ISO/date string. None for non-event types.
    event_id_key: Optional[tuple] = None


# pattern_id -> HeadlineMetric. Grounded in each detector's actual fire_result / trace
# inputs (see pm/insight/patterns.py); validated by the test suite against real fires.
HEADLINE_METRICS: dict[str, HeadlineMetric] = {
    "P1": HeadlineMetric(
        "P1", MONOTONIC_NUMERIC,
        Axis(("result", "captured_pct"), "p1_captured_min", HIGHER_FIRES, bound=1.0),
    ),
    "P2": HeadlineMetric(
        "P2", MULTI_AXIS,
        Axis(("result", "captured_pct"), "p2_captured_min", HIGHER_FIRES, bound=1.0),
        secondary=(
            Axis(("result", "iv_pctl"), "p2_iv_pctl_min", HIGHER_FIRES, "IV percentile gate"),
        ),
        note="Captured premium is the economic spine; the IV-percentile gate distinguishes P2 from P1.",
    ),
    "P3": HeadlineMetric(
        "P3", MULTI_AXIS,
        Axis(("inputs", "spot_vs_200d_ma", "value"), "p3_200d_break_threshold", SIGNED,
             "200d MA break (threshold sign flips put vs call)"),
        secondary=(
            Axis(("inputs", "option_captured_pct", "value"), "p3_captured_min", HIGHER_FIRES,
                 "captured gate"),
            Axis(("inputs", "return_horizons", "value"), "p3_return_5d_threshold", SIGNED,
                 "5-session move (return_5d within return_horizons)"),
        ),
        note="Direction-dependent: short put fires on a break down, short call on a break up.",
    ),
    "P4": HeadlineMetric(
        "P4", EVENT_RECURRENCE,
        Axis(("result", "captured_pct"), "p4_captured_min", HIGHER_FIRES, "captured gate"),
        secondary=(
            Axis(("inputs", "analyst_note_recent", "value"), None, EVENT,
                 "recent analyst note — the defining catalyst (re-surface on a new note)"),
        ),
        note="The trigger is a fresh analyst note (an event), not a drifting number.",
        event_id_key=("template_variables", "analyst_note_date"),
    ),
    "P5": HeadlineMetric(
        "P5", MULTI_AXIS,
        Axis(("result", "captured_pct"), "p5_captured_min", HIGHER_FIRES, bound=1.0),
        secondary=(
            Axis(("result", "dte"), "p5_dte_max", LOWER_FIRES,
                 "roll window — closer to expiry fires (monotonic time decay)"),
        ),
    ),
    "P6": HeadlineMetric(
        "P6", MONOTONIC_NUMERIC,
        Axis(("result", "pnl_pct"), "p6_pnl_pct_max", LOWER_FIRES, "path A loss (with time pressure)"),
        secondary=(
            Axis(("result", "pnl_pct"), "p6_extreme_pnl_pct_max", LOWER_FIRES,
                 "path B extreme loss (any expiry)"),
            Axis(("inputs", "option_dte", "value"), "p6_dte_max", LOWER_FIRES, "path A time gate"),
        ),
        note="P&L% is the headline; path A pairs it with a DTE gate, path B is P&L-only.",
    ),
    "P7": HeadlineMetric(
        "P7", EVENT_RECURRENCE,
        Axis(("inputs", "option_moneyness", "value"), None, HIGHER_FIRES, "depth ITM"),
        secondary=(
            Axis(("result", "extrinsic_estimate"), None, NOT_APPLICABLE,
                 "extrinsic estimate (the early-exercise economics: extrinsic < dividend)"),
            Axis(("result", "dividend"), None, NOT_APPLICABLE, "dividend amount"),
            Axis(("inputs", "days_to_ex_div", "value"), "p7_exdiv_window_days", EVENT,
                 "ex-div window"),
        ),
        note="Event-windowed: assignment economics around the ex-dividend date, not a metric drifting.",
        event_id_key=("template_variables", "ex_div_date"),
    ),
    "P8": HeadlineMetric(
        "P8", PROXY_ONLY,
        Axis(("inputs", "position.unrealized_pnl_pct", "value"), "p8_residual_pnl_pct_max",
             LOWER_FIRES, "residual leg P&L"),
        note="No clean single metric — the trigger is a roll-timing asymmetry; residual P&L% is a proxy.",
    ),
    "P9": HeadlineMetric(
        "P9", MONOTONIC_NUMERIC,
        Axis(("result", "nav_pct"), "p9_nav_pct_min", HIGHER_FIRES),
        note="Headline is size-of-NAV; the freshness window (p9_fresh_window_days) is a locked, "
             "self-expiring time bound, not a sensitivity dial.",
    ),
    "P10": HeadlineMetric(
        "P10", MONOTONIC_NUMERIC,
        Axis(("result", "pnl_pct"), "p10_pnl_pct_min", HIGHER_FIRES),
    ),
    "P11": HeadlineMetric(
        "P11", MULTI_AXIS,
        Axis(("result", "cash_pct"), "p11_cash_pct_min", HIGHER_FIRES),
        secondary=(
            Axis(("result", "days_since_trade"), "p11_idle_days_min", HIGHER_FIRES,
                 "idle window — a new trade resets it"),
        ),
        note="Account-level; cash share is the redeploy headline, idle days is partly event-reset.",
    ),
    "P12": HeadlineMetric(
        "P12", MONOTONIC_NUMERIC,
        Axis(("result", "nav_pct"), "p12_underlying_nav_pct_min", HIGHER_FIRES,
             "underlying-summed case"),
        secondary=(
            Axis(("result", "nav_pct"), "p12_single_position_nav_pct_min", HIGHER_FIRES,
                 "single-equity case"),
        ),
        note="Same headline (share of NAV) measured against two thresholds; result.case says which.",
    ),
    "P13": HeadlineMetric(
        "P13", MULTI_AXIS,
        Axis(("result", "iv_pctl"), "p13_iv_pctl_min", HIGHER_FIRES),
        secondary=(
            Axis(("result", "regime"), None, NOT_APPLICABLE, "MA stack regime (categorical gate)"),
            Axis(("result", "rsi_regime"), None, NOT_APPLICABLE, "RSI regime (categorical gate)"),
        ),
        note="IV percentile is the numeric headline; the trend/RSI gates are categorical state.",
    ),
    "P14": HeadlineMetric(
        "P14", EVENT_RECURRENCE,
        Axis(("result", "days_to_earnings"), "p14_earnings_window_days", EVENT,
             "earnings countdown"),
        secondary=(
            Axis(("result", "iv_pctl"), "p14_iv_pctl_min", HIGHER_FIRES, "IV percentile gate (OR)"),
            Axis(("result", "term"), "p14_term_structure_min", HIGHER_FIRES, "term-structure gate (OR)"),
        ),
        note="Earnings date is the defining event; the vol condition is an OR of two gates.",
        event_id_key=("template_variables", "earnings_date"),
    ),
    "P15": HeadlineMetric(
        "P15", EVENT_RECURRENCE,
        Axis(("result", "vol_units"), "p15_vol_multiplier_min", HIGHER_FIRES),
        note="A single-day EVENT, not a drifting number: each new qualifying move "
             "day is a new episode, so a muted alert re-surfaces on the next one "
             "(the old monotonic classification compared σ sizes across unrelated "
             "days — a permanent mute swallowed every future move below the "
             "baseline's size).",
        event_id_key=("template_variables", "move_date"),
    ),
    "P21": HeadlineMetric(
        "P21", EVENT_RECURRENCE,
        Axis(("result", "days_since_note"), "p21_note_window_bd", LOWER_FIRES,
             "business days since the note"),
        note="The note DATE is the event; a newer note on the name re-surfaces "
             "a muted alert. Direction (rating/target moved vs the prior note) "
             "arrives with the dated snapshot store.",
        event_id_key=("template_variables", "analyst_note_date"),
    ),
    "P22": HeadlineMetric(
        "P22", EVENT_RECURRENCE,
        Axis(("result", "bd_to_expiry"), "p22_expiry_window_bd", LOWER_FIRES,
             "business days to expiry"),
        secondary=(
            Axis(("result", "captured_pct"), "p5_captured_min", LOWER_FIRES,
                 "band gate: captured below the P5 close-out threshold"),
            Axis(("result", "pnl_pct"), "p6_pnl_pct_max", HIGHER_FIRES,
                 "band gate: P&L above the P6 stress threshold"),
        ),
        note="An expiry review is an episode bounded by its contract's expiry "
             "date: muting one silences it through that bell, and a rolled "
             "position re-entering the window (a NEW expiry, still in the band) "
             "re-surfaces. The band gates are P5/P6's own dials, not copies.",
        event_id_key=("template_variables", "expiry"),
    ),
    # ------------------------------------------------------------------
    # Structure fires (P16–P20). Their thresholds are structure_fires.py
    # module constants, not PatternConfig dials, so threshold_field is None
    # throughout; trace values live under ("inputs", <key>, "value") because a
    # structure fire's trace "result" is a string, not a dict. These entries
    # exist so a muted/acknowledged tier-1 structure fire has a way back — a
    # deepening breach or a NEW expiry episode re-surfaces it instead of
    # staying invisible forever.
    # ------------------------------------------------------------------
    "P16": HeadlineMetric(
        "P16", MONOTONIC_NUMERIC,
        Axis(("inputs", "naked_excess_contracts", "value"), None, HIGHER_FIRES,
             "uncovered short-call contracts"),
        note="A deepening coverage breach (more naked contracts than at mute time) re-surfaces.",
    ),
    "P17": HeadlineMetric(
        "P17", EVENT_RECURRENCE,
        Axis(("inputs", "extrinsic", "value"), None, LOWER_FIRES, "remaining time value"),
        note="A carry-risk episode is bounded by its leg's expiry; a NEW expiry "
             "(rolled or re-struck put back in carry breach) re-surfaces.",
        event_id_key=("inputs", "leg_expiry", "value"),
    ),
    "P18": HeadlineMetric(
        "P18", EVENT_RECURRENCE,
        Axis(("inputs", "spot", "value"), None, HIGHER_FIRES, "spot vs cap"),
        note="At-cap is an episode per short-call expiry; a rolled call capping "
             "again is a new episode and re-surfaces.",
        event_id_key=("inputs", "leg_expiry", "value"),
    ),
    "P19": HeadlineMetric(
        "P19", EVENT_RECURRENCE,
        Axis(("inputs", "dte", "value"), None, LOWER_FIRES, "days to expiry"),
        note="Pin risk is an expiry-day episode; a NEW expiry pinning again "
             "re-surfaces (the same episode stays muted through the bell).",
        event_id_key=("inputs", "leg_expiry", "value"),
    ),
    "P20": HeadlineMetric(
        "P20", EVENT_RECURRENCE,
        Axis(("inputs", "extrinsic", "value"), None, LOWER_FIRES, "put time value"),
        note="Monetize-the-put is an episode per put expiry; a rolled protective "
             "put hollowing out again is a new episode and re-surfaces.",
        event_id_key=("inputs", "leg_expiry", "value"),
    ),
}
