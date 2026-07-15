"""V1 pattern detectors — 15 patterns.

Each detector takes a ``PositionContext``-like bundle and a
``PatternConfig`` and returns a ``Fire`` (or None / list[Fire] for
account-level detectors). The Fire's trace satisfies the canonical
trace schema.

Convention:
- ``detect_pN(position, account_state, signals, config) -> Fire | None``
  for per-position patterns.
- ``detect_pN_account(account_state, config) -> list[Fire]`` for
  account-level patterns (P8, P11, P12, P14).

P4 reads D3 ``analyst_note_recent``: it fires when a captured-premium
short option also has a recent analyst note. Offline (or on names with
no analyst coverage) D3 is stale, so P4 stays silent there.

Variables that require live option pricing render as
``INDICATIVE_PLACEHOLDER`` in V1; the trace records the substitution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from pm.ingest.position_builder import Position
from pm.insight import templates as T
from pm.insight.signal_library import SignalDict, SignalValue


# ---------------------------------------------------------------------------
# Fire object
# ---------------------------------------------------------------------------

@dataclass
class SuppressionMark:
    """Marks a Fire's suppression state (``resurfaced`` is set by material-change re-surfacing).

    Attached by ``suppression_store`` in the load path *after* the engine has produced
    the fire — the fire is never removed from ``acc.fires``, so a restore (or snooze
    expiry) plus a re-apply brings it straight back (the no-recompute contract).

    Five kinds (``is_active`` = ``None`` or ``"resurfaced"``; everything else inactive —
    re-surfaced never looks like a plain snooze-expiry):
      - ``"suppressed"`` / ``"snoozed"`` — muted name-wide (``is_active`` False).
      - ``"acknowledged"`` — this ONE fire (keyed structure_id-or-position_id) was
        reviewed and set aside; inactive, but its captured baseline stays under the
        material-change net, so a materially moved condition flips it back to
        ``"resurfaced"``. Set by the alert-governance pass.
      - ``"disabled"`` — the whole pattern is toggled off in the Alert Manager;
        inactive, counted, recoverable, and FINAL (applied last — material change never
        overrides an explicit off-switch). Set by the alert-governance pass.
      - ``None`` on ``Fire.suppression`` — active (never muted, or snooze expired).
      - ``"resurfaced"`` — muted/acknowledged, but the condition moved materially since
        the baseline was captured, so it is **active again** (``is_active`` True) and
        carries the before→after delta so the surfaces can say *why* it came back.
    """
    kind: str                      # "suppressed" | "snoozed" | "acknowledged" | "disabled" | "resurfaced"
    until: Optional[str] = None    # 'YYYY-MM-DD' when snoozed; None otherwise
    # Set only when kind == "resurfaced" (material-change re-surfacing). The headline
    # metric and its baseline→current values, so the blotter / Alert Manager can show the
    # move. captured/current are numbers (numeric metrics) or ISO date strings (event metrics).
    metric: Optional[str] = None
    captured_value: Optional[Any] = None
    current_value: Optional[Any] = None


@dataclass
class Fire:
    pattern_id: str
    pattern_name: str
    tier: int
    account: str
    position_id: str
    underlying: str
    asset_class: str
    label: str
    rationale: str
    trace: dict[str, Any]
    fired_at: datetime
    skipped: bool = False
    skip_reason: Optional[str] = None
    # Set on the structure-aware fires (P16–P20) to their originating structure's
    # id, so confirming/rejecting a structure can add or remove exactly that
    # structure's fires. None on the per-position / account patterns (P1–P15).
    structure_id: Optional[str] = None
    # The rationale before any structure-leg context is appended, captured once so
    # the annotation can be rebuilt from a clean base — re-running never doubles it.
    rationale_base: Optional[str] = None
    # Set by suppression_store.apply_suppressions when an active suppression
    # matches (account, underlying, pattern_id). None = active alert. The fire stays
    # in acc.fires either way — active surfaces filter it out (suppression_store.
    # is_active), the modal's Muted footer and the Alert Manager show it to restore.
    suppression: Optional[SuppressionMark] = None


# ---------------------------------------------------------------------------
# PatternConfig — all pattern thresholds (p9 uses a calendar-day approximation
# of the "≤10 business days" rule; see p9_fresh_window_days).
# ---------------------------------------------------------------------------

@dataclass
class PatternConfig:
    # P1
    p1_captured_min: float = 0.75
    # P2
    p2_captured_min: float = 0.60
    p2_iv_pctl_min: float = 70.0
    # P3
    p3_captured_min: float = 0.60
    p3_200d_break_threshold: float = -0.02
    p3_return_5d_threshold: float = -0.03
    # P4
    p4_captured_min: float = 0.60
    # P5
    p5_dte_max: int = 45
    p5_captured_min: float = 0.70
    # P6
    p6_pnl_pct_max: float = -1.0          # ≤ −100%
    p6_dte_max: int = 60
    p6_extreme_pnl_pct_max: float = -2.0  # ≤ −200%
    # P7
    p7_exdiv_window_days: int = 7
    # P8 — business days (the spec's "within last 5 business days"; the
    # detector counts via bdate_range, weekend-safe).
    p8_recent_trade_window_days: int = 5
    p8_residual_pnl_pct_max: float = -1.0
    # P9 — calendar-days approximation of "≤10 business days".
    # Position.days_held is calendar days; 14 calendar ≈ 10 BD.
    p9_fresh_window_days: int = 14
    p9_nav_pct_min: float = 0.05
    # P10
    p10_pnl_pct_min: float = 2.0
    # P11
    p11_idle_days_min: int = 15
    p11_cash_pct_min: float = 0.05
    # P12
    p12_single_position_nav_pct_min: float = 0.25
    p12_underlying_nav_pct_min: float = 0.30
    # P13
    p13_iv_pctl_min: float = 70
    # P14
    p14_earnings_window_days: int = 14
    p14_iv_pctl_min: float = 60
    p14_term_structure_min: float = 2.0
    # P15
    p15_vol_multiplier_min: float = 1.5   # threshold for "Notable price move" alert


# ---------------------------------------------------------------------------
# Tier mapping: pattern_id -> (pattern_name, tier)
# ---------------------------------------------------------------------------

PATTERN_META: dict[str, tuple[str, int]] = {
    # pattern_id -> (pattern_name, tier)
    "P1": ("Captured short premium, ready to close", 2),
    "P2": ("Captured + IV elevated → close-and-rewrite", 2),
    "P3": ("Captured + adverse technical break", 1),
    "P4": ("Captured + analyst rating change", 1),
    "P5": ("Roll-due short option (three-roads)", 1),
    "P6": ("Stress position (deep underwater + time pressure)", 1),
    "P7": ("ITM short call + ex-div trap", 1),
    "P8": ("Recent roll asymmetry (multi-leg structure)", 1),
    "P9": ("Fresh significant position", 2),
    "P10": ("Big winner, partial monetization candidate", 2),
    "P11": ("Idle account, redeploy candidate", 2),
    "P12": ("Concentration", 2),
    "P13": ("Vol-rich covered-call setup", 3),
    "P14": ("Earnings catalyst setup", 3),
    "P15": ("Notable price move", 1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_value(signals: SignalDict, signal_id: str) -> Any:
    sv = signals.get(signal_id)
    if sv is None or sv.stale:
        return None
    return sv.value


def _signal_present(signals: SignalDict, signal_id: str) -> bool:
    sv = signals.get(signal_id)
    return sv is not None and not sv.stale and sv.value is not None


def _build_trace(
    *,
    inputs_from_signals: list[str],
    signals: SignalDict,
    position: Position,
    position_fields: Optional[list[str]] = None,
    config: PatternConfig,
    thresholds_used: dict[str, Any],
    computation: str,
    fire_result: dict[str, Any],
    template_variables: dict[str, Any],
    extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for sid in inputs_from_signals:
        sv = signals.get(sid)
        if sv is None:
            inputs[sid] = {"value": None, "source": "computed:not-available",
                            "as_of": None, "stale": True}
        else:
            inputs[sid] = {
                "value": sv.value,
                "source": f"signal:{sid}",
                "as_of": None,
                "stale": sv.stale,
            }
    for f in position_fields or []:
        inputs[f"position.{f}"] = {
            "value": getattr(position, f, None),
            "source": f"EXTRACT:Position.{f}",
            "as_of": None,
            "stale": False,
        }

    return {
        "inputs": inputs,
        "computation": computation,
        "thresholds": thresholds_used,
        "result": fire_result,
        "template_variables": template_variables,
        "template_extras": extras or {},
    }


def _is_short_option(position: Position) -> bool:
    return (position.asset_class == "option"
            and position.quantity is not None
            and position.quantity < 0)


def _is_long_option(position: Position) -> bool:
    return (position.asset_class == "option"
            and position.quantity is not None
            and position.quantity > 0)


def _now() -> datetime:
    return datetime.now()


def _make_fire(
    *,
    pattern_id: str,
    position: Position,
    account_state,
    label: str,
    rationale: str,
    trace: dict[str, Any],
) -> Fire:
    name, tier = PATTERN_META[pattern_id]
    return Fire(
        pattern_id=pattern_id,
        pattern_name=name,
        tier=tier,
        account=account_state.account,
        position_id=position.position_id if position else f"{account_state.account}:account",
        underlying=position.underlying_symbol or position.symbol if position else "",
        asset_class=position.asset_class if position else "account",
        label=label,
        rationale=rationale,
        trace=trace,
        fired_at=_now(),
    )


# ===========================================================================
# P1 — Captured short premium, ready to close
# ===========================================================================

def detect_p1(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if not _is_short_option(position):
        return None
    captured = _signal_value(signals, "option_captured_pct")
    if captured is None:
        return None
    if captured < config.p1_captured_min:
        return None

    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    extras = {}  # all template variables resolve from the registry
    label, label_vars = T.resolve_variables(T.P1_LABEL_TEMPLATE, ctx, extras)
    rationale, rationale_vars = T.resolve_variables(T.P1_RATIONALE_TEMPLATE, ctx, extras)
    template_vars = {**label_vars, **rationale_vars}

    trace = _build_trace(
        inputs_from_signals=["option_captured_pct", "option_dte"],
        signals=signals,
        position=position,
        position_fields=["cost_basis", "market_value", "quantity", "expiry"],
        config=config,
        thresholds_used={"captured_pct_min": config.p1_captured_min},
        computation="captured_pct = (|cost_basis| - |market_value|) / |cost_basis|",
        fire_result={"captured_pct": captured, "fired": True},
        template_variables=template_vars,
    )
    return _make_fire(pattern_id="P1", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P2 — Captured + IV elevated → close-and-rewrite
# ===========================================================================

def detect_p2(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if not _is_short_option(position):
        return None
    captured = _signal_value(signals, "option_captured_pct")
    iv_pctl = _signal_value(signals, "iv_3m_percentile_1y")
    if captured is None or iv_pctl is None:
        return None
    if captured < config.p2_captured_min or iv_pctl < config.p2_iv_pctl_min:
        return None

    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    extras = {"next_tenor": T.INDICATIVE_PLACEHOLDER}
    label, lv = T.resolve_variables(T.P2_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(T.P2_RATIONALE_TEMPLATE, ctx, extras)

    trace = _build_trace(
        inputs_from_signals=["option_captured_pct", "iv_3m_percentile_1y", "iv_3m_atm"],
        signals=signals,
        position=position,
        position_fields=["cost_basis", "market_value"],
        config=config,
        thresholds_used={
            "captured_pct_min": config.p2_captured_min,
            "iv_pctl_min": config.p2_iv_pctl_min,
        },
        computation="captured ≥ 60% AND iv_3m_percentile_1y ≥ 70",
        fire_result={"captured_pct": captured, "iv_pctl": iv_pctl, "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P2", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P3 — Captured + adverse technical break
# ===========================================================================

def detect_p3(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if not _is_short_option(position):
        return None
    captured = _signal_value(signals, "option_captured_pct")
    spot_vs_200d = _signal_value(signals, "spot_vs_200d_ma")
    rh = _signal_value(signals, "return_horizons")
    return_5d = rh.get("return_5d") if isinstance(rh, dict) else None

    if captured is None or spot_vs_200d is None or return_5d is None:
        return None
    if captured < config.p3_captured_min:
        return None

    right = (position.right or "").upper()
    direction_word = None

    if right == "PUT":
        if spot_vs_200d >= config.p3_200d_break_threshold:
            return None
        if return_5d >= config.p3_return_5d_threshold:
            return None
        direction_word = "weakening"
        template = T.P3_RATIONALE_TEMPLATE_PUT
    elif right == "CALL":
        # For short call: underlying strengthening
        if spot_vs_200d <= -config.p3_200d_break_threshold:  # ≤+0.02
            return None
        if return_5d <= -config.p3_return_5d_threshold:  # ≤+0.03
            return None
        moneyness = _signal_value(signals, "option_moneyness")
        if moneyness is None or moneyness <= -0.10:  # call too far OTM = not "moving toward ITM"
            return None
        direction_word = "strengthening"
        template = T.P3_RATIONALE_TEMPLATE_CALL
    else:
        return None

    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    pct_dist = (spot_vs_200d if right == "PUT" else spot_vs_200d)
    extras = {
        "direction_word": direction_word,
        "ma_200d_cross_date": T.INDICATIVE_PLACEHOLDER,
        "pct_below_200d": -spot_vs_200d if spot_vs_200d < 0 else 0,
        "pct_above_200d": spot_vs_200d if spot_vs_200d > 0 else 0,
    }
    label, lv = T.resolve_variables(T.P3_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(template, ctx, extras)

    trace = _build_trace(
        inputs_from_signals=[
            "option_captured_pct", "spot_vs_200d_ma", "return_horizons",
            "option_moneyness",
        ],
        signals=signals,
        position=position,
        position_fields=["right"],
        config=config,
        thresholds_used={
            "captured_pct_min": config.p3_captured_min,
            "spot_vs_200d_threshold": config.p3_200d_break_threshold,
            "return_5d_threshold": config.p3_return_5d_threshold,
        },
        computation=("PUT: spot_vs_200d ≤ −0.02 AND return_5d ≤ −0.03; "
                      "CALL: spot_vs_200d ≥ +0.02 AND return_5d ≥ +0.03"),
        fire_result={"direction_word": direction_word, "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P3", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P4 — Captured + recent analyst note  (D3 analyst_note_recent)
# ===========================================================================

def detect_p4(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if not _is_short_option(position):
        return None
    captured = _signal_value(signals, "option_captured_pct")
    if captured is None or captured < config.p4_captured_min:
        return None
    # D3 is stale offline / on names with no analyst coverage.
    if not _signal_present(signals, "analyst_note_recent"):
        return None
    note = _signal_value(signals, "analyst_note_recent")
    is_recent = note.get("is_recent") if isinstance(note, dict) else False
    if not is_recent:
        return None

    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P4_LABEL_TEMPLATE, ctx)
    rationale, rv = T.resolve_variables(T.P4_RATIONALE_TEMPLATE, ctx)
    trace = _build_trace(
        inputs_from_signals=["option_captured_pct", "analyst_note_recent",
                              "analyst_rating_and_target"],
        signals=signals,
        position=position,
        config=config,
        thresholds_used={"captured_pct_min": config.p4_captured_min},
        computation=("captured ≥ 60% AND analyst note is recent "
                      "(today or previous business day)"),
        fire_result={"captured_pct": captured,
                      "days_since_note": note.get("days_since") if isinstance(note, dict) else None,
                      "fired": True},
        template_variables={**lv, **rv},
    )
    return _make_fire(pattern_id="P4", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P5 — Roll-due short option (three-roads)
# ===========================================================================

def detect_p5(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if not _is_short_option(position):
        return None
    captured = _signal_value(signals, "option_captured_pct")
    dte = _signal_value(signals, "option_dte")
    if captured is None or dte is None:
        return None
    if dte > config.p5_dte_max or captured < config.p5_captured_min:
        return None

    # Determine direction words from option type + moneyness.
    right = (position.right or "").upper()
    direction_words = "up" if right == "CALL" else "down"

    iv_pctl = _signal_value(signals, "iv_3m_percentile_1y")
    # Best recommendation derived from vol regime
    if iv_pctl is None:
        best_rec = "review on terminal — IV percentile unavailable"
    elif iv_pctl >= 70:
        best_rec = "Close and rewrite at currently rich IV"
    elif iv_pctl <= 30:
        best_rec = "Hold theta; rewrite when IV firms"
    else:
        best_rec = "Roll out same strike to harvest more theta"

    extras = {
        "next_tenor": T.INDICATIVE_PLACEHOLDER,
        "roll_out_credit": T.INDICATIVE_PLACEHOLDER,
        "roll_wider_credit": T.INDICATIVE_PLACEHOLDER,
        "suggested_strike": T.INDICATIVE_PLACEHOLDER,
        "direction_words": direction_words,
        "best_recommendation": best_rec,
    }

    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P5_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(T.P5_RATIONALE_TEMPLATE, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["option_captured_pct", "option_dte", "option_moneyness",
                              "iv_3m_atm", "iv_3m_percentile_1y"],
        signals=signals,
        position=position,
        position_fields=["right", "expiry", "strike"],
        config=config,
        thresholds_used={"dte_max": config.p5_dte_max,
                          "captured_pct_min": config.p5_captured_min},
        computation="dte ≤ 45 AND captured ≥ 70%",
        fire_result={"dte": dte, "captured_pct": captured, "fired": True,
                      "best_recommendation": best_rec},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P5", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P6 — Stress position (deep underwater + time pressure)
# ===========================================================================

def detect_p6(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if position.asset_class != "option":
        return None
    pnl_pct = _signal_value(signals, "position_unrealized_pnl_pct")
    dte = _signal_value(signals, "option_dte")
    if pnl_pct is None:
        return None
    # Path A: pnl ≤ -100% AND dte ≤ 60 (short option only)
    # Path B: pnl ≤ -200%, any DTE (any option)
    path_a = (_is_short_option(position) and pnl_pct <= config.p6_pnl_pct_max
              and dte is not None and dte <= config.p6_dte_max)
    path_b = pnl_pct <= config.p6_extreme_pnl_pct_max
    if not (path_a or path_b):
        return None

    iv_3m = _signal_value(signals, "iv_3m_atm")
    # Template uses both standard and no-IV variants
    template = T.P6_RATIONALE_TEMPLATE if iv_3m is not None else T.P6_RATIONALE_TEMPLATE_NO_IV

    qty = position.quantity or 0
    strike = position.strike or 0
    capital_at_risk = abs(qty) * (position.multiplier or 100) * strike if (
        (position.right or "").upper() == "PUT") else "<not applicable for short calls>"

    extras = {
        "suggested_strike": T.INDICATIVE_PLACEHOLDER,
        "suggested_expiry": T.INDICATIVE_PLACEHOLDER,
        "capital_at_risk": capital_at_risk,
    }
    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P6_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(template, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["position_unrealized_pnl_pct", "option_dte",
                              "option_moneyness", "iv_3m_atm"],
        signals=signals,
        position=position,
        position_fields=["right", "strike", "quantity", "market_value"],
        config=config,
        thresholds_used={
            "pnl_pct_max_path_a": config.p6_pnl_pct_max,
            "dte_max_path_a": config.p6_dte_max,
            "pnl_pct_max_path_b": config.p6_extreme_pnl_pct_max,
        },
        computation=("path A: short option, pnl ≤ −100%, dte ≤ 60; "
                      "path B: any option, pnl ≤ −200%"),
        fire_result={"path": "A" if path_a else "B", "pnl_pct": pnl_pct,
                      "iv_available": iv_3m is not None, "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P6", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P7 — ITM short call + ex-div trap
# ===========================================================================

def detect_p7(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if not _is_short_option(position):
        return None
    if (position.right or "").upper() != "CALL":
        return None
    moneyness = _signal_value(signals, "option_moneyness")
    days_to_exdiv = _signal_value(signals, "days_to_ex_div")
    if moneyness is None or days_to_exdiv is None:
        return None
    if moneyness <= 0:  # not ITM
        return None
    if not (0 <= days_to_exdiv <= config.p7_exdiv_window_days):
        return None

    spot = T.TEMPLATE_VARIABLE_REGISTRY["spot"](
        T.TemplateContext(position=position, account_state=account_state,
                            signals=signals, config=config)
    )
    if spot is None:
        return None

    # Dividend amount: prefer a real Bloomberg forward projection; else estimate
    # from DVD_YLD (annual yield × spot / 4, quarterly). Both ride in the
    # days_to_ex_div / spot_vs_50d_ma traces, recorded upstream in the library.
    def _from_trace(input_name: str) -> Any:
        for sid in ("days_to_ex_div", "spot_vs_50d_ma"):
            sv = signals.get(sid)
            entry = ((getattr(sv, "trace", {}) or {}).get("inputs", {}).get(input_name)
                     if sv else None)
            if entry and entry.get("value") is not None:
                return entry.get("value")
        return None

    projected_dps = _from_trace("projected_dividend")
    if projected_dps is not None:
        try:
            dividend_amount = float(projected_dps)
        except (TypeError, ValueError):
            return None
        dividend_source = "projected"
    else:
        dvd_yld = _from_trace("DVD_YLD")
        if dvd_yld is None:
            return None
        try:
            dividend_amount = float(spot) * float(dvd_yld) / 100.0 / 4.0  # quarterly heuristic
        except (TypeError, ValueError):
            return None
        dividend_source = "heuristic"

    # Estimate extrinsic per contract.
    mv = position.market_value or 0
    qty = position.quantity or 0
    multiplier = position.multiplier or 100
    if qty == 0:
        return None
    mv_per_contract = abs(mv) / abs(qty) / multiplier
    intrinsic = max(0.0, float(spot) - float(position.strike or 0))
    extrinsic = max(0.0, mv_per_contract - intrinsic)

    if extrinsic >= dividend_amount:
        return None

    extras = {
        "dividend_amount": round(dividend_amount, 2),
        "extrinsic": round(extrinsic, 2),
    }
    # Source-aware copy: a Bloomberg projection is defensible as projected; the
    # yield estimate must say it is estimated (surface-the-uncertainty standard).
    template = (T.P7_RATIONALE_TEMPLATE if dividend_source == "projected"
                else T.P7_RATIONALE_TEMPLATE_HEURISTIC)
    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P7_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(template, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["option_moneyness", "days_to_ex_div", "iv_3m_atm"],
        signals=signals,
        position=position,
        position_fields=["strike", "market_value", "quantity"],
        config=config,
        thresholds_used={"exdiv_window_days": config.p7_exdiv_window_days},
        computation=("ITM short call AND ex-div ≤7 BD AND extrinsic estimate "
                      "(max(0, |MV|/(qty×100) − max(0, spot−strike))) < dividend "
                      "(Bloomberg projection when available, else spot × DVD_YLD/100 / 4)"),
        fire_result={
            "extrinsic_estimate": extrinsic, "dividend": dividend_amount,
            "dividend_source": dividend_source, "fired": True,
        },
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P7", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P8 — Recent roll asymmetry (multi-leg structure)  [account-level]
# ===========================================================================

def detect_p8_account(account_state, config: PatternConfig) -> list[Fire]:
    fires: list[Fire] = []
    # Group option positions by underlying_symbol
    by_under: dict[str, list[Position]] = {}
    for p in account_state.positions:
        if p.asset_class != "option" or p.quantity is None or p.quantity >= 0:
            continue
        if not p.underlying_symbol:
            continue
        by_under.setdefault(p.underlying_symbol, []).append(p)

    today = date.today()

    def _bdays_ago(d: date) -> int:
        # Business days between the trade date and today (weekend-safe): the
        # spec's window is "5 business days", so a Friday trade is 1 BD old on
        # Monday, not 3. Same convention as the signal library's
        # _safe_business_days_until (len(bdate_range) - 1).
        return int(len(pd.bdate_range(d, today)) - 1) if d <= today else 0

    for under, legs in by_under.items():
        if len(legs) < 2:
            continue
        trades = account_state.trades_by_underlying.get(under)
        if trades is None or trades.empty:
            continue

        # Identify which legs have trades within the window
        recent_legs: list[Position] = []
        residual_legs: list[Position] = []
        for leg in legs:
            leg_trades = trades[trades["option_contract_key"] == leg.option_contract_key]
            if leg_trades.empty:
                # No trades at all on this leg → potential residual
                pnl_pct = leg.unrealized_pnl_pct
                if pnl_pct is not None and pnl_pct <= config.p8_residual_pnl_pct_max:
                    residual_legs.append(leg)
                continue
            # Check the most recent trade date
            latest_dt = leg_trades["trade_date"].max()
            if isinstance(latest_dt, pd.Timestamp):
                latest_dt = latest_dt.date()
            if latest_dt and isinstance(latest_dt, date):
                bdays_ago = _bdays_ago(latest_dt)
                if bdays_ago <= config.p8_recent_trade_window_days:
                    recent_legs.append(leg)
                else:
                    pnl_pct = leg.unrealized_pnl_pct
                    if pnl_pct is not None and pnl_pct <= config.p8_residual_pnl_pct_max:
                        residual_legs.append(leg)

        if not recent_legs or not residual_legs:
            continue
        # Pick the most recently-touched leg as the "recent" leg for context
        most_recent = max(
            recent_legs,
            key=lambda leg: trades[trades["option_contract_key"] == leg.option_contract_key]
                                ["trade_date"].max()
        )
        recent_trades = trades[trades["option_contract_key"] == most_recent.option_contract_key]
        recent_trade_date = recent_trades["trade_date"].max()
        if isinstance(recent_trade_date, pd.Timestamp):
            recent_trade_date = recent_trade_date.date()
        recent_action = ""
        action_col = recent_trades.loc[
            recent_trades["trade_date"] == recent_trades["trade_date"].max(),
            "option_lifecycle_action",
        ]
        if not action_col.empty:
            recent_action = str(action_col.iloc[0])
        days_ago = _bdays_ago(recent_trade_date) if isinstance(recent_trade_date, date) else None

        for residual in residual_legs:
            extras = {
                "recent_leg_strike": f"${most_recent.strike:.0f}{most_recent.right[0]}",
                "recent_leg_right": (most_recent.right or "").title(),
                "recent_leg_dte": (most_recent.expiry - today).days if most_recent.expiry else None,
                "recent_trade_date": recent_trade_date,
                "recent_action": recent_action,
                "days_ago": days_ago,
                "residual_leg_strike": f"${residual.strike:.0f}{residual.right[0]}",
                "residual_leg_right": (residual.right or "").title(),
                "residual_leg_dte": (residual.expiry - today).days if residual.expiry else None,
                "residual_pnl_pct": residual.unrealized_pnl_pct,
            }
            ctx = T.TemplateContext(position=residual, account_state=account_state,
                                     signals={}, config=config)
            label, lv = T.resolve_variables(T.P8_LABEL_TEMPLATE, ctx, extras)
            rationale, rv = T.resolve_variables(T.P8_RATIONALE_TEMPLATE, ctx, extras)
            trace = _build_trace(
                inputs_from_signals=[],
                signals={},
                position=residual,
                position_fields=["option_contract_key", "unrealized_pnl_pct"],
                config=config,
                thresholds_used={
                    "recent_window_days": config.p8_recent_trade_window_days,
                    "residual_pnl_pct_max": config.p8_residual_pnl_pct_max,
                },
                computation=("group shorts by underlying; "
                              "residual = no trade in 5 BD AND pnl_pct ≤ −1.0"),
                fire_result={
                    "underlying": under,
                    "n_legs": len(legs),
                    "recent_leg": most_recent.position_id,
                    "residual_leg": residual.position_id,
                    "fired": True,
                },
                template_variables={**lv, **rv},
                extras=extras,
            )
            fires.append(_make_fire(pattern_id="P8", position=residual,
                                      account_state=account_state,
                                      label=label, rationale=rationale, trace=trace))
    return fires


# ===========================================================================
# P9 — Fresh significant position
# ===========================================================================

def detect_p9(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if position.open_date is None or position.days_held is None:
        return None
    if position.days_held > config.p9_fresh_window_days:
        return None
    nav_pct = _signal_value(signals, "position_size_pct_of_nav")
    if nav_pct is None or nav_pct < config.p9_nav_pct_min:
        return None

    extras = {"breakeven": T.INDICATIVE_PLACEHOLDER}
    template = (T.P9_RATIONALE_TEMPLATE_OPTION if position.asset_class == "option"
                else T.P9_RATIONALE_TEMPLATE_EQUITY)
    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P9_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(template, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["position_size_pct_of_nav", "option_moneyness"],
        signals=signals,
        position=position,
        position_fields=["open_date", "days_held", "cost_basis", "market_value"],
        config=config,
        thresholds_used={
            "fresh_window_days": config.p9_fresh_window_days,
            "nav_pct_min": config.p9_nav_pct_min,
        },
        computation=(f"days_held ≤ {config.p9_fresh_window_days} (calendar days, "
                      "approximating ≤10 BD) AND nav_pct ≥ 5%"),
        fire_result={"days_held": position.days_held, "nav_pct": nav_pct, "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P9", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P10 — Big winner, partial monetization candidate
# ===========================================================================

def detect_p10(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if not _is_long_option(position):
        return None
    pnl_pct = _signal_value(signals, "position_unrealized_pnl_pct")
    if pnl_pct is None or pnl_pct < config.p10_pnl_pct_min:
        return None

    quarter_lock = 0.25 * (position.unrealized_pnl or 0)
    extras = {"quarter_lock": round(quarter_lock, 0)}
    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P10_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(T.P10_RATIONALE_TEMPLATE, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["position_unrealized_pnl_pct", "option_dte",
                              "option_moneyness"],
        signals=signals,
        position=position,
        position_fields=["unrealized_pnl", "cost_basis"],
        config=config,
        thresholds_used={"pnl_pct_min": config.p10_pnl_pct_min},
        computation="long option AND pnl_pct ≥ +200%",
        fire_result={"pnl_pct": pnl_pct, "quarter_lock": quarter_lock, "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P10", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P11 — Idle account, redeploy candidate  [account-level]
# ===========================================================================

def detect_p11_account(account_state, config: PatternConfig) -> list[Fire]:
    trades = account_state.trades
    today = date.today()
    if trades is None or trades.empty:
        days_since_trade: Optional[int] = None
        last_trade_dt = None
    else:
        last_trade_dt = trades["trade_date"].max()
        if isinstance(last_trade_dt, pd.Timestamp):
            last_trade_dt = last_trade_dt.date()
        days_since_trade = (today - last_trade_dt).days if isinstance(last_trade_dt, date) else None

    if days_since_trade is not None and days_since_trade < config.p11_idle_days_min:
        return []

    cash_total = sum(
        abs(p.market_value)
        for p in account_state.positions
        if p.asset_class in ("cash", "fund_etf") and p.market_value is not None
    )
    cash_pct = cash_total / account_state.nav if account_state.nav else 0.0
    if cash_pct < config.p11_cash_pct_min:
        return []

    extras = {
        "days_since_trade": days_since_trade if days_since_trade is not None else "unknown",
        "last_trade_date": last_trade_dt,
        "cash_total": cash_total,
    }
    # Synthetic position for P11 (account-level fire)
    synthetic = _SyntheticAccountPosition(account_state.account)
    ctx = T.TemplateContext(position=synthetic, account_state=account_state,
                              signals={}, config=config)
    label, lv = T.resolve_variables(T.P11_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(T.P11_RATIONALE_TEMPLATE, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=[],
        signals={},
        position=synthetic,
        config=config,
        thresholds_used={
            "idle_days_min": config.p11_idle_days_min,
            "cash_pct_min": config.p11_cash_pct_min,
        },
        computation="days_since_trade ≥ 15 AND cash_pct ≥ 5%",
        fire_result={"days_since_trade": days_since_trade, "cash_pct": cash_pct,
                      "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    name, tier = PATTERN_META["P11"]
    return [Fire(
        pattern_id="P11",
        pattern_name=name,
        tier=tier,
        account=account_state.account,
        position_id=f"{account_state.account}:idle",
        underlying="",
        asset_class="account",
        label=label,
        rationale=rationale,
        trace=trace,
        fired_at=_now(),
    )]


@dataclass
class _SyntheticAccountPosition:
    account: str
    symbol: str = ""
    right: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[date] = None
    quantity: Optional[float] = None
    asset_class: str = "account"
    underlying_symbol: str = ""
    position_id: str = ""
    cost_basis: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    open_date: Optional[date] = None
    days_held: Optional[int] = None
    last_trade_date: Optional[date] = None
    last_trade_action: Optional[str] = None
    multiplier: int = 1


# ===========================================================================
# P12 — Concentration  [account-level: per-position + per-underlying-summed]
# ===========================================================================

def detect_p12_account(account_state, config: PatternConfig) -> list[Fire]:
    fires: list[Fire] = []
    fired_underlyings: set[str] = set()
    name, tier = PATTERN_META["P12"]
    nav = abs(account_state.nav) if account_state.nav else 0

    # Per-position case: single equity ≥ 25% of NAV
    for p in account_state.positions:
        if p.asset_class != "equity":
            continue
        if p.market_value is None or nav == 0:
            continue
        nav_pct = abs(p.market_value) / nav
        if nav_pct < config.p12_single_position_nav_pct_min:
            continue
        fires.append(_p12_make_fire(account_state, p, nav_pct, config, "single-position"))
        fired_underlyings.add(p.symbol or "")

    # Per-underlying-summed case: ≥ 30% summed across legs
    by_under: dict[str, float] = {}
    largest: dict[str, Position] = {}
    for p in account_state.positions:
        if p.asset_class in ("cash", "other"):
            continue
        sym = p.underlying_symbol or p.symbol or ""
        if not sym or sym in fired_underlyings:
            continue
        if p.market_value is None:
            continue
        by_under[sym] = by_under.get(sym, 0) + abs(p.market_value)
        if sym not in largest or abs(p.market_value) > abs(largest[sym].market_value or 0):
            largest[sym] = p

    for sym, total in by_under.items():
        if nav == 0:
            continue
        nav_pct = total / nav
        if nav_pct < config.p12_underlying_nav_pct_min:
            continue
        anchor = largest[sym]
        fires.append(_p12_make_fire(account_state, anchor, nav_pct, config,
                                       "underlying-summed"))

    return fires


def _p12_make_fire(account_state, anchor: Position, nav_pct: float,
                    config: PatternConfig, case: str) -> Fire:
    # Build the signals dict for the template — pull IV signals from the
    # cached account-level signals if present.
    signals = account_state.signals.get(anchor.symbol, {}) if hasattr(
        account_state, "signals") else {}
    iv_3m = _signal_value(signals, "iv_3m_atm")

    extras = {
        "position_value": abs(anchor.market_value or 0),
        "suggested_strike": T.INDICATIVE_PLACEHOLDER,
        "suggested_tenor": T.INDICATIVE_PLACEHOLDER,
        "protective_strike": T.INDICATIVE_PLACEHOLDER,
    }
    template = T.P12_RATIONALE_TEMPLATE if iv_3m is not None else T.P12_RATIONALE_TEMPLATE_NO_IV
    ctx = T.TemplateContext(position=anchor, account_state=account_state,
                              signals=signals, config=config)
    # Override the registry's nav_pct (which reads E1) with our computed summed value
    extras["nav_pct"] = nav_pct
    label, lv = T.resolve_variables(T.P12_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(template, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["position_size_pct_of_nav", "iv_3m_atm",
                              "iv_3m_percentile_1y"],
        signals=signals,
        position=anchor,
        position_fields=["symbol", "market_value"],
        config=config,
        thresholds_used={
            "single_position_nav_pct_min": config.p12_single_position_nav_pct_min,
            "underlying_nav_pct_min": config.p12_underlying_nav_pct_min,
        },
        computation=f"case={case}; nav_pct = {nav_pct:.3f}",
        fire_result={"case": case, "nav_pct": nav_pct, "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    name, tier = PATTERN_META["P12"]
    return Fire(
        pattern_id="P12",
        pattern_name=name,
        tier=tier,
        account=account_state.account,
        position_id=anchor.position_id,
        underlying=anchor.symbol or anchor.underlying_symbol or "",
        asset_class=anchor.asset_class,
        label=label,
        rationale=rationale,
        trace=trace,
        fired_at=_now(),
    )


# ===========================================================================
# P13 — Vol-rich covered-call setup  (per long-equity position)
# ===========================================================================

def detect_p13(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    if position.asset_class != "equity":
        return None
    if position.quantity is None or position.quantity <= 0:
        return None

    iv_pctl = _signal_value(signals, "iv_3m_percentile_1y")
    regime = _signal_value(signals, "ma_stack_regime")
    rsi = _signal_value(signals, "rsi_14d_regime")

    if iv_pctl is None or regime is None or rsi is None:
        return None
    if iv_pctl < config.p13_iv_pctl_min:
        return None
    if regime not in ("bullish_aligned", "mixed", "bullish_curling"):
        return None
    rsi_regime = rsi.get("regime") if isinstance(rsi, dict) else rsi
    if rsi_regime not in ("neutral", "strong"):
        return None

    extras = {
        "trend_regime": regime,
        "suggested_strike": T.INDICATIVE_PLACEHOLDER,
        "suggested_tenor": T.INDICATIVE_PLACEHOLDER,
        "premium": T.INDICATIVE_PLACEHOLDER,
        "premium_bps_nav": T.INDICATIVE_PLACEHOLDER,
    }
    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P13_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(T.P13_RATIONALE_TEMPLATE, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["iv_3m_percentile_1y", "ma_stack_regime",
                              "rsi_14d_regime", "iv_3m_atm",
                              "position_size_pct_of_nav"],
        signals=signals,
        position=position,
        config=config,
        thresholds_used={"iv_pctl_min": config.p13_iv_pctl_min},
        computation=("long equity AND iv_pctl ≥ 70 AND regime in "
                      "{bullish_aligned, mixed, bullish_curling} AND "
                      "rsi_regime in {neutral, strong}"),
        fire_result={"iv_pctl": iv_pctl, "regime": regime, "rsi_regime": rsi_regime,
                      "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P13", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ===========================================================================
# P14 — Earnings catalyst setup  [account-level: one fire per held underlying]
# ===========================================================================

def detect_p14_account(account_state, config: PatternConfig) -> list[Fire]:
    fires: list[Fire] = []
    # Group held positions by underlying
    by_under: dict[str, list[Position]] = {}
    for p in account_state.positions:
        if p.asset_class in ("cash", "other"):
            continue
        sym = p.underlying_symbol or p.symbol
        if not sym:
            continue
        by_under.setdefault(sym, []).append(p)

    for under, held in by_under.items():
        signals = account_state.signals.get(under, {}) if hasattr(account_state, "signals") else {}
        days = _signal_value(signals, "days_to_earnings")
        if days is None or not (1 <= days <= config.p14_earnings_window_days):
            continue
        iv_pctl = _signal_value(signals, "iv_3m_percentile_1y")
        term = _signal_value(signals, "iv_term_structure")
        iv_pctl_ok = iv_pctl is not None and iv_pctl >= config.p14_iv_pctl_min
        term_ok = term is not None and term >= config.p14_term_structure_min
        if not (iv_pctl_ok or term_ok):
            continue

        anchor = max(held, key=lambda p: abs(p.market_value or 0))
        ctx = T.TemplateContext(position=anchor, account_state=account_state,
                                  signals=signals, config=config)
        label, lv = T.resolve_variables(T.P14_LABEL_TEMPLATE, ctx)
        rationale, rv = T.resolve_variables(T.P14_RATIONALE_TEMPLATE, ctx)
        trace = _build_trace(
            inputs_from_signals=["days_to_earnings", "earnings_implied_move",
                                  "iv_3m_percentile_1y", "iv_term_structure",
                                  "iv_3m_atm", "iv_6m_atm"],
            signals=signals,
            position=anchor,
            config=config,
            thresholds_used={
                "earnings_window_days": config.p14_earnings_window_days,
                "iv_pctl_min": config.p14_iv_pctl_min,
                "term_structure_min": config.p14_term_structure_min,
            },
            computation=("earnings in [1, 14] BD AND (iv_pctl ≥ 60 OR "
                          "iv_term_structure ≥ +2)"),
            fire_result={"days_to_earnings": days, "iv_pctl": iv_pctl, "term": term,
                          "fired": True},
            template_variables={**lv, **rv},
        )
        name, tier = PATTERN_META["P14"]
        fires.append(Fire(
            pattern_id="P14",
            pattern_name=name,
            tier=tier,
            account=account_state.account,
            position_id=anchor.position_id,
            underlying=under,
            asset_class=anchor.asset_class,
            label=label,
            rationale=rationale,
            trace=trace,
            fired_at=_now(),
        ))
    return fires


# ===========================================================================
# P15 — Notable price move  (per held position; underlying's vol-adjusted move)
# ===========================================================================

def detect_p15(
    position: Position, account_state, signals: SignalDict, config: PatternConfig,
) -> Optional[Fire]:
    # Eligibility: held position (equity, fund_etf, or any option) — not cash/other.
    if position.asset_class not in ("equity", "fund_etf", "option"):
        return None
    vol_units = _signal_value(signals, "vol_adjusted_move")
    if vol_units is None:
        return None
    if vol_units < config.p15_vol_multiplier_min:
        return None
    rh = _signal_value(signals, "return_horizons")
    return_1d = rh.get("return_1d") if isinstance(rh, dict) else None
    if return_1d is None:
        return None

    rv_30d = _signal_value(signals, "rv_30d")  # percent number; None when B1 stale
    days_to_earn = _signal_value(signals, "days_to_earnings")
    note = _signal_value(signals, "analyst_note_recent")

    mv = position.market_value or 0
    sign = "-" if mv < 0 else ""
    position_value_signed = f"{sign}${abs(mv):,.0f}"

    # Earnings context — emitted only when earnings are within 14 BD (per spec).
    if days_to_earn is not None and days_to_earn <= 14:
        earnings_context_sentence = f"Earnings in {days_to_earn} business days. "
    else:
        earnings_context_sentence = ""

    # Analyst-note context — emitted only when a recent note exists.
    analyst_note_context_sentence = ""
    if isinstance(note, dict) and note.get("is_recent"):
        note_date = note.get("note_date")
        note_date_str = note_date.isoformat() if hasattr(note_date, "isoformat") else note_date
        analyst_note_context_sentence = (
            f"An analyst note was published {note.get('days_since')} BD ago "
            f"({note_date_str}) — possible catalyst. "
        )

    extras = {
        "direction_word": "rallied" if return_1d > 0 else "dropped",
        "position_value_signed": position_value_signed,
        "earnings_context_sentence": earnings_context_sentence,
        "analyst_note_context_sentence": analyst_note_context_sentence,
    }
    template = (T.P15_RATIONALE_TEMPLATE if rv_30d is not None
                else T.P15_RATIONALE_TEMPLATE_NO_RV)
    ctx = T.TemplateContext(position=position, account_state=account_state,
                              signals=signals, config=config)
    label, lv = T.resolve_variables(T.P15_LABEL_TEMPLATE, ctx, extras)
    rationale, rv = T.resolve_variables(template, ctx, extras)
    trace = _build_trace(
        inputs_from_signals=["vol_adjusted_move", "return_horizons", "rv_30d",
                              "days_to_earnings", "analyst_note_recent"],
        signals=signals,
        position=position,
        position_fields=["market_value"],
        config=config,
        thresholds_used={"p15_vol_multiplier_min": config.p15_vol_multiplier_min},
        computation="vol_adjusted_move >= p15_vol_multiplier_min",
        fire_result={"vol_units": vol_units, "return_1d": return_1d, "fired": True},
        template_variables={**lv, **rv},
        extras=extras,
    )
    return _make_fire(pattern_id="P15", position=position, account_state=account_state,
                       label=label, rationale=rationale, trace=trace)


# ---------------------------------------------------------------------------
# Public registry of detector callables
# ---------------------------------------------------------------------------

PER_POSITION_DETECTORS = (
    ("P1", detect_p1),
    ("P2", detect_p2),
    ("P3", detect_p3),
    ("P4", detect_p4),
    ("P5", detect_p5),
    ("P6", detect_p6),
    ("P7", detect_p7),
    ("P9", detect_p9),
    ("P10", detect_p10),
    ("P13", detect_p13),
    ("P15", detect_p15),
)

ACCOUNT_LEVEL_DETECTORS = (
    ("P8", detect_p8_account),
    ("P11", detect_p11_account),
    ("P12", detect_p12_account),
    ("P14", detect_p14_account),
)
