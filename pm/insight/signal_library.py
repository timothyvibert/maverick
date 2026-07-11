"""V1 signal library — 28 signals across groups A–F.

Each signal returns a ``SignalValue`` with a canonical audit trace. Stale or
missing required inputs propagate explicitly: ``value=None, stale=True``, with
the trace identifying which input was missing.

Public entry points:
- ``compute_signals_for_underlying`` — A, B, C, D, F groups (per-underlying)
- ``compute_position_signals``        — E group (per-position)

Group F (composite_score) wraps ``compute_composite_score`` from
``pm.core.composite_score`` and decomposes its 5 components into the trace so
the math is verifiable from the display.

V1 limitations:
- D2 (street consensus) requires BBG fields not in ``UNDERLYING_FIELDS``
  (``BEST_ANALYST_REC``, ``BEST_TARGET_PRICE``); it always returns stale
  in V1 with a trace note.
- D3 (``analyst_note_recent``) is live: it reads a pre-fetched
  ``INTERVAL_END_VALUE_DATE`` (BE998=UBS, PX395=Best Analyst Rating override
  pair) passed in as ``analyst_note_date``. Stale only when the override returns
  no date (offline or no analyst coverage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from pm.core.composite_score import compute_composite_score
from pm.core.vol_metrics import iv_percentile
from pm.ingest.position_builder import Position


# ---------------------------------------------------------------------------
# Output object + type alias
# ---------------------------------------------------------------------------

@dataclass
class SignalValue:
    """One signal's output. Trace is the load-bearing field."""
    signal_id: str
    value: Any
    display: str
    interpretation: Optional[str]
    trace: dict[str, Any]
    stale: bool


SignalDict = dict[str, SignalValue]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

EM_DASH = "—"


def _is_nan_or_none(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _coerce_float(v: Any) -> Optional[float]:
    if _is_nan_or_none(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stale(
    signal_id: str,
    missing_input: str,
    reason: str = "input NaN or missing",
    inputs: Optional[dict] = None,
) -> SignalValue:
    """Construct a stale SignalValue with a clean trace."""
    trace_inputs = dict(inputs or {})
    if missing_input not in trace_inputs:
        trace_inputs[missing_input] = {
            "value": None,
            "source": "—",
            "as_of": None,
            "stale": True,
        }
    return SignalValue(
        signal_id=signal_id,
        value=None,
        display=EM_DASH,
        interpretation=None,
        trace={
            "inputs": trace_inputs,
            "computation": f"skipped ({reason}: {missing_input})",
            "thresholds": {},
            "result": None,
        },
        stale=True,
    )


def _bbg_input(value: Any, field_name: str, as_of: Any = None) -> dict:
    """Helper to build a single trace.inputs entry from a BBG field."""
    return {
        "value": value,
        "source": f"BBG:{field_name}",
        "as_of": as_of,
        "stale": _is_nan_or_none(value),
    }


def _extract_input(value: Any, column_name: str, as_of: Any = None) -> dict:
    return {
        "value": value,
        "source": f"EXTRACT:{column_name}",
        "as_of": as_of,
        "stale": _is_nan_or_none(value),
    }


def _computed_input(value: Any, description: str) -> dict:
    return {
        "value": value,
        "source": f"computed:{description}",
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "stale": False,
    }


def _fmt_pct(v: Optional[float], signed: bool = True) -> str:
    if v is None:
        return EM_DASH
    spec = "{:+.1%}" if signed else "{:.1%}"
    return spec.format(v)


def _fmt_num(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return EM_DASH
    return f"{v:.{decimals}f}"


def _safe_business_days_until(target: Optional[date]) -> Optional[int]:
    if target is None:
        return None
    if isinstance(target, pd.Timestamp):
        target = target.date()
    if isinstance(target, datetime):
        target = target.date()
    if not isinstance(target, date):
        return None
    today = date.today()
    if target < today:
        return -int(len(pd.bdate_range(target, today)) - 1)
    return int(len(pd.bdate_range(today, target)) - 1)


def _today() -> date:
    """Indirection point for 'today' so tests can pin a deterministic
    reference date (D3 analyst_note_recent business-day math)."""
    return date.today()


def _business_days_since(note_date: date, today: date) -> int:
    """Business days from ``note_date`` to ``today`` (0 = today, 1 = the
    previous business day). Mirrors the ``pd.bdate_range`` idiom used by
    ``_safe_business_days_until`` for codebase consistency."""
    if note_date > today:
        return -int(len(pd.bdate_range(today, note_date)) - 1)
    return int(len(pd.bdate_range(note_date, today)) - 1)


# ===========================================================================
# Group A — Trend & Momentum (8 signals)
# ===========================================================================

def _compute_spot_vs_50d_ma(snap: dict) -> SignalValue:
    spot = _coerce_float(snap.get("PX_LAST"))
    ma = _coerce_float(snap.get("MOV_AVG_50D"))
    inputs = {
        "PX_LAST": _bbg_input(spot, "PX_LAST"),
        "MOV_AVG_50D": _bbg_input(ma, "MOV_AVG_50D"),
    }
    if spot is None or ma is None or ma == 0:
        return _stale("spot_vs_50d_ma", "PX_LAST" if spot is None else "MOV_AVG_50D", inputs=inputs)
    value = (spot / ma) - 1
    return SignalValue(
        signal_id="spot_vs_50d_ma",
        value=value,
        display=_fmt_pct(value),
        interpretation=("Above 50d MA — short-term momentum positive."
                        if value > 0 else "Below 50d MA — short-term weakness."),
        trace={
            "inputs": inputs,
            "computation": "(PX_LAST / MOV_AVG_50D) - 1",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


def _compute_spot_vs_200d_ma(snap: dict) -> SignalValue:
    spot = _coerce_float(snap.get("PX_LAST"))
    ma = _coerce_float(snap.get("MOV_AVG_200D"))
    inputs = {
        "PX_LAST": _bbg_input(spot, "PX_LAST"),
        "MOV_AVG_200D": _bbg_input(ma, "MOV_AVG_200D"),
    }
    if spot is None or ma is None or ma == 0:
        return _stale("spot_vs_200d_ma", "PX_LAST" if spot is None else "MOV_AVG_200D", inputs=inputs)
    value = (spot / ma) - 1
    return SignalValue(
        signal_id="spot_vs_200d_ma",
        value=value,
        display=_fmt_pct(value),
        interpretation=("Above 200d MA — long-term uptrend intact."
                        if value > 0 else "Below 200d MA — long-term downtrend."),
        trace={
            "inputs": inputs,
            "computation": "(PX_LAST / MOV_AVG_200D) - 1",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


def _compute_ma_stack_regime(snap: dict) -> SignalValue:
    spot = _coerce_float(snap.get("PX_LAST"))
    ma50 = _coerce_float(snap.get("MOV_AVG_50D"))
    ma200 = _coerce_float(snap.get("MOV_AVG_200D"))
    inputs = {
        "PX_LAST": _bbg_input(spot, "PX_LAST"),
        "MOV_AVG_50D": _bbg_input(ma50, "MOV_AVG_50D"),
        "MOV_AVG_200D": _bbg_input(ma200, "MOV_AVG_200D"),
    }
    if spot is None or ma50 is None or ma200 is None:
        return _stale("ma_stack_regime", "PX_LAST / MOV_AVG_50D / MOV_AVG_200D", inputs=inputs)

    if spot > ma50 > ma200:
        regime = "bullish_aligned"
        interp = "Bullish stack: spot above 50d above 200d."
    elif spot > ma200 and ma50 > ma200 and spot < ma50:
        regime = "bullish_curling"
        interp = "Bullish curling: spot above 200d, 50d above 200d, spot below 50d."
    elif spot < ma50 < ma200:
        regime = "bearish_aligned"
        interp = "Bearish stack: spot below 50d below 200d."
    elif spot < ma200 and ma50 < ma200 and spot > ma50:
        regime = "bearish_curling"
        interp = "Bearish curling: spot below 200d, 50d below 200d, spot above 50d."
    else:
        regime = "mixed"
        interp = "Mixed trend regime."

    return SignalValue(
        signal_id="ma_stack_regime",
        value=regime,
        display=regime,
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": "categorical (spot vs MOV_AVG_50D vs MOV_AVG_200D)",
            "thresholds": {},
            "result": regime,
        },
        stale=False,
    )


def _compute_return_horizons(snap: dict) -> SignalValue:
    field_map = {
        "return_1d": "CHG_PCT_1D",
        "return_5d": "CHG_PCT_5D",
        "return_3m": "CHG_PCT_3M",
        "return_ytd": "CHG_PCT_YTD",
        "return_1y": "CHG_PCT_1YR",
    }
    inputs: dict[str, Any] = {}
    value: dict[str, Optional[float]] = {}
    for key, field_name in field_map.items():
        raw = snap.get(field_name)
        v = _coerce_float(raw)
        # BBG returns percent values as integers (1.5 = 1.5%, not 0.015).
        # Divide by 100 to bring into decimal scale for downstream signal
        # consumers (e.g. return_5d < -0.03 in P3).
        if v is not None:
            v = v / 100.0
        value[key] = v
        inputs[field_name] = _bbg_input(raw, field_name)

    populated = [k for k, v in value.items() if v is not None]
    if not populated:
        return _stale("return_horizons", "CHG_PCT_*", inputs=inputs)

    display_lines = [
        f"1D {_fmt_pct(value['return_1d'])}",
        f"5D {_fmt_pct(value['return_5d'])}",
        f"3M {_fmt_pct(value['return_3m'])}",
        f"YTD {_fmt_pct(value['return_ytd'])}",
        f"1Y {_fmt_pct(value['return_1y'])}",
    ]
    return SignalValue(
        signal_id="return_horizons",
        value=value,
        display=" · ".join(display_lines),
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "direct read of CHG_PCT_{1D,5D,3M,YTD,1YR}; /100 to decimal",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


def _compute_rsi_14d_regime(snap: dict) -> SignalValue:
    rsi = _coerce_float(snap.get("RSI_14D"))
    inputs = {"RSI_14D": _bbg_input(rsi, "RSI_14D")}
    if rsi is None:
        return _stale("rsi_14d_regime", "RSI_14D", inputs=inputs)

    if rsi < 30:
        regime, interp = "oversold", f"Oversold (RSI {rsi:.0f})."
    elif rsi < 45:
        regime, interp = "weak", f"Weak (RSI {rsi:.0f})."
    elif rsi < 55:
        regime, interp = "neutral", f"Neutral (RSI {rsi:.0f})."
    elif rsi < 70:
        regime, interp = "strong", f"Strong (RSI {rsi:.0f})."
    else:
        regime, interp = "overbought", f"Overbought (RSI {rsi:.0f})."

    value = {"rsi": rsi, "regime": regime}
    return SignalValue(
        signal_id="rsi_14d_regime",
        value=value,
        display=f"{rsi:.0f} ({regime})",
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": "RSI_14D → bands (<30 / 30-45 / 45-55 / 55-70 / >70)",
            "thresholds": {"oversold_max": 30, "neutral_band": [45, 55], "overbought_min": 70},
            "result": value,
        },
        stale=False,
    )


def _compute_distance_from_52w_high(snap: dict) -> SignalValue:
    spot = _coerce_float(snap.get("PX_LAST"))
    hi = _coerce_float(snap.get("HIGH_52WEEK"))
    inputs = {
        "PX_LAST": _bbg_input(spot, "PX_LAST"),
        "HIGH_52WEEK": _bbg_input(hi, "HIGH_52WEEK"),
    }
    if spot is None or hi is None or hi == 0:
        return _stale("distance_from_52w_high", "PX_LAST" if spot is None else "HIGH_52WEEK", inputs=inputs)
    value = (spot / hi) - 1
    return SignalValue(
        signal_id="distance_from_52w_high",
        value=value,
        display=_fmt_pct(value),
        interpretation=("At/near 52w high." if value >= -0.02
                        else f"{_fmt_pct(value)} below 52w high."),
        trace={
            "inputs": inputs,
            "computation": "(PX_LAST / HIGH_52WEEK) - 1",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


def _compute_distance_from_52w_low(snap: dict) -> SignalValue:
    spot = _coerce_float(snap.get("PX_LAST"))
    lo = _coerce_float(snap.get("LOW_52WEEK"))
    inputs = {
        "PX_LAST": _bbg_input(spot, "PX_LAST"),
        "LOW_52WEEK": _bbg_input(lo, "LOW_52WEEK"),
    }
    if spot is None or lo is None or lo == 0:
        return _stale("distance_from_52w_low", "PX_LAST" if spot is None else "LOW_52WEEK", inputs=inputs)
    value = (spot / lo) - 1
    return SignalValue(
        signal_id="distance_from_52w_low",
        value=value,
        display=_fmt_pct(value),
        interpretation=("At/near 52w low — capitulation candidate." if value <= 0.05
                        else f"{_fmt_pct(value)} above 52w low."),
        trace={
            "inputs": inputs,
            "computation": "(PX_LAST / LOW_52WEEK) - 1",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


# A8 sanity floor on annualized RV (percent) before deriving sigma_daily.
# A dead-vol name (e.g. VOLATILITY_30D ~ 0.5%) would otherwise turn an
# ordinary 1% move into a >30σ "extreme" reading and spuriously trip P15.
# Floor: 5% annualized.
_A8_RV_FLOOR_ANN = 5.0


def _compute_vol_adjusted_move(snap: dict) -> SignalValue:
    """A8: today's return measured in multiples of daily realized vol.

    value = abs(log(1 + r_today)) / sigma_daily, where
    r_today = CHG_PCT_1D / 100 and
    sigma_daily = (max(VOLATILITY_30D, floor) / 100) / sqrt(252).
    The scalar is unsigned; ``r_today`` is recorded in the trace so
    consumers can recover the direction of the move.
    """
    chg_1d = _coerce_float(snap.get("CHG_PCT_1D"))
    vol_30d = _coerce_float(snap.get("VOLATILITY_30D"))
    inputs: dict[str, Any] = {
        "CHG_PCT_1D": _bbg_input(snap.get("CHG_PCT_1D"), "CHG_PCT_1D"),
        "VOLATILITY_30D": _bbg_input(snap.get("VOLATILITY_30D"), "VOLATILITY_30D"),
    }
    if chg_1d is None:
        return _stale("vol_adjusted_move", "CHG_PCT_1D", inputs=inputs)
    if vol_30d is None:
        return _stale("vol_adjusted_move", "VOLATILITY_30D", inputs=inputs)

    r_today = chg_1d / 100.0
    # Defensive: a <= -100% daily print is non-physical for equity, but guard
    # so log() never blows up.
    if 1.0 + r_today <= 0:
        return _stale("vol_adjusted_move", "CHG_PCT_1D",
                      reason="1 + r_today <= 0 (non-physical return)", inputs=inputs)

    vol_used = max(vol_30d, _A8_RV_FLOOR_ANN)
    sigma_daily = (vol_used / 100.0) / math.sqrt(252)
    if sigma_daily <= 0:
        return _stale("vol_adjusted_move", "VOLATILITY_30D",
                      reason="sigma_daily <= 0", inputs=inputs)

    value = abs(math.log(1.0 + r_today)) / sigma_daily

    inputs["r_today"] = _computed_input(r_today, "CHG_PCT_1D / 100")
    inputs["sigma_daily"] = _computed_input(
        sigma_daily,
        f"(max(VOLATILITY_30D, {_A8_RV_FLOOR_ANN}) / 100) / sqrt(252)",
    )

    if value < 1.0:
        interp = "Normal range — move within typical daily vol."
    elif value < 1.5:
        interp = "Above-average move."
    elif value < 2.5:
        interp = "Notable move."
    else:
        interp = "Extreme move."

    return SignalValue(
        signal_id="vol_adjusted_move",
        value=value,
        display=f"{value:.2f}σ",
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": ("abs(log(1 + r_today)) / sigma_daily where sigma_daily "
                            "= (VOLATILITY_30D / 100) / sqrt(252)"),
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


# ===========================================================================
# Group B — Volatility (7 signals)
# ===========================================================================

def _compute_rv_30d(snap: dict) -> SignalValue:
    rv = _coerce_float(snap.get("VOLATILITY_30D"))
    inputs = {"VOLATILITY_30D": _bbg_input(rv, "VOLATILITY_30D")}
    if rv is None:
        return _stale("rv_30d", "VOLATILITY_30D", inputs=inputs)
    return SignalValue(
        signal_id="rv_30d",
        value=rv,
        display=f"{rv:.1f}%",
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "direct read of VOLATILITY_30D (annualized %)",
            "thresholds": {},
            "result": rv,
        },
        stale=False,
    )


def _compute_iv_1m_atm(snap: dict) -> SignalValue:
    iv = _coerce_float(snap.get("CALL_IMP_VOL_30D"))
    inputs = {"CALL_IMP_VOL_30D": _bbg_input(iv, "CALL_IMP_VOL_30D")}
    if iv is None:
        return _stale("iv_1m_atm", "CALL_IMP_VOL_30D", inputs=inputs)
    return SignalValue(
        signal_id="iv_1m_atm",
        value=iv,
        display=f"{iv:.1f}%",
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "direct read of CALL_IMP_VOL_30D (substitute for 1MTH_IMPVOL)",
            "thresholds": {},
            "result": iv,
        },
        stale=False,
    )


def _compute_iv_3m_atm(snap: dict) -> SignalValue:
    iv = _coerce_float(snap.get("3MTH_IMPVOL_100.0%MNY_DF"))
    inputs = {"3MTH_IMPVOL_100.0%MNY_DF": _bbg_input(iv, "3MTH_IMPVOL_100.0%MNY_DF")}
    if iv is None:
        return _stale("iv_3m_atm", "3MTH_IMPVOL_100.0%MNY_DF", inputs=inputs)
    return SignalValue(
        signal_id="iv_3m_atm",
        value=iv,
        display=f"{iv:.1f}%",
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "direct read of 3MTH_IMPVOL_100.0%MNY_DF",
            "thresholds": {},
            "result": iv,
        },
        stale=False,
    )


def _compute_iv_6m_atm(snap: dict) -> SignalValue:
    iv = _coerce_float(snap.get("6MTH_IMPVOL_100.0%MNY_DF"))
    inputs = {"6MTH_IMPVOL_100.0%MNY_DF": _bbg_input(iv, "6MTH_IMPVOL_100.0%MNY_DF")}
    if iv is None:
        return _stale("iv_6m_atm", "6MTH_IMPVOL_100.0%MNY_DF", inputs=inputs)
    return SignalValue(
        signal_id="iv_6m_atm",
        value=iv,
        display=f"{iv:.1f}%",
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "direct read of 6MTH_IMPVOL_100.0%MNY_DF",
            "thresholds": {},
            "result": iv,
        },
        stale=False,
    )


def _compute_iv_3m_percentile_1y(snap: dict, iv_history: Optional[pd.Series]) -> SignalValue:
    current_iv = _coerce_float(snap.get("3MTH_IMPVOL_100.0%MNY_DF"))
    inputs = {
        "3MTH_IMPVOL_100.0%MNY_DF": _bbg_input(current_iv, "3MTH_IMPVOL_100.0%MNY_DF"),
        "iv_history": {
            "value": None if iv_history is None else f"<{len(iv_history)} obs>",
            "source": "BBG:BDH(3MTH_IMPVOL_100.0%MNY_DF, lookback=365)",
            "as_of": None,
            "stale": iv_history is None or len(iv_history.dropna()) < 30,
        },
    }
    if current_iv is None:
        return _stale("iv_3m_percentile_1y", "3MTH_IMPVOL_100.0%MNY_DF", inputs=inputs)
    if iv_history is None or len(iv_history.dropna()) < 30:
        return _stale("iv_3m_percentile_1y", "iv_history",
                      reason="history empty or <30 obs", inputs=inputs)

    pctl = iv_percentile(current_iv, iv_history)
    if pctl is None:
        return _stale("iv_3m_percentile_1y", "iv_history",
                      reason="percentile compute returned None", inputs=inputs)
    pctl_100 = float(pctl) * 100  # vol_metrics returns 0-1; spec uses 0-100
    interp = (
        f"IV in {pctl_100:.0f}th pctl of 1Y range — "
        + ("elevated, premium-selling regime." if pctl_100 >= 75
           else "depressed, premium-buying regime." if pctl_100 <= 25
           else "mid-range.")
    )
    return SignalValue(
        signal_id="iv_3m_percentile_1y",
        value=pctl_100,
        display=f"{pctl_100:.0f}th pctl",
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": "count(history ≤ current_iv) / count(history) × 100",
            "thresholds": {"elevated_min": 75, "depressed_max": 25},
            "result": pctl_100,
        },
        stale=False,
    )


def _compute_iv_term_structure(in_progress: SignalDict) -> SignalValue:
    iv3 = in_progress.get("iv_3m_atm")
    iv6 = in_progress.get("iv_6m_atm")
    inputs = {
        "iv_3m_atm": _computed_input(
            iv3.value if iv3 else None, "from signal iv_3m_atm"
        ),
        "iv_6m_atm": _computed_input(
            iv6.value if iv6 else None, "from signal iv_6m_atm"
        ),
    }
    if iv3 is None or iv6 is None or iv3.stale or iv6.stale:
        return _stale("iv_term_structure", "iv_3m_atm OR iv_6m_atm",
                      reason="prior signal stale", inputs=inputs)
    value = iv3.value - iv6.value
    interp = ("Backwardation (3M > 6M) — short-dated rich, often event-driven."
              if value > 0 else "Contango (3M < 6M) — normal term structure.")
    return SignalValue(
        signal_id="iv_term_structure",
        value=value,
        display=f"{value:+.1f}",
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": "iv_3m_atm - iv_6m_atm",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


def _compute_vrp_30d(in_progress: SignalDict) -> SignalValue:
    iv1 = in_progress.get("iv_1m_atm")
    rv = in_progress.get("rv_30d")
    inputs = {
        "iv_1m_atm": _computed_input(iv1.value if iv1 else None, "from signal iv_1m_atm"),
        "rv_30d": _computed_input(rv.value if rv else None, "from signal rv_30d"),
    }
    if iv1 is None or rv is None or iv1.stale or rv.stale:
        return _stale("vrp_30d", "iv_1m_atm OR rv_30d", reason="prior signal stale", inputs=inputs)
    value = iv1.value - rv.value
    interp = ("Options rich vs realized — premium-selling environment."
              if value > 5 else "Options cheap vs realized — premium-buying setup."
              if value < 0 else "VRP near zero — neutral.")
    return SignalValue(
        signal_id="vrp_30d",
        value=value,
        display=f"{value:+.1f}",
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": "iv_1m_atm - rv_30d (vol points)",
            "thresholds": {"rich_min": 5, "cheap_max": 0},
            "result": value,
        },
        stale=False,
    )


# ===========================================================================
# Group C — Catalysts (4 signals)
# ===========================================================================

def _compute_days_to_earnings(snap: dict) -> SignalValue:
    raw = snap.get("EXPECTED_REPORT_DT")
    inputs = {"EXPECTED_REPORT_DT": _bbg_input(raw, "EXPECTED_REPORT_DT")}
    if _is_nan_or_none(raw):
        return _stale("days_to_earnings", "EXPECTED_REPORT_DT", inputs=inputs)
    try:
        earn_date = pd.to_datetime(raw).date()
    except Exception:
        return _stale("days_to_earnings", "EXPECTED_REPORT_DT",
                      reason="unparseable date", inputs=inputs)
    bdays = _safe_business_days_until(earn_date)
    if bdays is None:
        return _stale("days_to_earnings", "EXPECTED_REPORT_DT",
                      reason="business-day calc failed", inputs=inputs)
    interp = ("Imminent — within 7 BD." if bdays <= 7
              else "Approaching — within 30 BD." if bdays <= 30
              else "Far.")
    return SignalValue(
        signal_id="days_to_earnings",
        value=bdays,
        display=f"{bdays} BD",
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": "business days from today to EXPECTED_REPORT_DT (pd.bdate_range)",
            "thresholds": {"imminent_max": 7, "approaching_max": 30},
            "result": bdays,
        },
        stale=False,
    )


def _compute_earnings_implied_move(snap: dict) -> SignalValue:
    raw = _coerce_float(snap.get("EARNINGS_RELATED_IMPLIED_MOVE"))
    inputs = {"EARNINGS_RELATED_IMPLIED_MOVE": _bbg_input(raw, "EARNINGS_RELATED_IMPLIED_MOVE")}
    if raw is None:
        return _stale("earnings_implied_move", "EARNINGS_RELATED_IMPLIED_MOVE", inputs=inputs)
    # BBG ships as percent (e.g., 4.5 means 4.5%). Convert to decimal.
    value = raw / 100.0
    return SignalValue(
        signal_id="earnings_implied_move",
        value=value,
        display=_fmt_pct(value, signed=False),
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "EARNINGS_RELATED_IMPLIED_MOVE / 100",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


_DVD_EX_FIELDS = ("DVD_EX_DT", "EQY_DVD_EX_DT", "DVD_EX_DATE", "EQY_DVD_EX_DATE")


def _compute_days_to_ex_div(snap: dict, projected_dividend: Optional[dict] = None) -> SignalValue:
    chosen_field = None
    chosen_date = None
    inputs: dict[str, Any] = {}
    for field_name in _DVD_EX_FIELDS:
        raw = snap.get(field_name)
        inputs[field_name] = _bbg_input(raw, field_name)
        if _is_nan_or_none(raw):
            continue
        try:
            parsed = pd.to_datetime(raw).date()
        except Exception:
            continue
        if chosen_date is None:
            chosen_date = parsed
            chosen_field = field_name

    # DVD_YLD rides along in the trace so the ex-div fire can fall back to the
    # yield heuristic (annual yield × spot / 4) when no forward forecast exists.
    inputs["DVD_YLD"] = _bbg_input(snap.get("DVD_YLD"), "DVD_YLD")

    # Forward dividend forecast: the precise per-share amount + ex-date the
    # ex-div fire prefers over the yield heuristic. A projected ex-date is the
    # forward event, so it governs the days-to-ex-div value when present.
    next_div = (projected_dividend or {}).get("next") or {}
    if next_div.get("dps") is not None:
        inputs["projected_dividend"] = {
            "value": next_div.get("dps"),
            "source": "BBG:BDVD_ALL_PROJECTIONS",
            "as_of": None,
            "stale": False,
        }
    proj_ex = next_div.get("ex_date")
    if proj_ex is not None:
        try:
            chosen_date = pd.to_datetime(proj_ex).date()
            chosen_field = "BDVD_ALL_PROJECTIONS"
        except Exception:
            pass

    if chosen_date is None:
        return _stale("days_to_ex_div", "DVD_EX_DT family", inputs=inputs)
    inputs["ex_div_date"] = _computed_input(chosen_date.isoformat(),
                                             f"from {chosen_field}")
    bdays = _safe_business_days_until(chosen_date)
    if bdays is None:
        return _stale("days_to_ex_div", "DVD_EX_DT family",
                      reason="business-day calc failed", inputs=inputs)
    interp = ("Imminent — ex-div within 7 BD." if 0 <= bdays <= 7
              else "Past." if bdays < 0
              else f"{bdays} BD away.")
    return SignalValue(
        signal_id="days_to_ex_div",
        value=bdays,
        display=f"{bdays} BD",
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": ("projected ex-date if a forward forecast exists, else "
                            f"first populated of {_DVD_EX_FIELDS}; business days "
                            "from today via pd.bdate_range"),
            "thresholds": {"imminent_max": 7},
            "result": bdays,
        },
        stale=False,
    )


def _compute_dte_nearest_expiry(positions: list[Position], underlying: str) -> SignalValue:
    """C4: minimum DTE across all option positions on this underlying in the account."""
    today = date.today()
    candidates: list[int] = []
    inputs: dict[str, Any] = {
        "positions_scanned": {
            "value": sum(1 for p in positions
                          if p.asset_class == "option" and p.underlying_symbol == underlying),
            "source": "EXTRACT:positions",
            "as_of": today.isoformat(),
            "stale": False,
        },
    }
    for p in positions:
        if p.asset_class != "option" or p.underlying_symbol != underlying:
            continue
        if p.expiry is None:
            continue
        dte = (p.expiry - today).days
        candidates.append(dte)
    if not candidates:
        return _stale("dte_nearest_expiry_in_account", "positions",
                      reason="no option positions on this underlying", inputs=inputs)
    value = min(candidates)
    return SignalValue(
        signal_id="dte_nearest_expiry_in_account",
        value=value,
        display=f"{value}d",
        interpretation=("Nearest option expiry imminent (<14d)." if value < 14
                        else f"Nearest option expiry in {value} calendar days."),
        trace={
            "inputs": inputs,
            "computation": "min((expiry - today).days) over option positions on this underlying",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


# ===========================================================================
# Group D — Sentiment & Ratings (3 signals)
# ===========================================================================

def _compute_analyst_rating_and_target(
    snap: dict, analyst_data: Optional[dict]
) -> SignalValue:
    inputs: dict[str, Any] = {
        "analyst_rating": {
            "value": (analyst_data or {}).get("analyst_rating"),
            "source": "BBG:BEST_ANALYST_REC[BE998=UBS]",
            "as_of": None,
            "stale": (analyst_data or {}).get("analyst_rating") is None,
        },
        "target": {
            "value": (analyst_data or {}).get("analyst_target"),
            "source": "BBG:BEST_TARGET_PRICE[BE998=UBS]",
            "as_of": None,
            "stale": (analyst_data or {}).get("analyst_target") is None,
        },
        "PX_LAST": _bbg_input(snap.get("PX_LAST"), "PX_LAST"),
    }
    rating = (analyst_data or {}).get("analyst_rating")
    target = (analyst_data or {}).get("analyst_target")
    spot = _coerce_float(snap.get("PX_LAST"))
    if rating is None and target is None:
        return _stale("analyst_rating_and_target", "analyst_data",
                      reason="analyst override returned no data", inputs=inputs)
    upside_pct = None
    if target is not None and spot is not None and spot > 0:
        upside_pct = (target / spot) - 1
    value = {"rating": rating, "target": target, "upside_pct": upside_pct}
    display = (f"{rating or EM_DASH} · target ${target:.2f} · "
               f"upside {_fmt_pct(upside_pct)}") if target is not None else (rating or EM_DASH)
    return SignalValue(
        signal_id="analyst_rating_and_target",
        value=value,
        display=display,
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "fetch_analyst_data(BE998=UBS); upside = (target/PX_LAST) - 1",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


def _compute_street_consensus(snap: dict) -> SignalValue:
    """V1: returns stale always (BEST_TARGET_PRICE / BEST_ANALYST_REC not in
    UNDERLYING_FIELDS). Plumbing deferred to V2."""
    inputs = {
        "BEST_ANALYST_REC": {
            "value": None, "source": "BBG:BEST_ANALYST_REC",
            "as_of": None, "stale": True,
        },
        "BEST_TARGET_PRICE": {
            "value": None, "source": "BBG:BEST_TARGET_PRICE",
            "as_of": None, "stale": True,
        },
    }
    return SignalValue(
        signal_id="street_consensus_rating_and_target",
        value=None,
        display=EM_DASH,
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": ("V1 limitation: street consensus requires non-overridden BDP "
                            "fetch; BEST_TARGET_PRICE / BEST_ANALYST_REC not currently in "
                            "UNDERLYING_FIELDS. Plumbing deferred to V2."),
            "thresholds": {},
            "result": None,
        },
        stale=True,
    )


def _compute_analyst_note_recent(analyst_note_date: Optional["pd.Timestamp"]) -> SignalValue:
    """D3: date of the most recent provider-sourced analyst
    note for this underlying, and whether it lands today or the previous
    business day.

    ``analyst_note_date`` is pre-fetched in ``portfolio_state`` via
    ``fetch_analyst_note_dates`` (INTERVAL_END_VALUE_DATE under the
    BE998=UBS + PX395=Best Analyst Rating override pair). None / NaT →
    stale (offline, or no analyst coverage on the name).
    """
    note_iso = None
    if isinstance(analyst_note_date, (pd.Timestamp, datetime, date)):
        note_iso = analyst_note_date.isoformat()
    inputs: dict[str, Any] = {
        "INTERVAL_END_VALUE_DATE": {
            "value": note_iso,
            "source": "BBG:INTERVAL_END_VALUE_DATE[BE998=UBS, PX395=Best Analyst Rating]",
            "as_of": None,
            "stale": _is_nan_or_none(analyst_note_date),
        },
    }
    if _is_nan_or_none(analyst_note_date):
        return _stale("analyst_note_recent", "INTERVAL_END_VALUE_DATE",
                      reason="analyst-note override returned no date", inputs=inputs)

    if isinstance(analyst_note_date, pd.Timestamp):
        note_d = analyst_note_date.date()
    elif isinstance(analyst_note_date, datetime):
        note_d = analyst_note_date.date()
    elif isinstance(analyst_note_date, date):
        note_d = analyst_note_date
    else:
        try:
            note_d = pd.to_datetime(analyst_note_date).date()
        except Exception:
            return _stale("analyst_note_recent", "INTERVAL_END_VALUE_DATE",
                          reason="unparseable note date", inputs=inputs)

    today = _today()
    days_since = _business_days_since(note_d, today)
    is_recent = days_since in (0, 1)
    value = {
        "note_date": note_d,
        "is_recent": is_recent,
        "days_since": days_since,
    }
    if is_recent:
        display = f"Note {note_d.isoformat()} ({days_since} BD)"
        interp = "A new analyst note was published recently — sales catalyst."
    else:
        display = f"Last note {note_d.isoformat()}"
        interp = "No recent analyst note."
    return SignalValue(
        signal_id="analyst_note_recent",
        value=value,
        display=display,
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": ("INTERVAL_END_VALUE_DATE via BDP[BE998=UBS, "
                            "PX395=Best Analyst Rating]; is_recent = note_date in "
                            "{today, prev_business_day}; days_since = business days "
                            "from note_date to today (pd.bdate_range)"),
            "thresholds": {"recent_max_bd": 1},
            "result": value,
        },
        stale=False,
    )


# ===========================================================================
# Group E — Position-specific (5 signals)
# ===========================================================================

def _compute_position_size_pct_of_nav(position: Position, account_nav: float) -> SignalValue:
    mv = _coerce_float(position.market_value)
    inputs = {
        "market_value": _extract_input(mv, "Market Value"),
        "account_nav": _computed_input(account_nav, "sum of signed market_value per account"),
    }
    if mv is None:
        return _stale("position_size_pct_of_nav", "market_value", inputs=inputs)
    if not account_nav:
        return _stale("position_size_pct_of_nav", "account_nav",
                      reason="account NAV zero", inputs=inputs)
    value = abs(mv) / abs(account_nav)
    return SignalValue(
        signal_id="position_size_pct_of_nav",
        value=value,
        display=_fmt_pct(value, signed=False),
        interpretation=(f"{value:.0%} of NAV — concentrated."
                        if value >= 0.25 else None),
        trace={
            "inputs": inputs,
            "computation": "abs(market_value) / abs(account_nav)",
            "thresholds": {"concentrated_min": 0.25},
            "result": value,
        },
        stale=False,
    )


def _compute_position_unrealized_pnl_pct(position: Position) -> SignalValue:
    pct = _coerce_float(position.unrealized_pnl_pct)
    inputs = {"unrealized_pnl_pct": _extract_input(pct, "Unrealized P&L %")}
    if pct is None:
        return _stale("position_unrealized_pnl_pct", "unrealized_pnl_pct", inputs=inputs)
    return SignalValue(
        signal_id="position_unrealized_pnl_pct",
        value=pct,
        display=_fmt_pct(pct),
        interpretation=None,
        trace={
            "inputs": inputs,
            "computation": "direct read of the extract's Unrealized P&L % (decimal scale)",
            "thresholds": {},
            "result": pct,
        },
        stale=False,
    )


def _compute_option_captured_pct(position: Position) -> SignalValue:
    """E3: for short options only; long options return value=None intentionally
    (not stale)."""
    inputs: dict[str, Any] = {
        "quantity": _extract_input(position.quantity, "Quantity"),
        "cost_basis": _extract_input(position.cost_basis, "Cost Basis"),
        "market_value": _extract_input(position.market_value, "Market Value"),
        "asset_class": _extract_input(position.asset_class, "Asset Class"),
    }
    if position.asset_class != "option":
        return SignalValue(
            signal_id="option_captured_pct",
            value=None,
            display=EM_DASH,
            interpretation=None,
            trace={
                "inputs": inputs,
                "computation": "N/A — not an option position",
                "thresholds": {},
                "result": None,
            },
            stale=False,
        )
    qty = _coerce_float(position.quantity)
    cb = _coerce_float(position.cost_basis)
    mv = _coerce_float(position.market_value)
    if qty is None or qty >= 0:
        return SignalValue(
            signal_id="option_captured_pct",
            value=None,
            display=EM_DASH,
            interpretation=None,
            trace={
                "inputs": inputs,
                "computation": "N/A — long option (captured_pct only defined for shorts)",
                "thresholds": {},
                "result": None,
            },
            stale=False,
        )
    if cb is None or cb == 0:
        return _stale("option_captured_pct", "cost_basis", inputs=inputs)
    if mv is None:
        return _stale("option_captured_pct", "market_value", inputs=inputs)
    captured = (abs(cb) - abs(mv)) / abs(cb)
    interp = ("Nearly fully captured (≥75%)." if captured >= 0.75
              else "Mostly captured (≥60%)." if captured >= 0.60
              else "Some captured." if captured > 0
              else "Position has moved against the trade.")
    return SignalValue(
        signal_id="option_captured_pct",
        value=captured,
        display=_fmt_pct(captured, signed=False if captured >= 0 else True),
        interpretation=interp,
        trace={
            "inputs": inputs,
            "computation": "(|cost_basis| - |market_value|) / |cost_basis|  (short options only)",
            "thresholds": {"nearly_full_min": 0.75, "mostly_min": 0.60},
            "result": captured,
        },
        stale=False,
    )


def _compute_option_dte(position: Position) -> SignalValue:
    inputs: dict[str, Any] = {
        "asset_class": _extract_input(position.asset_class, "Asset Class"),
        "expiry": _extract_input(position.expiry, "Option Expiration"),
        "today": _computed_input(date.today().isoformat(), "date.today()"),
    }
    if position.asset_class != "option" or position.expiry is None:
        return _stale("option_dte", "expiry",
                      reason="not an option / no expiry", inputs=inputs)
    dte = (position.expiry - date.today()).days
    return SignalValue(
        signal_id="option_dte",
        value=dte,
        display=f"{dte}d",
        interpretation=("Imminent — ≤14d." if dte <= 14
                        else "Front month." if dte <= 45
                        else "Back month."),
        trace={
            "inputs": inputs,
            "computation": "(position.expiry - today).days  (calendar days)",
            "thresholds": {"imminent_max": 14, "front_max": 45},
            "result": dte,
        },
        stale=False,
    )


def _compute_option_moneyness(position: Position, snap: Optional[dict]) -> SignalValue:
    inputs: dict[str, Any] = {
        "asset_class": _extract_input(position.asset_class, "Asset Class"),
        "right": _extract_input(position.right, "Option Type"),
        "strike": _extract_input(position.strike, "Option Strike"),
        "PX_LAST": _bbg_input((snap or {}).get("PX_LAST"), "PX_LAST"),
    }
    if position.asset_class != "option" or position.strike is None or position.right is None:
        return _stale("option_moneyness", "right/strike",
                      reason="not an option or missing right/strike", inputs=inputs)
    spot = _coerce_float((snap or {}).get("PX_LAST"))
    if spot is None:
        return _stale("option_moneyness", "PX_LAST", inputs=inputs)
    strike = float(position.strike)
    if strike == 0:
        return _stale("option_moneyness", "strike", reason="zero strike", inputs=inputs)
    if position.right == "CALL":
        value = (spot - strike) / strike
    else:
        value = (strike - spot) / strike
    return SignalValue(
        signal_id="option_moneyness",
        value=value,
        display=_fmt_pct(value),
        interpretation=("ITM." if value > 0 else "OTM."),
        trace={
            "inputs": inputs,
            "computation": "(PX_LAST - strike)/strike for calls; (strike - PX_LAST)/strike for puts",
            "thresholds": {},
            "result": value,
        },
        stale=False,
    )


# ===========================================================================
# Group F — Composite (1 signal, decomposed)
# ===========================================================================

def _compute_composite_score(legacy_signals: Optional[list]) -> SignalValue:
    """Wraps the existing 5-component composite, decomposing its components
    into the trace so the score is verifiable from the display."""
    inputs = {
        "legacy_signals_count": {
            "value": 0 if legacy_signals is None else len(legacy_signals),
            "source": "computed:from pm.core.portfolio_signals.compute_per_underlying_signals",
            "as_of": None,
            "stale": legacy_signals is None,
        },
    }
    if legacy_signals is None:
        return _stale("composite_score", "legacy_signals",
                      reason="upstream signals not available", inputs=inputs)
    try:
        score = compute_composite_score(legacy_signals)
    except Exception as exc:
        return _stale("composite_score", "compute_composite_score",
                      reason=f"compute raised: {exc}", inputs=inputs)
    value = {
        "total": score.total,
        "label": score.label,
        "components": score.components,
    }
    component_lines = [
        f"{name}: {c['raw']} × weight → {c['weighted']}"
        for name, c in score.components.items()
    ]
    return SignalValue(
        signal_id="composite_score",
        value=value,
        display=f"{score.total:.1f} ({score.label})",
        interpretation=" · ".join(component_lines),
        trace={
            "inputs": inputs,
            "computation": ("compute_composite_score(legacy_signals) — 5-component "
                            "weighted (signal_count 0.30 / vol_regime 0.25 / "
                            "trend_clarity 0.20 / event_pressure 0.15 / "
                            "signal_agreement 0.10)"),
            "thresholds": {
                "strong_min": 70, "moderate_min": 50, "weak_min": 30,
            },
            "result": value,
        },
        stale=False,
    )


# ===========================================================================
# Public entry points
# ===========================================================================

def compute_signals_for_underlying(
    underlying: str,
    snapshot_row: Optional[dict],
    iv_history: Optional[pd.Series],
    positions_in_account: list[Position],
    account_nav: float,
    analyst_data: Optional[dict],
    legacy_signals: Optional[list] = None,
    analyst_note_date: Optional["pd.Timestamp"] = None,
    projected_dividend: Optional[dict] = None,
) -> SignalDict:
    """Compute all underlying-level signals (groups A–D, F) for one
    underlying in one account context.

    Position-level signals (group E) are not computed here — call
    ``compute_position_signals`` per position and merge.
    """
    snap = dict(snapshot_row or {})
    out: SignalDict = {}

    # Group A
    out["spot_vs_50d_ma"] = _wrap(_compute_spot_vs_50d_ma, snap)
    out["spot_vs_200d_ma"] = _wrap(_compute_spot_vs_200d_ma, snap)
    out["ma_stack_regime"] = _wrap(_compute_ma_stack_regime, snap)
    out["return_horizons"] = _wrap(_compute_return_horizons, snap)
    out["rsi_14d_regime"] = _wrap(_compute_rsi_14d_regime, snap)
    out["distance_from_52w_high"] = _wrap(_compute_distance_from_52w_high, snap)
    out["distance_from_52w_low"] = _wrap(_compute_distance_from_52w_low, snap)
    out["vol_adjusted_move"] = _wrap(_compute_vol_adjusted_move, snap)

    # Group B (B6/B7 depend on B1/B2 and B3/B4 — compute in order)
    out["rv_30d"] = _wrap(_compute_rv_30d, snap)
    out["iv_1m_atm"] = _wrap(_compute_iv_1m_atm, snap)
    out["iv_3m_atm"] = _wrap(_compute_iv_3m_atm, snap)
    out["iv_6m_atm"] = _wrap(_compute_iv_6m_atm, snap)
    out["iv_3m_percentile_1y"] = _wrap(_compute_iv_3m_percentile_1y, snap, iv_history)
    out["iv_term_structure"] = _wrap(_compute_iv_term_structure, out)
    out["vrp_30d"] = _wrap(_compute_vrp_30d, out)

    # Group C
    out["days_to_earnings"] = _wrap(_compute_days_to_earnings, snap)
    out["earnings_implied_move"] = _wrap(_compute_earnings_implied_move, snap)
    out["days_to_ex_div"] = _wrap(_compute_days_to_ex_div, snap, projected_dividend)
    out["dte_nearest_expiry_in_account"] = _wrap(
        _compute_dte_nearest_expiry, positions_in_account, underlying,
    )

    # Group D
    out["analyst_rating_and_target"] = _wrap(_compute_analyst_rating_and_target, snap, analyst_data)
    out["street_consensus_rating_and_target"] = _wrap(_compute_street_consensus, snap)
    out["analyst_note_recent"] = _wrap(_compute_analyst_note_recent, analyst_note_date)

    # Group F
    out["composite_score"] = _wrap(_compute_composite_score, legacy_signals)

    return out


def compute_position_signals(
    position: Position,
    snapshot_row: Optional[dict],
    account_nav: float,
) -> SignalDict:
    """Compute E1–E5 for a single position. Caller merges with the
    underlying SignalDict before passing to detectors."""
    snap = dict(snapshot_row or {}) if snapshot_row is not None else None
    out: SignalDict = {
        "position_size_pct_of_nav": _wrap(_compute_position_size_pct_of_nav, position, account_nav),
        "position_unrealized_pnl_pct": _wrap(_compute_position_unrealized_pnl_pct, position),
        "option_captured_pct": _wrap(_compute_option_captured_pct, position),
        "option_dte": _wrap(_compute_option_dte, position),
        "option_moneyness": _wrap(_compute_option_moneyness, position, snap),
    }
    return out


def _wrap(fn, *args, **kwargs) -> SignalValue:
    """Catch unexpected exceptions and degrade to a stale SignalValue."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        signal_id = getattr(fn, "__name__", "unknown").replace("_compute_", "")
        return SignalValue(
            signal_id=signal_id,
            value=None,
            display=EM_DASH,
            interpretation=None,
            trace={
                "inputs": {},
                "computation": f"detector raised: {type(exc).__name__}: {exc}",
                "thresholds": {},
                "result": None,
            },
            stale=True,
        )
