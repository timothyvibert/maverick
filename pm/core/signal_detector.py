"""Signal detector — pure Python signal detection from SecurityProfile metrics.

Replaces 11 BQL filter() expressions with in-memory signal detection
applied to every security's data. No BQL or Dash dependencies.
"""
from __future__ import annotations

from pm.core.scanner_models import Signal, SecurityProfile


# ---------------------------------------------------------------------------
# Default thresholds (user-configurable via the UI)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "pctl_cheap": 0.20,       # iv_pctl below this → cheap_vol
    "pctl_rich": 0.80,        # iv_pctl above this → rich_vol
    "spread_pts": 5.0,        # |iv - rv| above this → iv_gt_rv or iv_lt_rv
    "z_score": 2.0,           # z-score above this → vol_spike
    "skew_pctl_high": 0.80,   # skew_pctl above this → steep_skew
    "skew_pctl_low": 0.20,    # skew_pctl below this → flat_skew
    "rr_pctl_low": 0.10,      # rr_pctl below this → unusual_rr (puts bid)
    "rr_pctl_high": 0.90,     # rr_pctl above this → unusual_rr (calls bid)
    "bf_pctl": 0.80,          # bf_pctl above this → elevated_tail
    "min_signals": 1,         # Minimum signals for a security to appear in results
}


# ---------------------------------------------------------------------------
# Vol signal detection
# ---------------------------------------------------------------------------

def _detect_vol_signals(profile: SecurityProfile, t: dict) -> list[Signal]:
    """Detect volatility-regime signals from a security's vol metrics."""
    signals: list[Signal] = []

    # cheap_vol
    if profile.iv_pctl is not None and profile.iv_pctl < t["pctl_cheap"]:
        val, thresh = profile.iv_pctl, t["pctl_cheap"]
        s = min((thresh - val) / thresh, 1.0)
        signals.append(Signal(
            signal_type="cheap_vol", strength=s,
            detail=f"{val:.0%} percentile", direction="neutral",
            metric_name="IV Percentile", metric_value=round(val * 100, 1),
            threshold_value=round(thresh * 100, 1),
            strength_formula=f"({thresh*100:.0f} \u2212 {val*100:.0f}) / {thresh*100:.0f} = {s:.2f}",
        ))

    # rich_vol
    if profile.iv_pctl is not None and profile.iv_pctl > t["pctl_rich"]:
        val, thresh = profile.iv_pctl, t["pctl_rich"]
        s = min((val - thresh) / (1.0 - thresh), 1.0)
        signals.append(Signal(
            signal_type="rich_vol", strength=s,
            detail=f"{val:.0%} percentile", direction="neutral",
            metric_name="IV Percentile", metric_value=round(val * 100, 1),
            threshold_value=round(thresh * 100, 1),
            strength_formula=f"({val*100:.0f} \u2212 {thresh*100:.0f}) / (100 \u2212 {thresh*100:.0f}) = {s:.2f}",
        ))

    # vol_spike
    if profile.z_score is not None and profile.z_score > t["z_score"]:
        val, thresh = profile.z_score, t["z_score"]
        s = min((val - thresh) / 2.0, 1.0)
        signals.append(Signal(
            signal_type="vol_spike", strength=s,
            detail=f"Z={val:.1f}\u03c3", direction="neutral",
            metric_name="Z-Score", metric_value=round(val, 2),
            threshold_value=round(thresh, 1),
            strength_formula=f"({val:.1f} \u2212 {thresh:.1f}) / 2.0 = {s:.2f}",
        ))

    # iv_gt_rv (IV > RV)
    if profile.spread is not None and profile.spread > t["spread_pts"]:
        val, thresh = profile.spread, t["spread_pts"]
        s = min(val / 20.0, 1.0)
        signals.append(Signal(
            signal_type="iv_gt_rv", strength=s,
            detail=f"+{val:.1f}pts", direction="neutral",
            metric_name="IV-RV Spread", metric_value=round(val, 1),
            threshold_value=round(thresh, 1),
            strength_formula=f"Gate: {val:.1f} > {thresh:.1f} | {val:.1f} / 20 = {s:.2f}",
        ))

    # iv_lt_rv (IV < RV)
    if profile.spread is not None and profile.spread < -t["spread_pts"]:
        val, thresh = profile.spread, t["spread_pts"]
        s = min(abs(val) / 20.0, 1.0)
        signals.append(Signal(
            signal_type="iv_lt_rv", strength=s,
            detail=f"{val:.1f}pts", direction="neutral",
            metric_name="IV-RV Spread", metric_value=round(val, 1),
            threshold_value=round(-thresh, 1),
            strength_formula=f"Gate: {val:.1f} < {-thresh:.1f} | |{val:.1f}| / 20 = {s:.2f}",
        ))

    # steep_skew
    if profile.skew_pctl is not None and profile.skew_pctl > t["skew_pctl_high"]:
        val, thresh = profile.skew_pctl, t["skew_pctl_high"]
        s = min((val - thresh) / (1.0 - thresh), 1.0)
        signals.append(Signal(
            signal_type="steep_skew", strength=s,
            detail=f"Skew {val:.0%} pctl", direction="neutral",
            metric_name="Skew Percentile", metric_value=round(val * 100, 1),
            threshold_value=round(thresh * 100, 1),
            strength_formula=f"({val*100:.0f} \u2212 {thresh*100:.0f}) / (100 \u2212 {thresh*100:.0f}) = {s:.2f}",
        ))

    # flat_skew
    if profile.skew_pctl is not None and profile.skew_pctl < t["skew_pctl_low"]:
        val, thresh = profile.skew_pctl, t["skew_pctl_low"]
        s = min((thresh - val) / thresh, 1.0)
        signals.append(Signal(
            signal_type="flat_skew", strength=s,
            detail=f"Skew {val:.0%} pctl", direction="neutral",
            metric_name="Skew Percentile", metric_value=round(val * 100, 1),
            threshold_value=round(thresh * 100, 1),
            strength_formula=f"({thresh*100:.0f} \u2212 {val*100:.0f}) / {thresh*100:.0f} = {s:.2f}",
        ))

    # inverted_term (1M > 6M)
    if profile.term is not None and profile.term > 0:
        val = profile.term
        s = min(val / 10.0, 1.0)
        signals.append(Signal(
            signal_type="inverted_term", strength=s,
            detail=f"1M-6M = +{val:.1f}pts", direction="neutral",
            metric_name="Term Spread (1M\u22126M)", metric_value=round(val, 1),
            threshold_value=0.0,
            strength_formula=f"{val:.1f} / 10.0 = {s:.2f}",
        ))

    # contango_term (6M > 1M, normal but steep)
    if profile.term is not None and profile.term < -5.0:
        val = profile.term
        s = min(abs(val) / 15.0, 1.0)
        signals.append(Signal(
            signal_type="contango_term", strength=s,
            detail=f"1M-6M = {val:.1f}pts", direction="neutral",
            metric_name="Term Spread (1M\u22126M)", metric_value=round(val, 1),
            threshold_value=-5.0,
            strength_formula=f"|{val:.1f}| / 15.0 = {s:.2f}",
        ))

    # unusual_rr (extreme risk reversal)
    if profile.rr_pctl is not None:
        if profile.rr_pctl < t["rr_pctl_low"]:
            val, thresh = profile.rr_pctl, t["rr_pctl_low"]
            s = min((thresh - val) / thresh, 1.0)
            signals.append(Signal(
                signal_type="unusual_rr", strength=s,
                detail=f"RR {val:.0%} pctl (puts bid)", direction="bearish",
                metric_name="RR Percentile", metric_value=round(val * 100, 1),
                threshold_value=round(thresh * 100, 1),
                strength_formula=f"({thresh*100:.0f} \u2212 {val*100:.0f}) / {thresh*100:.0f} = {s:.2f}",
            ))
        elif profile.rr_pctl > t["rr_pctl_high"]:
            val, thresh = profile.rr_pctl, t["rr_pctl_high"]
            s = min((val - thresh) / (1.0 - thresh), 1.0)
            signals.append(Signal(
                signal_type="unusual_rr", strength=s,
                detail=f"RR {val:.0%} pctl (calls bid)", direction="bullish",
                metric_name="RR Percentile", metric_value=round(val * 100, 1),
                threshold_value=round(thresh * 100, 1),
                strength_formula=f"({val*100:.0f} \u2212 {thresh*100:.0f}) / (100 \u2212 {thresh*100:.0f}) = {s:.2f}",
            ))

    # elevated_tail
    if profile.bf_pctl is not None and profile.bf_pctl > t["bf_pctl"]:
        val, thresh = profile.bf_pctl, t["bf_pctl"]
        s = min((val - thresh) / (1.0 - thresh), 1.0)
        signals.append(Signal(
            signal_type="elevated_tail", strength=s,
            detail=f"BF {val:.0%} pctl", direction="neutral",
            metric_name="BF Percentile", metric_value=round(val * 100, 1),
            threshold_value=round(thresh * 100, 1),
            strength_formula=f"({val*100:.0f} \u2212 {thresh*100:.0f}) / (100 \u2212 {thresh*100:.0f}) = {s:.2f}",
        ))

    return signals


# ---------------------------------------------------------------------------
# Trend classification (SMA ordering → 6 states)
# ---------------------------------------------------------------------------

_TREND_MAP = {
    ('L', 'S', 'M'): 1,  # Possible uptrend
    ('S', 'L', 'M'): 2,  # Confirmed uptrend
    ('S', 'M', 'L'): 3,  # Well-defined uptrend
    ('M', 'S', 'L'): 4,  # Possible downtrend
    ('M', 'L', 'S'): 5,  # Confirmed downtrend
    ('L', 'M', 'S'): 6,  # Well-defined downtrend
}

_TREND_LABELS = {
    1: "Possible uptrend (L>S>M)",
    2: "Confirmed uptrend (S>L>M)",
    3: "Well-defined uptrend (S>M>L)",
    4: "Possible downtrend (M>S>L)",
    5: "Confirmed downtrend (M>L>S)",
    6: "Well-defined downtrend (L>M>S)",
}


def classify_trend(sma_s: float, sma_m: float, sma_l: float) -> int:
    """Return trend state 1-6 from SMA ordering. 0 = undefined (e.g. ties)."""
    vals = [(sma_s, 'S'), (sma_m, 'M'), (sma_l, 'L')]
    order = tuple(label for _, label in sorted(vals, key=lambda x: x[0], reverse=True))
    return _TREND_MAP.get(order, 0)


def _detect_trend_signals(profile: SecurityProfile) -> list[Signal]:
    """Detect trend signal from pre-classified trend state."""
    signals: list[Signal] = []
    ts = profile.trend_state

    sma_vals = [("S", profile.sma_s), ("M", profile.sma_m), ("L", profile.sma_l)]
    sma_sorted = sorted(sma_vals, key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
    ordering = ">".join(f"{lbl}={v:.0f}" for lbl, v in sma_sorted if v is not None)

    if ts in (2, 3):
        s = 0.60 if ts == 2 else 0.85
        signals.append(Signal(
            signal_type="trend_bullish", strength=s,
            detail=_TREND_LABELS.get(ts, f"Trend {ts}"), direction="bullish",
            metric_name="SMA Ordering", metric_value=float(ts),
            threshold_value=None,
            strength_formula=f"{ordering} \u2192 Trend {ts} \u2192 {s:.2f}",
        ))
    elif ts in (5, 6):
        s = 0.60 if ts == 5 else 0.85
        signals.append(Signal(
            signal_type="trend_bearish", strength=s,
            detail=_TREND_LABELS.get(ts, f"Trend {ts}"), direction="bearish",
            metric_name="SMA Ordering", metric_value=float(ts),
            threshold_value=None,
            strength_formula=f"{ordering} \u2192 Trend {ts} \u2192 {s:.2f}",
        ))
    elif ts in (1, 4):
        direction = "bullish" if ts == 1 else "bearish"
        signals.append(Signal(
            signal_type="trend_transitional", strength=0.40,
            detail=_TREND_LABELS.get(ts, f"Trend {ts}"), direction=direction,
            metric_name="SMA Ordering", metric_value=float(ts),
            threshold_value=None,
            strength_formula=f"{ordering} \u2192 Trend {ts} \u2192 0.40",
        ))

    return signals


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------

_BREAKOUT_TIMEFRAMES = {
    '3mo': {'label': '3-month'},
    '6mo': {'label': '6-month'},
    '12mo': {'label': '12-month'},
}

_BREAKOUT_STRENGTH = {
    '3mo': 0.50,
    '6mo': 0.75,
    '12mo': 1.00,
}


def detect_breakouts(
    px: float,
    highs: dict,
    lows: dict,
) -> dict[str, str | None]:
    """Detect breakout/breakdown for each timeframe with novelty filter.

    A breakout requires BOTH:
    - px >= full window high (current extremum)
    - px > prior window high (novelty — not just re-testing old high)

    Returns dict: {'3mo': 'breakout'|'breakdown'|None, ...}
    """
    result: dict[str, str | None] = {}
    for tf in _BREAKOUT_TIMEFRAMES:
        h_full, h_prior = highs.get(tf, (None, None))
        l_full, l_prior = lows.get(tf, (None, None))

        if h_full is not None and h_prior is not None and px >= h_full and px > h_prior:
            result[tf] = 'breakout'
        elif l_full is not None and l_prior is not None and px <= l_full and px < l_prior:
            result[tf] = 'breakdown'
        else:
            result[tf] = None
    return result


def _detect_breakout_signals(profile: SecurityProfile) -> list[Signal]:
    """Convert breakout dict to Signal objects."""
    signals: list[Signal] = []
    px = profile.px
    for tf, status in profile.breakouts.items():
        if status is None:
            continue
        s = _BREAKOUT_STRENGTH.get(tf, 0.5)
        label = _BREAKOUT_TIMEFRAMES.get(tf, {}).get('label', tf)
        if status == 'breakout':
            signals.append(Signal(
                signal_type="breakout", strength=s,
                detail=f"{label} new high", direction="bullish",
                metric_name=f"Price vs {label} High",
                metric_value=round(px, 2) if px is not None else None,
                threshold_value=None,
                strength_formula=f"{label} breakout \u2192 fixed {s:.2f}",
            ))
        elif status == 'breakdown':
            signals.append(Signal(
                signal_type="breakdown", strength=s,
                detail=f"{label} new low", direction="bearish",
                metric_name=f"Price vs {label} Low",
                metric_value=round(px, 2) if px is not None else None,
                threshold_value=None,
                strength_formula=f"{label} breakdown \u2192 fixed {s:.2f}",
            ))
    return signals


# ---------------------------------------------------------------------------
# Master signal detector
# ---------------------------------------------------------------------------

def detect_signals(
    profile: SecurityProfile,
    thresholds: dict | None = None,
) -> list[Signal]:
    """Detect all active vol and technical signals for a security.

    Returns list of Signal objects.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    signals: list[Signal] = []

    signals.extend(_detect_vol_signals(profile, t))

    if profile.trend_state is not None and profile.trend_state != 0:
        signals.extend(_detect_trend_signals(profile))
    if profile.breakouts:
        signals.extend(_detect_breakout_signals(profile))

    return signals
