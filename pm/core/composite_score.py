"""Per-underlying composite score (0-100): five weighted components.

Inputs are signals already computed by pm.core.portfolio_signals — no new
data fetches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pm.core.scanner_models import Signal


# Component weights — tuned to favor signal density and vol regime
COMPOSITE_WEIGHTS = {
    "signal_count":      0.30,
    "vol_regime":        0.25,
    "trend_clarity":     0.20,
    "event_pressure":    0.15,
    "signal_agreement":  0.10,
}

# Which signal_types feed which component
VOL_SIGNAL_TYPES = {"iv_level", "iv_percentile", "vol_risk_premium", "iv_term"}
TREND_SIGNAL_TYPES = {"trend_200d", "momentum", "ytd_performance"}
EVENT_SIGNAL_TYPES = {"earnings_within_30d", "rsi_extreme", "breakout", "move_vs_iv"}

TOTAL_SIGNALS_POSSIBLE = 11   # the 11 signals in pm.core.portfolio_signals


@dataclass
class CompositeScore:
    total: float                              # 0-100
    components: dict                          # {name: {raw, weighted}}
    label: str                                # 'Strong' | 'Moderate' | 'Weak' | 'Quiet'


def compute_composite_score(signals: Optional[List[Signal]]) -> CompositeScore:
    if signals is None:
        signals = []
    signals = [s for s in signals if s is not None]

    fired = len(signals)

    # 1. Signal Count: how many of the 11 possible signals fired
    signal_count_raw = (fired / TOTAL_SIGNALS_POSSIBLE) * 100

    # 2. Vol Regime: max strength among vol signals * 100
    vol_strengths = [s.strength for s in signals if s.signal_type in VOL_SIGNAL_TYPES]
    vol_regime_raw = (max(vol_strengths) * 100) if vol_strengths else 0

    # 3. Trend Clarity: max strength among trend signals * 100
    trend_strengths = [s.strength for s in signals if s.signal_type in TREND_SIGNAL_TYPES]
    trend_clarity_raw = (max(trend_strengths) * 100) if trend_strengths else 0

    # 4. Event Pressure: max strength among event signals * 100
    event_strengths = [s.strength for s in signals if s.signal_type in EVENT_SIGNAL_TYPES]
    event_pressure_raw = (max(event_strengths) * 100) if event_strengths else 0

    # 5. Signal Agreement: directional consensus
    if fired == 0:
        agreement_raw = 0.0
    else:
        directions = [s.direction for s in signals]
        bullish = sum(1 for d in directions if d == "bullish")
        bearish = sum(1 for d in directions if d == "bearish")
        neutral = sum(1 for d in directions if d == "neutral")
        events = sum(1 for d in directions
                      if d in ("event", "premium-buy", "premium-sell"))
        max_dir = max(bullish, bearish, neutral, events)
        agreement_raw = (max_dir / fired) * 100

    components = {
        "signal_count": {
            "raw": round(signal_count_raw, 1),
            "weighted": round(signal_count_raw * COMPOSITE_WEIGHTS["signal_count"], 2),
        },
        "vol_regime": {
            "raw": round(vol_regime_raw, 1),
            "weighted": round(vol_regime_raw * COMPOSITE_WEIGHTS["vol_regime"], 2),
        },
        "trend_clarity": {
            "raw": round(trend_clarity_raw, 1),
            "weighted": round(trend_clarity_raw * COMPOSITE_WEIGHTS["trend_clarity"], 2),
        },
        "event_pressure": {
            "raw": round(event_pressure_raw, 1),
            "weighted": round(event_pressure_raw * COMPOSITE_WEIGHTS["event_pressure"], 2),
        },
        "signal_agreement": {
            "raw": round(agreement_raw, 1),
            "weighted": round(agreement_raw * COMPOSITE_WEIGHTS["signal_agreement"], 2),
        },
    }
    total = round(sum(c["weighted"] for c in components.values()), 1)

    if total >= 70:
        label = "Strong"
    elif total >= 50:
        label = "Moderate"
    elif total >= 30:
        label = "Weak"
    else:
        label = "Quiet"

    return CompositeScore(total=total, components=components, label=label)

