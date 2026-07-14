"""Per-underlying signal computation for portfolio review.

Each signal function takes a single-row pandas Series (from the underlying
snapshot DataFrame) and returns a Signal record or None. The runner
runs every signal against every underlying and returns a per-ticker dict
of signal lists.

Signal thresholds use institutional conventions documented in
``SIGNAL_DEFINITIONS`` — these are the bands a sales/trading desk would
use when discussing a position without context-specific historical
percentiles. The thresholds are intentionally simple universal bands;
per-name calibration (vol percentiles, sector-relative trend, etc.) is a
future extension when historical data fetches are added.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

# Signal lives in pm.core.scanner_models and is re-exported via
# signal_detector. Import via signal_detector.
from pm.core.signal_detector import Signal


# Field-name mapping — the BBG field names from the snapshot output.
# Centralized here so they can be extended.
FIELDS = {
    "spot":        "PX_LAST",
    "chg_1d":      "CHG_PCT_1D",
    "iv_3m":       "3MTH_IMPVOL_100.0%MNY_DF",
    "iv_6m":       "6MTH_IMPVOL_100.0%MNY_DF",
    "ytd":         "CHG_PCT_YTD",
    "ma_200d":     "MOV_AVG_200D",
    # --- Trend / momentum / vol fields ---
    "iv_1m":       "CALL_IMP_VOL_30D",        # 30D call IV — 1MTH_IMPVOL_100.0%MNY_DF
                                              # returns NULL via BDP for most US names
    "ma_50d":      "MOV_AVG_50D",
    "beta":        "BETA_ADJ_OVERRIDABLE",
    "rsi_14d":     "RSI_14D",
    "earn_dt":     "EXPECTED_REPORT_DT",      # NEXT_EARN_DT is NULL via BDP;
                                              # EXPECTED_REPORT_DT works.
    "eps_est":     "BEST_EPS",
    "div_yld":     "EQY_DVD_YLD_IND",
    "pcr":         "PUT_CALL_VOLUME_RATIO_CUR_DAY",
    "rv_30d":      "VOLATILITY_30D",          # BDP-available realized vol
                                              # (annualized %, 30D window)
    # --- Earnings implied-move field ---
    # First-choice fields EARN_DT_IMPLIED_MOVE_PCT / EARNINGS_IMPLIED_MOVE_PCT
    # both return NULL via BDP. EARNINGS_RELATED_IMPLIED_MOVE (already in
    # UNDERLYING_FIELDS) populates correctly. Avg-realized
    # earnings reaction has no working BDP field — left as None; UI shows "—".
    "earn_implied_move": "EARNINGS_RELATED_IMPLIED_MOVE",
    "earn_avg_realized": None,
}

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Signal definitions registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalDefinition:
    """Documentation for one signal — rendered in the Definitions panel."""
    signal_type: str
    display_name: str
    what_it_measures: str
    formula: str
    thresholds: list[tuple[str, str, str]]
    # Each tuple: (band_label, range_expression, direction)
    rationale: str


SIGNAL_DEFINITIONS: dict[str, SignalDefinition] = {
    "move_vs_iv": SignalDefinition(
        signal_type="move_vs_iv",
        display_name="Move vs Implied Vol",
        what_it_measures=(
            "Today's price move scaled by the option market's expected daily "
            "standard deviation. Tells you whether today's move was within "
            "what was priced in or a genuine surprise."
        ),
        formula=(
            "z = |CHG_PCT_1D| / (3M_ATM_IV / sqrt(252))   "
            "[i.e. today's % move divided by implied daily \u03c3]"
        ),
        thresholds=[
            ("Quiet",      "z < 1.0\u03c3",          "neutral"),
            ("Noteworthy", "1.0\u03c3 \u2264 z < 2.0\u03c3", "neutral"),
            ("Alert",      "2.0\u03c3 \u2264 z < 3.0\u03c3", "directional"),
            ("Severe",     "z \u2265 3.0\u03c3",     "directional"),
        ],
        rationale=(
            "2\u03c3 is the standard institutional threshold for a 'tradeable "
            "surprise' \u2014 a move beyond what implied vol priced in. "
            "Direction (UP / DOWN) is taken from the sign of CHG_PCT_1D."
        ),
    ),
    "trend_200d": SignalDefinition(
        signal_type="trend_200d",
        display_name="Trend (200D MA)",
        what_it_measures=(
            "Spot price relative to the 200-day moving average. The 200D MA "
            "is the canonical long-term trend filter used across the Street."
        ),
        formula="(PX_LAST - MOV_AVG_200D) / MOV_AVG_200D",
        thresholds=[
            ("Bullish",       "> +5%",       "bullish"),
            ("Near neutral",  "-5% to +5%",  "neutral"),
            ("Bearish",       "< -5%",       "bearish"),
        ],
        rationale=(
            "5% buffer around the 200D MA prevents whipsaws from being "
            "labeled as trend changes. Names trading 5%+ above 200D are "
            "in confirmed long-term uptrend; names 5%+ below are in confirmed "
            "downtrend; the buffer zone is transitional."
        ),
    ),
    "ytd_performance": SignalDefinition(
        signal_type="ytd_performance",
        display_name="YTD Performance",
        what_it_measures=(
            "Year-to-date total return bucketed into institutional categories. "
            "Sets context for trade sizing \u2014 leaders deserve different "
            "treatment than laggards."
        ),
        formula="CHG_PCT_YTD",
        thresholds=[
            ("Significant laggard", "< -20%",       "bearish"),
            ("Underperformer",      "-20% to -5%",  "bearish"),
            ("Flat",                "-5% to +5%",   "neutral"),
            ("In line",             "+5% to +20%",  "bullish"),
            ("Leader",              "> +20%",       "bullish"),
        ],
        rationale=(
            "\u00b15% / \u00b120% bands are the working buckets sell-side "
            "strategy desks use to classify YTD performance for tactical "
            "positioning discussions."
        ),
    ),
    "iv_level": SignalDefinition(
        signal_type="iv_level",
        display_name="3M ATM IV Level",
        what_it_measures=(
            "Absolute level of 3M at-the-money implied volatility. Identifies "
            "overlay opportunities (rich vol = sell premium) and hedging cost "
            "(very high vol = expensive to buy protection)."
        ),
        formula="3MTH_IMPVOL_100.0%MNY_DF (in vol points)",
        thresholds=[
            ("Low",       "< 20%",     "bullish for premium-buyers"),
            ("Normal",    "20%\u201340%",   "neutral"),
            ("Elevated",  "40%\u201360%",   "bullish for premium-sellers"),
            ("Very high", "> 60%",     "stress / event-driven"),
        ],
        rationale=(
            "Universal absolute bands \u2014 useful for a first-cut "
            "conversation. Per-name historical percentile rank is more "
            "precise but requires BDH historical pulls, which are not "
            "enabled here. For now: 30% IV on KO is rich, 30% IV on TSLA "
            "is cheap \u2014 the band-based view lets the salesperson "
            "eyeball either case."
        ),
    ),
    "iv_term": SignalDefinition(
        signal_type="iv_term",
        display_name="IV Term Structure (3M-6M)",
        what_it_measures=(
            "Differential between 3-month and 6-month ATM implied vol. "
            "Inverted curves (3M > 6M) signal event-driven near-term risk; "
            "contango (6M > 3M) is the normal calm regime."
        ),
        formula="iv_3m - iv_6m  (in vol points)",
        thresholds=[
            ("Inverted",  "> +2 pts",     "event-driven"),
            ("Flat",      "-2 to +2 pts", "neutral"),
            ("Contango",  "< -2 pts",     "calm"),
        ],
        rationale=(
            "2-point buffer absorbs noise. Inverted = the market is pricing "
            "a known catalyst (earnings, M&A, drug approval). Contango is "
            "the default for a name with no near-term catalyst."
        ),
    ),
    "earnings_within_30d": SignalDefinition(
        signal_type="earnings_within_30d",
        display_name="Earnings Window",
        what_it_measures=(
            "Days until the next reported earnings announcement. Anchors "
            "expected-move and IV-rich-zone trades."
        ),
        formula="EXPECTED_REPORT_DT \u2212 today (calendar days)",
        thresholds=[
            ("Imminent",    "\u2264 7 days",  "event"),
            ("Approaching", "8\u201314 days", "event"),
            ("Upcoming",    "15\u201330 days", "event"),
            ("Watch",       "31\u201360 days", "event"),
        ],
        rationale=(
            "30-day window is the standard pre-earnings positioning horizon "
            "for option overlays. 14-day is the IV-rich zone where premium "
            "selling has best risk/reward."
        ),
    ),
    "vol_risk_premium": SignalDefinition(
        signal_type="vol_risk_premium",
        display_name="Vol Risk Premium (1M\u201330D)",
        what_it_measures=(
            "Difference between 1M implied vol and 30-day realized vol. "
            "Persistent positive VRP \u21d2 premium-selling overlay; negative "
            "\u21d2 protection / long-vol opportunity."
        ),
        formula="iv_1m \u2212 rv_30d  (in vol points)",
        thresholds=[
            ("Rich VRP",       "> +8 pts",     "premium-sell"),
            ("Mild VRP",       "+3 to +8 pts", "premium-sell"),
            ("Fair",           "\u22123 to +3 pts", "neutral"),
            ("Cheap IV",       "\u22128 to \u22123 pts", "premium-buy"),
            ("Very cheap IV",  "< \u22128 pts",     "premium-buy"),
        ],
        rationale=(
            "VRP > +3 pts = persistent overlay opportunity; VRP < \u22123 = "
            "realized vol exceeds implied (recent shock or stale IV mark). "
            "8-pt threshold marks meaningful regimes."
        ),
    ),
    "iv_percentile": SignalDefinition(
        signal_type="iv_percentile",
        display_name="IV Percentile (1Y)",
        what_it_measures=(
            "Rank of current 1M IV within the trailing 1-year history of the "
            "same field. Per-name calibration of cheap vs rich IV."
        ),
        formula="rank(current_iv) within 1Y daily history",
        thresholds=[
            ("Cheap (\u226420th)",   "\u2264 0.20", "premium-buy"),
            ("Below average",        "0.20\u20130.40", "neutral"),
            ("Median",               "0.40\u20130.60", "neutral"),
            ("Above average",        "0.60\u20130.80", "neutral"),
            ("Rich (\u226580th)",    "\u2265 0.80", "premium-sell"),
        ],
        rationale=(
            "20th/80th-percentile bands are the standard cheap/rich IV "
            "cutoffs from the Vol Screener affinity matrix."
        ),
    ),
    "rsi_extreme": SignalDefinition(
        signal_type="rsi_extreme",
        display_name="RSI Extreme",
        what_it_measures=(
            "14-day Wilder RSI. Extremes (\u226570 / \u226430) flag mean-"
            "reversion candidates. Mid-range RSI does not fire a signal."
        ),
        formula="RSI(14) from BDP",
        thresholds=[
            ("Overbought", "\u2265 70", "bearish"),
            ("Oversold",   "\u2264 30", "bullish"),
        ],
        rationale=(
            "Wilder's classic 70/30 thresholds. RSI extremes are a mean-"
            "reversion timing input, not a primary direction call."
        ),
    ),
    "momentum": SignalDefinition(
        signal_type="momentum",
        display_name="Momentum (SMA stack)",
        what_it_measures=(
            "Trend regime via the ordering of 50/100/200-day SMAs. Maps to "
            "the 6-state Vol Screener trend classification."
        ),
        formula="classify_trend(50D, 100D, 200D)",
        thresholds=[
            ("Strong bullish",   "state 3 (S>M>L)", "bullish"),
            ("Bullish",          "state 2 (S>L>M)", "bullish"),
            ("Emerging bullish", "state 1 (L>S>M)", "bullish"),
            ("Emerging bearish", "state 4 (M>S>L)", "bearish"),
            ("Bearish",          "state 5 (M>L>S)", "bearish"),
            ("Strong bearish",   "state 6 (L>M>S)", "bearish"),
        ],
        rationale=(
            "50/100/200D SMA ordering classifies trend regime into 6 states "
            "per the Vol Screener convention. States 3 and 6 are strongest. "
            "100D SMA is BDH-computed; 50D and 200D come from BDP."
        ),
    ),
    "breakout": SignalDefinition(
        signal_type="breakout",
        display_name="Breakout / Breakdown",
        what_it_measures=(
            "Spot crossing trailing-window highs (breakout) or lows "
            "(breakdown) over 60 / 130 / 250 day windows."
        ),
        formula="detect_breakouts(spot, trailing highs/lows)",
        thresholds=[
            ("Breakout",  "spot > full-window high AND > prior-window high", "bullish"),
            ("Breakdown", "spot < full-window low AND < prior-window low",   "bearish"),
        ],
        rationale=(
            "Novelty-filtered: spot must exceed both the full-window AND "
            "prior-window high to count. Reduces noise vs. simple Donchian "
            "breakouts."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Individual signal functions
# ---------------------------------------------------------------------------

def _safe_get(row: pd.Series, key: str):
    """row.get with NaN-safe semantics. Returns the raw value or None when
    missing/NaN — caller should still pd.isna check numerics."""
    val = row.get(key)
    return val


def signal_move_vs_iv(row: pd.Series) -> Signal | None:
    chg = _safe_get(row, FIELDS["chg_1d"])
    iv = _safe_get(row, FIELDS["iv_3m"])
    if chg is None or iv is None or pd.isna(chg) or pd.isna(iv) or iv <= 0:
        return None
    chg = float(chg)
    iv = float(iv)
    daily_sigma_pct = iv / math.sqrt(TRADING_DAYS_PER_YEAR)
    z = abs(chg) / daily_sigma_pct
    direction = "bullish" if chg > 0 else "bearish" if chg < 0 else "neutral"
    if z >= 3.0:
        band = "Severe"
        strength = min(1.0, z / 4.0)
    elif z >= 2.0:
        band = "Alert"
        strength = 0.5 + (z - 2.0) * 0.25
    elif z >= 1.0:
        band = "Noteworthy"
        strength = 0.2 + (z - 1.0) * 0.3
    else:
        band = "Quiet"
        strength = z * 0.2
        direction = "neutral"
    return Signal(
        signal_type="move_vs_iv",
        strength=round(strength, 3),
        detail=f"{band}: {chg:+.2f}% move ({z:.1f}\u03c3 vs implied daily)",
        direction=direction,
        metric_name="z_score_vs_iv",
        metric_value=round(z, 2),
        threshold_value=2.0,
        strength_formula="|chg_1d| / (iv_3m / sqrt(252))",
    )


def signal_trend_200d(row: pd.Series) -> Signal | None:
    spot = _safe_get(row, FIELDS["spot"])
    ma = _safe_get(row, FIELDS["ma_200d"])
    if spot is None or ma is None or pd.isna(spot) or pd.isna(ma) or ma <= 0:
        return None
    spot = float(spot)
    ma = float(ma)
    premium = (spot - ma) / ma
    if premium > 0.05:
        band, direction = "Bullish", "bullish"
        strength = min(1.0, 0.5 + premium * 5)
    elif premium < -0.05:
        band, direction = "Bearish", "bearish"
        strength = min(1.0, 0.5 + abs(premium) * 5)
    else:
        band, direction = "Near neutral", "neutral"
        strength = 0.3
    return Signal(
        signal_type="trend_200d",
        strength=round(strength, 3),
        detail=f"{band}: spot {premium*100:+.1f}% vs 200D MA",
        direction=direction,
        metric_name="px_vs_ma200d_pct",
        metric_value=round(premium * 100, 2),
        threshold_value=5.0,
        strength_formula="(spot - ma_200d) / ma_200d",
    )


def signal_ytd_performance(row: pd.Series) -> Signal | None:
    ytd = _safe_get(row, FIELDS["ytd"])
    if ytd is None or pd.isna(ytd):
        return None
    ytd = float(ytd)
    if ytd < -20:
        band, direction, strength = "Significant laggard", "bearish", 0.9
    elif ytd < -5:
        band, direction, strength = "Underperformer", "bearish", 0.5
    elif ytd <= 5:
        band, direction, strength = "Flat", "neutral", 0.2
    elif ytd <= 20:
        band, direction, strength = "In line", "bullish", 0.5
    else:
        band, direction, strength = "Leader", "bullish", 0.9
    return Signal(
        signal_type="ytd_performance",
        strength=strength,
        detail=f"{band}: {ytd:+.1f}% YTD",
        direction=direction,
        metric_name="ytd_return_pct",
        metric_value=round(ytd, 2),
        threshold_value=0.0,
        strength_formula="CHG_PCT_YTD bucketed",
    )


def signal_iv_level(row: pd.Series) -> Signal | None:
    iv = _safe_get(row, FIELDS["iv_3m"])
    if iv is None or pd.isna(iv):
        return None
    iv = float(iv)
    if iv < 20:
        band, strength = "Low", 0.3
    elif iv < 40:
        band, strength = "Normal", 0.2
    elif iv < 60:
        band, strength = "Elevated", 0.6
    else:
        band, strength = "Very high", 0.95
    return Signal(
        signal_type="iv_level",
        strength=strength,
        detail=f"{band}: {iv:.1f}% 3M ATM IV",
        direction="neutral",
        metric_name="iv_3m_pct",
        metric_value=round(iv, 2),
        threshold_value=40.0,
        strength_formula="absolute IV bands",
    )


def signal_iv_term(row: pd.Series) -> Signal | None:
    iv3 = _safe_get(row, FIELDS["iv_3m"])
    iv6 = _safe_get(row, FIELDS["iv_6m"])
    if iv3 is None or iv6 is None or pd.isna(iv3) or pd.isna(iv6):
        return None
    iv3 = float(iv3)
    iv6 = float(iv6)
    diff = iv3 - iv6
    if diff > 2:
        band, direction = "Inverted", "neutral"
        strength = min(1.0, 0.5 + diff * 0.05)
    elif diff < -2:
        band, direction = "Contango", "neutral"
        strength = min(1.0, 0.3 + abs(diff) * 0.03)
    else:
        band, direction, strength = "Flat", "neutral", 0.2
    return Signal(
        signal_type="iv_term",
        strength=round(strength, 3),
        detail=f"{band}: 3M-6M = {diff:+.1f} pts",
        direction=direction,
        metric_name="iv_term_diff_pts",
        metric_value=round(diff, 2),
        threshold_value=2.0,
        strength_formula="iv_3m - iv_6m",
    )


# ---------------------------------------------------------------------------
# History-derived signals — some take kwargs from the caller
# (BDH price / IV history, computed RV / IV percentile).
# ---------------------------------------------------------------------------

def signal_earnings_within_30d(row: pd.Series) -> Signal | None:
    earn_dt = _safe_get(row, FIELDS["earn_dt"])
    if earn_dt is None:
        return None
    try:
        if pd.isna(earn_dt):
            return None
    except (TypeError, ValueError):
        return None
    try:
        earn_date = pd.to_datetime(earn_dt).date()
    except Exception:
        return None
    today = pd.Timestamp.today().date()
    days_to_earnings = (earn_date - today).days
    if days_to_earnings < 0 or days_to_earnings > 60:
        return None
    if days_to_earnings <= 7:
        band, strength = "Imminent", 0.95
    elif days_to_earnings <= 14:
        band, strength = "Approaching", 0.7
    elif days_to_earnings <= 30:
        band, strength = "Upcoming", 0.4
    else:
        band, strength = "Watch", 0.2
    return Signal(
        signal_type="earnings_within_30d",
        strength=strength,
        detail=f"{band}: earnings in {days_to_earnings} days ({earn_date})",
        direction="event",
        metric_name="days_to_earnings",
        metric_value=float(days_to_earnings),
        threshold_value=30.0,
        strength_formula="bucketed by days-to-earnings",
    )


def signal_vol_risk_premium(
    row: pd.Series,
    realized_vol_30d: float | None = None,
) -> Signal | None:
    iv = _safe_get(row, FIELDS["iv_1m"])
    if iv is None or realized_vol_30d is None:
        return None
    try:
        if pd.isna(iv):
            return None
    except (TypeError, ValueError):
        return None
    iv = float(iv)
    rv = float(realized_vol_30d)
    vrp = iv - rv
    if vrp > 8:
        band, direction, strength = "Rich VRP", "premium-sell", 0.85
    elif vrp > 3:
        band, direction, strength = "Mild VRP", "premium-sell", 0.5
    elif vrp > -3:
        band, direction, strength = "Fair", "neutral", 0.2
    elif vrp > -8:
        band, direction, strength = "Cheap IV", "premium-buy", 0.5
    else:
        band, direction, strength = "Very cheap IV", "premium-buy", 0.85
    return Signal(
        signal_type="vol_risk_premium",
        strength=strength,
        detail=f"{band}: 1M IV {iv:.1f}% \u2212 30D RV {rv:.1f}% = {vrp:+.1f} pts",
        direction=direction,
        metric_name="vrp_pts",
        metric_value=round(vrp, 2),
        threshold_value=3.0,
        strength_formula="iv_1m \u2212 rv_30d",
    )


def signal_iv_percentile_signal(
    row: pd.Series,
    iv_pctl: float | None = None,
) -> Signal | None:
    if iv_pctl is None:
        return None
    if iv_pctl <= 0.20:
        band, direction, strength = "Cheap (\u226420th)", "premium-buy", 0.85
    elif iv_pctl <= 0.40:
        band, direction, strength = "Below average", "neutral", 0.4
    elif iv_pctl <= 0.60:
        band, direction, strength = "Median", "neutral", 0.2
    elif iv_pctl <= 0.80:
        band, direction, strength = "Above average", "neutral", 0.4
    else:
        band, direction, strength = "Rich (\u226580th)", "premium-sell", 0.85
    return Signal(
        signal_type="iv_percentile",
        strength=strength,
        detail=f"{band}: IV at {iv_pctl*100:.0f}th %ile of trailing 1Y",
        direction=direction,
        metric_name="iv_1y_percentile",
        metric_value=round(iv_pctl, 3),
        threshold_value=0.80,
        strength_formula="rank of current IV in 1Y series",
    )


def signal_rsi_extreme(row: pd.Series) -> Signal | None:
    rsi = _safe_get(row, FIELDS["rsi_14d"])
    if rsi is None:
        return None
    try:
        if pd.isna(rsi):
            return None
    except (TypeError, ValueError):
        return None
    rsi = float(rsi)
    if rsi >= 70:
        band, direction = "Overbought", "bearish"
        strength = min(1.0, (rsi - 70) / 15 + 0.5)
        threshold = 70.0
    elif rsi <= 30:
        band, direction = "Oversold", "bullish"
        strength = min(1.0, (30 - rsi) / 15 + 0.5)
        threshold = 30.0
    else:
        return None
    return Signal(
        signal_type="rsi_extreme",
        strength=round(strength, 3),
        detail=f"{band}: RSI(14) = {rsi:.1f}",
        direction=direction,
        metric_name="rsi_14d",
        metric_value=round(rsi, 1),
        threshold_value=threshold,
        strength_formula="14-day Wilder RSI from BDP",
    )


def signal_momentum(
    row: pd.Series,
    sma_100d: float | None = None,
) -> Signal | None:
    """Wraps classify_trend(short=50D, mid=100D, long=200D) into a signal."""
    from pm.core.signal_detector import classify_trend

    short = _safe_get(row, FIELDS["ma_50d"])
    long = _safe_get(row, FIELDS["ma_200d"])
    if short is None or long is None or sma_100d is None:
        return None
    try:
        if pd.isna(short) or pd.isna(long):
            return None
    except (TypeError, ValueError):
        return None

    trend_state = classify_trend(float(short), float(sma_100d), float(long))
    state_map = {
        1: ("Emerging bullish", "bullish", 0.4),
        2: ("Bullish",          "bullish", 0.7),
        3: ("Strong bullish",   "bullish", 0.95),
        4: ("Emerging bearish", "bearish", 0.4),
        5: ("Bearish",          "bearish", 0.7),
        6: ("Strong bearish",   "bearish", 0.95),
    }
    if trend_state not in state_map:
        return None
    band, direction, strength = state_map[trend_state]
    return Signal(
        signal_type="momentum",
        strength=strength,
        detail=f"{band}: trend state {trend_state} (50/100/200D SMA stack)",
        direction=direction,
        metric_name="trend_state",
        metric_value=float(trend_state),
        threshold_value=3.0,
        strength_formula="classify_trend(50D, 100D, 200D)",
    )


def signal_breakout(
    row: pd.Series,
    highs: dict[int, float] | None = None,
    lows: dict[int, float] | None = None,
) -> Signal | None:
    """Wraps detect_breakouts into a signal.

    ``highs`` / ``lows`` are dicts keyed by lookback days (e.g. 60, 130, 250)
    with the *current* trailing extreme as value. We adapt them into
    ``signal_detector.detect_breakouts``'s ``{tf: (full_high, prior_high)}``
    convention by deriving the prior-window extremum as the next-larger
    window's value (e.g. prior_high for 60d ~= 130d high — strict-novelty
    filter ensures both must be exceeded).
    """
    from pm.core.signal_detector import detect_breakouts

    spot = _safe_get(row, FIELDS["spot"])
    if spot is None or not highs or not lows:
        return None
    try:
        if pd.isna(spot):
            return None
    except (TypeError, ValueError):
        return None

    # Map (60, 130, 250) → ('3mo', '6mo', '12mo') with prior-window = next-up
    tf_map = {60: "3mo", 130: "6mo", 250: "12mo"}
    sorted_windows = sorted(tf_map.keys())
    highs_arg: dict[str, tuple[float | None, float | None]] = {}
    lows_arg: dict[str, tuple[float | None, float | None]] = {}
    for i, w in enumerate(sorted_windows):
        prior_w = sorted_windows[i + 1] if i + 1 < len(sorted_windows) else w
        highs_arg[tf_map[w]] = (highs.get(w), highs.get(prior_w))
        lows_arg[tf_map[w]] = (lows.get(w), lows.get(prior_w))

    breakouts = detect_breakouts(float(spot), highs_arg, lows_arg)
    fired_keys = [k for k, v in breakouts.items() if v]
    if not fired_keys:
        return None

    direction = (
        "bullish" if any(breakouts[k] == "breakout" for k in fired_keys)
        else "bearish"
    )
    band = "Breakout" if direction == "bullish" else "Breakdown"
    return Signal(
        signal_type="breakout",
        strength=0.8,
        detail=f"{band}: {', '.join(fired_keys)} (vs trailing windows)",
        direction=direction,
        metric_name="breakouts_fired",
        metric_value=float(len(fired_keys)),
        threshold_value=1.0,
        strength_formula="detect_breakouts(spot, trailing_highs, trailing_lows)",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

ALL_SIGNAL_FUNCTIONS: list[Callable[[pd.Series], Signal | None]] = [
    signal_move_vs_iv,
    signal_trend_200d,
    signal_ytd_performance,
    signal_iv_level,
    signal_iv_term,
]


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def compute_per_underlying_signals(
    underlying_snapshot: pd.DataFrame,
    bloomberg_available: bool = True,
    iv_histories: Optional[dict] = None,
) -> dict[str, list[Signal]]:
    """Run all signal functions against every row of the snapshot.

    Returns a dict keyed by ticker (snapshot index value), value = list of
    non-None Signal records. Tickers with all signals returning None get an
    empty list (still present in the dict).

    History-derived signals (vol_risk_premium, iv_percentile, momentum,
    breakout) require BDH fetches; those degrade gracefully if BBG is
    unavailable or returns nothing.

    ``iv_histories``: an already-fetched ``{ticker: pd.Series}`` 1Y IV-history
    dict to reuse — the load path fetches it once and shares it with the
    insight prefetch, so the same BDH is not issued twice per load. ``None``
    means fetch here (standalone callers).
    """
    from pm.core.vol_metrics import (
        iv_percentile,
        realized_vol_from_prices,
        trailing_high_low,
    )

    out: dict[str, list[Signal]] = {}
    if underlying_snapshot is None or underlying_snapshot.empty:
        return out

    tickers = list(underlying_snapshot.index)
    price_hist: dict[str, pd.Series] = {}
    iv_hist: dict[str, pd.Series] = dict(iv_histories) if iv_histories is not None else {}
    if bloomberg_available and tickers:
        try:
            from pm.core.bloomberg_client import (
                fetch_iv_history,
                fetch_price_history,
            )
            price_hist = fetch_price_history(tickers, lookback_days=365)
            if iv_histories is None:
                iv_hist = fetch_iv_history(tickers, lookback_days=365)
        except Exception:
            # Pipeline must keep going with degraded signals on history
            # fetch failure.
            pass

    for ticker, row in underlying_snapshot.iterrows():
        ph = price_hist.get(ticker)
        ih = iv_hist.get(ticker)

        # Per-ticker history-derived inputs
        # Use BDP VOLATILITY_30D as the primary RV source (probed live).
        # Fall back to BDH-computed RV if BDP value is missing.
        rv_30d = _safe_get(row, FIELDS["rv_30d"])
        try:
            if rv_30d is None or pd.isna(rv_30d):
                rv_30d = (
                    realized_vol_from_prices(ph, window_days=30)
                    if ph is not None else None
                )
            else:
                rv_30d = float(rv_30d)
        except (TypeError, ValueError):
            rv_30d = (
                realized_vol_from_prices(ph, window_days=30)
                if ph is not None else None
            )

        iv_1m_current = _safe_get(row, FIELDS["iv_1m"])
        iv_pctl = (
            iv_percentile(iv_1m_current, ih) if ih is not None else None
        )
        sma_100d = (
            float(ph.tail(100).mean())
            if ph is not None and len(ph) >= 100 else None
        )

        highs: dict[int, float] = {}
        lows: dict[int, float] = {}
        if ph is not None:
            for w in (60, 130, 250):
                hi, lo = trailing_high_low(ph, w)
                if hi is not None:
                    highs[w] = hi
                if lo is not None:
                    lows[w] = lo

        sigs: list[Signal] = []

        # Stateless signals
        for fn in ALL_SIGNAL_FUNCTIONS:
            s = _safe_call(fn, row)
            if s is not None:
                sigs.append(s)

        # History-derived signals
        for fn, kwargs in [
            (signal_earnings_within_30d, {}),
            (signal_vol_risk_premium,    {"realized_vol_30d": rv_30d}),
            (signal_iv_percentile_signal, {"iv_pctl": iv_pctl}),
            (signal_rsi_extreme,         {}),
            (signal_momentum,            {"sma_100d": sma_100d}),
            (signal_breakout,            {"highs": highs, "lows": lows}),
        ]:
            s = _safe_call(fn, row, **kwargs)
            if s is not None:
                sigs.append(s)

        out[ticker] = sigs
    return out
