"""Per-position recommendation engine.

Each rule examines a PositionContext and the signals fired for the
underlying; returns a Recommendation with action label, rationale, and
trigger references. Rules are evaluated in priority order — first match
wins per position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from pm.core.position_context import PositionContext
from pm.core.scanner_models import Signal


# ─────────────────────────────────────────────────────────────────────
# Action vocabulary — the universe of labels a rule can return
# ─────────────────────────────────────────────────────────────────────
ACTIONS = {
    "CLOSE":             "Take profit / cut loss now",
    "ROLL_OUT":          "Extend tenor (same strike, later expiry)",
    "ROLL_DOWN":         "Lower strike (same expiry)",
    "ROLL_UP":           "Raise strike (same expiry)",
    "ROLL_OUT_AND_DOWN": "Lower strike + extend tenor (defensive)",
    "ROLL_UP_AND_OUT":   "Raise strike + extend tenor (avoid assignment on covered call)",
    "HARVEST_THETA":     "Hold short premium and let decay work",
    "ADD_OVERLAY":       "Sell calls vs long stock OR sell puts vs cash",
    "ADD_HEDGE":         "Buy puts / put spreads for downside protection",
    "TRIM":              "Reduce position size",
    "ADD":               "Increase position",
    "MONITOR":           "No action — watch listed triggers",
}

PRIORITY = {
    "urgent":        1,
    "opportunistic": 2,
    "monitor":       3,
}


@dataclass
class Recommendation:
    position_id: str
    instrument_type: str           # 'option' | 'equity'
    action: str                    # one of ACTIONS keys
    priority: str                  # 'urgent' | 'opportunistic' | 'monitor'
    rationale: str
    triggering_signals: list[str]
    metrics: dict
    rule_id: str
    estimated_dollar_value: Optional[float] = None


@dataclass(frozen=True)
class RecommendationRuleDefinition:
    rule_id: str
    display_name: str
    applies_to: str
    triggers: list[str]
    action: str
    rationale: str
    institutional_source: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_signal(signals, signal_type: str, *, contains: str | None = None,
                direction: str | None = None) -> bool:
    """Convenience: was any Signal of this type fired matching the optional
    detail-substring or direction?
    """
    for s in signals:
        if s.signal_type != signal_type:
            continue
        if contains and contains not in (s.detail or ""):
            continue
        if direction and s.direction != direction:
            continue
        return True
    return False


def _signal_types(signals, types: list[str]) -> list[str]:
    """Return the subset of types that fired (preserves caller's order)."""
    fired = {s.signal_type for s in signals}
    return [t for t in types if t in fired]


# ─────────────────────────────────────────────────────────────────────
# 2.1 SHORT PUT rules
# ─────────────────────────────────────────────────────────────────────

def rule_short_put_take_profit(ctx, signals):
    if ctx.instrument_type != "option" or ctx.right != "PUT" or ctx.quantity >= 0:
        return None
    if ctx.pct_captured is None or ctx.dte is None:
        return None
    if ctx.pct_captured >= 0.50 and ctx.dte > 14:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="CLOSE",
            priority="opportunistic",
            rationale=(
                f"Short put has captured {ctx.pct_captured*100:.0f}% of premium "
                f"with {ctx.dte}d remaining. Industry convention: take 50%, redeploy."
            ),
            triggering_signals=[],
            metrics={"pct_captured": ctx.pct_captured, "dte": ctx.dte},
            rule_id="short_put_take_profit_50",
            estimated_dollar_value=abs(ctx.cost_basis or 0) - abs(ctx.market_value),
        )
    return None


def rule_short_put_assignment_risk(ctx, signals):
    if ctx.instrument_type != "option" or ctx.right != "PUT" or ctx.quantity >= 0:
        return None
    if ctx.dte is None or ctx.moneyness is None or ctx.delta is None:
        return None
    if ctx.moneyness > 0.02 and ctx.dte < 21 and ctx.delta > -0.85:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="ROLL_OUT_AND_DOWN",
            priority="urgent",
            rationale=(
                f"Short put is {ctx.moneyness*100:.1f}% ITM with {ctx.dte}d to expiry "
                f"(\u03b4 {ctx.delta:.2f}). Roll out + down to defer assignment, "
                f"lower strike, reset theta clock."
            ),
            triggering_signals=[],
            metrics={
                "moneyness_pct": ctx.moneyness * 100,
                "dte": ctx.dte,
                "delta": ctx.delta,
            },
            rule_id="short_put_assignment_risk",
        )
    return None


def rule_short_put_near_expiry_otm(ctx, signals):
    if ctx.instrument_type != "option" or ctx.right != "PUT" or ctx.quantity >= 0:
        return None
    if ctx.dte is None or ctx.moneyness is None or ctx.delta is None:
        return None
    iv_elevated = _has_signal(signals, "iv_level", contains="Elevated")
    vrp_rich = _has_signal(signals, "vol_risk_premium", contains="Rich")
    if (ctx.moneyness < -0.05 and ctx.dte < 14
            and ctx.delta > -0.20 and (iv_elevated or vrp_rich)):
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="ROLL_OUT",
            priority="opportunistic",
            rationale=(
                f"Short put OTM ({ctx.moneyness*100:.1f}%) with {ctx.dte}d to expiry. "
                f"IV is {'rich' if vrp_rich else 'elevated'}; roll out to capture "
                f"new premium at the same strike."
            ),
            triggering_signals=_signal_types(signals,
                                              ["iv_level", "vol_risk_premium"]),
            metrics={"moneyness_pct": ctx.moneyness * 100, "dte": ctx.dte},
            rule_id="short_put_roll_otm_near_expiry",
        )
    return None


def rule_short_put_pre_earnings_hedge(ctx, signals):
    if ctx.instrument_type != "option" or ctx.right != "PUT" or ctx.quantity >= 0:
        return None
    if ctx.days_to_earnings is None or ctx.delta is None:
        return None
    if ctx.days_to_earnings <= 7 and ctx.delta < -0.25:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="CLOSE",
            priority="urgent",
            rationale=(
                f"Earnings in {ctx.days_to_earnings}d on a \u03b4 {ctx.delta:.2f} "
                f"short put. Gap-down risk exceeds remaining premium; close or buy "
                f"put spread to cap."
            ),
            triggering_signals=["earnings_within_30d"],
            metrics={"days_to_earnings": ctx.days_to_earnings, "delta": ctx.delta},
            rule_id="short_put_pre_earnings_close",
        )
    return None


def rule_short_put_harvest_theta(ctx, signals):
    if ctx.instrument_type != "option" or ctx.right != "PUT" or ctx.quantity >= 0:
        return None
    if ctx.pct_captured is None or ctx.dte is None or ctx.moneyness is None:
        return None
    # Materially underwater positions don't qualify as healthy theta capture;
    # let them fall through to the assignment-risk / monitor rules below.
    if ctx.pct_captured < -0.50:
        return None
    if ctx.moneyness < -0.02 and ctx.pct_captured < 0.50 and ctx.dte > 21:
        if ctx.pct_captured >= 0:
            rationale = (
                f"Short put OTM with {ctx.dte}d remaining and "
                f"{ctx.pct_captured*100:.0f}% of premium captured. "
                f"Hold; theta accruing in your favor."
            )
        else:
            rationale = (
                f"Short put OTM with {ctx.dte}d remaining; "
                f"position currently {abs(ctx.pct_captured)*100:.0f}% "
                f"underwater on premium received. "
                f"Theta should accrue if no breach; monitor underlying for "
                f"deterioration."
            )
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="HARVEST_THETA",
            priority="monitor",
            rationale=rationale,
            triggering_signals=[],
            metrics={"pct_captured": ctx.pct_captured, "dte": ctx.dte},
            rule_id="short_put_harvest_theta",
        )
    return None


# ─────────────────────────────────────────────────────────────────────
# 2.2 SHORT CALL rules
# ─────────────────────────────────────────────────────────────────────

def rule_short_call_take_profit(ctx, signals):
    if ctx.instrument_type != "option" or ctx.right != "CALL" or ctx.quantity >= 0:
        return None
    if ctx.pct_captured is None or ctx.dte is None:
        return None
    if ctx.pct_captured >= 0.50 and ctx.dte > 14:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="CLOSE",
            priority="opportunistic",
            rationale=(
                f"Short call has captured {ctx.pct_captured*100:.0f}% of premium "
                f"with {ctx.dte}d remaining. Industry convention: take 50%, redeploy."
            ),
            triggering_signals=[],
            metrics={"pct_captured": ctx.pct_captured, "dte": ctx.dte},
            rule_id="short_call_take_profit_50",
            estimated_dollar_value=abs(ctx.cost_basis or 0) - abs(ctx.market_value),
        )
    return None


def rule_short_call_assignment_risk(ctx, signals):
    """Covered call ITM and risk of stock being called away."""
    if ctx.instrument_type != "option" or ctx.right != "CALL" or ctx.quantity >= 0:
        return None
    if ctx.dte is None or ctx.moneyness is None or ctx.delta is None:
        return None
    if (ctx.moneyness > 0.02 and ctx.dte < 21
            and ctx.delta > 0.60 and ctx.has_long_stock):
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="ROLL_UP_AND_OUT",
            priority="urgent",
            rationale=(
                f"Short call (covered) is {ctx.moneyness*100:.1f}% ITM with "
                f"{ctx.dte}d to expiry (\u03b4 {ctx.delta:.2f}). Roll up + out to "
                f"protect long stock from assignment, raise strike, reset clock."
            ),
            triggering_signals=[],
            metrics={
                "moneyness_pct": ctx.moneyness * 100,
                "dte": ctx.dte,
                "delta": ctx.delta,
            },
            rule_id="short_call_assignment_risk",
        )
    return None


def rule_short_call_near_expiry_iv_elevated(ctx, signals):
    """Roll out OTM short call when IV is still rich."""
    if ctx.instrument_type != "option" or ctx.right != "CALL" or ctx.quantity >= 0:
        return None
    if ctx.dte is None or ctx.moneyness is None or ctx.delta is None:
        return None
    iv_elevated = _has_signal(signals, "iv_level", contains="Elevated")
    vrp_rich = _has_signal(signals, "vol_risk_premium", contains="Rich")
    if (ctx.moneyness < -0.05 and ctx.dte < 14
            and ctx.delta < 0.20 and (iv_elevated or vrp_rich)):
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="ROLL_OUT",
            priority="opportunistic",
            rationale=(
                f"Short call OTM ({ctx.moneyness*100:.1f}%) with {ctx.dte}d to "
                f"expiry. IV is {'rich' if vrp_rich else 'elevated'}; roll out to "
                f"capture new premium at the same strike."
            ),
            triggering_signals=_signal_types(signals,
                                              ["iv_level", "vol_risk_premium"]),
            metrics={"moneyness_pct": ctx.moneyness * 100, "dte": ctx.dte},
            rule_id="short_call_roll_otm_near_expiry",
        )
    return None


def rule_short_call_harvest_theta(ctx, signals):
    if ctx.instrument_type != "option" or ctx.right != "CALL" or ctx.quantity >= 0:
        return None
    if ctx.pct_captured is None or ctx.dte is None or ctx.moneyness is None:
        return None
    # Same materially-underwater guard as the put variant.
    if ctx.pct_captured < -0.50:
        return None
    if ctx.moneyness < -0.02 and ctx.pct_captured < 0.50 and ctx.dte > 21:
        if ctx.pct_captured >= 0:
            rationale = (
                f"Short call OTM with {ctx.dte}d remaining and "
                f"{ctx.pct_captured*100:.0f}% of premium captured. "
                f"Hold; theta accruing in your favor."
            )
        else:
            rationale = (
                f"Short call OTM with {ctx.dte}d remaining; "
                f"position currently {abs(ctx.pct_captured)*100:.0f}% "
                f"underwater on premium received. "
                f"Theta should accrue if no breach; monitor underlying for "
                f"deterioration."
            )
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="HARVEST_THETA",
            priority="monitor",
            rationale=rationale,
            triggering_signals=[],
            metrics={"pct_captured": ctx.pct_captured, "dte": ctx.dte},
            rule_id="short_call_harvest_theta",
        )
    return None


# ─────────────────────────────────────────────────────────────────────
# 2.3 LONG OPTION rules
# ─────────────────────────────────────────────────────────────────────

def rule_long_option_take_profit(ctx, signals):
    if ctx.instrument_type != "option" or ctx.quantity <= 0:
        return None
    if ctx.pct_pnl is None:
        return None
    if ctx.pct_pnl >= 0.50:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="CLOSE",
            priority="opportunistic",
            rationale=(
                f"Long {ctx.right.lower() if ctx.right else 'option'} has gained "
                f"{ctx.pct_pnl*100:.0f}%. Take profit; theta and gamma decay accelerate "
                f"from here."
            ),
            triggering_signals=[],
            metrics={"pct_pnl": ctx.pct_pnl},
            rule_id="long_option_take_profit",
            estimated_dollar_value=ctx.market_value - (ctx.cost_basis or 0),
        )
    return None


def rule_long_option_decay_killing(ctx, signals):
    """Long option that has lost most of its delta and most of its premium."""
    if ctx.instrument_type != "option" or ctx.quantity <= 0:
        return None
    if (ctx.delta is None or ctx.dte is None or ctx.pct_pnl is None):
        return None
    if (abs(ctx.delta) < 0.10 and ctx.dte < 14 and ctx.pct_pnl < -0.50):
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="CLOSE",
            priority="opportunistic",
            rationale=(
                f"Long option down {ctx.pct_pnl*100:.0f}% with \u03b4 {ctx.delta:.2f} "
                f"and {ctx.dte}d left. Cut the residual; redeploy into a fresh hedge "
                f"if still needed."
            ),
            triggering_signals=[],
            metrics={"pct_pnl": ctx.pct_pnl, "delta": ctx.delta, "dte": ctx.dte},
            rule_id="long_option_decay_killing",
        )
    return None


def rule_long_protection_decaying(ctx, signals):
    """Long put hedge with DTE running out but still relevant delta — roll."""
    if ctx.instrument_type != "option" or ctx.right != "PUT" or ctx.quantity <= 0:
        return None
    if ctx.dte is None or ctx.delta is None:
        return None
    if ctx.dte < 21 and abs(ctx.delta) > 0.10:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="option",
            action="ROLL_OUT",
            priority="opportunistic",
            rationale=(
                f"Long put protection has {ctx.dte}d left and \u03b4 {ctx.delta:.2f} "
                f"— still meaningful coverage. Roll out to extend protection before "
                f"gamma decay accelerates."
            ),
            triggering_signals=[],
            metrics={"dte": ctx.dte, "delta": ctx.delta},
            rule_id="long_protection_decaying",
        )
    return None


# ─────────────────────────────────────────────────────────────────────
# 2.4 LONG STOCK rules
# ─────────────────────────────────────────────────────────────────────

def rule_long_stock_pre_earnings_hedge(ctx, signals):
    if ctx.instrument_type != "equity" or ctx.quantity <= 0:
        return None
    if ctx.has_long_protection:
        return None
    if ctx.days_to_earnings is None:
        return None
    if 1 <= ctx.days_to_earnings <= 14 and ctx.portfolio_pct_of_nav > 0.05:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="equity",
            action="ADD_HEDGE",
            priority="urgent",
            rationale=(
                f"Earnings in {ctx.days_to_earnings}d on a "
                f"{ctx.portfolio_pct_of_nav*100:.1f}% NAV position with no "
                f"protective puts. Initiate collar or buy put spread for downside cap."
            ),
            triggering_signals=["earnings_within_30d"],
            metrics={
                "days_to_earnings": ctx.days_to_earnings,
                "pct_of_nav": ctx.portfolio_pct_of_nav,
            },
            rule_id="long_stock_earnings_hedge",
        )
    return None


def rule_long_stock_overlay_opportunity(ctx, signals):
    """Sell covered calls when IV is rich and trend isn't crashing."""
    if ctx.instrument_type != "equity" or ctx.quantity <= 0:
        return None
    if ctx.has_short_calls:
        return None
    iv_rich = _has_signal(signals, "iv_level", contains="Elevated") or \
              _has_signal(signals, "iv_level", contains="Very high")
    vrp_rich = _has_signal(signals, "vol_risk_premium", contains="Rich")
    iv_pctl_rich = _has_signal(signals, "iv_percentile", contains="Rich")
    trend_bear = _has_signal(signals, "trend_200d", direction="bearish")
    rsi_overbought = _has_signal(signals, "rsi_extreme", direction="bearish")

    if (iv_rich or vrp_rich or iv_pctl_rich) and not trend_bear:
        rationale_pieces = []
        if iv_rich:        rationale_pieces.append("IV elevated")
        if vrp_rich:       rationale_pieces.append("VRP rich")
        if iv_pctl_rich:   rationale_pieces.append("IV at top of 1Y range")
        if rsi_overbought: rationale_pieces.append(
            "RSI overbought (call premium extra valuable)"
        )
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="equity",
            action="ADD_OVERLAY",
            priority="opportunistic",
            rationale=(
                f"Long stock with {' / '.join(rationale_pieces)}. "
                f"Sell ~30-delta calls 30-45d out for premium without giving up "
                f"meaningful upside."
            ),
            triggering_signals=_signal_types(signals,
                ["iv_level", "vol_risk_premium", "iv_percentile", "rsi_extreme"]),
            metrics={"market_value": ctx.market_value},
            rule_id="long_stock_overlay_iv_rich",
        )
    return None


def rule_long_stock_trim_extended(ctx, signals):
    if ctx.instrument_type != "equity" or ctx.quantity <= 0:
        return None
    is_leader = _has_signal(signals, "ytd_performance", contains="Leader")
    is_overbought = _has_signal(signals, "rsi_extreme", direction="bearish")
    if is_leader and is_overbought:
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="equity",
            action="TRIM",
            priority="opportunistic",
            rationale=(
                "Position is YTD leader and shows overbought RSI. Trim 25-33% to "
                "lock in gains; consider redeploying into laggards or keeping dry "
                "powder."
            ),
            triggering_signals=["ytd_performance", "rsi_extreme"],
            metrics={"market_value": ctx.market_value},
            rule_id="long_stock_trim_overbought_leader",
        )
    return None


def rule_long_stock_oversold_value(ctx, signals):
    if ctx.instrument_type != "equity" or ctx.quantity <= 0:
        return None
    is_laggard = _has_signal(signals, "ytd_performance", contains="laggard") or \
                 _has_signal(signals, "ytd_performance", contains="Underperformer")
    is_oversold = _has_signal(signals, "rsi_extreme", direction="bullish")
    iv_cheap = _has_signal(signals, "iv_percentile", contains="Cheap")
    trend_bear = _has_signal(signals, "trend_200d", direction="bearish")
    if is_laggard and is_oversold and not trend_bear:
        if iv_cheap:
            action = "ADD"
            sub = ("IV is cheap \u2014 direct adds avoid premium-selling at "
                   "unrewarding levels")
        else:
            action = "ADD_OVERLAY"
            sub = ("Sell cash-secured puts for entry at lower strike with premium "
                   "income")
        return Recommendation(
            position_id=ctx.bbg_ticker,
            instrument_type="equity",
            action=action,
            priority="opportunistic",
            rationale=(
                f"Laggard with oversold RSI; trend not yet bearish. {sub}."
            ),
            triggering_signals=_signal_types(signals,
                ["ytd_performance", "rsi_extreme", "iv_percentile"]),
            metrics={"market_value": ctx.market_value},
            rule_id=("long_stock_oversold_add" if action == "ADD"
                     else "long_stock_oversold_csp"),
        )
    return None


def rule_long_stock_default_monitor(ctx, signals):
    if ctx.instrument_type != "equity" or ctx.quantity <= 0:
        return None
    return Recommendation(
        position_id=ctx.bbg_ticker,
        instrument_type="equity",
        action="MONITOR",
        priority="monitor",
        rationale="No actionable signals at current levels; watch list.",
        triggering_signals=[],
        metrics={},
        rule_id="long_stock_default_monitor",
    )


def rule_option_default_monitor(ctx, signals):
    """Catch-all so every option position has a recommendation."""
    if ctx.instrument_type != "option":
        return None
    return Recommendation(
        position_id=ctx.bbg_ticker,
        instrument_type="option",
        action="MONITOR",
        priority="monitor",
        rationale="No rule fired; position outside trigger thresholds.",
        triggering_signals=[],
        metrics={},
        rule_id="option_default_monitor",
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

SHORT_PUT_RULES: list[Callable] = [
    rule_short_put_assignment_risk,
    rule_short_put_pre_earnings_hedge,
    rule_short_put_take_profit,
    rule_short_put_near_expiry_otm,
    rule_short_put_harvest_theta,
    rule_option_default_monitor,
]
SHORT_CALL_RULES: list[Callable] = [
    rule_short_call_assignment_risk,
    rule_short_call_take_profit,
    rule_short_call_near_expiry_iv_elevated,
    rule_short_call_harvest_theta,
    rule_option_default_monitor,
]
LONG_OPTION_RULES: list[Callable] = [
    rule_long_option_take_profit,
    rule_long_protection_decaying,
    rule_long_option_decay_killing,
    rule_option_default_monitor,
]
LONG_STOCK_RULES: list[Callable] = [
    rule_long_stock_pre_earnings_hedge,
    rule_long_stock_overlay_opportunity,
    rule_long_stock_trim_extended,
    rule_long_stock_oversold_value,
    rule_long_stock_default_monitor,
]


def _select_rules(ctx: PositionContext) -> list[Callable]:
    if ctx.instrument_type == "equity" and ctx.quantity > 0:
        return LONG_STOCK_RULES
    if ctx.instrument_type != "option":
        return []
    if ctx.right == "PUT" and ctx.quantity < 0:
        return SHORT_PUT_RULES
    if ctx.right == "CALL" and ctx.quantity < 0:
        return SHORT_CALL_RULES
    if ctx.quantity > 0:
        return LONG_OPTION_RULES
    return []


def compute_recommendations(
    contexts: list[PositionContext],
    signals_by_ticker: dict[str, list[Signal]],
) -> list[Recommendation]:
    out: list[Recommendation] = []
    for ctx in contexts:
        signals = signals_by_ticker.get(ctx.underlying_bbg_ticker, [])
        for rule in _select_rules(ctx):
            try:
                rec = rule(ctx, signals)
            except Exception:
                continue
            if rec is not None:
                out.append(rec)
                break  # first match wins
    return out


# ─────────────────────────────────────────────────────────────────────
# RECOMMENDATION_DEFINITIONS — one entry per rule_id any rule returns
# ─────────────────────────────────────────────────────────────────────

RECOMMENDATION_DEFINITIONS: dict[str, RecommendationRuleDefinition] = {
    # SHORT PUT ----------------------------------------------------------
    "short_put_take_profit_50": RecommendationRuleDefinition(
        rule_id="short_put_take_profit_50",
        display_name="Short Put \u2014 50% Take-Profit",
        applies_to="short_put",
        triggers=["pct_captured \u2265 50%", "DTE > 14"],
        action="CLOSE",
        rationale=(
            "Sell-side options desks routinely take 50% of max profit on short "
            "premium positions and redeploy. Holding the last 50% sacrifices "
            "decay efficiency: gamma rises and the remaining theta tail isn't "
            "compensated for tail risk."
        ),
        institutional_source="Industry-standard short-premium management convention.",
    ),
    "short_put_assignment_risk": RecommendationRuleDefinition(
        rule_id="short_put_assignment_risk",
        display_name="Short Put \u2014 Assignment Risk Defense",
        applies_to="short_put",
        triggers=["Moneyness > 2% ITM", "DTE < 21d", "Delta > -0.85 (still rollable)"],
        action="ROLL_OUT_AND_DOWN",
        rationale=(
            "An ITM short put within 3 weeks of expiry has rapidly accelerating "
            "gamma. Rolling out (later expiry) and down (lower strike) defers "
            "assignment, restarts the theta clock, and reduces strike price."
        ),
        institutional_source="Standard defensive roll for near-expiry ITM short premium.",
    ),
    "short_put_roll_otm_near_expiry": RecommendationRuleDefinition(
        rule_id="short_put_roll_otm_near_expiry",
        display_name="Short Put \u2014 Roll OTM Near Expiry",
        applies_to="short_put",
        triggers=["Moneyness < -5% (OTM)", "DTE < 14",
                  "Delta > -0.20", "IV elevated OR VRP rich"],
        action="ROLL_OUT",
        rationale=(
            "OTM short put about to expire worthless and IV is still rich \u2014 "
            "roll to a later expiry at the same strike to capture another cycle "
            "of premium. Avoids the cash drag of waiting for expiry."
        ),
        institutional_source="Premium-seller standard practice in elevated IV regimes.",
    ),
    "short_put_pre_earnings_close": RecommendationRuleDefinition(
        rule_id="short_put_pre_earnings_close",
        display_name="Short Put \u2014 Close Before Earnings",
        applies_to="short_put",
        triggers=["Earnings within 7 days", "Delta < -0.25"],
        action="CLOSE",
        rationale=(
            "An earnings event within a week, on a meaningful-delta short put, "
            "introduces gap-down risk that exceeds the residual premium. Close "
            "or convert to a put spread to cap downside."
        ),
        institutional_source="Pre-earnings risk management discipline.",
    ),
    "short_put_harvest_theta": RecommendationRuleDefinition(
        rule_id="short_put_harvest_theta",
        display_name="Short Put \u2014 Harvest Theta",
        applies_to="short_put",
        triggers=["Moneyness < -2% (OTM)", "pct_captured < 50%", "DTE > 21"],
        action="HARVEST_THETA",
        rationale=(
            "Healthy short put state: out-of-the-money, plenty of time, less than "
            "half the premium captured. Theta is accruing in your favor; let "
            "the position work."
        ),
        institutional_source="Premium-seller patience discipline.",
    ),

    # SHORT CALL ---------------------------------------------------------
    "short_call_take_profit_50": RecommendationRuleDefinition(
        rule_id="short_call_take_profit_50",
        display_name="Short Call \u2014 50% Take-Profit",
        applies_to="short_call",
        triggers=["pct_captured \u2265 50%", "DTE > 14"],
        action="CLOSE",
        rationale=(
            "Mirror of the short-put 50% take-profit convention; same rationale "
            "applies on the call side."
        ),
        institutional_source="Industry-standard short-premium management convention.",
    ),
    "short_call_assignment_risk": RecommendationRuleDefinition(
        rule_id="short_call_assignment_risk",
        display_name="Short Call \u2014 Assignment Risk (Covered)",
        applies_to="short_call",
        triggers=["Moneyness > 2% ITM", "DTE < 21d",
                  "Delta > 0.60", "Has long stock on same name"],
        action="ROLL_UP_AND_OUT",
        rationale=(
            "Covered call gone ITM near expiry threatens to have the long stock "
            "called away. Roll up (higher strike) and out (later expiry) to keep "
            "the overlay alive without surrendering the underlying."
        ),
        institutional_source="Covered-call overlay management discipline.",
    ),
    "short_call_roll_otm_near_expiry": RecommendationRuleDefinition(
        rule_id="short_call_roll_otm_near_expiry",
        display_name="Short Call \u2014 Roll OTM Near Expiry",
        applies_to="short_call",
        triggers=["Moneyness < -5% (OTM)", "DTE < 14",
                  "Delta < 0.20", "IV elevated OR VRP rich"],
        action="ROLL_OUT",
        rationale=(
            "Same logic as the short-put OTM roll: harvest a fresh cycle of "
            "premium when IV is still attractive."
        ),
        institutional_source="Premium-seller standard practice in elevated IV regimes.",
    ),
    "short_call_harvest_theta": RecommendationRuleDefinition(
        rule_id="short_call_harvest_theta",
        display_name="Short Call \u2014 Harvest Theta",
        applies_to="short_call",
        triggers=["Moneyness < -2% (OTM)", "pct_captured < 50%", "DTE > 21"],
        action="HARVEST_THETA",
        rationale="Mirror of the short-put harvest-theta state.",
        institutional_source="Premium-seller patience discipline.",
    ),

    # LONG OPTION --------------------------------------------------------
    "long_option_take_profit": RecommendationRuleDefinition(
        rule_id="long_option_take_profit",
        display_name="Long Option \u2014 50%+ Take-Profit",
        applies_to="long_option",
        triggers=["pct_pnl \u2265 50%"],
        action="CLOSE",
        rationale=(
            "Long debit positions face accelerating theta and gamma decay once "
            "in the money. Take 50%+ profits and redeploy into the next conviction "
            "trade rather than ride the curve down."
        ),
        institutional_source="Long-premium take-profit convention.",
    ),
    "long_option_decay_killing": RecommendationRuleDefinition(
        rule_id="long_option_decay_killing",
        display_name="Long Option \u2014 Cut Decayed Loser",
        applies_to="long_option",
        triggers=["|delta| < 0.10", "DTE < 14", "pct_pnl < -50%"],
        action="CLOSE",
        rationale=(
            "When delta has collapsed below 10% with little time left, the option "
            "is functionally a lottery ticket. Cut, free the residual capital, and "
            "re-enter at a strike that still carries delta if the thesis still holds."
        ),
        institutional_source="Long-premium loss-cutting discipline.",
    ),
    "long_protection_decaying": RecommendationRuleDefinition(
        rule_id="long_protection_decaying",
        display_name="Long Put \u2014 Roll Hedge Forward",
        applies_to="long_option",
        triggers=["right=PUT, long", "DTE < 21d", "|delta| > 0.10"],
        action="ROLL_OUT",
        rationale=(
            "Protective put that still has meaningful delta but limited tenor. "
            "Roll forward before gamma decay accelerates into expiry to keep the "
            "downside cap intact."
        ),
        institutional_source="Hedge-management discipline.",
    ),

    # LONG STOCK ---------------------------------------------------------
    "long_stock_earnings_hedge": RecommendationRuleDefinition(
        rule_id="long_stock_earnings_hedge",
        display_name="Long Stock \u2014 Pre-Earnings Hedge",
        applies_to="equity",
        triggers=["Days to earnings 1\u201314",
                  "Position > 5% NAV",
                  "No long protection in place"],
        action="ADD_HEDGE",
        rationale=(
            "Concentrated long-stock exposure into earnings without protection is "
            "a known unforced error. Even a wide collar materially reduces gap-down "
            "risk for a small premium outlay."
        ),
        institutional_source="Pre-earnings risk management discipline.",
    ),
    "long_stock_overlay_iv_rich": RecommendationRuleDefinition(
        rule_id="long_stock_overlay_iv_rich",
        display_name="Long Stock \u2014 Overlay (IV Rich)",
        applies_to="equity",
        triggers=["No existing short calls",
                  "IV elevated OR VRP rich OR IV percentile rich",
                  "Trend not bearish"],
        action="ADD_OVERLAY",
        rationale=(
            "Long stock + rich IV without an overlay is leaving income on the "
            "table. A 30-delta overlay 30-45d out generates premium at strikes "
            "that retain meaningful upside participation."
        ),
        institutional_source="Vol Screener affinity matrix: rich vol + bullish "
                             "trend favors covered call.",
    ),
    "long_stock_trim_overbought_leader": RecommendationRuleDefinition(
        rule_id="long_stock_trim_overbought_leader",
        display_name="Long Stock \u2014 Trim Extended Leader",
        applies_to="equity",
        triggers=["YTD Leader (>+20%)", "RSI overbought (\u2265 70)"],
        action="TRIM",
        rationale=(
            "A YTD leader showing overbought RSI is statistically due for mean "
            "reversion. Locking in 25\u201333% of the gain hedges that risk and "
            "frees capital for laggards."
        ),
        institutional_source="Mean-reversion + position-management discipline.",
    ),
    "long_stock_oversold_csp": RecommendationRuleDefinition(
        rule_id="long_stock_oversold_csp",
        display_name="Long Stock \u2014 Oversold (CSP Setup)",
        applies_to="equity",
        triggers=["YTD laggard or underperformer",
                  "RSI oversold (\u2264 30)",
                  "Trend not bearish",
                  "IV not cheap"],
        action="ADD_OVERLAY",
        rationale=(
            "Laggard at oversold RSI with elevated IV \u2014 best expressed via "
            "cash-secured put (sell premium AT or below current price) so you're "
            "paid to wait for a lower entry."
        ),
        institutional_source="CSP entry technique for accumulating high-conviction "
                             "names at discount.",
    ),
    "long_stock_oversold_add": RecommendationRuleDefinition(
        rule_id="long_stock_oversold_add",
        display_name="Long Stock \u2014 Oversold (Direct Add)",
        applies_to="equity",
        triggers=["YTD laggard or underperformer",
                  "RSI oversold (\u2264 30)",
                  "Trend not bearish",
                  "IV cheap (\u2264 20th percentile)"],
        action="ADD",
        rationale=(
            "Same setup as the CSP version, but IV is too cheap to be paid for "
            "selling premium. Direct add avoids selling cheap optionality."
        ),
        institutional_source="Mean-reversion add discipline; CSP only when IV "
                             "compensates.",
    ),
    "long_stock_default_monitor": RecommendationRuleDefinition(
        rule_id="long_stock_default_monitor",
        display_name="Long Stock \u2014 Monitor",
        applies_to="equity",
        triggers=["No other equity rule fired"],
        action="MONITOR",
        rationale=(
            "No actionable signals at current levels. Watch the trend, IV, and "
            "earnings calendar; act when a trigger fires."
        ),
        institutional_source="Default state.",
    ),

    # GENERIC ------------------------------------------------------------
    "option_default_monitor": RecommendationRuleDefinition(
        rule_id="option_default_monitor",
        display_name="Option \u2014 Monitor",
        applies_to="option",
        triggers=["No other option rule fired"],
        action="MONITOR",
        rationale=(
            "Position is outside all trigger thresholds. Watch DTE, moneyness, "
            "and IV; act when a rule fires."
        ),
        institutional_source="Default state.",
    ),
}
