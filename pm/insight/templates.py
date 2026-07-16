"""Pattern label / rationale templates + variable-resolution helper.

All variable substitution flows through `resolve_variables`, which renders
``None`` values uniformly as ``"—"`` (em-dash) and applies Python format
specifiers from the registry.

Variables that would require live option pricing (roll credits,
breakevens, suggested strikes/tenors) are rendered as
``INDICATIVE_PLACEHOLDER`` in V1; the trace records that they were
placeholder-substituted.
"""
from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any, Callable, Optional


INDICATIVE_PLACEHOLDER = "<indicative — terminal quote required>"
EM_DASH = "—"


# ---------------------------------------------------------------------------
# Context object passed to every registry callable
# ---------------------------------------------------------------------------

@dataclass
class TemplateContext:
    """Bundle of references a registry callable may read from.

    Detectors construct this object and pass it to ``resolve_variables``
    along with the template string and any pattern-specific ``extras``.
    """
    position: Any                          # pm.ingest.position_builder.Position (avoid cycle)
    account_state: Any                     # pm.store.portfolio_state.AccountState
    signals: dict                          # SignalDict for position.underlying
    config: Any                            # PatternConfig (avoid cycle)


# ---------------------------------------------------------------------------
# Helpers used by registry callables
# ---------------------------------------------------------------------------

def _signal_value(ctx: TemplateContext, signal_id: str) -> Any:
    """Read SignalValue.value from the bundle, or None if absent / stale."""
    sv = ctx.signals.get(signal_id)
    if sv is None:
        return None
    if getattr(sv, "stale", False):
        return None
    return getattr(sv, "value", None)


def _signal_trace_input(ctx: TemplateContext, signal_id: str, input_name: str) -> Any:
    """Pull a specific raw input from a signal's trace.inputs dict."""
    sv = ctx.signals.get(signal_id)
    if sv is None:
        return None
    inputs = getattr(sv, "trace", {}).get("inputs", {})
    entry = inputs.get(input_name)
    if entry is None:
        return None
    return entry.get("value")


def _safe_abs(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return abs(float(v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Template variable registry
# ---------------------------------------------------------------------------
#
# Each entry: variable_name -> callable(TemplateContext) -> raw value (or None).
# Detectors may override / extend via the `extras` dict on resolve_variables.

TEMPLATE_VARIABLE_REGISTRY: dict[str, Callable[[TemplateContext], Any]] = {
    # ----- Position identity --------------------------------------------------
    "symbol": lambda ctx: ctx.position.symbol if ctx.position else None,
    "right": lambda ctx: (
        ctx.position.right.title() if (ctx.position and ctx.position.right) else None
    ),
    "strike": lambda ctx: ctx.position.strike if ctx.position else None,
    "expiry": lambda ctx: ctx.position.expiry if ctx.position else None,
    "quantity": lambda ctx: ctx.position.quantity if ctx.position else None,

    # ----- Position economics -------------------------------------------------
    "cost_basis_abs": lambda ctx: _safe_abs(getattr(ctx.position, "cost_basis", None)),
    "market_value_abs": lambda ctx: _safe_abs(getattr(ctx.position, "market_value", None)),
    "pnl_dollars": lambda ctx: getattr(ctx.position, "unrealized_pnl", None),
    "notional": lambda ctx: _safe_abs(getattr(ctx.position, "cost_basis", None)),

    # ----- Signal-derived (per source column) -----------------------
    "dte": lambda ctx: _signal_value(ctx, "option_dte"),
    "captured_pct": lambda ctx: _signal_value(ctx, "option_captured_pct"),
    "pnl_pct": lambda ctx: _signal_value(ctx, "position_unrealized_pnl_pct"),
    "nav_pct": lambda ctx: _signal_value(ctx, "position_size_pct_of_nav"),
    "spot": lambda ctx: _signal_trace_input(ctx, "spot_vs_50d_ma", "PX_LAST"),
    "moneyness": lambda ctx: _signal_value(ctx, "option_moneyness"),
    "itm_otm": lambda ctx: (
        "ITM" if (_signal_value(ctx, "option_moneyness") or 0) > 0 else "OTM"
    ),
    "iv_3m": lambda ctx: _signal_value(ctx, "iv_3m_atm"),
    "iv_6m": lambda ctx: _signal_value(ctx, "iv_6m_atm"),
    "iv_pctl": lambda ctx: _signal_value(ctx, "iv_3m_percentile_1y"),
    "iv_term_structure": lambda ctx: _signal_value(ctx, "iv_term_structure"),
    "ma_200d": lambda ctx: _signal_trace_input(ctx, "spot_vs_200d_ma", "MOV_AVG_200D"),
    "ma_stack_regime": lambda ctx: _signal_value(ctx, "ma_stack_regime"),
    "rsi_14d": lambda ctx: (
        _signal_value(ctx, "rsi_14d_regime") or {}
    ).get("rsi") if isinstance(_signal_value(ctx, "rsi_14d_regime"), dict) else None,
    "return_5d": lambda ctx: (
        _signal_value(ctx, "return_horizons") or {}
    ).get("return_5d") if isinstance(_signal_value(ctx, "return_horizons"), dict) else None,
    "earnings_date": lambda ctx: _signal_trace_input(ctx, "days_to_earnings", "EXPECTED_REPORT_DT"),
    "days_to_earnings": lambda ctx: _signal_value(ctx, "days_to_earnings"),
    "earnings_implied_move": lambda ctx: _signal_value(ctx, "earnings_implied_move"),
    "ex_div_date": lambda ctx: _signal_trace_input(ctx, "days_to_ex_div", "ex_div_date"),
    "days_to_exdiv": lambda ctx: _signal_value(ctx, "days_to_ex_div"),
    "analyst_rating": lambda ctx: (
        _signal_value(ctx, "analyst_rating_and_target") or {}
    ).get("rating") if isinstance(_signal_value(ctx, "analyst_rating_and_target"), dict) else None,
    "analyst_target": lambda ctx: (
        _signal_value(ctx, "analyst_rating_and_target") or {}
    ).get("target") if isinstance(_signal_value(ctx, "analyst_rating_and_target"), dict) else None,
    "analyst_upside": lambda ctx: (
        _signal_value(ctx, "analyst_rating_and_target") or {}
    ).get("upside_pct") if isinstance(_signal_value(ctx, "analyst_rating_and_target"), dict) else None,
    "analyst_note_date": lambda ctx: (
        _signal_value(ctx, "analyst_note_recent") or {}
    ).get("note_date") if isinstance(_signal_value(ctx, "analyst_note_recent"), dict) else None,
    "days_since_note": lambda ctx: (
        _signal_value(ctx, "analyst_note_recent") or {}
    ).get("days_since") if isinstance(_signal_value(ctx, "analyst_note_recent"), dict) else None,

    # ----- A8 / P15 (vol-adjusted move) --------------------------------------
    "vol_units": lambda ctx: _signal_value(ctx, "vol_adjusted_move"),
    "return_pct": lambda ctx: (
        _signal_value(ctx, "return_horizons") or {}
    ).get("return_1d") if isinstance(_signal_value(ctx, "return_horizons"), dict) else None,
    "abs_return_pct": lambda ctx: _abs_return_1d(ctx),
    "direction_word": lambda ctx: _direction_word(ctx),
    # B1 rv_30d is stored as a percent number (22.5); divide so a `:.1%`
    # format spec renders "22.5%" rather than "2250.0%".
    "rv_30d": lambda ctx: (
        _signal_value(ctx, "rv_30d") / 100.0
        if _signal_value(ctx, "rv_30d") is not None else None
    ),
    "position_nav_pct": lambda ctx: _signal_value(ctx, "position_size_pct_of_nav"),
    "position_value_signed": lambda ctx: _position_value_signed(ctx),
    # P15 context sentences are detector-supplied via `extras`; default empty
    # so a missing extra renders as "" rather than the em-dash.
    "earnings_context_sentence": lambda ctx: "",
    "analyst_note_context_sentence": lambda ctx: "",

    # ----- Position trade history --------------------------------------------
    "open_date": lambda ctx: getattr(ctx.position, "open_date", None),
    "days_held": lambda ctx: getattr(ctx.position, "days_held", None),
    "last_trade_date": lambda ctx: getattr(ctx.position, "last_trade_date", None),
    "last_trade_action": lambda ctx: getattr(ctx.position, "last_trade_action", None),

    # ----- Account-scoped ----------------------------------------------------
    "account": lambda ctx: ctx.account_state.account if ctx.account_state else None,
    "nav": lambda ctx: ctx.account_state.nav if ctx.account_state else None,
    "cash_total": lambda ctx: _cash_total(ctx),
    "cash_pct": lambda ctx: (
        _cash_total(ctx) / ctx.account_state.nav
        if ctx.account_state and ctx.account_state.nav else None
    ),
    "n_legs": lambda ctx: _count_option_legs(ctx),
}


def _cash_total(ctx: TemplateContext) -> Optional[float]:
    if ctx.account_state is None:
        return None
    return sum(
        abs(p.market_value)
        for p in ctx.account_state.positions
        if p.asset_class in ("cash", "fund_etf") and p.market_value is not None
    )


def _count_option_legs(ctx: TemplateContext) -> int:
    if ctx.account_state is None or ctx.position is None:
        return 0
    return sum(
        1 for p in ctx.account_state.positions
        if p.asset_class == "option"
        and p.underlying_symbol == ctx.position.underlying_symbol
    )


def _abs_return_1d(ctx: TemplateContext) -> Optional[float]:
    rh = _signal_value(ctx, "return_horizons")
    r1d = rh.get("return_1d") if isinstance(rh, dict) else None
    return abs(r1d) if r1d is not None else None


def _direction_word(ctx: TemplateContext) -> Optional[str]:
    rh = _signal_value(ctx, "return_horizons")
    r1d = rh.get("return_1d") if isinstance(rh, dict) else None
    if r1d is None:
        return None
    return "rallied" if r1d > 0 else "dropped"


def _position_value_signed(ctx: TemplateContext) -> Optional[str]:
    mv = getattr(ctx.position, "market_value", None) if ctx.position else None
    if mv is None:
        return None
    return f"{'-' if mv < 0 else ''}${abs(mv):,.0f}"


# ---------------------------------------------------------------------------
# Resolution + rendering
# ---------------------------------------------------------------------------

class _ToleratesNoneFormatter(Formatter):
    """Renders None as EM_DASH regardless of the format_spec."""

    def format_field(self, value, format_spec):
        if value is None:
            return EM_DASH
        try:
            return super().format_field(value, format_spec)
        except (TypeError, ValueError):
            return str(value)


_FORMATTER = _ToleratesNoneFormatter()


def _format_date(v) -> str:
    if v is None:
        return EM_DASH
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def resolve_variables(
    template: str,
    ctx: TemplateContext,
    extras: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Render ``template`` against the registry + per-detector extras.

    Returns ``(rendered_string, resolved_values)`` where
    ``resolved_values`` is the dict actually substituted (suitable for
    inclusion in a Fire's trace.inputs.template_variables).

    Variables in ``extras`` override registry entries (detectors use this
    to pass computed values like roll structures or indicative placeholders).
    """
    extras = extras or {}

    field_names = set()
    for _literal, field_name, _format_spec, _conversion in _FORMATTER.parse(template):
        if field_name:
            field_names.add(field_name.split(".", 1)[0].split("[", 1)[0])

    resolved: dict[str, Any] = {}
    for name in field_names:
        if name in extras:
            resolved[name] = extras[name]
        elif name in TEMPLATE_VARIABLE_REGISTRY:
            try:
                resolved[name] = TEMPLATE_VARIABLE_REGISTRY[name](ctx)
            except Exception:
                resolved[name] = None
        else:
            resolved[name] = None

    # Dates render as ISO strings (Python's default repr is too noisy).
    render_kwargs = {}
    for name, value in resolved.items():
        if hasattr(value, "isoformat") and not isinstance(value, (int, float)):
            render_kwargs[name] = _format_date(value)
        else:
            render_kwargs[name] = value

    rendered = _FORMATTER.vformat(template, (), render_kwargs)
    return rendered, resolved


# ===========================================================================
# Pattern templates.
# ===========================================================================

# ----- P1 -----
P1_LABEL_TEMPLATE = (
    "{symbol} {right} ${strike} — {captured_pct:.0%} captured, ready to close"
)
P1_RATIONALE_TEMPLATE = (
    "{symbol} short {right} ${strike} expiring {expiry}: {captured_pct:.0%} of "
    "premium captured (cost basis {cost_basis_abs}, current MV {market_value_abs}). "
    "With {dte} days to expiry the remaining decay is small. "
    "**Close-and-redeploy candidate**: closing locks ${pnl_dollars} of "
    "premium-margin for the account; capacity could be re-deployed into a fresh "
    "write at a higher strike or further-dated tenor."
)

# ----- P2 -----
P2_LABEL_TEMPLATE = (
    "{symbol} {right} ${strike} — {captured_pct:.0%} captured + IV pctl {iv_pctl:.0f}"
)
P2_RATIONALE_TEMPLATE = (
    "{symbol} short {right} ${strike}: {captured_pct:.0%} captured AND 3M IV is "
    "in the {iv_pctl:.0f}th percentile of its 1Y range. **Close-and-rewrite "
    "setup**: closing harvests ${pnl_dollars} of premium; rewriting at a similar "
    "strike further-dated re-establishes premium at currently rich IV. Specific "
    "re-write candidate: same strike, {next_tenor} expiry — indicative IV "
    "{iv_3m:.1f}%."
)

# ----- P3 -----
P3_LABEL_TEMPLATE = (
    "{symbol} {right} ${strike} — {captured_pct:.0%} + {direction_word} adverse"
)
P3_RATIONALE_TEMPLATE_PUT = (
    "{symbol} short put ${strike}: {captured_pct:.0%} captured, BUT {symbol} "
    "broke its 200d MA on {ma_200d_cross_date} (spot ${spot} vs 200d MA "
    "${ma_200d}, {pct_below_200d:.1%} below) AND has fallen {return_5d:.1%} in "
    "the last 5 sessions. The profit cushion is shrinking while the catalyst "
    "is in motion. **Close before further weakness erodes the captured "
    "premium**. Closing locks ${pnl_dollars}."
)
P3_RATIONALE_TEMPLATE_CALL = (
    "{symbol} short call ${strike}: {captured_pct:.0%} captured, BUT {symbol} "
    "is extending above its 200d MA (spot ${spot} vs 200d MA ${ma_200d}, "
    "{pct_above_200d:.1%} above) AND has rallied {return_5d:.1%} in the last 5 "
    "sessions while the call is closing toward ITM. **Close or roll up-and-out "
    "before continued strength erodes the captured premium**. Closing locks "
    "${pnl_dollars}."
)

# ----- P4 -----
P4_LABEL_TEMPLATE = (
    "{symbol} {right} ${strike} — {captured_pct:.0%} + analyst note {analyst_note_date}"
)
P4_RATIONALE_TEMPLATE = (
    "{symbol} short {right} ${strike}: {captured_pct:.0%} captured AND an analyst "
    "published a note on {analyst_note_date} ({days_since_note} BD ago). **Review "
    "the position in context of the updated view**: if the new view argues "
    "against this strike's risk-reward, close; if it supports it, hold or "
    "rewrite further out."
)

# ----- P5 -----
P5_LABEL_TEMPLATE = (
    "{symbol} {right} ${strike} — roll-due ({dte} DTE, {captured_pct:.0%})"
)
P5_RATIONALE_TEMPLATE = (
    "{symbol} short {right} ${strike} expires {expiry} ({dte} DTE) with "
    "{captured_pct:.0%} captured. **Three roll structures:**\n"
    "• **Roll out, same strike** — capture additional theta at the same risk "
    "profile. Indicative {next_tenor} ${strike} {right} credit ~${roll_out_credit}.\n"
    "• **Roll {direction_words}-and-out** — widen the strike for more cushion. "
    "Indicative {next_tenor} ${suggested_strike} {right} credit ~${roll_wider_credit}.\n"
    "• **Close and rewrite** — book the captured premium, re-establish a fresh "
    "structure on current IV (currently {iv_3m:.1f}%, {iv_pctl:.0f}th pctl).\n"
    "\n"
    "Recommended based on current vol regime: {best_recommendation}."
)

# ----- P6 -----
P6_LABEL_TEMPLATE = (
    "{symbol} {right} ${strike} — STRESS ({pnl_pct:.0%}, {dte} DTE)"
)
P6_RATIONALE_TEMPLATE = (
    "{symbol} short {right} ${strike}: P&L {pnl_pct:.0%} (${pnl_dollars}), "
    "{dte} DTE, spot ${spot} vs strike ${strike} ({moneyness:.1%} {itm_otm}). "
    "**Three roads:**\n"
    "• **Take the loss now** — close at current MV ${market_value_abs}; remove "
    "the position from the book and any tail risk.\n"
    "• **Roll for credit (if feasible)** — at current IV {iv_3m:.1f}%, roll "
    "candidate {suggested_strike}/{suggested_expiry} for indicative net credit "
    "(verify on terminal — if cannot achieve credit, this road is closed).\n"
    "• **Accept assignment, convert** — if short put, prepare to take stock at "
    "${strike} and convert to a covered-call program. Capital at risk: "
    "${capital_at_risk}."
)
P6_RATIONALE_TEMPLATE_NO_IV = (
    "{symbol} short {right} ${strike}: P&L {pnl_pct:.0%} (${pnl_dollars}), "
    "{dte} DTE, spot ${spot} vs strike ${strike} ({moneyness:.1%} {itm_otm}). "
    "**Two roads (IV unavailable — roll road omitted):**\n"
    "• **Take the loss now** — close at current MV ${market_value_abs}; remove "
    "the position from the book and any tail risk.\n"
    "• **Accept assignment, convert** — if short put, prepare to take stock at "
    "${strike} and convert to a covered-call program. Capital at risk: "
    "${capital_at_risk}."
)

# ----- P7 -----
# Two rationale variants by dividend source (detect_p7 selects): the live one
# names the Bloomberg projection (defensible, worded as projected — not declared);
# the heuristic one flags the dividend as estimated from yield.
P7_LABEL_TEMPLATE = (
    "{symbol} short Call ${strike} — ex-div trap ({days_to_exdiv}d, div ${dividend_amount})"
)
P7_RATIONALE_TEMPLATE = (
    "{symbol} short call ${strike}: ITM by {moneyness:.1%} with ex-dividend in "
    "{days_to_exdiv} business days ({ex_div_date}). Estimated extrinsic value "
    "${extrinsic} vs the Bloomberg projected dividend of ${dividend_amount} on "
    "{ex_div_date} — **early-exercise economics favor the holder of the long "
    "call**. Action: close or roll up-and-out before ex-div to avoid forced "
    "assignment."
)
P7_RATIONALE_TEMPLATE_HEURISTIC = (
    "{symbol} short call ${strike}: ITM by {moneyness:.1%} with ex-dividend in "
    "{days_to_exdiv} business days ({ex_div_date}). Estimated extrinsic value "
    "${extrinsic} vs an estimated dividend of ${dividend_amount} (from {symbol}'s "
    "yield) — **early-exercise economics favor the holder of the long call**. "
    "Action: close or roll up-and-out before ex-div to avoid forced assignment."
)

# ----- P8 -----
P8_LABEL_TEMPLATE = (
    "{symbol} — roll asymmetry ({recent_leg_strike} rolled {days_ago}d ago, "
    "{residual_leg_strike} not touched, {residual_pnl_pct:.0%})"
)
P8_RATIONALE_TEMPLATE = (
    "{symbol} multi-leg structure: {n_legs} short legs in this account. "
    "**{recent_leg_strike} {recent_leg_right}** ({recent_leg_dte} DTE) was "
    "traded {days_ago} business days ago ({recent_action} on "
    "{recent_trade_date}). **{residual_leg_strike} {residual_leg_right}** "
    "({residual_leg_dte} DTE) sitting at {residual_pnl_pct:.0%} has NOT been "
    "touched in this period. **Question to raise**: was the residual leg "
    "intentionally retained, or was it missed in the recent roll? If missed: "
    "same-day roll candidate to align the structure."
)

# ----- P9 -----
P9_LABEL_TEMPLATE = (
    "{symbol} — fresh position ({days_held}d old, {nav_pct:.0%} NAV)"
)
# days_held is a CALENDAR-day count (position_builder), so the copy says
# "days", not "business days" — the text must match the arithmetic.
P9_RATIONALE_TEMPLATE_OPTION = (
    "{symbol} {right} ${strike} expiring {expiry}: opened {open_date} "
    "({days_held} days ago), notional ${notional}, {nav_pct:.0%} of "
    "account NAV. Spot ${spot} vs strike ${strike} ({moneyness:.1%} {itm_otm}, "
    "breakeven ${breakeven}). **Watch-list item** — significant fresh "
    "exposure; ensure client is comfortable with current positioning."
)
P9_RATIONALE_TEMPLATE_EQUITY = (
    "{symbol} equity: opened {open_date} ({days_held} days ago), "
    "notional ${notional}, {nav_pct:.0%} of account NAV. Spot ${spot}. "
    "**Watch-list item** — significant fresh exposure; ensure client is "
    "comfortable with current positioning."
)

# ----- P10 -----
P10_LABEL_TEMPLATE = "{symbol} {right} ${strike} — winner ({pnl_pct:.0%})"
P10_RATIONALE_TEMPLATE = (
    "{symbol} long {right} ${strike} ({dte} DTE): {pnl_pct:.0%} "
    "(${pnl_dollars}) on cost basis ${cost_basis_abs}. Spot ${spot} vs strike "
    "${strike} ({moneyness:.1%} ITM). **Partial monetization candidate**: "
    "selling 25% locks ${quarter_lock} of profit; remaining 75% preserves the "
    "thesis. Alternative: roll-up-and-out for premium credit if conviction in "
    "further upside is intact."
)

# ----- P11 -----
P11_LABEL_TEMPLATE = "{account} — idle ({days_since_trade}d), {cash_pct:.0%} cash"
# days_since_trade is a CALENDAR-day count (detect_p11) — copy matches.
P11_RATIONALE_TEMPLATE = (
    "{account}: last trade {last_trade_date} ({days_since_trade} days ago). "
    "Cash + sweep positions = ${cash_total} ({cash_pct:.0%} of "
    "NAV). **Redeploy candidates** drawn from existing equity holdings with "
    "vol-rich setups (see P13 fires for this account)."
)

# ----- P12 -----
P12_LABEL_TEMPLATE = "{symbol} — concentrated ({nav_pct:.0%} of NAV)"
P12_RATIONALE_TEMPLATE = (
    "{account}: {symbol} is {nav_pct:.0%} of NAV (${position_value} of ${nav}). "
    "**Overlay candidate**: covered-call sleeve on the equity sleeve writes "
    "monthly premium without giving up material upside; specific candidate "
    "{suggested_strike} call {suggested_tenor} based on current IV "
    "({iv_3m:.1f}%, {iv_pctl:.0f}th pctl). Alternative: collar — same call "
    "sale plus protective put at {protective_strike}."
)
P12_RATIONALE_TEMPLATE_NO_IV = (
    "{account}: {symbol} is {nav_pct:.0%} of NAV (${position_value} of ${nav}). "
    "**Overlay candidate**: covered-call sleeve on the equity sleeve writes "
    "monthly premium without giving up material upside. Specific structure "
    "indicative — fetch live IV quote for sizing."
)

# ----- P13 -----
P13_LABEL_TEMPLATE = "{symbol} — vol-rich CC setup (IV {iv_pctl:.0f}p, {trend_regime})"
P13_RATIONALE_TEMPLATE = (
    "{symbol} long equity ({nav_pct:.0%} of NAV): IV 3M at {iv_pctl:.0f}th "
    "percentile of 1Y range AND trend regime {ma_stack_regime}. **Vol-rich "
    "covered-call candidate**: at current IV {iv_3m:.1f}%, writing "
    "{suggested_strike} {suggested_tenor} calls captures indicative "
    "${premium} ({premium_bps_nav} bps on NAV)."
)

# ----- P14 -----
P14_LABEL_TEMPLATE = "{symbol} — earnings in {days_to_earnings}d, IV pctl {iv_pctl:.0f}"
P14_RATIONALE_TEMPLATE = (
    "{symbol}: reports {earnings_date} ({days_to_earnings} business days). "
    "Implied move {earnings_implied_move:.1%}. IV 3M at {iv_pctl:.0f}th pctl, "
    "term structure {iv_term_structure:+.1f} (front {iv_3m:.1f} vs back "
    "{iv_6m:.1f}). **Pre-earnings structures**: directional call/put spread, "
    "calendar spread (sell front-month, buy back-month), or collar on existing "
    "exposure depending on directional view."
)

# ----- P15 -----
P15_LABEL_TEMPLATE = (
    "{symbol} — {direction_word} {abs_return_pct:.1%} today ({vol_units:.1f}σ)"
)
P15_RATIONALE_TEMPLATE = (
    "{symbol} moved {return_pct:+.1%} today ({vol_units:.1f}σ on 30d realized "
    "vol of {rv_30d:.1%}). Position size {position_nav_pct:.1%} of NAV "
    "({position_value_signed}). {earnings_context_sentence}"
    "{analyst_note_context_sentence}**Worth flagging to the client** — material "
    "move on a held position is a natural reason for a call."
)
P15_RATIONALE_TEMPLATE_NO_RV = (
    "{symbol} moved {return_pct:+.1%} today ({vol_units:.1f}σ, realized vol "
    "unavailable). Position size {position_nav_pct:.1%} of NAV "
    "({position_value_signed}). {earnings_context_sentence}"
    "{analyst_note_context_sentence}**Worth flagging to the client** — material "
    "move on a held position is a natural reason for a call."
)

P21_LABEL_TEMPLATE = (
    "{symbol} — research note published {note_recency}"
)
P21_RATIONALE_TEMPLATE = (
    "An analyst note on {symbol} is dated {analyst_note_date} ({note_recency}); "
    "the account holds the name. Current view: {analyst_rating} · target "
    "${analyst_target:,.2f} · upside {analyst_upside:.0%}. **Review the holding in "
    "light of the note** — a fresh view is a natural client touchpoint. "
    "(Whether the rating or target MOVED vs the prior note is not yet tracked; "
    "this alert fires on recency, not direction.)"
)
