"""Pitch synthesizer.

Groups individual Recommendations into 3-4 themed Action Items a salesperson
can use directly in a client conversation. Each PitchTheme has a name,
client-facing headline, supporting recommendations, and where computable, an
estimated dollar value.
"""
from __future__ import annotations

from dataclasses import dataclass

from pm.core.recommender import Recommendation


@dataclass
class PitchTheme:
    theme_name: str
    headline: str
    recommendations: list[Recommendation]
    summary_metric: str


THEMES = [
    "Yield Enhancement",
    "Risk Mitigation",
    "Dead-Weight Purge",
    "Tactical Opportunity",
]


# Rule_ids that count as "purge" themes when accompanying a CLOSE action.
_PURGE_RULE_IDS = frozenset({
    "short_put_take_profit_50",
    "short_call_take_profit_50",
    "long_option_take_profit",
    "long_option_decay_killing",
})


def _theme_for(rec: Recommendation) -> str | None:
    """Map a single recommendation to a pitch theme; None means 'don't surface'."""
    if rec.action in ("ADD_OVERLAY", "HARVEST_THETA"):
        return "Yield Enhancement"
    if rec.action in ("ADD_HEDGE", "ROLL_OUT_AND_DOWN", "ROLL_UP_AND_OUT"):
        return "Risk Mitigation"
    if rec.action == "CLOSE" and rec.rule_id in _PURGE_RULE_IDS:
        return "Dead-Weight Purge"
    if rec.action in ("ADD", "TRIM"):
        return "Tactical Opportunity"
    # MONITOR / non-purge CLOSE / ROLL_OUT (non-defensive) / etc. — not surfaced
    return None


def synthesize_pitch(recommendations: list[Recommendation]) -> list[PitchTheme]:
    """Bucket recommendations into the 4 themes. Return only themes with \u22651 rec."""
    buckets: dict[str, list[Recommendation]] = {t: [] for t in THEMES}
    for r in recommendations:
        t = _theme_for(r)
        if t is not None:
            buckets[t].append(r)

    out: list[PitchTheme] = []
    for theme_name in THEMES:
        recs = buckets[theme_name]
        if not recs:
            continue
        out.append(PitchTheme(
            theme_name=theme_name,
            headline=_build_headline(theme_name, recs),
            recommendations=recs,
            summary_metric=_summary_metric(theme_name, recs),
        ))
    return out


def _build_headline(theme_name: str, recs: list[Recommendation]) -> str:
    n = len(recs)
    # Extract the underlying symbol from each position_id (first whitespace token).
    names = sorted({r.position_id.split()[0] for r in recs})
    sample = ", ".join(names[:3]) + ("\u2026" if len(names) > 3 else "")
    headlines = {
        "Yield Enhancement": (
            f"{n} positions identified for premium-collection overlays "
            f"on {sample} \u2014 generate income on stable holdings."
        ),
        "Risk Mitigation": (
            f"{n} positions need defensive action: {sample}. Hedge "
            f"earnings exposure, roll defensive shorts, cap downside."
        ),
        "Dead-Weight Purge": (
            f"{n} positions to close at target: {sample}. Take 50%+ profit, "
            f"redeploy capital into higher-conviction trades."
        ),
        "Tactical Opportunity": (
            f"{n} names show actionable directional setups: {sample}. "
            f"Express conviction with defined-risk structures."
        ),
    }
    return headlines[theme_name]


def _summary_metric(theme_name: str, recs: list[Recommendation]) -> str:
    if theme_name in ("Yield Enhancement", "Dead-Weight Purge"):
        total = sum((r.estimated_dollar_value or 0) for r in recs)
        if total != 0:
            return f"~${abs(total):,.0f} premium / capital"
    return f"{len(recs)} positions"
