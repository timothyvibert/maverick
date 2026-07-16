"""Alert taxonomy: pattern_id -> group.

A presentation/filter lookup that lets the morning blotter slice alerts by the
kind of action they imply. Deliberately NOT stamped onto the ``Fire`` dataclass
and not part of fire creation — it is a contained lookup the blotter row-builder
(and future features) import.

Covers the full live pattern inventory: the per-position / account engine
patterns (``patterns.PATTERN_META`` — P1-P15, P21) and the structure fires
P16-P20 (``structure_fires._META``). ``test_pattern_groups`` asserts this
stays exact.
"""
from __future__ import annotations

from typing import Optional

# pattern_id -> group key. Four groups, by the action the alert implies.
PATTERN_GROUP: dict[str, str] = {
    # Position management — manage / exit an existing position.
    "P1": "position",
    "P2": "position",
    "P3": "position",
    "P4": "position",
    "P5": "position",
    "P6": "position",
    "P7": "position",
    "P8": "position",
    "P10": "position",
    "P16": "position",
    "P17": "position",
    "P18": "position",
    "P19": "position",
    "P20": "position",
    # Market / opportunity.
    "P11": "market",
    "P13": "market",
    "P15": "market",
    # Catalyst — a calendar/event setup (earnings, research notes).
    "P14": "catalyst",
    "P21": "catalyst",
    # Informational — context, not a direct action.
    "P9": "informational",
    "P12": "informational",
}

# Group key -> display label.
GROUP_LABELS: dict[str, str] = {
    "position": "Position",
    "market": "Market",
    "catalyst": "Catalyst",
    "informational": "Informational",
}

# Canonical display / control order.
GROUP_ORDER: tuple[str, ...] = ("position", "market", "catalyst", "informational")


def group_for(pattern_id: str) -> Optional[str]:
    """The group of a pattern, or None if unmapped. The inventory test keeps the
    map exhaustive, so None means a genuinely unknown id (and the alert is then
    never hidden by the group filter — see ``apply_alert_filters``)."""
    return PATTERN_GROUP.get(pattern_id)


def all_pattern_meta() -> dict[str, tuple[str, int]]:
    """pattern_id -> (display name, tier) across the full live inventory (the
    engine patterns incl. P21, plus the P16-P20 structure fires). The single
    place that unions the two metadata sources — used by the inventory guard
    test."""
    from pm.insight.patterns import PATTERN_META
    from pm.insight.structure_fires import _META as STRUCTURE_META
    return {**PATTERN_META, **STRUCTURE_META}
