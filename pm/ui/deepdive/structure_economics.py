"""Tier-1 structure economics — pure presentation aggregation, no pricing engine.

Each structure leg is a signed-quantity SLICE of its position, so the position's
cost_basis / market_value are pro-rated by ``allocated_qty / quantity`` and summed
across the legs. This reads fields already on Position; it does not recompute
anything upstream. A leg whose quantity is None/0 (the loader yields None
quantities when the Quantity column is absent) cannot be pro-rated, so the
dependent aggregate degrades to None ("—") rather than crashing.

Tier-2 economics (breakeven, max profit/loss, structure Greeks, theoretical
value) need layer-2 pricing and are intentionally absent — the view renders a
``PENDING_PRICING`` marker for them, not a fabricated number.
"""
from __future__ import annotations

from typing import Optional

# The allocation ledger (how much of each position the structures claim vs leave
# standalone) now lives with the structure model in pm.insight.structures, so the
# By-Structure view and the portfolio exposure rollup share one conservation rule.
# Re-exported here so existing callers keep importing it from this module.
from pm.insight.structures import reconcile_allocations  # noqa: F401

# Tier-2 placeholder — a clean "incomplete" marker, distinct from the legacy
# "<indicative — terminal quote required>" alert-rationale token.
PENDING_PRICING = "pending pricing"


def _slice_fraction(allocated_qty, quantity) -> Optional[float]:
    """``allocated_qty / quantity``, guarding a None/0 quantity."""
    try:
        if quantity is None or float(quantity) == 0.0:
            return None
        return float(allocated_qty) / float(quantity)
    except (TypeError, ValueError):
        return None


def leg_slice(leg, position):
    """(cost, market_value, pnl, premium, available) for one leg's slice.
    ``available`` is False when the slice can't be pro-rated (no position or a
    None/0 quantity); premium is the option-leg cost component (0 for non-options)."""
    if position is None:
        return None, None, None, None, False
    frac = _slice_fraction(leg.allocated_qty, position.quantity)
    if frac is None:
        return None, None, None, None, False
    cost = position.cost_basis * frac if position.cost_basis is not None else None
    mval = position.market_value * frac if position.market_value is not None else None
    pnl = (mval - cost) if (cost is not None and mval is not None) else None
    premium = cost if position.asset_class == "option" else 0.0
    return cost, mval, pnl, premium, True


def structure_economics(structure, by_id: dict) -> dict:
    """Tier-1 economics for one structure. ``by_id`` maps position_id -> Position.

    Net debit/credit = sum of signed sliced cost_basis (paid vs received); net P&L
    = sum of sliced (market_value − cost_basis); net premium = the option-leg cost
    component; plus strikes, expiries, signed net quantity. Any leg that can't be
    pro-rated degrades the dependent sums to None (rendered "—"); ``degraded`` flags
    that so the row can show it.
    """
    costs, mvals, pnls, premiums = [], [], [], []
    strikes, expiries = [], []
    net_qty: Optional[float] = 0.0
    degraded = False

    for leg in structure.legs:
        pos = by_id.get(leg.position_id)
        cost, mval, pnl, premium, ok = leg_slice(leg, pos)
        if not ok:
            degraded = True
        costs.append(cost); mvals.append(mval); pnls.append(pnl); premiums.append(premium)
        try:
            if net_qty is not None:
                net_qty += float(leg.allocated_qty)
        except (TypeError, ValueError):
            net_qty = None
        if pos is not None and pos.asset_class == "option":
            if pos.strike is not None:
                strikes.append(float(pos.strike))
            if pos.expiry is not None:
                expiries.append(pos.expiry)

    def _sum(vals):
        # Degrade (not fake) if any contributing leg is unavailable.
        return None if any(v is None for v in vals) else sum(vals)

    return {
        "net_quantity": net_qty,
        "net_debit_credit": _sum(costs),
        "net_pnl": _sum(pnls),
        "net_premium": _sum(premiums),
        "strikes": sorted(set(strikes)),
        "expiries": sorted(set(expiries)),
        "degraded": degraded,
        # Tier-2 — pending layer-2 pricing.
        "breakeven": PENDING_PRICING,
        "max_profit_loss": PENDING_PRICING,
        "greeks": PENDING_PRICING,
        "theoretical_value": PENDING_PRICING,
    }
