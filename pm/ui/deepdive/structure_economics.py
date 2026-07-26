"""Tier-1 structure economics — pure presentation aggregation, no pricing engine.

Each structure leg is a signed-quantity SLICE of its position, so the position's
cost_basis / market_value are pro-rated by ``allocated_qty / quantity`` and summed
across the legs. This reads fields already on Position; it does not recompute
anything upstream. A leg whose quantity is None/0 (the loader yields None
quantities when the Quantity column is absent) cannot be pro-rated, so the
dependent aggregate degrades to None ("—") rather than crashing.

Tier-2 economics (breakevens, max profit/loss, PoP) are the pricing engine's —
computed in the load path by ``pm.risk.payoff.run_structure_tier2`` and stored on
``AccountState.structure_tier2``; the grid and the structure modal read that
record, and this module carries no placeholder for them.
"""
from __future__ import annotations

from typing import Optional

# The allocation ledger (how much of each position the structures claim vs leave
# standalone) now lives with the structure model in pm.insight.structures, so the
# By-Structure view and the portfolio exposure rollup share one conservation rule.
# Re-exported here so existing callers keep importing it from this module.
from pm.insight.structures import reconcile_allocations  # noqa: F401


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
    component; plus strikes and expiries. Any leg that can't be pro-rated degrades
    the dependent sums to None (rendered "—"); ``degraded`` flags that so the row
    can show it.

    Quantities are PER ASSET CLASS — shares and contracts are different units and
    are never added together: ``shares_allocated`` (signed stock shares the
    structure's legs claim) with ``shares_total`` (the backing stock positions'
    full quantity, None when unknown — the display shows the allocated/total
    coverage cue only when both are known and differ), and ``contracts_net`` (the
    signed net option contracts, the same sum an option-only structure has always
    shown). A class with no legs — or a leg whose slice can't be read — yields
    None for that class, never a guess. ``net_quantity`` (the legacy cross-class
    sum) is retained key-compatible for existing readers; no display renders it.
    """
    costs, mvals, pnls, premiums = [], [], [], []
    strikes, expiries = [], []
    net_qty: Optional[float] = 0.0
    sh_sum = ct_sum = 0.0
    sh_seen = ct_seen = sh_bad = ct_bad = False
    stock_positions: dict = {}
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
        if pos is not None:
            is_option = pos.asset_class == "option"
            try:
                q = float(leg.allocated_qty)
            except (TypeError, ValueError):
                q = None
            if is_option:
                ct_seen = True
                if q is None:
                    ct_bad = True
                else:
                    ct_sum += q
            else:
                sh_seen = True
                if q is None:
                    sh_bad = True
                else:
                    sh_sum += q
                stock_positions[leg.position_id] = pos
        if pos is not None and pos.asset_class == "option":
            if pos.strike is not None:
                strikes.append(float(pos.strike))
            if pos.expiry is not None:
                expiries.append(pos.expiry)

    shares_total: Optional[float] = None
    if stock_positions:
        try:
            shares_total = sum(float(p.quantity) for p in stock_positions.values())
        except (TypeError, ValueError):
            shares_total = None

    def _sum(vals):
        # Degrade (not fake) if any contributing leg is unavailable.
        return None if any(v is None for v in vals) else sum(vals)

    return {
        "net_quantity": net_qty,
        "shares_allocated": (sh_sum if sh_seen and not sh_bad else None),
        "shares_total": shares_total,
        "contracts_net": (ct_sum if ct_seen and not ct_bad else None),
        "net_debit_credit": _sum(costs),
        "net_pnl": _sum(pnls),
        "net_premium": _sum(premiums),
        "strikes": sorted(set(strikes)),
        "expiries": sorted(set(expiries)),
        "degraded": degraded,
    }
