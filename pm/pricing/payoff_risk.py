"""Payoff, scenario, and risk helpers (vectorized over a spot grid).

Expiry payoffs (intrinsic hockey-sticks), a single-state repricer, and the
strategy statistics: breakevens (slope-aware), probability-of-profit (lognormal),
max profit/loss, and the vectorized strategy greeks. Pricing-bearing helpers route
American -> BS2002 and European -> 6-arg BS (q absorbed via S_eff).
"""
import numpy as np
import pandas as pd

from pm.pricing.american_bs2002 import bs2002_price
from pm.pricing.conventions import norm_cdf, year_frac
from pm.pricing.european import bs_price


def spot_grid(spot, n_points=200, range_pct=0.5):
    """Return a monotone-increasing spot grid spanning [spot*(1-rp), spot*(1+rp)].
    Fail-fast on invalid inputs (spot <= 0, range_pct outside (0,1], n_points < 2)."""
    spot = float(spot)
    range_pct = float(range_pct)
    n_points = int(n_points)
    if not (spot > 0):
        raise ValueError(f"spot_grid: spot must be > 0, got {spot}")
    if not (0 < range_pct <= 1):
        raise ValueError(f"spot_grid: range_pct must be in (0, 1], got {range_pct}")
    if n_points < 2:
        raise ValueError(f"spot_grid: n_points must be >= 2, got {n_points}")
    return np.linspace(spot * (1 - range_pct), spot * (1 + range_pct), n_points)


def payoff_at_expiry(legs, spot_grid):
    """Gross intrinsic payoff at expiry, vectorized over spot_grid. No time value,
    no discounting. Units: per-share-dollars * contracts (chart layer applies
    contract_size). Each leg needs qty, opt_type ('Call'/'Put'/'Stock'), and K
    (Call/Put) or cost_basis (Stock)."""
    spot_grid = np.asarray(spot_grid, dtype=float)
    total = np.zeros_like(spot_grid)
    for leg in legs:
        qty = leg['qty']
        opt_type = leg['opt_type']
        if opt_type == 'Call':
            K = float(leg['K'])
            total = total + qty * np.maximum(spot_grid - K, 0.0)
        elif opt_type == 'Put':
            K = float(leg['K'])
            total = total + qty * np.maximum(K - spot_grid, 0.0)
        elif opt_type == 'Stock':
            cost_basis = float(leg['cost_basis'])
            total = total + qty * (spot_grid - cost_basis)
        else:
            raise ValueError(
                f"payoff_at_expiry: unknown opt_type {opt_type!r} "
                f"(expected 'Call', 'Put', or 'Stock')"
            )
    return total


def payoff_net_at_expiry(legs, spot_grid):
    """Net P&L at expiry: gross intrinsic minus initial debit (plus credit).

    Per-contract dollars: option intrinsic and premium are both multiplied by the
    100-share contract multiplier; stock legs stay per-share * qty. Each option leg
    requires 'mid' (legs without it contribute zero premium). Long leg at mid > 0
    -> debit (negative); short leg -> credit (positive).
    """
    spot_grid = np.asarray(spot_grid, dtype=float)
    total = np.zeros_like(spot_grid)
    for leg in legs:
        qty = float(leg.get('qty', 0.0))
        opt_type = leg.get('opt_type')
        if opt_type == 'Call':
            K = float(leg['K'])
            mid = float(leg.get('mid') or 0.0)
            total = total + qty * np.maximum(spot_grid - K, 0.0) * 100.0
            total = total - qty * mid * 100.0
        elif opt_type == 'Put':
            K = float(leg['K'])
            mid = float(leg.get('mid') or 0.0)
            total = total + qty * np.maximum(K - spot_grid, 0.0) * 100.0
            total = total - qty * mid * 100.0
        elif opt_type == 'Stock':
            cost_basis = float(leg['cost_basis'])
            total = total + qty * (spot_grid - cost_basis)
        else:
            raise ValueError(
                f"payoff_net_at_expiry: unknown opt_type {opt_type!r} "
                f"(expected 'Call', 'Put', or 'Stock')"
            )
    return total


def pnl_at_state(legs, perturbed_spot, perturbed_today,
                 vol_shift_pts, r_perturbed, q):
    """Qty-signed total option value at a perturbed (spot, date, vol, rate) state.

    Stock legs and sigma=None legs are skipped; legs whose perturbed T <= 0 settle
    to intrinsic. American legs price via BS2002, European via 6-arg BS on S_eff.
    """
    total = 0.0
    for leg in legs:
        if leg['opt_type'] == 'Stock':
            continue
        if leg.get('sigma') is None:
            continue
        T_pert = year_frac(perturbed_today, leg['expiry'])
        if T_pert <= 0:
            if leg['opt_type'] == 'Call':
                intrinsic = max(perturbed_spot - leg['K'], 0.0)
            else:
                intrinsic = max(leg['K'] - perturbed_spot, 0.0)
            total += int(leg['qty']) * intrinsic
            continue
        sigma_pert = max(float(leg['sigma']) + vol_shift_pts / 100.0, 0.01)
        if leg.get('style', 'American') == 'American':
            price = bs2002_price(perturbed_spot, leg['K'], T_pert,
                                 r_perturbed, q, sigma_pert,
                                 leg['opt_type'])
        else:
            S_eff = perturbed_spot * np.exp(-q * T_pert)
            price = bs_price(S_eff, leg['K'], T_pert,
                             r_perturbed, sigma_pert, leg['opt_type'])
        total += int(leg['qty']) * float(price)
    return total


def strategy_breakevens(spot_grid, expiry_curve):
    """Spot prices where the at-expiry NET P&L curve crosses zero.

    Requires expiry_curve to be NET P&L (intrinsic + net premium), not gross
    intrinsic. Only strict sign-change crossings with a non-trivial local slope
    count (plateau-zero touches and grid noise below 1% of the curve scale per grid
    step are filtered); near-coincident roots within half a grid step are deduped.
    Returns a sorted list of breakeven spots (empty if no crossing / degenerate).
    """
    spot_grid = np.asarray(spot_grid, dtype=np.float64)
    curve = np.asarray(expiry_curve, dtype=np.float64)
    if spot_grid.size != curve.size or spot_grid.size < 2:
        return []

    curve_scale = float(np.nanmax(curve) - np.nanmin(curve))
    if curve_scale <= 0:
        return []
    dx = float(np.median(np.diff(spot_grid)))
    slope_threshold = 0.01 * curve_scale / max(dx, 1e-9)

    breakevens = []
    for i in range(len(curve) - 1):
        y0, y1 = curve[i], curve[i + 1]
        x0, x1 = spot_grid[i], spot_grid[i + 1]
        if y0 * y1 < 0:   # strict sign change only
            local_slope = abs((y1 - y0) / max(x1 - x0, 1e-9))
            if local_slope < slope_threshold:
                continue
            t = -y0 / (y1 - y0)
            breakevens.append(float(x0 + t * (x1 - x0)))

    if breakevens:
        dedup_tol = max(dx * 0.5, 1e-6)
        deduped = [breakevens[0]]
        for b in breakevens[1:]:
            if b - deduped[-1] > dedup_tol:
                deduped.append(b)
        breakevens = deduped

    return sorted(breakevens)


def pop_lognormal(spot, sigma, T, r, q, spot_grid, expiry_curve):
    """Probability of profit at expiry under lognormal risk-neutral dynamics.

    Closed-form over the profit region(s): the intervals where the NET P&L curve is
    positive, delimited by its zero-crossings (breakevens). Under lognormal S_T
    (ln S_T ~ N(ln(spot) + (r - q - sigma^2/2) T, sigma sqrt(T))), each profit interval
    [a, b] contributes Phi(d(b)) - Phi(d(a)), with d(x) = (ln(x/spot) - drift) /
    (sigma sqrt(T)); an unbounded tail uses Phi(0)=0 at S->0 and Phi(inf)=1 at S->inf.
    The spot grid is used ONLY to locate the profit intervals (sign + linear crossing),
    NOT for the probability mass -- so no tail is truncated and nothing is renormalized.
    Returns a float in [0, 1], or NaN if T <= 0 / sigma <= 0 / spot <= 0.

    Assumes the P&L is monotonic in spot beyond the grid edges (true for standard option
    structures), so a profit run touching the left / right grid edge extends to 0 / +inf.
    """
    grid = np.asarray(spot_grid, dtype=np.float64)
    curve = np.asarray(expiry_curve, dtype=np.float64)
    if grid.size != curve.size or grid.size < 2:
        return float('nan')
    if not (T > 0) or not (sigma > 0) or not (spot > 0):
        return float('nan')

    sigma_sqT = sigma * np.sqrt(T)
    drift = (r - q - 0.5 * sigma * sigma) * T

    def _ln_cdf(x):
        # Lognormal CDF P(S_T <= x): 0 at S->0, 1 at S->inf.
        if x <= 0:
            return 0.0
        if not np.isfinite(x):
            return 1.0
        return float(norm_cdf((np.log(x / spot) - drift) / sigma_sqT))

    def _cross(k):
        # Linear zero-crossing of the curve between grid[k-1] and grid[k].
        x0, x1 = grid[k - 1], grid[k]
        y0, y1 = curve[k - 1], curve[k]
        return float(x0 + (x1 - x0) * (0.0 - y0) / (y1 - y0))

    profit = curve > 0
    n = curve.size
    pop = 0.0
    i = 0
    while i < n:
        if not profit[i]:
            i += 1
            continue
        j = i
        while j < n and profit[j]:
            j += 1
        # Profit run covers grid indices [i, j-1].
        a = 0.0 if i == 0 else _cross(i)            # left boundary (0 -> reaches S=0)
        b = float('inf') if j == n else _cross(j)   # right boundary (inf -> reaches S=inf)
        pop += _ln_cdf(b) - _ln_cdf(a)
        i = j

    return max(0.0, min(1.0, pop))


def strategy_max_profit_loss(spot_grid, expiry_curve, qty_legs):
    """Max profit + max loss at expiry from the NET P&L curve.

    Requires expiry_curve to be NET P&L. The high-spot slope
    slope_high = stock_qty + 100 * net_call_qty (each contract = 100 shares) flags an
    UNBOUNDED side as S -> inf (> 0 gain, < 0 loss). The low-spot side (S -> 0) is always
    bounded (S >= 0), but a net-short-put structure's max loss sits at S = 0 (put
    assignment, stock to zero) -- off the typical +/-50% grid, so the grid minimum
    understates it. When the low-spot slope slope_low = stock_qty - 100 * net_put_qty is
    > 0 (P&L falls toward S = 0) AND the curve's left edge is in its linear
    below-all-strikes tail, that tail is extended to S = 0 for the true max loss.
    Returns max_profit / max_loss (None if that side is unbounded), their spots, and the
    unbounded_gain / unbounded_loss flags.
    """
    curve = np.asarray(expiry_curve, dtype=np.float64)
    grid = np.asarray(spot_grid, dtype=np.float64)
    if curve.size == 0:
        return {
            'max_profit': None, 'max_loss': None,
            'max_profit_at_spot': None, 'max_loss_at_spot': None,
            'unbounded_gain': False, 'unbounded_loss': False,
        }

    stock_qty = sum(int(l['qty']) for l in qty_legs
                    if l.get('opt_type') == 'Stock')
    net_call_qty = sum(int(l['qty']) for l in qty_legs
                       if l.get('opt_type') == 'Call')
    net_put_qty = sum(int(l['qty']) for l in qty_legs
                      if l.get('opt_type') == 'Put')
    slope_high = stock_qty + 100 * net_call_qty
    slope_low = stock_qty - 100 * net_put_qty

    unbounded_gain = (slope_high > 0)
    unbounded_loss = (slope_high < 0)

    max_profit_idx = int(np.argmax(curve))
    max_loss_idx = int(np.argmin(curve))
    max_loss_val = float(curve[max_loss_idx])
    max_loss_spot = float(grid[max_loss_idx])

    # Net-short-put left tail: the max loss sits at S = 0, below the grid. If the
    # curve's left edge is in its linear below-all-strikes tail (local slope ==
    # slope_low), extend it to S = 0 for the true max loss (= strike - premium, x100).
    if slope_low > 0 and grid.size >= 2:
        left_slope = (curve[1] - curve[0]) / (grid[1] - grid[0])
        if abs(left_slope - slope_low) <= 0.01 * abs(slope_low) + 1e-6:
            val_at_zero = float(curve[0] - slope_low * grid[0])
            if val_at_zero < max_loss_val:
                max_loss_val = val_at_zero
                max_loss_spot = 0.0

    return {
        'max_profit':         (None if unbounded_gain
                               else float(curve[max_profit_idx])),
        'max_loss':           (None if unbounded_loss
                               else max_loss_val),
        'max_profit_at_spot': float(grid[max_profit_idx]),
        'max_loss_at_spot':   max_loss_spot,
        'unbounded_gain':     unbounded_gain,
        'unbounded_loss':     unbounded_loss,
    }


def strategy_greeks_vectorized(S_array, legs, r, q, today=None):
    """Aggregate strategy greeks (delta, gamma, vega, theta) at each spot in
    S_array, via bump-and-revalue using the vectorized BS2002 (American) / BS
    (European, via S_eff) pricers. Stock legs contribute delta=qty only.

    Bumps match the scalar BS2002 greeks: delta/gamma spot bump 0.01*S; vega sigma
    bump 0.01 per vol point; theta backward one business day (T - 1/252).
    """
    S_array = np.asarray(S_array, dtype=np.float64)
    if today is None:
        today = pd.Timestamp.today().normalize()

    delta_total = np.zeros_like(S_array)
    gamma_total = np.zeros_like(S_array)
    vega_total = np.zeros_like(S_array)
    theta_total = np.zeros_like(S_array)

    for leg in legs:
        if leg['opt_type'] == 'Stock':
            qty = int(leg['qty'])
            delta_total += qty
            continue

        if leg.get('sigma') is None:
            continue

        T = max(year_frac(today, leg['expiry']), 1e-4)
        sigma = float(leg['sigma'])
        K = float(leg['K'])
        qty = int(leg['qty'])
        opt_type = leg['opt_type']
        style = leg.get('style', 'American')

        bs_S = np.maximum(0.01 * S_array, 1e-4)
        bs_v = 0.01
        T_minus_1bd = max(T - 1.0 / 252.0, 1e-8)

        if style == 'American':
            p = bs2002_price(S_array, K, T, r, q, sigma, opt_type)
            pup = bs2002_price(S_array + bs_S, K, T, r, q, sigma, opt_type)
            pdn = bs2002_price(S_array - bs_S, K, T, r, q, sigma, opt_type)
            pvu = bs2002_price(S_array, K, T, r, q, sigma + bs_v, opt_type)
            pvd = bs2002_price(S_array, K, T, r, q, max(sigma - bs_v, 1e-4), opt_type)
            pt1 = bs2002_price(S_array, K, T_minus_1bd, r, q, sigma, opt_type)
        else:
            S_eff = S_array * np.exp(-q * T)
            S_eff_up = (S_array + bs_S) * np.exp(-q * T)
            S_eff_dn = (S_array - bs_S) * np.exp(-q * T)
            S_eff_t1 = S_array * np.exp(-q * T_minus_1bd)
            p = bs_price(S_eff, K, T, r, sigma, opt_type)
            pup = bs_price(S_eff_up, K, T, r, sigma, opt_type)
            pdn = bs_price(S_eff_dn, K, T, r, sigma, opt_type)
            pvu = bs_price(S_eff, K, T, r, sigma + bs_v, opt_type)
            pvd = bs_price(S_eff, K, T, r, max(sigma - bs_v, 1e-4), opt_type)
            pt1 = bs_price(S_eff_t1, K, T_minus_1bd, r, sigma, opt_type)

        leg_delta = (pup - pdn) / (2.0 * bs_S)
        leg_gamma = (pup - 2.0 * p + pdn) / (bs_S * bs_S)
        leg_vega = (pvu - pvd) / 2.0
        leg_theta = (pt1 - p)

        delta_total += qty * leg_delta
        gamma_total += qty * leg_gamma
        vega_total += qty * leg_vega
        theta_total += qty * leg_theta

    return {
        'delta': delta_total,
        'gamma': gamma_total,
        'vega': vega_total,
        'theta': theta_total,
    }
