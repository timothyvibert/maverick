"""Implied-volatility solver (Brent two-tier bracket + arbitrage pre-check).

Solves for sigma from an observed price. Dispatches to BS2002 for American mode and
6-arg BS (with q absorbed via S_eff) for European mode. The solver always returns a
finite positive float OR None — never NaN, 0, a sentinel, or a raised exception — so
a chain-surface build can skip a bad point without poisoning aggregates.

scipy.optimize.brentq is imported lazily inside the solver so that importing the
pricing package (for BS / greeks / payoff) never pulls scipy; only an actual IV solve
does.
"""
import math

from pm.pricing.american_bs2002 import bs2002_price
from pm.pricing.european import bs_price

# Bracket floor 1% sigma: the empirical BS2002 stability bound. The overflow root
# cause is (X/S)^kap inside the BS2002 phi/psi terms (X the early-exercise boundary,
# kap = 2b/sigma^2 + (2*gamma-1)); as sigma -> 0 with b > 0, kap -> infinity and
# (X/S)^kap overflows. 1% IV is below any listed option, so coverage is unchanged.
IV_BRACKET_MIN = 1e-2            # 1% sigma floor
IV_BRACKET_MAX_PRIMARY = 5.0     # 500% -- covers ~99.9% of listed options
IV_BRACKET_MAX_FALLBACK = 20.0   # 2000% -- extreme distressed-name tail
IV_XTOL = 1e-8                   # Brent absolute tolerance on sigma
IV_RTOL = 1e-7                   # Brent relative tolerance on sigma
IV_MAXITER = 100                 # Brent max iterations (typically < 30)
IV_INTRINSIC_BUFFER = 1e-4       # |price - intrinsic| < buffer treated as zero-vol


def implied_vol(price_obs, S, K, T, r, q, opt_type, model='American'):
    """Solve for implied volatility from an observed option price.

    ``model``: 'American' (BS2002) or 'European' (BS). Returns a float in
    (IV_BRACKET_MIN, IV_BRACKET_MAX_FALLBACK) on success, or None on any failure:
    invalid inputs, arbitrage violation, no bracketing solution, or solve failure.
    Never raises, never returns NaN / 0 / a sentinel.
    """
    from scipy.optimize import brentq  # lazy: only an actual IV solve pulls scipy

    # Input sanity.
    try:
        price_obs = float(price_obs)
        S = float(S); K = float(K); T = float(T); r = float(r); q = float(q)
    except (TypeError, ValueError):
        return None
    if not (S > 0 and K > 0 and T > 0 and price_obs > 0):
        return None
    if opt_type not in ('Call', 'Put'):
        return None
    if model not in ('American', 'European'):
        return None
    for x in (price_obs, S, K, T, r, q):
        if not (x == x and abs(x) != float('inf')):
            return None

    # No-arbitrage pre-check. American bounds are undiscounted; European bounds use
    # the discounted forward (tighter).
    if opt_type == 'Call':
        if model == 'American':
            intrinsic = max(S - K, 0.0)
            upper = S
        else:  # European
            intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
            upper = S * math.exp(-q * T)
    else:  # Put
        if model == 'American':
            intrinsic = max(K - S, 0.0)
            upper = K
        else:  # European
            intrinsic = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
            upper = K * math.exp(-r * T)

    if price_obs < intrinsic - IV_INTRINSIC_BUFFER:
        return None  # below intrinsic
    if price_obs >= upper:
        return None  # at/above upper bound -- cannot bracket
    if price_obs <= intrinsic + IV_INTRINSIC_BUFFER:
        # Effectively at intrinsic -> sigma ~ 0; return the bracket floor.
        return IV_BRACKET_MIN

    # Build the pricer residual. American: BS2002 takes q natively. European: 6-arg
    # BS, absorb q via S_eff = S * exp(-q*T).
    if model == 'American':
        def _residual(v):
            return float(bs2002_price(S, K, T, r, q, v, opt_type)) - price_obs
    else:  # European
        S_eff = S * math.exp(-q * T)
        def _residual(v):
            return float(bs_price(S_eff, K, T, r, v, opt_type)) - price_obs

    # Primary bracket.
    try:
        iv = brentq(
            _residual,
            IV_BRACKET_MIN,
            IV_BRACKET_MAX_PRIMARY,
            xtol=IV_XTOL,
            rtol=IV_RTOL,
            maxiter=IV_MAXITER,
        )
        return float(iv)
    except ValueError:
        # No sign change in the primary bracket -- try the wider fallback.
        pass
    except (RuntimeError, OverflowError):
        pass

    # Fallback bracket.
    try:
        iv = brentq(
            _residual,
            IV_BRACKET_MIN,
            IV_BRACKET_MAX_FALLBACK,
            xtol=IV_XTOL,
            rtol=IV_RTOL,
            maxiter=IV_MAXITER,
        )
        return float(iv)
    except (ValueError, RuntimeError, OverflowError):
        return None
