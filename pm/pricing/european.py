"""European options — closed-form Black-Scholes (analytic price and greeks).

The 6-arg core (``bs_price`` / ``bs_greeks``) prices on the dividend-stripped spot
with q = 0 implicit; the common-interface ``price`` / ``greeks`` absorb a continuous
yield via S_eff = S * exp(-q*T) at the caller. Greeks are fully analytic (delta,
gamma, vega, theta, rho); there is no div_rho (q is carried in the spot, not a
separate channel).

Conventions (OVME-match): vega per 1 vol point; theta per business day
(theta_per_year / 252); rho per 1 bp.
"""
import math

import numpy as np

from pm.pricing.conventions import norm_cdf, norm_pdf


def bs_price(S, K, T, r, sigma, opt_type):
    """Vectorized Black-Scholes price. S scalar or ndarray; K/T/r/sigma scalars.

    Expects S to already be the dividend-stripped spot (q = 0 implicit).
    """
    S = np.asarray(S, dtype=float)
    sigma_safe = max(sigma, 1e-6)
    T_safe = max(T, 1e-6)
    sqrtT = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrtT)
    d2 = d1 - sigma_safe * sqrtT
    if opt_type == 'Call':
        price = S * norm_cdf(d1) - K * np.exp(-r * T_safe) * norm_cdf(d2)
    else:
        price = K * np.exp(-r * T_safe) * norm_cdf(-d2) - S * norm_cdf(-d1)
    # T <= 0 returns intrinsic regardless of the safe computation above.
    if T <= 0:
        return np.maximum(S - K, 0) if opt_type == 'Call' else np.maximum(K - S, 0)
    return price


def bs_greeks(S, K, T, r, sigma, opt_type):
    """Vectorized analytic BS greeks: {price, delta, gamma, vega, theta, rho}.

    Returns rho (per 1 bp) but no div_rho. q = 0 implicit (caller passes the
    stripped spot). Theta is per business day (theta_per_year / 252, OVME-match).
    """
    S = np.asarray(S, dtype=float)
    sigma_safe = max(sigma, 1e-6)
    T_safe = max(T, 1e-6)
    sqrtT = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrtT)
    d2 = d1 - sigma_safe * sqrtT

    if opt_type == 'Call':
        price = S * norm_cdf(d1) - K * np.exp(-r * T_safe) * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta_per_year = (-S * norm_pdf(d1) * sigma_safe / (2 * sqrtT)
                          - r * K * np.exp(-r * T_safe) * norm_cdf(d2))
        rho_per_bp = K * T_safe * np.exp(-r * T_safe) * norm_cdf(d2) / 10000.0
    else:  # Put
        price = K * np.exp(-r * T_safe) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1.0  # = -N(-d1)
        theta_per_year = (-S * norm_pdf(d1) * sigma_safe / (2 * sqrtT)
                          + r * K * np.exp(-r * T_safe) * norm_cdf(-d2))
        rho_per_bp = -K * T_safe * np.exp(-r * T_safe) * norm_cdf(-d2) / 10000.0

    gamma = norm_pdf(d1) / (S * sigma_safe * sqrtT)
    vega_per_volpt = S * norm_pdf(d1) * sqrtT / 100.0
    theta_per_day = theta_per_year / 252.0  # OVME-match (per business day)

    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'vega': vega_per_volpt, 'theta': theta_per_day, 'rho': rho_per_bp,
    }


# --- common interface: price() / greeks() with continuous-q absorbed via S_eff ---

def price(S, K, T, r, q, sigma, opt_type, *, divs=None, n_steps=None):
    """European price; q absorbed via S_eff = S * exp(-q*T) then 6-arg BS."""
    if not np.isscalar(S):
        S_eff = np.asarray(S, dtype=np.float64) * np.exp(-q * T)
    else:
        S_eff = S * math.exp(-q * T)
    return bs_price(S_eff, K, T, r, sigma, opt_type)


def greeks(S, K, T, r, q, sigma, opt_type, *, divs=None, today=None, n_steps=None):
    """European greeks on S_eff; bs_greeks already returns rho, so the
    setdefault('rho', ...) is a no-op — only div_rho is padded (BS has none)."""
    S_eff = S * math.exp(-q * T)
    g = dict(bs_greeks(S_eff, K, T, r, sigma, opt_type))
    g.setdefault('rho', 0.0)
    g.setdefault('div_rho', 0.0)
    return g
