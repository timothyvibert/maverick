"""American options — Bjerksund-Stensland (2002) two-step closed-form approximation.

Mathematical reference: Bjerksund & Stensland (2002), "Closed Form Valuation of
American Options". Implements Proposition 1 verbatim (twelve terms: six phi at the
golden-split time t, six psi at maturity T). Two non-obvious points (do not change
without re-verifying against the paper):
  (1) the inner boundary x = X_{T-t} uses time T-t (~0.382*T), NOT t (~0.618*T);
  (2) psi -> M(d, D, rho) is called with POSITIVE d, D because the paper's d/D
      already carry the leading minus sign (App. B Eq. B.8 + Prop. 1).

The bivariate-normal CDF M is computed by the Drezner-Wesolowsky (1990) algorithm
(12-point Gauss-Legendre quadrature, Genz Fortran port), restricted to |rho| < 0.925
— always satisfied here since rho = sqrt(t/T) ~ 0.786.

Continuous cost-of-carry b = r - q. This is the only engine carrying a continuous q;
BS European and CRR American run q = 0 on the stripped spot. Greeks are
bump-and-revalue. The accuracy envelope of the two-step boundary is wide
(approximately 200 bps Tier-1 / 500 bps in the deep-ITM, high-sigma Tier-2 corner) —
the dedicated validation sweep characterizes it.
"""
import math

import numpy as np

from pm.pricing.conventions import norm_cdf

# Bump sizes for the finite-difference rho / div_rho greeks (1 bp).
Q_BUMP_BPS = 0.0001
R_BUMP_BPS = 0.0001

# Drezner-Wesolowsky 1990: 12-point Gauss-Legendre nodes/weights for the
# |rho| < 0.925 regime. Source: jblevins.org/mirror/amiller/genz2d3d.f90 (Alan Genz).
_BS2002_GL_X = np.array([
    -0.9815606342467191, -0.9041172563704750, -0.7699026741943050,
    -0.5873179542866171, -0.3678314989981802, -0.1252334085114692,
])
_BS2002_GL_W = np.array([
    0.04717533638651177, 0.1069393259953183,  0.1600783285433464,
    0.2031674267230659,  0.2334925365383547,  0.2491470458134029,
])


def M_lower_dw1990(a, b, rho):
    """Lower-orthant standard bivariate-normal CDF P(X <= a, Y <= b), correlation rho.

    Drezner-Wesolowsky 1990, restricted to |rho| < 0.925. Vectorized over a, b.
    Genz's bvnd returns the upper orthant P(X > h, Y > k); the lower orthant is
    M(a, b, rho) = bvnd(-a, -b, rho) since (-X, -Y) has the same correlation rho.
    Matches a library bivariate-normal CDF to ~1e-15 on the BS2002 input range.
    """
    h = -np.asarray(a, dtype=np.float64)
    k = -np.asarray(b, dtype=np.float64)
    hk = h * k
    hs = (h*h + k*k) / 2.0
    asr = math.asin(rho)
    bcast_shape = np.broadcast_shapes(h.shape, k.shape) if (h.shape or k.shape) else ()
    bvn = np.zeros(bcast_shape)
    # 12-point Gauss-Legendre integration (6 nodes, doubled with +/- sign per Genz).
    for i in range(6):
        for sign in (-1.0, +1.0):
            sn = math.sin(asr * (sign * _BS2002_GL_X[i] + 1.0) / 2.0)
            bvn = bvn + _BS2002_GL_W[i] * np.exp((sn*hk - hs) / (1.0 - sn*sn))
    return bvn * asr / (4.0 * math.pi) + norm_cdf(-h) * norm_cdf(-k)


def _phi(S, t, gamma, h, X, r, b, v):
    """phi(S, t | gamma, H, X) per paper Eq. (7), with lambda Eq. (8), kappa Eq. (9).
    Vectorized over S (and h, X if arrays). Requires h <= X."""
    sqrt_t = math.sqrt(t)
    d1 = -(np.log(S/h) + (b + (gamma - 0.5)*v**2) * t) / (v * sqrt_t)
    d2 = d1 - 2 * np.log(X/S) / (v * sqrt_t)
    lam = -r + gamma*b + 0.5*gamma*(gamma - 1)*v**2
    kap = 2*b/v**2 + (2*gamma - 1)
    return math.exp(lam*t) * (S**gamma) * (norm_cdf(d1) - (X/S)**kap * norm_cdf(d2))


def _psi(S, T, gamma, H, X_outer, x_inner, t1, r, b, v):
    """psi(S, T | gamma, H, X, x, t) per paper Prop. 1 (derived in App. B).

    The d_i use time t1; the D_i use time T; all eight carry a leading minus sign
    per Prop. 1. The four bivariate-normal calls then take POSITIVE d, D (the
    leading minus is already embedded). Vectorized over S (and H, X_outer, x_inner).
    """
    sqrt_t1 = math.sqrt(t1)
    sqrt_T = math.sqrt(T)
    drift_t1 = (b + (gamma - 0.5)*v**2) * t1
    drift_T = (b + (gamma - 0.5)*v**2) * T

    d1 = -(np.log(S / x_inner) + drift_t1) / (v * sqrt_t1)
    d2 = -(np.log(X_outer**2 / (S * x_inner)) + drift_t1) / (v * sqrt_t1)
    d3 = -(np.log(S / x_inner) - drift_t1) / (v * sqrt_t1)
    d4 = -(np.log(X_outer**2 / (S * x_inner)) - drift_t1) / (v * sqrt_t1)
    D1 = -(np.log(S / H) + drift_T) / (v * sqrt_T)
    D2 = -(np.log(X_outer**2 / (S * H)) + drift_T) / (v * sqrt_T)
    D3 = -(np.log(x_inner**2 / (S * H)) + drift_T) / (v * sqrt_T)
    D4 = -(np.log(S * x_inner**2 / (H * X_outer**2)) + drift_T) / (v * sqrt_T)

    rho = math.sqrt(t1 / T)            # ~0.786 (golden-split ratio)
    lam = -r + gamma*b + 0.5*gamma*(gamma - 1)*v**2          # Eq. (8)
    kap = 2*b/v**2 + (2*gamma - 1)                            # Eq. (9)

    return math.exp(lam * T) * (S**gamma) * (
        M_lower_dw1990(d1, D1, +rho)
        - (X_outer/S)**kap * M_lower_dw1990(d2, D2, +rho)
        - (x_inner/S)**kap * M_lower_dw1990(d3, D3, -rho)
        + (x_inner/X_outer)**kap * M_lower_dw1990(d4, D4, -rho)
    )


def _call_core(S, K, T, r, b, v):
    """Bjerksund-Stensland 2002 American CALL value (paper Prop. 1), vectorized over
    S or K. Puts use the Eq. (19) transform P(S,K,T,r,b,v) = C(K,S,T,r-b,-b,v).

    Edge cases: T <= 0 -> intrinsic; b >= r (q <= 0) -> European call with carry b
    (no early-exercise premium, paper footnote 2); S >= outer boundary X -> intrinsic.
    """
    S_arr = np.asarray(S, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)

    if T <= 0:
        return np.maximum(S_arr - K_arr, 0.0)

    # b >= r -> no early-exercise premium -> European call with cost-of-carry b.
    if b >= r:
        d1 = (np.log(S_arr / K_arr) + (b + 0.5*v**2)*T) / (v*math.sqrt(T))
        d2 = d1 - v*math.sqrt(T)
        return (S_arr * math.exp((b - r)*T) * norm_cdf(d1)
                - K_arr * math.exp(-r * T) * norm_cdf(d2))

    # Golden-ratio time split (Eq. 16).
    t = 0.5 * (math.sqrt(5) - 1) * T
    Tmt = T - t                # ~0.382*T -- inner-boundary time

    beta = (0.5 - b/v**2) + math.sqrt((b/v**2 - 0.5)**2 + 2*r/v**2)   # Eq. (6)

    Binf = (beta / (beta - 1)) * K_arr            # Eq. (12)
    B0 = np.maximum(K_arr, (r/(r - b)) * K_arr)    # Eq. (13)

    # Outer boundary X = X_T (Eq. 10-11 at time T).
    h_X = -(b*T + 2*v*math.sqrt(T)) * (K_arr**2 / ((Binf - B0) * B0))
    X = B0 + (Binf - B0) * (1.0 - np.exp(h_X))

    # Inner boundary x = X_{T-t} (Eq. 10-11 with tau = T-t).
    h_x = -(b*Tmt + 2*v*math.sqrt(Tmt)) * (K_arr**2 / ((Binf - B0) * B0))
    x = B0 + (Binf - B0) * (1.0 - np.exp(h_x))

    aX = (X - K_arr) * X**(-beta)    # Eq. (5) at the two boundaries
    ax = (x - K_arr) * x**(-beta)

    val = (
        aX * S_arr**beta
        - aX * _phi(S_arr, t, beta, X, X, r, b, v)
        +      _phi(S_arr, t, 1.0,  X, X, r, b, v)
        -      _phi(S_arr, t, 1.0,  x, X, r, b, v)
        - K_arr * _phi(S_arr, t, 0.0,  X, X, r, b, v)
        + K_arr * _phi(S_arr, t, 0.0,  x, X, r, b, v)
        + ax * _phi(S_arr, t, beta, x, X, r, b, v)
        - ax * _psi(S_arr, T, beta, x, X, x, t, r, b, v)
        +      _psi(S_arr, T, 1.0,  x, X, x, t, r, b, v)
        -      _psi(S_arr, T, 1.0,  K_arr, X, x, t, r, b, v)
        - K_arr * _psi(S_arr, T, 0.0,  x, X, x, t, r, b, v)
        + K_arr * _psi(S_arr, T, 0.0,  K_arr, X, x, t, r, b, v)
    )
    # Deep-ITM region (S >= outer boundary) overrides with intrinsic value.
    val = np.where(S_arr >= X, S_arr - K_arr, val)
    # sigma -> 0 stability: (X/S)^kap with kap = 2b/sigma^2 + (2gamma-1) can overflow
    # the phi/psi terms; the analytic sigma -> 0 limit of an American call is intrinsic
    # max(S-K, 0), so substitute it wherever the numerical evaluation went non-finite.
    # Exact (limit theorem), and a no-op wherever BS2002 is numerically stable.
    return np.where(np.isfinite(val), val, np.maximum(S_arr - K_arr, 0.0))


def bs2002_price(S, K, T, r, q, v, opt_type):
    """Public BS2002 American option price (continuous-q fast path). b = r - q.
    Returns a scalar if S and K are scalars; an ndarray otherwise."""
    b = r - q
    if opt_type == 'Call':
        return _call_core(S, K, T, r, b, v)
    if opt_type == 'Put':
        return _call_core(K, S, T, r - b, -b, v)   # Eq. (19) transform
    raise ValueError(f"opt_type must be 'Call' or 'Put', got {opt_type!r}")


def bs2002_greeks(S, K, T, r, q, v, opt_type, today=None):
    """All seven BS2002 greeks via bump-and-revalue. Scalar inputs/outputs.

    Conventions: delta/gamma central diff (spot bump 0.01*S); vega per 1 vol point
    (sigma bump 0.01); theta per business day (revalue at T - 1/252, OVME-match);
    rho/div_rho per 1 bp.
    """
    T_minus_1bd = max(T - 1.0/252.0, 1e-8)

    p = float(bs2002_price(S, K, T, r, q, v, opt_type))
    bs_S = max(0.01 * S, 1e-4)
    p_up = float(bs2002_price(S + bs_S, K, T, r, q, v, opt_type))
    p_dn = float(bs2002_price(S - bs_S, K, T, r, q, v, opt_type))
    delta = (p_up - p_dn) / (2.0 * bs_S)
    gamma = (p_up - 2.0*p + p_dn) / (bs_S * bs_S)
    bs_v = 0.01
    p_vu = float(bs2002_price(S, K, T, r, q, v + bs_v, opt_type))
    p_vd = float(bs2002_price(S, K, T, r, q, max(v - bs_v, 1e-4), opt_type))
    vega = (p_vu - p_vd) / 2.0    # already per 1 vol point
    p_t1 = float(bs2002_price(S, K, T_minus_1bd, r, q, v, opt_type))
    theta = (p_t1 - p)            # daily theta directly (T already in years)
    bs_r = R_BUMP_BPS / 10000.0
    p_ru = float(bs2002_price(S, K, T, r + bs_r, q, v, opt_type))
    p_rd = float(bs2002_price(S, K, T, r - bs_r, q, v, opt_type))
    rho = (p_ru - p_rd) / (2.0 * bs_r) / 10000.0
    bs_q = Q_BUMP_BPS / 10000.0
    p_qu = float(bs2002_price(S, K, T, r, q + bs_q, v, opt_type))
    p_qd = float(bs2002_price(S, K, T, r, max(q - bs_q, 0.0), v, opt_type))
    div_rho = (p_qu - p_qd) / (2.0 * bs_q) / 10000.0

    return {
        'price': p, 'delta': delta, 'gamma': gamma, 'vega': vega,
        'theta': theta, 'rho': rho, 'div_rho': div_rho,
    }


# --- common interface ---

def price(S, K, T, r, q, sigma, opt_type, *, divs=None, n_steps=None):
    return bs2002_price(S, K, T, r, q, sigma, opt_type)


def greeks(S, K, T, r, q, sigma, opt_type, *, divs=None, today=None, n_steps=None):
    return bs2002_greeks(S, K, T, r, q, sigma, opt_type, today=today)
