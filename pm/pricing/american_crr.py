"""American options — Cox-Ross-Rubinstein binomial lattice (converges to truth).

The tree is built on the dividend-stripped risky spot S_risky_0 = spot - PV(divs)
with q = 0 implicit; discrete cash dividends are absorbed into the root and the
actual ex-div spot S_actual = S_risky + PV_remaining is recovered at every interior
node so the early-exercise test sees the true ex-div spot. ``crr_price_continuous_q``
is the same lattice with a continuous yield (calibration reference). Greeks are
bump-and-revalue, EXCEPT gamma, which is read directly off the lattice's step-2 nodes
(smooth and convergent in n, avoiding the bump second-difference sawtooth); theta is
discrete-div-safe (revalue one business day forward with a re-anchored div schedule).

Day-count note (intentional, OVME-match): the option tenor T is busday/252 while
PV(divs) discounts at calendar/365 — do not normalize.
"""
import numpy as np
import pandas as pd

from pm.pricing.dividends import pv_dividends_at_node, strip_spot

DEFAULT_CRR_STEPS = 500   # accurate path
FAST_CRR_STEPS = 200      # IV-solver inner-loop fast path


def _tree_node_gamma(vals3, spots3):
    """Gamma off the lattice's three step-2 nodes (ascending spot order): the second
    difference of option value with respect to the node spot levels. Uses the tree's
    own grid, so it is smooth and convergent in n -- unlike a bump-and-revalue second
    difference, which carries the lattice sawtooth. (For CRR the middle step-2 node
    sits at the current spot, so this is gamma evaluated at the money.)"""
    S_dd, S_ud, S_uu = spots3
    V_dd, V_ud, V_uu = vals3
    delta_up = (V_uu - V_ud) / (S_uu - S_ud)
    delta_dn = (V_ud - V_dd) / (S_ud - S_dd)
    return float((delta_up - delta_dn) / ((S_uu - S_dd) / 2.0))


def crr_price(S_full, K, T, r, sigma, divs_df, opt_type,
              n_steps=DEFAULT_CRR_STEPS, today=None, return_gamma_nodes=False):
    """CRR American option on a strip-spot tree with discrete cash dividends.

    ``S_full``: the un-stripped market spot. ``sigma``: vol of S_risky. Note
    the shipped IV solver does NOT back-solve under this convention: its
    American mode solves under BS2002 continuous-q on the full spot - a
    deliberate, bounded mismatch used only on the iv_mid-missing fallback
    (see pm.risk.pricing_adapter). ``divs_df``: canonical
    [EX_DATE, DIVIDENDS] DataFrame, or None / empty for no-div names.
    """
    if today is None:
        today = pd.Timestamp.today().normalize()
    if T <= 0:
        intrinsic = (max(S_full - K, 0) if opt_type == 'Call'
                     else max(K - S_full, 0))
        return (intrinsic, None) if return_gamma_nodes else intrinsic

    # Risky spot at the root -- the tree grows from this (divs absorbed here).
    S_risky_0 = strip_spot(S_full, divs_df, r, today, T)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    S_risky_terminal = S_risky_0 * (u ** j) * (d ** (n_steps - j))
    # No remaining divs at terminal time: S_actual_terminal = S_risky_terminal.
    S_actual_terminal = S_risky_terminal

    if opt_type == 'Call':
        V = np.maximum(S_actual_terminal - K, 0)
    else:
        V = np.maximum(K - S_actual_terminal, 0)

    # Backward induction (vectorized over node index j; the loop is time only).
    gamma_nodes = None
    for i in range(n_steps - 1, -1, -1):
        t_node = i * dt
        j_arr = np.arange(i + 1)
        S_risky_node = S_risky_0 * (u ** j_arr) * (d ** (i - j_arr))
        # Recover the actual ex-div spot at this node from the remaining divs.
        pv_rem = pv_dividends_at_node(divs_df, r, today, t_node, T)
        S_actual_node = S_risky_node + pv_rem

        continuation = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        if opt_type == 'Call':
            intrinsic = np.maximum(S_actual_node - K, 0)
        else:
            intrinsic = np.maximum(K - S_actual_node, 0)
        V = np.maximum(continuation, intrinsic)
        if return_gamma_nodes and i == 2:
            gamma_nodes = (V.copy(), S_actual_node.copy())

    return (V[0], gamma_nodes) if return_gamma_nodes else V[0]


def crr_price_continuous_q(S, K, T, r, q, sigma, opt_type,
                           n_steps=DEFAULT_CRR_STEPS, return_gamma_nodes=False):
    """CRR American tree with a continuous dividend yield q (no discrete divs, no
    strip). Calibration reference for the BS2002 path; not on any load path."""
    if T <= 0:
        intrinsic = (max(S - K, 0) if opt_type == 'Call' else max(K - S, 0))
        return (intrinsic, None) if return_gamma_nodes else intrinsic
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    S_terminal = S * (u ** j) * (d ** (n_steps - j))
    if opt_type == 'Call':
        V = np.maximum(S_terminal - K, 0)
    else:
        V = np.maximum(K - S_terminal, 0)

    gamma_nodes = None
    for i in range(n_steps - 1, -1, -1):
        j_arr = np.arange(i + 1)
        S_node = S * (u ** j_arr) * (d ** (i - j_arr))
        continuation = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        if opt_type == 'Call':
            intrinsic = np.maximum(S_node - K, 0)
        else:
            intrinsic = np.maximum(K - S_node, 0)
        V = np.maximum(continuation, intrinsic)
        if return_gamma_nodes and i == 2:
            gamma_nodes = (V.copy(), S_node.copy())

    return (V[0], gamma_nodes) if return_gamma_nodes else V[0]


def crr_greeks(S_full, K, T, r, sigma, divs_df, opt_type,
               n_steps=DEFAULT_CRR_STEPS, today=None):
    """CRR price + greeks, bump-and-revalue throughout (discrete-div safe).

    Theta revalues one business day forward (re-anchored div schedule, T - 1/252),
    not a finite difference at adjacent tree nodes (which biases on discrete-div
    names). Conventions: vega per 1 vol point; theta per business day (OVME-match);
    rho per 1 bp; div_rho per 1 bp of the dividend amount (relative bump).
    Returns {price, delta, gamma, vega, theta, rho, div_rho}.
    """
    if today is None:
        today = pd.Timestamp.today().normalize()

    p_base, gamma_nodes = crr_price(S_full, K, T, r, sigma, divs_df, opt_type,
                                    n_steps, today, return_gamma_nodes=True)

    p_up = crr_price(S_full * 1.01, K, T, r, sigma, divs_df, opt_type, n_steps, today)
    p_dn = crr_price(S_full * 0.99, K, T, r, sigma, divs_df, opt_type, n_steps, today)
    delta = (p_up - p_dn) / (S_full * 0.02)
    # Gamma off the lattice nodes (smooth/convergent); the spot bump still gives delta.
    gamma = (_tree_node_gamma(*gamma_nodes) if gamma_nodes is not None
             else (p_up - 2 * p_base + p_dn) / (S_full * 0.01) ** 2)

    p_v_up = crr_price(S_full, K, T, r, sigma + 0.01, divs_df, opt_type, n_steps, today)
    p_v_dn = crr_price(S_full, K, T, r, sigma - 0.01, divs_df, opt_type, n_steps, today)
    vega = (p_v_up - p_v_dn) / 2.0  # per 1 vol point

    p_r_up = crr_price(S_full, K, T, r + 0.0001, sigma, divs_df, opt_type, n_steps, today)
    p_r_dn = crr_price(S_full, K, T, r - 0.0001, sigma, divs_df, opt_type, n_steps, today)
    rho = (p_r_up - p_r_dn) / 2.0  # per 1 bp

    # Theta -- revalue one business day forward (OVME-match, per business day).
    today_next = pd.Timestamp(
        np.busday_offset(today.date(), 1).astype('datetime64[D]'))
    T_next = T - 1.0 / 252.0
    if T_next > 0:
        p_next = crr_price(S_full, K, T_next, r, sigma, divs_df, opt_type, n_steps, today_next)
        theta_per_day = p_next - p_base
    else:
        theta_per_day = (max(S_full - K, 0) - p_base if opt_type == 'Call'
                         else max(K - S_full, 0) - p_base)

    # div_rho -- relative +/- 1 bp bump of the dividend amounts.
    if divs_df is None or divs_df.empty:
        div_rho = 0.0
    else:
        divs_up = divs_df.copy(); divs_up['DIVIDENDS'] *= 1.0001
        divs_dn = divs_df.copy(); divs_dn['DIVIDENDS'] *= 0.9999
        p_d_up = crr_price(S_full, K, T, r, sigma, divs_up, opt_type, n_steps, today)
        p_d_dn = crr_price(S_full, K, T, r, sigma, divs_dn, opt_type, n_steps, today)
        div_rho = (p_d_up - p_d_dn) / 2.0

    return {
        'price': p_base, 'delta': delta, 'gamma': gamma,
        'vega': vega, 'theta': theta_per_day, 'rho': rho,
        'div_rho': div_rho,
    }


def crr_greeks_continuous_q(S, K, T, r, q, sigma, opt_type,
                            n_steps=DEFAULT_CRR_STEPS, today=None):
    """CRR price + greeks on the continuous-q lattice, bump-and-revalue throughout.

    The continuous-q counterpart of crr_greeks, with the same conventions: vega per
    1 vol point; theta per business day (one-business-day T - 1/252 reprice); rho per
    1 bp (central, /2 -- not /10000); div_rho the sensitivity to the continuous yield
    q via a central +/- 1 bp bump (mirrors the rho bump). today is accepted for
    interface symmetry but does not affect a continuous-q price (there is no dividend
    schedule to re-anchor). Returns {price, delta, gamma, vega, theta, rho, div_rho}.
    """
    p_base, gamma_nodes = crr_price_continuous_q(S, K, T, r, q, sigma, opt_type,
                                                 n_steps, return_gamma_nodes=True)

    p_up = crr_price_continuous_q(S * 1.01, K, T, r, q, sigma, opt_type, n_steps)
    p_dn = crr_price_continuous_q(S * 0.99, K, T, r, q, sigma, opt_type, n_steps)
    delta = (p_up - p_dn) / (S * 0.02)
    # Gamma off the lattice nodes (smooth/convergent); the spot bump still gives delta.
    gamma = (_tree_node_gamma(*gamma_nodes) if gamma_nodes is not None
             else (p_up - 2 * p_base + p_dn) / (S * 0.01) ** 2)

    p_v_up = crr_price_continuous_q(S, K, T, r, q, sigma + 0.01, opt_type, n_steps)
    p_v_dn = crr_price_continuous_q(S, K, T, r, q, sigma - 0.01, opt_type, n_steps)
    vega = (p_v_up - p_v_dn) / 2.0  # per 1 vol point

    p_r_up = crr_price_continuous_q(S, K, T, r + 0.0001, q, sigma, opt_type, n_steps)
    p_r_dn = crr_price_continuous_q(S, K, T, r - 0.0001, q, sigma, opt_type, n_steps)
    rho = (p_r_up - p_r_dn) / 2.0  # per 1 bp

    # Theta -- revalue one business day forward (per business day). No dividend
    # schedule on the continuous-q path, so no calendar re-anchor is needed.
    T_next = T - 1.0 / 252.0
    if T_next > 0:
        p_next = crr_price_continuous_q(S, K, T_next, r, q, sigma, opt_type, n_steps)
        theta_per_day = p_next - p_base
    else:
        theta_per_day = (max(S - K, 0) - p_base if opt_type == 'Call'
                         else max(K - S, 0) - p_base)

    # div_rho -- sensitivity to the continuous yield q, central +/- 1 bp bump.
    p_q_up = crr_price_continuous_q(S, K, T, r, q + 0.0001, sigma, opt_type, n_steps)
    p_q_dn = crr_price_continuous_q(S, K, T, r, q - 0.0001, sigma, opt_type, n_steps)
    div_rho = (p_q_up - p_q_dn) / 2.0

    return {
        'price': p_base, 'delta': delta, 'gamma': gamma,
        'vega': vega, 'theta': theta_per_day, 'rho': rho,
        'div_rho': div_rho,
    }


# --- common interface ---

def price(S, K, T, r, q, sigma, opt_type, *, divs=None, n_steps=DEFAULT_CRR_STEPS):
    """American CRR price; discrete divs if given, else continuous-q. Scalar S
    prices directly; array S loops (the lattice is scalar-per-evaluation)."""
    is_array = not np.isscalar(S)
    has_divs = divs is not None and len(divs) > 0
    if not is_array:
        if has_divs:
            return crr_price(S, K, T, r, sigma, divs, opt_type, n_steps=n_steps)
        return crr_price_continuous_q(S, K, T, r, q, sigma, opt_type, n_steps=n_steps)
    S_arr = np.asarray(S, dtype=np.float64)
    out = np.empty(S_arr.shape, dtype=np.float64)
    if has_divs:
        for i, s in enumerate(S_arr.flat):
            out.flat[i] = crr_price(float(s), K, T, r, sigma, divs, opt_type, n_steps=n_steps)
    else:
        for i, s in enumerate(S_arr.flat):
            out.flat[i] = crr_price_continuous_q(float(s), K, T, r, q, sigma, opt_type, n_steps=n_steps)
    return out


def greeks(S, K, T, r, q, sigma, opt_type, *, divs=None, today=None, n_steps=DEFAULT_CRR_STEPS):
    """American CRR greeks: discrete-dividend (crr_greeks) when a div schedule is
    given, else the continuous-q lattice (crr_greeks_continuous_q)."""
    if divs is not None and len(divs) > 0:
        return crr_greeks(S, K, T, r, sigma, divs, opt_type, n_steps=n_steps, today=today)
    return crr_greeks_continuous_q(S, K, T, r, q, sigma, opt_type, n_steps=n_steps, today=today)
