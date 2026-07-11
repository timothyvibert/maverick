"""Strategy dispatch + aggregation — the model-selection layer.

Contains no numerical kernel of its own: it routes each leg to the right engine
via a ``(style, mode)`` registry and aggregates by signed quantity (price = sum of
qty * leg_price; greeks = qty-weighted sum of per-leg greek dicts).

The default American mode is 'truth' = CRR (the convergent, true-American mark). BS2002
is retained under 'fast' as a fast closed-form cross-check / standing regression canary;
a future PDE engine registers under ('American', 'pde') with no caller change:
    ('European', *)        -> european         (q absorbed via S_eff)
    ('American', 'truth')  -> american_crr      (default)
    ('American', 'fast')   -> american_bs2002   (cross-check)
"""
import math

import numpy as np

from pm.pricing import american_bs2002, american_crr, european
from pm.pricing.american_crr import DEFAULT_CRR_STEPS

REGISTRY = {
    ('European', 'fast'): european,
    ('European', 'truth'): european,
    ('American', 'fast'): american_bs2002,
    ('American', 'truth'): american_crr,
}


def price_leg(S, K, T, r, q, sigma, opt_type,
              style='American', mode='truth', divs=None,
              n_steps=DEFAULT_CRR_STEPS):
    """Price a single option leg, routing to the engine selected by (style, mode).

    ``S`` raw spot (European absorbs q via S_eff internally). ``divs`` is a discrete
    [EX_DATE, DIVIDENDS] DataFrame, used only for American truth mode. Returns a
    scalar if S is scalar, an ndarray if S is an array. Always finite.
    """
    if style not in ('American', 'European'):
        raise ValueError("style must be 'American' or 'European', got %r" % (style,))
    if mode not in ('fast', 'truth'):
        raise ValueError("mode must be 'fast' or 'truth', got %r" % (mode,))
    if opt_type not in ('Call', 'Put'):
        raise ValueError("opt_type must be 'Call' or 'Put', got %r" % (opt_type,))

    engine = REGISTRY[(style, mode)]
    return engine.price(S, K, T, r, q, sigma, opt_type, divs=divs, n_steps=n_steps)


def price_strategy(S, legs, r, q, mode='truth', divs=None,
                   n_steps=DEFAULT_CRR_STEPS):
    """Aggregate price across legs, signed by quantity.

    Returns {'total', 'leg_prices', 'leg_qtys'}. Each leg requires K, T, sigma,
    opt_type; qty (default 1, positive long / negative short) and style (default
    'American') are optional. r, q are shared across legs.
    """
    if not legs:
        zero = 0.0 if np.isscalar(S) else np.zeros_like(np.asarray(S, dtype=np.float64))
        return {'total': zero, 'leg_prices': [], 'leg_qtys': []}

    if np.isscalar(S):
        total = 0.0
    else:
        total = np.zeros_like(np.asarray(S, dtype=np.float64))

    leg_prices = []
    leg_qtys = []
    for leg in legs:
        K = leg['K']
        T = leg['T']
        sigma = leg['sigma']
        opt_type = leg['opt_type']
        qty = int(leg.get('qty', 1))
        style = leg.get('style', 'American')

        leg_price = price_leg(
            S, K, T, r, q, sigma, opt_type,
            style=style, mode=mode, divs=divs, n_steps=n_steps,
        )
        leg_prices.append(leg_price)
        leg_qtys.append(qty)
        total = total + qty * leg_price

    return {'total': total, 'leg_prices': leg_prices, 'leg_qtys': leg_qtys}


def strategy_greeks(S, legs, r, q, today=None,
                    mode='truth', divs=None, n_steps=DEFAULT_CRR_STEPS):
    """Aggregate greeks across legs (qty-weighted sum). Scalar S only.

    European legs use analytic BS greeks on S_eff (rho is real; div_rho padded to
    0). American legs use CRR (the 'truth' default -- discrete divs via crr_greeks,
    else the continuous-q lattice via crr_greeks_continuous_q) or BS2002 (the 'fast'
    cross-check). Returns the aggregated greek dict plus 'leg_greeks' (per-leg dicts).
    """
    if not np.isscalar(S):
        raise ValueError("strategy_greeks requires scalar S; "
                         "use price_strategy for vectorized payoff curves")

    if not legs:
        zeros = {k: 0.0 for k in ('price', 'delta', 'gamma', 'vega', 'theta', 'rho', 'div_rho')}
        zeros['leg_greeks'] = []
        return zeros

    aggregate = {'price': 0.0, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0,
                 'theta': 0.0, 'rho': 0.0, 'div_rho': 0.0}
    leg_greeks_list = []

    for leg in legs:
        K = leg['K']
        T = leg['T']
        sigma = leg['sigma']
        opt_type = leg['opt_type']
        qty = int(leg.get('qty', 1))
        style = leg.get('style', 'American')

        if style == 'European':
            # bs_greeks returns rho (analytic), so setdefault('rho', ...) is a
            # no-op; only div_rho is genuinely padded (BS has no dividend-rho).
            S_eff = S * math.exp(-q * T)
            g_bs = european.bs_greeks(S_eff, K, T, r, sigma, opt_type)
            g = dict(g_bs)
            g.setdefault('rho', 0.0)
            g.setdefault('div_rho', 0.0)
        elif mode == 'fast':
            g = american_bs2002.bs2002_greeks(S, K, T, r, q, sigma, opt_type, today=today)
        else:  # mode == 'truth' American (the default)
            if divs is not None and len(divs) > 0:
                g = american_crr.crr_greeks(S, K, T, r, sigma, divs, opt_type,
                                            n_steps=n_steps, today=today)
            else:
                g = american_crr.crr_greeks_continuous_q(S, K, T, r, q, sigma, opt_type,
                                                         n_steps=n_steps, today=today)

        leg_greeks_list.append(g)
        for key in aggregate:
            aggregate[key] = aggregate[key] + qty * g[key]

    aggregate['leg_greeks'] = leg_greeks_list
    return aggregate


def avg_iv(legs):
    """Notional-weighted (|qty|-weighted) average IV across option legs. Stock legs
    and sigma=None legs are excluded. Returns NaN if no valid legs. (The notional
    weight |qty| * spot * 100 reduces to |qty| since spot is shared across legs.)"""
    weights = []
    sigmas = []
    for leg in legs:
        if leg.get('opt_type') == 'Stock':
            continue
        if leg.get('sigma') is None:
            continue
        weights.append(abs(int(leg['qty'])))
        sigmas.append(float(leg['sigma']))
    if not sigmas or sum(weights) == 0:
        return float('nan')
    return float(np.average(sigmas, weights=weights))
