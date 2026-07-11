"""Discrete-cash-dividend handling: the escrowed-dividend (stripped-spot) model.

BS European and CRR American price on the dividend-stripped spot
``S_risky = spot - PV(divs)`` with q = 0 implicit; the continuous-yield
conversion (``divs_to_q``) is used ONLY by the BS2002 fast path and must never
be called from the BS or CRR paths.

Day-count note (intentional, OVME-match): the option tenor uses business-days/252
(see ``conventions.year_frac``) while every dividend PV here discounts at
calendar-days / 365. The busday-vs-calendar mismatch reproduces OVME exactly —
do not normalize it.
"""
import numpy as np

from pm.pricing.conventions import PricingValidationError

# Dividend-PV day-count: calendar days / 365 (distinct from the 252 option clock).
DIV_PV_DAYS_PER_YEAR = 365.0


def pv_dividends(divs_df, r, today, T_years):
    """Present value at t=0 of the discrete cash dividends paid in (today, today+T].

    ``divs_df``: a [EX_DATE, DIVIDENDS] DataFrame (or None / empty).
    ``r``: continuously compounded rate. Each dividend is discounted at its own
    (ex_date - today) calendar-day / 365 offset (OVME-match) — not the option
    tenor's busday/252.
    """
    if divs_df is None or divs_df.empty:
        return 0.0
    pv = 0.0
    for _, row in divs_df.iterrows():
        t = (row['EX_DATE'] - today).days / DIV_PV_DAYS_PER_YEAR
        if 0 < t <= T_years:
            pv += row['DIVIDENDS'] * np.exp(-r * t)
    return pv


def strip_spot(spot, divs_df, r, today, T_years):
    """Return S_risky = spot - PV(divs), the stripped spot on which BS / CRR run
    with q = 0 implicit.

    Raises PricingValidationError if spot - PV <= 0 (pricing on a non-positive
    stripped spot is undefined).
    """
    pv = pv_dividends(divs_df, r, today, T_years)
    s_risky = spot - pv
    if s_risky <= 0:
        raise PricingValidationError(
            f'Strip-spot underflow: spot={spot:.4f}, PV(divs)={pv:.4f}, '
            f'S_risky={s_risky:.4f} <= 0. Likely an extreme dividend '
            f'forecast or wrong sign on divs_df.')
    return s_risky


def pv_dividends_at_node(divs_df, r, today, t_node_years, T_years):
    """PV of the REMAINING dividends as of CRR node time t_node, discounted from
    each ex-date back to t_node.

    Lets the CRR tree recover S_actual = S_risky + PV_at_node at every interior
    node so the early-exercise test sees the true ex-div spot. All time
    arithmetic uses calendar days / 365 (consistent with pv_dividends).
    """
    if divs_df is None or divs_df.empty:
        return 0.0
    pv = 0.0
    for _, row in divs_df.iterrows():
        t_ex = (row['EX_DATE'] - today).days / DIV_PV_DAYS_PER_YEAR
        if t_node_years < t_ex <= T_years:
            pv += row['DIVIDENDS'] * np.exp(-r * (t_ex - t_node_years))
    return pv


def divs_to_q(divs_df, spot, r, T_years, today):
    """Convert discrete dividends to an equivalent continuous yield,
    q = PV(divs) / (spot * T), matching OVME.

    Used ONLY for the BS2002 fast path — do not call from the BS or CRR paths,
    which use strip_spot.
    """
    if T_years <= 0:
        return 0.0
    pv = pv_dividends(divs_df, r, today, T_years)
    return pv / (spot * T_years)
