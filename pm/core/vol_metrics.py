"""Vol metric computations — pure pandas/numpy over historical series.

All functions take a single pd.Series (or two for spread) and return a
scalar. NaN inputs return NaN/None outputs (no exception).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def iv_percentile(current_iv: float, iv_history: pd.Series) -> float | None:
    """Rank of ``current_iv`` within the historical series, in [0, 1].
    Returns None if history is empty / too short or current_iv is NaN.
    Minimum 30 observations required for a meaningful percentile.
    """
    if iv_history is None or current_iv is None:
        return None
    try:
        if pd.isna(current_iv):
            return None
    except (TypeError, ValueError):
        return None
    if len(iv_history) < 30:
        return None
    s = iv_history.dropna()
    if len(s) < 30:
        return None
    return float((s <= current_iv).mean())


def iv_z_score(current_iv: float, iv_history: pd.Series) -> float | None:
    """(current_iv - mean) / stdev of the history series. None on empty/short."""
    if iv_history is None or current_iv is None:
        return None
    try:
        if pd.isna(current_iv):
            return None
    except (TypeError, ValueError):
        return None
    s = iv_history.dropna()
    if len(s) < 30:
        return None
    sd = s.std()
    if sd == 0 or pd.isna(sd):
        return None
    return float((current_iv - s.mean()) / sd)


def realized_vol_from_prices(
    price_history: pd.Series,
    window_days: int = 30,
) -> float | None:
    """Annualized realized vol over the trailing ``window_days`` of prices.
    std(log returns) * sqrt(252) * 100, returned in vol points.
    """
    if price_history is None or len(price_history) < window_days + 1:
        return None
    s = price_history.dropna().tail(window_days + 1)
    if len(s) < window_days + 1:
        return None
    log_returns = np.log(s / s.shift(1)).dropna()
    if log_returns.empty:
        return None
    sd = log_returns.std()
    if sd == 0 or pd.isna(sd):
        return None
    return float(sd * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def vol_risk_premium(implied_vol: float, realized_vol: float) -> float | None:
    """IV − RV in vol points. Positive = IV is rich vs realized."""
    if implied_vol is None or realized_vol is None:
        return None
    try:
        if pd.isna(implied_vol) or pd.isna(realized_vol):
            return None
    except (TypeError, ValueError):
        return None
    return float(implied_vol - realized_vol)


def trailing_high_low(
    price_history: pd.Series,
    window_days: int,
) -> tuple[float | None, float | None]:
    """High and low of the trailing ``window_days`` of prices."""
    if price_history is None or len(price_history) < window_days:
        return None, None
    s = price_history.dropna().tail(window_days)
    if s.empty:
        return None, None
    return float(s.max()), float(s.min())
