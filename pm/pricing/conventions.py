"""Foundation conventions for the pricing engines.

Holds the zero-dependency primitives every engine shares: the option-tenor
day-count, the standard-normal CDF/PDF, and the validation exception. The normal
CDF/PDF use stdlib ``math.erfc`` (``Phi(x) = 0.5*erfc(-x/sqrt(2))``) so the pricing
package adds no scipy to the import surface.
"""
import math

import numpy as np
import pandas as pd

from pm.pricing.trading_calendar import holidays_between

# Option-engine tenor day-count: NYSE trading days / 252. The terminal's option
# clock excludes exchange holidays as well as weekends (probed live, side by
# side); plain weekday counting overstates the tenor by one day per holiday.
DAYS_PER_YEAR = 252


class PricingValidationError(Exception):
    """Raised when an input makes pricing undefined (e.g. a non-positive
    dividend-stripped spot). Engine-local — engines never rely on an
    ambient/global exception name."""


_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_erfc_vec = np.vectorize(math.erfc, otypes=[float])


def norm_cdf(x):
    """Standard-normal CDF, Phi(x) = 0.5*erfc(-x/sqrt(2)). Scalar or ndarray.

    Exact to one ULP versus a library ndtr (scipy's norm.cdf is the same
    erfc-based formula), so it is a drop-in with no economic difference.
    """
    if np.isscalar(x):
        return 0.5 * math.erfc(-x / _SQRT2)
    arr = np.asarray(x, dtype=float)
    return 0.5 * _erfc_vec(-arr / _SQRT2)


def norm_pdf(x):
    """Standard-normal PDF. Scalar or ndarray."""
    if np.isscalar(x):
        return _INV_SQRT_2PI * math.exp(-0.5 * x * x)
    arr = np.asarray(x, dtype=float)
    return _INV_SQRT_2PI * np.exp(-0.5 * arr * arr)


def year_frac(today, expiry):
    """Years-to-expiry as a trading-day fraction — NYSE trading days in
    [today, expiry) divided by 252 — the single tenor entry point for option
    pricing.

    Trading days exclude weekends AND NYSE full-day holidays
    (``trading_calendar.holidays_between``, same half-open window as
    ``np.busday_count``). This matches the terminal's option clock: a live
    side-by-side showed it excludes exchange holidays, so the earlier plain
    weekday count overstated T by one day per holiday spanned. Unscheduled
    exchange closures are not captured — see the trading_calendar module note.

    Returns 0.0 if expiry <= today; callers requiring strictly positive T clamp
    to a small floor (production call sites use 1e-4).
    """
    today = pd.Timestamp(today).normalize().date()
    expiry = pd.Timestamp(expiry).normalize().date()
    if expiry <= today:
        return 0.0
    holidays = holidays_between(today, expiry)
    return np.busday_count(today, expiry, holidays=list(holidays)) / DAYS_PER_YEAR
