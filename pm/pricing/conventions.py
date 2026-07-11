"""Foundation conventions for the pricing engines.

Holds the zero-dependency primitives every engine shares: the option-tenor
day-count, the standard-normal CDF/PDF, and the validation exception. The normal
CDF/PDF use stdlib ``math.erfc`` (``Phi(x) = 0.5*erfc(-x/sqrt(2))``) so the pricing
package adds no scipy to the import surface.
"""
import math

import numpy as np
import pandas as pd

# Option-engine tenor day-count: business days / 252, matching Bloomberg OVME.
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
    """Years-to-expiry as a business-day fraction (busday_count / 252), the
    single tenor entry point for option pricing (OVME-match day-count).

    Returns 0.0 if expiry <= today; callers requiring strictly positive T clamp
    to a small floor (production call sites use 1e-4).
    """
    today = pd.Timestamp(today).normalize().date()
    expiry = pd.Timestamp(expiry).normalize().date()
    if expiry <= today:
        return 0.0
    return np.busday_count(today, expiry) / DAYS_PER_YEAR
