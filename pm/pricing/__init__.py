"""pm.pricing — modular option-pricing engines behind a common interface.

Each engine module (european, american_bs2002, american_crr) exposes the same
``price()`` / ``greeks()`` contract; ``strategy`` selects among them via a
``(style, mode)`` registry. Importing this package pulls only numpy/pandas — the
normal CDF/PDF use stdlib ``math.erfc`` and the implied-vol solver lazily imports
``scipy.optimize.brentq`` only when an actual solve runs, so core/offline paths
stay free of scipy at import time.
"""
