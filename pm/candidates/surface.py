"""Vol-surface fit + IV+pp for an option-chain slice, plus IV-rank and the OVDV
cross-check.

The surface is a global polynomial in log-moneyness ``m = ln(K/S)`` and ``sqrt(T)``
(``T = DTE/365``): ``IV ~= a + b*m + c*m^2 + d*sqrtT + e*m*sqrtT + f*m^2*sqrtT``,
fit in percent-IV space by weighted least squares with robust IRLS re-weighting.

It is a STATISTICAL SMOOTHER, not an arbitrage-free surface: nothing forces the
fitted total variance to rise in T or the risk-neutral density to stay non-negative.
The fitted line is a reference the per-contract IV is measured against, never a
priced surface — callers disclose that and cross-check against Bloomberg's published
OVDV grid (``ovdv_compare``). The fit degrades honestly: the curvature-vs-maturity
term drops below 3 expiries, and too few points fall back to a flat surface
(``iv_excess = 0``) rather than reporting a fake-perfect fit.

IV+pp is a SHAPE measure (rich vs the chain's own smile); it is level-blind by
construction, so it is always paired with IV-rank (the level context) upstream.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Fit gates + filter thresholds.
_MIN_RESIDUAL_DOF = 2        # need (n_fit - n_terms) >= this, else a flat fallback
_F_TERM_MIN_EXPIRIES = 3     # the m^2*sqrtT curvature term needs >= 3 distinct expiries
_DELTA_LO, _DELTA_HI = 0.10, 0.95   # |delta| band for the fit subset
_MAX_SPREAD_PCT = 0.25       # (ask-bid)/mid above this reads stale/illiquid -> out of the fit
_EARNINGS_DTE_MAX = 60       # earnings exclusion applies to <= this DTE
_IRLS_ITERS = 3
_HUBER_K = 1.345


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None   # drop NaN


@dataclass
class Contract:
    """One slice contract, annotated by the fit. IVs are in PERCENT."""
    strike: float
    expiry: date
    right: str                    # 'CALL' | 'PUT'
    iv: Optional[float]
    delta: Optional[float] = None
    vega: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    ticker: Optional[str] = None
    m: Optional[float] = None
    T: Optional[float] = None
    in_fit: bool = False
    iv_fitted: Optional[float] = None
    iv_excess: Optional[float] = None
    iv_source: str = "bbg"        # 'bbg' | 'solved' | 'none'


@dataclass
class SurfaceFit:
    coeffs: Optional[list]        # [a, b, c, d, e, f][:n_terms], or None when degraded
    n_terms: int                  # 5 or 6
    n_fit: int
    n_expiries: int
    residual_std: Optional[float]
    degraded: bool                # True = flat fallback (iv_excess = 0 everywhere)
    reason: str

    def evaluate(self, m: float, T: float) -> Optional[float]:
        if self.degraded or not self.coeffs:
            return None
        return float(np.dot(self.coeffs, _basis(m, math.sqrt(max(T, 0.0)), self.n_terms)))


def _basis(m: float, sqrtT: float, n_terms: int) -> np.ndarray:
    full = [1.0, m, m * m, sqrtT, m * sqrtT, m * m * sqrtT]
    return np.array(full[:n_terms])


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _passes_fit_filters(c: Contract, spot: float, earnings_date, today) -> bool:
    """The fit-subset filter: all contracts still display, only the REGRESSION is
    filtered. OTM-only (deep-ITM carry inflated put-call-parity IV), |delta| band,
    spread, and earnings exclusion for near-dated contracts spanning the report."""
    otm = (c.strike >= spot) if c.right == "CALL" else (c.strike <= spot)
    if not otm:
        return False
    if c.delta is not None and not (_DELTA_LO <= abs(c.delta) <= _DELTA_HI):
        return False
    if c.bid is not None and c.ask is not None and c.mid and c.mid > 0:
        if (c.ask - c.bid) / c.mid > _MAX_SPREAD_PCT:
            return False
    if earnings_date is not None:
        dte = (c.expiry - today).days
        if dte <= _EARNINGS_DTE_MAX and today <= earnings_date <= c.expiry:
            return False
    return True


def fit_surface(contracts, spot, *, today: Optional[date] = None,
                earnings_date=None) -> SurfaceFit:
    """Fit the surface over the filtered subset of *contracts*. Sets ``m``/``T``/
    ``in_fit`` on every contract with a usable IV so ``apply_iv_pp`` can evaluate the
    full set. ``today`` is injectable for deterministic tests."""
    today = today or date.today()
    S = float(spot)
    for c in contracts:
        c.in_fit, c.m, c.T = False, None, None
        iv = _num(c.iv)
        if iv is None or iv <= 0 or not c.strike or c.strike <= 0 or S <= 0:
            continue
        dte = (c.expiry - today).days
        if dte <= 0:
            continue
        c.T = dte / 365.0
        c.m = math.log(c.strike / S)
        c.in_fit = _passes_fit_filters(c, S, earnings_date, today)

    fit_rows = [c for c in contracts if c.in_fit]
    n_expiries = len({c.expiry for c in fit_rows})
    n_terms = 6 if n_expiries >= _F_TERM_MIN_EXPIRIES else 5
    n_fit = len(fit_rows)

    if n_fit - n_terms < _MIN_RESIDUAL_DOF:
        return SurfaceFit(coeffs=None, n_terms=n_terms, n_fit=n_fit,
                          n_expiries=n_expiries, residual_std=None, degraded=True,
                          reason=f"insufficient fit ({n_fit} pts / {n_expiries} expiries)")

    X = np.array([_basis(c.m, math.sqrt(c.T), n_terms) for c in fit_rows])
    y = np.array([float(c.iv) for c in fit_rows])
    w = np.array([c.vega if (c.vega and c.vega > 0) else 1.0 for c in fit_rows])
    coeffs, resid_std = _wls_irls(X, y, w)
    return SurfaceFit(coeffs=[float(b) for b in coeffs], n_terms=n_terms, n_fit=n_fit,
                      n_expiries=n_expiries, residual_std=resid_std, degraded=False,
                      reason=f"{n_terms}-term fit on {n_fit} pts / {n_expiries} expiries")


def _wls(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    return beta, float(np.sqrt(np.sum(resid ** 2) / dof))


def _mad(x) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)


def _wls_irls(X, y, w):
    """WLS with a few Huber IRLS passes so a stale outlier can't bend the surface
    toward itself (the very outliers the screen hunts)."""
    beta, resid_std = _wls(X, y, w)
    for _ in range(_IRLS_ITERS):
        resid = y - X @ beta
        scale = _mad(resid)
        if scale <= 0:
            break
        u = resid / (_HUBER_K * scale)
        rw = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-9))
        beta, resid_std = _wls(X, y, w * rw)
    return beta, resid_std


def apply_iv_pp(contracts, fit: SurfaceFit):
    """Set ``iv_fitted`` / ``iv_excess`` per contract. When the whole fit is degraded,
    the honest flat surface applies (``iv_fitted = iv``, ``iv_excess = 0``); otherwise
    every contract with coordinates — in the fit subset or not — is measured against
    the fitted surface."""
    for c in contracts:
        iv = _num(c.iv)
        if iv is None:
            c.iv_fitted, c.iv_excess = None, None
            continue
        if fit.degraded or c.m is None or c.T is None:
            c.iv_fitted, c.iv_excess = iv, 0.0
        else:
            fitted = fit.evaluate(c.m, c.T)
            c.iv_fitted = fitted
            c.iv_excess = (iv - fitted) if fitted is not None else None
    return contracts


# ---------------------------------------------------------------------------
# Slice orchestration (parse the cached slice frame, solve absent IVs, fit)
# ---------------------------------------------------------------------------

def _solve_iv(c: Contract, spot: float, today: date, r: float, q: float) -> Optional[float]:
    """Solve an American IV (percent) from the contract mid when BBG's iv_mid is
    absent. Best-effort: returns None (never NaN) on any failure."""
    dte = (c.expiry - today).days
    if dte <= 0 or not c.mid or c.mid <= 0:
        return None
    from pm.pricing.implied_vol import implied_vol
    opt = "Call" if c.right == "CALL" else "Put"
    iv = implied_vol(c.mid, spot, c.strike, dte / 365.0, r, q, opt, model="American")
    return iv * 100.0 if iv is not None else None   # solver returns a decimal; surface is percent


def build_slice_surface(slice_df, spot, *, today: Optional[date] = None,
                        earnings_date=None, r: float = 0.045, q: float = 0.0) -> dict:
    """Build the surface + IV+pp for a cached slice frame (indexed by option ticker,
    columns from ``fetch_option_snapshots``). Returns ``{'surface': SurfaceFit,
    'contracts': [Contract, ...]}``. Absent per-contract IVs are solved best-effort."""
    if slice_df is None or getattr(slice_df, "empty", True):
        return {"surface": None, "contracts": []}
    from pm.core.ticker_utils import parse_option_description
    today = today or date.today()

    contracts: list[Contract] = []
    for tk, row in slice_df.iterrows():
        parsed = parse_option_description(str(tk))
        if not parsed:
            continue
        c = Contract(
            strike=parsed["strike"], expiry=parsed["expiry"], right=parsed["right"],
            iv=_num(row.get("iv_mid")), delta=_num(row.get("delta_mid")),
            vega=_num(row.get("vega")), bid=_num(row.get("BID")),
            ask=_num(row.get("ASK")), mid=_num(row.get("PX_MID")), ticker=str(tk),
        )
        if c.iv is None:
            solved = _solve_iv(c, spot, today, r, q)
            c.iv, c.iv_source = (solved, "solved") if solved is not None else (None, "none")
        contracts.append(c)

    fit = fit_surface(contracts, spot, today=today, earnings_date=earnings_date)
    apply_iv_pp(contracts, fit)
    return {"surface": fit, "contracts": contracts}


# ---------------------------------------------------------------------------
# IV-rank + OVDV cross-check
# ---------------------------------------------------------------------------

def iv_rank(current_iv, iv_history):
    """Trailing-range percentile of *current_iv* within *iv_history* (a pandas Series),
    in [0, 1] or None below the min-observation floor. Thin wrapper over the shared
    ``vol_metrics.iv_percentile`` (both in the same, percent, units)."""
    from pm.core.vol_metrics import iv_percentile
    return iv_percentile(current_iv, iv_history)


def ovdv_compare(fit: SurfaceFit, ovdv_grid: dict, *, tol: float = 1.5) -> list:
    """Compare our fitted surface to Bloomberg's published OVDV grid near ATM. Each
    row: our value vs BBG's at (tenor_months, moneyness%), the difference, and a flag
    ('ok' within *tol* vol points, 'diverge' beyond, 'ovdv-gap' where BBG published
    nothing — never interpolated). A divergence is a cross-check flag (our fit pulled
    by a bad print, or a real structural feature; and the two use slightly different
    IV conventions), not a failure."""
    rows = []
    for (months, mny), bbg_iv in sorted(ovdv_grid.items()):
        if bbg_iv is None or (isinstance(bbg_iv, float) and bbg_iv != bbg_iv):
            rows.append({"months": months, "mny": mny, "bbg": None, "ours": None,
                         "diff": None, "flag": "ovdv-gap"})
            continue
        ours = None if (fit is None or fit.degraded) else fit.evaluate(math.log(mny / 100.0), months / 12.0)
        diff = (ours - bbg_iv) if ours is not None else None
        flag = "diverge" if diff is None or abs(diff) > tol else "ok"
        rows.append({"months": months, "mny": mny, "bbg": float(bbg_iv),
                     "ours": ours, "diff": diff, "flag": flag})
    return rows
