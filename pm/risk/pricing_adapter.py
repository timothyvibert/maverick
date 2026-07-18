"""Position + BBG snapshot -> ``pm.pricing`` engine-leg adapter.

The first consumer of ``pm.pricing`` in the live app and the producer side of the
scenario layer. Pure and read-only over an already-loaded
``PortfolioState`` -- no Bloomberg, no recompute, no Dash.

Every per-leg pricing input is resolved per the convention map calibrated at the
Calibrated convention map (measured against the live Terminal):

  * spot  S      -- underlying ``PX_LAST`` (snapshot.underlyings)
  * strike K     -- ``Position.strike`` (``OPT_STRIKE_PX`` as a cross-check)
  * tenor  T     -- ``year_frac(today, Position.expiry)`` busday/252
                    (the snapshot ``dte`` is NaN on this Terminal -- never used)
  * rate   r     -- ``risk_free_curve`` via ``pick_rate_for_dte`` (decimal),
                    else the ``risk_free_rate`` scalar fallback
  * vol    sigma -- ``iv_mid`` / 100 (percent->decimal); brentq IV-solve on the
                    option ``PX_MID`` as the fallback (returns ``None``, never NaN)
  * style        -- ``OPTION_EXERCISE_TYPE_REALTIME`` -> 'American'/'European'
                    (read per leg; default 'American' + warn only if absent)
  * dividends    -- discrete schedule (``projected_dividends_by_ticker``), ex-date
                    in (today, expiry] -> ``divs_df=[EX_DATE,DIVIDENDS]`` for the
                    CRR-truth strip-spot path; continuous-q from
                    ``EQY_DVD_YLD_IND``/100 is the BS2002-fast treatment and the
                    no-schedule fallback. Per name, never mixed.
  * opt_type     -- CALL/PUT -> 'Call'/'Put' (capitalization is load-bearing)

A single ``price_leg(S,K,T,r,q,sigma,opt_type, style, mode, divs=divs_df)`` call
then routes correctly per mode without any further branching: truth-CRR consumes
``divs_df`` (and ignores ``q``) when a discrete schedule exists, else the
continuous-q lattice; fast-BS2002 always consumes ``q`` (the continuous-equivalent
of the schedule). That is the "per name, never mixed" guarantee.

Greek/price source boundary: the exposure layer reads BBG snapshot greeks for the
*current* state; this adapter feeds the engine for *hypothetical*-state repricing.
Conventions stay engine-native here (gamma per $1^2, theta per business day); the
BBG-comparison transforms live in ``reconciliation.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from pm.core.bloomberg_client import pick_rate_for_dte
from pm.pricing import dividends as _div
from pm.pricing import strategy as _strategy
from pm.pricing.conventions import year_frac
from pm.pricing.implied_vol import implied_vol
from pm.core import clock

logger = logging.getLogger(__name__)

_OPT_TYPE_MAP = {"CALL": "Call", "PUT": "Put", "C": "Call", "P": "Put"}
_DEFAULT_STYLE = "American"
_T_FLOOR = 1e-6


@dataclass
class EngineLeg:
    """One option position resolved to ``pm.pricing`` inputs (engine-native units).

    ``divs_df`` (discrete, ex<=expiry) drives the CRR-truth path; ``q`` is the
    continuous-equivalent yield used by the BS2002-fast path and by truth when no
    schedule exists. ``priceable`` is False when spot or sigma could not be
    resolved (an empty/nonexistent contract row, or an IV solve that failed) --
    such a leg is retained for reporting but must be skipped before pricing.
    """
    position_id: str
    bbg_ticker: str
    underlying_bbg: Optional[str]
    account: str
    opt_type: str                 # 'Call' | 'Put'
    style: str                    # 'American' | 'European'
    K: float
    expiry: date
    T: float                      # year-fraction, busday/252
    qty: float                    # signed contracts (long +, short -)
    multiplier: int
    spot: Optional[float]
    sigma: Optional[float]        # decimal; None if unsolvable
    sigma_source: str             # 'iv_mid' | 'solved' | 'missing'
    mid: Optional[float]
    r: float
    q: float                      # continuous-equivalent yield
    divs_df: Optional[pd.DataFrame]
    div_mode: str                 # 'discrete' | 'continuous_q' | 'none'
    today: pd.Timestamp
    priceable: bool
    warnings: list = field(default_factory=list)

    # -- engine leg-dict shapes (the two shapes fail SILENTLY on a missing key,
    #    so both are populated explicitly) ----------------------------------
    def to_strategy_leg(self) -> dict:
        """Leg dict for ``strategy.price_strategy`` / ``strategy_greeks`` (T precomputed)."""
        return {
            "K": self.K, "T": self.T, "sigma": self.sigma,
            "opt_type": self.opt_type, "qty": int(self.qty), "style": self.style,
        }

    def to_payoff_leg(self) -> dict:
        """Leg dict for ``payoff_risk.*`` (carries ``expiry`` + ``mid``; computes T itself)."""
        return {
            "opt_type": self.opt_type, "sigma": self.sigma, "expiry": self.expiry,
            "K": self.K, "qty": int(self.qty), "style": self.style,
            "mid": self.mid if self.mid is not None else 0.0,
        }

    # -- per-share engine price / greeks (the reconciliation + 2b point marks) --
    def price(self, mode: str = "truth", spot: Optional[float] = None,
              sigma: Optional[float] = None) -> Optional[float]:
        S = self.spot if spot is None else spot
        sig = self.sigma if sigma is None else sigma
        if not _finite(S) or not _finite(sig):
            return None
        return float(_strategy.price_leg(
            S, self.K, self.T, self.r, self.q, sig, self.opt_type,
            style=self.style, mode=mode, divs=self.divs_df))

    def greeks(self, mode: str = "truth", spot: Optional[float] = None,
               sigma: Optional[float] = None) -> Optional[dict]:
        """Per-share engine greeks (engine-native units). ``None`` if not priceable."""
        S = self.spot if spot is None else spot
        sig = self.sigma if sigma is None else sigma
        if not _finite(S) or not _finite(sig):
            return None
        engine = _strategy.REGISTRY[(self.style, mode)]
        return engine.greeks(S, self.K, self.T, self.r, self.q, sig, self.opt_type,
                             divs=self.divs_df, today=self.today)


def build_engine_legs(state, account_state, today=None) -> list[EngineLeg]:
    """Resolve every option position on ``account_state`` to an ``EngineLeg``.

    ``state`` supplies the load-path market context (``risk_free_curve`` /
    ``risk_free_rate`` / ``projected_dividends_by_ticker``); ``account_state``
    supplies ``positions`` + ``snapshot``. Reads already-loaded state only.
    """
    today = _normalize_today(today)
    today_d = today.date()
    snapshot = getattr(account_state, "snapshot", None)
    opt_snap = getattr(snapshot, "options", None)
    und_snap = getattr(snapshot, "underlyings", None)
    curve = getattr(state, "risk_free_curve", None) or []
    rfr = getattr(state, "risk_free_rate", 0.04)
    pdivs = getattr(state, "projected_dividends_by_ticker", None) or {}

    legs: list[EngineLeg] = []
    for p in getattr(account_state, "positions", []) or []:
        if getattr(p, "asset_class", None) != "option":
            continue
        leg = _build_one(p, opt_snap, und_snap, curve, rfr, pdivs, today, today_d)
        if leg is not None:
            legs.append(leg)
    return legs


def _build_one(p, opt_snap, und_snap, curve, rfr, pdivs, today, today_d) -> Optional[EngineLeg]:
    warns: list[str] = []
    opt_type = _OPT_TYPE_MAP.get(str(getattr(p, "option_type", "")).upper())
    if opt_type is None:
        return None  # not a recognizable option leg

    orow = _snap_row(opt_snap, getattr(p, "bbg_ticker", None))
    K = _num(getattr(p, "strike", None))
    if K is None and orow is not None:
        K = _num(orow.get("OPT_STRIKE_PX"))
    expiry = getattr(p, "expiry", None)
    if K is None or expiry is None or not isinstance(expiry, date):
        return None  # cannot place the leg on the strike/tenor axes

    T = year_frac(today_d, expiry)
    if T <= 0:
        warns.append("expired or zero tenor")
    T = max(T, _T_FLOOR)
    dte = (pd.Timestamp(expiry) - today).days

    pick = pick_rate_for_dte(curve, dte)
    r = pick["rate"] if (pick and pick.get("rate") is not None) else rfr

    style = orow.get("style") if orow is not None else None
    if style not in ("American", "European"):
        style = _DEFAULT_STYLE
        warns.append(f"exercise style unavailable; defaulted '{style}'")

    und_bbg = getattr(p, "underlying_bbg_ticker", None)
    spot = _num(_snap_val(und_snap, und_bbg, "PX_LAST"))
    divs_df, q, div_mode = _resolve_dividends(
        und_bbg, und_snap, pdivs, spot, r, T, today, expiry)

    iv = _num(orow.get("iv_mid")) if orow is not None else None
    mid = _num(orow.get("PX_MID")) if orow is not None else None
    sigma: Optional[float] = None
    sigma_source = "missing"
    if iv is not None and iv > 0:
        sigma, sigma_source = iv / 100.0, "iv_mid"          # iv_mid is in percent
    elif _finite(spot) and mid is not None and mid > 0:
        solved = implied_vol(mid, spot, K, T, r, q, opt_type, model=style)
        if solved is not None:
            sigma, sigma_source = solved, "solved"
        else:
            warns.append("iv_mid missing and IV solve failed")
    elif orow is None:
        warns.append("no option snapshot row (nonexistent contract?)")
    else:
        warns.append("no iv_mid and no usable mid for IV solve")

    priceable = _finite(spot) and sigma is not None
    if orow is not None and _all_nan(orow):
        warns.append("empty option snapshot row")

    return EngineLeg(
        position_id=getattr(p, "position_id", getattr(p, "bbg_ticker", "")),
        bbg_ticker=getattr(p, "bbg_ticker", ""),
        underlying_bbg=und_bbg,
        account=getattr(p, "account", ""),
        opt_type=opt_type, style=style, K=float(K), expiry=expiry, T=float(T),
        qty=float(getattr(p, "quantity", 0) or 0),
        multiplier=int(getattr(p, "multiplier", 100) or 100),
        spot=spot, sigma=sigma, sigma_source=sigma_source, mid=mid,
        r=float(r), q=float(q), divs_df=divs_df, div_mode=div_mode,
        today=today, priceable=priceable, warnings=warns,
    )


def _resolve_dividends(und_bbg, und_snap, pdivs, spot, r, T, today, expiry):
    """Discrete schedule (ex in (today, expiry]) -> (divs_df, q_continuous_equiv, 'discrete');
    else continuous-q from EQY_DVD_YLD_IND -> (None, q, 'continuous_q'); else (None, 0, 'none').
    """
    schedule = []
    if und_bbg and und_bbg in pdivs:
        schedule = (pdivs.get(und_bbg) or {}).get("schedule") or []
    expiry_ts = pd.Timestamp(expiry)
    ex_dates, amts = [], []
    for d in schedule:
        ex, dps = d.get("ex_date"), d.get("dps")
        if ex is None or dps is None:
            continue
        ex_ts = pd.Timestamp(ex)
        if today < ex_ts <= expiry_ts and float(dps) > 0:
            ex_dates.append(ex_ts)
            amts.append(float(dps))
    if ex_dates:
        divs_df = pd.DataFrame({"EX_DATE": ex_dates, "DIVIDENDS": amts})
        q = 0.0
        if _finite(spot) and spot > 0:
            try:
                q = float(_div.divs_to_q(divs_df, spot, r, T, today))
            except Exception:  # noqa: BLE001 -- a bad strip never blocks the leg
                q = 0.0
        return divs_df, q, "discrete"

    y = _num(_snap_val(und_snap, und_bbg, "EQY_DVD_YLD_IND"))
    if y is not None and y > 0:
        return None, y / 100.0, "continuous_q"     # indicated yield is in percent
    return None, 0.0, "none"


# --------------------------------------------------------------------------
# small read helpers
# --------------------------------------------------------------------------
def _normalize_today(today) -> pd.Timestamp:
    if today is None:
        return pd.Timestamp(clock.today())
    return pd.Timestamp(today).normalize()


def _snap_row(snap, ticker):
    if snap is None or ticker is None:
        return None
    try:
        if ticker not in snap.index:
            return None
        row = snap.loc[ticker]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _snap_val(snap, idx, col):
    if snap is None or idx is None:
        return None
    try:
        if idx not in snap.index or col not in getattr(snap, "columns", []):
            return None
        v = snap.loc[idx, col]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else None
    return v


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and np.isfinite(v)


def _all_nan(row) -> bool:
    try:
        return bool(row.isna().all())
    except Exception:  # noqa: BLE001
        return False
