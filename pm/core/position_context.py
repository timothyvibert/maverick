"""Per-position derived metrics for the recommendation engine.

Builds a ``PositionContext`` per position from (Position record, BBG
snapshot row, option snapshot row, per-account NAV) so rule functions
don't recompute the same things.

V1 takes ``list[Position]`` directly instead of ``ParsedPortfolio``.
Cash and Other positions are skipped (no recommendations on them).
Funds / ETFs are treated as equities — ``instrument_type='equity'`` so
existing equity rules apply when underlying data is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from pm.core.portfolio_signals import FIELDS
from pm.ingest.position_builder import Position
from pm.core import clock


@dataclass
class PositionContext:
    """All the derived facts a recommendation rule might need."""
    # Identification
    bbg_ticker: str
    underlying_symbol: str
    underlying_bbg_ticker: str
    instrument_type: str            # 'equity' | 'option' | 'warrant' | 'other'

    # For options
    right: Optional[str]            # 'PUT' | 'CALL' | None for equity
    strike: Optional[float]
    expiry: Optional[date]
    dte: Optional[int]

    # Position economics
    quantity: float                  # signed (negative = short)
    multiplier: int                  # 1 for equity, 100 for option
    cost_basis: Optional[float]      # signed
    market_value: float              # signed
    pct_pnl: Optional[float]         # +0.50 = 50% profit, regardless of side

    # Live market
    spot: Optional[float]
    delta: Optional[float]           # raw greek (per-share); not multiplied by qty
    vega: Optional[float]
    theta: Optional[float]
    gamma: Optional[float]
    iv_1m: Optional[float]
    iv_3m: Optional[float]

    # Derived
    moneyness: Optional[float]       # signed: positive = ITM
    pct_captured: Optional[float]    # = pct_pnl
    portfolio_pct_of_nav: float
    has_long_stock: bool
    has_short_calls: bool
    has_short_puts: bool
    has_long_protection: bool

    # Underlying-level convenience copies
    rsi_14d: Optional[float]
    earnings_date: Optional[date]
    days_to_earnings: Optional[int]
    beta: Optional[float]
    iv_percentile: Optional[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Position asset classes that get a PositionContext (i.e. ones the
# recommender can act on).
_TRADEABLE_ASSET_CLASSES = ("option", "equity", "fund_etf")


def _signed_moneyness(spot, strike, right) -> Optional[float]:
    """ITM-positive moneyness. For calls: (S-K)/K. For puts: -(S-K)/K = (K-S)/K."""
    if spot is None or strike is None or strike == 0:
        return None
    try:
        if pd.isna(spot) or pd.isna(strike):
            return None
    except (TypeError, ValueError):
        return None
    raw = (float(spot) - float(strike)) / float(strike)
    if right == "CALL":
        return raw
    if right == "PUT":
        return -raw
    return None


def _dte(expiry, today: Optional[date] = None) -> Optional[int]:
    if expiry is None:
        return None
    today = today or clock.today()
    try:
        return (expiry - today).days
    except Exception:
        return None


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _spot_for(ticker, snapshot_underlyings) -> Optional[float]:
    if (snapshot_underlyings is None or snapshot_underlyings.empty
            or ticker not in snapshot_underlyings.index
            or "PX_LAST" not in snapshot_underlyings.columns):
        return None
    return _safe_float(snapshot_underlyings.loc[ticker, "PX_LAST"])


def _greek_for(opt_ticker, snapshot_options, col) -> Optional[float]:
    if (snapshot_options is None or snapshot_options.empty
            or opt_ticker not in snapshot_options.index
            or col not in snapshot_options.columns):
        return None
    return _safe_float(snapshot_options.loc[opt_ticker, col])


def _earnings_date(ticker, snapshot_underlyings) -> Optional[date]:
    col = FIELDS["earn_dt"]
    if (snapshot_underlyings is None or snapshot_underlyings.empty
            or ticker not in snapshot_underlyings.index
            or col not in snapshot_underlyings.columns):
        return None
    val = snapshot_underlyings.loc[ticker, col]
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _underlying_field(
    snapshot_underlyings, ticker: str, field_key: str,
) -> Optional[float]:
    """Pull a numeric field from the underlying snapshot, NaN-safe."""
    if (snapshot_underlyings is None or snapshot_underlyings.empty
            or ticker not in snapshot_underlyings.index
            or field_key not in snapshot_underlyings.columns):
        return None
    return _safe_float(snapshot_underlyings.loc[ticker, field_key])


def _build_overlay_map(positions: list[Position]) -> dict[str, dict[str, bool]]:
    """For each underlying symbol seen across stock + option positions,
    compute booleans: has_long_stock, has_short_calls, has_short_puts,
    has_long_protection. Keyed by symbol.

    Cash and Other positions are ignored — they have no underlying overlay.
    """
    out: dict[str, dict[str, bool]] = {}

    syms: set[str] = set()
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and p.symbol:
            syms.add(p.symbol)
        elif p.asset_class == "option" and p.underlying_symbol:
            syms.add(p.underlying_symbol)

    for sym in syms:
        long_stock = False
        short_calls = False
        short_puts = False
        long_protection = False
        for p in positions:
            if p.asset_class in ("equity", "fund_etf") and p.symbol == sym:
                if p.quantity is not None and p.quantity > 0:
                    long_stock = True
            elif p.asset_class == "option" and p.underlying_symbol == sym:
                if p.right == "CALL" and p.quantity is not None and p.quantity < 0:
                    short_calls = True
                elif p.right == "PUT" and p.quantity is not None and p.quantity < 0:
                    short_puts = True
                elif p.right == "PUT" and p.quantity is not None and p.quantity > 0:
                    long_protection = True
        out[sym] = {
            "has_long_stock": long_stock,
            "has_short_calls": short_calls,
            "has_short_puts": short_puts,
            "has_long_protection": long_protection,
        }
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_position_contexts(
    positions: list[Position],
    snapshot,                 # PortfolioSnapshot
    account_nav: float,
    iv_pctl_by_ticker: Optional[dict[str, float]] = None,
) -> list[PositionContext]:
    """Build a PositionContext for every tradeable position in
    ``positions`` (option / equity / fund_etf). Skips cash and other.

    ``snapshot`` is a PortfolioSnapshot with ``.underlyings`` (indexed
    by bbg_ticker) and ``.options`` (indexed by option bbg_ticker).
    ``account_nav`` is the sum of |market_value| for the positions'
    account; portfolio_pct_of_nav is computed against it.
    """
    iv_pctl_by_ticker = iv_pctl_by_ticker or {}
    underlyings = snapshot.underlyings if hasattr(snapshot, "underlyings") else None
    options = snapshot.options if hasattr(snapshot, "options") else None

    overlay_map = _build_overlay_map(positions)

    nav = float(account_nav) if account_nav else 0.0

    contexts: list[PositionContext] = []

    for p in positions:
        if p.asset_class not in _TRADEABLE_ASSET_CLASSES:
            continue

        if p.asset_class == "option":
            ctx = _build_option_context(p, underlyings, options, overlay_map, iv_pctl_by_ticker, nav)
        else:
            ctx = _build_equity_context(p, underlyings, overlay_map, iv_pctl_by_ticker, nav)

        if ctx is not None:
            contexts.append(ctx)

    return contexts


def _build_equity_context(
    p: Position,
    underlyings,
    overlay_map: dict[str, dict[str, bool]],
    iv_pctl_by_ticker: dict[str, float],
    nav: float,
) -> Optional[PositionContext]:
    sym = p.symbol or ""
    bbg = p.bbg_ticker or ""
    mv = _safe_float(p.market_value) or 0.0
    spot = _spot_for(bbg, underlyings)
    ovl = overlay_map.get(sym, {})
    beta = _underlying_field(underlyings, bbg, FIELDS["beta"])
    rsi = _underlying_field(underlyings, bbg, FIELDS["rsi_14d"])
    iv_3m = _underlying_field(underlyings, bbg, FIELDS["iv_3m"])
    iv_1m = _underlying_field(underlyings, bbg, FIELDS["iv_1m"])
    ed = _earnings_date(bbg, underlyings)
    d2e = _dte(ed)

    return PositionContext(
        bbg_ticker=bbg,
        underlying_symbol=sym,
        underlying_bbg_ticker=bbg,
        instrument_type="equity",
        right=None,
        strike=None,
        expiry=None,
        dte=None,
        quantity=_safe_float(p.quantity) or 0.0,
        multiplier=int(p.multiplier or 1),
        cost_basis=_safe_float(p.cost_basis),
        market_value=mv,
        pct_pnl=_safe_float(p.pct_pnl),
        spot=spot,
        delta=1.0,
        vega=0.0, theta=0.0, gamma=0.0,
        iv_1m=iv_1m, iv_3m=iv_3m,
        moneyness=None,
        pct_captured=_safe_float(p.pct_pnl),
        portfolio_pct_of_nav=(abs(mv) / nav) if nav else 0.0,
        has_long_stock=ovl.get("has_long_stock", False),
        has_short_calls=ovl.get("has_short_calls", False),
        has_short_puts=ovl.get("has_short_puts", False),
        has_long_protection=ovl.get("has_long_protection", False),
        rsi_14d=rsi,
        earnings_date=ed,
        days_to_earnings=d2e,
        beta=beta,
        iv_percentile=iv_pctl_by_ticker.get(bbg),
    )


def _build_option_context(
    p: Position,
    underlyings,
    options,
    overlay_map: dict[str, dict[str, bool]],
    iv_pctl_by_ticker: dict[str, float],
    nav: float,
) -> Optional[PositionContext]:
    sym = p.underlying_symbol or ""
    opt_ticker = p.bbg_ticker or ""
    under_ticker = p.underlying_bbg_ticker or ""
    mv = _safe_float(p.market_value) or 0.0
    spot = _spot_for(under_ticker, underlyings)
    strike = _safe_float(p.strike)
    right = p.right if isinstance(p.right, str) else None
    expiry = p.expiry
    dte = _dte(expiry) if isinstance(expiry, date) else None

    ovl = overlay_map.get(sym, {})
    beta = _underlying_field(underlyings, under_ticker, FIELDS["beta"])
    rsi = _underlying_field(underlyings, under_ticker, FIELDS["rsi_14d"])
    iv_3m = _underlying_field(underlyings, under_ticker, FIELDS["iv_3m"])
    iv_1m = _underlying_field(underlyings, under_ticker, FIELDS["iv_1m"])
    ed = _earnings_date(under_ticker, underlyings)
    d2e = _dte(ed)

    return PositionContext(
        bbg_ticker=opt_ticker,
        underlying_symbol=sym,
        underlying_bbg_ticker=under_ticker,
        instrument_type="option",
        right=right,
        strike=strike,
        expiry=expiry if isinstance(expiry, date) else None,
        dte=dte,
        quantity=_safe_float(p.quantity) or 0.0,
        multiplier=int(p.multiplier or 100),
        cost_basis=_safe_float(p.cost_basis),
        market_value=mv,
        pct_pnl=_safe_float(p.pct_pnl),
        spot=spot,
        delta=_greek_for(opt_ticker, options, "delta_mid"),
        vega=_greek_for(opt_ticker, options, "vega"),
        theta=_greek_for(opt_ticker, options, "theta"),
        gamma=_greek_for(opt_ticker, options, "gamma"),
        iv_1m=iv_1m, iv_3m=iv_3m,
        moneyness=_signed_moneyness(spot, strike, right),
        pct_captured=_safe_float(p.pct_pnl),
        portfolio_pct_of_nav=(abs(mv) / nav) if nav else 0.0,
        has_long_stock=ovl.get("has_long_stock", False),
        has_short_calls=ovl.get("has_short_calls", False),
        has_short_puts=ovl.get("has_short_puts", False),
        has_long_protection=ovl.get("has_long_protection", False),
        rsi_14d=rsi,
        earnings_date=ed,
        days_to_earnings=d2e,
        beta=beta,
        iv_percentile=iv_pctl_by_ticker.get(under_ticker),
    )
