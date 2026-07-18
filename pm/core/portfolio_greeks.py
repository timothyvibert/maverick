"""Portfolio-level greeks aggregation.

Conventions:
- Quantity carries sign: short option positions have negative quantity, so
  greek-times-quantity automatically flips sign for shorts. Don't double-flip.
- Multiplier (100 for options, 1 for equities) is applied EXPLICITLY here.
- Spot is the underlying's PX_LAST from the snapshot.
- Equity positions contribute to delta only; vega/theta/gamma are zero.

Output greeks (all in USD or USD-equivalent for non-US — FX translation is
a v2 concern):

- ``dollar_delta``:   $ exposure per $1 move in the underlying's spot
- ``dollar_vega``:    $ P&L per +1 vol point (1.0% IV) move
- ``dollar_theta``:   $ P&L per 1 calendar day passing
- ``dollar_gamma``:   $ change in dollar_delta per **1% move** in spot
                      (= qty * mult * gamma * spot). The Bloomberg ``GAMMA``
                      snapshot field is dDelta per 1% underlying move —
                      live-confirmed against the engine (ratio == S/100 on
                      names both above and below $100) — so dollarizing it
                      with spot yields a per-1% quantity, NOT per-$1. The
                      scenario section's engine dollar-gamma is per-$1; the
                      two are labeled distinctly and must not be compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pm.ingest.position_builder import Position


# Asset classes that contribute greek rows. Fund / ETF treated as equity
# (delta-one); cash and other are skipped.
_EQUITY_LIKE = ("equity", "fund_etf")


@dataclass
class PortfolioGreeks:
    """Aggregate portfolio greeks plus the per-position rows that fed them."""
    by_position: pd.DataFrame
    # cols: bbg_ticker, instrument_type ('equity' | 'option'),
    #       underlying_ticker, quantity, multiplier, spot,
    #       delta, vega, theta, gamma,
    #       dollar_delta, dollar_vega, dollar_theta, dollar_gamma

    totals: dict
    # keys: dollar_delta, dollar_vega, dollar_theta, dollar_gamma,
    #       net_long_options_count,
    #       net_short_options_count, coverage_ratio_by_underlying,
    #       greeks_coverage.
    # A dollar-greek total is None (never 0.0) when EVERY eligible row was
    # missing that greek — a zero from all-missing data is not a real zero.
    # greeks_coverage = {col: {"n_missing": int, "n_total": int}} counts the
    # missing rows per dollar greek so the UI can cue partial coverage
    # (delta is eligible on every greek-bearing row; gamma/vega/theta only on
    # options — an equity's 0.0 vega is genuine, not missing).

    warnings: list[str]


_BY_POSITION_COLS = [
    # position_id is the canonical Position key — it lets exposure / rollup views
    # join a greek row back to its holding (and its expiry) without re-deriving the
    # option ticker. bbg_ticker stays as the market identifier.
    "position_id",
    "bbg_ticker", "instrument_type", "underlying_ticker", "right",
    "quantity", "multiplier", "spot",
    "delta", "vega", "theta", "gamma",
    "dollar_delta", "dollar_vega", "dollar_theta", "dollar_gamma",
]


def _spot_from(under_snap: pd.DataFrame, ticker: str) -> float:
    if under_snap is None or under_snap.empty or ticker not in under_snap.index:
        return float("nan")
    if "PX_LAST" not in under_snap.columns:
        return float("nan")
    val = under_snap.loc[ticker, "PX_LAST"]
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return float("nan")


def _greek_from(opt_snap: pd.DataFrame, ticker: str, col: str) -> float:
    if opt_snap is None or opt_snap.empty or ticker not in opt_snap.index:
        return float("nan")
    if col not in opt_snap.columns:
        return float("nan")
    val = opt_snap.loc[ticker, col]
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return float("nan")


def compute_portfolio_greeks(
    positions: list[Position],
    underlying_snapshot: pd.DataFrame,
    option_snapshot: pd.DataFrame,
) -> PortfolioGreeks:
    """Combine positions with live snapshots into per-position and
    aggregate greeks.

    ``underlying_snapshot`` is indexed by underlying bbg_ticker (e.g.
    ``'AAPL US Equity'``); must contain a ``PX_LAST`` column. Either
    snapshot may be empty (BBG unavailable) — the function still returns a
    PortfolioGreeks with NaN-filled rows so the UI can render.

    ``option_snapshot`` is indexed by option bbg_ticker (e.g.
    ``'BX US 5/15/26 P105 Equity'``); must contain ``delta``, ``vega``,
    ``theta``, ``gamma`` columns.

    Cash and Other positions are skipped (no greeks). Fund / ETF rows
    contribute as equity (delta=1).
    """
    warnings: list[str] = []
    rows: list[dict] = []

    equity_like_positions = [p for p in positions if p.asset_class in _EQUITY_LIKE]
    option_positions = [p for p in positions if p.asset_class == "option"]

    # -- Equity / fund rows -------------------------------------------------
    for p in equity_like_positions:
        bbg = p.bbg_ticker or None
        qty = p.quantity
        mult = p.multiplier or 1
        spot = _spot_from(underlying_snapshot, bbg) if bbg else float("nan")

        if bbg and (underlying_snapshot is None or underlying_snapshot.empty
                    or bbg not in underlying_snapshot.index):
            warnings.append(f"No underlying snapshot for {bbg}.")
        elif bbg and np.isnan(spot):
            warnings.append(f"PX_LAST missing for {bbg}.")

        try:
            qty_f = float(qty) if qty is not None else float("nan")
        except (TypeError, ValueError):
            qty_f = float("nan")

        delta = 1.0
        vega = 0.0
        theta = 0.0
        gamma = 0.0
        dollar_delta = qty_f * spot
        dollar_vega = 0.0
        dollar_theta = 0.0
        dollar_gamma = 0.0

        rows.append({
            "position_id": p.position_id,
            "bbg_ticker": bbg,
            "instrument_type": "equity",
            "underlying_ticker": bbg,
            "right": None,
            "quantity": qty_f,
            "multiplier": int(mult),
            "spot": spot,
            "delta": delta,
            "vega": vega,
            "theta": theta,
            "gamma": gamma,
            "dollar_delta": dollar_delta,
            "dollar_vega": dollar_vega,
            "dollar_theta": dollar_theta,
            "dollar_gamma": dollar_gamma,
        })

    # -- Option rows --------------------------------------------------------
    for p in option_positions:
        opt_ticker = p.bbg_ticker
        under_ticker = p.underlying_bbg_ticker
        right = p.right
        qty = p.quantity
        mult = p.multiplier or 100

        spot = _spot_from(underlying_snapshot, under_ticker) if under_ticker else float("nan")
        if under_ticker and (underlying_snapshot is None or underlying_snapshot.empty
                              or under_ticker not in underlying_snapshot.index):
            warnings.append(f"No underlying snapshot for {under_ticker}.")
        elif under_ticker and np.isnan(spot):
            warnings.append(f"PX_LAST missing for {under_ticker}.")

        if option_snapshot is None or option_snapshot.empty or opt_ticker not in option_snapshot.index:
            warnings.append(f"No option snapshot for {opt_ticker}.")
            delta = vega = theta = gamma = float("nan")
        else:
            delta = _greek_from(option_snapshot, opt_ticker, "delta_mid")
            vega = _greek_from(option_snapshot, opt_ticker, "vega")
            theta = _greek_from(option_snapshot, opt_ticker, "theta")
            gamma = _greek_from(option_snapshot, opt_ticker, "gamma")
            for label, val in [("delta", delta), ("vega", vega),
                               ("theta", theta), ("gamma", gamma)]:
                if np.isnan(val):
                    warnings.append(f"{label} missing for {opt_ticker}.")

        try:
            qty_f = float(qty) if qty is not None else float("nan")
        except (TypeError, ValueError):
            qty_f = float("nan")

        dollar_delta = qty_f * mult * delta * spot
        dollar_vega = qty_f * mult * vega
        dollar_theta = qty_f * mult * theta
        dollar_gamma = qty_f * mult * gamma * spot  # $Δ per 1% spot move (BBG GAMMA is per-1%)

        rows.append({
            "position_id": p.position_id,
            "bbg_ticker": opt_ticker,
            "instrument_type": "option",
            "underlying_ticker": under_ticker,
            "right": right,
            "quantity": qty_f,
            "multiplier": int(mult),
            "spot": spot,
            "delta": delta,
            "vega": vega,
            "theta": theta,
            "gamma": gamma,
            "dollar_delta": dollar_delta,
            "dollar_vega": dollar_vega,
            "dollar_theta": dollar_theta,
            "dollar_gamma": dollar_gamma,
        })

    by_pos = pd.DataFrame(rows, columns=_BY_POSITION_COLS)

    # -- Totals -------------------------------------------------------------
    # Skipna sums, with one honesty rule: when every eligible row is missing a
    # greek, the total is None — an all-missing book must never render as a
    # confident $0. Coverage counts ride along so partial missingness can be
    # cued ("n of m missing") next to the summed number.
    option_rows = by_pos[by_pos["instrument_type"] == "option"] if not by_pos.empty else by_pos
    totals: dict = {}
    greeks_coverage: dict[str, dict] = {}
    for col in ("dollar_delta", "dollar_vega", "dollar_theta", "dollar_gamma"):
        eligible = by_pos if col == "dollar_delta" else option_rows
        n_total = int(len(eligible))
        n_missing = int(eligible[col].isna().sum()) if n_total else 0
        greeks_coverage[col] = {"n_missing": n_missing, "n_total": n_total}
        if n_total and n_missing == n_total:
            totals[col] = None
        else:
            totals[col] = float(by_pos[col].sum(skipna=True)) if not by_pos.empty else 0.0
    totals["greeks_coverage"] = greeks_coverage

    opts = by_pos[by_pos["instrument_type"] == "option"]
    totals["net_long_options_count"] = int((opts["quantity"] > 0).sum()) if not opts.empty else 0
    totals["net_short_options_count"] = int((opts["quantity"] < 0).sum()) if not opts.empty else 0

    # -- Coverage ratio (long stock vs short calls per underlying) ---------
    coverage: dict[str, float] = {}
    if equity_like_positions and option_positions:
        for symbol in {p.underlying_symbol for p in option_positions if p.underlying_symbol}:
            short_calls = [
                p for p in option_positions
                if p.underlying_symbol == symbol and p.right == "CALL"
                and p.quantity is not None and p.quantity < 0
            ]
            long_stock = [
                p for p in equity_like_positions
                if p.symbol == symbol and p.quantity is not None and p.quantity > 0
            ]
            if not short_calls or not long_stock:
                continue
            short_call_shares = float(sum(abs(p.quantity) * (p.multiplier or 100) for p in short_calls))
            long_stock_shares = float(sum(p.quantity for p in long_stock))
            if long_stock_shares > 0:
                coverage[str(symbol)] = short_call_shares / long_stock_shares
    totals["coverage_ratio_by_underlying"] = coverage

    return PortfolioGreeks(
        by_position=by_pos,
        totals=totals,
        warnings=warnings,
    )
