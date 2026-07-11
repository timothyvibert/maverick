"""Portfolio-level snapshot assembly. Combines a list of
``Position`` records with live Bloomberg data into a single
``PortfolioSnapshot`` ready for display.

V1 takes ``list[Position]`` directly. Cash / Other positions are skipped (no BBG ticker).
Funds / ETFs are fetched the same as equities — missing data tolerated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pm.core.bloomberg_client import (
    OPTION_SNAPSHOT_FIELDS,
    UNDERLYING_FIELDS,
    fetch_option_chain,
    fetch_option_snapshots,
    fetch_spx_betas,
    fetch_underlying_snapshots,
)
from pm.core.ticker_utils import match_option_ticker

# SPX-relative beta columns merged onto the underlying snapshot for the exposure
# view (sourced via a separate override-aware pull — see bloomberg_client.fetch_spx_betas).
_SPX_BETA_COLS = ("EQY_BETA", "EQY_RAW_BETA")
from pm.ingest.position_builder import Position


# Asset classes that have a tradeable underlying we want a BBG row for.
_UNDERLYING_ASSET_CLASSES = ("equity", "fund_etf")


@dataclass
class PortfolioSnapshot:
    """Snapshot bundle. v0.3 has underlying + option data."""
    underlyings: pd.DataFrame
    # index = underlying bbg_ticker, cols = UNDERLYING_FIELDS + 'security_name'
    #         + SPX-relative betas 'EQY_BETA' / 'EQY_RAW_BETA' (for the exposure view)
    options: pd.DataFrame
    # index = option bbg_ticker, cols = OPTION_SNAPSHOT_FIELDS + canonical greek cols
    #         + 'style' ('American'/'European', for the scenario pricing adapter)
    fetch_warnings: list[str] = field(default_factory=list)
    bloomberg_available: bool = False


def fetch_portfolio_snapshot(
    positions: list[Position],
    bloomberg_available: bool,
) -> PortfolioSnapshot:
    """Fetch live snapshots for every unique underlying and every option
    in ``positions``. Routes to a no-op result when
    ``bloomberg_available`` is False so the rest of the app can render
    gracefully without a Terminal.
    """
    if not bloomberg_available:
        return PortfolioSnapshot(
            underlyings=_empty_underlyings_df(),
            options=_empty_options_df(),
            fetch_warnings=["Bloomberg unavailable — snapshot skipped."],
            bloomberg_available=False,
        )

    warnings: list[str] = []

    # ---- Underlyings -----------------------------------------------------
    tickers = _unique_underlying_tickers(positions)
    if tickers:
        under_df = fetch_underlying_snapshots(tickers)
        # Missing-underlying detection runs on the batched fields only (below),
        # BEFORE the additive SPX-beta columns are merged, so that warning set is
        # unchanged by this feature.
        missing_underlyings: list[str] = []
        if not under_df.empty:
            for t in tickers:
                if t not in under_df.index:
                    missing_underlyings.append(t)
                    continue
                if under_df.loc[t].isna().all():
                    missing_underlyings.append(t)
        else:
            missing_underlyings = list(tickers)
        if missing_underlyings:
            warnings.append(
                f"BBG returned no data for {len(missing_underlyings)} "
                f"underlying ticker(s): {missing_underlyings[:5]}"
                + ("..." if len(missing_underlyings) > 5 else "")
            )
        # Merge the SPX-relative betas (separate override-aware pull) onto the
        # underlying snapshot keyed by ticker, so the exposure view reads one
        # coherent benchmark. Kept apart from the batched pull above so the SPX
        # override never touches the default BETA_ADJ_OVERRIDABLE that pull returns.
        spx_betas = fetch_spx_betas(tickers)
        for col in _SPX_BETA_COLS:
            under_df[col] = (spx_betas[col].reindex(under_df.index)
                             if col in getattr(spx_betas, "columns", []) else pd.NA)
    else:
        under_df = _empty_underlyings_df()
        warnings.append("No underlyings to fetch.")

    # ---- Options ---------------------------------------------------------
    option_tickers = [
        p.bbg_ticker for p in positions
        if p.asset_class == "option" and p.bbg_ticker
    ]
    option_tickers = sorted(set(option_tickers))

    if option_tickers:
        opts_df = fetch_option_snapshots(option_tickers)
        missing_options = _missing_option_tickers(opts_df, option_tickers)
        if missing_options:
            # A held option's ticker is built from its EQUITY root, which is wrong
            # for names whose option root differs (NESN SW -> NES1 SW). Enumerate
            # each missing name's listed chain once, re-key the leg to the true
            # ticker, and re-fetch — so its greeks/IV/mark populate instead of
            # reading as an all-NaN gap. Names that still don't resolve keep their
            # best-effort ticker and get a specific warning.
            opts_df = _resolve_missing_options(
                positions, opts_df, missing_options, warnings,
            )
    else:
        opts_df = _empty_options_df()

    return PortfolioSnapshot(
        underlyings=under_df,
        options=opts_df,
        fetch_warnings=warnings,
        bloomberg_available=True,
    )


def _missing_option_tickers(opts_df: pd.DataFrame, option_tickers: list[str]) -> list[str]:
    """Option tickers BBG returned no data for — absent from the frame or an
    all-NaN row (the shape an unresolved constructed ticker produces)."""
    if opts_df.empty:
        return list(option_tickers)
    missing: list[str] = []
    for t in option_tickers:
        if t not in opts_df.index or opts_df.loc[t].isna().all():
            missing.append(t)
    return missing


def _unresolved_option_warning(p: Position) -> str:
    exp = p.expiry.isoformat() if p.expiry else "?"
    right = p.right or "?"
    strike = p.strike if p.strike is not None else "?"
    name = p.underlying_bbg_ticker or p.underlying_symbol or p.bbg_ticker
    return f"unresolved after OPT_CHAIN: {name} {exp} {right} {strike}"


def _resolve_missing_options(
    positions: list[Position],
    opts_df: pd.DataFrame,
    missing_options: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    """Recover held options whose constructed (equity-root) ticker didn't resolve.

    For each missing option ticker: enumerate its underlier's listed chain once
    (cached per underlier), match on (expiry, strike, right), and on a hit re-key
    every position carrying that ticker to the true listed string and re-fetch it.
    Names that don't resolve keep their best-effort ticker and get a specific
    warning (never a silent all-NaN, never a skip). Mutates ``Position.bbg_ticker``
    in place — the single-source key the snapshot index and every downstream
    lookup read — and records the original on ``provisional_bbg_ticker``. Returns
    the updated options frame (re-indexed to the re-keyed tickers).
    """
    missing_set = set(missing_options)
    reps_by_ticker: dict[str, list[Position]] = {}
    for p in positions:
        if p.asset_class == "option" and p.bbg_ticker in missing_set:
            reps_by_ticker.setdefault(p.bbg_ticker, []).append(p)

    chain_cache: dict[str, list[str]] = {}
    resolved: dict[str, str] = {}   # old constructed ticker -> true listed ticker
    for old_ticker in missing_options:
        reps = reps_by_ticker.get(old_ticker)
        if not reps:
            continue
        p0 = reps[0]
        underlier = p0.underlying_bbg_ticker
        canonical = None
        if underlier:
            if underlier not in chain_cache:
                chain_cache[underlier] = fetch_option_chain(underlier)
            canonical = match_option_ticker(
                chain_cache[underlier], p0.expiry, p0.strike, p0.right,
            )
        if not canonical or canonical == old_ticker:
            warnings.append(_unresolved_option_warning(p0))
            continue
        for p in reps:
            p.provisional_bbg_ticker = p.bbg_ticker
            p.bbg_ticker = canonical
        resolved[old_ticker] = canonical

    if not resolved:
        return opts_df

    refetched = fetch_option_snapshots(sorted(set(resolved.values())))
    opts_df = opts_df.drop(index=[t for t in resolved if t in opts_df.index],
                           errors="ignore")
    opts_df = pd.concat([opts_df, refetched])

    # A ticker that matched the chain but still returns no snapshot data (rare):
    # flag it too, so a re-keyed-but-empty leg is never a silent gap.
    for old_ticker, new_ticker in resolved.items():
        if new_ticker not in opts_df.index or opts_df.loc[new_ticker].isna().all():
            warnings.append(_unresolved_option_warning(reps_by_ticker[old_ticker][0]))
    return opts_df


def _unique_underlying_tickers(positions: list[Position]) -> list[str]:
    """Sorted union of:
      - bbg_ticker on equity / fund_etf positions, and
      - underlying_bbg_ticker on option positions.
    Drops empty / None values.
    """
    tickers: set[str] = set()
    for p in positions:
        if p.asset_class in _UNDERLYING_ASSET_CLASSES:
            if p.bbg_ticker:
                tickers.add(p.bbg_ticker)
        elif p.asset_class == "option":
            if p.underlying_bbg_ticker:
                tickers.add(p.underlying_bbg_ticker)
    return sorted(tickers)


def _empty_underlyings_df() -> pd.DataFrame:
    cols = ["security_name"] + list(UNDERLYING_FIELDS) + list(_SPX_BETA_COLS)
    return pd.DataFrame(columns=cols)


def _empty_options_df() -> pd.DataFrame:
    """Columns mirror what fetch_option_snapshots produces (sans 'security'
    since it's the index).
    """
    cols = [
        "BID", "ASK", "PX_MID", "PX_LAST", "IVOL_MID", "IVOL",
        "DAYS_TO_EXPIRATION", "DAYS_EXPIRE", "OPT_STRIKE_PX", "OPT_PUT_CALL",
        "DELTA_MID_RT", "THETA", "THETA_MID", "GAMMA", "VEGA", "RHO",
        "OPEN_INT", "PX_VOLUME", "OPTION_EXERCISE_TYPE_REALTIME",
        "dte", "delta_mid", "theta", "gamma", "vega", "rho", "iv_mid",
        "oi", "volume", "style",
    ]
    return pd.DataFrame(columns=cols)
