"""Portfolio-level snapshot assembly. Combines a list of
``Position`` records with live Bloomberg data into a single
``PortfolioSnapshot`` ready for display.

V1 takes ``list[Position]`` directly. Cash / Other positions are skipped (no BBG ticker).
Funds / ETFs are fetched the same as equities — missing data tolerated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from datetime import date

from pm.core.bloomberg_client import (
    OPTION_SNAPSHOT_FIELDS,
    UNDERLYING_FIELDS,
    fetch_option_chain,
    fetch_option_snapshots,
    fetch_security_identity,
    fetch_spx_betas,
    fetch_underlying_snapshots,
)
from pm.core.ticker_utils import match_option_ticker
from pm.ingest.extract_loader import URGENT_FLAG

# SPX-relative beta columns merged onto the underlying snapshot for the exposure
# view (sourced via a separate override-aware pull — see bloomberg_client.fetch_spx_betas).
_SPX_BETA_COLS = ("EQY_BETA", "EQY_RAW_BETA")
from pm.ingest.position_builder import Position
from pm.core import clock


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
        # Venue-resolution net: the extract's country codes are UNTRUSTED, so a
        # constructed ticker can be dead (no such listing) or — worse — resolve
        # to a different security on the wrong venue. Recover what can be
        # identity-validated, flag the rest loudly. Runs BEFORE the SPX-beta
        # pull so recovered names get betas under their live ticker.
        under_df = _resolve_missing_underlyings(positions, under_df, tickers, warnings)
        tickers = _unique_underlying_tickers(positions)   # pick up any re-keys
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
            # An already-expired holding can never match a listed chain — tag it
            # honestly and skip its chain fetch instead of mis-diagnosing it as
            # a ticker-resolution failure.
            missing_options = _flag_expired_options(positions, missing_options, warnings)
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


# ---------------------------------------------------------------------------
# Venue resolution for underlying tickers (defense-in-depth for untrusted
# extract country codes)
# ---------------------------------------------------------------------------

def _split_venue(ticker: str):
    """('RACE', 'NA') from 'RACE NA Equity'; (None, None) when not that shape."""
    parts = ticker.rsplit(" ", 2)
    if len(parts) != 3 or parts[2] != "Equity":
        return None, None
    return parts[0], parts[1]


def _reps_by_underlying(positions: list[Position]) -> dict[str, list[Position]]:
    """Positions keyed by the underlying ticker they ride on (equity/fund rows
    by their own bbg_ticker; option rows by underlying_bbg_ticker)."""
    reps: dict[str, list[Position]] = {}
    for p in positions:
        if p.asset_class in _UNDERLYING_ASSET_CLASSES and p.bbg_ticker:
            reps.setdefault(p.bbg_ticker, []).append(p)
        elif p.asset_class == "option" and p.underlying_bbg_ticker:
            reps.setdefault(p.underlying_bbg_ticker, []).append(p)
    return reps


def _expected_isin(reps: list[Position]):
    """The extract's own identity for the name (first non-empty ISIN)."""
    for p in reps:
        isin = getattr(p, "isin", None)
        if isinstance(isin, str) and isin.strip():
            return isin.strip().upper()
    return None


def _name_tokens(s) -> set:
    import re
    return {t for t in re.split(r"[^A-Z0-9]+", str(s or "").upper()) if len(t) >= 4}


def _identity_basis(expected_isin, position_name, ident_row: dict):
    """Why a probed ticker may be ACCEPTED as this name — or None.

    Exact ISIN equality against the extract's own ISIN is the strong basis;
    name-token agreement is the audited fallback only when the extract carries
    no ISIN. Presence of data alone is never enough — a wrong-venue ticker can
    resolve to a different instrument with a matching root."""
    isin = ident_row.get("ID_ISIN")
    if expected_isin:
        if isinstance(isin, str) and isin.strip().upper() == expected_isin:
            return "ISIN match"
        return None
    name = ident_row.get("NAME")
    if position_name and isinstance(name, str) and (_name_tokens(position_name) & _name_tokens(name)):
        return "name match (no ISIN in extract)"
    return None


def _ident_row(df, ticker: str) -> dict:
    if df is None or getattr(df, "empty", True) or ticker not in df.index:
        return {}
    row = df.loc[ticker]
    return {k: (None if pd.isna(row.get(k)) else row.get(k))
            for k in ("PX_LAST", "NAME", "ID_ISIN", "CRNCY")}


def _describe_ident(row: dict) -> str:
    if not row or all(v is None for v in row.values()):
        return "returned nothing"
    if row.get("PX_LAST") is None:
        return (f"resolved to {row.get('NAME')!r} (ISIN {row.get('ID_ISIN')}), no live price")
    return f"resolved to {row.get('NAME')!r} (ISIN {row.get('ID_ISIN')}), price {row.get('PX_LAST')}"


def _resolve_missing_underlyings(
    positions: list[Position],
    under_df: pd.DataFrame,
    tickers: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    """Recover mis-venued underlying tickers; loudly flag what can't be recovered.

    The extract's country code is untrusted input, so the constructed ticker can
    (a) resolve to nothing, or (b) resolve to a DIFFERENT security on that venue
    (observed live: a wrong-venue suffix on a NYSE root returned a defunct local
    line with the same root — identity fields populated, price dead). Therefore:

    * a ticker is suspect when it has no usable PX_LAST, or when its probed
      ISIN contradicts the extract's own ISIN for the name (even with a price);
    * recovery tries the ``<root> US Equity`` variant (the book's symbology is
      US) and ACCEPTS it only on identity validation — exact ISIN equality with
      the extract, else name-token agreement when the extract has no ISIN —
      plus a live price; never blind acceptance;
    * every re-key is audited in place (``provisional_bbg_ticker`` on
      equity/fund rows, ``provisional_underlying_bbg_ticker`` on option rows —
      the option's OWN constructed ticker is then recovered by the option-chain
      pass below, which enumerates the chain on the re-keyed underlier);
    * anything still unresolved gets an URGENT per-name flag naming the ticker
      tried, what came back, and the market value at stake — never a silent
      blank; a confirmed wrong-security row is dropped from the snapshot so a
      wrong price can never flow into signals.

    Mutates ``positions`` and ``warnings``; returns the updated frame.
    """
    reps_map = _reps_by_underlying(positions)

    def _px(t):
        if under_df.empty or t not in under_df.index:
            return None
        v = under_df.loc[t].get("PX_LAST")
        return None if pd.isna(v) else v

    dead = [t for t in tickers if _px(t) is None]
    foreign = [t for t in tickers if (_split_venue(t)[1] or "US") != "US"]
    audit_set = sorted(set(dead) | set(foreign))
    if not audit_set:
        return under_df

    ident = fetch_security_identity(audit_set)

    # What each suspect's CONSTRUCTED ticker actually is, per the identity probe.
    suspects: list[tuple[str, dict, bool]] = []   # (ticker, ident row, wrong_security)
    for t in audit_set:
        reps = reps_map.get(t) or []
        if not reps:
            continue
        row = _ident_row(ident, t)
        exp_isin = _expected_isin(reps)
        probed_isin = row.get("ID_ISIN")
        wrong = bool(exp_isin and isinstance(probed_isin, str)
                     and probed_isin.strip().upper() != exp_isin)
        if t in dead or wrong:
            suspects.append((t, row, wrong))

    if not suspects:
        return under_df

    variants = {}
    for t, _row, _wrong in suspects:
        root, suffix = _split_venue(t)
        if root and suffix != "US":
            variants[t] = f"{root} US Equity"
    vident = fetch_security_identity(sorted(set(variants.values())))

    recovered: dict[str, str] = {}
    drop_rows: list[str] = []
    for t, row, wrong in suspects:
        reps = reps_map[t]
        exp_isin = _expected_isin(reps)
        display = next((p.name for p in reps if p.name), None) \
            or reps[0].underlying_symbol or reps[0].symbol or t
        mv = sum(abs(p.market_value or 0.0) for p in reps)
        variant = variants.get(t)
        vrow = _ident_row(vident, variant) if variant else {}
        basis = _identity_basis(exp_isin, display, vrow) if variant else None

        if variant and basis and vrow.get("PX_LAST") is not None:
            for p in reps:
                if p.asset_class in _UNDERLYING_ASSET_CLASSES and p.bbg_ticker == t:
                    p.provisional_bbg_ticker = p.bbg_ticker
                    p.bbg_ticker = variant
                elif p.asset_class == "option" and p.underlying_bbg_ticker == t:
                    p.provisional_underlying_bbg_ticker = p.underlying_bbg_ticker
                    p.underlying_bbg_ticker = variant
            recovered[t] = variant
            drop_rows.append(t)
            warnings.append(
                f"Underlying re-keyed {t} -> {variant} ({basis}; constructed ticker "
                f"{_describe_ident(row)}); original kept for audit."
            )
            continue

        tried = (f"tried {variant}: {_describe_ident(vrow)}" if variant
                 else "no venue variant to try (already US-suffixed)")
        warnings.append(
            f"{URGENT_FLAG} Market data unresolved for {display} ({t} "
            f"{_describe_ident(row)}; {tried}); MV {mv:,.0f} affected — "
            f"market data, signals and alerts are dead on this name."
        )
        if wrong:
            # Never let a wrong security's numbers flow into signals: better an
            # honest all-NaN (stale) row than a confidently wrong price.
            drop_rows.append(t)

    if drop_rows:
        under_df = under_df.drop(index=[t for t in drop_rows if t in under_df.index],
                                 errors="ignore")
    if recovered:
        refetched = fetch_underlying_snapshots(sorted(set(recovered.values())))
        under_df = refetched if under_df.empty else pd.concat([under_df, refetched])
        for old, new in recovered.items():
            if new not in under_df.index or pd.isna(under_df.loc[new].get("PX_LAST")):
                warnings.append(
                    f"{URGENT_FLAG} Re-keyed underlying {new} (from {old}) still "
                    f"returned no market data — signals stay stale on this name."
                )
    return under_df


def _flag_expired_options(
    positions: list[Position],
    missing_options: list[str],
    warnings: list[str],
    today=None,
) -> list[str]:
    """Split already-expired holdings out of the missing-option set BEFORE the
    chain pass: a listed chain never contains an expired contract, so the pass
    would burn one chain fetch per name and mis-diagnose the row as a ticker-
    resolution failure ('unresolved after OPT_CHAIN'). Emit an honest expired
    tag instead. Returns the still-live missing tickers."""
    today = today or clock.today()
    by_ticker: dict[str, Position] = {}
    for p in positions:
        if p.asset_class == "option" and p.bbg_ticker in missing_options:
            by_ticker.setdefault(p.bbg_ticker, p)
    live: list[str] = []
    for t in missing_options:
        p = by_ticker.get(t)
        if p is not None and p.expiry is not None and p.expiry < today:
            days = (today - p.expiry).days
            name = p.underlying_bbg_ticker or p.underlying_symbol or p.bbg_ticker
            warnings.append(
                f"expired option still on the book: {name} {p.expiry.isoformat()} "
                f"{p.right or '?'} {p.strike if p.strike is not None else '?'} "
                f"({days}d past expiry) — no market data; pending removal from the extract."
            )
            continue
        live.append(t)
    return live


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
                constructed=old_ticker,
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
