"""Loader for the holdings-extract xlsx files.

The extract is a two-sheet workbook (`Holdings`, `Trades`) emitted by an
upstream portfolio system. Filenames are `<source>_extract_YYYYMMDD_HHMMSS.xlsx`
(any lowercase-alphanumeric source prefix is accepted).

This module is the only piece of the codebase that reads the raw xlsx.
Downstream consumers (`pm.ingest.position_builder`, `pm.store`) take the
produced ``PortfolioExtract`` and never touch the file again.

Incoming headers are matched to the canonical field names below in precedence
order — exact, then an explicit alias map, then case/whitespace-insensitive,
then a bounded fuzzy near-miss (high threshold, abstains on ambiguity, logged).
Unrecognized extra columns are ignored; missing columns degrade gracefully
(load-bearing → affected rows skipped with a flag; optional → left blank), so a
real extract with a renamed/added/dropped column loads rather than failing.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-name normalization maps (raw header in xlsx -> snake_case)
# ---------------------------------------------------------------------------

HOLDINGS_COLUMN_MAP: dict[str, str] = {
    "Account": "account",
    "Position Date": "position_date",
    "Asset Class": "asset_class",
    "Instrument Type": "instrument_type",
    "Product Name": "product_name",
    "Underlying Ticker": "underlying_ticker",
    "Underlying Name": "underlying_name",
    "Underlying ISIN": "underlying_isin",
    "Underlying Issuer Country Code": "underlying_issuer_country_code",
    "Option Type": "option_type",
    "Option Expiration": "option_expiration",
    "Option Strike": "option_strike",
    "Ticker Final": "ticker_final",
    "CUSIP Final": "cusip_final",
    "ISIN Final": "isin_final",
    "Issuer Country Code Final": "issuer_country_code_final",
    "ISIN Country Code Final": "isin_country_code_final",
    "Listing Hint Country Code": "listing_hint_country_code",
    "Quantity": "quantity",
    "Valuation Price": "valuation_price",
    "Market Value": "market_value",
    "Cost Basis": "cost_basis",
    "Unrealized P&L": "unrealized_pnl",
    "Unrealized P&L %": "unrealized_pnl_pct",
    "Tax Lot Count": "tax_lot_count",
    "Option Contract Key": "option_contract_key",
}

TRADES_COLUMN_MAP: dict[str, str] = {
    "Account": "account",
    "Trade Date": "trade_date",
    "Buy/Sell": "buy_sell",
    "Option Lifecycle Action": "option_lifecycle_action",
    "Asset Class": "asset_class",
    "Instrument Type": "instrument_type",
    "Product Name": "product_name",
    "Underlying Ticker": "underlying_ticker",
    "Underlying Name": "underlying_name",
    "Underlying ISIN": "underlying_isin",
    "Underlying Issuer Country Code": "underlying_issuer_country_code",
    "Option Type": "option_type",
    "Option Expiration": "option_expiration",
    "Option Strike": "option_strike",
    "Ticker Final": "ticker_final",
    "CUSIP Final": "cusip_final",
    "ISIN Final": "isin_final",
    "Issuer Country Code Final": "issuer_country_code_final",
    "ISIN Country Code Final": "isin_country_code_final",
    "Listing Hint Country Code": "listing_hint_country_code",
    "Quantity": "quantity",
    "Principal Amount": "principal_amount",
    "Trade Type Code": "trade_type_code",
    "Cancel Code": "cancel_code",
    "Option Contract Key": "option_contract_key",
}

# Explicit alias map: alternative raw headers -> canonical snake field. Kept
# minimal and reviewable — only equivalences actually seen in real extracts. A
# genuinely new variant is caught by the fuzzy layer (and logged) until it is
# promoted to an explicit alias here.
ALIAS_MAP: dict[str, str] = {
    "Account Number": "account",
}

# Load-bearing canonical fields: a row cannot be built without them, so an
# absent column means the affected rows are skipped (see position_builder). Two
# are required for every row; the rest are required only for rows of one class.
LOAD_BEARING_ALWAYS: tuple[str, ...] = ("account", "asset_class")
LOAD_BEARING_BY_CLASS: dict[str, tuple[str, ...]] = {
    "option": ("option_contract_key", "underlying_ticker", "option_type",
               "option_strike", "option_expiration"),
    "equity": ("ticker_final",),
    "fund_etf": ("ticker_final",),
}

# High-impact optional: an absent column skips no rows but nulls a field used
# book-wide, so the in-app flag calls it out as urgent, like a load-bearing miss.
HIGH_IMPACT_OPTIONAL: dict[str, str] = {
    "quantity": "Quantity column missing — position sizes, structures, and P&L unavailable",
}

# Trades-sheet columns the trade-history join cannot run without. The builder
# stays deliberately silent when they are absent (see position_builder's
# _attach_trade_history early returns), so the loader is the one place the
# consequence gets said out loud.
TRADES_LOAD_BEARING: dict[str, str] = {
    "account": ("Trades 'Account' column missing — equity/fund trade history "
                "cannot be attributed to accounts (join disabled)"),
    "trade_date": ("Trades 'Trade Date' column missing — open dates, days held, "
                   "recency and roll reads all blank"),
    "ticker_final": ("Trades 'Ticker Final' column missing — equity/fund trade "
                     "history join disabled"),
    "option_contract_key": ("Trades 'Option Contract Key' column missing — "
                            "option trade history join disabled"),
}

# High-impact pair: without both, the open-date derivation (and everything on
# it: days held, the behaviour profile's lifecycle read) degrades book-wide.
TRADES_HIGH_IMPACT: dict[str, str] = {
    "buy_sell": ("Trades 'Buy/Sell' column missing — open-date derivation and "
                 "the behaviour profile degrade"),
    "option_lifecycle_action": ("Trades 'Option Lifecycle Action' column missing "
                                "— open-date derivation and the behaviour "
                                "profile degrade"),
}

# Prefix marking a flag the status bar surfaces as urgent (amber): a missing
# load-bearing column or a high-impact optional.
URGENT_FLAG = "⚠"  # ⚠

# Fuzzy header matching is a last resort: high cutoff, one-to-one, and it
# abstains when two canonical targets are plausible (financial fields are
# unforgiving — e.g. Unrealized P&L vs Unrealized P&L % must never collapse).
_FUZZY_CUTOFF = 0.9


# Asset Class raw value -> canonical
ASSET_CLASS_MAP: dict[str, str] = {
    "Option": "option",
    "Equity": "equity",
    "Fund / ETF": "fund_etf",
    "Cash": "cash",
    "Other": "other",
}

# Option Type raw -> canonical CALL/PUT
OPTION_TYPE_MAP: dict[str, str] = {
    "Call": "CALL",
    "Put": "PUT",
    "CALL": "CALL",
    "PUT": "PUT",
}

_HOLDINGS_NUMERIC_COLS = (
    "quantity", "valuation_price", "market_value", "cost_basis",
    "unrealized_pnl", "unrealized_pnl_pct", "tax_lot_count", "option_strike",
)
_TRADES_NUMERIC_COLS = (
    "quantity", "principal_amount", "trade_type_code", "option_strike",
)
_HOLDINGS_DATE_COLS = ("position_date", "option_expiration")
_TRADES_DATE_COLS = ("trade_date", "option_expiration")

_FILENAME_RE = re.compile(r"^[a-z0-9]+_extract_(\d{8})_(\d{6})\.xlsx$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PortfolioExtract:
    """Result of loading a single extract xlsx file."""
    extract_ts: datetime
    source_path: Path
    holdings: pd.DataFrame
    trades: pd.DataFrame
    accounts: list[str]
    parse_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_latest_extract(data_dir: Path) -> Path | None:
    """Return the newest `<source>_extract_YYYYMMDD_HHMMSS.xlsx` in ``data_dir``
    by the filename timestamp. Returns None if no matching file exists.

    Filename timestamps (not mtimes) are used because xlsx files are often
    copied around with stale mtimes.
    """
    if not data_dir.exists():
        return None
    candidates: list[tuple[datetime, Path]] = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        ts = _parse_filename_ts(path.name)
        if ts is not None:
            candidates.append((ts, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _parse_filename_ts(name: str) -> datetime | None:
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main load
# ---------------------------------------------------------------------------

def load_portfolio_extract(path: Path) -> PortfolioExtract:
    """Load the Holdings + Trades sheets, normalize headers + dtypes, and
    accumulate parse warnings.

    Degrades gracefully rather than failing on schema drift: headers are
    matched by precedence (exact / alias / case-insensitive / bounded fuzzy),
    unknown extra columns are ignored, and missing columns are flagged by tier
    (load-bearing → affected rows skipped; optional → left blank). Never raises
    for missing columns or individual bad rows — those become warnings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Extract not found: {path}")

    extract_ts = _parse_filename_ts(path.name)
    if extract_ts is None:
        extract_ts = datetime.fromtimestamp(path.stat().st_mtime)

    warnings: list[str] = []

    holdings = _load_sheet(
        path, sheet_name="Holdings",
        column_map=HOLDINGS_COLUMN_MAP,
        numeric_cols=_HOLDINGS_NUMERIC_COLS,
        date_cols=_HOLDINGS_DATE_COLS,
        warnings=warnings,
    )
    trades = _load_sheet(
        path, sheet_name="Trades",
        column_map=TRADES_COLUMN_MAP,
        numeric_cols=_TRADES_NUMERIC_COLS,
        date_cols=_TRADES_DATE_COLS,
        warnings=warnings,
    )

    holdings = _normalize_categoricals(holdings)
    trades = _normalize_categoricals(trades)

    _flag_columns(holdings, "Holdings", HOLDINGS_COLUMN_MAP, warnings)
    _flag_columns(trades, "Trades", TRADES_COLUMN_MAP, warnings)

    _row_level_warnings(holdings, warnings)

    accounts: list[str]
    if "account" in holdings.columns:
        accounts = sorted(holdings["account"].dropna().astype(str).unique().tolist())
    else:
        accounts = []

    return PortfolioExtract(
        extract_ts=extract_ts,
        source_path=path,
        holdings=holdings,
        trades=trades,
        accounts=accounts,
        parse_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Per-sheet helpers
# ---------------------------------------------------------------------------

def _norm_header(s: object) -> str:
    """Lowercase, collapse internal whitespace, strip — for case/whitespace-
    insensitive header matching."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _normalize_headers(
    incoming: list[str], column_map: dict[str, str], sheet_name: str,
    warnings: list[str],
) -> dict[str, str]:
    """Map incoming raw headers to canonical snake field names, in precedence
    order: exact → alias → case/whitespace-insensitive → bounded fuzzy.

    Returns the rename dict ``{incoming_header: canonical_snake}``. Headers that
    resolve to nothing are left out (ignored as unknown extras). Fuzzy matches
    are appended to *warnings* and logged; an ambiguous near-miss (two plausible
    targets) is flagged and left unmatched rather than guessed. One incoming
    header maps to at most one canonical field, and vice versa.
    """
    norm_lookup: dict[str, str] = {}
    for raw_key, snake in {**column_map, **ALIAS_MAP}.items():
        norm_lookup.setdefault(_norm_header(raw_key), snake)

    rename: dict[str, str] = {}
    taken: set[str] = set()
    unresolved: list[str] = []

    for header in incoming:
        if header in column_map:
            snake = column_map[header]
        elif header in ALIAS_MAP:
            snake = ALIAS_MAP[header]
        elif _norm_header(header) in norm_lookup:
            snake = norm_lookup[_norm_header(header)]
        else:
            unresolved.append(header)
            continue
        rename[header] = snake
        taken.add(snake)

    # Fuzzy — last resort, only against canonical names still unclaimed.
    remaining = {snake: _norm_header(raw_key)
                 for raw_key, snake in column_map.items() if snake not in taken}
    for header in unresolved:
        if not remaining:
            break
        matches = difflib.get_close_matches(
            _norm_header(header), list(remaining.values()), n=2, cutoff=_FUZZY_CUTOFF,
        )
        if len(matches) == 1:
            snakes = [s for s, n in remaining.items() if n == matches[0]]
            if len(snakes) == 1:                       # unambiguous one-to-one
                snake = snakes[0]
                rename[header] = snake
                remaining.pop(snake, None)
                warnings.append(
                    f"{sheet_name}: fuzzy-matched header '{header}' → '{snake}' — review."
                )
                logger.warning("Header fuzzy match in %s: %r → %r", sheet_name, header, snake)
                continue
        if len(matches) >= 2:                          # ambiguous → never guess
            warnings.append(
                f"{sheet_name}: header '{header}' is an ambiguous near-miss "
                f"({matches}) — left unmatched."
            )
            logger.warning(
                "Ambiguous header near-miss in %s: %r ≈ %s; left unmatched",
                sheet_name, header, matches,
            )
        # 0 matches (or ambiguous) → unknown extra, ignored silently
    return rename


def _load_sheet(
    path: Path,
    *,
    sheet_name: str,
    column_map: dict[str, str],
    numeric_cols: tuple[str, ...],
    date_cols: tuple[str, ...],
    warnings: list[str],
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name)

    rename = _normalize_headers(list(raw.columns), column_map, sheet_name, warnings)
    df = raw.rename(columns=rename).copy()

    # Coerce only the columns that resolved — a missing column is tolerated here
    # and flagged by tier in _flag_columns; downstream reads it as absent/None.
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col == "quantity":
            mask_dash = df[col].astype(object).apply(
                lambda v: isinstance(v, str) and v.strip() == "-"
            )
            if mask_dash.any():
                warnings.append(
                    f"{sheet_name}: {int(mask_dash.sum())} row(s) had "
                    f"Quantity '-' (coerced to NaN; routed via asset_class)."
                )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in date_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].apply(lambda v: v.date() if isinstance(v, pd.Timestamp) else None)

    return df


def _flag_columns(
    df: pd.DataFrame, sheet_name: str, column_map: dict[str, str], warnings: list[str],
) -> None:
    """Flag canonical columns that did not resolve, by tier. Holdings and
    Trades each carry load-bearing + high-impact tiers (urgent, with an
    affected-row count where one applies); every other absent column is an
    ordinary optional rolled into a single note.
    One summary per missing column — never one note per skipped row (the builder
    stays silent on absent columns; see position_builder)."""
    absent = set(column_map.values()) - set(df.columns)
    if not absent:
        return

    n_rows = len(df)
    by_class = df["asset_class"].value_counts() if "asset_class" in df.columns else None
    flagged: set[str] = set()

    if sheet_name == "Holdings":
        for col in LOAD_BEARING_ALWAYS:
            if col in absent:
                warnings.append(
                    f"{URGENT_FLAG} {sheet_name}: '{col}' column missing — "
                    f"{n_rows} row(s) unbuildable (skipped)."
                )
                flagged.add(col)
        if by_class is not None:
            for cls, cols in LOAD_BEARING_BY_CLASS.items():
                n_cls = int(by_class.get(cls, 0))
                for col in cols:
                    if col in absent and col not in flagged:
                        warnings.append(
                            f"{URGENT_FLAG} {sheet_name}: '{col}' column missing — "
                            f"{n_cls} {cls} row(s) skipped."
                        )
                        flagged.add(col)
        for col, message in HIGH_IMPACT_OPTIONAL.items():
            if col in absent and col not in flagged:
                warnings.append(f"{URGENT_FLAG} {sheet_name}: {message}.")
                flagged.add(col)

    elif sheet_name == "Trades":
        for col, message in TRADES_LOAD_BEARING.items():
            if col in absent:
                warnings.append(f"{URGENT_FLAG} {message} ({n_rows} trade row(s)).")
                flagged.add(col)
        for col, message in TRADES_HIGH_IMPACT.items():
            if col in absent and col not in flagged:
                warnings.append(f"{URGENT_FLAG} {message}.")
                flagged.add(col)

    optional_absent = sorted(absent - flagged)
    if optional_absent:
        warnings.append(
            f"{sheet_name}: optional column(s) absent — "
            f"{', '.join(optional_absent)} (left blank)."
        )


def _normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    if "asset_class" in df.columns:
        df["asset_class"] = df["asset_class"].map(
            lambda v: ASSET_CLASS_MAP.get(v, v) if isinstance(v, str) else v
        )
    if "instrument_type" in df.columns:
        df["instrument_type"] = df["instrument_type"].map(
            lambda v: ASSET_CLASS_MAP.get(v, v) if isinstance(v, str) else v
        )
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].map(
            lambda v: OPTION_TYPE_MAP.get(v, v) if isinstance(v, str) else v
        )
    return df


def _row_level_warnings(holdings: pd.DataFrame, warnings: list[str]) -> None:
    """Soft warnings for unexpected NaN combinations."""
    if "asset_class" not in holdings.columns:
        return

    options = holdings[holdings["asset_class"] == "option"]
    if "underlying_ticker" in options.columns:
        bad = options[options["underlying_ticker"].isna()]
        if not bad.empty:
            warnings.append(
                f"Holdings: {len(bad)} option row(s) have NaN underlying_ticker."
            )
    if "option_contract_key" in options.columns:
        bad = options[options["option_contract_key"].isna()]
        if not bad.empty:
            warnings.append(
                f"Holdings: {len(bad)} option row(s) have NaN option_contract_key."
            )

    non_cash_non_other = holdings[~holdings["asset_class"].isin(["cash", "other"])]
    if "ticker_final" in non_cash_non_other.columns:
        bad = non_cash_non_other[non_cash_non_other["ticker_final"].isna()]
        if not bad.empty:
            warnings.append(
                f"Holdings: {len(bad)} non-cash/non-other row(s) have NaN ticker_final."
            )
