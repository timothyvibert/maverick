"""Build the canonical ``Position`` record list from an ``PortfolioExtract``.

A ``Position`` is the canonical normalized holdings row. It carries
everything the downstream schema-coupling modules need
(``portfolio_snapshot``, ``position_context``, ``portfolio_greeks``,
``portfolio_diagnostics``) without exposing the raw xlsx column layout.

This module is also responsible for constructing Bloomberg-format
tickers (e.g. ``'AAPL US Equity'``, ``'AAPL US 1/21/28 C300 Equity'``)
and joining each Holdings row to its Trades-sheet history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from pm.core.ticker_utils import construct_option_ticker
from pm.ingest.extract_loader import URGENT_FLAG, PortfolioExtract
from pm.core import clock


# ---------------------------------------------------------------------------
# Country code -> Bloomberg exchange suffix
# ---------------------------------------------------------------------------
# Keyed by ISO-3166 alpha-2 codes as they appear in the extract's
# `Issuer Country Code Final` / `Listing Hint Country Code` columns.
# Unknown codes fall back to "US" and a parse warning is emitted; that is
# acceptable for V1 since the sample has only one non-US row and
# exceptions accumulate empirically as new names appear.

COUNTRY_TO_BBG_SUFFIX: dict[str, str] = {
    "US": "US",
    "CH": "SW",  # Switzerland (SIX)
    "DE": "GY",  # Germany (XETRA)
    "NL": "NA",  # Netherlands (Euronext Amsterdam)
    "GB": "LN",  # UK (LSE)
    "FR": "FP",  # France (Euronext Paris)
    "IT": "IM",  # Italy (Borsa Italiana)
    "ES": "SM",  # Spain (BME)
    "SE": "SS",  # Sweden (Nasdaq Stockholm)
    "DK": "DC",  # Denmark
    "NO": "NO",  # Norway (Oslo)
    "FI": "FH",  # Finland (Helsinki)
    "BE": "BB",  # Belgium (Euronext Brussels)
    "JP": "JT",  # Japan (Tokyo)
    "IE": "ID",  # Ireland (Euronext Dublin)
    # Incorporation-haven codes: issuers domiciled here have no home venue of
    # that country — in this book's US symbology they are US-listed lines
    # (e.g. Cayman-incorporated ADRs). Map to US instead of warning per row;
    # the post-fetch venue-resolution pass validates identity regardless.
    "KY": "US",  # Cayman Islands
    "BM": "US",  # Bermuda
    "JE": "US",  # Jersey
    "LU": "US",  # Luxembourg
}


# ---------------------------------------------------------------------------
# Position record
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """One row from the Holdings sheet, normalized and trade-joined.

    Note: ``Position.position_id`` is a distinct identifier from
    ``Recommendation.position_id``. The latter is the BBG-formatted
    ticker (set by ``compute_recommendations``); ``Position.position_id``
    is the canonical row identifier (option_contract_key for options;
    ticker_final for equities/funds; product_name + per-account suffix
    for cash/other). V2 will rename one of them.
    """
    # Identity
    account: str
    position_id: str
    asset_class: str            # 'option' | 'equity' | 'fund_etf' | 'cash' | 'other'
    instrument_type: str        # routed class; keeps the raw extract class (e.g.
                                # 'Warrant') when an unknown class is held as 'other'

    # Tickers
    symbol: str                 # ticker_final for non-options; underlying_ticker for options
    bbg_ticker: str             # BBG-format. '' for cash/other.
    underlying_symbol: Optional[str]
    underlying_bbg_ticker: Optional[str]

    # Economics (all signed where applicable)
    quantity: Optional[float]
    multiplier: int
    valuation_price: Optional[float]
    market_value: float
    cost_basis: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pnl_pct: Optional[float]
    pct_pnl: Optional[float]    # alias of unrealized_pnl_pct for PositionContext compat

    # Option-specific
    option_type: Optional[str]
    right: Optional[str]        # alias of option_type
    strike: Optional[float]
    expiry: Optional[date]
    option_contract_key: Optional[str]

    # Trade-history-derived
    open_date: Optional[date] = None
    days_held: Optional[int] = None
    last_trade_date: Optional[date] = None
    last_trade_action: Optional[str] = None
    n_trades: int = 0
    transfer_inferred: bool = False  # True when any trade-history field above was
                                     # derived from another account's trades (the
                                     # book-transfer fallback join, options only).

    # Forward-compat
    style: Optional[str] = None   # The extract has no Style column; always None in V1.

    # Display
    name: Optional[str] = None    # Human-readable security name, surfaced from
                                  # the extract's Product Name / Underlying Name
                                  # column. Display-only; no engine input.

    # Set by the enrichment layer when a held option's ticker (built here from the
    # equity root) is re-keyed to the true listed ticker resolved from the option
    # chain: the original constructed string, kept for audit. None when the built
    # ticker resolved as-is (every US single-name). Non-keyed; diagnostic only.
    provisional_bbg_ticker: Optional[str] = None

    # Same audit contract for the venue-resolution pass on the UNDERLYING key:
    # when a mis-coded underlying ticker (wrong exchange suffix from an untrusted
    # extract country code) is re-keyed to an identity-validated listing, the
    # original constructed string is kept here. Non-keyed; diagnostic only.
    provisional_underlying_bbg_ticker: Optional[str] = None

    # Identity key for venue resolution: the instrument's ISIN (extract
    # 'ISIN Final') on equity/fund rows, the UNDERLYING's ISIN (extract
    # 'Underlying ISIN') on option rows. The extract's country codes are
    # untrusted input; the ISIN is the authoritative identity a recovered
    # ticker must match before it is accepted. None when the extract has none.
    isin: Optional[str] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_positions(extract: PortfolioExtract) -> list[Position]:
    """One ``Position`` per Holdings row, joined to per-account trade
    history. Mutates ``extract.parse_warnings`` to append any new
    warnings raised here.
    """
    holdings = extract.holdings
    trades = extract.trades
    warnings = extract.parse_warnings

    positions: list[Position] = []
    cash_other_counters: dict[str, int] = {}

    for _, row in holdings.iterrows():
        # When a load-bearing column is ENTIRELY absent the loader has already
        # emitted one summary flag with the affected-row count, so these per-row
        # skips stay silent (no N-note wall). A present column with a bad value
        # in some rows is a genuine per-row data issue and is still warned.
        asset_class = row.get("asset_class")
        if not isinstance(asset_class, str):
            if "asset_class" in row.index:
                warnings.append(
                    f"Holdings: skipped row with non-string asset_class ({asset_class!r})."
                )
            continue

        account = str(row.get("account") or "")
        if not account:
            if "account" in row.index:
                warnings.append("Holdings: skipped row with empty account.")
            continue

        position = _build_one(row, account, asset_class, cash_other_counters, warnings)
        if position is None:
            continue

        _attach_trade_history(position, trades)

        positions.append(position)

    return _consolidate_duplicate_ids(positions, warnings)


def _consolidate_duplicate_ids(positions: list[Position],
                               warnings: list[str]) -> list[Position]:
    """Merge Holdings rows that collide on ``(account, position_id)`` — the
    multi-lot shape (two tax-lot rows for one ticker or one option contract).

    Every keyed consumer downstream (AG-Grid rowIds, ``by_id`` position maps,
    the structure-allocation ledger, per-position greeks) assumes the key is
    unique; a collision silently drops one lot from those maps while detection
    sums the quantities — inflating slice economics against a half-sized
    position. Consolidating at ingest makes the invariant true by
    construction: quantities, market values, cost bases and P&L sum; the P&L
    percent is recomputed from the sums; identity/contract fields and the
    trade-history join (both lots join the same contract history) come from
    the first lot. Cash/other rows never reach here — they are de-collided
    with a per-account suffix at build time.

    Emits one warning per merged key naming the lot count, so a multi-lot
    extract is visible in the load notes rather than silently reshaped.
    """
    by_key: dict[tuple[str, str], Position] = {}
    lots: dict[tuple[str, str], int] = {}
    out: list[Position] = []
    for p in positions:
        key = (p.account, p.position_id)
        first = by_key.get(key)
        if first is None:
            by_key[key] = p
            lots[key] = 1
            out.append(p)
            continue
        lots[key] += 1
        for field_name in ("quantity", "market_value", "cost_basis", "unrealized_pnl"):
            a = getattr(first, field_name)
            b = getattr(p, field_name)
            # Sum when both lots carry the value; a lot with the value missing
            # makes the merged value honestly unknown (None), never a half-sum
            # presented as a whole. market_value is builder-defaulted to 0.0,
            # so it always sums.
            setattr(first, field_name,
                    (a + b) if (a is not None and b is not None) else None)
        pnl, cb = first.unrealized_pnl, first.cost_basis
        # The extract quotes the percent as a fraction of |cost basis|
        # (verified against the live sample: pct == pnl / |cb|, no x100).
        first.unrealized_pnl_pct = (
            pnl / abs(cb) if (pnl is not None and cb) else None)
        first.pct_pnl = first.unrealized_pnl_pct

    for key, n in lots.items():
        if n > 1:
            account, pid = key
            merged = by_key[key]
            qty = merged.quantity
            warnings.append(
                f"Holdings: {n} rows for {account}/{pid} (multi-lot extract) "
                f"consolidated into one position"
                + (f" of {qty:g}" if qty is not None else "")
                + "; quantities, market value, cost basis and P&L summed."
            )
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_one(
    row: pd.Series,
    account: str,
    asset_class: str,
    cash_other_counters: dict[str, int],
    warnings: list[str],
) -> Optional[Position]:
    market_value = _coerce_float(row.get("market_value"))
    if market_value is None:
        market_value = 0.0

    quantity = _coerce_float(row.get("quantity"))
    cost_basis = _coerce_float(row.get("cost_basis"))
    unrealized_pnl = _coerce_float(row.get("unrealized_pnl"))
    unrealized_pnl_pct = _coerce_float(row.get("unrealized_pnl_pct"))
    valuation_price = _coerce_float(row.get("valuation_price"))

    if asset_class == "option":
        return _build_option(
            row, account, market_value, quantity, cost_basis,
            unrealized_pnl, unrealized_pnl_pct, valuation_price, warnings,
        )
    if asset_class in ("equity", "fund_etf"):
        return _build_equity_or_fund(
            row, account, asset_class, market_value, quantity, cost_basis,
            unrealized_pnl, unrealized_pnl_pct, valuation_price, warnings,
        )
    if asset_class in ("cash", "other"):
        return _build_cash_or_other(
            row, account, asset_class, market_value, quantity, valuation_price,
            cash_other_counters,
        )

    # An unknown asset class (e.g. a warrant) must never silently vanish from
    # the book: route it to 'other' — held, shown in the grid, NAV-counted, and
    # inert to pricing/greeks/signals/structures — with the raw class kept on
    # instrument_type and an urgent flag naming the position and its size.
    position = _build_cash_or_other(
        row, account, "other", market_value, quantity, valuation_price,
        cash_other_counters, instrument_type=asset_class,
    )
    warnings.append(
        f"{URGENT_FLAG} Holdings: unknown asset_class {asset_class!r} for account "
        f"{account} ({position.name or position.symbol}, MV {market_value:,.0f}) — "
        f"held as 'other' (shown + NAV-counted, not priced)."
    )
    return position


def _build_option(
    row: pd.Series, account: str, market_value: float, quantity: Optional[float],
    cost_basis: Optional[float], unrealized_pnl: Optional[float],
    unrealized_pnl_pct: Optional[float], valuation_price: Optional[float],
    warnings: list[str],
) -> Optional[Position]:
    # As in build_positions: warn per row only when the column is present (a real
    # per-row gap); an absent column is the loader's single summary flag's job.
    contract_key = row.get("option_contract_key")
    if not isinstance(contract_key, str) or not contract_key:
        if "option_contract_key" in row.index:
            warnings.append(
                f"Holdings option row missing option_contract_key (account={account}); skipped."
            )
        return None

    underlying_ticker = row.get("underlying_ticker")
    if not isinstance(underlying_ticker, str) or not underlying_ticker:
        if "underlying_ticker" in row.index:
            warnings.append(
                f"Holdings option {contract_key} missing underlying_ticker; skipped."
            )
        return None

    option_type = row.get("option_type")
    if option_type not in ("CALL", "PUT"):
        if "option_type" in row.index:
            warnings.append(
                f"Holdings option {contract_key} has invalid option_type {option_type!r}; skipped."
            )
        return None

    strike = _coerce_float(row.get("option_strike"))
    expiry = row.get("option_expiration")
    if not isinstance(expiry, date) or strike is None:
        if "option_strike" in row.index and "option_expiration" in row.index:
            warnings.append(
                f"Holdings option {contract_key} missing strike/expiry; skipped."
            )
        return None

    # Index underliers (SPX, NDX, …) build "<root> Index" and carry the Index
    # sector through the option ticker — the extract has no instrument-type
    # signal for indices, so the config allowlist is the honest source. The
    # equity-shaped default would enumerate a dead security's chain and read
    # the book as riskless.
    from pm.config import INDEX_UNDERLIERS
    if (underlying_ticker or "").upper() in INDEX_UNDERLIERS:
        underlying_bbg = f"{underlying_ticker.upper()} Index"
        sector_hint = "Index"
    else:
        underlying_cc = _pick_country_code(
            row, "underlying_issuer_country_code", "issuer_country_code_final",
            "listing_hint_country_code",
        )
        underlying_bbg = _build_equity_bbg_ticker(underlying_ticker, underlying_cc, warnings)
        sector_hint = "Equity"

    try:
        option_bbg = construct_option_ticker(
            underlying_bbg, expiry, option_type, strike, sector_hint=sector_hint,
        )
    except ValueError as exc:
        warnings.append(
            f"Holdings option {contract_key}: construct_option_ticker failed ({exc}); skipped."
        )
        return None

    return Position(
        account=account,
        position_id=contract_key,
        asset_class="option",
        instrument_type="option",
        symbol=underlying_ticker,
        bbg_ticker=option_bbg,
        underlying_symbol=underlying_ticker,
        underlying_bbg_ticker=underlying_bbg,
        quantity=quantity,
        multiplier=100,
        valuation_price=valuation_price,
        market_value=market_value,
        cost_basis=cost_basis,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        pct_pnl=unrealized_pnl_pct,
        option_type=option_type,
        right=option_type,
        strike=strike,
        expiry=expiry,
        option_contract_key=contract_key,
        name=_security_name(row, "option"),
        isin=_clean_str(row.get("underlying_isin")),
    )


def _build_equity_or_fund(
    row: pd.Series, account: str, asset_class: str, market_value: float,
    quantity: Optional[float], cost_basis: Optional[float],
    unrealized_pnl: Optional[float], unrealized_pnl_pct: Optional[float],
    valuation_price: Optional[float], warnings: list[str],
) -> Optional[Position]:
    ticker_final = row.get("ticker_final")
    if not isinstance(ticker_final, str) or not ticker_final:
        if "ticker_final" in row.index:
            warnings.append(
                f"Holdings {asset_class} row missing ticker_final (account={account}); skipped."
            )
        return None

    cc = _pick_country_code(
        row, "issuer_country_code_final", "listing_hint_country_code",
        "underlying_issuer_country_code",
    )
    bbg_ticker = _build_equity_bbg_ticker(ticker_final, cc, warnings)

    return Position(
        account=account,
        position_id=ticker_final,
        asset_class=asset_class,
        instrument_type=asset_class,
        symbol=ticker_final,
        bbg_ticker=bbg_ticker,
        underlying_symbol=None,
        underlying_bbg_ticker=None,
        quantity=quantity,
        multiplier=1,
        valuation_price=valuation_price,
        market_value=market_value,
        cost_basis=cost_basis,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        pct_pnl=unrealized_pnl_pct,
        option_type=None,
        right=None,
        strike=None,
        expiry=None,
        option_contract_key=None,
        name=_security_name(row, asset_class),
        isin=_clean_str(row.get("isin_final")),
    )


def _build_cash_or_other(
    row: pd.Series, account: str, asset_class: str, market_value: float,
    quantity: Optional[float], valuation_price: Optional[float],
    cash_other_counters: dict[str, int],
    instrument_type: Optional[str] = None,
) -> Position:
    product_name = row.get("product_name")
    base = product_name if isinstance(product_name, str) and product_name else asset_class.upper()
    counter_key = (account, base)
    cash_other_counters[counter_key] = cash_other_counters.get(counter_key, 0) + 1
    suffix = cash_other_counters[counter_key]
    position_id = f"{base}__{suffix}" if suffix > 1 else base

    symbol_value = row.get("ticker_final")
    symbol = symbol_value if isinstance(symbol_value, str) else base

    return Position(
        account=account,
        position_id=position_id,
        asset_class=asset_class,
        # An unknown extract class routed here keeps its raw string (e.g.
        # "Warrant") so the position stays identifiable; native cash/other
        # rows carry their own class.
        instrument_type=instrument_type or asset_class,
        symbol=symbol,
        bbg_ticker="",
        underlying_symbol=None,
        underlying_bbg_ticker=None,
        quantity=quantity,
        multiplier=1,
        valuation_price=valuation_price,
        market_value=market_value,
        cost_basis=None,
        unrealized_pnl=None,
        unrealized_pnl_pct=None,
        pct_pnl=None,
        option_type=None,
        right=None,
        strike=None,
        expiry=None,
        option_contract_key=None,
        name=_security_name(row, asset_class),
    )


def _attach_trade_history(position: Position, trades: pd.DataFrame) -> None:
    if trades is None or trades.empty:
        return

    # Options join account-scoped FIRST on (account, option_contract_key); the
    # join widens to the contract key alone only when the holding account's own
    # rows cannot supply a derivation (cross-account journal entries / book
    # transfers), and anything derived that way marks the position
    # transfer_inferred. Equities/funds join on (account, ticker_final).
    book_wide = None
    if position.asset_class == "option" and position.option_contract_key:
        if "option_contract_key" not in trades.columns:
            return
        book_wide = trades[trades["option_contract_key"] == position.option_contract_key]
        if "account" in book_wide.columns:
            matches = book_wide[book_wide["account"] == position.account]
        else:
            # No account column at all: unscoped is the only join available (the
            # loader already flags the dropped column urgently).
            matches = book_wide
        if matches.empty and not book_wide.empty:
            # The holding account never traded this contract: transfer fallback.
            matches = book_wide
            position.transfer_inferred = True
    elif position.asset_class in ("equity", "fund_etf"):
        if "ticker_final" not in trades.columns or "account" not in trades.columns:
            return
        matches = trades[
            (trades["account"] == position.account)
            & (trades["ticker_final"] == position.symbol)
        ]
    else:
        return

    if matches.empty:
        return

    position.n_trades = int(len(matches))

    sorted_matches = matches.sort_values("trade_date") if "trade_date" in matches.columns else matches
    open_rows = sorted_matches
    if "option_lifecycle_action" in sorted_matches.columns:
        open_rows = sorted_matches[
            sorted_matches["option_lifecycle_action"].isin(["Buy to Open", "Sell to Open"])
        ]
    if open_rows.empty and "buy_sell" in sorted_matches.columns and position.asset_class != "option":
        # For equities / funds in the absence of lifecycle action, treat
        # the earliest Buy as the opening trade for longs and earliest
        # Sell for shorts.
        if position.quantity is not None and position.quantity >= 0:
            open_rows = sorted_matches[sorted_matches["buy_sell"] == "Buy"]
        else:
            open_rows = sorted_matches[sorted_matches["buy_sell"] == "Sell"]

    if (
        open_rows.empty
        and not position.transfer_inferred
        and book_wide is not None
        and len(book_wide) > len(matches)
        and "option_lifecycle_action" in book_wide.columns
    ):
        # The account traded the contract but holds no opening trade of its own
        # (transferred in, then adjusted): widen the OPEN derivation — and only
        # that — to the whole book, marked.
        wide_sorted = (
            book_wide.sort_values("trade_date") if "trade_date" in book_wide.columns else book_wide
        )
        wide_open = wide_sorted[
            wide_sorted["option_lifecycle_action"].isin(["Buy to Open", "Sell to Open"])
        ]
        if not wide_open.empty:
            open_rows = wide_open
            position.transfer_inferred = True

    if not open_rows.empty:
        first_open = open_rows.iloc[0]
        open_dt = first_open.get("trade_date")
        if isinstance(open_dt, date):
            position.open_date = open_dt
            position.days_held = max((clock.today() - open_dt).days, 0)

    last = sorted_matches.iloc[-1]
    last_dt = last.get("trade_date")
    if isinstance(last_dt, date):
        position.last_trade_date = last_dt
    last_action = last.get("option_lifecycle_action")
    if isinstance(last_action, str):
        position.last_trade_action = last_action
    elif isinstance(last.get("buy_sell"), str):
        position.last_trade_action = str(last["buy_sell"])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_str(value: object) -> Optional[str]:
    """A non-empty trimmed string, or None (handles NaN / blanks)."""
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return None


def _security_name(row: pd.Series, asset_class: str) -> Optional[str]:
    """Human-readable name for a holdings row. Options prefer the underlying
    company name; everything else prefers the product name. Both come straight
    from the extract — no lookup, no BBG."""
    product = _clean_str(row.get("product_name"))
    underlying = _clean_str(row.get("underlying_name"))
    if asset_class == "option":
        return underlying or product
    return product or underlying


def _pick_country_code(row: pd.Series, *cols: str) -> Optional[str]:
    for col in cols:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _is_otc_f_share(ticker: str) -> bool:
    """Five-letter tickers ending in 'F' are US OTC symbology for a foreign
    ordinary line (e.g. a European name's OTC trading line) — US-traded
    regardless of the issuer's incorporation country. The extract's country
    code on such rows describes the issuer, not the venue."""
    return len(ticker) == 5 and ticker.isalpha() and ticker.isupper() and ticker.endswith("F")


def _build_equity_bbg_ticker(
    ticker: str, country_code: Optional[str], warnings: list[str],
) -> str:
    suffix = "US"
    if _is_otc_f_share(ticker):
        return f"{ticker} US Equity"
    if country_code:
        mapped = COUNTRY_TO_BBG_SUFFIX.get(country_code.upper())
        if mapped is None:
            msg = (
                f"No BBG exchange suffix for country code {country_code!r} "
                f"(ticker {ticker}); defaulting to 'US'. Add the mapping if BBG fails."
            )
            # One note per (code, ticker) — a name held across several rows
            # (e.g. one per option leg) must not wall the status bar.
            if msg not in warnings:
                warnings.append(msg)
        else:
            suffix = mapped
    return f"{ticker} {suffix} Equity"
