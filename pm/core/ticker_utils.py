"""Ticker construction utilities.

Includes the option-ticker construction primitives. The live BBG
validation (`validate_tickers`, `resolve_option_ticker_from_strike`)
calls Bloomberg in production; here `validate_tickers` is a stub that
raises if invoked without a monkeypatch (the live implementation lives
in `pm.core.bloomberg_client`). The resolver can be tested via a
monkey-patched validator.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Sequence

import pandas as pd


# Sector keywords that may appear at the end of a Bloomberg ticker
MARKET_SECTOR_KEYWORDS = {"EQUITY", "INDEX", "CURNCY", "COMDTY"}

# Strike offset ladder used by resolve_option_ticker_from_strike to walk
# nearby strikes when the exact strike isn't tradeable.
DEFAULT_STRIKE_OFFSETS = [-0.5, 0.5, -1.0, 1.0, -2.5, 2.5, -5.0, 5.0]


def _normalize_put_call(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().upper()
    if text in {"C", "CALL"}:
        return "CALL"
    if text in {"P", "PUT"}:
        return "PUT"
    return None


def format_bbg_expiry(expiry: object) -> str:
    if isinstance(expiry, (date, datetime, pd.Timestamp)):
        expiry_date = expiry.date() if isinstance(expiry, datetime) else expiry
    else:
        parsed = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(parsed):
            raise ValueError("Invalid expiry provided for Bloomberg ticker.")
        expiry_date = parsed.date()
    month = expiry_date.month
    day = expiry_date.day
    year = expiry_date.year % 100
    return f"{month}/{day}/{year:02d}"


def _strip_sector_suffix(underlying: str) -> str:
    parts = underlying.strip().split()
    if not parts:
        return ""
    if parts[-1].upper() in MARKET_SECTOR_KEYWORDS:
        parts = parts[:-1]
    return " ".join(parts).strip()


def _infer_sector_suffix(underlying: str, sector_hint: Optional[str]) -> str:
    hint = (sector_hint or "").strip()
    if hint:
        hint_key = hint.upper()
        if hint_key in MARKET_SECTOR_KEYWORDS:
            return hint_key.title()
        return hint
    if "INDEX" in underlying.upper():
        return "Index"
    return "Equity"


def _format_strike_for_ticker(strike: float) -> str:
    try:
        value = float(strike)
    except (TypeError, ValueError):
        raise ValueError("Strike must be numeric for ticker construction.")
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text


def construct_option_ticker(
    underlying: str,
    expiry: str,
    put_call: str,
    strike: float,
    sector_hint: Optional[str] = None,
) -> str:
    base = _strip_sector_suffix(underlying)
    if not base:
        raise ValueError("Underlying is required for ticker construction.")
    side = _normalize_put_call(put_call)
    if side is None:
        raise ValueError("put_call must be CALL or PUT.")
    prefix = "C" if side == "CALL" else "P"
    expiry_text = format_bbg_expiry(expiry)
    strike_text = _format_strike_for_ticker(strike)
    sector = _infer_sector_suffix(underlying, sector_hint)
    return f"{base} {expiry_text} {prefix}{strike_text} {sector}".strip()


def validate_tickers(tickers: list[str]) -> pd.DataFrame:
    """Live-probe seam — no live implementation exists in this codebase.

    A real implementation would BDP-probe the tickers and return the subset
    that resolve to a genuine Bloomberg contract. Tests monkeypatch this;
    any un-patched call raises.
    """
    raise NotImplementedError(
        "validate_tickers has no live implementation — monkeypatch it in tests."
    )


def resolve_option_ticker_from_strike(
    underlying: str,
    expiry: str,
    put_call: str,
    strike: float,
    sector_hint: Optional[str] = None,
    offsets: Optional[Sequence[float]] = None,
) -> Optional[str]:
    exact = construct_option_ticker(
        underlying, expiry, put_call, strike, sector_hint
    )
    exact_df = validate_tickers([exact])
    if not exact_df.empty:
        return exact

    offset_list = list(offsets) if offsets is not None else DEFAULT_STRIKE_OFFSETS
    candidates: list[str] = []
    seen: set[str] = set()
    for offset in offset_list:
        if offset == 0:
            continue
        candidate_strike = float(strike) + float(offset)
        if candidate_strike <= 0:
            continue
        ticker = construct_option_ticker(
            underlying, expiry, put_call, candidate_strike, sector_hint
        )
        key = ticker.upper()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(ticker)

    if not candidates:
        return None
    validated = validate_tickers(candidates)
    if validated.empty:
        return None
    valid_set = {
        str(value).strip().upper()
        for value in validated["security"].dropna().tolist()
    }
    for ticker in candidates:
        if ticker.upper() in valid_set:
            return ticker
    return None


def _value_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Option-description parsing + matching (chain resolution)
# ---------------------------------------------------------------------------
# A canonical Bloomberg option ticker (the OPT_CHAIN 'Security Description') reads
# 'NES1 SW 07/03/26 C67.5 Equity': a possibly-multi-token root, then the expiry
# (MM/DD/YY), then <C|P><strike>, then the market sector. Parsing from the right
# keeps a multi-token root intact.


def _parse_bbg_expiry_token(token: object) -> Optional[date]:
    """A python date from a Bloomberg 'MM/DD/YY' expiry token, else None."""
    parsed = pd.to_datetime(str(token).strip(), format="%m/%d/%y", errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _coerce_expiry_date(value: object) -> Optional[date]:
    """A python date from a date / datetime / Timestamp / parseable string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def parse_option_description(description: object) -> Optional[dict]:
    """Parse a canonical option ticker string into
    ``{'ticker', 'root', 'expiry': date, 'right': 'CALL'|'PUT', 'strike': float}``,
    or ``None`` when it doesn't parse. ``ticker`` is the original string verbatim
    (the value that round-trips through a snapshot fetch)."""
    if not description:
        return None
    text = str(description).strip()
    parts = text.split()
    if len(parts) < 4:
        return None
    opt_token = parts[-2]
    right = _normalize_put_call(opt_token[:1])
    strike = _value_or_none(opt_token[1:])
    expiry = _parse_bbg_expiry_token(parts[-3])
    if right is None or strike is None or expiry is None:
        return None
    return {
        "ticker": text,
        "root": " ".join(parts[:-3]),
        "expiry": expiry,
        "right": right,
        "strike": strike,
    }


def match_option_ticker(
    chain: Sequence[str], expiry: object, strike: object, right: object,
    constructed: Optional[str] = None,
) -> Optional[str]:
    """The canonical option ticker in *chain* matching (*expiry*, *strike*,
    *right*), or ``None`` when no listed contract matches. *chain* is a list of
    'Security Description' strings (see ``pm.core.bloomberg_client.fetch_option_chain``).
    Returns the matched string VERBATIM so it round-trips through the snapshot
    fetch's reindex.

    A chain can list more than one contract at the same coordinates under
    different roots — a corporate-action-adjusted line (e.g. an appended-digit
    root) beside the standard one, with different deliverables. A lone match is
    returned as before; on a multi-root collision the entry whose root exactly
    equals *constructed*'s root wins, and anything else returns ``None`` so the
    caller keeps its honest unresolved warning instead of silently re-keying to
    the wrong contract."""
    want_expiry = _coerce_expiry_date(expiry)
    want_right = _normalize_put_call(right)
    want_strike = _value_or_none(strike)
    if want_expiry is None or want_right is None or want_strike is None:
        return None
    matches: list[dict] = []
    for description in chain or []:
        parsed = parse_option_description(description)
        if parsed is None:
            continue
        if (parsed["right"] == want_right
                and parsed["expiry"] == want_expiry
                and abs(parsed["strike"] - want_strike) <= 1e-6):
            matches.append(parsed)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]["ticker"]
    want_root = None
    if constructed:
        parsed_constructed = parse_option_description(constructed)
        if parsed_constructed is not None:
            want_root = parsed_constructed["root"]
    root_hits = [m for m in matches if want_root is not None and m["root"] == want_root]
    if len(root_hits) == 1:
        return root_hits[0]["ticker"]
    return None


# ---------------------------------------------------------------------------
# Targeted slice selection (the on-demand chain pull's client-side filter)
# ---------------------------------------------------------------------------

def _is_third_friday(d: date) -> bool:
    """True when *d* is the standard monthly option expiry — the 3rd Friday of its
    month (the Friday landing on day 15–21)."""
    return d.weekday() == 4 and 15 <= d.day <= 21


def filter_chain_slice(
    parsed_chain,
    spot: object,
    ref_strike: object,
    *,
    horizon_expiry: Optional[date] = None,
    n_expiries: int = 3,
    moneyness_pct: float = 0.15,
    rights: Sequence[str] = ("CALL", "PUT"),
    monthlies_only: bool = True,
    today: Optional[date] = None,
) -> list[str]:
    """The targeted slice of *parsed_chain* around *spot* and the held (*ref_strike*)
    contract: the canonical tickers for ``n_expiries`` expiries bracketing the roll
    horizon and strikes within ``moneyness_pct`` of spot OR the held strike, on the
    requested rights.

    *parsed_chain* is the enumerated chain — each item a ``parse_option_description``
    dict or a raw description string (parsed here). Expiries are taken ``>= today``;
    with ``monthlies_only`` (default) only standard 3rd-Friday expiries are kept, so a
    liquid name's weeklies don't balloon the count. Expiry selection is forward-biased
    (the roll-out direction), back-filling earlier monthlies only when fewer than
    ``n_expiries`` lie ahead. Pure and BBG-free; the caller fetches snapshots for the
    returned tickers. ``today`` defaults to the current date; it is injectable so the
    selection is deterministic under test.
    """
    if today is None:
        today = date.today()
    want_rights = {r for r in (_normalize_put_call(x) for x in rights) if r}
    spot_v = _value_or_none(spot)
    if spot_v is None or spot_v <= 0 or not want_rights:
        return []
    ref = _value_or_none(ref_strike)

    # Normalise to parsed dicts; drop unparseable rows, non-positive strikes (the
    # '.01'-type adjusted entries are excluded by the moneyness band regardless), past
    # expiries, and off-right rows.
    items = []
    for entry in parsed_chain or []:
        parsed = entry if isinstance(entry, dict) else parse_option_description(entry)
        if not parsed:
            continue
        strike, expiry, r = parsed.get("strike"), parsed.get("expiry"), parsed.get("right")
        if strike is None or strike <= 0 or expiry is None or expiry < today:
            continue
        if r not in want_rights:
            continue
        items.append(parsed)
    if not items:
        return []

    horizon = horizon_expiry or min(p["expiry"] for p in items)
    expiries = sorted({p["expiry"] for p in items
                       if not monthlies_only or _is_third_friday(p["expiry"])})
    if not expiries:
        return []
    forward = [e for e in expiries if e >= horizon]
    chosen = set(forward[:n_expiries])
    if len(chosen) < n_expiries:
        back = [e for e in expiries if e < horizon][-(n_expiries - len(chosen)):]
        chosen |= set(back)

    def _in_band(k: float) -> bool:
        if abs(k - spot_v) / spot_v <= moneyness_pct:
            return True
        return ref is not None and ref > 0 and abs(k - ref) / ref <= moneyness_pct

    out = {p["ticker"] for p in items
           if p["expiry"] in chosen and _in_band(p["strike"])}
    return sorted(out)
