"""Bloomberg adapter for Portfolio Manager.

A thin wrapper over ``polars_bloomberg`` (BLPAPI). Provides:
  * a session context manager + lazy ``polars_bloomberg`` import,
  * ``fetch_spot``, ``fetch_underlying_snapshot``, ``fetch_underlying_snapshots``,
  * ``fetch_option_snapshot``, ``fetch_option_snapshots``,
  * ``fetch_risk_free_rate`` (single-tenor treasury picker) +
    ``fetch_risk_free_curve`` / ``pick_rate_for_dte`` (full UST curve for the
    carry fire and layer-2 pricing),
  * ``fetch_projected_dividend`` / ``fetch_projected_dividends`` (forward
    dividend forecast, single + batched),
  * ``fetch_analyst_data`` / ``fetch_analyst_data_batch`` (BE998=UBS
    analyst-data override, single + batched),
  * ``is_bloomberg_available`` (startup probe).

``fetch_projected_dividend`` returns the forward dividend forecast in a fixed
shape, parsed from the bulk BDVD_PR_EX_DTS_DVD_AMTS_W_ANN field via plain BDP
(this environment's polars_bloomberg has no ``.bds``); when the field yields
nothing, the ex-div fire degrades to the yield heuristic. Further helpers:
``fetch_option_chain`` / ``fetch_option_snapshots`` (chain enumeration and
contract snapshots), ``fetch_iv_history``, ``fetch_ovdv_grid``,
``fetch_spx_betas``, and ``fetch_analyst_note_dates``.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTION_SNAPSHOT_FIELDS = [
    "BID",
    "ASK",
    "PX_MID",
    "MID",
    "PX_LAST",
    "IVOL_MID",
    "IVOL",
    "DAYS_TO_EXPIRATION",
    "DAYS_EXPIRE",
    "OPT_STRIKE_PX",
    "OPT_PUT_CALL",
    "DELTA_MID_RT",
    "THETA",
    "THETA_MID",
    "GAMMA",
    "VEGA",
    "RHO",
    # Per-contract liquidity for the scanner chain table. OPEN_INT / PX_VOLUME are the
    # standard reference mnemonics; the terminal field for a given entitlement is
    # confirmed by a live probe, and _normalize_option_fields coalesces any alternate
    # column (OPT_OPEN_INTEREST / VOLUME / VOLM) that a request adds.
    "OPEN_INT",
    "PX_VOLUME",
    # Exercise style ('American' / 'European') — drives the pricing-engine style
    # key (pm.pricing registry). Read per leg; the scenario layer never defaults it.
    "OPTION_EXERCISE_TYPE_REALTIME",
]

MARKET_SECTOR_KEYWORDS = {"EQUITY", "INDEX", "CURNCY", "COMDTY"}

UNDERLYING_FIELDS = [
    "PX_LAST",
    "NAME",
    "GICS_SECTOR_NAME",
    "INDUSTRY_SECTOR",
    "52WK_HIGH",
    "52WK_LOW",
    "HIGH_52WEEK",
    "LOW_52WEEK",
    "HIGH_DT_52WEEK",
    "LOW_DT_52WEEK",
    "CHG_PCT_1YR",
    "EQY_TRR_PCT_1YR",
    "CHG_PCT_5D",
    "CHG_PCT_3M",
    "CHG_PCT_YTD",
    "VOL_PERCENTILE",
    "3MTH_IMPVOL_100.0%MNY_DF",
    "EARNINGS_RELATED_IMPLIED_MOVE",
    "CHG_PCT_1D",
    "CHG_NET_1D",
    "6MTH_IMPVOL_100.0%MNY_DF",
    "DVD_YLD",
    "EQY_DVD_YLD_IND",
    "DVD_EX_DT",
    "EQY_DVD_EX_DT",
    "DVD_EX_DATE",
    "EQY_DVD_EX_DATE",
    "EXPECTED_REPORT_DT",
    "EARNINGS_ANNOUNCEMENT_DATE",
    "MOV_AVG_200D",   # 200-day moving average (used by Cash-Secured Put template)
    "PUT_CALL_OPEN_INTEREST_RATIO",
    "PUT_CALL_VOLUME_RATIO_CUR_DAY",
    # --- Trend / momentum / vol fields ---
    "BETA_ADJ_OVERRIDABLE",   # 2y daily beta vs market index, adjusted
    "RSI_14D",                # 14-day Wilder RSI
    "MOV_AVG_50D",            # 50-day SMA (paired with MOV_AVG_200D for trend stack)
    "CALL_IMP_VOL_30D",       # 30D call IV — substitute for 1M ATM IV (BDP doesn't
                              # populate 1MTH_IMPVOL_100.0%MNY_DF for most US names)
    "BEST_EPS",               # Consensus EPS estimate (next reporting period)
    "VOLATILITY_30D",         # 30-day realized vol (annualized %)
    # Street consensus (D2) — deliberately UN-overridden on this batched pull, so
    # these are the whole street, never the provider view (which comes from the
    # separate overridden fetch). Confirmed live: BEST_ANALYST_RATING = numeric
    # 1-5 consensus score; the provider-only BEST_ANALYST_REC stays out.
    "BEST_ANALYST_RATING",
    "BEST_TARGET_PRICE",
]
DIVIDEND_FIELDS = [
    "DVD_EX_DT",
    "EQY_DVD_EX_DT",
    "DVD_EX_DATE",
    "EQY_DVD_EX_DATE",
]


# Treasury index tickers ordered by maturity threshold (days)
_TREASURY_MAP = [
    (30,    "USGG1M Index",   "1M UST"),
    (90,    "USGG3M Index",   "3M UST"),
    (180,   "USGG6M Index",   "6M UST"),
    (365,   "USGG12M Index",  "12M UST"),
    (730,   "USGG2YR Index",  "2Y UST"),
    (1095,  "USGG3YR Index",  "3Y UST"),
    (1825,  "USGG5YR Index",  "5Y UST"),
    (2555,  "USGG7YR Index",  "7Y UST"),
    (3650,  "USGG10YR Index", "10Y UST"),
    (7300,  "USGG20YR Index", "20Y UST"),
    (99999, "USGG30YR Index", "30Y UST"),
]

_RFR_DEFAULT: dict = {"rate": 0.0, "ticker": "", "label": ""}


# ---------------------------------------------------------------------------
# Session + DataFrame helpers# ---------------------------------------------------------------------------

@contextmanager
def with_session():
    from polars_bloomberg import BQuery

    with BQuery() as query:
        yield query


def _to_pandas(data: object) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    return pd.DataFrame(data)


def _ensure_security_column(df: pd.DataFrame) -> pd.DataFrame:
    if "security" in df.columns:
        return df
    for candidate in ["Security", "SECURITY", "ticker", "Ticker"]:
        if candidate in df.columns:
            return df.rename(columns={candidate: "security"})
    if df.index.name:
        return df.reset_index().rename(columns={df.index.name: "security"})
    return df.reset_index().rename(columns={"index": "security"})


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


# ---------------------------------------------------------------------------
# Spot / snapshot# ---------------------------------------------------------------------------

def fetch_spot(ticker: str) -> float:
    with with_session() as query:
        raw = query.bdp([ticker], ["PX_LAST"])
    df = _ensure_security_column(_to_pandas(raw))
    if "PX_LAST" not in df.columns or df.empty:
        return float("nan")
    row = df.loc[df["security"] == ticker]
    if row.empty:
        row = df.iloc[[0]]
    value = row["PX_LAST"].iloc[0]
    if pd.isna(value):
        return float("nan")
    return float(value)


def fetch_underlying_snapshot(ticker: str) -> pd.Series:
    if not ticker:
        return pd.Series(dtype=object)
    with with_session() as query:
        raw = query.bdp([ticker], UNDERLYING_FIELDS)
    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, UNDERLYING_FIELDS)
    if df.empty:
        return pd.Series(dtype=object)
    row = df.loc[df["security"] == ticker]
    if row.empty:
        row = df.iloc[[0]]
    record = row.iloc[0].copy()
    def _clean_value(value: object) -> object:
        try:
            if pd.isna(value):
                return None
        except Exception:
            logger.debug("pd.isna check failed for value: %r", value)
        return value

    high_52week = record.get("HIGH_52WEEK")
    if pd.isna(high_52week):
        high_52week = record.get("52WK_HIGH")
    low_52week = record.get("LOW_52WEEK")
    if pd.isna(low_52week):
        low_52week = record.get("52WK_LOW")
    record["high_52week"] = _clean_value(high_52week)
    record["low_52week"] = _clean_value(low_52week)
    record["week_52_high"] = record.get("high_52week")
    record["week_52_low"] = record.get("low_52week")
    record["high_dt_52week"] = _clean_value(record.get("HIGH_DT_52WEEK"))
    record["low_dt_52week"] = _clean_value(record.get("LOW_DT_52WEEK"))
    record["chg_pct_1yr"] = _clean_value(record.get("CHG_PCT_1YR"))
    record["eqy_trr_pct_1yr"] = _clean_value(record.get("EQY_TRR_PCT_1YR"))
    record["chg_pct_5d"] = _clean_value(record.get("CHG_PCT_5D"))
    record["chg_pct_3m"] = _clean_value(record.get("CHG_PCT_3M"))
    record["chg_pct_ytd"] = _clean_value(record.get("CHG_PCT_YTD"))
    record["vol_percentile"] = _clean_value(record.get("VOL_PERCENTILE"))
    record["impvol_3m_atm"] = _clean_value(record.get("3MTH_IMPVOL_100.0%MNY_DF"))
    record["chg_pct_1d"] = _clean_value(record.get("CHG_PCT_1D"))
    record["chg_net_1d"] = _clean_value(record.get("CHG_NET_1D"))
    record["impvol_6m_atm"] = _clean_value(record.get("6MTH_IMPVOL_100.0%MNY_DF"))
    record["earnings_related_implied_move"] = _clean_value(
        record.get("EARNINGS_RELATED_IMPLIED_MOVE")
    )
    projection = fetch_projected_dividend(ticker)
    next_div = projection.get("next") or {}
    bdp_ex_div_date = get_next_dividend_date(ticker, snapshot=record.to_dict())
    record["ex_div_date"] = next_div.get("ex_date") or bdp_ex_div_date
    record["projected_dividend"] = next_div.get("dps")
    record["dividend_schedule"] = projection.get("schedule") or []
    record["mov_avg_200d"] = _clean_value(record.get("MOV_AVG_200D"))
    record["put_call_oi_ratio"] = _clean_value(record.get("PUT_CALL_OPEN_INTEREST_RATIO"))
    record["put_call_vol_ratio"] = _clean_value(record.get("PUT_CALL_VOLUME_RATIO_CUR_DAY"))
    analyst = fetch_analyst_data(ticker)
    record["analyst_rating"] = analyst.get("analyst_rating")
    record["analyst_target"] = analyst.get("analyst_target")
    return record


# ---------------------------------------------------------------------------
# Provider-overridden analyst data# ---------------------------------------------------------------------------

def fetch_analyst_data(ticker: str) -> Dict[str, object]:
    """Fetch provider-specific analyst rating and target price from Bloomberg.

    Uses BDP with the BE998=UBS override for provider-specific data. Fails
    CLOSED: if the overridden call cannot be made (override API unavailable)
    or fails, this returns absent data — never a bare non-overridden retry,
    which would return street consensus that downstream traces label as
    provider research. Absent data flows through as a stale signal instead
    (same posture as ``fetch_spx_betas``).
    """
    if not ticker:
        return {}
    fields = ["BEST_ANALYST_REC", "BEST_TARGET_PRICE"]
    result: Dict[str, object] = {}
    try:
        with with_session() as query:
            bdp_fn = getattr(query, "bdp", None)
            if bdp_fn is None:
                return result
            # polars_bloomberg BQuery.bdp overrides: list[tuple] | None
            try:
                raw = bdp_fn([ticker], fields, overrides=[("BE998", "UBS")])
            except TypeError:
                # bdp without override support — provider-specific data is
                # impossible here; surface nothing rather than street consensus.
                logger.warning("Analyst pull: bdp overrides unsupported; "
                               "provider analyst data unavailable for %s", ticker)
                return result
            except Exception as exc:
                logger.warning("Analyst pull with override failed for %s: %s "
                               "(failing closed — no bare retry)", ticker, exc)
                return result
            df = _ensure_security_column(_to_pandas(raw))
            df = _ensure_columns(df, fields)
            if df.empty:
                return result
            row = df.loc[df["security"] == ticker]
            if row.empty:
                row = df.iloc[[0]]
            rec = row.iloc[0]
            rating = rec.get("BEST_ANALYST_REC")
            target = rec.get("BEST_TARGET_PRICE")
            if rating is not None:
                try:
                    if not pd.isna(rating):
                        result["analyst_rating"] = str(rating).strip()
                except (TypeError, ValueError):
                    result["analyst_rating"] = str(rating).strip()
            if target is not None:
                try:
                    if not pd.isna(target):
                        result["analyst_target"] = float(target)
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        logger.warning("Analyst data fetch failed for %s: %s", ticker, e)
    return result


def fetch_analyst_data_batch(tickers: list[str]) -> Dict[str, Dict[str, object]]:
    """Provider-specific analyst rating and target price for N tickers in ONE
    overridden BDP round trip — the batched analog of ``fetch_analyst_data``
    (measured on the live book: 23 single-ticker calls ~13s -> one call ~3s).

    Uses BDP with the BE998=UBS override. Returns ``{ticker: {"analyst_rating":
    str, "analyst_target": float}}`` with the same per-ticker value shaping as
    the singular fetcher: a missing value's key is absent, and every requested
    ticker is present (worst case ``{}``). Fails CLOSED exactly like the
    singular fetcher: if the overridden call cannot be made (override API
    unavailable) or fails, every ticker maps to ``{}`` — never a bare
    non-overridden retry, which would return street consensus that downstream
    traces label as provider research. Never raises.
    """
    out: Dict[str, Dict[str, object]] = {t: {} for t in tickers}
    if not tickers:
        return out
    fields = ["BEST_ANALYST_REC", "BEST_TARGET_PRICE"]
    try:
        with with_session() as query:
            bdp_fn = getattr(query, "bdp", None)
            if bdp_fn is None:
                return out
            # polars_bloomberg BQuery.bdp overrides: list[tuple] | None
            try:
                raw = bdp_fn(list(tickers), fields, overrides=[("BE998", "UBS")])
            except TypeError:
                # bdp without override support — provider-specific data is
                # impossible here; surface nothing rather than street consensus.
                logger.warning("Analyst pull: bdp overrides unsupported; "
                               "provider analyst data unavailable")
                return out
            except Exception as exc:
                logger.warning("Batched analyst pull with override failed: %s "
                               "(failing closed — no bare retry)", exc)
                return out
            df = _ensure_security_column(_to_pandas(raw))
            df = _ensure_columns(df, fields)
            if df.empty:
                return out
            df = df.set_index("security")
            for t in tickers:
                if t not in df.index:
                    continue
                rec = df.loc[t]
                rating = rec.get("BEST_ANALYST_REC")
                target = rec.get("BEST_TARGET_PRICE")
                if rating is not None:
                    try:
                        if not pd.isna(rating):
                            out[t]["analyst_rating"] = str(rating).strip()
                    except (TypeError, ValueError):
                        out[t]["analyst_rating"] = str(rating).strip()
                if target is not None:
                    try:
                        if not pd.isna(target):
                            out[t]["analyst_target"] = float(target)
                    except (TypeError, ValueError):
                        pass
    except Exception as e:
        logger.warning("Batched analyst data fetch failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# Risk-free rate# ---------------------------------------------------------------------------

def fetch_risk_free_rate(dte: int) -> dict:
    """Fetch risk-free rate from US Treasury indices matched to option DTE.

    Selects the first treasury index whose maturity threshold >= *dte*,
    fetches PX_LAST (yield as %), and returns a dict with the rate as a
    decimal (e.g. 4.5% → 0.045), the Bloomberg ticker, and a short label.
    Returns ``_RFR_DEFAULT`` on any failure.
    """
    if dte <= 0:
        return dict(_RFR_DEFAULT)
    # Pick the first bucket where dte <= threshold
    best = next((t for t in _TREASURY_MAP if dte <= t[0]), _TREASURY_MAP[-1])
    best_ticker, best_label = best[1], best[2]
    try:
        with with_session() as query:
            raw = query.bdp([best_ticker], ["PX_LAST"])
        df = _ensure_security_column(_to_pandas(raw))
        if "PX_LAST" not in df.columns or df.empty:
            return dict(_RFR_DEFAULT)
        row = df.loc[df["security"] == best_ticker]
        if row.empty:
            row = df.iloc[[0]]
        value = row["PX_LAST"].iloc[0]
        if pd.isna(value):
            return dict(_RFR_DEFAULT)
        return {"rate": float(value) / 100.0, "ticker": best_ticker, "label": best_label}
    except Exception:
        logger.warning("Risk-free rate lookup failed, using default")
        return dict(_RFR_DEFAULT)


def fetch_risk_free_curve() -> list:
    """Fetch the whole US Treasury curve (USGG 1M–30Y) in one BDP round trip.

    Returns an ordered list (short to long) of
    ``{"max_days", "ticker", "label", "rate"}`` where ``rate`` is a decimal
    (4.5% -> 0.045) or ``None`` when BBG returned nothing for that tenor. The
    carry fire picks the tenor nearest a leg's DTE via ``pick_rate_for_dte``;
    the full curve is kept (not collapsed to one point) so layer-2 pricing can
    reuse it. Returns ``[]`` on any failure — callers fall back to the
    configured scalar rate. Never raises.
    """
    tickers = [t[1] for t in _TREASURY_MAP]
    try:
        with with_session() as query:
            raw = query.bdp(tickers, ["PX_LAST"])
        df = _ensure_security_column(_to_pandas(raw))
    except Exception:
        logger.warning("Treasury-curve fetch failed, falling back to scalar rate")
        return []
    if df.empty or "PX_LAST" not in df.columns:
        return []
    px_by_ticker = {row.get("security"): row.get("PX_LAST") for _, row in df.iterrows()}
    curve: list = []
    for max_days, ticker, label in _TREASURY_MAP:
        px = px_by_ticker.get(ticker)
        rate = None
        if px is not None and not pd.isna(px):
            try:
                rate = float(px) / 100.0
            except (TypeError, ValueError):
                rate = None
        curve.append({"max_days": max_days, "ticker": ticker, "label": label, "rate": rate})
    return curve


def pick_rate_for_dte(curve: list, dte: Optional[int]) -> Optional[dict]:
    """Pick the curve tenor nearest *dte*: the first bucket whose ``max_days`` is
    >= dte (same bucketing as ``fetch_risk_free_rate``), else the longest tenor.
    Skips tenors BBG left without a rate. Returns the tenor dict (with a non-None
    ``rate``) or ``None`` when the curve is empty / has no usable rate."""
    if not curve or dte is None or dte <= 0:
        return None
    pick = next((t for t in curve if dte <= t.get("max_days", 0)), curve[-1])
    if pick.get("rate") is None:
        pick = next((t for t in curve if t.get("rate") is not None), None)
    return pick if (pick and pick.get("rate") is not None) else None


# ---------------------------------------------------------------------------
# Generic helpers# ---------------------------------------------------------------------------

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


def _has_market_sector(ticker: str) -> bool:
    parts = ticker.strip().split()
    if not parts:
        return False
    return parts[-1].upper() in MARKET_SECTOR_KEYWORDS


def resolve_security(user_ticker: str) -> str:
    if _has_market_sector(user_ticker):
        return user_ticker

    with with_session() as query:
        raw = query.bsrch(user_ticker)
    df = _ensure_security_column(_to_pandas(raw))
    if df.empty or "security" not in df.columns:
        return user_ticker
    first = df["security"].dropna()
    if first.empty:
        return user_ticker
    return str(first.iloc[0])


def normalize_iso_date(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if text == "":
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _normalize_dividend_date(value: object) -> Optional[date]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def get_next_dividend_date(
    ticker: str, snapshot: Optional[Dict[str, object]] = None
) -> Optional[date]:
    if snapshot:
        for field in DIVIDEND_FIELDS:
            value = snapshot.get(field)
            parsed = _normalize_dividend_date(value)
            if parsed is not None:
                return parsed

    if not ticker:
        return None

    with with_session() as query:
        raw = query.bdp([ticker], DIVIDEND_FIELDS)
    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, DIVIDEND_FIELDS)
    if df.empty:
        return None
    row = df.loc[df["security"] == ticker]
    if row.empty:
        row = df.iloc[[0]]
    record = row.iloc[0]
    for field in DIVIDEND_FIELDS:
        parsed = _normalize_dividend_date(record.get(field))
        if parsed is not None:
            return parsed
    return None


# ---------------------------------------------------------------------------
# Projected dividends — Bloomberg Dividend Forecast (BDS BDVD_ALL_PROJECTIONS)
# ---------------------------------------------------------------------------
# fetch_projected_dividend returns one fixed shape so callers never branch on
# source:
#   {"next": {"ex_date": date|None, "declared_date": date|None,
#             "dps": float|None} | None,
#    "schedule": [{"ex_date": date, "dps": float}, ...]}   # forward-ordered
# The bulk BDS parse fills next + the full forward schedule; the single-value
# BDP estimate fills next only (schedule stays []). The ex-div fire (P7) reads
# next.dps; when it is absent the fire falls back to the DVD_YLD yield heuristic,
# so a missing forecast degrades gracefully. Layer-2 pricing will consume the
# full schedule.

def _empty_projection() -> Dict[str, object]:
    return {"next": None, "schedule": []}


def fetch_projected_dividend(ticker: str) -> Dict[str, object]:
    """Forward dividend forecast for *ticker* in the fixed
    ``{"next": {...}, "schedule": [...]}`` shape documented above.

    Order of preference: the BDS bulk schedule (BDVD_ALL_PROJECTIONS, mnemonic
    DV140) for the full forward set, then the single-value BDP estimate for a
    next-only forecast, then the empty shape. Never raises.
    """
    if not ticker:
        return _empty_projection()
    schedule = _fetch_bdvd_schedule(ticker)
    if schedule:
        first = schedule[0]
        return {
            "next": {"ex_date": first.get("ex_date"),
                     "declared_date": first.get("declared_date"),
                     "dps": first.get("dps")},
            "schedule": [{"ex_date": r.get("ex_date"), "dps": r.get("dps")}
                         for r in schedule],
        }
    next_div = _fetch_bdvd_next_estimate(ticker)
    if next_div and next_div.get("dps") is not None:
        return {"next": next_div, "schedule": []}
    return _empty_projection()


def fetch_projected_dividends(tickers: list[str]) -> Dict[str, Dict[str, object]]:
    """Forward dividend forecasts for N tickers in ONE bulk-field BDP round trip
    — the batched analog of ``fetch_projected_dividend``, same fixed
    ``{"next": {...}, "schedule": [...]}`` shape per ticker, every requested
    ticker present (measured on the live book: 23 single-ticker calls ~8s ->
    one call ~1.4s).

    A multi-security bdp on the bulk BDVD field returns one array-of-row-dicts
    cell per security (confirmed live 2026-07: identical parse to the
    per-ticker call on every name). Per ticker the fallback chain matches the
    singular fetcher: bulk schedule, else the single-value next estimate, else
    the empty shape. Never raises; a failed batch call degrades every name to
    the empty shape (the ex-div fire then uses the yield heuristic).
    """
    out: Dict[str, Dict[str, object]] = {t: _empty_projection() for t in tickers}
    if not tickers:
        return out
    try:
        with with_session() as query:
            raw = query.bdp(list(tickers), [_BDVD_SCHEDULE_FIELD])
        df = _ensure_security_column(_to_pandas(raw))
    except Exception as exc:
        logger.warning("Batched BDVD schedule fetch failed: %s", exc)
        return out
    if df.empty or _BDVD_SCHEDULE_FIELD not in df.columns:
        return out
    df = df.set_index("security")
    for t in tickers:
        try:
            cell = df.loc[t, _BDVD_SCHEDULE_FIELD] if t in df.index else None
            schedule = _parse_bdvd_rows(cell)
        except Exception as exc:
            logger.warning("BDVD schedule parse for %s failed: %s", t, exc)
            continue
        if schedule:
            first = schedule[0]
            out[t] = {
                "next": {"ex_date": first.get("ex_date"),
                         "declared_date": first.get("declared_date"),
                         "dps": first.get("dps")},
                "schedule": [{"ex_date": r.get("ex_date"), "dps": r.get("dps")}
                             for r in schedule],
            }
            continue
        next_div = _fetch_bdvd_next_estimate(t)
        if next_div and next_div.get("dps") is not None:
            out[t] = {"next": next_div, "schedule": []}
    return out


# DV147 bulk forward-dividend schedule. Confirmed live (2026-06): this env's
# polars_bloomberg BQuery has NO ``.bds`` method, but a plain ``bdp`` on this bulk
# field returns, per security, a numpy array of row dicts
# ``{'Ex Date': datetime, 'Amount Per Share': float, 'Projected/Confirmed': str}``.
_BDVD_SCHEDULE_FIELD = "BDVD_PR_EX_DTS_DVD_AMTS_W_ANN"


def _parse_bdvd_rows(cell: object) -> list:
    """Parse a BDVD bulk cell (numpy array / list of row dicts) into the
    forward-ordered ``[{"ex_date": date, "declared_date": date|None, "dps": float},
    ...]`` contract. Drops rows missing an ex-date or carrying a non-positive
    amount. Never raises.
    """
    if cell is None:
        return []
    try:
        rows = list(cell)
    except TypeError:
        return []
    out: list = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        ex = _normalize_dividend_date(rec.get("Ex Date") or rec.get("Ex-Date"))
        try:
            dps = float(rec.get("Amount Per Share"))
        except (TypeError, ValueError):
            continue
        if ex is None or not (dps > 0):
            continue
        out.append({
            "ex_date": ex,
            "declared_date": _normalize_dividend_date(rec.get("Projected Declared Date")),
            "dps": dps,
        })
    out.sort(key=lambda r: r["ex_date"])
    return out


def _fetch_bdvd_schedule(ticker: str) -> list:
    """Full forward dividend schedule from the BDVD bulk field
    ``BDVD_PR_EX_DTS_DVD_AMTS_W_ANN`` (DV147), ex-date ascending:
    ``[{"ex_date", "declared_date", "dps"}, ...]``.

    Access is a plain ``bdp`` on the bulk field (this env has no ``.bds``) whose
    per-security cell is a numpy array of row dicts — see ``_parse_bdvd_rows``.
    Never raises; returns ``[]`` on any failure so the ex-div fire degrades to the
    yield heuristic and the pricing adapter to continuous-q.
    """
    if not ticker:
        return []
    try:
        with with_session() as query:
            raw = query.bdp([ticker], [_BDVD_SCHEDULE_FIELD])
        df = _ensure_security_column(_to_pandas(raw))
        if df.empty or _BDVD_SCHEDULE_FIELD not in df.columns:
            return []
        match = df.loc[df["security"] == ticker]
        cell = (match.iloc[0] if not match.empty else df.iloc[0])[_BDVD_SCHEDULE_FIELD]
        return _parse_bdvd_rows(cell)
    except Exception as exc:
        logger.warning("BDVD schedule fetch for %s failed: %s", ticker, exc)
        return []


def _fetch_bdvd_next_estimate(ticker: str) -> Optional[Dict[str, object]]:
    """Single next projected dividend (``{"ex_date", "declared_date", "dps"}``)
    from the BDP estimate fields (BDVD_NEXT_EST_EX_DT / BDVD_NEXT_EST_DECL_DT +
    the next per-share amount), used as a next-only fallback when the bulk
    schedule is unavailable.

    The per-share-amount token is confirmed via the same terminal probe; until
    then this returns ``None``. Never raises.
    """
    return None


# ---------------------------------------------------------------------------
# Batched underlying snapshot — NEW (single BDP round trip for all tickers)
# ---------------------------------------------------------------------------

def fetch_underlying_snapshots(tickers: list[str]) -> pd.DataFrame:
    """Batch BDP for the full ``UNDERLYING_FIELDS`` set across N tickers.

    Returns a pandas DataFrame indexed by Bloomberg ticker (e.g.
    ``'AAPL US Equity'``), one column per field in ``UNDERLYING_FIELDS``
    plus ``security_name`` (= the ``NAME`` field). Missing values are
    NaN, never None and never the string ``'N.A.'`` / ``'#N/A N/A'``.

    On any error (no session, request rejected, partial response), logs
    at WARNING and returns an empty DataFrame with the expected columns.
    Does NOT raise.

    ``tickers`` must already include the market sector suffix (e.g.
    ``'AAPL US Equity'``). Tickers are taken as given — this function
    performs no re-resolution; upstream construction owns correctness.
    """
    cols = ["security_name"] + list(UNDERLYING_FIELDS)
    if not tickers:
        return pd.DataFrame(columns=cols)

    try:
        with with_session() as query:
            raw = query.bdp(list(tickers), UNDERLYING_FIELDS)
    except Exception as exc:
        logger.warning("Batched BDP for underlyings failed: %s", exc)
        return pd.DataFrame(columns=cols)

    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, UNDERLYING_FIELDS)

    if df.empty:
        return pd.DataFrame(columns=cols)

    # Normalize Bloomberg sentinel strings to NaN. Bloomberg occasionally
    # ships "N.A." / "#N/A N/A" / "" rather than missing values.
    sentinels = {"N.A.", "#N/A N/A", "#N/A Field Not Applicable", ""}
    for col in UNDERLYING_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: pd.NA if isinstance(v, str) and v.strip() in sentinels else v
            )

    df["security_name"] = df.get("NAME", pd.NA)

    df = df.set_index("security")
    # Reindex so callers see a row per requested ticker, even if BBG dropped some
    df = df.reindex(list(tickers))
    return df[cols]


SECURITY_IDENTITY_FIELDS = ["PX_LAST", "NAME", "ID_ISIN", "CRNCY"]


def fetch_security_identity(tickers: list[str]) -> pd.DataFrame:
    """Batch BDP for the identity of each ticker: price, name, ISIN, currency.

    The venue-resolution pass uses this to decide whether a constructed ticker
    resolved to the RIGHT security — a mis-suffixed ticker can return data for
    a different instrument entirely (confirmed live: a wrong-venue suffix on a
    NYSE root returned a defunct local line with a matching root), so presence
    of data is not proof of identity; the ISIN is.

    Returns a DataFrame indexed by ticker with ``SECURITY_IDENTITY_FIELDS``
    columns, reindexed to the request (an unresolvable ticker is an all-NaN
    row). On any error logs at WARNING and returns an empty DataFrame with the
    expected columns. Does NOT raise.
    """
    cols = list(SECURITY_IDENTITY_FIELDS)
    if not tickers:
        return pd.DataFrame(columns=cols)

    try:
        with with_session() as query:
            raw = query.bdp(list(tickers), cols)
    except Exception as exc:
        logger.warning("Batched BDP for security identity failed: %s", exc)
        return pd.DataFrame(columns=cols)

    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, cols)
    if df.empty:
        return pd.DataFrame(columns=cols)

    sentinels = {"N.A.", "#N/A N/A", "#N/A Field Not Applicable", ""}
    for col in cols:
        df[col] = df[col].apply(
            lambda v: pd.NA if isinstance(v, str) and v.strip() in sentinels else v
        )

    df = df.set_index("security")
    df = df.reindex(list(tickers))
    return df[cols]


# ---------------------------------------------------------------------------
# SPX-relative betas — separate override-aware BDP for the exposure view
# ---------------------------------------------------------------------------
# Why this is a separate pull and not two more UNDERLYING_FIELDS:
# the exposure view needs every name's beta measured against ONE benchmark (SPX),
# so "net market exposure" is coherent. That requires the BETA_OVERRIDE_REL_INDEX=SPX
# override — but a BDP override applies to the whole request, and the batched
# UNDERLYING_FIELDS pull carries BETA_ADJ_OVERRIDABLE (the snapshot's default beta the
# legacy weighted-beta chip / signals read). Putting the SPX override on that batched
# call would silently re-point the legacy beta for non-US names (confirmed live:
# NESN 0.84 -> 0.35 vs SPX). So the SPX betas come from their own override-aware call,
# leaving the default beta untouched.
#
# Source-field note (confirmed live): the EQY_BETA / EQY_RAW_BETA fields IGNORE
# BETA_OVERRIDE_REL_INDEX, so an SPX-relative beta cannot come from them. The
# *_OVERRIDABLE fields DO respect it; at their default they equal EQY_BETA (same
# 2y-weekly methodology), and the override only changes the benchmark to SPX. The
# results are stored under the EQY_BETA / EQY_RAW_BETA column names the exposure
# aggregation reads (US names already default to SPX; non-US names move to SPX).
_SPX_BETA_OVERRIDE = [("BETA_OVERRIDE_REL_INDEX", "SPX Index")]
_SPX_BETA_SOURCE_FIELDS = ["BETA_ADJ_OVERRIDABLE", "BETA_RAW_OVERRIDABLE"]
SPX_BETA_COLUMNS = {"BETA_ADJ_OVERRIDABLE": "EQY_BETA",
                    "BETA_RAW_OVERRIDABLE": "EQY_RAW_BETA"}


def fetch_spx_betas(tickers: list[str]) -> pd.DataFrame:
    """SPX-relative 2y-weekly betas (adjusted + raw) per underlying.

    Returns a DataFrame indexed by Bloomberg ticker with columns ``EQY_BETA``
    (adjusted) and ``EQY_RAW_BETA`` (raw), both measured against SPX via the
    BETA_OVERRIDE_REL_INDEX override (see the module comment above for why this is a
    separate pull and why the values are sourced from the *_OVERRIDABLE fields).
    Missing values are NaN. On any error (no session, override unsupported, partial
    response) logs at WARNING and returns an empty DataFrame with the expected
    columns. Never raises.
    """
    cols = ["EQY_BETA", "EQY_RAW_BETA"]
    if not tickers:
        return pd.DataFrame(columns=cols)

    try:
        with with_session() as query:
            raw = query.bdp(list(tickers), _SPX_BETA_SOURCE_FIELDS,
                            overrides=_SPX_BETA_OVERRIDE)
    except TypeError:
        # polars_bloomberg without override support — an SPX-relative beta is
        # impossible here, so surface nothing rather than a wrong default-index beta.
        logger.warning("SPX-beta pull: bdp overrides unsupported; SPX betas unavailable")
        return pd.DataFrame(columns=cols)
    except Exception as exc:
        logger.warning("SPX-beta pull failed: %s", exc)
        return pd.DataFrame(columns=cols)

    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, _SPX_BETA_SOURCE_FIELDS)
    if df.empty:
        return pd.DataFrame(columns=cols)

    sentinels = {"N.A.", "#N/A N/A", "#N/A Field Not Applicable", ""}
    for col in _SPX_BETA_SOURCE_FIELDS:
        df[col] = df[col].apply(
            lambda v: pd.NA if isinstance(v, str) and v.strip() in sentinels else v
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns=SPX_BETA_COLUMNS).set_index("security")
    df = df.reindex(list(tickers))
    return df[cols]


# ---------------------------------------------------------------------------
# Analyst-note dates — batched BDP for D3
# ---------------------------------------------------------------------------

def fetch_analyst_note_dates(
    query,
    tickers: list[str],
) -> pd.DataFrame:
    """Batch BDP for INTERVAL_END_VALUE_DATE across N tickers with the analyst-source
    analyst-note override pair.

    Overrides:
        BE998 = UBS
        PX395 = Best Analyst Rating

    Returns a DataFrame indexed by ticker with one column: 'analyst_note_date'.
    Coerces to pandas Timestamp / NaT.

    ``query`` is an open ``polars_bloomberg.BQuery`` session (caller-owned,
    mirroring the override-aware ``fetch_analyst_data``). On any error
    (no session, request rejected, override unsupported) logs at WARNING and
    returns an empty DataFrame with the expected column — it fails CLOSED,
    never retrying without the override pair (a bare retry would return a
    non-provider note date labeled as provider data downstream). Does NOT
    raise.

    ``tickers`` must already include the market sector suffix (e.g.
    ``'AAPL US Equity'``); the parser produces these correctly.
    """
    field = "INTERVAL_END_VALUE_DATE"
    cols = ["analyst_note_date"]
    if not tickers:
        return pd.DataFrame(columns=cols)

    try:
        bdp_fn = getattr(query, "bdp", None)
        if bdp_fn is None:
            return pd.DataFrame(columns=cols)
        # polars_bloomberg BQuery.bdp overrides: list[tuple] | None
        try:
            raw = bdp_fn(
                list(tickers),
                [field],
                overrides=[("BE998", "UBS"), ("PX395", "Best Analyst Rating")],
            )
        except TypeError:
            # bdp without override support — the provider-scoped note date is
            # impossible here; surface nothing rather than a bare-retry date.
            logger.warning("Analyst-note pull: bdp overrides unsupported; "
                           "note dates unavailable")
            return pd.DataFrame(columns=cols)
    except Exception as exc:
        logger.warning("Batched BDP for analyst-note dates failed: %s", exc)
        return pd.DataFrame(columns=cols)

    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, [field])
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Normalize Bloomberg sentinel strings to NaN before date coercion.
    sentinels = {"N.A.", "#N/A N/A", "#N/A Field Not Applicable", ""}
    df[field] = df[field].apply(
        lambda v: pd.NA if isinstance(v, str) and v.strip() in sentinels else v
    )
    df["analyst_note_date"] = pd.to_datetime(df[field], errors="coerce")

    df = df.set_index("security")
    df = df.reindex(list(tickers))
    return df[cols]


# ---------------------------------------------------------------------------
# Option snapshot + batch wrapper
# ---------------------------------------------------------------------------
#
# ``fetch_option_snapshot`` normalizes BBG field aliases (DAYS_EXPIRE → dte,
# MID → PX_MID, IVOL_MID/IVOL → iv_mid, etc.). That logic lives in
# ``_normalize_option_fields`` so both the per-call function and the batch
# wrapper share it.

_OPTION_SNAPSHOT_OUTPUT_COLS = [
    "security",
    "BID",
    "ASK",
    "PX_MID",
    "PX_LAST",
    "IVOL_MID",
    "IVOL",
    "DAYS_TO_EXPIRATION",
    "DAYS_EXPIRE",
    "OPT_STRIKE_PX",
    "OPT_PUT_CALL",
    "DELTA_MID_RT",
    "THETA",
    "THETA_MID",
    "GAMMA",
    "VEGA",
    "RHO",
    "OPEN_INT",
    "PX_VOLUME",
    "OPTION_EXERCISE_TYPE_REALTIME",
    "dte",
    "delta_mid",
    "theta",
    "gamma",
    "vega",
    "rho",
    "iv_mid",
    "oi",
    "volume",
    "style",
]


def _normalize_exercise_style(value: object) -> Optional[str]:
    """Map a Bloomberg ``OPTION_EXERCISE_TYPE_REALTIME`` value to the pricing
    engine's style key. Returns ``'American'`` / ``'European'`` or ``None`` for
    anything unrecognized (incl. sentinels / NaN) so the adapter can decide a
    safe default rather than mis-route the registry.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text.startswith("amer"):
        return "American"
    if text.startswith("euro"):
        return "European"
    return None


def _normalize_option_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize option fields on a BDP frame.

    Coalesces MID into PX_MID, casts numeric fields, and exposes lowercase
    canonical columns (dte, delta_mid, theta, gamma, vega, rho, iv_mid, style).
    """
    if "MID" in df.columns:
        df["PX_MID"] = pd.to_numeric(df["PX_MID"], errors="coerce")
        mid_values = pd.to_numeric(df["MID"], errors="coerce")
        df["PX_MID"] = df["PX_MID"].fillna(mid_values)
    if "MID" in df.columns:
        df = df.drop(columns=["MID"])

    numeric_fields = [
        "BID",
        "ASK",
        "PX_MID",
        "PX_LAST",
        "IVOL_MID",
        "IVOL",
        "DAYS_TO_EXPIRATION",
        "DAYS_EXPIRE",
        "OPT_STRIKE_PX",
        "DELTA_MID_RT",
        "THETA",
        "THETA_MID",
        "GAMMA",
        "VEGA",
        "RHO",
        "OPEN_INT",
        "PX_VOLUME",
    ]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    dte = pd.to_numeric(df.get("DAYS_TO_EXPIRATION"), errors="coerce")
    if "DAYS_EXPIRE" in df.columns:
        dte_alt = pd.to_numeric(df.get("DAYS_EXPIRE"), errors="coerce")
        dte = dte.fillna(dte_alt)
    df["dte"] = dte

    df["delta_mid"] = pd.to_numeric(df.get("DELTA_MID_RT"), errors="coerce")
    theta = pd.to_numeric(df.get("THETA"), errors="coerce")
    if "THETA_MID" in df.columns:
        theta = theta.fillna(pd.to_numeric(df.get("THETA_MID"), errors="coerce"))
    df["theta"] = theta
    df["gamma"] = pd.to_numeric(df.get("GAMMA"), errors="coerce")
    df["vega"] = pd.to_numeric(df.get("VEGA"), errors="coerce")
    df["rho"] = pd.to_numeric(df.get("RHO"), errors="coerce")
    iv_mid = pd.to_numeric(df.get("IVOL_MID"), errors="coerce")
    if "IVOL" in df.columns:
        iv_mid = iv_mid.fillna(pd.to_numeric(df.get("IVOL"), errors="coerce"))
    df["iv_mid"] = iv_mid
    # Per-contract open interest / volume, coalesced from whichever mnemonic the
    # terminal populates (OPEN_INT / PX_VOLUME the primaries; alternates picked up
    # if a request adds them). Absent everywhere -> NaN, which the view shows as '—'.
    oi = pd.to_numeric(df.get("OPEN_INT"), errors="coerce")
    for alt in ("OPT_OPEN_INTEREST", "OPEN_INTEREST"):
        if alt in df.columns:
            oi = oi.fillna(pd.to_numeric(df.get(alt), errors="coerce"))
    df["oi"] = oi
    volume = pd.to_numeric(df.get("PX_VOLUME"), errors="coerce")
    for alt in ("VOLUME", "VOLM"):
        if alt in df.columns:
            volume = volume.fillna(pd.to_numeric(df.get(alt), errors="coerce"))
    df["volume"] = volume
    # Exercise style normalized to the engine's registry key ('American' /
    # 'European'); anything BBG doesn't classify becomes None (the adapter then
    # defaults + warns rather than mis-routing).
    df["style"] = df.get("OPTION_EXERCISE_TYPE_REALTIME").map(_normalize_exercise_style) \
        if "OPTION_EXERCISE_TYPE_REALTIME" in df.columns else None
    return df


def fetch_option_snapshot(option_tickers: list[str]) -> pd.DataFrame:
    if not option_tickers:
        return pd.DataFrame(columns=_OPTION_SNAPSHOT_OUTPUT_COLS)

    with with_session() as query:
        raw = query.bdp(option_tickers, OPTION_SNAPSHOT_FIELDS)
    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, OPTION_SNAPSHOT_FIELDS)
    df = _normalize_option_fields(df)
    return df[_OPTION_SNAPSHOT_OUTPUT_COLS]


def fetch_option_snapshots(option_tickers: list[str]) -> pd.DataFrame:
    """Batch BDP for ``OPTION_SNAPSHOT_FIELDS`` across N option tickers in
    a single Bloomberg round trip.

    Returns a pandas DataFrame indexed by option ticker (e.g.
    ``'BX US 5/15/26 P105 Equity'``). Greek columns follow the source
    adapter's normalization (``DAYS_EXPIRE`` → ``dte``, ``MID`` / ``PX_MID``
    coalesced, ``IVOL_MID`` / ``IVOL`` coalesced, ``DELTA_MID_RT`` exposed
    as ``delta_mid``). Missing values are NaN.

    On any error (no session, request rejected, partial response), logs at
    WARNING and returns an empty DataFrame with the expected columns.
    Does NOT raise.

    ``option_tickers`` must be Bloomberg-formatted (e.g.
    ``'BX US 5/15/26 P105 Equity'``). The parser produces these correctly.
    """
    cols = [c for c in _OPTION_SNAPSHOT_OUTPUT_COLS if c != "security"]
    if not option_tickers:
        return pd.DataFrame(columns=cols)

    try:
        with with_session() as query:
            raw = query.bdp(list(option_tickers), OPTION_SNAPSHOT_FIELDS)
    except Exception as exc:
        logger.warning("Batched BDP for options failed: %s", exc)
        return pd.DataFrame(columns=cols)

    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, OPTION_SNAPSHOT_FIELDS)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = _normalize_option_fields(df)
    df = df.set_index("security")
    df = df.reindex(list(option_tickers))
    # Same column order as fetch_option_snapshot, minus the 'security' col
    return df[cols]


# ---------------------------------------------------------------------------
# Option chain enumeration — OPT_CHAIN via a plain bdp on the bulk field
# ---------------------------------------------------------------------------
# The listed option chain for an underlier is read the same way as the BDVD
# dividend schedule: a plain bdp on the bulk OPT_CHAIN field (this env has no
# .bds), whose per-security cell is an array of row dicts
# {'Security Description': '<canonical option ticker>'}. Enumerating on the
# underlier's EQUITY ticker returns the true option root even when it differs
# (e.g. NESN SW -> NES1 SW), which is what lets the snapshot layer recover a held
# option whose ticker was built from the equity root.
_OPT_CHAIN_FIELD = "OPT_CHAIN"
_OPT_CHAIN_DESC_KEY = "Security Description"


def _parse_opt_chain_cell(cell: object) -> list[str]:
    """Parse an OPT_CHAIN bulk cell (array / list of row dicts) into the list of
    canonical option-ticker strings, in listed order. Drops rows without a
    description. Never raises."""
    if cell is None:
        return []
    try:
        rows = list(cell)
    except TypeError:
        return []
    out: list[str] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        desc = rec.get(_OPT_CHAIN_DESC_KEY)
        if desc is None:
            continue
        text = str(desc).strip()
        if text:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# OVDV published surface — the reachable near-ATM moneyness x tenor grid
# ---------------------------------------------------------------------------
# Bloomberg's published vol surface (the OVDV grid) is read as the
# ``<N>MTH_IMPVOL_<MNY>%MNY_DF`` field family. Confirmed live (2026-07): only the
# near-the-money region resolves via BDP — tenors 3M/6M/12M x moneyness 90-110%; the
# 1M/2M tenors and the 80%/120% wings return nothing. Those gaps are reported as None,
# never interpolated. Used as an independent cross-check against our own fitted surface.
_OVDV_TENOR_MONTHS = (3, 6, 12)
_OVDV_MONEYNESS = (90.0, 95.0, 100.0, 105.0, 110.0)


def _ovdv_field(months: int, mny: float) -> str:
    return "%dMTH_IMPVOL_%.1f%%MNY_DF" % (months, mny)


def fetch_ovdv_grid(underlier: str) -> dict:
    """Bloomberg's published near-ATM vol grid for *underlier* (its equity ticker) as
    ``{(tenor_months, moneyness_pct): iv_percent | None}`` over the reachable region
    (3M/6M/12M x 90-110%). ``None`` marks a value BBG did not publish — callers must
    not interpolate it. Returns an all-None grid on any failure. Never raises."""
    keys = [(n, m) for n in _OVDV_TENOR_MONTHS for m in _OVDV_MONEYNESS]
    if not underlier:
        return {}
    try:
        with with_session() as query:
            raw = query.bdp([underlier], [_ovdv_field(n, m) for n, m in keys])
        df = _ensure_security_column(_to_pandas(raw))
    except Exception as exc:
        logger.warning("OVDV grid fetch for %s failed: %s", underlier, exc)
        return {k: None for k in keys}
    if df.empty:
        return {k: None for k in keys}
    match = df.loc[df["security"] == underlier]
    rec = (match.iloc[0] if not match.empty else df.iloc[0])
    out: dict = {}
    for n, m in keys:
        field = _ovdv_field(n, m)
        v = rec.get(field) if field in rec.index else None
        try:
            out[(n, m)] = float(v) if (v is not None and float(v) == float(v)) else None
        except (TypeError, ValueError):
            out[(n, m)] = None
    return out


def fetch_option_chain(underlier: str) -> list[str]:
    """The listed option chain for *underlier* (its EQUITY bbg ticker) as a list
    of canonical option-ticker strings, e.g. ``'NES1 SW 07/03/26 C67 Equity'``.

    Access is a plain ``bdp`` on the bulk ``OPT_CHAIN`` field (this env has no
    ``.bds``); the per-security cell is an array of row dicts — see
    ``_parse_opt_chain_cell``. Returns ``[]`` on any failure (no session, empty
    response, parse error) so callers degrade to the constructed best-effort
    ticker rather than raising. Never raises.
    """
    if not underlier:
        return []
    try:
        with with_session() as query:
            raw = query.bdp([underlier], [_OPT_CHAIN_FIELD])
        df = _ensure_security_column(_to_pandas(raw))
        if df.empty or _OPT_CHAIN_FIELD not in df.columns:
            return []
        match = df.loc[df["security"] == underlier]
        cell = (match.iloc[0] if not match.empty else df.iloc[0])[_OPT_CHAIN_FIELD]
        return _parse_opt_chain_cell(cell)
    except Exception as exc:
        logger.warning("OPT_CHAIN fetch for %s failed: %s", underlier, exc)
        return []


# ---------------------------------------------------------------------------
# BDH historical fetchers
# ---------------------------------------------------------------------------

def _bdh_to_dict(
    df_in: object,
    field: str,
) -> dict[str, "pd.Series"]:
    """Coerce a polars-bloomberg BDH response into ``{ticker: Series}``."""
    df = _to_pandas(df_in)
    if df is None or df.empty:
        return {}

    # polars-bloomberg ships BDH responses with ['security', 'date', <field>]
    # columns. Defend against alternate shapes.
    df = _ensure_security_column(df)
    if "date" not in df.columns:
        for cand in ["DATE", "Date", "AS_OF_DATE", "asof_date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    if "date" not in df.columns or field not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df[field] = pd.to_numeric(df[field], errors="coerce")

    out: dict[str, pd.Series] = {}
    for ticker, sub in df.groupby("security"):
        s = sub.set_index("date")[field].sort_index()
        out[str(ticker)] = s
    return out


def fetch_price_history(
    tickers: list[str],
    lookback_days: int = 365,
) -> dict[str, "pd.Series"]:
    """BDH PX_LAST for N tickers over the lookback window. Returns
    ``{ticker: pd.Series of daily close prices indexed by date}``.

    On error, logs WARNING and returns ``{}``. Does NOT raise. Tickers
    that come back with no rows are silently omitted from the result.
    """
    if not tickers:
        return {}
    end = date.today()
    start = end - pd.Timedelta(days=lookback_days).to_pytimedelta()
    try:
        with with_session() as q:
            raw = q.bdh(
                securities=list(tickers),
                fields=["PX_LAST"],
                start_date=start,
                end_date=end,
            )
    except Exception as exc:
        logger.warning("BDH price-history fetch failed: %s", exc)
        return {}
    return _bdh_to_dict(raw, "PX_LAST")


def fetch_iv_history(
    tickers: list[str],
    lookback_days: int = 365,
    iv_field: str = "3MTH_IMPVOL_100.0%MNY_DF",
) -> dict[str, "pd.Series"]:
    """BDH for the specified IV field over lookback. Returns
    ``{ticker: pd.Series of daily IV indexed by date}``.

    Default ``iv_field`` is ``3MTH_IMPVOL_100.0%MNY_DF`` (3M ATM IV) —
    BDH-historical of 3M ATM IV does populate even when the BDP point-
    in-time of the same field doesn't. On error, returns ``{}``.
    """
    if not tickers:
        return {}
    end = date.today()
    start = end - pd.Timedelta(days=lookback_days).to_pytimedelta()
    try:
        with with_session() as q:
            raw = q.bdh(
                securities=list(tickers),
                fields=[iv_field],
                start_date=start,
                end_date=end,
            )
    except Exception as exc:
        logger.warning("BDH IV-history fetch failed: %s", exc)
        return {}
    return _bdh_to_dict(raw, iv_field)


# ---------------------------------------------------------------------------
# Startup probe — NEW
# ---------------------------------------------------------------------------

def is_bloomberg_available(probe_ticker: str = "AAPL US Equity") -> bool:
    """Probe the Bloomberg connection at startup. Returns True iff a BDP
    call for PX_LAST against ``probe_ticker`` succeeds.

    Wraps every failure path: missing ``polars_bloomberg`` install, no
    Bloomberg Terminal running, network/auth errors. Logs the failure
    reason at DEBUG. Never raises.
    """
    try:
        with with_session() as q:
            df = q.bdp(securities=[probe_ticker], fields=["PX_LAST"])
        # polars-bloomberg returns a polars DataFrame; check it has a row
        return df is not None and len(df) > 0
    except Exception as exc:
        logger.debug("Bloomberg probe failed: %s", exc)
        return False
