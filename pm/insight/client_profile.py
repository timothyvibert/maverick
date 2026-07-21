"""Per-account client behavioural profile from the trade history.

A read-only summary of *how an account trades* — strategy posture, direction,
tenor, sector lean, sizing, cadence — derived purely from the account's own
trade rows plus the already-loaded underlying snapshot. It is computed once in
the load path (mirroring the exposure / scenario passes) and stored on
``AccountState.client_profile``; the UI later only reads it. No Bloomberg, no
recompute.

Design boundary — only the *robust flow tier* lives here. Every dimension is
derivable from a single self-describing trade row (its lifecycle action,
instrument, quantity, principal, date), so it survives a trade window that is
shallower than the positions' tenor. Dimensions that need opening/closing trades
to be paired (holding period, roll behaviour, realised income) are deliberately
absent — they belong to a later, separately-gated pass and are not declared here.

Strategy posture is read from the trade flow alone (what the account *opens*).
It never claims a short call is "covered" or "naked": that is a property of the
holdings at the time of the trade, which the flow does not carry, so conflating
it with the current structure list would assert facts the data does not contain.

The load-bearing piece is the lifecycle normalisation: ``option_lifecycle_action``
arrives as a raw string with no enumerated vocabulary, and a wrong mapping is
invisible and poisons every downstream count. So the canonical set is explicit,
``buy_sell`` is cross-checked against it, cancelled rows are dropped, and any
string outside the canonical set fails loud rather than being silently bucketed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import median
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle vocabulary (the gate)
# ---------------------------------------------------------------------------

# Canonical option lifecycle actions -> (side, stage). ``side`` is the order
# verb (it agrees with the Buy/Sell column); ``stage`` is whether the trade
# opens or closes a position — the part Buy/Sell does NOT encode, which is why
# this map is keyed on the lifecycle string and not derivable from Buy/Sell.
_LIFECYCLE: dict[str, tuple[str, str]] = {
    "Buy to Open": ("buy", "open"),
    "Sell to Open": ("sell", "open"),
    "Sell to Close": ("sell", "close"),
    "Buy to Close": ("buy", "close"),
}

# Recognised but directionally-neutral lifecycle events: terminal mechanics, not
# a discretionary open/close. Kept out of the open/close flow counts, but valid
# (they must not trip the fail-loud guard).
_NEUTRAL_ACTIONS: frozenset[str] = frozenset(
    {"Exercise", "Assignment", "Expiration", "Expired", "Expire", "Exercised", "Assigned"}
)

_OPTION_CLASS = "option"
_EQUITY_CLASSES = ("equity", "fund_etf")


class UnknownLifecycleAction(ValueError):
    """A trade carries an ``option_lifecycle_action`` outside the canonical set.

    Raised by ``_canonical_action`` rather than silently bucketed: an
    unrecognised lifecycle string would quietly mis-state strategy / direction /
    coverage. ``normalize_trades`` catches it per ROW: the offending row is
    staged ``"unknown"`` (excluded from every open/close-based read, kept for
    flow-level ones), the raw string reaches the profile's ``degraded_reason``
    and the load notes — the rest of the account's profile stays live instead
    of nulling wholesale."""


# ---------------------------------------------------------------------------
# Sector lookup — self-contained, matching pm.risk.exposure._sector_of semantics
# ---------------------------------------------------------------------------
# Replicated here (rather than imported from pm.risk) so the insight layer does
# not depend on the risk layer — that would invert the documented
# insight -> structures -> risk direction. Behaviour is kept identical to the
# exposure helper so the two bases never disagree on a name's sector; a future
# pass that already touches exposure should lift both into one shared utility.
SECTOR_FIELD = "GICS_SECTOR_NAME"
UNCLASSIFIED = "Unclassified"

# Provisional dials. The representative ~35-account / multi-year extract needed
# to calibrate these is not available, so they are sized by judgement, not data,
# and are expected to move. They are deliberately monotone so coverage / sample
# size only ever raises confidence as it grows.
_SHORT_DTE_MAX = 45      # days; at-open tenor <= this reads "short-dated"
_SWING_DTE_MAX = 365     # days; <= this reads "swing"; longer reads "LEAPS"
_CONF_MED_MIN = 4        # sample size for a "medium"-confidence dimension
_CONF_HIGH_MIN = 10      # sample size for a "high"-confidence dimension
_BAND_MED_MIN = 6        # trades for a "medium" overall coverage band
_BAND_HIGH_MIN = 15      # trades (with paired evidence) for a "high" band
_BAND_HIGH_PAIRED = 0.34  # paired fraction also required for "high"
_BAND_WINDOW_MIN_DAYS = 30  # a span shorter than this caps the band at "low" —
                            # a dense burst is not a behavioural history
_CADENCE_MIN_WINDOW_DAYS = 30  # shorter spans can't be annualised into a rate
_TOP_N = 5               # sector / name lean is reported top-N

# Fragile-tier dials (provisional — behaviour-tested, not calibrated).
_HOLDS_MIN_POSITIONS = 3   # held option positions with a derivable open to read a median
_ROLLS_MIN_CLOSES = 6      # closing trades needed to characterise a roll tendency
_ROLL_WINDOW_BD = 1        # a reopen within this many FORWARD business days of a
                           # close counts as a roll (captures Fri-close/Mon-reopen;
                           # a prior-day open is not a roll of a later close)
_ROLL_HIGH = 0.5           # roll-like share at/above this reads "rolls"
_ROLL_LOW = 0.25           # roll-like share below this reads "closes_early"


def _sector_of(underlyings, ticker: Optional[str]) -> str:
    """GICS sector for one underlying (in BBG-ticker form), or 'Unclassified'
    when the snapshot, name, or field is absent or blank — never raises. Index
    underliers ("… Index" tickers) carry no GICS sector by nature and read
    "Index", not missing-data (kept identical to the exposure helper)."""
    fallback = ("Index" if str(ticker or "").strip().endswith(" Index")
                else UNCLASSIFIED)
    if underlyings is None or not ticker:
        return fallback
    try:
        if SECTOR_FIELD not in getattr(underlyings, "columns", []):
            return fallback
        if ticker not in underlyings.index:
            return fallback
        val = underlyings.loc[ticker, SECTOR_FIELD]
    except Exception:
        return fallback
    # A duplicated index yields a Series — take the first row defensively.
    if hasattr(val, "iloc"):
        val = val.iloc[0] if len(val) else None
    if val is None:
        return fallback
    try:
        if pd.isna(val):
            return fallback
    except (TypeError, ValueError):
        pass
    return str(val).strip() or fallback


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class Coverage:
    """How much trade history backs the profile, and how much to trust it.

    ``paired_fraction`` is the in-window round-trip rate: option contracts whose
    history holds BOTH an opening and a closing trade, over all contracts seen in
    the trades (no FIFO). It is a strict, different measure from
    ``positions_with_derivable_open_fraction`` — held option positions whose
    contract has any opening trade in the window — which carries the
    opens-predate-window caveat separately."""
    n_trades: int
    n_opens: int
    n_closes: int
    n_excluded_cancels: int
    window_days: Optional[int]
    paired_fraction: float
    positions_with_derivable_open_fraction: Optional[float]
    band: str  # 'low' | 'medium' | 'high'


@dataclass
class StrategyBias:
    """Posture mix over *opening* trades (what the account establishes): option
    opens bucketed by side x right, plus equity buys as long stock. Weights sum
    to 1 over opening trades; empty when there are none."""
    weights: dict = field(default_factory=dict)
    n_opening: int = 0
    confidence: str = "low"


@dataclass
class DirectionBias:
    """Directional lean over opening trades. ``call_put_skew`` is over option
    opens only; ``long_short_skew`` is by opened delta sign (long call / short
    put / long stock are long; short call / long put are short). Both in
    [-1, +1]; None when there is no opening flow to read."""
    call_put_skew: Optional[float] = None
    long_short_skew: Optional[float] = None
    confidence: str = "low"


@dataclass
class TenorPref:
    """At-open tenor preference over option opens — days to expiry at the trade
    date. ``bucket`` reads the median (short / swing / leaps). Non-option opens
    carry no expiry and are skipped."""
    median_dte_at_open: Optional[float] = None
    bucket: Optional[str] = None
    distribution: dict = field(default_factory=dict)
    n_opens: int = 0
    confidence: str = "low"


@dataclass
class SectorLean:
    """Where the account trades, weighted by trade count. ``top`` is by GICS
    sector (needs the snapshot); ``by_name`` is the always-available underlying
    lean. ``classified_fraction`` says how much of the flow resolved to a real
    sector — exited names absent from the snapshot read 'Unclassified'."""
    top: list = field(default_factory=list)        # [(sector, weight), ...]
    by_name: list = field(default_factory=list)    # [(underlying, weight), ...]
    classified_fraction: float = 0.0
    confidence: str = "low"


@dataclass
class Sizing:
    """Trade sizing and name concentration. ``median_principal`` is per-trade
    absolute principal; ``concentration_hhi`` is the Herfindahl over underlyings
    by absolute principal (1/N diversified .. 1 single-name)."""
    median_principal: Optional[float] = None
    concentration_hhi: Optional[float] = None
    confidence: str = "low"


@dataclass
class Cadence:
    """Trading frequency. ``trades_per_month`` annualises the account's own
    trade-date span; ``clustering`` (earnings / month-end / post-move) is left
    'n/a' for now — it needs a deeper, separately-gated pass."""
    trades_per_month: Optional[float] = None
    clustering: str = "n/a"
    confidence: str = "low"


@dataclass
class HoldingPeriod:
    """Current-book holding-period proxy: median days-held-so-far across held
    option positions that have a derivable opening trade. DOUBLY LIMITED — it is
    current-book only (survivorship-biased toward longer holds: positions that were
    short-held and already closed are absent) AND counts only positions whose
    contract has a derivable open (``Position.days_held`` joins the account's own
    trades first; an open found only in another account's history is used as a
    marked book-transfer fallback and counted here in ``n_transfer_inferred``).
    Not a realised holding period; None below the sample floor."""
    median_days_held: Optional[float] = None
    n_positions: int = 0
    confidence: str = "low"
    n_transfer_inferred: int = 0


@dataclass
class RollBehavior:
    """Roll tendency from a HEURISTIC — not a verified FIFO/LIFO pairing. A
    'rolled close' is a closing trade with a same-underlying, same-right reopen on
    a different contract within a day (STC↔BTO long, BTC↔STO short). ``tendency``
    reads the roll-like share of closing activity; ``n_events`` is the observed
    roll-like count, surfaced even when the tendency is too thin to characterise."""
    tendency: str = "unknown"   # 'rolls' | 'closes_early' | 'mixed' | 'unknown'
    n_events: int = 0
    n_closes: int = 0
    confidence: str = "low"


@dataclass
class ClientProfile:
    """The behavioural profile for one account — a robust flow tier plus a fragile
    lifecycle tier (holds proxy + roll heuristic; realised income deferred)."""
    account: str
    coverage: Coverage
    strategy_bias: StrategyBias
    direction_bias: DirectionBias
    tenor_pref: TenorPref
    sector_lean: SectorLean
    sizing: Sizing
    cadence: Cadence
    holding_period: Optional[HoldingPeriod] = None
    roll_behavior: Optional[RollBehavior] = None
    headline: str = ""
    # Set when trade rows carried unrecognised lifecycle actions: those rows are
    # excluded from open/close-based reads and this names them (rendered on the
    # section + echoed into the load notes). None = clean vocabulary.
    degraded_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _confidence(n: Optional[int]) -> str:
    """Provisional per-dimension confidence from its sample size (monotone)."""
    if n is None or n < _CONF_MED_MIN:
        return "low"
    if n < _CONF_HIGH_MIN:
        return "medium"
    return "high"


def _band(n_trades: int, paired_fraction: float,
          window_days: Optional[int] = None) -> str:
    """Overall coverage band from trade count (primary), paired evidence
    (secondary), and a window-span floor: a history spanning under
    ``_BAND_WINDOW_MIN_DAYS`` caps at 'low' regardless of trade count — a short
    dense burst is activity, not a behavioural read (the same reason cadence
    refuses to annualise it). Monotone non-decreasing in every input, so more
    history never lowers the band. Thresholds are provisional (no calibration
    extract available). ``window_days is None`` (undated trades) does not cap —
    absence of dates is not evidence of a burst."""
    if window_days is not None and window_days < _BAND_WINDOW_MIN_DAYS:
        return "low"
    pf = paired_fraction or 0.0
    if n_trades >= _BAND_HIGH_MIN and pf >= _BAND_HIGH_PAIRED:
        return "high"
    if n_trades >= _BAND_MED_MIN:
        return "medium"
    return "low"


def _as_date(v) -> Optional[date]:
    """A python date from a date / datetime / Timestamp cell, else None."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.Timestamp(v).date()
    except Exception:
        return None


def _blank(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip() == ""


def _canonical_action(action) -> Optional[tuple[str, str]]:
    """(side, stage) for a lifecycle string, ('', 'neutral') for a recognised
    terminal event, or None when the cell is blank. Raises on any other string."""
    if _blank(action):
        return None
    a = str(action).strip()
    if a in _LIFECYCLE:
        return _LIFECYCLE[a]
    if a in _NEUTRAL_ACTIONS:
        return ("", "neutral")
    raise UnknownLifecycleAction(a)


# ---------------------------------------------------------------------------
# Normalisation (the gate)
# ---------------------------------------------------------------------------

def normalize_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """Validate and tag the account's trades, returning
    (clean_frame, n_cancels, unknown_actions).

    - Cancelled / busted rows (``cancel_code`` populated) are dropped.
    - Each row gets ``_side`` / ``_stage``: from the lifecycle map for options;
      from Buy/Sell for equities (buy -> open/long, sell -> exit).
    - ``buy_sell`` is cross-checked against the mapped side; a disagreement is
      logged, not fatal.
    - Any ``option_lifecycle_action`` outside the canonical set is NEVER
      silently bucketed: the row is staged ``"unknown"`` (excluded from every
      open/close-based read, kept for flow-level ones) and the raw string is
      returned in ``unknown_actions`` so the profile can stamp its degraded
      reason and the load notes can name it. One bad vocabulary row no longer
      nulls the whole profile.
    """
    if trades is None or getattr(trades, "empty", True):
        return (trades if trades is not None else pd.DataFrame()), 0, []

    df = trades.copy()

    # Drop cancels first (a real cancel_code surfaces as a populated cell; the
    # column is all-NaN when there are no cancels).
    n_cancels = 0
    if "cancel_code" in df.columns:
        cancelled = df["cancel_code"].notna()
        n_cancels = int(cancelled.sum())
        df = df[~cancelled].copy()

    sides: list[str] = []
    stages: list[str] = []
    mismatches = 0
    unknown_actions: list[str] = []
    has_lc = "option_lifecycle_action" in df.columns
    has_bs = "buy_sell" in df.columns

    for _, row in df.iterrows():
        ac = str(row.get("asset_class") or "").strip().lower()
        bs_word = str(row.get("buy_sell") or "").strip().lower().split(" ")[0] if has_bs else ""
        try:
            mapped = _canonical_action(row.get("option_lifecycle_action")) if has_lc else None
        except UnknownLifecycleAction as exc:
            unknown_actions.append(str(exc))
            sides.append(bs_word)
            stages.append("unknown")
            continue

        if ac == _OPTION_CLASS:
            if mapped is None:
                # An option without a lifecycle action — can't place it on the
                # open/close axis; keep it (for sector/sizing) but stage-neutral.
                side, stage = bs_word, "neutral"
            else:
                side, stage = mapped
        elif ac in _EQUITY_CLASSES:
            # Equities carry no option lifecycle. A buy establishes long stock; a
            # sell is a reduce/exit (no short marker in the data, so never short).
            side = bs_word
            stage = "open" if bs_word == "buy" else "exit"
        else:
            side, stage = bs_word, "neutral"

        if side and bs_word and side != bs_word:
            mismatches += 1
        sides.append(side)
        stages.append(stage)

    df["_side"] = sides
    df["_stage"] = stages
    if mismatches:
        logger.warning(
            "client profile: %d trade row(s) where Buy/Sell disagrees with the "
            "lifecycle side", mismatches,
        )
    return df, n_cancels, unknown_actions


# ---------------------------------------------------------------------------
# Dimension builders (pure functions over the normalised frame)
# ---------------------------------------------------------------------------

_OPT_OPEN_BUCKET = {
    ("buy", "CALL"): "long_call",
    ("sell", "CALL"): "short_call",
    ("buy", "PUT"): "long_put",
    ("sell", "PUT"): "short_put",
}
_LONG_DELTA = {"long_call", "short_put", "long_stock"}
_SHORT_DELTA = {"short_call", "long_put"}


def _opening_buckets(df: pd.DataFrame) -> list[str]:
    """The strategy bucket for every opening trade (option opens by side x right,
    equity buys as long stock)."""
    out: list[str] = []
    for _, row in df.iterrows():
        if row.get("_stage") != "open":
            continue
        ac = str(row.get("asset_class") or "").strip().lower()
        if ac == _OPTION_CLASS:
            right = str(row.get("option_type") or "").strip().upper()
            # An option open whose side/right can't be read still counts as an
            # opening trade — surfaced as 'unclassified_open' rather than silently
            # dropped, so the posture mix stays faithful to the opening flow (and
            # the opening count agrees with the coverage open count).
            out.append(_OPT_OPEN_BUCKET.get((row.get("_side"), right), "unclassified_open"))
        elif ac in _EQUITY_CLASSES:
            out.append("long_stock")
    return out


def _build_strategy_bias(df: pd.DataFrame) -> StrategyBias:
    buckets = _opening_buckets(df)
    n = len(buckets)
    if not n:
        return StrategyBias(weights={}, n_opening=0, confidence="low")
    weights: dict[str, float] = {}
    for b in buckets:
        weights[b] = weights.get(b, 0.0) + 1.0 / n
    return StrategyBias(weights=weights, n_opening=n, confidence=_confidence(n))


def _build_direction_bias(df: pd.DataFrame) -> DirectionBias:
    buckets = _opening_buckets(df)
    n = len(buckets)
    if not n:
        return DirectionBias(confidence="low")
    calls = sum(b in ("long_call", "short_call") for b in buckets)
    puts = sum(b in ("long_put", "short_put") for b in buckets)
    longs = sum(b in _LONG_DELTA for b in buckets)
    shorts = sum(b in _SHORT_DELTA for b in buckets)
    cp = (calls - puts) / (calls + puts) if (calls + puts) else None
    ls = (longs - shorts) / (longs + shorts) if (longs + shorts) else None
    return DirectionBias(call_put_skew=cp, long_short_skew=ls, confidence=_confidence(n))


def _build_tenor_pref(df: pd.DataFrame) -> TenorPref:
    dtes: list[int] = []
    for _, row in df.iterrows():
        if row.get("_stage") != "open":
            continue
        if str(row.get("asset_class") or "").strip().lower() != _OPTION_CLASS:
            continue
        td = _as_date(row.get("trade_date"))
        ex = _as_date(row.get("option_expiration"))
        if td is None or ex is None:
            continue
        dtes.append((ex - td).days)
    if not dtes:
        return TenorPref(confidence="low")
    med = float(median(dtes))
    dist = {"short": 0, "swing": 0, "leaps": 0}
    for d in dtes:
        dist[_dte_bucket(d)] += 1
    total = len(dtes)
    distribution = {k: v / total for k, v in dist.items()}
    return TenorPref(
        median_dte_at_open=med,
        bucket=_dte_bucket(med),
        distribution=distribution,
        n_opens=total,
        confidence=_confidence(total),
    )


def _dte_bucket(dte: float) -> str:
    if dte <= _SHORT_DTE_MAX:
        return "short"
    if dte <= _SWING_DTE_MAX:
        return "swing"
    return "leaps"


def _underlying_key(row) -> Optional[str]:
    """Raw underlying symbol for a trade: underlying_ticker for options,
    ticker_final for equities/funds."""
    ac = str(row.get("asset_class") or "").strip().lower()
    raw = row.get("underlying_ticker") if ac == _OPTION_CLASS else row.get("ticker_final")
    if _blank(raw):
        raw = row.get("underlying_ticker")
    return None if _blank(raw) else str(raw).strip()


def _raw_to_bbg(positions) -> dict[str, str]:
    """Map a raw underlying symbol to its BBG-ticker form, from the held
    positions (the snapshot is keyed on BBG form; trades carry raw symbols)."""
    m: dict[str, str] = {}
    for p in positions or []:
        ac = str(getattr(p, "asset_class", "") or "").strip().lower()
        if ac == _OPTION_CLASS:
            raw, bbg = getattr(p, "underlying_symbol", None), getattr(p, "underlying_bbg_ticker", None)
        elif ac in _EQUITY_CLASSES:
            raw, bbg = getattr(p, "symbol", None), getattr(p, "bbg_ticker", None)
        else:
            continue
        if raw and bbg:
            m.setdefault(str(raw).strip(), str(bbg).strip())
    return m


def _build_sector_lean(df: pd.DataFrame, positions, underlyings) -> SectorLean:
    raw_to_bbg = _raw_to_bbg(positions)
    name_w: dict[str, float] = {}
    sector_w: dict[str, float] = {}
    classified = 0
    total = 0
    for _, row in df.iterrows():
        name = _underlying_key(row)
        if name is None:
            continue
        total += 1
        name_w[name] = name_w.get(name, 0.0) + 1.0
        sector = _sector_of(underlyings, raw_to_bbg.get(name))
        sector_w[sector] = sector_w.get(sector, 0.0) + 1.0
        if sector != UNCLASSIFIED:
            classified += 1
    if not total:
        return SectorLean(confidence="low")

    def _top(d: dict[str, float]) -> list:
        ranked = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(k, v / total) for k, v in ranked[:_TOP_N]]

    # Only surface a sector ranking when at least one name resolved; otherwise the
    # ranking would be a single 'Unclassified' bar, which says nothing.
    top_sectors = _top({k: v for k, v in sector_w.items() if k != UNCLASSIFIED}) if classified else []
    return SectorLean(
        top=top_sectors,
        by_name=_top(name_w),
        classified_fraction=classified / total,
        confidence=_confidence(total),
    )


def _build_sizing(df: pd.DataFrame) -> Sizing:
    if "principal_amount" not in df.columns:
        return Sizing(confidence="low")
    principal = pd.to_numeric(df["principal_amount"], errors="coerce").abs().dropna()
    if principal.empty:
        return Sizing(confidence="low")
    med = float(principal.median())

    # Name concentration by absolute principal (Herfindahl).
    by_name: dict[str, float] = {}
    for _, row in df.iterrows():
        name = _underlying_key(row)
        amt = pd.to_numeric(pd.Series([row.get("principal_amount")]), errors="coerce").iloc[0]
        if name is None or pd.isna(amt):
            continue
        by_name[name] = by_name.get(name, 0.0) + abs(float(amt))
    hhi = None
    grand = sum(by_name.values())
    if grand > 0:
        hhi = sum((v / grand) ** 2 for v in by_name.values())
    return Sizing(median_principal=med, concentration_hhi=hhi, confidence=_confidence(len(principal)))


def _build_cadence(df: pd.DataFrame, window_days: Optional[int]) -> Cadence:
    n = len(df)
    # Annualising a sub-month burst (e.g. seven trades over six days) would invent
    # a precise-looking rate the history can't support, so a span shorter than a
    # month yields no rate and low confidence rather than a misleading number.
    if not n or not window_days or window_days < _CADENCE_MIN_WINDOW_DAYS:
        return Cadence(trades_per_month=None, clustering="n/a", confidence="low")
    months = window_days / 30.44
    tpm = n / months if months > 0 else None
    return Cadence(trades_per_month=tpm, clustering="n/a", confidence=_confidence(n))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _window_days(df: pd.DataFrame) -> Optional[int]:
    if "trade_date" not in df.columns:
        return None
    dates = [d for d in (_as_date(v) for v in df["trade_date"]) if d is not None]
    if len(dates) < 2:
        return 0 if dates else None
    return (max(dates) - min(dates)).days


def _build_coverage(df: pd.DataFrame, positions, n_cancels: int) -> Coverage:
    n_trades = len(df)
    n_opens = int((df["_stage"] == "open").sum()) if "_stage" in df.columns else 0
    n_closes = int((df["_stage"] == "close").sum()) if "_stage" in df.columns else 0
    window_days = _window_days(df)

    # Round-trip fraction: option contracts whose history holds both an open and
    # a close (no FIFO, no qty netting — presence of each stage).
    paired_fraction = 0.0
    if "option_contract_key" in df.columns and "_stage" in df.columns:
        opt = df[df["asset_class"].astype(str).str.lower() == _OPTION_CLASS] if "asset_class" in df.columns else df
        keys = {}
        for _, row in opt.iterrows():
            key = row.get("option_contract_key")
            if _blank(key):
                continue
            entry = keys.setdefault(str(key), {"open": False, "close": False})
            if row.get("_stage") in ("open", "close"):
                entry[row["_stage"]] = True
        if keys:
            both = sum(1 for v in keys.values() if v["open"] and v["close"])
            paired_fraction = both / len(keys)

    # Held option positions with any opening trade on the same contract (carries
    # the opens-predate-window caveat distinct from the round-trip rate above).
    pos_open_frac = _positions_with_derivable_open(df, positions)

    band = _band(n_trades, paired_fraction, window_days)
    return Coverage(
        n_trades=n_trades,
        n_opens=n_opens,
        n_closes=n_closes,
        n_excluded_cancels=n_cancels,
        window_days=window_days,
        paired_fraction=paired_fraction,
        positions_with_derivable_open_fraction=pos_open_frac,
        band=band,
    )


def _positions_with_derivable_open(df: pd.DataFrame, positions) -> Optional[float]:
    held_keys = [
        str(getattr(p, "option_contract_key"))
        for p in (positions or [])
        if str(getattr(p, "asset_class", "") or "").strip().lower() == _OPTION_CLASS
        and not _blank(getattr(p, "option_contract_key", None))
    ]
    if not held_keys:
        return None
    open_keys: set[str] = set()
    if "option_contract_key" in df.columns and "_stage" in df.columns:
        opens = df[df["_stage"] == "open"]
        for _, row in opens.iterrows():
            key = row.get("option_contract_key")
            if not _blank(key):
                open_keys.add(str(key))
    derivable = sum(1 for k in held_keys if k in open_keys)
    return derivable / len(held_keys)


# ---------------------------------------------------------------------------
# Fragile lifecycle tier (holds proxy + roll heuristic; income deferred)
# ---------------------------------------------------------------------------

def _build_holding_period(positions) -> Optional[HoldingPeriod]:
    """Median days-held-so-far across currently-held option positions that carry a
    derivable open (``Position.days_held`` = today − the first opening trade found
    by the account-scoped-first join; transfer-inferred opens are counted and
    disclosed). None below the sample floor. Doubly limited — see HoldingPeriod."""
    held: list[int] = []
    n_transfer = 0
    for p in positions or []:
        if str(getattr(p, "asset_class", "") or "").strip().lower() != _OPTION_CLASS:
            continue
        dh = getattr(p, "days_held", None)
        if dh is None:
            continue
        try:
            held.append(int(dh))
        except (TypeError, ValueError):
            continue
        if getattr(p, "transfer_inferred", False):
            n_transfer += 1
    if len(held) < _HOLDS_MIN_POSITIONS:
        return None
    return HoldingPeriod(median_days_held=float(median(held)), n_positions=len(held),
                         confidence=_confidence(len(held)),
                         n_transfer_inferred=n_transfer)


def _build_roll_behavior(df: pd.DataFrame) -> Optional[RollBehavior]:
    """Roll tendency from the roll-like-clustering heuristic (see RollBehavior).
    None when there are no option opening or closing trades to read; otherwise a
    RollBehavior whose tendency is gated to 'unknown' below the close-count floor
    (n_events still surfaced). n_events counts roll-like CLOSES (the per-close
    share), so one reopen near two closes counts twice — intentional for the share
    metric, see RollBehavior."""
    if df is None or getattr(df, "empty", True) or "_stage" not in df.columns:
        return None
    opt = df[df["asset_class"].astype(str).str.lower() == _OPTION_CLASS] \
        if "asset_class" in df.columns else df
    closes = opt[opt["_stage"] == "close"]
    opens = opt[opt["_stage"] == "open"]
    n_closes = int(len(closes))
    if n_closes == 0 and len(opens) == 0:
        return None

    # Pre-extract the opens once (small books; an explicit loop keeps the side /
    # same-name / different-contract / window rule legible).
    open_recs = []
    for _, op in opens.iterrows():
        od = _as_date(op.get("trade_date"))
        if od is None or _blank(op.get("underlying_ticker")):
            continue
        open_recs.append((str(op.get("underlying_ticker")), str(op.get("option_type") or "").upper(),
                          op.get("option_contract_key"), op.get("_side"), od))

    n_events = 0
    for _, cl in closes.iterrows():
        und = cl.get("underlying_ticker")
        right = str(cl.get("option_type") or "").upper()
        ckey = cl.get("option_contract_key")
        cdate = _as_date(cl.get("trade_date"))
        if _blank(und) or not right or cdate is None:
            continue
        # STC (sell-close of a long) rolls into a BTO; BTC (buy-close of a short) into a STO.
        want_open_side = "buy" if cl.get("_side") == "sell" else "sell"
        for o_und, o_right, o_key, o_side, o_date in open_recs:
            if o_und != str(und) or o_right != right or o_side != want_open_side:
                continue
            if not _blank(o_key) and not _blank(ckey) and str(o_key) == str(ckey):
                continue  # same contract — a re-add, not a roll
            # Forward-only, in business days: a roll REPLACES the closed leg, so
            # its open comes at or after the close (same day, or the next
            # business day — Friday close / Monday reopen counts). An open that
            # PRECEDES the close is an existing position, not a roll.
            if 0 <= np.busday_count(cdate, o_date) <= _ROLL_WINDOW_BD:
                n_events += 1
                break

    if n_closes < _ROLLS_MIN_CLOSES:
        return RollBehavior(tendency="unknown", n_events=n_events, n_closes=n_closes, confidence="low")
    share = n_events / n_closes if n_closes else 0.0
    if share >= _ROLL_HIGH:
        tendency = "rolls"
    elif share < _ROLL_LOW:
        tendency = "closes_early"
    else:
        tendency = "mixed"
    return RollBehavior(tendency=tendency, n_events=n_events, n_closes=n_closes,
                        confidence=_confidence(n_closes))


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

_POSTURE_LABEL = {
    "long_call": "call buyer",
    "short_call": "call writer",
    "long_put": "put buyer",
    "short_put": "put seller",
    "long_stock": "equity accumulator",
    "unclassified_open": "unclassified opens",
}
_TENOR_LABEL = {"short": "short-dated", "swing": "swing tenor", "leaps": "LEAPS tenor"}


def _build_headline(strat: StrategyBias, tenor: TenorPref, sector: SectorLean, band: str) -> str:
    parts: list[str] = []
    if strat.weights:
        top = max(strat.weights, key=strat.weights.get)
        parts.append(_POSTURE_LABEL.get(top, top))
    if tenor.bucket:
        parts.append(_TENOR_LABEL.get(tenor.bucket, tenor.bucket))
    if sector.top:
        parts.append(f"{sector.top[0][0]}-leaning")
    elif sector.by_name:
        parts.append(f"{sector.by_name[0][0]}-concentrated")
    core = " · ".join(parts) if parts else "insufficient trade history"
    return f"Thin history — {core}" if band == "low" else core


# ---------------------------------------------------------------------------
# Assembly + load-path entry
# ---------------------------------------------------------------------------

def compute_account_profile(account_state) -> ClientProfile:
    """Build the behavioural profile for one account from its trades + snapshot +
    positions. Pure: reads already-loaded state, no Bloomberg, no recompute."""
    account = getattr(account_state, "account", "")
    trades = getattr(account_state, "trades", None)
    positions = getattr(account_state, "positions", None)
    snapshot = getattr(account_state, "snapshot", None)
    underlyings = getattr(snapshot, "underlyings", None)

    df, n_cancels, unknown_actions = normalize_trades(trades)
    degraded_reason = None
    if unknown_actions:
        uniq = sorted(set(unknown_actions))
        degraded_reason = (
            f"{len(unknown_actions)} trade row(s) carry unrecognised lifecycle "
            f"action(s) {uniq} — excluded from open/close-based reads (posture, "
            "direction, tenor, round-trips, rolls); flow-level reads (sector, "
            "sizing, cadence, window) include them."
        )

    if df is None or df.empty:
        coverage = Coverage(
            n_trades=0, n_opens=0, n_closes=0, n_excluded_cancels=n_cancels,
            window_days=None, paired_fraction=0.0,
            positions_with_derivable_open_fraction=_positions_with_derivable_open(
                df if df is not None else pd.DataFrame(), positions),
            band="low",
        )
        return ClientProfile(
            account=account, coverage=coverage,
            strategy_bias=StrategyBias(), direction_bias=DirectionBias(),
            tenor_pref=TenorPref(), sector_lean=SectorLean(), sizing=Sizing(),
            cadence=Cadence(),
            # Holds is positions-derived (Position.days_held; the join is
            # account-scoped first, cross-account only as the marked transfer
            # fallback), so it can still read for a positions-only / transferred-in
            # account with an empty trade blotter; rolls needs trades, so stays None.
            holding_period=_build_holding_period(positions),
            headline="No trade history.",
            degraded_reason=degraded_reason,
        )

    coverage = _build_coverage(df, positions, n_cancels)
    strategy_bias = _build_strategy_bias(df)
    direction_bias = _build_direction_bias(df)
    tenor_pref = _build_tenor_pref(df)
    sector_lean = _build_sector_lean(df, positions, underlyings)
    sizing = _build_sizing(df)
    cadence = _build_cadence(df, coverage.window_days)
    holding_period = _build_holding_period(positions)
    roll_behavior = _build_roll_behavior(df)
    headline = _build_headline(strategy_bias, tenor_pref, sector_lean, coverage.band)

    return ClientProfile(
        account=account, coverage=coverage, strategy_bias=strategy_bias,
        direction_bias=direction_bias, tenor_pref=tenor_pref, sector_lean=sector_lean,
        sizing=sizing, cadence=cadence, holding_period=holding_period,
        roll_behavior=roll_behavior, headline=headline,
        degraded_reason=degraded_reason,
    )


def run_account_profile(state) -> None:
    """Compute and attach the client profile for every account, in the load path.

    Reads each account's already-loaded trades + snapshot + positions and stores
    the result on ``acc.client_profile`` — no Bloomberg, no recompute. One bad
    account degrades to ``None`` and is logged, leaving the rest intact; a
    profile computed around unrecognised lifecycle rows carries its
    ``degraded_reason`` into the load notes (replace-not-stack, the [insight]
    idiom) so the operator surface names the offending strings."""
    warnings = getattr(state, "all_warnings", None)
    if warnings is not None:
        state.all_warnings = [w for w in warnings if not w.startswith("[profile]")]
    for acc in getattr(state, "accounts", {}).values():
        try:
            acc.client_profile = compute_account_profile(acc)
        except Exception:
            logger.exception(
                "client profile failed for account %s", getattr(acc, "account", "?")
            )
            acc.client_profile = None
            continue
        reason = getattr(acc.client_profile, "degraded_reason", None)
        if reason and getattr(state, "all_warnings", None) is not None:
            state.all_warnings.append(
                f"[profile] {getattr(acc, 'account', '?')}: {reason}")
