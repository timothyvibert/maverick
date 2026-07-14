"""Portfolio exposure aggregation — where an account is economically exposed now.

The first layer of the risk-analytics views. It answers
"where is this book exposed *right now*" by aggregating the **Bloomberg snapshot
greeks the load path already computed** (``AccountState.greeks.by_position`` —
delta/gamma/vega/theta straight off the option snapshot) into account-level dollar
exposure, beta-adjusts it to a single SPX benchmark, and rolls it up
position → structure → account. It is the *current-state* lens: it reads the live
greeks, it does **not** re-price anything (the pricing engine is for the hypothetical
states of the scenario layer).

Pure and read-only — no Dash, no Bloomberg, no pricing-engine import. It reads only
already-loaded state and is duck-typed so tests pass lightweight stand-ins.

What it produces (every number carries a trace of its inputs):
- **Net dollar greeks** — Δ/Γ/V/Θ for the account.
- **Dollar beta (net market exposure)** — Σ(position dollar-delta × the name's SPX
  beta). Computed under both the adjusted and raw beta so the panel can show both;
  ``beta_source`` selects which is the headline (adjusted by default — the stable,
  desk-matching number).
- **Market value vs economic exposure** — the book's mark vs its delta-equivalent
  underlying exposure, side by side (an option's premium understates its direction).
- **Vega by tenor bucket** — vega's term structure, on its own finer buckets.
- **Position → structure → account rollup** — each structure's exposure, the
  unstructured residual, and the account total, conserved: structured + unstructured
  == account. The split rides the shared allocation ledger
  (``reconcile_allocations``), so contended legs count once and rejected structures
  drop out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from pm.insight.structures import reconcile_allocations

# Bloomberg beta fields read from the underlying snapshot. Both are pulled
# SPX-relative (a single consistent benchmark, so net market exposure is coherent)
# at the 2y-weekly window — wired with the BETA_OVERRIDE_REL_INDEX=SPX override in
# the data layer. The exposure default is the adjusted (Blume-shrunk) beta; the raw
# beta is the current empirical sensitivity, surfaced alongside.
BETA_FIELD_ADJUSTED = "EQY_BETA"       # SPX 2y weekly, adjusted (exposure default)
BETA_FIELD_RAW = "EQY_RAW_BETA"        # SPX 2y weekly, raw

# GICS sector, read from the same underlying snapshot for the option-aware sector
# breakdown (Analytics). The same field the legacy stock-MV sector diagnostic reads,
# so the two bases line up name-for-name.
SECTOR_FIELD = "GICS_SECTOR_NAME"

# Vega tenor buckets — deliberately finer and term-structure-shaped, NOT the
# expiry-ladder's strike-obligation buckets (≤30/31-60/61-90/>90d): vega is a
# vol-term-structure exposure, and the ladder's single >90d bucket would collapse
# all long-dated vega into one number. Bucketed on calendar days to expiry.
VEGA_TENOR_BUCKETS = [
    ("≤1m", lambda d: d <= 30),
    ("1–3m", lambda d: 31 <= d <= 90),
    ("3–6m", lambda d: 91 <= d <= 180),
    ("6–12m", lambda d: 181 <= d <= 365),
    (">12m", lambda d: d > 365),
]


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

@dataclass
class ExposureNode:
    """Aggregated dollar exposure for one slice of the book — the whole account, a
    single structure, the structured total, or the unstructured residual. Dollar
    beta is the beta-weighted dollar-delta under each beta (None when no contributing
    name had a beta). Each dollar greek is likewise None when NO contributing slice
    carried that greek — an all-missing slice must never read as $0."""
    label: str
    dollar_delta: Optional[float]
    dollar_gamma: Optional[float]
    dollar_vega: Optional[float]
    dollar_theta: Optional[float]
    dollar_beta_adjusted: Optional[float]
    dollar_beta_raw: Optional[float]
    market_value: float
    n_positions: int
    structure_id: Optional[str] = None
    structure_type: Optional[str] = None
    contention_group: Optional[str] = None
    degraded: bool = False           # a contributing slice couldn't be trusted/allocated
    trace: dict = field(default_factory=dict)


@dataclass
class VegaTenorBucket:
    label: str
    dollar_vega: float
    n_options: int
    # options in the bucket whose vega is missing from the snapshot — the bucket's
    # dollar_vega understates by these; n_missing_vega == n_options means the
    # bucket's $0 is pure missing data, not zero exposure.
    n_missing_vega: int = 0


@dataclass
class AccountExposure:
    """The exposure view for one account: the account total, the conserved
    structured/unstructured split, the per-structure breakdown, vega's term
    structure, and the warnings/trace behind the numbers."""
    account: str
    beta_source: str                 # "adjusted" | "raw" — the headline beta choice
    total: ExposureNode              # the whole account == structured + unstructured
    structured: ExposureNode         # the part inside recognised structures
    unstructured: ExposureNode       # the standalone residual
    structures: list[ExposureNode]   # per-structure breakdown (contention alternatives flagged)
    vega_by_tenor: list[VegaTenorBucket]
    warnings: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    # tickers whose row is missing one or more dollar greeks (the beta analogue:
    # missing_beta discloses names excluded from dollar-beta; this discloses names
    # whose greeks are absent, so the greek totals understate). Rendered in the
    # panel provenance.
    missing_greeks: list[str] = field(default_factory=list)
    # expired options still on the book (stale extract) — excluded from the vega
    # tenor ladder; rendered as an "expired (n)" cue on that panel.
    n_expired_options: int = 0

    @property
    def net_market_exposure(self) -> Optional[float]:
        """The headline dollar beta under the selected ``beta_source`` — the
        account's net market (SPX-equivalent) exposure."""
        node = self.total
        return (node.dollar_beta_adjusted if self.beta_source == "adjusted"
                else node.dollar_beta_raw)

    @property
    def economic_exposure(self) -> Optional[float]:
        """Delta-equivalent underlying exposure — the economic side of the
        market-value-vs-economic contrast (== net dollar delta; None when no
        contributing row had a delta)."""
        return self.total.dollar_delta


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    """Coerce to float; None / NaN / non-numeric -> None (skipna semantics)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cell_lookup(underlyings, ticker: Optional[str], field_name: str):
    """Raw snapshot cell for (ticker, field_name), or None when the snapshot is
    absent, the name isn't in it, or the field is missing — never raises. Shared by
    the beta lookup (numeric) and the sector lookup (string)."""
    if underlyings is None or not ticker:
        return None
    try:
        if field_name not in getattr(underlyings, "columns", []):
            return None
        if ticker not in underlyings.index:
            return None
        val = underlyings.loc[ticker, field_name]
    except Exception:
        return None
    # A duplicated index yields a Series — take the first row defensively.
    if hasattr(val, "iloc"):
        val = val.iloc[0] if len(val) else None
    return val


def _beta_lookup(underlyings, ticker: Optional[str], field_name: str) -> Optional[float]:
    """Beta for one underlying from the snapshot, by ticker + field name. None when
    the snapshot is absent, the name isn't in it, the field is missing, or the value
    is non-numeric — never raises, never substitutes 0."""
    return _num(_cell_lookup(underlyings, ticker, field_name))


def _sector_of(underlyings, ticker: Optional[str]) -> str:
    """GICS sector for one underlying — 'Unclassified' when the snapshot/name/field
    is absent or blank. Mirrors the diagnostics fallback so the two bases never
    disagree on a name's sector."""
    val = _cell_lookup(underlyings, ticker, SECTOR_FIELD)
    if val is None:
        return "Unclassified"
    try:
        if pd.isna(val):
            return "Unclassified"
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s or "Unclassified"


def _new_acc() -> dict:
    return {"dd": 0.0, "dg": 0.0, "dv": 0.0, "dt": 0.0,
            "dd_has": False, "dg_has": False, "dv_has": False, "dt_has": False,
            "ba": 0.0, "br": 0.0, "ba_has": False, "br_has": False,
            "mv": 0.0, "n": 0, "degraded": False, "pids": []}


def _add_slice(acc: dict, frac: float, row: dict, *, degraded: bool = False) -> None:
    """Add ``frac`` of one position's dollar greeks to an accumulator. Greeks are
    skipna (a missing BBG greek contributes nothing, matching ``greeks.totals``),
    with a has-flag per greek so a node where NOTHING accrued finalizes to None,
    not a fake 0.0; dollar-beta only accrues where both the delta and the name's
    beta are present."""
    dd = row["dd"]
    for key, val in (("dd", dd), ("dg", row["dg"]), ("dv", row["dv"]), ("dt", row["dt"])):
        if val is not None:
            acc[key] += frac * val
            acc[key + "_has"] = True
    if dd is not None and row["beta_adj"] is not None:
        acc["ba"] += frac * dd * row["beta_adj"]
        acc["ba_has"] = True
    if dd is not None and row["beta_raw"] is not None:
        acc["br"] += frac * dd * row["beta_raw"]
        acc["br_has"] = True
    if row["mv"] is not None:
        acc["mv"] += frac * row["mv"]
    acc["n"] += 1
    if degraded:
        acc["degraded"] = True
    acc["pids"].append(row["pid"])


def _finalize(acc: dict, label: str, **extra) -> ExposureNode:
    # A slice with no rows at all genuinely holds zero exposure (0.0); a slice
    # WITH rows where nothing accrued for a greek is missing data (None).
    def _val(key: str):
        if acc[key + "_has"]:
            return acc[key]
        return 0.0 if acc["n"] == 0 else None

    return ExposureNode(
        label=label,
        dollar_delta=_val("dd"),
        dollar_gamma=_val("dg"),
        dollar_vega=_val("dv"),
        dollar_theta=_val("dt"),
        dollar_beta_adjusted=(acc["ba"] if acc["ba_has"] else None),
        dollar_beta_raw=(acc["br"] if acc["br_has"] else None),
        market_value=acc["mv"], n_positions=acc["n"], degraded=acc["degraded"],
        trace={"position_ids": list(acc["pids"]),
               "computation": "Σ slice_fraction × position dollar-greek; "
                              "dollar_beta = Σ slice dollar-delta × SPX beta"},
        **extra,
    )


# ---------------------------------------------------------------------------
# The compute
# ---------------------------------------------------------------------------

def compute_account_exposure(account_state, *, beta_source: str = "adjusted",
                             as_of: Optional[date] = None) -> AccountExposure:
    """Aggregate one account's live snapshot greeks into the exposure view.

    Reads ``account_state.greeks.by_position`` (per-position dollar greeks),
    ``account_state.snapshot.underlyings`` (the SPX beta fields + the index of
    underlyings), ``account_state.positions`` (market value + expiry), and
    ``account_state.structures`` (the rollup). Pure: no Bloomberg, no recompute, no
    pricing-engine call.

    ``beta_source`` ("adjusted" | "raw") selects which beta is the headline net
    market exposure; both dollar betas are always computed so the panel can toggle
    without recomputing. ``as_of`` pins the reference date for the vega tenor
    buckets (defaults to today), mirroring the expiry ladder.
    """
    if beta_source not in ("adjusted", "raw"):
        raise ValueError("beta_source must be 'adjusted' or 'raw', got %r" % (beta_source,))
    today = as_of or date.today()

    account = getattr(account_state, "account", "")
    positions = list(getattr(account_state, "positions", []) or [])
    pos_by_id = {p.position_id: p for p in positions}
    greeks = getattr(account_state, "greeks", None)
    by_position = getattr(greeks, "by_position", None)
    snapshot = getattr(account_state, "snapshot", None)
    underlyings = getattr(snapshot, "underlyings", None)
    warnings: list[str] = []

    # Beta per underlying, looked up once per name (so a missing beta warns once).
    beta_cache: dict[str, tuple] = {}
    missing_beta: list[str] = []

    def _betas(underlying: Optional[str]) -> tuple:
        key = underlying or ""
        if key in beta_cache:
            return beta_cache[key]
        ba = _beta_lookup(underlyings, underlying, BETA_FIELD_ADJUSTED)
        br = _beta_lookup(underlyings, underlying, BETA_FIELD_RAW)
        beta_cache[key] = (ba, br)
        if ba is None and br is None and underlying:
            missing_beta.append(underlying)
        return ba, br

    # Build one greek row per greek-bearing position (equity + option). Cash/other
    # carry no greeks and are absent from by_position — they contribute nothing.
    # Names with a missing dollar greek are collected for the provenance line
    # (the beta analogue): the totals silently understate by exactly these rows.
    rows_by_pid: dict[str, dict] = {}
    missing_greeks: list[str] = []
    if by_position is not None and not getattr(by_position, "empty", True):
        for _, r in by_position.iterrows():
            pid = r.get("position_id")
            if pid is None:
                continue
            pos = pos_by_id.get(pid)
            underlying = r.get("underlying_ticker")
            ba, br = _betas(underlying)
            row = {
                "pid": pid,
                "dd": _num(r.get("dollar_delta")),
                "dg": _num(r.get("dollar_gamma")),
                "dv": _num(r.get("dollar_vega")),
                "dt": _num(r.get("dollar_theta")),
                "beta_adj": ba,
                "beta_raw": br,
                "mv": _num(getattr(pos, "market_value", None)) if pos is not None else None,
                "underlying": underlying,
                "instrument_type": r.get("instrument_type"),
                "expiry": getattr(pos, "expiry", None) if pos is not None else None,
            }
            rows_by_pid[pid] = row
            if any(row[k] is None for k in ("dd", "dg", "dv", "dt")):
                name = str(r.get("bbg_ticker") or underlying or pid)
                if name not in missing_greeks:
                    missing_greeks.append(name)

    # The conserved split: how much of each position the structures claim vs leave
    # standalone (rejected-skip + contention-collapse live in this ledger).
    reconcile = reconcile_allocations(account_state)

    total_acc = _new_acc()
    structured_acc = _new_acc()
    unstructured_acc = _new_acc()
    for pid, row in rows_by_pid.items():
        _add_slice(total_acc, 1.0, row)
        rec = reconcile.get(pid) or {}
        full = rec.get("quantity")
        alloc = rec.get("allocated", 0.0)
        rem = rec.get("remainder")
        ok = rec.get("ok", True)
        # A position we can't trust a slice for (no/zero quantity, over-allocation,
        # sign flip) rolls up wholly unstructured rather than fabricating a split.
        if full in (None, 0) or rem is None or not ok:
            _add_slice(unstructured_acc, 1.0, row, degraded=not ok)
            continue
        frac_s = alloc / full
        frac_u = rem / full
        if frac_s != 0:
            _add_slice(structured_acc, frac_s, row)
        if frac_u != 0:
            _add_slice(unstructured_acc, frac_u, row)

    total = _finalize(total_acc, label="Account")
    structured = _finalize(structured_acc, label="Structured")
    unstructured = _finalize(unstructured_acc, label="Unstructured")

    # Per-structure breakdown (display): each structure's own legs, sliced. Contention
    # alternatives each appear (flagged); they are NOT summed into the structured
    # total — that total comes from the reconcile ledger, which counts a contended
    # leg once.
    structure_nodes: list[ExposureNode] = []
    for st in (getattr(account_state, "structures", []) or []):
        if getattr(st, "status", None) == "rejected":
            continue
        acc = _new_acc()
        for leg in getattr(st, "legs", []):
            row = rows_by_pid.get(leg.position_id)
            if row is None:
                continue
            rec = reconcile.get(leg.position_id) or {}
            full = rec.get("quantity")
            if full in (None, 0):
                acc["degraded"] = True
                continue
            try:
                frac = float(leg.allocated_qty) / float(full)
            except (TypeError, ValueError, ZeroDivisionError):
                acc["degraded"] = True
                continue
            _add_slice(acc, frac, row)
        structure_nodes.append(_finalize(
            acc, label=getattr(st, "type", "structure"),
            structure_id=getattr(st, "structure_id", None),
            structure_type=getattr(st, "type", None),
            contention_group=getattr(st, "contention_group", None),
        ))

    # Vega term structure — option positions bucketed by days to expiry. An option
    # whose vega is missing still counts into its bucket (n) but is tallied as
    # missing so the UI can dash an all-missing bucket instead of showing $0.
    # An EXPIRED option (negative DTE — a stale extract is a designed-for mode)
    # is excluded entirely: a dead contract must not read as ≤1m vega.
    buckets = {label: {"dv": 0.0, "n": 0, "nm": 0} for label, _ in VEGA_TENOR_BUCKETS}
    n_no_expiry = 0
    n_expired = 0
    for row in rows_by_pid.values():
        if row["instrument_type"] != "option":
            continue
        exp = row["expiry"]
        if not isinstance(exp, date):
            n_no_expiry += 1
            continue
        dte = (exp - today).days
        if dte < 0:
            n_expired += 1
            continue
        for label, pred in VEGA_TENOR_BUCKETS:
            if pred(dte):
                buckets[label]["n"] += 1
                if row["dv"] is not None:
                    buckets[label]["dv"] += row["dv"]
                else:
                    buckets[label]["nm"] += 1
                break
    vega_by_tenor = [VegaTenorBucket(label, buckets[label]["dv"], buckets[label]["n"],
                                     buckets[label]["nm"])
                     for label, _ in VEGA_TENOR_BUCKETS]
    if n_no_expiry:
        warnings.append(f"{n_no_expiry} option position(s) had no expiry — "
                        "excluded from the vega tenor ladder.")
    if n_expired:
        warnings.append(f"{n_expired} expired option position(s) still on the book — "
                        "excluded from the vega tenor ladder.")

    # Warnings: names with no SPX beta drop out of dollar-beta (never zeroed).
    for name in missing_beta:
        warnings.append(f"SPX beta unavailable for {name} — excluded from dollar-beta.")
    if any(n.degraded for n in (total, structured, unstructured)) or \
            any(n.degraded for n in structure_nodes):
        warnings.append("Some positions could not be cleanly allocated to structures "
                        "(over-allocation or missing quantity) — see degraded nodes.")

    trace = {
        "inputs": {
            "greek_source": "Bloomberg option-snapshot greeks "
                            "(delta_mid/gamma/vega/theta) via greeks.by_position",
            "beta_field_adjusted": BETA_FIELD_ADJUSTED,
            "beta_field_raw": BETA_FIELD_RAW,
            "beta_source": beta_source,
            "n_positions_with_greeks": len(rows_by_pid),
            "names_missing_beta": list(missing_beta),
            "names_missing_greeks": list(missing_greeks),
            "as_of": today.isoformat(),
        },
        "computation": "dollar greeks summed across positions; dollar_beta = "
                       "Σ position dollar-delta × the name's SPX beta; rollup splits "
                       "each position by its structure allocation (reconcile_allocations) "
                       "so structured + unstructured == account.",
        "result": "account exposure + conserved position→structure→account rollup",
    }

    return AccountExposure(
        account=account,
        beta_source=beta_source,
        total=total,
        structured=structured,
        unstructured=unstructured,
        structures=structure_nodes,
        vega_by_tenor=vega_by_tenor,
        warnings=warnings,
        trace=trace,
        missing_greeks=missing_greeks,
        n_expired_options=n_expired,
    )


def run_account_exposure(state) -> None:
    """Compute and attach the exposure view for every account, in the load path.

    Reads each account's already-loaded greeks + structures + snapshot and stores the
    result on ``acc.exposure`` — no Bloomberg, no recompute. Called after structure
    detection so the rollup sees the recognised structures; the UI only reads the
    stored view."""
    for acc in state.accounts.values():
        acc.exposure = compute_account_exposure(acc)


# ---------------------------------------------------------------------------
# Economic-exposure breakdowns (per-name, per-sector) — the option-aware basis for
# the Analytics sector + concentration cards. Same delta-$ economic exposure as the
# rollup, just sliced by underlying / GICS sector instead of by structure. Pure reads
# of greeks.by_position (+ the underlying snapshot for sector / NAV for the %),
# conserved: Σ over names == Σ over sectors == the account's net dollar-delta.
# ---------------------------------------------------------------------------

def _iter_greek_rows(account_state):
    """Yield the per-position greek rows (Series) from ``greeks.by_position`` — empty
    when no greeks were computed."""
    greeks = getattr(account_state, "greeks", None)
    by_position = getattr(greeks, "by_position", None)
    if by_position is None or getattr(by_position, "empty", True):
        return
    for _, r in by_position.iterrows():
        yield r


def _symbol_by_underlying(account_state) -> dict:
    """Map an underlying bbg-ticker → a short display symbol, taken from the positions
    (option legs carry ``underlying_symbol``; equity/fund carry ``symbol``)."""
    out: dict = {}
    for p in getattr(account_state, "positions", []) or []:
        ac = getattr(p, "asset_class", None)
        if ac == "option":
            ut, sym = getattr(p, "underlying_bbg_ticker", None), getattr(p, "underlying_symbol", None)
        elif ac in ("equity", "fund_etf"):
            ut, sym = getattr(p, "bbg_ticker", None), getattr(p, "symbol", None)
        else:
            continue
        if ut and sym and ut not in out:
            out[ut] = sym
    return out


def _short_name(ticker: Optional[str]) -> str:
    """Fallback display name from a bbg ticker: 'AAPL US Equity' → 'AAPL'."""
    if not ticker:
        return "—"
    return str(ticker).split(" ")[0]


def economic_exposure_by_underlying(account_state) -> list[dict]:
    """Per-underlying net economic exposure — Σ ``dollar_delta`` over a name's stock +
    option legs — descending by magnitude. Each dict: ``underlying_ticker``,
    ``symbol`` (short display name), ``dollar_delta`` (signed delta-$), ``pct_nav``
    (signed delta-$ ÷ |NAV|, None when NAV is absent/zero).

    Conserved: Σ ``dollar_delta`` == the account's net dollar-delta
    (== ``AccountExposure.economic_exposure``). Options are **netted against stock**,
    so a covered call's name shows below its stock market value and a standalone short
    call / CSP shows negative — the whole point of the economic basis. Pure read of
    ``greeks.by_position`` + ``positions`` — no Bloomberg, no recompute."""
    nav = abs(_num(getattr(account_state, "nav", None)) or 0.0)
    sym_by_ut = _symbol_by_underlying(account_state)
    sums: dict = {}
    for r in _iter_greek_rows(account_state):
        dd = _num(r.get("dollar_delta"))
        if dd is None:
            continue
        ut = r.get("underlying_ticker")
        sums[ut] = sums.get(ut, 0.0) + dd
    out = [{
        "underlying_ticker": ut,
        "symbol": sym_by_ut.get(ut) or _short_name(ut),
        "dollar_delta": dd,
        "pct_nav": (dd / nav) if nav else None,
    } for ut, dd in sums.items()]
    out.sort(key=lambda d: abs(d["dollar_delta"]), reverse=True)
    return out


def economic_exposure_by_sector(account_state) -> list[dict]:
    """Per-GICS-sector net economic exposure — Σ ``dollar_delta`` of the names in each
    sector — descending by magnitude. Each dict: ``sector``, ``dollar_delta`` (signed
    delta-$), ``pct_nav`` (signed delta-$ ÷ |NAV|).

    Same conserved delta-$ basis as :func:`economic_exposure_by_underlying`; a name
    with no GICS sector rolls into 'Unclassified'. Unlike the legacy stock-MV sector
    diagnostic, **options are included and netted** — a short call/put reduces (or
    flips) its sector's exposure rather than vanishing. Pure read of
    ``greeks.by_position`` + the underlying snapshot."""
    nav = abs(_num(getattr(account_state, "nav", None)) or 0.0)
    snapshot = getattr(account_state, "snapshot", None)
    underlyings = getattr(snapshot, "underlyings", None)
    sums: dict = {}
    for r in _iter_greek_rows(account_state):
        dd = _num(r.get("dollar_delta"))
        if dd is None:
            continue
        sector = _sector_of(underlyings, r.get("underlying_ticker"))
        sums[sector] = sums.get(sector, 0.0) + dd
    out = [{"sector": s, "dollar_delta": v, "pct_nav": (v / nav) if nav else None}
           for s, v in sums.items()]
    out.sort(key=lambda d: abs(d["dollar_delta"]), reverse=True)
    return out


def economic_exposure_missing(account_state) -> dict:
    """What the economic-exposure breakdowns silently leave out: greek rows whose
    ``dollar_delta`` is missing are skipped by both by-underlying and by-sector,
    so a name whose every row is missing vanishes from the bars and a partially-
    missing name understates. Returns ``{"n_rows": int, "names": [str]}`` — the
    skipped row count and the affected display symbols — so the panels can say so
    instead of rendering a silently-partial book. Pure read of
    ``greeks.by_position``."""
    sym_by_ut = _symbol_by_underlying(account_state)
    n_rows = 0
    names: list[str] = []
    for r in _iter_greek_rows(account_state):
        if _num(r.get("dollar_delta")) is not None:
            continue
        n_rows += 1
        ut = r.get("underlying_ticker")
        sym = sym_by_ut.get(ut) or _short_name(ut)
        if sym not in names:
            names.append(sym)
    return {"n_rows": n_rows, "names": names}

