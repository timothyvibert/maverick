"""Top-level in-memory state container for the Portfolio Manager app.

The state is keyed by account. All accounts load in one pass: one BBG
roundtrip across the union of unique underlyings, then per-account
slices feed the per-account context / recommendation / greeks /
diagnostics pipelines.

Runtime state is in-memory and rebuilt on every load. Durable app state
(structure resolutions, alert suppressions, threshold overrides) persists in
the local SQLite store behind ``pm.store.db``; dated day-over-day snapshots
are a planned follow-on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from pm.config import DEFAULT_RISK_FREE_RATE
from pm.core.bloomberg_client import is_bloomberg_available
from pm.core.composite_score import compute_all_composite_scores
from pm.core.pitch_synthesizer import synthesize_pitch
from pm.core.portfolio_diagnostics import (
    PortfolioDiagnostics,
    compute_portfolio_diagnostics,
)
from pm.core.portfolio_greeks import PortfolioGreeks, compute_portfolio_greeks
from pm.core.portfolio_signals import compute_per_underlying_signals
from pm.core.portfolio_snapshot import (
    PortfolioSnapshot,
    fetch_portfolio_snapshot,
)
from pm.core.position_context import PositionContext, build_position_contexts
from pm.core.recommender import Recommendation, compute_recommendations
from pm.ingest.extract_loader import PortfolioExtract, find_latest_extract, load_portfolio_extract
from pm.ingest.position_builder import Position, build_positions

if TYPE_CHECKING:
    from pm.insight.client_profile import ClientProfile
    from pm.risk.exposure import AccountExposure


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-account state
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    """Everything Tab 2 needs for a single account.

    Note: two signal containers coexist by design.
      - ``signals_by_ticker``: legacy ``pm.core.portfolio_signals.Signal``
        objects keyed by BBG-format ticker (e.g. ``"AAPL US Equity"``).
        Produced by ``compute_per_underlying_signals`` and consumed by
        the legacy ``recommender.py``.
      - ``signals``: V1 insight ``SignalValue`` objects keyed by *bare
        symbol* (e.g. ``"AAPL"``).
        Produced by ``pm.insight.engine.run_insight_engine`` and
        consumed by the new pattern detectors / signal-sheet UI.

    The new ``fires`` list is populated by the insight engine in the
    same pass.
    """
    account: str
    nav: float
    positions: list[Position]
    snapshot: PortfolioSnapshot
    contexts: list[PositionContext]
    recommendations: list[Recommendation]
    diagnostics: PortfolioDiagnostics
    greeks: PortfolioGreeks
    composite_scores: dict
    signals_by_ticker: dict
    pitch_themes: list
    trades: pd.DataFrame
    trades_by_underlying: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Insight engine output (populated by run_insight_engine):
    signals: dict = field(default_factory=dict)  # underlying-symbol -> SignalDict (A–D, F)
    fires: list = field(default_factory=list)    # list[pm.insight.patterns.Fire]
    # Per-position merged SignalDict (underlying A–D/F layered with the
    # position's E group), keyed by Position.position_id. The engine already
    # builds this for the detectors; persisting it lets the signal-sheet UI
    # read Group E without recomputing.
    position_signals: dict = field(default_factory=dict)  # position_id -> SignalDict
    # Multi-leg structure groupings (covered call, collar, vertical, covered /
    # cash-secured put, straddle / strangle), detected in the load path after the
    # engine. The UI reads these and never recomputes. See pm.insight.structures.
    structures: list = field(default_factory=list)  # list[pm.insight.structures.Structure]
    # Portfolio exposure aggregation (net dollar greeks, SPX-relative dollar-beta,
    # market value vs economic exposure, vega by tenor, position->structure->account
    # rollup), computed in the load path after structure detection. The UI reads it
    # and never recomputes. See pm.risk.exposure.
    exposure: Optional["AccountExposure"] = None
    # Deterministic stress / scenario views (co-moving shock table at truth-CRR +
    # the beta-mapped portfolio P&L curve at fast BS2002). NOT populated by the
    # load path — the scenario section prices live via price_scenario (fast), and
    # the truth table is only written by an explicit run_account_scenario call
    # (a future commit-at-truth view). None until then. See pm.risk.scenario.
    scenario: Optional["AccountScenario"] = None
    # Per-structure Tier-2 economics (zero-shock breakevens + max P/L + PoP), precomputed
    # in the load path after exposure. The By-Structure grid reads the breakeven
    # from here; the payoff drawer carries the full set. structure_id -> dict.
    structure_tier2: dict = field(default_factory=dict)
    # Assignment risk + entry-cost economics, precomputed in the load path after
    # the Tier-2 pass: {"positions": {position_id -> record}, "structures":
    # {structure_id -> record}}. Position records carry P(assignment) on short
    # option legs (European N(d2) at the leg's own IV; None with a reason when
    # unpriceable), the early-exercise qualifier, and the position's entry-cost
    # breakeven(s) — extract-only, so breakevens fill with Bloomberg off.
    # Structure records carry the capped short-leg union. The UI reads this and
    # never recomputes. See pm.risk.assignment.
    assignment: dict = field(default_factory=dict)
    # Client behavioural profile derived from the account's trade history (strategy
    # posture, direction, tenor, sector lean, sizing, cadence + a coverage band),
    # computed in the load path after the risk runs. The UI reads it and never
    # recomputes. See pm.insight.client_profile.
    client_profile: Optional["ClientProfile"] = None


@dataclass
class PortfolioState:
    """Top-level state container. The single source of truth for the
    running app."""
    extract: PortfolioExtract
    loaded_at: datetime
    bloomberg_ok: bool
    accounts: dict[str, AccountState]
    all_warnings: list[str]
    # Pre-fetched BBG data the insight engine consumes (engine itself
    # does no I/O — all fetches happen here at load time).
    iv_histories: dict = field(default_factory=dict)        # BBG-ticker -> pd.Series
    analyst_data_by_ticker: dict = field(default_factory=dict)  # BBG-ticker -> dict
    analyst_note_dates_by_ticker: dict = field(default_factory=dict)  # BBG-ticker -> pd.Timestamp | None
    # Forward dividend forecast per underlying BBG ticker, in the
    # bloomberg_client.fetch_projected_dividend shape ({next, schedule}). Feeds
    # the ex-div fire's real-dividend path; empty when BBG is off or unforecast.
    projected_dividends_by_ticker: dict = field(default_factory=dict)
    # US Treasury curve (short->long tenor dicts) for the deep-ITM short-put
    # carry check; the fire picks the tenor nearest each leg's DTE. Empty when
    # BBG is off — the carry calc then uses the risk_free_rate scalar below.
    risk_free_curve: list = field(default_factory=list)
    # BBG-off fallback short rate for the carry check (see config). Used only
    # when risk_free_curve is empty; the fire flags it as an estimate.
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    # On-demand option-chain slice cache, owned by the scanner's sanctioned write
    # path (state_access.pull_slice). Two-level: {'chains': {underlier: {chain,
    # pulled_at}}, 'slices': {(underlier, window): {candidates, df, spot, pulled_at}}}.
    # Freshly empty each load, so a reload (which swaps in a new PortfolioState) drops
    # every cached slice — the marks never survive a Refresh stale.
    slice_cache: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_portfolio_state(
    data_dir: Path,
    bbg_ok: Optional[bool] = None,
    extract_path: Optional[Path] = None,
) -> PortfolioState:
    """End-to-end load: read a holdings extract, build positions, fetch a single BBG
    snapshot across the union of underlyings, then slice per account.

    ``bbg_ok``:
      - ``None``: probe Bloomberg via ``is_bloomberg_available()``.
      - ``True`` / ``False``: trust the caller (used by tests).
    ``extract_path``:
      - ``None``: read the latest extract in ``data_dir`` (pick up a newer file).
      - a path: read that specific extract (re-enrich the current file without
        switching to a newer one — backs the "Refresh BBG" path).
    """
    latest = Path(extract_path) if extract_path is not None else find_latest_extract(data_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No `<source>_extract_YYYYMMDD_HHMMSS.xlsx` found in {data_dir}."
        )

    extract = load_portfolio_extract(latest)
    positions = build_positions(extract)

    if bbg_ok is None:
        bbg_ok = is_bloomberg_available()

    global_snapshot = fetch_portfolio_snapshot(positions, bbg_ok)

    # 1Y IV history — ONE BDH across the union of underlyings, fetched here and
    # shared by BOTH consumers (the legacy signal pass's IV percentile and the
    # insight engine's B5). The two passes previously issued the same BDH twice
    # per load. Runs after the snapshot so venue-recovered tickers are final.
    iv_histories: dict = {}
    if bbg_ok:
        tickers = _underlying_ticker_union(positions)
        if tickers:
            try:
                from pm.core.bloomberg_client import fetch_iv_history
                iv_histories = fetch_iv_history(sorted(tickers), lookback_days=365) or {}
            except Exception as exc:
                logger.warning("IV-history prefetch failed: %s", exc)

    global_signals = compute_per_underlying_signals(
        global_snapshot.underlyings, bloomberg_available=bbg_ok,
        iv_histories=iv_histories,
    )

    # Pre-fetch the remaining BBG inputs the insight engine consumes (analyst
    # override for D1; analyst-note dates for D3; forward dividend forecast for
    # the ex-div fire). The IV histories fetched above pass through unchanged.
    iv_histories, analyst_data, analyst_note_dates, projected_dividends = _prefetch_insight_inputs(
        positions, bbg_ok, iv_histories=iv_histories,
    )

    # Treasury curve for the carry fire (one BDP round trip). Empty offline →
    # the carry calc falls back to the DEFAULT_RISK_FREE_RATE scalar.
    risk_free_curve: list = []
    if bbg_ok:
        try:
            from pm.core.bloomberg_client import fetch_risk_free_curve
            risk_free_curve = fetch_risk_free_curve() or []
        except Exception as exc:
            logger.warning("Treasury-curve prefetch failed: %s", exc)

    accounts: dict[str, AccountState] = {}
    for account_id in extract.accounts:
        accounts[account_id] = _build_account_state(
            account_id=account_id,
            all_positions=positions,
            all_trades=extract.trades,
            global_snapshot=global_snapshot,
            global_signals=global_signals,
        )

    all_warnings = list(extract.parse_warnings) + list(global_snapshot.fetch_warnings)
    for acc in accounts.values():
        if acc.greeks.warnings:
            sample = acc.greeks.warnings[:3]
            rest = len(acc.greeks.warnings) - len(sample)
            all_warnings.extend(f"[{acc.account}] {w}" for w in sample)
            if rest > 0:
                all_warnings.append(
                    f"[{acc.account}] … and {rest} more greeks warning(s)."
                )

    state = PortfolioState(
        extract=extract,
        loaded_at=datetime.now(),
        bloomberg_ok=bbg_ok,
        accounts=accounts,
        all_warnings=all_warnings,
        iv_histories=iv_histories,
        analyst_data_by_ticker=analyst_data,
        analyst_note_dates_by_ticker=analyst_note_dates,
        projected_dividends_by_ticker=projected_dividends,
        risk_free_curve=risk_free_curve,
    )

    # Run the insight engine, which writes signals + fires onto each
    # AccountState and may append further warnings. The engine reads the desk's
    # persisted threshold overrides: build_pattern_config layers any
    # saved overrides over the PatternConfig defaults, so a threshold edit takes
    # effect on the next reload (the persist-then-reload apply path) — every
    # reload route (both Refresh buttons -> reload_state -> here) picks them up.
    from pm.insight.engine import run_insight_engine
    from pm.store.settings_store import build_pattern_config
    run_insight_engine(state, build_pattern_config())

    # Detect multi-leg structures from the grouped holdings (pure; reads the
    # built positions, writes Structure proposals onto each AccountState).
    from pm.insight.structures import run_structure_detection
    run_structure_detection(state)

    # Portfolio exposure aggregation — net dollar greeks, SPX-relative dollar-beta,
    # market value vs economic exposure, vega by tenor, and the
    # position -> structure -> account rollup. Reads the already-loaded greeks +
    # structures + snapshot (no Bloomberg, no recompute) and stores the view on each
    # AccountState. Runs after detection so the rollup sees the recognised structures.
    from pm.risk.exposure import run_account_exposure
    run_account_exposure(state)

    # The scenario section prices live through the read-only price_scenario
    # recompute (fast vectorized BS2002) — nothing rendered reads a precomputed
    # scenario table, so none is computed here. run_account_scenario (the
    # truth-CRR preset table) remains available as an explicit call for a future
    # commit-at-truth view; in the load path it cost seconds per account and its
    # output was consumed by nothing.

    # Per-structure Tier-2 economics — zero-shock payoff per structure (engine
    # legs built once per account, reused), storing breakeven / max P/L / PoP on
    # acc.structure_tier2. Fills the By-Structure grid's Breakeven column (was the
    # 'pending pricing' stub). Pure/read-only; runs after exposure so the
    # structures + snapshot are in place.
    from pm.risk.payoff import run_structure_tier2
    run_structure_tier2(state)

    # Assignment risk — P(assignment) per short option leg (the leg's own IV;
    # None with a reason when unpriceable), the per-structure short-leg union,
    # the early-exercise qualifier, and entry-cost breakevens per position,
    # stored on acc.assignment. Reuses the engine legs + the payoff assembler;
    # pure/read-only, no Bloomberg call. Runs after the Tier-2 pass so the
    # structures + snapshot are in place.
    from pm.risk.assignment import run_account_assignment
    run_account_assignment(state)

    # Client behavioural profile from the trade history — strategy posture,
    # direction, tenor, sector lean, sizing, cadence + a coverage band, per
    # account. Pure read of the already-loaded trades + snapshot + positions (no
    # Bloomberg, no recompute); one bad account degrades to None, not a crash.
    from pm.insight.client_profile import run_account_profile
    run_account_profile(state)

    # Structure-aware management fires (coverage breach, ex-div context, carry,
    # at-cap, pin, collar-monetize). Appends Fires to each AccountState and
    # annotates leg fires that belong to a confirmed structure.
    from pm.insight.structure_fires import run_structure_fires
    run_structure_fires(state)

    # Apply persisted alert suppressions: mark each fire whose
    # (account, underlying, pattern_id) is actively suppressed or snoozed. Runs
    # AFTER the engine, so it only flags already-computed fires — no recompute,
    # no Bloomberg — and an un-suppress brings the fire straight back next load.
    from pm.store.suppression_store import apply_suppressions
    apply_suppressions(state)

    # Material-change re-surfacing: over the just-marked fires, flip a muted
    # alert back to active when its condition has moved materially since mute time —
    # comparing the suppression's captured baseline against the current fire's
    # already-computed trace. Read-path mark only: no recompute, no Bloomberg.
    from pm.store.suppression_store import apply_material_change
    apply_material_change(state)

    # Alert governance, in order: per-fire acknowledgements (with their own
    # material-change net), then per-pattern disables LAST — an explicit
    # off-switch is final and overrides even a re-surfaced mark. Both are
    # marking passes over already-computed fires: nothing is dropped from
    # acc.fires, so flipping a toggle back restores the alerts in place.
    from pm.store.alert_governance import apply_acknowledgements, apply_pattern_disables
    apply_acknowledgements(state)
    apply_pattern_disables(state)

    return state


def reapply_thresholds(state: PortfolioState) -> PortfolioState:
    """Re-evaluate the alert set over ALREADY-LOADED state under the currently
    persisted threshold overrides — the recompute-only Apply path. No extract
    re-read, no Bloomberg call: a threshold edit changes which fires the engine
    PRODUCES, so the fires must be re-derived, but every input the engine reads
    (snapshot, IV histories, analyst data, dividends, curve) is already on the
    state and is unchanged by a dial edit.

    Re-runs exactly the passes whose output depends on PatternConfig or on the
    fire set it produces, in load-path order:
      1. ``run_insight_engine(state, build_pattern_config())`` — replaces each
         account's signals / fires / position_signals wholesale (the engine
         resets them per account; it never appends to a prior run's output).
      2. ``run_structure_fires(state)`` — P16–P20 read module constants, not
         PatternConfig, but they live in ``acc.fires`` which the engine just
         reset, so they are re-derived and the structure-leg annotations
         re-attached (both idempotent by design).
      3. ``apply_suppressions`` + ``apply_material_change`` — the re-run
         produces fresh, unmarked Fire objects; the marking passes re-flag
         them from the persisted store, same as a load.
    Structure detection, exposure, tier-2 economics, assignment risk and the
    client profile do NOT re-run: none reads PatternConfig, and each lives on
    its own AccountState field untouched by the engine. ``state.all_warnings``
    is left as loaded — this path adds no load events to warn about.

    Mutates and returns the same state object.
    """
    from pm.insight.engine import run_insight_engine
    from pm.insight.structure_fires import run_structure_fires
    from pm.store.alert_governance import apply_acknowledgements, apply_pattern_disables
    from pm.store.settings_store import build_pattern_config
    from pm.store.suppression_store import apply_material_change, apply_suppressions

    run_insight_engine(state, build_pattern_config())
    run_structure_fires(state)
    apply_suppressions(state)
    apply_material_change(state)
    apply_acknowledgements(state)
    apply_pattern_disables(state)
    return state


def refresh_portfolio_state(
    current: PortfolioState | None,
    data_dir: Path,
    reuse_extract: bool = False,
) -> PortfolioState:
    """Re-runs ``load_portfolio_state``. Re-probes Bloomberg every time —
    one BDP call is cheap and avoids carrying stale connection state if
    the Terminal comes online mid-session. Logs whether the extract
    timestamp changed.

    ``reuse_extract``: when True (and a prior state exists), re-read the *current*
    extract file rather than the latest in the data dir — re-pulling market data on
    the same holdings ("Refresh BBG"). When False, read the latest file ("Refresh
    Acct Data" / initial load), picking up a newer extract if one has landed."""
    extract_path = (current.extract.source_path
                    if (reuse_extract and current is not None) else None)
    new_state = load_portfolio_state(data_dir, bbg_ok=None, extract_path=extract_path)
    if current is not None:
        if new_state.extract.extract_ts != current.extract.extract_ts:
            logger.info(
                "Refresh: extract timestamp changed %s -> %s",
                current.extract.extract_ts, new_state.extract.extract_ts,
            )
        else:
            logger.info(
                "Refresh: extract timestamp unchanged (%s).",
                new_state.extract.extract_ts,
            )
    return new_state


# ---------------------------------------------------------------------------
# Insight-input pre-fetch (BBG I/O happens here, not in the engine)
# ---------------------------------------------------------------------------

def _underlying_ticker_union(positions: list[Position]) -> set[str]:
    """Unique underlying BBG tickers across the book: equity/fund rows by their
    own ticker, option rows by their underlier. The set every whole-book
    prefetch (IV history, analyst data, note dates, dividends) runs over."""
    tickers: set[str] = set()
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and p.bbg_ticker:
            tickers.add(p.bbg_ticker)
        elif p.asset_class == "option" and p.underlying_bbg_ticker:
            tickers.add(p.underlying_bbg_ticker)
    return tickers


def _prefetch_insight_inputs(
    positions: list[Position], bbg_ok: bool,
    iv_histories: Optional[dict] = None,
) -> tuple[dict, dict, dict, dict]:
    """Fetch the BBG-derived inputs the insight engine needs:
      - 1Y IV history per unique underlying BBG ticker (B5) — or, when the
        caller already fetched it (the load path shares one BDH with the
        legacy signal pass), passed through unchanged via ``iv_histories``
      - provider-overridden analyst data per unique underlying (D1)
      - analyst-note dates per unique underlying (D3)
      - forward dividend forecast per unique underlying (ex-div fire / P7)
    Returns (iv_histories, analyst_data_by_ticker, analyst_note_dates_by_ticker,
    projected_dividends_by_ticker). All four are empty when ``bbg_ok`` is False
    or no underlyings are present.
    """
    analyst_data: dict = {}
    analyst_note_dates: dict = {}
    projected_dividends: dict = {}
    if not bbg_ok:
        return iv_histories or {}, analyst_data, analyst_note_dates, projected_dividends

    tickers = _underlying_ticker_union(positions)
    if not tickers:
        return iv_histories or {}, analyst_data, analyst_note_dates, projected_dividends

    # IV histories — one BDH call across all tickers, unless already fetched
    if iv_histories is None:
        iv_histories = {}
        try:
            from pm.core.bloomberg_client import fetch_iv_history
            iv_histories = fetch_iv_history(sorted(tickers), lookback_days=365) or {}
        except Exception as exc:
            logger.warning("Insight prefetch: IV history failed: %s", exc)

    # Analyst data — ONE batched BDP with the provider override across all
    # names (was one single-ticker call per name; measured on the live book:
    # ~13s of serial round trips -> ~3s). Fails closed to per-ticker {}.
    try:
        from pm.core.bloomberg_client import fetch_analyst_data_batch
        analyst_data = fetch_analyst_data_batch(sorted(tickers))
    except Exception as exc:
        logger.warning("Insight prefetch: analyst data fetch failed: %s", exc)
        analyst_data = {t: {} for t in tickers}

    # Analyst-note dates (D3) — one batched BDP with the override pair
    try:
        from pm.core.bloomberg_client import fetch_analyst_note_dates, with_session
        with with_session() as query:
            note_df = fetch_analyst_note_dates(query, sorted(tickers))
        for t in sorted(tickers):
            ts = None
            if note_df is not None and not note_df.empty and t in note_df.index:
                val = note_df.loc[t, "analyst_note_date"]
                if not pd.isna(val):
                    ts = val
            analyst_note_dates[t] = ts
    except Exception as exc:
        logger.warning("Insight prefetch: analyst-note dates failed: %s", exc)

    # Forward dividend forecast — ONE batched BDP on the bulk BDVD field across
    # all names (was one call per ticker; measured on the live book: ~8s of
    # serial round trips -> ~1.4s). A failed batch degrades every name to the
    # empty shape, matching the singular fetcher's per-name degradation.
    try:
        from pm.core.bloomberg_client import fetch_projected_dividends
        projected_dividends = fetch_projected_dividends(sorted(tickers))
    except Exception as exc:
        logger.warning("Insight prefetch: projected dividends fetch failed: %s", exc)
        projected_dividends = {t: {} for t in tickers}

    return iv_histories, analyst_data, analyst_note_dates, projected_dividends


# ---------------------------------------------------------------------------
# Per-account assembly
# ---------------------------------------------------------------------------

def _build_account_state(
    *,
    account_id: str,
    all_positions: list[Position],
    all_trades: pd.DataFrame,
    global_snapshot: PortfolioSnapshot,
    global_signals: dict,
) -> AccountState:
    positions = [p for p in all_positions if p.account == account_id]
    # NAV = net asset value (signed sum of market values across all
    # positions in the account). Short option market values are negative
    # and subtract correctly. portfolio_pct_of_nav uses abs(mv)/nav.
    nav = sum(p.market_value for p in positions if p.market_value is not None)

    account_snapshot = _slice_snapshot(global_snapshot, positions)
    iv_pctl = _iv_pctl_dict(global_signals)

    contexts = build_position_contexts(
        positions, account_snapshot, account_nav=nav, iv_pctl_by_ticker=iv_pctl,
    )
    recommendations = compute_recommendations(contexts, global_signals)
    greeks = compute_portfolio_greeks(
        positions, account_snapshot.underlyings, account_snapshot.options,
    )
    diagnostics = compute_portfolio_diagnostics(
        positions, account_snapshot.underlyings,
    )

    account_signals = {
        ticker: sigs for ticker, sigs in global_signals.items()
        if ticker in {
            p.bbg_ticker for p in positions if p.asset_class in ("equity", "fund_etf")
        } | {
            p.underlying_bbg_ticker for p in positions if p.asset_class == "option"
        }
    }
    composite_scores = compute_all_composite_scores(account_signals)
    pitch_themes = synthesize_pitch(recommendations)

    trades = _filter_trades(all_trades, account_id)
    trades_by_underlying = _group_trades_by_underlying(trades, positions)

    return AccountState(
        account=account_id,
        nav=float(nav),
        positions=positions,
        snapshot=account_snapshot,
        contexts=contexts,
        recommendations=recommendations,
        diagnostics=diagnostics,
        greeks=greeks,
        composite_scores=composite_scores,
        signals_by_ticker=account_signals,
        pitch_themes=pitch_themes,
        trades=trades,
        trades_by_underlying=trades_by_underlying,
    )


def _slice_snapshot(
    global_snapshot: PortfolioSnapshot,
    positions: list[Position],
) -> PortfolioSnapshot:
    """Filter the global snapshot to only the tickers this account holds."""
    under_tickers: set[str] = set()
    opt_tickers: set[str] = set()
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and p.bbg_ticker:
            under_tickers.add(p.bbg_ticker)
        elif p.asset_class == "option":
            if p.underlying_bbg_ticker:
                under_tickers.add(p.underlying_bbg_ticker)
            if p.bbg_ticker:
                opt_tickers.add(p.bbg_ticker)

    under_df = global_snapshot.underlyings
    if not under_df.empty:
        keep = [t for t in under_tickers if t in under_df.index]
        under_df = under_df.loc[keep] if keep else under_df.iloc[0:0]

    opt_df = global_snapshot.options
    if not opt_df.empty:
        keep_opts = [t for t in opt_tickers if t in opt_df.index]
        opt_df = opt_df.loc[keep_opts] if keep_opts else opt_df.iloc[0:0]

    return PortfolioSnapshot(
        underlyings=under_df,
        options=opt_df,
        fetch_warnings=list(global_snapshot.fetch_warnings),
        bloomberg_available=global_snapshot.bloomberg_available,
    )


def _filter_trades(trades: pd.DataFrame, account_id: str) -> pd.DataFrame:
    if trades is None or trades.empty or "account" not in trades.columns:
        return trades.iloc[0:0] if trades is not None else pd.DataFrame()
    df = trades[trades["account"] == account_id].copy()
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
    return df


def _group_trades_by_underlying(
    trades: pd.DataFrame,
    positions: list[Position],
) -> dict[str, pd.DataFrame]:
    """Key by underlying_ticker for options and ticker_final for
    equities/funds. Each value is a Trade-sheet slice sorted by trade_date asc.

    The set of keys is the union of:
      - option underlying_tickers seen in the trades themselves, and
      - equity/fund tickers seen in the account's positions
    so that every position with potential trade activity gets a slot
    (possibly empty).
    """
    out: dict[str, pd.DataFrame] = {}
    if trades is None or trades.empty:
        for p in positions:
            if p.asset_class in ("equity", "fund_etf") and p.symbol:
                out.setdefault(p.symbol, pd.DataFrame(columns=trades.columns if trades is not None else []))
        return out

    sorted_trades = trades.sort_values("trade_date") if "trade_date" in trades.columns else trades

    # Options: key by underlying_ticker
    if "underlying_ticker" in sorted_trades.columns and "asset_class" in sorted_trades.columns:
        opt_trades = sorted_trades[sorted_trades["asset_class"] == "option"]
        for ticker, sub in opt_trades.groupby("underlying_ticker"):
            if isinstance(ticker, str) and ticker:
                out[ticker] = sub.reset_index(drop=True)

    # Equity / fund trades: key by ticker_final
    if "ticker_final" in sorted_trades.columns and "asset_class" in sorted_trades.columns:
        eq_trades = sorted_trades[sorted_trades["asset_class"].isin(("equity", "fund_etf"))]
        for ticker, sub in eq_trades.groupby("ticker_final"):
            if isinstance(ticker, str) and ticker:
                out[ticker] = sub.reset_index(drop=True)

    # Make sure every held equity/fund has a slot (possibly empty)
    empty_template = sorted_trades.iloc[0:0]
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and p.symbol:
            out.setdefault(p.symbol, empty_template)

    return out


def _iv_pctl_dict(signals_by_ticker: dict) -> dict[str, float]:
    """Extract iv_percentile.metric_value into a {ticker: pctl} dict."""
    out: dict[str, float] = {}
    for ticker, sigs in signals_by_ticker.items():
        for s in sigs:
            if s.signal_type == "iv_percentile":
                out[ticker] = s.metric_value
                break
    return out
