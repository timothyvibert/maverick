"""Insight engine entry point.

Composes the signal library + pattern detectors.

The engine is purely computational: no BBG I/O, no disk I/O. All BBG
data (underlying snapshot, IV histories, analyst data) is fetched
upstream in ``pm.store.portfolio_state.load_portfolio_state`` and
attached to ``PortfolioState`` before the engine runs.

Side effects:
- Populates ``account_state.signals`` (underlying → SignalDict).
- Populates ``account_state.fires`` (list[Fire]).
- Replaces the ``[insight]``-prefixed lines in ``state.all_warnings`` with this
  run's stale-skip notes (capped): one line per (account, pattern, missing
  signal) naming how many names could not be evaluated. Replacement, not
  append, so a threshold recompute over the same state never stacks
  duplicates. An eligibility non-fire stays silent; only a required signal
  that is missing/stale reports (see ``patterns._required_signal``).
"""
from __future__ import annotations

import logging
from typing import Optional

from pm.insight import patterns as _patterns
from pm.insight.patterns import (
    ACCOUNT_LEVEL_DETECTORS,
    Fire,
    PER_POSITION_DETECTORS,
    PatternConfig,
)
from pm.insight.signal_library import (
    SignalDict,
    SignalValue,
    compute_position_signals,
    compute_signals_for_underlying,
)


logger = logging.getLogger(__name__)

_WARNING_CAP = 20


def _unique_underlyings(account_state) -> list[str]:
    """Set of bare-symbol underlying names for the account."""
    out: set[str] = set()
    for p in account_state.positions:
        if p.asset_class in ("equity", "fund_etf") and p.symbol:
            out.add(p.symbol)
        elif p.asset_class == "option" and p.underlying_symbol:
            out.add(p.underlying_symbol)
    return sorted(out)


def _bbg_ticker_for_underlying(account_state, symbol: str) -> Optional[str]:
    """First BBG ticker we find for this symbol across the account."""
    for p in account_state.positions:
        if p.asset_class in ("equity", "fund_etf") and p.symbol == symbol:
            return p.bbg_ticker or None
        if p.asset_class == "option" and p.underlying_symbol == symbol:
            return p.underlying_bbg_ticker or None
    return None


def _snapshot_row_for(account_state, bbg_ticker: Optional[str]) -> Optional[dict]:
    if not bbg_ticker:
        return None
    df = account_state.snapshot.underlyings
    if df is None or df.empty or bbg_ticker not in df.index:
        return None
    series = df.loc[bbg_ticker]
    return {col: series[col] for col in df.columns}


def _merge_signals(underlying: SignalDict, position: SignalDict) -> SignalDict:
    """Per the spec: position signals (E*) layer on top of underlying signals
    (A/B/C/D/F). No key collisions expected, but position values win on tie."""
    merged: SignalDict = dict(underlying)
    merged.update(position)
    return merged


def run_insight_engine(
    state,
    config: Optional[PatternConfig] = None,
) -> list[Fire]:
    """Evaluate all signals + all patterns across all accounts.

    Returns the flat list of fires; also mutates each
    ``AccountState.signals`` and ``AccountState.fires``.
    """
    config = config or PatternConfig()
    all_fires: list[Fire] = []
    skip_warnings: list[str] = []
    skips_by_account: dict[str, dict] = {}

    iv_histories = getattr(state, "iv_histories", {}) or {}
    analyst_data_by_ticker = getattr(state, "analyst_data_by_ticker", {}) or {}
    analyst_note_dates_by_ticker = getattr(state, "analyst_note_dates_by_ticker", {}) or {}
    projected_dividends_by_ticker = getattr(state, "projected_dividends_by_ticker", {}) or {}

    for account_id, account_state in state.accounts.items():
        # ---- Stage 1: compute per-underlying signals ----------------------
        underlying_signals: dict[str, SignalDict] = {}
        for symbol in _unique_underlyings(account_state):
            bbg = _bbg_ticker_for_underlying(account_state, symbol)
            snap = _snapshot_row_for(account_state, bbg)
            iv_hist = iv_histories.get(bbg) if bbg else None
            analyst = analyst_data_by_ticker.get(bbg) if bbg else None
            analyst_note = analyst_note_dates_by_ticker.get(bbg) if bbg else None
            projected_div = projected_dividends_by_ticker.get(bbg) if bbg else None
            legacy = (account_state.signals_by_ticker.get(bbg)
                      if bbg and hasattr(account_state, "signals_by_ticker") else None)
            sig_dict = compute_signals_for_underlying(
                underlying=symbol,
                snapshot_row=snap,
                iv_history=iv_hist,
                positions_in_account=account_state.positions,
                account_nav=account_state.nav,
                analyst_data=analyst,
                legacy_signals=legacy,
                analyst_note_date=analyst_note,
                projected_dividend=projected_div,
                # The one definition of "recent note" every consumer shares —
                # the editable research-note recency dial.
                note_recent_window_bd=config.p21_note_window_bd,
            )
            underlying_signals[symbol] = sig_dict

        account_state.signals = underlying_signals
        account_state.fires = []
        account_state.position_signals = {}

        # ---- Stage 2: per-position signals + per-position detectors -------
        _patterns.begin_skip_collection()
        account_fires: list[Fire] = []
        for position in account_state.positions:
            symbol = position.underlying_symbol or position.symbol
            if not symbol:
                continue  # cash/other with no symbol
            base = underlying_signals.get(symbol, {})
            snap = None
            bbg = _bbg_ticker_for_underlying(account_state, symbol)
            if bbg:
                snap = _snapshot_row_for(account_state, bbg)
            pos_signals = compute_position_signals(position, snap, account_state.nav)
            merged = _merge_signals(base, pos_signals)
            # Persist the merged dict (A–D/F + this position's E group) so the
            # signal-sheet UI can read Group E without recomputing.
            account_state.position_signals[position.position_id] = merged

            for pattern_id, detector in PER_POSITION_DETECTORS:
                try:
                    fire = detector(position, account_state, merged, config)
                except Exception as exc:
                    logger.warning(
                        "Detector %s raised for %s/%s: %s",
                        pattern_id, account_id, position.position_id, exc,
                    )
                    fire = None
                if fire is not None:
                    account_fires.append(fire)

        # ---- Stage 3: account-level detectors -----------------------------
        for pattern_id, detector in ACCOUNT_LEVEL_DETECTORS:
            try:
                fires = detector(account_state, config)
            except Exception as exc:
                logger.warning(
                    "Account-level detector %s raised for %s: %s",
                    pattern_id, account_id, exc,
                )
                fires = []
            account_fires.extend(fires or [])

        account_state.fires = account_fires
        all_fires.extend(account_fires)

        # ---- Stage 3.5: aggregate this account's stale-skips --------------
        by_gap: dict[tuple[str, str], set[str]] = {}
        for pid, entity, sid in _patterns.drain_skip_collection():
            by_gap.setdefault((pid, sid), set()).add(entity)
        for (pid, sid), names in sorted(by_gap.items()):
            noun = "name" if len(names) == 1 else "names"
            skip_warnings.append(
                f"[insight] {account_id}: {pid} not evaluated on "
                f"{len(names)} {noun} — {sid} unavailable"
            )
        if by_gap:
            # Structured form of the same gaps, for the status-bar coverage
            # chip: (pattern, name) deduped so one dark name skipped by two
            # required signals counts once.
            skips_by_account[account_id] = {
                "n_not_evaluated": len({(pid, nm) for (pid, _sid), nms in by_gap.items()
                                        for nm in nms}),
                "n_patterns": len({pid for (pid, _sid) in by_gap}),
                "gaps": [{"pattern": pid, "signal": sid, "n_names": len(nms)}
                         for (pid, sid), nms in sorted(by_gap.items())],
            }

    # ---- Stage 4: stale-skip warnings ------------------------------------
    # Replace this engine's previous notes rather than stacking them: a
    # threshold recompute re-runs the engine over the same loaded state.
    if getattr(state, "all_warnings", None) is not None:
        state.all_warnings = [w for w in state.all_warnings
                              if not w.startswith("[insight]")]
        cap = skip_warnings[:_WARNING_CAP]
        state.all_warnings.extend(cap)
        rest = len(skip_warnings) - len(cap)
        if rest > 0:
            state.all_warnings.append(f"[insight] … and {rest} more skip warning(s).")
    # Whole-run product, assigned not merged — the recompute replaces it.
    state.insight_skips = skips_by_account

    return all_fires
