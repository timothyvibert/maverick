"""Read/own the UI layer's runtime state.

The UI never recomputes — it reads what ``run_insight_engine`` already
produced and attached to ``PortfolioState``. The singleton PortfolioState is
OWNED here (``_RUNTIME``), because this module is only ever imported as
``pm.ui.state_access`` — never executed as ``__main__``. ``pm/app.py`` is the
entry point and, under ``python -m pm.app``, runs as ``__main__``; a global
stored there would be a *different* object from the ``pm.app`` callbacks
import, so the state would be invisible to them. Owning it here gives one
canonical instance for both the entry point and every callback.

The scanner's service layer lives in ``pm.scanner`` (slice / candidates /
ticket / view), Dash-free and singleton-free: every implementation takes the
loaded state explicitly. The facade below resolves the singleton ONCE per
call and threads that one object through the whole chain — so a scan
computes against a single coherent state, and this module stays the one
seam the UI imports. Shared pure reads (``coerce_float``, ``position_by_id``
and friends) live in ``pm.store.state_reads`` and are re-exported here.
"""
from __future__ import annotations

import threading
from typing import Optional

from pm.ingest.position_builder import Position
from pm.insight.patterns import Fire
from pm.insight.signal_library import SignalDict, SignalValue
from pm.scanner import (candidates as _scan_candidates, slice as _scan_slice,
                        ticket as _scan_ticket, view as _scan_view)
from pm.scanner.candidates import (  # noqa: F401 — re-exported scanner helpers
    STRUCTURE_AUTO, _contemporaneous_mid, _held_option_delta, _held_stock,
    _per_share_basis, _resolve_scan_structure_id, _structure_roll_context)
from pm.scanner.slice import (  # noqa: F401 — re-exported scanner helpers
    _as_date, _band_filter_df, _contract_metrics, _div_yield,
    _drop_candidate_caches, _resolve_expiry_range, _scan_sig, _slice_iv_rank,
    _slice_surface, _snapshot_underlying_row, _spot_from_snapshot,
    _spot_slice_df)
from pm.scanner.view import (  # noqa: F401 — re-exported scanner helpers
    _empty_scan_note, _slice_contracts, _window_expiry_counts)
from pm.store.portfolio_state import PortfolioState
from pm.store.state_reads import (  # noqa: F401 — shared pure reads, re-exported
    coerce_float, is_missing, position_by_id, structure_for_position)


# ---------------------------------------------------------------------------
# Signal-sheet group catalog (display order + display names per Part 1/3.9).
# Groups A–D, F come from AccountState.signals[underlying]; group E is
# per-position and read from AccountState.position_signals[position_id].
# ---------------------------------------------------------------------------

GROUP_A = ("A — Trend & Momentum", [
    ("spot_vs_50d_ma", "Spot vs 50d MA"),
    ("spot_vs_200d_ma", "Spot vs 200d MA"),
    ("ma_stack_regime", "MA stack regime"),
    ("return_horizons", "Returns (1D / 5D / 3M / YTD / 1Y)"),
    ("rsi_14d_regime", "RSI 14d + regime"),
    ("distance_from_52w_high", "Distance from 52w high"),
    ("distance_from_52w_low", "Distance from 52w low"),
    ("vol_adjusted_move", "Vol-adjusted move (today)"),
])
GROUP_B = ("B — Volatility", [
    ("rv_30d", "Realized vol (30d)"),
    ("iv_1m_atm", "IV 1M ATM"),
    ("iv_3m_atm", "IV 3M ATM"),
    ("iv_6m_atm", "IV 6M ATM"),
    ("iv_3m_percentile_1y", "IV 3M percentile (1Y range)"),
    ("iv_term_structure", "IV term structure (3M − 6M)"),
    ("vrp_30d", "Vol risk premium (1M IV − 30d RV)"),
])
GROUP_C = ("C — Catalysts", [
    ("days_to_earnings", "Days to earnings"),
    ("earnings_implied_move", "Earnings implied move"),
    ("days_to_ex_div", "Days to ex-dividend"),
    ("dte_nearest_expiry_in_account", "DTE to nearest expiry (account)"),
])
GROUP_D = ("D — Sentiment & Ratings", [
    ("analyst_rating_and_target", "Analyst rating / target / upside"),
    ("street_consensus_rating_and_target", "Street rating / target / upside"),
    ("analyst_note_recent", "Analyst note (recent)"),
])
GROUP_E = ("E — Position-specific", [
    ("position_size_pct_of_nav", "Position size (% of NAV)"),
    ("position_unrealized_pnl_pct", "P&L %"),
    ("option_captured_pct", "Premium captured (%)"),
    ("option_dte", "DTE"),
    ("option_moneyness", "Moneyness"),
])
GROUP_F = ("F — Composite", [
    ("composite_score", "Composite score (0–100)"),
])

# A–D, F come from the per-underlying SignalDict; E is per-position.
UNDERLYING_GROUPS = [GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_F]
POSITION_GROUP = GROUP_E


# ---------------------------------------------------------------------------
# Global runtime state — OWNED HERE.
#
# This must live in a module that is only ever imported as ``pm.ui.state_access``
# (never executed as ``__main__``). If the global lived in ``pm/app.py``,
# ``python -m pm.app`` would run that file as ``__main__`` — a *separate* module
# object from the ``pm.app`` that ``get_state`` imports — so state set at startup
# would be invisible to callbacks (get_state() → None → dead drawers). Keeping
# the singleton here guarantees one instance for both the entry point and every
# callback.
# ---------------------------------------------------------------------------

_RUNTIME: dict = {"state": None, "active_account": None}

# The server is threaded (app.run(threaded=True)) with zero framework-level
# serialization, so the owner provides its own:
#   _RELOAD_LOCK serializes the full-load route. A caller that arrives while a
#   load is in flight (a double-clicked Refresh) queues on the lock and, once
#   through, sees the sequence number moved and returns the fresh state instead
#   of starting a second multi-second Bloomberg pull.
#   _WRITE_LOCK serializes the fast sanctioned write paths (resolve_structure,
#   suppress_alert / restore_alert, recompute_thresholds) against each other so
#   two overlapping writes cannot interleave their state mutations. It is never
#   held across a Bloomberg call, so it cannot stall the UI.
# The two locks never nest.
_RELOAD_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()

_RELOAD_SEQ = 0   # bumped after each completed load; lets a queued caller see one finished while it waited


def get_state() -> Optional[PortfolioState]:
    """Return the current global PortfolioState, or None if not loaded."""
    return _RUNTIME.get("state")


def set_state(state: Optional[PortfolioState],
              active_account: Optional[str] = None) -> Optional[PortfolioState]:
    """Install the global PortfolioState (called once at app build)."""
    _RUNTIME["state"] = state
    if active_account is not None:
        _RUNTIME["active_account"] = active_account
    return state


def reload_state(reuse_extract: bool = False) -> Optional[PortfolioState]:
    """Refresh the global PortfolioState in place. Returns the new state.

    ``reuse_extract``: re-enrich the current extract file ("Refresh BBG"); when
    False, read the latest extract in the data dir ("Refresh Acct Data" / first load).

    Serialized: one full load runs at a time. A caller that arrives while a
    load is in flight waits for it and returns THAT load's state rather than
    starting a second Bloomberg pull — so a double-clicked Refresh costs one
    load, not two. (A cross-button race — e.g. Refresh Acct Data queued behind
    an in-flight Refresh BBG — also coalesces into the in-flight load; clicking
    again once it completes runs the intended route.)"""
    global _RELOAD_SEQ
    from pm.config import EXTRACT_DATA_DIR
    from pm.store.portfolio_state import refresh_portfolio_state
    seq_at_entry = _RELOAD_SEQ
    with _RELOAD_LOCK:
        if _RELOAD_SEQ != seq_at_entry:
            # A concurrent reload completed while this caller waited on the
            # lock; its freshly-built state answers this request too.
            return _RUNTIME.get("state")
        prev = _RUNTIME.get("state")
        new_state = refresh_portfolio_state(prev, EXTRACT_DATA_DIR, reuse_extract=reuse_extract)
        _RUNTIME["state"] = new_state
        _RELOAD_SEQ += 1
        return new_state


def recompute_thresholds() -> Optional[PortfolioState]:
    """Re-derive the alert set over the loaded state under the persisted
    threshold overrides — the Apply path's recompute. A sanctioned
    owned-state write path (like ``resolve_structure``): it re-runs the
    engine + structure fires + suppression marking over data already on the
    state, with **no Bloomberg call and no extract re-read** — a dial edit
    changes which fires the engine produces, not the market data they read.
    Every other load-path product (snapshot, structures, exposure, tier-2,
    client profile) is untouched. Returns the same state object, or None when
    nothing is loaded (callers then fall back to a full reload)."""
    state = _RUNTIME.get("state")
    if state is None:
        return None
    from pm.store.portfolio_state import reapply_thresholds
    with _WRITE_LOCK:
        return reapply_thresholds(state)


def refresh_scanner_params() -> bool:
    """Invalidate every cached candidate ranking so the next Scan re-ranks under
    the persisted scanner dials — the Thresholds-tab Apply path for the SCANNER
    group. A sanctioned owned-state write under the write lock: cached rankings
    embed the parameters they were ranked under, so a dial change must drop
    them or a reopened scan would serve an order the dials no longer describe.
    Raw slices, chains and per-expiry frames stay (no Bloomberg); fires,
    signals and structures are untouched (no engine recompute — scanner dials
    change candidate ORDER, never alerts). Returns True when a loaded state
    was re-marked."""
    with _WRITE_LOCK:
        state = _RUNTIME.get("state")
        if state is None:
            return False
        for account in state.accounts:
            _drop_candidate_caches(state, account)
        (getattr(state, "slice_cache", None) or {}).pop("overlay_ranked", None)
        return True


def price_scenario(
    account: str, *, spot_pct: float = 0.0, vol_pts: float = 0.0,
    rate_bps: float = 0.0, time_days: int = 0, target=None, mode: str = "fast",
) -> Optional[dict]:
    """The one sanctioned scenario recompute (the live dial). Reprices the account's
    book over a co-moving shock — spot (beta-mapped) / vol pts / rate bps / time —
    purely over already-loaded state: **no Bloomberg, no reload, and (unlike
    ``resolve_structure``) no write-back to ``_RUNTIME``** — a hypothetical must not
    mutate owned state. Returns ``{account, positions[], grid}`` or None.

    ``mode='fast'`` (vectorized BS2002) drives the live dial + heatmap grid;
    ``mode='truth'`` (CRR) is for a committed point. The spot×vol grid is always fast
    — a sweep is never priced at truth.
    """
    state = _RUNTIME.get("state")
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    from pm.risk.scenario import ShockSpec, shock_reprice, spot_vol_grid
    shock = ShockSpec(name="custom", label="custom", spot_pct=spot_pct, vol_pts=vol_pts,
                      rate_bps=rate_bps, time_days=int(time_days))
    # The impact table is always the full book (every position's P&L under the shock);
    # ``target`` drills only the heatmap surface, never the table.
    impact = shock_reprice(state, acc, shock, mode=mode)
    grid = spot_vol_grid(state, acc, rate_bps=rate_bps, time_days=int(time_days), target=target,
                         point_spot_pct=spot_pct, point_vol_pts=vol_pts)
    return {
        "account": {"pnl": impact["account_pnl"], "pnl_pct": impact["account_pnl_pct"],
                    "axes": shock.axes(), "mode": mode, "target": target,
                    "n_priced": impact.get("n_priced"),
                    "n_skipped": impact.get("n_skipped"),
                    "beta_excluded_names": impact.get("beta_excluded_names"),
                    # current-vs-stressed exposure totals (engine-priced both
                    # sides, account scope) — the reshape view's substrate
                    "exposures": impact.get("exposures")},
        "positions": impact["rows"],
        "grid": grid,
    }


def price_payoff(
    account: str, *, structure_id: Optional[str] = None,
    position_id: Optional[str] = None, shock: Optional[dict] = None,
):
    """The structure/position-level read-only payoff recompute — the payoff drawer's
    live dial, the per-level analogue of ``price_scenario``. Looks up the target (a
    structure by id, else a standalone position) in the loaded state and returns its
    ``PayoffResult``. Read-only: **no Bloomberg, no reload, no ``_RUNTIME`` write-back**
    — a hypothetical must not mutate owned state. None if the state/account/target is
    missing. ``shock`` is ``{spot_pct, vol_pts, rate_bps, time_days}`` (None = base)."""
    state = _RUNTIME.get("state")
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    target = None
    if structure_id:
        target = next((s for s in (acc.structures or []) if s.structure_id == structure_id), None)
    elif position_id:
        target = next((p for p in acc.positions if p.position_id == position_id), None)
    if target is None:
        return None
    from pm.risk.payoff import structure_payoff
    return structure_payoff(state, acc, target, shock=shock)


# ---------------------------------------------------------------------------
# Scanner facade — the sanctioned scanner seams, delegating to pm.scanner.
# The implementations (pm/scanner/{slice,candidates,ticket,view}.py) take the
# loaded state explicitly; each wrapper here resolves the singleton ONCE and
# threads that one object through the whole call chain, so a scan computes
# against a single coherent state snapshot. Signatures and defaults mirror
# the implementations exactly. pull_slice remains the sanctioned owned-state
# WRITE path (it fetches live data into the state-attached slice_cache);
# scanner_candidate resolves through it on a cold cache, so only the warm
# path is a pure read. See each implementation for the full contract.
# ---------------------------------------------------------------------------

def pull_slice(account: str, position_id: str, *, refresh: bool = False,
               refresh_chain: bool = False, n_expiries: int = 3,
               moneyness_pct: float = 0.15, rights=("CALL", "PUT"),
               expiry_type: str = "monthly", dte_range=None) -> Optional[dict]:
    """The scanner's sanctioned on-demand slice pull (pm.scanner.slice)."""
    return _scan_slice.pull_slice(
        _RUNTIME.get("state"), account, position_id, refresh=refresh,
        refresh_chain=refresh_chain, n_expiries=n_expiries,
        moneyness_pct=moneyness_pct, rights=rights, expiry_type=expiry_type,
        dte_range=dte_range)


def chain_expiries(account: str, position_id: str,
                   expiry_type: str = "monthly") -> list:
    """Listed expiries of the selected type — the DTE control's bounds
    (pm.scanner.slice)."""
    return _scan_slice.chain_expiries(_RUNTIME.get("state"), account,
                                      position_id, expiry_type)


def scanner_candidate(account: str, position_id: str, objective: str, rank: int,
                      *, n_expiries: int = 3, structure_id=STRUCTURE_AUTO,
                      dte_range=None, delta_band=None, rolled_pids=None,
                      expiry_type: str = "monthly"):
    """The cached ranked candidate at (objective, rank) (pm.scanner.candidates)."""
    return _scan_candidates.scanner_candidate(
        _RUNTIME.get("state"), account, position_id, objective, rank,
        n_expiries=n_expiries, structure_id=structure_id, dte_range=dte_range,
        delta_band=delta_band, rolled_pids=rolled_pids, expiry_type=expiry_type)


def price_candidate(account: str, position_id: str, objective: str, rank: int, *,
                    shock=None, n_expiries: int = 3, structure_id=STRUCTURE_AUTO,
                    dte_range=None, delta_band=None, rolled_pids=None,
                    expiry_type: str = "monthly"):
    """Read-only payoff reprice of a ranked candidate (pm.scanner.candidates)."""
    return _scan_candidates.price_candidate(
        _RUNTIME.get("state"), account, position_id, objective, rank,
        shock=shock, n_expiries=n_expiries, structure_id=structure_id,
        dte_range=dte_range, delta_band=delta_band, rolled_pids=rolled_pids,
        expiry_type=expiry_type)


def generate_slice_candidates(account: str, position_id: str, *, objectives=None,
                              cap: int = 15, n_expiries: int = 3,
                              structure_id=STRUCTURE_AUTO, dte_range=None,
                              delta_band=None, expiry_type: str = "monthly"):
    """Generate + price the adjustment candidates (pm.scanner.candidates)."""
    return _scan_candidates.generate_slice_candidates(
        _RUNTIME.get("state"), account, position_id, objectives=objectives,
        cap=cap, n_expiries=n_expiries, structure_id=structure_id,
        dte_range=dte_range, delta_band=delta_band, expiry_type=expiry_type)


def rank_slice_candidates(account: str, position_id: str, *, objectives=None,
                          cap: int = 15, n_expiries: int = 3,
                          structure_id=STRUCTURE_AUTO, dte_range=None,
                          delta_band=None, expiry_type: str = "monthly"):
    """Generate + price + rank, grouped by objective (pm.scanner.candidates)."""
    return _scan_candidates.rank_slice_candidates(
        _RUNTIME.get("state"), account, position_id, objectives=objectives,
        cap=cap, n_expiries=n_expiries, structure_id=structure_id,
        dte_range=dte_range, delta_band=delta_band, expiry_type=expiry_type)


def generate_joint_candidates(account: str, position_id: str, rolled_pids, *,
                              objectives=None, cap: int = 15, n_expiries: int = 3,
                              structure_id=STRUCTURE_AUTO, delta_band=None,
                              dte_range=None, expiry_type: str = "monthly"):
    """Joint-roll candidates for a set of structure legs (pm.scanner.candidates)."""
    return _scan_candidates.generate_joint_candidates(
        _RUNTIME.get("state"), account, position_id, rolled_pids,
        objectives=objectives, cap=cap, n_expiries=n_expiries,
        structure_id=structure_id, delta_band=delta_band, dte_range=dte_range,
        expiry_type=expiry_type)


def rank_joint_candidates(account: str, position_id: str, rolled_pids, *,
                          objectives=None, cap: int = 15, n_expiries: int = 3,
                          structure_id=STRUCTURE_AUTO, delta_band=None,
                          dte_range=None, expiry_type: str = "monthly"):
    """Generate + rank the joint-roll candidates (pm.scanner.candidates)."""
    return _scan_candidates.rank_joint_candidates(
        _RUNTIME.get("state"), account, position_id, rolled_pids,
        objectives=objectives, cap=cap, n_expiries=n_expiries,
        structure_id=structure_id, delta_band=delta_band, dte_range=dte_range,
        expiry_type=expiry_type)


def build_adjustment_ticket(account: str, position_id: str, *, objective=None,
                            rank=None, structure_id=STRUCTURE_AUTO, dte_range=None,
                            delta_band=None, rolled_pids=None, capture_pids=None,
                            n_expiries: int = 3, expiry_type: str = "monthly"):
    """The adjustment ticket for the selected candidate and/or capture marks
    (pm.scanner.ticket) — a read-only compose; a proposal, never an order."""
    return _scan_ticket.build_adjustment_ticket(
        _RUNTIME.get("state"), account, position_id, objective=objective,
        rank=rank, structure_id=structure_id, dte_range=dte_range,
        delta_band=delta_band, rolled_pids=rolled_pids,
        capture_pids=capture_pids, n_expiries=n_expiries,
        expiry_type=expiry_type)


def scanner_roster(account: str, position_id: str, *, structure_id=STRUCTURE_AUTO):
    """The Managing band's roster rows + stored economics (pm.scanner.view)."""
    return _scan_view.scanner_roster(_RUNTIME.get("state"), account, position_id,
                                     structure_id=structure_id)


def scanner_view_data(account: str, position_id: str, *, objectives=None,
                      cap: int = 15, refresh: bool = False, n_expiries: int = 3,
                      structure_id=STRUCTURE_AUTO, dte_range=None, delta_band=None,
                      rolled_pids=None, expiry_type: str = "monthly") -> Optional[dict]:
    """Read-only packaging for the scanner drawer (pm.scanner.view)."""
    return _scan_view.scanner_view_data(
        _RUNTIME.get("state"), account, position_id, objectives=objectives,
        cap=cap, refresh=refresh, n_expiries=n_expiries,
        structure_id=structure_id, dte_range=dte_range, delta_band=delta_band,
        rolled_pids=rolled_pids, expiry_type=expiry_type)


def resolve_structure(
    account: str, structure_id: str, resolution: str,
    chosen_type: Optional[str] = None, edited_legs: Optional[list] = None,
) -> bool:
    """Confirm / reject / choose-alternative / edit a structure proposal. Writes the
    resolution through the structure store, re-applies it to the in-memory state's
    structures (flipping status), then re-derives that one structure's management fires
    so the now-eligible fires appear (or the no-longer-eligible ones disappear) without
    a reload.

    This stays within the no-recompute contract: it is a transactional state update in
    the single owner, reading only data already on the state (snapshot spot, holdings
    mark, the treasury curve / fallback rate) — no Bloomberg fetch, no signal recompute.
    It is idempotent — the affected structure's fires are removed by structure_id and
    re-derived each time, and the leg-context annotations rebuild from a clean base — so
    repeated confirm/reject produces no duplicate fires and no doubled annotations.
    Returns True on success."""
    from pm.insight.structure_fires import attach_structure_context, rederive_structure_fires
    from pm.store import structure_store
    state = _RUNTIME.get("state")
    if state is None:
        return False
    with _WRITE_LOCK:
        acc = state.accounts.get(account)
        if acc is None:
            return False
        target = next((s for s in acc.structures if s.structure_id == structure_id), None)
        if target is None:
            return False
        if chosen_type is None:
            # Record the reading the decision was made against — the apply pass
            # demotes to a fresh proposal if the same legs later re-detect as a
            # different type (a confirm of one reading never silently carries).
            chosen_type = target.type
        leg_pids = structure_store.decision_leg_pids(acc.structures, target)
        structure_store.save_resolution(
            account, leg_pids, resolution, chosen_type=chosen_type, edited_legs=edited_legs)
        structure_store.apply_resolutions(account, acc.structures)
        # Swap in the affected structure's fires by structure_id: drop its prior fires,
        # then append the freshly re-derived set. Unified across confirm and reject — a
        # reject re-derives too, so the structure's non-confirmation-gated fires survive
        # exactly as a full reload would produce them while the gated ones drop. A
        # contention choose flips the WHOLE group (winner confirmed, siblings rejected)
        # and the pin fire speaks through one group representative — so every member is
        # swapped, not just the target, else a sibling's stale fire lingers until the
        # next full load. Then rebuild leg-context annotations from each fire's clean
        # base (idempotent). Built off to the side and assigned once, so a concurrent
        # render never sees the list mid-rebuild.
        group = getattr(target, "contention_group", None)
        members = ([s for s in acc.structures
                    if getattr(s, "contention_group", None) == group]
                   if group else [target])
        member_ids = {s.structure_id for s in members}
        fires = [f for f in acc.fires if f.structure_id not in member_ids]
        for m in members:
            fires.extend(rederive_structure_fires(state, acc, m))
        acc.fires = fires
        attach_structure_context(acc)
        # Re-mark this account's fires so a just-confirmed fire that matches an
        # active suppression is muted without a reload — same marking logic as the load
        # path, reading only the persisted suppressions (no recompute).
        from pm.store import suppression_store
        suppression_store.remark_account(acc)
        # The scanner's cached candidates embed structure context (kept legs,
        # allocated slices) — a changed resolution invalidates them for this
        # account. Raw slices stay; the next scan regenerates Bloomberg-free
        # from the cached slice.
        _drop_candidate_caches(state, account)
        return True


# ---------------------------------------------------------------------------
# Alert suppression write path — the single shared accessor/restore.
# Both the modal's Muted footer (Part B) and the Alert Manager (Part C) call these;
# there is no second mechanism. Like resolve_structure, each is a transactional
# update in the single state owner: it writes the persisted suppression, then
# re-marks the affected account's fires in place (no reload, no recompute) so every
# surface reflects the change immediately.
# ---------------------------------------------------------------------------

def suppress_alert(account: str, name: str, pattern_id: str, *,
                   suppressed_until: Optional[str] = None,
                   trace: Optional[dict] = None,
                   rationale: Optional[str] = None) -> bool:
    """Suppress (``suppressed_until=None``) or snooze the alert ``(account, name,
    pattern_id)``. The captured baseline is anchored to the MOST-EXTREME matching
    instance on the name — the same instance the material-change comparison
    measures against — not the clicked row's, so muting the weaker of two
    same-name instances can never flip straight back to ``resurfaced`` in the
    same interaction (the passed ``trace``/``rationale`` remain the fallback when
    no live state or no comparable headline exists). Re-marks the account so the
    muted fire drops from the active surfaces at once. Returns True on success."""
    from pm.store import suppression_store
    with _WRITE_LOCK:
        state = _RUNTIME.get("state")
        acc = state.accounts.get(account) if state is not None else None
        if acc is not None:
            matching = [f for f in acc.fires
                        if f.underlying == name and f.pattern_id == pattern_id]
            baseline = suppression_store.pick_baseline_fire(matching, pattern_id)
            if baseline is not None:
                trace, rationale = baseline.trace, baseline.rationale
        suppression_store.suppress(account, name, pattern_id,
                                   suppressed_until=suppressed_until,
                                   trace=trace, rationale=rationale)
        if acc is None:
            return False
        suppression_store.remark_account(acc)
        return True


def restore_alert(account: str, name: str, pattern_id: str,
                  fire_key: Optional[str] = None) -> bool:
    """Remove the suppression — and, when ``fire_key`` is given, any per-fire
    acknowledgement under the same key — then re-mark the account so the alert
    returns to the active surfaces without a reload. Returns True on success."""
    from pm.store import alert_governance, suppression_store
    with _WRITE_LOCK:
        suppression_store.restore(account, name, pattern_id)
        if fire_key:
            alert_governance.unacknowledge(account, pattern_id, fire_key)
        state = _RUNTIME.get("state")
        if state is None:
            return False
        acc = state.accounts.get(account)
        if acc is None:
            return False
        suppression_store.remark_account(acc)
        return True


def acknowledge_alert(account: str, position_id: str, pattern_id: str) -> bool:
    """Acknowledge ONE fire — the per-fire quiet path (distinct from muting the
    whole pattern on the name). Persists the ack keyed by the fire's
    structure_id-or-position_id with the fire's trace as the material-change
    baseline, then re-marks the account in place. A sanctioned write path under
    the write lock; no reload, no recompute. Returns True on success."""
    from pm.store import alert_governance, suppression_store
    with _WRITE_LOCK:
        state = _RUNTIME.get("state")
        if state is None:
            return False
        acc = state.accounts.get(account)
        if acc is None:
            return False
        fire = next((f for f in acc.fires
                     if f.position_id == position_id and f.pattern_id == pattern_id), None)
        if fire is None:
            return False
        alert_governance.acknowledge(
            account, pattern_id, alert_governance.fire_key(fire),
            trace=fire.trace, rationale=fire.rationale)
        suppression_store.remark_account(acc)
        return True


def unacknowledge_alert(account: str, pattern_id: str, fire_key: str) -> bool:
    """Remove one acknowledgement and re-mark the account (the Alert Manager's
    per-ack Restore). Returns True on success."""
    from pm.store import alert_governance, suppression_store
    with _WRITE_LOCK:
        alert_governance.unacknowledge(account, pattern_id, fire_key)
        state = _RUNTIME.get("state")
        if state is None:
            return False
        acc = state.accounts.get(account)
        if acc is None:
            return False
        suppression_store.remark_account(acc)
        return True


def set_pattern_enabled(pattern_id: str, enabled: bool) -> bool:
    """Flip one pattern's persisted on/off toggle and re-mark EVERY account in
    place — collapse-to-muted, never drop: a toggled-off pattern's fires stay on
    ``acc.fires`` marked ``disabled`` (counted and recoverable), and toggling
    back on restores them with no reload and no recompute. A sanctioned write
    path under the write lock. Returns True on success."""
    from pm.store import alert_governance, suppression_store
    with _WRITE_LOCK:
        alert_governance.set_pattern_enabled(pattern_id, enabled)
        state = _RUNTIME.get("state")
        if state is None:
            return False
        for acc in state.accounts.values():
            suppression_store.remark_account(acc)
        return True


# ---------------------------------------------------------------------------
# Fire / signal / position lookups
# ---------------------------------------------------------------------------

def all_fires(state: PortfolioState) -> list[Fire]:
    """Flat list of every fire across all accounts."""
    out: list[Fire] = []
    for acc in state.accounts.values():
        out.extend(acc.fires)
    return out


def fires_for_account(state: PortfolioState, account: str) -> list[Fire]:
    acc = state.accounts.get(account)
    return list(acc.fires) if acc else []


def fires_for_underlying(state: PortfolioState, account: str, underlying: str) -> list[Fire]:
    """All fires on a given underlying within one account (for the signal sheet)."""
    acc = state.accounts.get(account)
    if acc is None:
        return []
    return [f for f in acc.fires if f.underlying == underlying]


def fires_for_position(state: PortfolioState, account: str, position_id: str) -> list[Fire]:
    """All fires (alerts) on one position, most-severe first — for the modal's
    Alert view, which stacks every alert on a consolidated position row."""
    acc = state.accounts.get(account)
    if acc is None:
        return []
    fires = [f for f in acc.fires if f.position_id == position_id]
    return sorted(fires, key=lambda f: f.tier)


def signals_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> Optional[SignalDict]:
    acc = state.accounts.get(account)
    if acc is None:
        return None
    return acc.signals.get(underlying)


def position_signals_for(
    state: PortfolioState, account: str, position_id: str,
) -> Optional[SignalDict]:
    """The merged per-position SignalDict (carries Group E), or None."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    return acc.position_signals.get(position_id)


def fire_by_id(
    state: PortfolioState, account: str, position_id: str, pattern_id: str,
) -> Optional[Fire]:
    """Locate a single fire for drawer rendering."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    for f in acc.fires:
        if f.position_id == position_id and f.pattern_id == pattern_id:
            return f
    return None


def positions_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> list[Position]:
    """Held positions whose (underlying_symbol or symbol) == underlying."""
    acc = state.accounts.get(account)
    if acc is None:
        return []
    return [p for p in acc.positions
            if (p.underlying_symbol or p.symbol) == underlying]


# ---------------------------------------------------------------------------
# Snapshot access (for the signal-sheet header)
# ---------------------------------------------------------------------------

def bbg_ticker_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> Optional[str]:
    """First BBG ticker we find for this bare-symbol underlying in the account."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    for p in acc.positions:
        if p.asset_class in ("equity", "fund_etf") and p.symbol == underlying:
            return p.bbg_ticker or None
        if p.asset_class == "option" and p.underlying_symbol == underlying:
            return p.underlying_bbg_ticker or None
    return None


def snapshot_row_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> Optional[dict]:
    """The snapshot row (dict of BBG fields) for an underlying, or None.
    Read-only — pulls the row already fetched onto AccountState.snapshot."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    bbg = bbg_ticker_for_underlying(state, account, underlying)
    if not bbg:
        return None
    df = acc.snapshot.underlyings
    if df is None or df.empty or bbg not in df.index:
        return None
    series = df.loc[bbg]
    return {col: series[col] for col in df.columns}
