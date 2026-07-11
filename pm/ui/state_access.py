"""Read/own the UI layer's runtime state.

The UI never recomputes — it reads what ``run_insight_engine`` already
produced and attached to ``PortfolioState``. The singleton PortfolioState is
OWNED here (``_RUNTIME``), because this module is only ever imported as
``pm.ui.state_access`` — never executed as ``__main__``. ``pm/app.py`` is the
entry point and, under ``python -m pm.app``, runs as ``__main__``; a global
stored there would be a *different* object from the ``pm.app`` callbacks
import, so the state would be invisible to them. Owning it here gives one
canonical instance for both the entry point and every callback.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from pm.ingest.position_builder import Position
from pm.insight.patterns import Fire
from pm.insight.signal_library import SignalDict, SignalValue
from pm.store.portfolio_state import AccountState, PortfolioState


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
    False, read the latest extract in the data dir ("Refresh Acct Data" / first load)."""
    from pm.config import EXTRACT_DATA_DIR
    from pm.store.portfolio_state import refresh_portfolio_state
    prev = _RUNTIME.get("state")
    new_state = refresh_portfolio_state(prev, EXTRACT_DATA_DIR, reuse_extract=reuse_extract)
    _RUNTIME["state"] = new_state
    return new_state


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
    grid = spot_vol_grid(state, acc, rate_bps=rate_bps, time_days=int(time_days), target=target)
    return {
        "account": {"pnl": impact["account_pnl"], "pnl_pct": impact["account_pnl_pct"],
                    "axes": shock.axes(), "mode": mode, "target": target},
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


def scanner_candidate(account: str, position_id: str, objective: str, rank: int,
                      *, n_expiries: int = 3):
    """The cached ranked scanner candidate at ``(objective, rank)`` for a held position,
    or None. A pure read of the slice the scanner already pulled + ranked.

    ``n_expiries`` must be the scanner's ACTIVE window (the drawer threads its
    window store through): the slice cache is keyed by window, so reading the
    default while the table shows an expanded window would resolve (objective,
    rank) against a different, stale ranking than the row the user clicked.
    Held-stock (overlay) positions have no option slice — their ranking is read
    from the per-position overlay cache instead."""
    state = _RUNTIME.get("state")
    if state is None:
        return None
    pos = position_by_id(state, account, position_id)
    if pos is None:
        return None
    if pos.asset_class == "option":
        sl = pull_slice(account, position_id, n_expiries=n_expiries)  # cache hit after the scan
        ranked_map = sl.get("candidates_ranked") if sl else None
    else:
        ranked_map = state.slice_cache.get("overlay_ranked", {}).get(position_id)
    ranked = (ranked_map or {}).get(objective) or []
    return next((r for r in ranked if getattr(r, "rank", None) == rank), None)


def price_candidate(account: str, position_id: str, objective: str, rank: int, *,
                    shock=None, n_expiries: int = 3):
    """The candidate side of the current-vs-candidate comparison — a read-only payoff
    reprice of a ranked candidate's resulting position under a shock. Prices the
    candidate's engine legs through the pure ``compute_payoff`` (the same engine the
    structure payoff wraps) at the SLICE's spot (option rolls) or the snapshot spot
    (stock overlays), so the candidate curve shares the current position's grid.
    ``n_expiries`` is the scanner's active window (see ``scanner_candidate``).
    No Bloomberg beyond the already-cached slice, no reload, no ``_RUNTIME`` write,
    no engine change. Returns the ``compute_payoff`` dict or None."""
    rc = scanner_candidate(account, position_id, objective, rank, n_expiries=n_expiries)
    if rc is None:
        return None
    state = _RUNTIME.get("state")
    acc = state.accounts.get(account) if state else None
    pos = position_by_id(state, account, position_id) if state else None
    if acc is None or pos is None:
        return None
    if pos.asset_class == "option":
        sl = pull_slice(account, position_id, n_expiries=n_expiries)
        spot = sl.get("spot") if sl else None
    else:
        spot = _spot_from_snapshot(acc, pos.bbg_ticker)
    cand = getattr(rc, "candidate", None)
    legs = getattr(cand, "legs", None)
    if spot is None or not (spot > 0) or not legs:
        return None
    from datetime import date
    from pm.candidates.generate import _build_tier1
    from pm.risk.payoff import compute_payoff
    try:
        tier1 = _build_tier1(legs, date.today())
        return compute_payoff(legs, float(spot), tier1, shock=shock)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("price_candidate failed for %s", position_id)
        return None


# ---------------------------------------------------------------------------
# On-demand option-chain slice pull (the scanner's data layer).
# A SANCTIONED owned-state WRITE path — like resolve_structure / suppress_alert,
# and categorically UNLIKE the read-only price_scenario / price_payoff: it fetches
# live data and writes it into the state-attached slice_cache. The cache is fresh
# each load, so a reload drops every slice (no stale marks survive a Refresh).
# ---------------------------------------------------------------------------

def _spot_from_snapshot(acc: AccountState, underlier: str) -> Optional[float]:
    """Underlier spot from the morning snapshot (no re-pull)."""
    df = getattr(getattr(acc, "snapshot", None), "underlyings", None)
    if df is None or getattr(df, "empty", True) or underlier not in df.index:
        return None
    return coerce_float(df.loc[underlier, "PX_LAST"] if "PX_LAST" in df.columns else None)


def pull_slice(
    account: str, position_id: str, *, refresh: bool = False,
    refresh_chain: bool = False, n_expiries: int = 3, moneyness_pct: float = 0.15,
    rights=("CALL", "PUT"), monthlies_only: bool = True,
) -> Optional[dict]:
    """Pull the targeted option-chain slice for a held position and cache it on the
    loaded state. A SANCTIONED owned-state WRITE path (parallel to ``resolve_structure``;
    it deliberately writes fetched data into owned state, unlike the read-only
    ``price_scenario`` / ``price_payoff``, which must never mutate it).

    Enumerates the underlier's listed chain **once per underlier** (cached), filters to
    the window around spot and the held strike (``ticker_utils.filter_chain_slice``), and
    snapshots the survivors. The held leg's own greeks/IV come from the morning snapshot
    and are never re-pulled here — this fetches candidate contracts only.

    Returns ``{key, underlier, candidates, df, spot, pulled_at}`` or ``None`` (no state /
    position / spot, or Bloomberg off). ``refresh`` re-snapshots with fresh greeks/IV
    (reusing the cached chain); ``refresh_chain`` additionally re-enumerates the chain.
    Re-opening the same window without ``refresh`` is a cache hit — no Bloomberg call."""
    state = _RUNTIME.get("state")
    if state is None or not getattr(state, "bloomberg_ok", False):
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    pos = next((p for p in acc.positions
                if p.position_id == position_id and p.asset_class == "option"), None)
    if pos is None or not pos.underlying_bbg_ticker or pos.strike is None:
        return None
    underlier = pos.underlying_bbg_ticker

    spot = _spot_from_snapshot(acc, underlier)
    if spot is None:
        return None

    cache = state.slice_cache
    chains = cache.setdefault("chains", {})
    slices = cache.setdefault("slices", {})

    key = (underlier, round(float(pos.strike), 4), pos.expiry, int(n_expiries),
           round(float(moneyness_pct), 4), bool(monthlies_only),
           tuple(sorted(str(r).upper() for r in rights)))
    if not refresh and key in slices:
        return slices[key]

    from pm.core.ticker_utils import filter_chain_slice, parse_option_description

    chain_entry = chains.get(underlier)
    if chain_entry is None or refresh_chain:
        from pm.core.bloomberg_client import fetch_option_chain
        parsed = [d for d in (parse_option_description(s) for s in fetch_option_chain(underlier)) if d]
        chain_entry = {"chain": parsed, "pulled_at": datetime.now()}
        chains[underlier] = chain_entry

    candidates = filter_chain_slice(
        chain_entry["chain"], spot, pos.strike, horizon_expiry=pos.expiry,
        n_expiries=n_expiries, moneyness_pct=moneyness_pct, rights=rights,
        monthlies_only=monthlies_only,
    )

    from pm.core.bloomberg_client import fetch_option_snapshots
    df = fetch_option_snapshots(candidates) if candidates else None

    # Surface + IV+pp (a read-only derivation of the cached slice) and IV-rank (a
    # name-level metric cached beside the chain). Best-effort — a fit failure must not
    # break the slice pull.
    surface, iv_pp = _slice_surface(acc, underlier, spot, df)
    result = {"key": key, "underlier": underlier, "candidates": candidates,
              "df": df, "spot": spot, "pulled_at": datetime.now(),
              "surface": surface, "iv_pp": iv_pp,
              "iv_rank": _slice_iv_rank(state, acc, underlier)}
    slices[key] = result
    return result


def _snapshot_underlying_row(acc: AccountState, underlier: str):
    df = getattr(getattr(acc, "snapshot", None), "underlyings", None)
    if df is None or getattr(df, "empty", True) or underlier not in df.index:
        return None
    return df.loc[underlier]


def _as_date(v):
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v, errors="coerce")
    except Exception:
        return None
    return None if pd.isna(ts) else ts.date()


def _slice_surface(acc: AccountState, underlier: str, spot: float, df):
    """Fit the surface + IV+pp over the pulled slice. Returns (SurfaceFit|None,
    iv_pp rows|None); never raises."""
    if df is None or getattr(df, "empty", True):
        return None, None
    try:
        from pm.candidates.surface import build_slice_surface
        row = _snapshot_underlying_row(acc, underlier)
        earnings = _as_date(row.get("EXPECTED_REPORT_DT")) if row is not None else None
        built = build_slice_surface(df, spot, earnings_date=earnings)
        rows = [{"ticker": c.ticker, "strike": c.strike, "expiry": c.expiry,
                 "right": c.right, "iv": c.iv, "iv_fitted": c.iv_fitted,
                 "iv_excess": c.iv_excess, "in_fit": c.in_fit, "iv_source": c.iv_source}
                for c in built["contracts"]]
        return built["surface"], rows
    except Exception:
        import logging
        logging.getLogger(__name__).exception("slice surface fit failed for %s", underlier)
        return None, None


def _slice_iv_rank(state: PortfolioState, acc: AccountState, underlier: str):
    """Trailing 52-week IV-rank of the name's 3M ATM IV, cached per underlier (name-
    level, one BBG history pull per name per load). Returns a dict or None."""
    cache = state.slice_cache.setdefault("iv_rank", {})
    if underlier in cache:
        return cache[underlier]
    val = None
    try:
        row = _snapshot_underlying_row(acc, underlier)
        current = coerce_float(row.get("3MTH_IMPVOL_100.0%MNY_DF")) if row is not None else None
        if current is not None:
            from pm.candidates.surface import iv_rank
            from pm.core.bloomberg_client import fetch_iv_history
            hist = fetch_iv_history([underlier], lookback_days=400).get(underlier)
            n = int(hist.notna().sum()) if hist is not None else 0
            val = {"current_3m_atm": current,
                   "percentile": iv_rank(current, hist) if hist is not None else None,
                   "n_obs": n}
    except Exception:
        import logging
        logging.getLogger(__name__).exception("iv-rank failed for %s", underlier)
    cache[underlier] = val
    return val


# ---------------------------------------------------------------------------
# Candidate generation + per-candidate economics (the scanner's pricing step).
# A sanctioned owned-state derivation: reads the cached slice, prices each candidate
# through the validated payoff engine (one compute_payoff call per candidate), and
# attaches the result to the slice entry. No new pricing math.
# ---------------------------------------------------------------------------

def _per_share_basis(pos) -> Optional[float]:
    cb = coerce_float(getattr(pos, "cost_basis", None))
    qty = coerce_float(getattr(pos, "quantity", None))
    return cb / qty if (cb is not None and qty) else None


def _held_option_delta(acc: AccountState, pos) -> Optional[float]:
    opts = getattr(getattr(acc, "snapshot", None), "options", None)
    if opts is not None and not getattr(opts, "empty", True) and pos.bbg_ticker in opts.index:
        return coerce_float(opts.loc[pos.bbg_ticker].get("delta_mid"))
    return None


def _held_stock(acc: AccountState, opt_pos):
    """(shares, cost_basis_per_share) for a long stock position on the option's
    underlier, else None — so a covered roll prices as covered, a naked one as naked."""
    for p in acc.positions:
        if p.asset_class in ("equity", "fund_etf") and (p.quantity or 0) > 0 \
                and p.symbol == opt_pos.underlying_symbol:
            basis = _per_share_basis(p)
            if basis is not None:
                return (int(p.quantity), basis)
    return None


def _contemporaneous_mid(pos, sl) -> Optional[float]:
    """The held option's current mid for the net-credit arithmetic: from the slice if
    present, else an explicit fresh pull (the held leg can lie outside the slice
    window). The held leg's risk/greeks DISPLAY still uses the morning snapshot."""
    df = sl.get("df")
    tk = pos.bbg_ticker
    if df is not None and not getattr(df, "empty", True) and tk in df.index:
        m = coerce_float(df.loc[tk].get("PX_MID"))
        if m is not None:
            return m
    try:
        from pm.core.bloomberg_client import fetch_option_snapshots
        one = fetch_option_snapshots([tk])
        return coerce_float(one.loc[tk].get("PX_MID")) if tk in one.index else None
    except Exception:
        return None


def _spot_slice_df(state: PortfolioState, underlier: str, spot: float):
    """A spot-centered near-dated slice frame for a stock overlay (there is no held
    strike/expiry to anchor on). Reuses the per-underlier chain cache."""
    from datetime import date, timedelta
    from pm.core.ticker_utils import filter_chain_slice, parse_option_description
    from pm.core.bloomberg_client import fetch_option_snapshots
    chains = state.slice_cache.setdefault("chains", {})
    entry = chains.get(underlier)
    if entry is None:
        from pm.core.bloomberg_client import fetch_option_chain
        parsed = [d for d in (parse_option_description(s) for s in fetch_option_chain(underlier)) if d]
        entry = {"chain": parsed, "pulled_at": datetime.now()}
        chains[underlier] = entry
    today = date.today()
    tickers = filter_chain_slice(entry["chain"], spot, spot,
                                 horizon_expiry=today + timedelta(days=90),
                                 n_expiries=3, moneyness_pct=0.15)
    return fetch_option_snapshots(tickers) if tickers else None


def generate_slice_candidates(account: str, position_id: str, *, objectives=None, cap: int = 15,
                              n_expiries: int = 3):
    """Generate + price the adjustment candidates for a held position — rolls for a held
    option, single-leg overlays for held stock. Attaches the priced candidates to the
    slice cache entry and returns them (or None)."""
    state = _RUNTIME.get("state")
    if state is None or not getattr(state, "bloomberg_ok", False):
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    pos = next((p for p in acc.positions if p.position_id == position_id), None)
    if pos is None:
        return None

    curve = getattr(state, "risk_free_curve", None) or []
    rfr = getattr(state, "risk_free_rate", 0.045)

    from pm.candidates.generate import candidates_from_slice, overlays_from_slice
    cands = []
    try:
        if pos.asset_class == "option":
            sl = pull_slice(account, position_id, n_expiries=n_expiries)
            if sl is None:
                return None
            q = _div_yield(acc, sl["underlier"])
            held = {"strike": pos.strike, "expiry": pos.expiry, "right": pos.right,
                    "quantity": pos.quantity, "delta": _held_option_delta(acc, pos)}
            cands = candidates_from_slice(
                sl["df"], held, _contemporaneous_mid(pos, sl), sl["spot"],
                held_stock=_held_stock(acc, pos), risk_free_curve=curve,
                risk_free_rate=rfr, div_yield=q, objectives=objectives, cap=cap)
            sl["candidates_priced"] = cands
        elif pos.asset_class in ("equity", "fund_etf"):
            spot = _spot_from_snapshot(acc, pos.bbg_ticker)
            basis = _per_share_basis(pos)
            if spot is None or basis is None or not pos.quantity:
                return []
            df = _spot_slice_df(state, pos.bbg_ticker, spot)
            # Retain the overlay slice so the scanner chain table can read each
            # candidate's contract liquidity (the option-roll path keeps its slice; the
            # overlay path builds a transient frame, so stash it under the position).
            state.slice_cache.setdefault("overlay_dfs", {})[position_id] = df
            q = _div_yield(acc, pos.bbg_ticker)
            cands = overlays_from_slice(df, spot, int(pos.quantity), basis,
                                        risk_free_curve=curve, risk_free_rate=rfr,
                                        div_yield=q, cap=cap)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("candidate generation failed for %s", position_id)
    return cands


def rank_slice_candidates(account: str, position_id: str, *, objectives=None, cap: int = 15,
                          n_expiries: int = 3):
    """Generate + price + rank the adjustment candidates for a held position, grouped
    by objective. A pure, read-only derivation over already-loaded state: it reads the
    account's client profile and the slice's IV+pp rows, ranks each objective's
    candidates through ``pm.candidates.ranking`` (no Bloomberg, no state write beyond
    caching the result on the option slice for reuse), and returns
    ``{objective: [RankedCandidate, ...]}`` (or None). Render is M5's job."""
    from datetime import date

    state = _RUNTIME.get("state")
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    pos = next((p for p in acc.positions if p.position_id == position_id), None)
    if pos is None:
        return None
    # Free pill switches: an option scan's full ranking (all objectives) is cached on the
    # slice at pull time; an overlay scan's ranking is cached per position (a held-stock
    # scan has no option slice). Either way a pill change reads the cache rather than
    # re-generating + re-pricing — and the comparison tab resolves clicked rows from the
    # SAME cache, so the overlay path is no longer compare-dead.
    if pos.asset_class == "option":
        cached_sl = pull_slice(account, position_id, n_expiries=n_expiries)
        if cached_sl and cached_sl.get("candidates_ranked"):
            return cached_sl["candidates_ranked"]
    else:
        cached = state.slice_cache.get("overlay_ranked", {}).get(position_id)
        if cached:
            return cached

    cands = generate_slice_candidates(account, position_id, objectives=objectives, cap=cap,
                                      n_expiries=n_expiries)
    if not cands:
        return None

    profile = getattr(acc, "client_profile", None)

    # IV+pp rows + the held leg's Δ / DTE are available only on the option roll path
    # (a stock overlay has no held option leg and no anchored slice); the ranker
    # degrades cleanly when they are absent.
    iv_pp = None
    held = None
    sl = None
    if pos.asset_class == "option":
        sl = pull_slice(account, position_id, n_expiries=n_expiries)
        iv_pp = sl.get("iv_pp") if sl else None
        held = {"delta": _held_option_delta(acc, pos),
                "dte": (pos.expiry - date.today()).days if pos.expiry else None,
                "strike": pos.strike}

    from pm.candidates.ranking import rank_candidates
    by_objective: dict = {}
    for c in cands:
        by_objective.setdefault(c.objective, []).append(c)
    ranked = {obj: rank_candidates(cs, objective=obj, client_profile=profile,
                                   iv_pp=iv_pp, held=held)
              for obj, cs in by_objective.items()}

    if sl is not None:
        sl["candidates_ranked"] = ranked
    elif pos.asset_class in ("equity", "fund_etf"):
        # The overlay analog of the slice-attached ranking cache: keyed per
        # position beside overlay_dfs, read by scanner_candidate / the pill
        # cache-hit above. Dropped by a scanner Refresh (scanner_view_data).
        state.slice_cache.setdefault("overlay_ranked", {})[position_id] = ranked
    return ranked


def scanner_view_data(account: str, position_id: str, *, objectives=None, cap: int = 15,
                      refresh: bool = False, n_expiries: int = 3) -> Optional[dict]:
    """Read-only packaging for the scanner drawer: the ranked adjustment candidates for a
    held position, plus the slice metadata the view stamps. Delegates to
    ``rank_slice_candidates`` (the sanctioned on-demand pull + rank); ``refresh`` re-pulls
    the option slice first. Returns ``{ranked, pulled_at, spot, underlier, kind}`` or
    ``None`` (no state / position, Bloomberg off, or nothing to scan). The pull is the
    only market I/O — the view itself never recomputes."""
    state = _RUNTIME.get("state")
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    pos = position_by_id(state, account, position_id)
    if pos is None:
        return None

    # A refresh re-snapshots the option slice (the sanctioned write path); on the
    # overlay path it drops the position's cached df + ranking so the scan genuinely
    # re-generates instead of cache-hitting.
    if refresh and pos.asset_class == "option":
        pull_slice(account, position_id, refresh=True, n_expiries=n_expiries)
    elif refresh:
        state.slice_cache.get("overlay_dfs", {}).pop(position_id, None)
        state.slice_cache.get("overlay_ranked", {}).pop(position_id, None)

    ranked = rank_slice_candidates(account, position_id, objectives=objectives, cap=cap,
                                   n_expiries=n_expiries)
    if ranked is None:
        return None

    pulled_at = spot = underlier = None
    df = surface = iv_pp = None
    if pos.asset_class == "option":
        sl = pull_slice(account, position_id, n_expiries=n_expiries)   # cache hit after the rank
        if sl:
            pulled_at, spot, underlier, df = (sl.get("pulled_at"), sl.get("spot"),
                                              sl.get("underlier"), sl.get("df"))
            surface, iv_pp = sl.get("surface"), sl.get("iv_pp")
    else:
        underlier = pos.bbg_ticker
        spot = _spot_from_snapshot(acc, pos.bbg_ticker)
        df = state.slice_cache.get("overlay_dfs", {}).get(position_id)

    metrics = _contract_metrics(df)
    return {"ranked": ranked, "pulled_at": pulled_at, "spot": spot,
            "underlier": underlier, "kind": pos.asset_class,
            "contract_metrics": metrics, "surface": surface, "iv_pp": iv_pp,
            "contracts": _slice_contracts(df, iv_pp, metrics),
            "held_strike": getattr(pos, "strike", None)}


def _slice_contracts(df, iv_pp, metrics) -> list:
    """The full cached slice as browse rows — every snapshotted contract with its strike,
    expiry, right, liquidity, IV and (option path) fitted IV / IV+pp. The option-roll path
    joins the IV+pp rows; the overlay path parses the ticker (no fitted surface)."""
    out: list = []
    if iv_pp:
        for r in iv_pp:
            m = metrics.get(r.get("ticker"), {})
            out.append({"ticker": r.get("ticker"), "strike": r.get("strike"),
                        "expiry": r.get("expiry"), "right": r.get("right"),
                        "iv": r.get("iv"), "iv_fitted": r.get("iv_fitted"),
                        "iv_excess": r.get("iv_excess"), "in_fit": r.get("in_fit"),
                        "bid": m.get("bid"), "ask": m.get("ask"), "mid": m.get("mid"),
                        "delta": m.get("delta"), "oi": m.get("oi")})
    elif df is not None and not getattr(df, "empty", True):
        from pm.core.ticker_utils import parse_option_description
        for tk, m in metrics.items():
            p = parse_option_description(str(tk)) or {}
            out.append({"ticker": tk, "strike": p.get("strike"), "expiry": p.get("expiry"),
                        "right": p.get("right"), "iv": m.get("iv"), "iv_fitted": None,
                        "iv_excess": None, "in_fit": False, "bid": m.get("bid"),
                        "ask": m.get("ask"), "mid": m.get("mid"), "delta": m.get("delta"),
                        "oi": m.get("oi")})
    return out


def structure_for_position(state, account: str, position_id: str):
    """The non-rejected structure whose legs include this position (confirmed preferred
    over proposed), or None. Lets a popup opened on an alert/holding price the enclosing
    covered structure in its Payoff/Scanner tabs, falling back to the standalone leg."""
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    hits = [s for s in (getattr(acc, "structures", None) or [])
            if getattr(s, "status", None) != "rejected"
            and any(getattr(lg, "position_id", None) == position_id for lg in (s.legs or []))]
    if not hits:
        return None
    hits.sort(key=lambda s: 0 if getattr(s, "status", None) == "confirmed" else 1)
    return hits[0].structure_id


def _contract_metrics(df) -> dict:
    """Per-ticker liquidity/greeks for the chain table, keyed by option ticker (the
    slice df index): bid/ask/mid/iv/delta/oi/volume. Empty when there is no slice."""
    out: dict = {}
    if df is None or getattr(df, "empty", True):
        return out
    for tk, row in df.iterrows():
        out[str(tk)] = {
            "bid": coerce_float(row.get("BID")), "ask": coerce_float(row.get("ASK")),
            "mid": coerce_float(row.get("PX_MID")), "iv": coerce_float(row.get("iv_mid")),
            "delta": coerce_float(row.get("delta_mid")), "oi": coerce_float(row.get("oi")),
            "volume": coerce_float(row.get("volume")),
        }
    return out


def _div_yield(acc: AccountState, underlier: str) -> float:
    row = _snapshot_underlying_row(acc, underlier)
    if row is None:
        return 0.0
    y = coerce_float(row.get("EQY_DVD_YLD_IND"))
    return (y / 100.0) if y is not None else 0.0


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
    acc = state.accounts.get(account)
    if acc is None:
        return False
    target = next((s for s in acc.structures if s.structure_id == structure_id), None)
    if target is None:
        return False
    if chosen_type is None and resolution == structure_store.CONFIRMED:
        chosen_type = target.type  # the confirmed/chosen reading's type
    leg_pids = structure_store.decision_leg_pids(acc.structures, target)
    structure_store.save_resolution(
        account, leg_pids, resolution, chosen_type=chosen_type, edited_legs=edited_legs)
    structure_store.apply_resolutions(account, acc.structures)
    # Swap in the affected structure's fires by structure_id: drop its prior fires,
    # then append the freshly re-derived set. Unified across confirm and reject — a
    # reject re-derives too, so the structure's non-confirmation-gated fires survive
    # exactly as a full reload would produce them while the gated ones drop. Then
    # rebuild leg-context annotations from each fire's clean base (idempotent).
    acc.fires = [f for f in acc.fires if f.structure_id != structure_id]
    acc.fires.extend(rederive_structure_fires(state, acc, target))
    attach_structure_context(acc)
    # Item 9: re-mark this account's fires so a just-confirmed fire that matches an
    # active suppression is muted without a reload — same marking logic as the load
    # path, reading only the persisted suppressions (no recompute).
    from pm.store import suppression_store
    suppression_store.remark_account(acc)
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
    pattern_id)``; ``trace``/``rationale`` capture the acting fire's baseline (store
    only). Re-marks the account so the muted fire drops from the active surfaces at
    once. Returns True on success."""
    from pm.store import suppression_store
    suppression_store.suppress(account, name, pattern_id,
                               suppressed_until=suppressed_until,
                               trace=trace, rationale=rationale)
    state = _RUNTIME.get("state")
    if state is None:
        return False
    acc = state.accounts.get(account)
    if acc is None:
        return False
    suppression_store.remark_account(acc)
    return True


def restore_alert(account: str, name: str, pattern_id: str) -> bool:
    """Remove the suppression and re-mark the account so the alert returns to the
    active surfaces without a reload. Returns True on success."""
    from pm.store import suppression_store
    suppression_store.restore(account, name, pattern_id)
    state = _RUNTIME.get("state")
    if state is None:
        return False
    acc = state.accounts.get(account)
    if acc is None:
        return False
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


def position_by_id(
    state: PortfolioState, account: str, position_id: str,
) -> Optional[Position]:
    acc = state.accounts.get(account)
    if acc is None:
        return None
    for p in acc.positions:
        if p.position_id == position_id:
            return p
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


# ---------------------------------------------------------------------------
# Small shared coercion helpers (display layer)
# ---------------------------------------------------------------------------

def is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def coerce_float(v: Any) -> Optional[float]:
    if is_missing(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
