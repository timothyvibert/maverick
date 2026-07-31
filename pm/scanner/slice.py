"""The scanner's data layer: on-demand option-chain slices and their caches.

Owns the chain / slice / per-expiry-frame / iv-rank caches on
``PortfolioState.slice_cache`` (fresh each load, so a reload drops every
cached slice by construction) and the pure derivations that ride the pull:
the fitted vol surface + IV+pp, the name-level IV-rank, and the per-contract
liquidity metrics. Every function takes the loaded state (or account)
explicitly — the runtime singleton lives in ``pm.ui.state_access``, which
delegates here. Bloomberg fetchers are imported at CALL time inside each
function; that call-time binding is load-bearing (tests patch the fetchers on
``pm.core.bloomberg_client``) and must not be hoisted to module level.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from pm.store.portfolio_state import AccountState, PortfolioState
from pm.store.state_reads import coerce_float


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


def chain_expiries(state, account: str, position_id: str,
                   expiry_type: str = "monthly") -> list:
    """The underlier's LISTED expiries of the selected type (sorted dates) from the
    cached chain — the DTE slider's bounds derive from these, so the control can
    never ask for an expiry that doesn't exist, and under 'weekly'/'all' it can
    represent near-dated windows a monthly-only ladder cannot. Enumerates the chain
    on first touch (the same cached one-per-underlier pull the scan uses). Empty
    list when unavailable."""
    if state is None or not getattr(state, "bloomberg_ok", False):
        return []
    acc = state.accounts.get(account)
    pos = next((p for p in (acc.positions if acc else [])
                if p.position_id == position_id), None)
    underlier = getattr(pos, "underlying_bbg_ticker", None) or getattr(pos, "bbg_ticker", None)
    if not underlier:
        return []
    from pm.core.ticker_utils import expiry_type_admits, parse_option_description
    chains = state.slice_cache.setdefault("chains", {})
    entry = chains.get(underlier)
    if entry is None:
        from pm.core.bloomberg_client import fetch_option_chain
        parsed = [d for d in (parse_option_description(s)
                              for s in fetch_option_chain(underlier)) if d]
        entry = {"chain": parsed, "pulled_at": datetime.now()}
        chains[underlier] = entry
    today = date.today()
    return sorted({p["expiry"] for p in entry["chain"]
                   if p.get("expiry") and p["expiry"] > today
                   and expiry_type_admits(p["expiry"], expiry_type)})


def _resolve_expiry_range(dte_range) -> Optional[tuple]:
    """(first, last) calendar window for a (lo, hi) DTE range — the chain filter
    then intersects it with the LISTED expiries (client-side; chain overrides are
    unavailable on this terminal)."""
    if dte_range is None:
        return None
    lo, hi = dte_range
    today = date.today()
    return (today + timedelta(days=int(lo)), today + timedelta(days=int(hi)))


def pull_slice(
    state, account: str, position_id: str, *, refresh: bool = False,
    refresh_chain: bool = False, n_expiries: int = 3, moneyness_pct: float = 0.15,
    rights=("CALL", "PUT"), expiry_type: str = "monthly", dte_range=None,
) -> Optional[dict]:
    """Pull the targeted option-chain slice for a held position and cache it on the
    loaded state. A SANCTIONED owned-state WRITE path (parallel to ``resolve_structure``;
    it deliberately writes fetched data into owned state, unlike the read-only
    ``price_scenario`` / ``price_payoff``, which must never mutate it).

    Enumerates the underlier's listed chain **once per underlier** (cached), filters to
    the window around spot and the held strike (``ticker_utils.filter_chain_slice``), and
    snapshots the survivors. The held leg's own greeks/IV come from the morning snapshot
    and are never re-pulled here — this fetches candidate contracts only.

    Returns ``{key, underlier, candidates, df, spot, spot_asof, pulled_at}`` or ``None``
    (no state / position / spot, or Bloomberg off). ``spot`` is the underlier's live
    PX_LAST fetched in the same snapshot request as the candidates (``spot_asof ==
    "live"``); it falls back to the morning-snapshot spot (``"snapshot"``) when the
    fetch cannot supply it. ``refresh`` re-snapshots with fresh greeks/IV
    (reusing the cached chain); ``refresh_chain`` additionally re-enumerates the chain.
    Re-opening the same window without ``refresh`` is a cache hit — no Bloomberg call."""
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

    rights_key = tuple(sorted(str(r).upper() for r in rights))
    if dte_range is None:
        key = (underlier, round(float(pos.strike), 4), pos.expiry, int(n_expiries),
               round(float(moneyness_pct), 4), str(expiry_type), rights_key)
        if not refresh and key in slices:
            return slices[key]

    from pm.core.ticker_utils import (filter_chain_slice, expiry_type_admits,
                                      parse_option_description)

    chain_entry = chains.get(underlier)
    if chain_entry is None or refresh_chain:
        from pm.core.bloomberg_client import fetch_option_chain
        parsed = [d for d in (parse_option_description(s) for s in fetch_option_chain(underlier)) if d]
        chain_entry = {"chain": parsed, "pulled_at": datetime.now()}
        chains[underlier] = chain_entry

    from pm.core.bloomberg_client import fetch_option_snapshots
    df = None
    spot_asof = "snapshot"

    if dte_range is not None:
        # The RANGE family: the DTE range resolves to LISTED expiries of the
        # selected type from the cached chain (client-side — the control can never
        # ask for an unlisted expiry), keyed by the resolved expiry tuple.
        # Snapshots cache PER EXPIRY (``exp_frames``), so widening the range
        # fetches exactly the missing expiries in one batched request and reuses
        # every frame already pulled.
        e_lo, e_hi = _resolve_expiry_range(dte_range)
        today_d = date.today()
        chosen = tuple(sorted({p["expiry"] for p in chain_entry["chain"]
                               if p.get("expiry") and today_d < p["expiry"]
                               and e_lo <= p["expiry"] <= e_hi
                               and expiry_type_admits(p["expiry"], expiry_type)}))
        key = (underlier, round(float(pos.strike), 4), pos.expiry, ("rng",) + chosen,
               round(float(moneyness_pct), 4), str(expiry_type), rights_key)
        if not refresh and key in slices:
            return slices[key]
        frames = cache.setdefault("exp_frames", {})

        def _fkey(exp):
            return (underlier, exp, round(float(pos.strike), 4),
                    round(float(moneyness_pct), 4), str(expiry_type), rights_key)

        tickers_by_exp = {
            exp: filter_chain_slice(chain_entry["chain"], spot, pos.strike,
                                    expiry_range=(exp, exp), moneyness_pct=moneyness_pct,
                                    rights=rights, expiry_type=expiry_type)
            for exp in chosen}
        missing = [exp for exp in chosen
                   if refresh or _fkey(exp) not in frames]
        if missing:
            to_fetch = sorted({tk for exp in missing for tk in tickers_by_exp[exp]})
            if to_fetch:
                fetched = fetch_option_snapshots(to_fetch + [underlier])
                if fetched is not None and underlier in fetched.index:
                    live_spot = coerce_float(fetched.loc[underlier].get("PX_LAST"))
                    fetched = fetched.drop(index=underlier)
                    if live_spot is not None and live_spot > 0:
                        spot = live_spot
                        spot_asof = "live"
                for exp in missing:
                    tks = [t for t in tickers_by_exp[exp]
                           if fetched is not None and t in fetched.index]
                    frames[_fkey(exp)] = fetched.loc[tks] if tks else None
            else:
                for exp in missing:
                    frames[_fkey(exp)] = None
        parts = [frames.get(_fkey(exp)) for exp in chosen]
        parts = [p for p in parts if p is not None and not getattr(p, "empty", True)]
        df = pd.concat(parts) if parts else None
        candidates = sorted({tk for exp in chosen for tk in tickers_by_exp[exp]})
    else:
        candidates = filter_chain_slice(
            chain_entry["chain"], spot, pos.strike, horizon_expiry=pos.expiry,
            n_expiries=n_expiries, moneyness_pct=moneyness_pct, rights=rights,
            expiry_type=expiry_type,
        )
        if candidates:
            # The underlier rides in the SAME batched snapshot request (one extra
            # security, no extra round trip): its live PX_LAST re-anchors the slice
            # spot so the candidate quotes and the spot they price against are
            # contemporaneous. The chain window above is still cut on the
            # morning-snapshot spot — the fresh value only exists once this fetch
            # returns. On any fetch failure (empty frame, missing row/field) the
            # snapshot spot stands and the stamp says so.
            df = fetch_option_snapshots(list(candidates) + [underlier])
            if df is not None and underlier in df.index:
                live_spot = coerce_float(df.loc[underlier].get("PX_LAST"))
                df = df.drop(index=underlier)
                if live_spot is not None and live_spot > 0:
                    spot = live_spot
                    spot_asof = "live"

    # Surface + IV+pp (a read-only derivation of the cached slice) and IV-rank (a
    # name-level metric cached beside the chain). Best-effort — a fit failure must not
    # break the slice pull. Both price against the (freshened) slice spot.
    surface, iv_pp = _slice_surface(acc, underlier, spot, df)
    result = {"key": key, "underlier": underlier, "candidates": candidates,
              "df": df, "spot": spot, "spot_asof": spot_asof,
              "pulled_at": datetime.now(),
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


def _scan_sig(dte_range, delta_band, expiry_type: str = "monthly") -> tuple:
    """A hashable signature of the scan controls, for cache keys. Carries the
    expiry-type so a Monthly scan and a Weekly/All scan at identical dials can
    never collide in any cache keyed through it."""
    rng = tuple(dte_range) if dte_range is not None else None
    band = tuple(delta_band) if delta_band is not None else None
    return (rng, band, str(expiry_type))


def _band_filter_df(df, delta_band):
    """The slice restricted to |delta| inside the band. A band means the delta must
    be KNOWN — rows without one drop (matching the joint path's rule). None-band
    returns the frame untouched."""
    if delta_band is None or df is None or getattr(df, "empty", True):
        return df
    lo, hi = float(delta_band[0]), float(delta_band[1])
    if "delta_mid" not in df.columns:
        return df.iloc[0:0]
    d = df["delta_mid"].abs()
    return df[(d >= lo) & (d <= hi)]


def _drop_candidate_caches(state, account: str) -> None:
    """Drop the account's cached scanner candidates/rankings (raw slices and
    per-expiry frames stay). The cached rankings embed structure context — kept
    legs, allocated slices — so a structure confirm/reject/choose/edit must
    invalidate them or a reopened scan serves candidates priced against a
    resolution the user just changed."""
    sc = getattr(state, "slice_cache", None) or {}
    for entry in sc.get("slices", {}).values():
        for cache_name in ("candidates_ranked", "candidates_priced"):
            m = entry.get(cache_name)
            if m:
                for k in [k for k in m if k[0] == account]:
                    m.pop(k, None)
    jr = sc.get("joint_ranked")
    if jr:
        for k in [k for k in jr if k[0] == account]:
            jr.pop(k, None)


def _spot_slice_df(state: PortfolioState, underlier: str, spot: float, dte_range=None,
                   expiry_type: str = "monthly"):
    """A spot-centered slice frame for a stock overlay (there is no held
    strike/expiry to anchor on). Reuses the per-underlier chain cache. The default
    window is the first three expiries of the selected type from ~30 days out, so
    the standard 30-45 day write is in the universe; an explicit ``dte_range``
    (the scan's DTE slider) overrides it. ``expiry_type`` is passed EXPLICITLY on
    both branches — this path once inherited filter_chain_slice's own
    monthly-only default invisibly, which kept weekly writes out of every
    overlay scan."""
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
    if dte_range is not None:
        tickers = filter_chain_slice(entry["chain"], spot, spot,
                                     expiry_range=_resolve_expiry_range(dte_range),
                                     moneyness_pct=0.15, expiry_type=expiry_type)
    else:
        tickers = filter_chain_slice(entry["chain"], spot, spot,
                                     horizon_expiry=today + timedelta(days=30),
                                     n_expiries=3, moneyness_pct=0.15,
                                     expiry_type=expiry_type)
    return fetch_option_snapshots(tickers) if tickers else None


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
