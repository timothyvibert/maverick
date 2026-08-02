"""Candidate generation, pricing and ranking for the scanner.

Single-leg rolls, stock overlays and joint (multi-leg) rolls, each priced as
the RESULTING structure through the pure payoff engine — kept sibling legs at
entry basis, every new leg at the current slice mid. Rankings cache on the
slice / state so pill switches and the comparison tab stay Bloomberg-free;
the cache-key shapes (including ``_scan_sig``'s expiry-type dimension) are
part of the contract and must not change. Every entry point takes the loaded
state explicitly — the singleton lives in ``pm.ui.state_access``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pm.scanner.slice import (_band_filter_df, _contract_metrics, _div_yield,
                              _scan_sig, _spot_from_snapshot, _spot_slice_df,
                              pull_slice)
from pm.store.portfolio_state import AccountState, PortfolioState
from pm.store.state_reads import (coerce_float, position_by_id,
                                  structure_for_position)


# ---------------------------------------------------------------------------
# Candidate generation + per-candidate economics (the scanner's pricing step).
# A sanctioned owned-state derivation: reads the cached slice, prices each candidate
# through the validated payoff engine (one compute_payoff call per candidate), and
# attaches the result to the slice entry. No new pricing math.
# ---------------------------------------------------------------------------


# Sentinel for the scanner's structure anchor: "no explicit structure given —
# resolve it the same way the payoff tab does" (structure_for_position). The drawer
# always passes drawer-state's structure_id VERBATIM (None means scan the leg
# standalone — e.g. the scenario drill route's own-axis contract), so every open
# route's scan and its compare's current side read ONE structure by construction.
STRUCTURE_AUTO = object()


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


def _resolve_scan_structure_id(state, account: str, position_id: str, structure_id):
    if structure_id is STRUCTURE_AUTO:
        return structure_for_position(state, account, position_id)
    return structure_id


def _structure_roll_context(state: PortfolioState, acc: AccountState, pos, sid):
    """(sibling_legs, rolled_qty, warnings) for rolling ``pos`` inside structure
    ``sid``: the structure's OTHER legs as entry-basis payoff leg dicts — the same
    assembly the payoff panel prices (allocated slices, contract-multiplier
    normalized) — plus the rolled leg's own signed standard-contract slice and the
    assembly's warnings for the KEPT side. ``(None, None, None)`` when there is no
    structure, the assembly is degraded, or it cannot name a non-zero rolled slice
    (degrade to the position-anchored scan rather than guess)."""
    if not sid:
        return None, None, None
    struct = next((s for s in (getattr(acc, "structures", None) or [])
                   if s.structure_id == sid), None)
    if struct is None:
        return None, None, None
    try:
        from pm.risk.payoff import build_structure_payoff_legs
        asm = build_structure_payoff_legs(state, acc, struct)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("structure context failed for %s", pos.position_id)
        return None, None, None
    if asm.get("degraded"):
        return None, None, None      # a partial read must not masquerade as the structure
    legs = list(asm.get("leg_dicts") or [])
    rolled = [d for d in legs
              if d.get("position_id") == pos.position_id
              and d.get("opt_type") in ("Call", "Put")]
    if len(rolled) != 1 or not rolled[0].get("qty"):
        return None, None, None
    kept = [d for d in legs if d is not rolled[0]]
    pid_prefix = f"{pos.position_id}:"
    kept_warns = [w for w in (asm.get("warnings") or []) if not w.startswith(pid_prefix)]
    return kept, rolled[0].get("qty"), kept_warns


def generate_slice_candidates(state, account: str, position_id: str, *,
                              objectives=None, cap: int = 15,
                              n_expiries: int = 3, structure_id=STRUCTURE_AUTO,
                              dte_range=None, delta_band=None,
                              expiry_type: str = "monthly"):
    """Generate + price the adjustment candidates for a held position — rolls for a held
    option, single-leg overlays for held stock. Attaches the priced candidates to the
    slice cache entry and returns them (or None)."""
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
    from pm.candidates.objectives import build_harvest_params
    hp = build_harvest_params()          # persisted scanner dials, read per scan
    cands = []
    try:
        if pos.asset_class == "option":
            sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                            dte_range=dte_range, expiry_type=expiry_type)
            if sl is None:
                return None
            q = _div_yield(acc, sl["underlier"])
            held = {"strike": pos.strike, "expiry": pos.expiry, "right": pos.right,
                    "quantity": pos.quantity, "delta": _held_option_delta(acc, pos),
                    "multiplier": getattr(pos, "multiplier", None)}
            # A leg inside a detected structure rolls WITHIN it: the kept sibling
            # legs (entry basis, allocated slices) ride into generation so every
            # candidate prices as the resulting structure. A standalone option
            # keeps the covered-call heuristic (held_stock).
            sid = _resolve_scan_structure_id(state, account, position_id, structure_id)
            sibling_legs, rolled_qty, ctx_warns = _structure_roll_context(state, acc, pos, sid)
            cands = candidates_from_slice(
                _band_filter_df(sl["df"], delta_band), held,
                _contemporaneous_mid(pos, sl), sl["spot"],
                held_stock=_held_stock(acc, pos), sibling_legs=sibling_legs,
                rolled_qty=rolled_qty, context_warnings=ctx_warns, risk_free_curve=curve,
                risk_free_rate=rfr, div_yield=q, objectives=objectives, cap=cap,
                harvest_params=hp)
            # Keyed per position AND structure anchor: the raw slice is shared
            # across positions/accounts holding the same contract, but the priced
            # candidates embed ONE position's structure context and must never
            # leak across positions, accounts, or open routes. (The slice entry
            # itself is window-keyed; the |delta| band scopes the nested key.)
            sl.setdefault("candidates_priced", {})[
                (account, position_id, sid,
                 _scan_sig(None, delta_band, expiry_type))] = cands
        elif pos.asset_class in ("equity", "fund_etf"):
            spot = _spot_from_snapshot(acc, pos.bbg_ticker)
            basis = _per_share_basis(pos)
            if spot is None or basis is None or not pos.quantity:
                return []
            df = _band_filter_df(_spot_slice_df(state, pos.bbg_ticker, spot,
                                                dte_range=dte_range,
                                                expiry_type=expiry_type), delta_band)
            # Retain the overlay slice so the scanner chain table can read each
            # candidate's contract liquidity (the option-roll path keeps its slice; the
            # overlay path builds a transient frame, so stash it under the position,
            # scoped by the scan controls).
            state.slice_cache.setdefault("overlay_dfs", {})[
                (position_id, _scan_sig(dte_range, delta_band, expiry_type))] = df
            # The overlay slice's own as-of (the option path's pulled_at analog)
            # — the adjustment ticket quotes it, so it must exist to be quoted.
            state.slice_cache.setdefault("overlay_pulled", {})[
                (position_id, _scan_sig(dte_range, delta_band, expiry_type))] = datetime.now()
            q = _div_yield(acc, pos.bbg_ticker)
            cands = overlays_from_slice(df, spot, int(pos.quantity), basis,
                                        risk_free_curve=curve, risk_free_rate=rfr,
                                        div_yield=q, cap=cap, harvest_params=hp)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("candidate generation failed for %s", position_id)
    return cands


def rank_slice_candidates(state, account: str, position_id: str, *,
                          objectives=None, cap: int = 15,
                          n_expiries: int = 3, structure_id=STRUCTURE_AUTO,
                          dte_range=None, delta_band=None,
                          expiry_type: str = "monthly"):
    """Generate + price + rank the adjustment candidates for a held position, grouped
    by objective. Reads the account's client profile and the slice's IV+pp rows and
    ranks each objective's candidates through ``pm.candidates.ranking``; the only
    state write is caching the result on the option slice for reuse. On a cache
    MISS it delegates to ``generate_slice_candidates``, which resolves the slice
    via the sanctioned on-demand Bloomberg pull — only the warm path runs
    Bloomberg-free. Returns ``{objective: [RankedCandidate, ...]}`` (or None)."""
    from datetime import date

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
    sid = (_resolve_scan_structure_id(state, account, position_id, structure_id)
           if pos.asset_class == "option" else None)
    band_sig = _scan_sig(None, delta_band, expiry_type)
    if pos.asset_class == "option":
        cached_sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                               dte_range=dte_range, expiry_type=expiry_type)
        cached_ranked = ((cached_sl.get("candidates_ranked") or {})
                         .get((account, position_id, sid, band_sig))
                         if cached_sl else None)
        if cached_ranked:
            return cached_ranked
    else:
        cached = state.slice_cache.get("overlay_ranked", {}).get(
            (position_id, _scan_sig(dte_range, delta_band, expiry_type)))
        if cached:
            return cached

    cands = generate_slice_candidates(state, account, position_id,
                                      objectives=objectives, cap=cap,
                                      n_expiries=n_expiries, structure_id=sid,
                                      dte_range=dte_range, delta_band=delta_band,
                                      expiry_type=expiry_type)
    if not cands:
        return None

    profile = getattr(acc, "client_profile", None)

    # IV+pp rows + the held leg's Δ / DTE are available only on the option roll path
    # (a stock overlay has no held option leg and no anchored slice); the ranker
    # degrades cleanly when they are absent. The liquidity map (bid/ask/OI per
    # ticker, for Harvest's execution adjustment + flags) reads whichever frame
    # this scan priced from — the option slice, else the stashed overlay frame.
    iv_pp = None
    held = None
    sl = None
    liq_df = None
    if pos.asset_class == "option":
        sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                        dte_range=dte_range, expiry_type=expiry_type)
        iv_pp = sl.get("iv_pp") if sl else None
        held = {"delta": _held_option_delta(acc, pos),
                "dte": (pos.expiry - date.today()).days if pos.expiry else None,
                "strike": pos.strike, "right": pos.right}
        liq_df = sl.get("df") if sl else None
    else:
        liq_df = state.slice_cache.get("overlay_dfs", {}).get(
            (position_id, _scan_sig(dte_range, delta_band, expiry_type)))

    from pm.candidates.objectives import build_harvest_params
    from pm.candidates.ranking import rank_candidates
    liquidity = _contract_metrics(liq_df)
    hp = build_harvest_params()
    by_objective: dict = {}
    for c in cands:
        by_objective.setdefault(c.objective, []).append(c)
    ranked = {obj: rank_candidates(cs, objective=obj, client_profile=profile,
                                   iv_pp=iv_pp, held=held, liquidity=liquidity,
                                   params=hp)
              for obj, cs in by_objective.items()}

    if sl is not None:
        # Keyed per position + structure anchor + |delta| band (see
        # generate_slice_candidates): the ranking embeds one structure context;
        # same-contract slices are shared; the slice entry is window-keyed.
        sl.setdefault("candidates_ranked", {})[
            (account, position_id, sid, band_sig)] = ranked
    elif pos.asset_class in ("equity", "fund_etf"):
        # The overlay analog of the slice-attached ranking cache: keyed per
        # position + scan controls beside overlay_dfs, read by scanner_candidate /
        # the cache-hit above. Dropped by a scanner Refresh (scanner_view_data).
        state.slice_cache.setdefault("overlay_ranked", {})[
            (position_id, _scan_sig(dte_range, delta_band, expiry_type))] = ranked
    return ranked


def generate_joint_candidates(state, account: str, position_id: str, rolled_pids, *,
                              objectives=None, cap: int = 15, n_expiries: int = 3,
                              structure_id=STRUCTURE_AUTO, delta_band=None,
                              dte_range=None, expiry_type: str = "monthly"):
    """Joint-roll candidates: roll a SET of the enclosing structure's option legs
    together to one common new expiry, priced as the resulting structure (kept
    siblings at entry basis, every new leg at the current slice mid — the same
    seam single-leg rolls use). ``rolled_pids`` is a set of the structure's leg
    position ids; the scanned ``position_id`` anchors the slice — targets draw
    from the ANCHOR's pulled window (its forward monthlies, its moneyness band);
    a rolled sibling beyond that window honestly yields nothing (re-anchoring
    belongs to the scan controls). A one-element set IS the single-leg path
    end-to-end (delegated — identical by construction, with that path's normal
    ranking cache; ``delta_band`` does not apply there and is ignored). The
    multi-leg path writes no caches — the leg-roster surface will own joint
    caching when it lands — and may pull one single-ticker mid per rolled leg
    that sits outside the slice (the joint analogue of the sanctioned held-leg
    mid; uncached, so each regenerate re-pulls). Returns the priced candidates,
    or None (no state / Bloomberg off / no enclosing structure / a rolled pid
    that is not one of its sized option legs)."""
    if state is None or not getattr(state, "bloomberg_ok", False):
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    pos = next((p for p in acc.positions if p.position_id == position_id), None)
    if pos is None or pos.asset_class != "option":
        return None
    pids = {str(p) for p in (rolled_pids or [])} or {position_id}
    if len(pids) == 1:
        return generate_slice_candidates(state, account, next(iter(pids)), objectives=objectives,
                                         cap=cap, n_expiries=n_expiries,
                                         structure_id=structure_id,
                                         dte_range=dte_range, delta_band=delta_band,
                                         expiry_type=expiry_type)

    sid = _resolve_scan_structure_id(state, account, position_id, structure_id)
    if not sid:
        return None
    struct = next((s for s in (getattr(acc, "structures", None) or [])
                   if s.structure_id == sid), None)
    if struct is None:
        return None
    try:
        from pm.risk.payoff import build_structure_payoff_legs
        asm = build_structure_payoff_legs(state, acc, struct)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("joint structure context failed for %s", sid)
        return None
    if asm.get("degraded"):
        return None
    legs = list(asm.get("leg_dicts") or [])
    rolled_dicts = [d for d in legs
                    if d.get("position_id") in pids and d.get("opt_type") in ("Call", "Put")]
    if {d["position_id"] for d in rolled_dicts} != pids \
            or any(not d.get("qty") for d in rolled_dicts):
        return None                     # every pid must be a sized option leg
    kept = [d for d in legs if d not in rolled_dicts]

    sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                    dte_range=dte_range, expiry_type=expiry_type)
    if sl is None:
        return None
    by_id = {p.position_id: p for p in acc.positions}
    prefixes = tuple(f"{p}:" for p in pids)
    warns = [w for w in (asm.get("warnings") or []) if not w.startswith(prefixes)]
    rolled = []
    for d in rolled_dicts:
        p = by_id.get(d["position_id"])
        rolled.append({
            "position_id": d["position_id"], "strike": d["K"], "expiry": d["expiry"],
            "right": "CALL" if d["opt_type"] == "Call" else "PUT", "qty": d["qty"],
            "mid": _contemporaneous_mid(p, sl) if p is not None else None,
            "delta": _held_option_delta(acc, p) if p is not None else None,
        })
        full_std = (abs(coerce_float(getattr(p, "quantity", None)) or 0.0)
                    * ((coerce_float(getattr(p, "multiplier", None)) or 100.0) / 100.0))
        if full_std and abs(coerce_float(d["qty"]) or 0.0) + 1e-9 < full_std:
            warns.append(f"{d['position_id']}: rolls the structure's "
                         f"{abs(d['qty']):g}-contract slice of a {full_std:g}-contract "
                         "position — the remainder sits outside this structure")

    from pm.candidates.generate import joint_candidates_from_slice
    try:
        return joint_candidates_from_slice(
            sl["df"], rolled, sl["spot"], sibling_legs=kept, context_warnings=warns,
            risk_free_curve=getattr(state, "risk_free_curve", None) or [],
            risk_free_rate=getattr(state, "risk_free_rate", 0.045),
            div_yield=_div_yield(acc, sl["underlier"]), objectives=objectives,
            cap=cap, delta_band=delta_band)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("joint generation failed for %s", sid)
        return None


def rank_joint_candidates(state, account: str, position_id: str, rolled_pids, *,
                          objectives=None, cap: int = 15, n_expiries: int = 3,
                          structure_id=STRUCTURE_AUTO, delta_band=None,
                          dte_range=None, expiry_type: str = "monthly"):
    """Generate + rank the joint-roll candidates, grouped by objective — the
    joint analogue of ``rank_slice_candidates``. A one-element rolled set is the
    single-leg path END-TO-END (delegated here too, so its ranking carries the
    held-leg context and the normal cache — identical to a direct single-leg
    scan). Multi-leg sets rank with ``held=None``: each joint candidate carries
    its objective's own metric (``joint_driver``) and its common-expiry tenor.
    Returns ``{objective: [RankedCandidate, ...]}`` or None."""
    pids = {str(p) for p in (rolled_pids or [])} or {position_id}
    if len(pids) == 1:
        return rank_slice_candidates(state, account, next(iter(pids)), objectives=objectives,
                                     cap=cap, n_expiries=n_expiries,
                                     structure_id=structure_id,
                                     dte_range=dte_range, delta_band=delta_band,
                                     expiry_type=expiry_type)
    if state is None:
        return None
    # The roster owns joint caching: keyed by anchor + structure + the rolled SET
    # + the scan controls, dropped by Refresh and by structure resolutions.
    sid = _resolve_scan_structure_id(state, account, position_id, structure_id)
    jkey = (account, position_id, sid, frozenset(pids),
            _scan_sig(dte_range, delta_band, expiry_type))
    cached = state.slice_cache.get("joint_ranked", {}).get(jkey)
    if cached:
        return cached
    cands = generate_joint_candidates(state, account, position_id, rolled_pids,
                                      objectives=objectives, cap=cap,
                                      n_expiries=n_expiries, structure_id=sid,
                                      delta_band=delta_band, dte_range=dte_range,
                                      expiry_type=expiry_type)
    if not cands:
        return None
    acc = state.accounts.get(account)
    sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                    dte_range=dte_range, expiry_type=expiry_type)
    iv_pp = sl.get("iv_pp") if sl else None
    from pm.candidates.ranking import rank_candidates
    by_objective: dict = {}
    for c in cands:
        by_objective.setdefault(c.objective, []).append(c)
    ranked = {obj: rank_candidates(cs, objective=obj,
                                   client_profile=getattr(acc, "client_profile", None),
                                   iv_pp=iv_pp, held=None)
              for obj, cs in by_objective.items()}
    state.slice_cache.setdefault("joint_ranked", {})[jkey] = ranked
    return ranked


def scanner_candidate(state, account: str, position_id: str, objective: str, rank: int,
                      *, n_expiries: int = 3, structure_id=STRUCTURE_AUTO,
                      dte_range=None, delta_band=None, rolled_pids=None,
                      expiry_type: str = "monthly"):
    """The cached ranked scanner candidate at ``(objective, rank)`` for a held position,
    or None. Reads the slice the scanner already pulled + ranked — but a COLD
    cache (fresh reload, first touch) resolves through ``pull_slice``, which
    performs the sanctioned on-demand Bloomberg pull; only the warm path is a
    pure read.

    ``n_expiries`` must be the scanner's ACTIVE window (the drawer threads its
    window store through): the slice cache is keyed by window, so reading the
    default while the table shows an expanded window would resolve (objective,
    rank) against a different, stale ranking than the row the user clicked.
    Held-stock (overlay) positions have no option slice — their ranking is read
    from the per-position overlay cache instead."""
    if state is None:
        return None
    pos = position_by_id(state, account, position_id)
    if pos is None:
        return None
    pids = {str(p) for p in (rolled_pids or [])}
    if len(pids) > 1:
        # A joint roll resolves from the joint cache (rank_joint_candidates owns it).
        sid = _resolve_scan_structure_id(state, account, position_id, structure_id)
        ranked_map = state.slice_cache.get("joint_ranked", {}).get(
            (account, position_id, sid, frozenset(pids),
             _scan_sig(dte_range, delta_band, expiry_type)))
    elif pos.asset_class == "option":
        # Same sentinel contract as the scan itself: the drawer passes drawer-state's
        # structure_id verbatim (None = standalone), so the (obj, rank) lookup lands
        # on the ranking the clicked table was rendered from.
        sid = _resolve_scan_structure_id(state, account, position_id, structure_id)
        sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                        dte_range=dte_range, expiry_type=expiry_type)
        ranked_map = ((sl.get("candidates_ranked") or {})
                      .get((account, position_id, sid,
                            _scan_sig(None, delta_band, expiry_type)))
                      if sl else None)
    else:
        ranked_map = state.slice_cache.get("overlay_ranked", {}).get(
            (position_id, _scan_sig(dte_range, delta_band, expiry_type)))
    ranked = (ranked_map or {}).get(objective) or []
    return next((r for r in ranked if getattr(r, "rank", None) == rank), None)


def price_candidate(state, account: str, position_id: str, objective: str, rank: int, *,
                    shock=None, n_expiries: int = 3, structure_id=STRUCTURE_AUTO,
                    dte_range=None, delta_band=None, rolled_pids=None,
                    expiry_type: str = "monthly"):
    """The candidate side of the current-vs-candidate comparison — a read-only payoff
    reprice of a ranked candidate's resulting position under a shock. Prices the
    candidate's engine legs through the pure ``compute_payoff`` (the same engine the
    structure payoff wraps) at the SLICE's spot (option rolls) or the snapshot spot
    (stock overlays), so the candidate curve shares the current position's grid.
    ``n_expiries`` is the scanner's active window (see ``scanner_candidate``).
    No Bloomberg beyond the already-cached slice, no reload, no ``_RUNTIME`` write,
    no engine change. Returns the ``compute_payoff`` dict or None."""
    rc = scanner_candidate(state, account, position_id, objective, rank, n_expiries=n_expiries,
                           structure_id=structure_id, dte_range=dte_range,
                           delta_band=delta_band, rolled_pids=rolled_pids,
                           expiry_type=expiry_type)
    if rc is None:
        return None
    acc = state.accounts.get(account) if state else None
    pos = position_by_id(state, account, position_id) if state else None
    if acc is None or pos is None:
        return None
    if pos.asset_class == "option":
        sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                        dte_range=dte_range, expiry_type=expiry_type)
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
        today = date.today()   # one as-of for tier1 AND the payoff (consistency)
        tier1 = _build_tier1(legs, today)
        return compute_payoff(legs, float(spot), tier1, shock=shock, today=today)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("price_candidate failed for %s", position_id)
        return None
