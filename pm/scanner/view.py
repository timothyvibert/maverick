"""Read-only packaging for the scanner drawer.

``scanner_view_data`` assembles the ranked candidates (single-leg, overlay,
or joint) plus the slice metadata the view stamps — spot / as-of, the chain
rows, the fitted surface + IV+pp, IV-rank, the listed expiries the DTE
control derives its bounds from, and the honest empty-scan note.
``scanner_roster`` reads the Managing band's rows off already-loaded state.
The pull is the only market I/O — the view itself never recomputes. Takes
the loaded state explicitly — the singleton lives in ``pm.ui.state_access``.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pm.scanner.candidates import (STRUCTURE_AUTO, _resolve_scan_structure_id,
                                   rank_joint_candidates,
                                   rank_slice_candidates)
from pm.scanner.slice import (_contract_metrics, _resolve_expiry_range,
                              _scan_sig, _snapshot_underlying_row,
                              _spot_from_snapshot, chain_expiries, pull_slice)
from pm.store.state_reads import coerce_float, position_by_id


def scanner_roster(state, account: str, position_id: str, *, structure_id=STRUCTURE_AUTO):
    """The Managing band's data, read-only from loaded state: the enclosing
    structure's legs as roster rows (role · contract · DTE · Δ · morning-snapshot
    mid · the |Δ| assignment proxy on short legs · the opened-on marker) plus the
    structure's stored current economics (the load-path Tier-2 record and the
    snapshot net delta — zero recompute). A standalone position yields its single
    row; None when nothing resolves."""
    if state is None:
        return None
    acc = state.accounts.get(account)
    pos = position_by_id(state, account, position_id) if state else None
    if acc is None or pos is None:
        return None
    sid = _resolve_scan_structure_id(state, account, position_id, structure_id)
    struct = next((s for s in (getattr(acc, "structures", None) or [])
                   if s.structure_id == sid), None) if sid else None
    by_id = {p.position_id: p for p in acc.positions}
    opts = getattr(getattr(acc, "snapshot", None), "options", None)

    def _snap(tk, col):
        if opts is not None and not getattr(opts, "empty", True) and tk in opts.index:
            return coerce_float(opts.loc[tk].get(col))
        return None

    def _row(pid, alloc, role):
        p = by_id.get(pid)
        if p is None:
            return None
        if getattr(p, "asset_class", None) == "option":
            d = _snap(p.bbg_ticker, "delta_mid")
            qty = alloc if alloc is not None else p.quantity
            return {"position_id": pid, "role": role or "",
                    "contract": f"{p.strike:g} {(p.right or '?')[0]}",
                    "qty": qty,
                    "dte": (p.expiry - date.today()).days if p.expiry else None,
                    "delta": d, "mid": _snap(p.bbg_ticker, "PX_MID"),
                    "p_assign": (abs(d) if (d is not None and (qty or 0) < 0) else None),
                    "is_option": True, "anchor": pid == position_id}
        qty = alloc if alloc is not None else p.quantity
        return {"position_id": pid, "role": role or "stock",
                "contract": "stock", "qty": qty, "dte": None, "delta": None,
                "mid": None, "p_assign": None, "is_option": False,
                "anchor": pid == position_id}

    if struct is not None:
        rows = [r for r in (_row(lg.position_id, lg.allocated_qty, lg.role)
                            for lg in struct.legs) if r is not None]
        tier2 = (getattr(acc, "structure_tier2", None) or {}).get(sid) or {}
        net_delta = 0.0
        have_delta = False
        for lg in struct.legs:
            p = by_id.get(lg.position_id)
            if p is None:
                continue
            if getattr(p, "asset_class", None) == "option":
                d = _snap(p.bbg_ticker, "delta_mid")
                if d is not None:
                    net_delta += d * (lg.allocated_qty or 0) * 100.0
                    have_delta = True
            else:
                net_delta += (lg.allocated_qty or 0)
                have_delta = True
        econ = {"structure_type": getattr(struct, "type", None),
                "status": getattr(struct, "status", None),
                "net_delta": net_delta if have_delta else None,
                "tier2": tier2}
        return {"rows": rows, "sid": sid, "econ": econ}
    row = _row(position_id, None, getattr(pos, "right", None) or pos.asset_class)
    return {"rows": [row] if row else [], "sid": None,
            "econ": {"structure_type": None, "status": None, "net_delta": None,
                     "tier2": {}}}


def _window_expiry_counts(state, underlier, dte_range, expiry_type):
    """(n_selected_type, n_other_types) LISTED expiries inside the DTE window,
    read from the already-cached chain — no fetch. (None, None) when the chain
    or the window is unavailable, so the caller falls back to the generic note."""
    entry = (state.slice_cache.get("chains", {}) or {}).get(underlier)
    if not entry or dte_range is None:
        return (None, None)
    from pm.core.ticker_utils import expiry_type_admits
    e_lo, e_hi = _resolve_expiry_range(dte_range)
    today_d = date.today()
    wins = {p["expiry"] for p in entry.get("chain") or []
            if p.get("expiry") and today_d < p["expiry"] and e_lo <= p["expiry"] <= e_hi}
    sel = {e for e in wins if expiry_type_admits(e, expiry_type)}
    return (len(sel), len(wins) - len(sel))


def _empty_scan_note(expiry_type, n_sel, n_other) -> str:
    """The empty-scan note, honest about WHY: expiries of the selected type were
    filtered out of the window (say so, name the broader selection), the window
    is genuinely empty of listed expiries, or contracts existed and the band /
    objectives left nothing."""
    if n_sel == 0 and n_other:
        label = "Monthly" if expiry_type == "monthly" else "Weekly"
        broader = "Weekly or All" if expiry_type == "monthly" else "All"
        plural = "y" if n_other == 1 else "ies"
        return (f"no {label} expiries in the DTE window — {n_other} listed "
                f"expir{plural} of other types inside it; switch Expiries "
                f"to {broader}")
    if n_sel == 0 and n_other == 0:
        return "no listed expiries in the DTE window — widen the window and press Scan"
    return ("no candidates pass the current Delta Band / objectives — adjust a "
            "control and press Scan")


def scanner_view_data(state, account: str, position_id: str, *, objectives=None, cap: int = 15,
                      refresh: bool = False, n_expiries: int = 3,
                      structure_id=STRUCTURE_AUTO, dte_range=None, delta_band=None,
                      rolled_pids=None, expiry_type: str = "monthly") -> Optional[dict]:
    """Read-only packaging for the scanner drawer: the ranked adjustment candidates
    for a held position (single-leg, overlay, or — with 2+ ``rolled_pids`` — the
    joint path), plus the slice metadata the view stamps: spot/as-of, the full
    chain rows, the fitted surface + IV+pp, IV-rank (level context beside the
    shape metric), realized-vol ratio, fit quality, and the listed expiries the
    DTE control derives its bounds from. ``refresh`` re-pulls the option slice /
    drops the overlay + joint caches first. The pull is the only market I/O —
    the view itself never recomputes."""
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    pos = position_by_id(state, account, position_id)
    if pos is None:
        return None

    pids = {str(p) for p in (rolled_pids or [])}
    joint = len(pids) > 1

    # A refresh re-snapshots the option slice (the sanctioned write path); on the
    # overlay path it drops the position's cached dfs + rankings (all control
    # signatures) so the scan genuinely re-generates instead of cache-hitting;
    # joint rankings for this anchor drop either way.
    if refresh and pos.asset_class == "option":
        pull_slice(state, account, position_id, refresh=True, n_expiries=n_expiries,
                   dte_range=dte_range, expiry_type=expiry_type)
    elif refresh:
        for cname in ("overlay_dfs", "overlay_ranked", "overlay_pulled"):
            m = state.slice_cache.get(cname, {})
            for k in [k for k in m if isinstance(k, tuple) and k[0] == position_id]:
                m.pop(k, None)
    if refresh:
        jr = state.slice_cache.get("joint_ranked", {})
        for k in [k for k in jr if k[0] == account and k[1] == position_id]:
            jr.pop(k, None)

    note = None
    if joint:
        ranked = rank_joint_candidates(state, account, position_id, pids, objectives=objectives,
                                       cap=cap, n_expiries=n_expiries,
                                       structure_id=structure_id,
                                       delta_band=delta_band, dte_range=dte_range,
                                       expiry_type=expiry_type)
        if ranked is None:
            # Honest empty state: the joint path found nothing — most often no
            # admissible common expiry/strike inside the fetched window for one
            # of the rolled legs (the increment-3 logged case), else no structure.
            ranked = {}
            note = ("no admissible roll for the selected legs within the fetched "
                    "window — widen the DTE range, or check the legs share a "
                    "detected structure")
    else:
        ranked = rank_slice_candidates(state, account, position_id,
                                       objectives=objectives, cap=cap,
                                       n_expiries=n_expiries, structure_id=structure_id,
                                       dte_range=dte_range, delta_band=delta_band,
                                       expiry_type=expiry_type)
        if ranked is None and (dte_range is not None or delta_band is not None):
            ranked = {}
            note_underlier = (pos.underlying_bbg_ticker if pos.asset_class == "option"
                              else pos.bbg_ticker)
            n_sel, n_other = _window_expiry_counts(state, note_underlier,
                                                   dte_range, expiry_type)
            note = _empty_scan_note(expiry_type, n_sel, n_other)
        elif ranked is None:
            return None

    pulled_at = spot = underlier = spot_asof = None
    df = surface = iv_pp = iv_rank = None
    if pos.asset_class == "option":
        sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                        dte_range=dte_range, expiry_type=expiry_type)
        if sl:
            pulled_at, spot, underlier, df = (sl.get("pulled_at"), sl.get("spot"),
                                              sl.get("underlier"), sl.get("df"))
            surface, iv_pp = sl.get("surface"), sl.get("iv_pp")
            spot_asof = sl.get("spot_asof")
            iv_rank = sl.get("iv_rank")
    else:
        underlier = pos.bbg_ticker
        spot = _spot_from_snapshot(acc, pos.bbg_ticker)
        df = state.slice_cache.get("overlay_dfs", {}).get(
            (position_id, _scan_sig(dte_range, delta_band, expiry_type)))

    # Level + quality context for the cap line: IV-rank (the 52-week percentile the
    # slice already carries), current IV over 30-day realized, and the fit's R².
    row = _snapshot_underlying_row(acc, underlier) if underlier else None
    rv30 = coerce_float(row.get("VOLATILITY_30D")) if row is not None else None
    cur_iv = (iv_rank or {}).get("current_3m_atm") if iv_rank else None
    iv_rv = (cur_iv / rv30) if (cur_iv is not None and rv30 and rv30 > 0) else None
    day_pct = coerce_float(row.get("CHG_PCT_1D")) if row is not None else None

    metrics = _contract_metrics(df)
    return {"ranked": ranked, "pulled_at": pulled_at, "spot": spot,
            "spot_asof": spot_asof, "day_pct": day_pct,
            "underlier": underlier, "kind": pos.asset_class,
            "contract_metrics": metrics, "surface": surface, "iv_pp": iv_pp,
            "contracts": _slice_contracts(df, iv_pp, metrics),
            "held_strike": getattr(pos, "strike", None),
            "iv_rank": iv_rank, "iv_rv_ratio": iv_rv, "rv30": rv30,
            "fit_r2": getattr(surface, "r2", None) if surface is not None else None,
            "listed_expiries": chain_expiries(state, account, position_id, expiry_type),
            "note": note, "joint": joint}


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
