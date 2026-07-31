"""The adjustment-ticket compose — the scanner's one-transaction proposal.

Read-only: composes the cached ranked candidate, the structure assembly's
ledger slices, the sanctioned held-leg mid pull, the pure payoff engine and
the pure structure detector over COPIED positions. No state write-back, no
reload, nothing persisted, and never any order placement. Takes the loaded
state explicitly — the singleton lives in ``pm.ui.state_access``.
"""
from __future__ import annotations

from datetime import date

from pm.scanner.candidates import (STRUCTURE_AUTO, _contemporaneous_mid,
                                   _per_share_basis,
                                   _resolve_scan_structure_id,
                                   scanner_candidate)
from pm.scanner.slice import _scan_sig, _spot_from_snapshot, pull_slice
from pm.store.state_reads import coerce_float, position_by_id


def build_adjustment_ticket(state, account: str, position_id: str, *, objective=None,
                            rank=None, structure_id=STRUCTURE_AUTO, dte_range=None,
                            delta_band=None, rolled_pids=None, capture_pids=None,
                            n_expiries: int = 3, expiry_type: str = "monthly"):
    """The adjustment ticket for the scanner's selected candidate and/or the
    roster's capture marks — the whole adjustment as ONE transaction: a
    close-set plus an open-set at per-leg contemporaneous mids, the net cash,
    the resulting position's priced economics, and a factual coverage flag when
    the transaction leaves a short uncovered.

    A PROPOSAL, read-only: composes the cached ranked candidate
    (``scanner_candidate``), the structure assembly's ledger slices, the
    sanctioned held-leg mid pull, the pure payoff engine and the pure structure
    detector over COPIED positions — no ``_RUNTIME`` write-back, no reload,
    nothing persisted, and never any order placement. ``capture_pids`` marks
    roster legs to close OUTSIDE the rolled set; with no candidate selected the
    ticket is close-only (captures alone). Returns an ``AdjustmentTicket`` or
    None (no state / no candidate resolved / nothing to trade)."""
    if state is None:
        return None
    acc = state.accounts.get(account)
    if acc is None:
        return None
    pos = position_by_id(state, account, position_id)
    if pos is None:
        return None
    captures = [str(p) for p in (capture_pids or [])]
    rc = None
    if objective is not None and rank is not None:
        rc = scanner_candidate(state, account, position_id, objective, rank,
                               n_expiries=n_expiries, structure_id=structure_id,
                               dte_range=dte_range, delta_band=delta_band,
                               rolled_pids=(rolled_pids
                                            if rolled_pids and len(set(rolled_pids)) > 1
                                            else None),
                               expiry_type=expiry_type)
        if rc is None:
            return None
    cand = getattr(rc, "candidate", None)
    if cand is None and not captures:
        return None

    from pm.candidates import ticket as tkt

    by_id = {p.position_id: p for p in acc.positions}
    sym = pos.underlying_symbol or pos.symbol
    warnings: list = []

    # Structure context — the ledger's allocated slices (REAL contracts), the
    # same assembly every scan reads. A degraded read falls back to
    # full-position semantics, matching the position-anchored scan, and says so.
    sid = _resolve_scan_structure_id(state, account, position_id, structure_id)
    struct = next((s for s in (getattr(acc, "structures", None) or [])
                   if s.structure_id == sid), None) if sid else None
    alloc_by_pid: dict = {}
    asm_legs: list = []
    if struct is not None:
        try:
            from pm.risk.payoff import build_structure_payoff_legs
            asm = build_structure_payoff_legs(state, acc, struct)
        except Exception:
            asm = {"degraded": True}
        if asm.get("degraded"):
            struct = None
            warnings.append("structure context degraded — the ticket uses "
                            "full-position quantities")
        else:
            asm_legs = list(asm.get("leg_dicts") or [])
            alloc_by_pid = {lg.position_id: lg.allocated_qty for lg in struct.legs}

    sl = pull_slice(state, account, position_id, n_expiries=n_expiries,
                    dte_range=dte_range,
                    expiry_type=expiry_type) if pos.asset_class == "option" else None
    spot = (sl.get("spot") if sl else None) or _spot_from_snapshot(
        acc, pos.underlying_bbg_ticker if pos.asset_class == "option" else pos.bbg_ticker)
    as_of = (sl.get("pulled_at") if sl
             else state.slice_cache.get("overlay_pulled", {}).get(
                 (position_id, _scan_sig(dte_range, delta_band, expiry_type))))

    def _desc(right, strike, expiry):
        r = "C" if str(right or "").upper().startswith("C") else "P"
        exp = f"{expiry:%Y-%m-%d}" if expiry is not None else "?"
        k = f"{strike:g}" if strike is not None else "?"
        return f"{sym} {exp} {k} {r}"

    def _mid_for(p):
        return _contemporaneous_mid(p, sl if sl is not None else {"df": None})

    def _slice_note(p, held_qty, alloc, verb, noun):
        full = abs(coerce_float(p.quantity) or 0.0)
        if alloc is not None and abs(held_qty) + 1e-9 < full:
            return (f"{verb} the structure's {abs(held_qty):g}-{noun} slice; "
                    f"{full - abs(held_qty):g} {noun}s remain outside")
        return None

    # The close side of the ROLL — what the selected candidate replaces. Derived
    # from the pids the RANKING actually rolled: the rolled set on the joint
    # path, the scanned position on the single-leg path (the tick set does not
    # re-anchor a single-leg scan).
    pids = {str(p) for p in (rolled_pids or [])}
    rolled = (sorted(pids) if len(pids) > 1 else [position_id]) \
        if (cand is not None and pos.asset_class == "option") else []
    close_set: list = []
    for pid in rolled:
        p = by_id.get(pid)
        if p is None or p.asset_class != "option":
            continue
        alloc = alloc_by_pid.get(pid) if struct is not None else None
        held_qty = coerce_float(alloc) if alloc is not None \
            else (coerce_float(p.quantity) or 0.0)
        if not held_qty:
            continue
        close_set.append(tkt.close_leg(
            description=_desc(p.right, p.strike, p.expiry), held_qty=held_qty,
            mid=_mid_for(p), multiplier=coerce_float(p.multiplier) or 100.0,
            position_id=pid, right=(p.right or "").upper() or None,
            strike=coerce_float(p.strike), expiry=p.expiry,
            note=_slice_note(p, held_qty, alloc, "closes", "contract")))

    # Capture/close lines — roster legs OUTSIDE the rolled set (a leg already in
    # the roll is closed by the roll). Priced at the same contemporaneous marks;
    # run/decay is stated vs ENTRY basis, the house accounting convention.
    for pid in captures:
        if pid in rolled:
            continue
        p = by_id.get(pid)
        if p is None:
            continue
        alloc = alloc_by_pid.get(pid) if struct is not None else None
        held_qty = coerce_float(alloc) if alloc is not None \
            else (coerce_float(p.quantity) or 0.0)
        if not held_qty:
            continue
        if p.asset_class == "option":
            mult = coerce_float(p.multiplier) or 100.0
            qf, cb = coerce_float(p.quantity), coerce_float(p.cost_basis)
            entry = cb / (qf * mult) if (cb is not None and qf) else None
            close_set.append(tkt.close_leg(
                description=_desc(p.right, p.strike, p.expiry), held_qty=held_qty,
                mid=_mid_for(p), multiplier=mult, position_id=pid,
                right=(p.right or "").upper() or None,
                strike=coerce_float(p.strike), expiry=p.expiry, is_capture=True,
                entry_per_share=entry,
                note=_slice_note(p, held_qty, alloc, "captures", "contract")))
        elif p.asset_class in ("equity", "fund_etf"):
            close_set.append(tkt.close_leg(
                description=f"{sym} stock", held_qty=held_qty, mid=spot,
                multiplier=1.0, position_id=pid, is_capture=True,
                entry_per_share=_per_share_basis(p),
                note=_slice_note(p, held_qty, alloc, "captures", "share")))

    # The open side — the legs the candidate's transaction OPENS (standard
    # listed contracts at the slice mid). A fractional standard-contract
    # quantity (a non-100 held that does not map to whole contracts) is
    # disclosed, never rounded.
    open_set: list = []
    if cand is not None:
        for lg in (getattr(cand, "legs", None) or []):
            if not lg.get("opened") or lg.get("opt_type") not in ("Call", "Put"):
                continue
            qty = coerce_float(lg.get("qty")) or 0.0
            if not qty:
                continue
            note = None
            if abs(qty - round(qty)) > 1e-9:
                note = ("fractional standard-contract quantity — the held "
                        "contract's size does not map to whole standard contracts")
            right = "CALL" if lg["opt_type"] == "Call" else "PUT"
            open_set.append(tkt.open_leg(
                description=_desc(right, lg.get("K"), lg.get("expiry")), qty=qty,
                mid=lg.get("mid"), position_id=lg.get("position_id"), right=right,
                strike=coerce_float(lg.get("K")), expiry=lg.get("expiry"), note=note))

    if not close_set and not open_set:
        return None

    # The RESULTING legs after the WHOLE transaction: the candidate's resulting
    # structure minus any captured legs (or, close-only, the held legs minus the
    # captures) — priced by the same pure engine call every candidate uses.
    capture_ids = {lg.position_id for lg in close_set if lg.is_capture}
    stock_captured = any(lg.is_capture and lg.right is None for lg in close_set)
    if cand is not None:
        base = list(getattr(cand, "legs", None) or [])
    elif struct is not None:
        base = asm_legs
    else:
        try:
            from pm.risk.payoff import build_structure_payoff_legs
            asm1 = build_structure_payoff_legs(state, acc, pos)
            base = list(asm1.get("leg_dicts") or []) if not asm1.get("degraded") else []
        except Exception:
            base = []
    resulting_legs = [d for d in base
                      if d.get("position_id") not in capture_ids
                      and not (stock_captured and d.get("position_id") == "held_stock")]

    res_econ = None
    if resulting_legs and spot and spot > 0:
        try:
            from pm.candidates.generate import _build_tier1
            from pm.risk.payoff import compute_payoff
            today = date.today()
            r = compute_payoff(resulting_legs, float(spot),
                               _build_tier1(resulting_legs, today), today=today)
            res_econ = dict(r.get("economics") or {})
            if r.get("breakevens") is not None:
                res_econ["breakevens"] = [float(b) for b in r["breakevens"]]
        except Exception:
            import logging
            logging.getLogger(__name__).exception("ticket economics failed for %s",
                                                  position_id)

    # Coverage projection — the pure detector over a COPIED book, before vs
    # after, at NAME level (stock seniority needs the whole book); the flag is
    # factual role arithmetic, never advice. The resulting LABEL is scoped to
    # exactly the legs the economics price — a name-level type here would name
    # a different book than the numbers beside it.
    conv = None
    label = None
    try:
        closes_proj = [(lg.position_id, -lg.trade_qty) for lg in close_set
                       if lg.position_id is not None]
        opens_proj = [{"right": lg.right, "strike": lg.strike,
                       "expiry": lg.expiry, "qty": lg.trade_qty}
                      for lg in open_set]
        trades = getattr(acc, "trades_by_underlying", None)
        before = tkt.project_structures(account, acc.positions, trades, sym)
        projected = tkt.apply_transaction(acc.positions, closes_proj, opens_proj,
                                          underlying_symbol=sym)
        after = tkt.project_structures(account, projected, trades, sym)
        conv = tkt.coverage_conversion(tkt.uncovered_counts(before),
                                       tkt.uncovered_counts(after))
        label = tkt.resulting_label(resulting_legs, sym)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("ticket projection failed for %s",
                                              position_id)

    if not resulting_legs:
        label, res_econ = "flat", None
    elif not label:
        label = f"{len(resulting_legs)}-leg position"
    resulting = {"label": label, "economics": res_econ,
                 "legs_summary": tkt.legs_summary(resulting_legs),
                 "line": tkt.resulting_line(label, res_econ)}

    # Client-context blocks for the copy text — three labeled clocks: the
    # spot's own as-of, the extract date under the position marks, and the
    # mids' pull time (the ticket's as_of). Research provider arrives as
    # runtime DATA on the analyst record; a covered name shows it, an
    # uncovered name says so, Bloomberg-off dashes without a claimed reason.
    und_bbg = (pos.underlying_bbg_ticker if pos.asset_class == "option"
               else pos.bbg_ticker)
    if getattr(state, "bloomberg_ok", False):
        rec = (getattr(state, "analyst_data_by_ticker", None) or {}).get(und_bbg) or {}
        analyst = {"provider": rec.get("provider"),
                   "rating": rec.get("analyst_rating"),
                   "target": rec.get("analyst_target"),
                   "reason": None if rec else "not covered"}
    else:
        analyst = {"provider": None, "rating": None, "target": None, "reason": None}
    spot_kind = ("live" if (sl and sl.get("spot") is not None
                            and sl.get("spot_asof") == "live")
                 else ("snapshot" if spot is not None else None))
    spot_info = {"spot": spot, "kind": spot_kind,
                 "asof": sl.get("pulled_at") if sl else None}

    from pm.ui.deepdive.structure_economics import leg_slice, structure_economics
    extract_ts = getattr(getattr(state, "extract", None), "extract_ts", None)
    pos_legs: list = []
    coverage = None
    if struct is not None:
        e = structure_economics(struct, by_id)
        rights = {(by_id[lg.position_id].right or "").upper()
                  for lg in struct.legs
                  if by_id.get(lg.position_id) is not None
                  and by_id[lg.position_id].asset_class == "option"}
        n_ct = e.get("contracts_net")
        if len(rights) == 1 and n_ct:
            r = next(iter(rights))
            noun = (f"{'short' if n_ct < 0 else 'long'} "
                    f"{'calls' if r == 'CALL' else 'puts'}")
            coverage = tkt.coverage_line(e.get("shares_allocated"),
                                         e.get("shares_total"), n_ct, noun)
        for lg in struct.legs:
            p = by_id.get(lg.position_id)
            cost, mv, pnl, _prem, _ok = leg_slice(lg, p)
            pct = (pnl / abs(cost)) if (pnl is not None and cost) else None
            if p is not None and p.asset_class == "option":
                contract = (f"{lg.allocated_qty:+,.0f}  "
                            f"{_desc(p.right, p.strike, p.expiry)}")
            else:
                contract = f"{(lg.allocated_qty or 0):,.0f} sh"
            pos_legs.append({"label": (lg.role or "leg").replace("_", " "),
                             "contract": contract, "basis": cost, "mv": mv,
                             "pnl": pnl, "pct": pct})
    else:
        cost = coerce_float(getattr(pos, "cost_basis", None))
        mv = coerce_float(getattr(pos, "market_value", None))
        pnl = coerce_float(getattr(pos, "unrealized_pnl", None))
        if pnl is None and mv is not None and cost is not None:
            pnl = mv - cost
        pct = (pnl / abs(cost)) if (pnl is not None and cost) else None
        q = coerce_float(pos.quantity) or 0.0
        if pos.asset_class == "option":
            kind = "call" if str(pos.right or "").upper().startswith("C") else "put"
            leg_label = f"{'short' if q < 0 else 'long'} {kind}"
            contract = f"{q:+,.0f}  {_desc(pos.right, pos.strike, pos.expiry)}"
        else:
            leg_label = f"{'short' if q < 0 else 'long'} stock"
            contract = f"{q:,.0f} sh"
        pos_legs.append({"label": leg_label, "contract": contract, "basis": cost,
                         "mv": mv, "pnl": pnl, "pct": pct})
    position_block = {
        "asof": f"{extract_ts:%Y-%m-%d}" if extract_ts else None,
        "legs": pos_legs, "coverage": coverage}

    return tkt.assemble(close_set, open_set, account=account, underlier=sym,
                        as_of=as_of, resulting=resulting, conversion=conv,
                        warnings=warnings, analyst=analyst, spot_info=spot_info,
                        position_block=position_block)
