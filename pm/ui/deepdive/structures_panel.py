"""Tab 2 — By Structure view: the structures grid rows/columns + the structure
detail modal.

The holdings table toggles between By Position (the per-leg grid) and By Structure
(one row per detected structure with Tier-1 economics). Both are AG-Grid Community;
expand-to-legs is faked with indented leg rows toggled by a caret cell, and a row
click opens the shared modal — the structure's legs, Tier-1 economics, and the
Confirm / Reject / Edit / Choose affordances. Those affordances reuse the existing
buttons + ``state_access.resolve_structure`` (the single write path); this module
only renders.

Pure presentation: reads ``AccountState.structures`` + ``Position`` fields and the
Tier-1 aggregation in ``structure_economics``; never recomputes. Tier-2 economics
(breakeven, max P/L, Greeks, theoretical value) await layer-2 pricing and render as
``PENDING_PRICING`` markers.
"""
from __future__ import annotations

from datetime import date

from dash import dcc, html

from pm.insight import structures as S
from pm.store.suppression_store import is_active
from pm.ui.blotter.grid import format_position_descriptor
from pm.ui.deepdive.formatters import MONEY_FULL_FMT, QTY_FMT, SIGNED_COLOR_STYLE
from pm.ui.deepdive.structure_economics import (
    PENDING_PRICING,
    leg_slice,
    structure_economics,
)

_PRIMARY = {S.COVERED_CALL, S.COLLAR, S.VERTICAL, S.COVERED_PUT,
            S.CASH_SECURED_PUT, S.STRADDLE, S.STRANGLE}
_SUB = {S.RESIDUAL_LONG, S.NAKED_EXCESS_SHORT_CALL}
_CONFIRMED = ("confirmed", "edited")
_BAND_LABEL = {S.HIGH: "High", S.MEDIUM: "Medium", S.LOW_AMBIGUOUS: "Low-Ambiguous"}
_TYPE_LABEL = {
    S.COVERED_CALL: "Covered call", S.COLLAR: "Collar", S.VERTICAL: "Vertical",
    S.COVERED_PUT: "Covered put", S.CASH_SECURED_PUT: "Cash-secured put",
    S.STRADDLE: "Straddle", S.STRANGLE: "Strangle",
    S.RESIDUAL_LONG: "Uncovered long (residual)",
    S.NAKED_EXCESS_SHORT_CALL: "Naked-excess short call",
}
_STATUS_LABEL = {"proposed": "Proposed", "confirmed": "Confirmed",
                 "edited": "Confirmed (edited)", "rejected": "Rejected"}


# ---------------------------------------------------------------------------
# Small formatters (Python-side, for the modal)
# ---------------------------------------------------------------------------
def _money(v) -> str:
    if v is None:
        return "—"
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def _qty(v) -> str:
    return "—" if v is None else f"{v:g}"


def _expiry(d) -> str:
    try:
        return d.strftime("%b-%y")
    except Exception:
        return str(d)


def _strikes_str(strikes) -> str:
    return " / ".join(f"{x:g}" for x in strikes) if strikes else ""


def _expiries_str(expiries) -> str:
    return " / ".join(_expiry(x) for x in expiries) if expiries else ""


def _nearest_dte(expiries):
    """Integer days to the nearest leg expiry — the desk reads expiry in DTE."""
    if not expiries:
        return None
    try:
        return (min(expiries) - date.today()).days
    except Exception:
        return None


# ---------------------------------------------------------------------------
# By Structure grid — columns + rows
# ---------------------------------------------------------------------------
def build_structure_columns() -> list[dict]:
    """Columns for the By Structure grid, leading with identity + economics and
    demoting workflow metadata (Band / Status) to the right. Tier-1 economics are
    real; the single 'Pricing (T2)' column flags the layer-2 economics as pending
    (the modal breaks them out). The leading caret toggles in-grid leg expansion.

    Numerics mirror the By Position grid (shared formatters, same green/red P&L).
    Community filters only: numbers get a number filter, the multi-value Strikes /
    Expiry text columns a text filter (they render '440 / 460', 'Jun-26'). Filtering
    works cleanly in the default collapsed view; with legs expanded a filter can hide
    a parent and leave its leg rows orphaned — structure-aware filtering is a separate
    piece of work."""
    return [
        {"field": "caret", "headerName": "", "width": 40, "sortable": False,
         "filter": False, "cellClass": "struct-caret-cell"},
        {"field": "label", "headerName": "Structure", "flex": 2, "minWidth": 240,
         "filter": "agTextColumnFilter", "tooltipField": "label", "cellClass": "dd-cell-ellipsis"},
        {"field": "net_qty", "headerName": "Net Qty", "width": 92,
         "type": "rightAligned", "valueFormatter": QTY_FMT, "filter": "agNumberColumnFilter"},
        {"field": "strikes", "headerName": "Strikes", "width": 110,
         "filter": "agTextColumnFilter"},
        {"field": "expiry", "headerName": "Expiry", "width": 96,
         "filter": "agTextColumnFilter"},
        {"field": "dte", "headerName": "DTE", "width": 70, "type": "rightAligned",
         "valueFormatter": {"function": "params.value == null ? '' : params.value"},
         "filter": "agNumberColumnFilter"},
        {"field": "net_debit_credit", "headerName": "Net Deb/Cr", "width": 124,
         "type": "rightAligned", "valueFormatter": MONEY_FULL_FMT, "filter": "agNumberColumnFilter"},
        {"field": "net_pnl", "headerName": "Net P&L", "width": 120,
         "type": "rightAligned", "valueFormatter": MONEY_FULL_FMT,
         "cellStyle": SIGNED_COLOR_STYLE, "filter": "agNumberColumnFilter"},
        {"field": "net_premium", "headerName": "Net Premium", "width": 124,
         "type": "rightAligned", "valueFormatter": MONEY_FULL_FMT, "filter": "agNumberColumnFilter"},
        {"field": "t2_pricing", "headerName": "Breakeven", "width": 124,
         "filter": "agTextColumnFilter", "tooltipField": "t2_tooltip"},
        {"field": "band", "headerName": "Band", "width": 116, "filter": "agTextColumnFilter"},
        {"field": "status", "headerName": "Status", "width": 130, "filter": "agTextColumnFilter"},
    ]


def _breakeven_str(t2_entry) -> str:
    """Format a structure's zero-shock breakeven(s) for the grid cell, '/'-joined.
    '—' when no Tier-2 (unpriceable / no spot). The stored breakevens are analytic
    (complete by construction), so an empty list truthfully means the curve never
    crosses zero and renders as the curve's constant sign — 'always profitable' /
    'always loss' — never an ambiguous 'none'. A multi-expiry structure carries a
    dagger; the cell tooltip names the evaluation horizon."""
    if not t2_entry:
        return "—"
    bes = t2_entry.get("breakevens") or []
    if not bes:
        if t2_entry.get("always_profitable"):
            s = "always profitable"
        elif t2_entry.get("always_loss"):
            s = "always loss"
        else:
            s = "—"
    else:
        s = " / ".join(f"{b:,.2f}" for b in bes)
    if t2_entry.get("multi_expiry"):
        s += " †"
    return s


def _breakeven_tooltip(t2_entry) -> str:
    """Cell tooltip for the Breakeven column: names the multi-expiry horizon."""
    if t2_entry and t2_entry.get("multi_expiry"):
        when = t2_entry.get("eval_date") or "the nearest leg expiry"
        return (f"† multi-expiry structure — evaluated at nearest expiry {when}; "
                "far legs at model value on that date.")
    return ""


def _structure_row(s, by_id, expanded: set, expandable: bool = True, t2_map=None) -> dict:
    e = structure_economics(s, by_id)
    return {
        "_row_id": f"structure::{s.structure_id}",
        "_kind": "structure",
        "_structure_id": s.structure_id,
        "_expandable": expandable,
        "caret": ("▾" if s.structure_id in expanded else "▸") if expandable else "",
        "label": f"{_TYPE_LABEL.get(s.type, s.type)} · {s.underlying}",
        "band": _BAND_LABEL.get(s.confidence_band, s.confidence_band),
        "status": _STATUS_LABEL.get(s.status, s.status),
        "strikes": _strikes_str(e["strikes"]),
        "expiry": _expiries_str(e["expiries"]),
        "dte": _nearest_dte(e["expiries"]),
        "net_qty": e["net_quantity"],
        "net_debit_credit": e["net_debit_credit"],
        "net_pnl": e["net_pnl"],
        "net_premium": e["net_premium"],
        "t2_pricing": _breakeven_str((t2_map or {}).get(s.structure_id)),
        "t2_tooltip": _breakeven_tooltip((t2_map or {}).get(s.structure_id)),
    }


def _leg_row(s, leg, by_id, idx: int) -> dict:
    pos = by_id.get(leg.position_id)
    cost, _mval, pnl, premium, _ok = leg_slice(leg, pos)
    desc = format_position_descriptor(pos) if pos is not None else leg.position_id
    return {
        "_row_id": f"leg::{s.structure_id}::{leg.position_id}::{idx}",
        "_kind": "leg",
        "_structure_id": s.structure_id,
        "caret": "",
        "label": f" ↳ {leg.role.replace('_', ' ')}: {desc}",
        "band": "", "status": "",
        "strikes": "", "expiry": "", "dte": None,
        "net_qty": leg.allocated_qty,
        "net_debit_credit": cost,
        "net_pnl": pnl,
        "net_premium": premium,
        "t2_pricing": "",
    }


def _substructure_row(sub, by_id, t2_map=None) -> dict:
    e = structure_economics(sub, by_id)
    return {
        "_row_id": f"sub::{sub.structure_id}",
        "_kind": "substructure",
        "_structure_id": sub.structure_id,
        "caret": "",
        "label": f" ↳ {_TYPE_LABEL.get(sub.type, sub.type)}",
        "band": _BAND_LABEL.get(sub.confidence_band, sub.confidence_band),
        "status": _STATUS_LABEL.get(sub.status, sub.status),
        "strikes": _strikes_str(e["strikes"]),
        "expiry": _expiries_str(e["expiries"]),
        "dte": _nearest_dte(e["expiries"]),
        "net_qty": e["net_quantity"],
        "net_debit_credit": e["net_debit_credit"],
        "net_pnl": e["net_pnl"],
        "net_premium": e["net_premium"],
        "t2_pricing": _breakeven_str((t2_map or {}).get(sub.structure_id)),
        "t2_tooltip": _breakeven_tooltip((t2_map or {}).get(sub.structure_id)),
    }


def _has_tier1_attention(s, t1_sids, t1_pids) -> bool:
    """A structure draws attention if a tier-1 fire targets it directly (by
    structure_id) or lands on any of its leg positions."""
    if s.structure_id in t1_sids:
        return True
    return any(leg.position_id in t1_pids for leg in s.legs)


def build_structure_rows(account_state, state, expanded_sids=None) -> list[dict]:
    """Flat rows for the By Structure grid: contention rows first, then the detected
    structures ordered by attention — a tier-1 fire (on the structure or any of its
    legs) floats it up, then largest absolute Net P&L. Each structure's indented legs
    and residual/naked sub-rows ride beneath it when expanded. Structures only — a
    large *unstructured* position shows in the By Position view, not here. Row ids are
    unique/stable."""
    expanded = set(expanded_sids or [])
    by_id = {p.position_id: p for p in account_state.positions}
    t2_map = getattr(account_state, "structure_tier2", None) or {}
    structures = list(account_state.structures or [])
    rows: list[dict] = []

    # Tier-1 attention set, built once for the account: structures a T1 fire targets
    # by structure_id, and positions a T1 fire lands on (a structure inherits attention
    # from a fire on any of its legs). Read-only over already-loaded fires; a
    # suppressed/snoozed T1 fire no longer floats its structure.
    fires = [f for f in (getattr(account_state, "fires", []) or []) if is_active(f)]
    t1_sids = {f.structure_id for f in fires if f.tier == 1 and f.structure_id}
    t1_pids = {f.position_id for f in fires if f.tier == 1}

    # 1) Contention groups — one row each (resolved → the chosen reading).
    groups: dict = {}
    for s in structures:
        if s.contention_group:
            groups.setdefault(s.contention_group, []).append(s)
    for group, alts in sorted(groups.items()):
        chosen = next((a for a in alts if a.status in _CONFIRMED), None)
        if chosen is not None:
            rows.append(_structure_row(chosen, by_id, expanded, t2_map=t2_map))
            if chosen.structure_id in expanded:
                rows.extend(_leg_row(chosen, leg, by_id, i) for i, leg in enumerate(chosen.legs))
        else:
            rows.append({
                # carry a structure_id (not the group) so the click handler can
                # parse it from rowId; render_structure_detail derives the group.
                "_row_id": f"contention::{alts[0].structure_id}",
                "_kind": "contention",
                "_structure_id": alts[0].structure_id,
                "_group": group,
                "caret": "",
                "label": f"Contention · {alts[0].underlying}",
                "band": _BAND_LABEL.get(S.LOW_AMBIGUOUS),
                "status": "Needs your choice",
                "strikes": "", "expiry": "", "dte": None, "net_qty": None,
                "net_debit_credit": None, "net_pnl": None, "net_premium": None,
                "t2_pricing": "—",
            })

    # 2) Primary structures (not contended, not rejected), ordered by attention.
    #    Reorder whole structure BLOCKS — not an AG-Grid row sort — so each block's
    #    legs and residual/naked sub-rows stay attached beneath it when expanded.
    primaries = [s for s in structures
                 if not s.contention_group and s.status != "rejected" and s.type in _PRIMARY]

    def _attention_key(s):
        npnl = structure_economics(s, by_id)["net_pnl"]
        return (0 if _has_tier1_attention(s, t1_sids, t1_pids) else 1,
                -abs(npnl) if npnl is not None else 0.0)

    for s in sorted(primaries, key=_attention_key):
        rows.append(_structure_row(s, by_id, expanded, t2_map=t2_map))
        if s.structure_id in expanded:
            rows.extend(_leg_row(s, leg, by_id, i) for i, leg in enumerate(s.legs))
            for sub in structures:
                if (sub.type in _SUB and sub.underlying == s.underlying
                        and sub.status != "rejected"):
                    rows.append(_substructure_row(sub, by_id, t2_map=t2_map))

    for r in rows:
        r["_account"] = account_state.account
    return rows


# ---------------------------------------------------------------------------
# Structure detail modal — Tier-1 + legs + affordances (reuses resolve_structure)
# ---------------------------------------------------------------------------
def _chip(band) -> html.Span:
    return html.Span(_BAND_LABEL.get(band, band), className=f"struct-chip struct-chip-{band}")


def _econ_item(label: str, value: str, pending: bool = False) -> html.Div:
    return html.Div(className="struct-econ-item", children=[
        html.Span(label, className="struct-econ-label"),
        html.Span(value, className="struct-econ-value"
                  + (" struct-econ-pending" if pending else "")),
    ])


def _tier1_block(e: dict) -> html.Div:
    return html.Div(className="struct-econ", children=[
        _econ_item("Net debit/credit", _money(e["net_debit_credit"])),
        _econ_item("Net P&L", _money(e["net_pnl"])),
        _econ_item("Net premium", _money(e["net_premium"])),
        _econ_item("Net quantity", _qty(e["net_quantity"])),
        _econ_item("Strikes", _strikes_str(e["strikes"]) or "—"),
        _econ_item("Expiries", _expiries_str(e["expiries"]) or "—"),
    ])


def _tier2_block() -> html.Div:
    return html.Div(className="struct-econ struct-econ-t2", children=[
        _econ_item("Breakeven", PENDING_PRICING, pending=True),
        _econ_item("Max profit/loss", PENDING_PRICING, pending=True),
        _econ_item("Greeks", PENDING_PRICING, pending=True),
        _econ_item("Theoretical value", PENDING_PRICING, pending=True),
    ])


def _legs_block(account, s, by_id, removable: bool) -> html.Div:
    rows = []
    for leg in s.legs:
        pos = by_id.get(leg.position_id)
        cost, _mv, pnl, _prem, _ok = leg_slice(leg, pos)
        desc = format_position_descriptor(pos) if pos is not None else leg.position_id
        children = [
            html.Span(leg.role.replace("_", " "), className="struct-leg-role"),
            html.Span(desc, className="struct-leg-id"),
            html.Span(f"{leg.allocated_qty:g}", className="struct-leg-qty"),
            html.Span(_money(cost), className="struct-leg-qty"),
            html.Span(_money(pnl), className="struct-leg-qty"),
        ]
        if removable and len(s.legs) > 1:
            children.append(html.Button(
                "remove", className="struct-leg-remove",
                id={"type": "struct-removeleg", "account": account,
                    "sid": s.structure_id, "leg": leg.position_id}, n_clicks=0))
        rows.append(html.Div(children, className="struct-leg-row"))
    return html.Div(rows, className="struct-legs")


def _sub_block(account, structures, underlying, by_id) -> list:
    out = []
    for sub in structures:
        if sub.type in _SUB and sub.underlying == underlying and sub.status != "rejected":
            e = structure_economics(sub, by_id)
            out.append(html.Div(className="struct-subrow", children=[
                html.Span(_TYPE_LABEL.get(sub.type, sub.type), className="struct-subrow-label"),
                html.Span(f"net {_money(e['net_debit_credit'])} · P&L {_money(e['net_pnl'])}",
                          className="struct-subrow-note"),
            ]))
    return out


def _single_detail(account, s, structures, by_id) -> html.Div:
    confirmed = s.status in _CONFIRMED
    e = structure_economics(s, by_id)
    head = [
        html.Span(f"{_TYPE_LABEL.get(s.type, s.type)} · {s.underlying}",
                  className="struct-card-title"),
        _chip(s.confidence_band),
        html.Span(_STATUS_LABEL.get(s.status, s.status),
                  className="struct-confirmed-by" if confirmed else "struct-status-proposed"),
    ]
    if confirmed and s.resolved_at:
        head.append(html.Span(f"on {s.resolved_at[:10]}", className="struct-confirmed-by"))

    if confirmed:
        actions = [html.Button("Undo", className="struct-btn",
                               id={"type": "struct-reject", "account": account,
                                   "sid": s.structure_id}, n_clicks=0)]
    else:
        actions = [
            html.Button("Confirm", className="struct-btn struct-btn-primary",
                        id={"type": "struct-confirm", "account": account,
                            "sid": s.structure_id}, n_clicks=0),
            html.Button("Reject", className="struct-btn",
                        id={"type": "struct-reject", "account": account,
                            "sid": s.structure_id}, n_clicks=0),
            html.Span("Edit: remove a leg below", className="struct-edit-hint"),
        ]

    return html.Div(className="drawer-content struct-detail", children=[
        html.Div(head, className="struct-card-head"),
        html.Div("Tier-1 economics", className="drawer-section-label"),
        _tier1_block(e),
        html.Div("Tier-2 — pending layer-2 pricing", className="drawer-section-label"),
        _tier2_block(),
        html.Div("Legs", className="drawer-section-label"),
        _legs_block(account, s, by_id, removable=not confirmed),
        *_sub_block(account, structures, s.underlying, by_id),
        html.Div(actions, className="struct-actions"),
    ])


def _contention_detail(account, target, structures, by_id) -> html.Div:
    group = target.contention_group
    alts = [a for a in structures if a.contention_group == group]
    options = [{"label": f"{_TYPE_LABEL.get(a.type, a.type)} — "
                         f"net {_money(structure_economics(a, by_id)['net_debit_credit'])}",
                "value": a.structure_id} for a in alts]
    return html.Div(className="drawer-content struct-detail", children=[
        html.Div([html.Span(f"Contention · {target.underlying}", className="struct-card-title"),
                  _chip(S.LOW_AMBIGUOUS),
                  html.Span("these legs read two ways — choose one",
                            className="struct-contention-note")],
                 className="struct-card-head"),
        dcc.RadioItems(id={"type": "struct-radio", "group": group}, options=options,
                       value=None, className="struct-radio"),
        html.Div([
            html.Button("Confirm choice", className="struct-btn struct-btn-primary",
                        id={"type": "struct-choose", "account": account, "group": group},
                        n_clicks=0),
            html.Button("Reject both", className="struct-btn",
                        id={"type": "struct-reject", "account": account,
                            "sid": alts[0].structure_id}, n_clicks=0),
        ], className="struct-actions"),
    ])


def render_structure_detail(account, structure_id, state) -> html.Div:
    """Modal body for a structure row: Tier-1 economics, legs, Tier-2 placeholders,
    and the Confirm / Reject / Edit / Choose affordances (which call the existing
    resolve_structure). A contended structure renders the alternative chooser —
    explicit choice required, no one-click confirm."""
    acc = state.accounts.get(account) if state else None
    if acc is None:
        return html.Div("No account selected.", className="trace-muted")
    structures = list(acc.structures or [])
    target = next((s for s in structures if s.structure_id == structure_id), None)
    if target is None:
        return html.Div("Structure no longer present.", className="trace-muted")
    by_id = {p.position_id: p for p in acc.positions}
    if target.contention_group and target.status not in _CONFIRMED:
        return _contention_detail(account, target, structures, by_id)
    return _single_detail(account, target, structures, by_id)
