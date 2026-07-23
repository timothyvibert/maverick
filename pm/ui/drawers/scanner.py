"""Scanner drawer — the order-entry surface for rolling a position or a structure.

Layout (the approved order-entry idiom): identity header (ticker · spot · day move ·
account · as-of) → MANAGING (the structure's legs as a roster grid whose checkboxes
ARE the rolled set, plus the structure's stored current economics) → SCAN (objective
tokens, the shared DTE and |Δ| range dials, Scan) → RICHNESS (the full-width smile:
in-fit dots filled, filtered hollow, the fitted line dashed, the selected candidate
ringed; IV-rank · IV/RV · fit R² beneath) → CANDIDATES (the ranked answer set as a
grid, the full chain collapsed behind a "+N in chain" line, the widen-window pull) →
CURRENT VS ADJUSTED (kept legs at entry, new legs at mid — one accounting basis) →
PAYOFF (the adjusted structure at expiry and today) → the shock dials.

One rolled leg scans the single-leg path; two or more drive the joint path (one
common new expiry, spreads keep their width). Render-only: every number comes from
``state_access`` (the sanctioned on-demand pull + the cached rankings); the view
never prices or ranks anything itself. Honest states render — a joint roll with no
admissible target, a truncated enumeration, a partial-slice remainder, a degraded
fit — they are never hidden. All colour is the shared --pm-*/--pos/--neg tokens.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import dash_ag_grid as dag
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from pm.candidates.generate import _COSTLESS_PER_SHARE
from pm.ui import state_access as sa
from pm.ui.deepdive.aggregations import _fmt_money
from pm.ui.dial_sync import register_dial_sync, register_range_dial_sync

# Objective labels + the order the tokens read in (the recommender seed picks the
# default, the tokens override). Any objective the ranker emits still shows.
_OBJ_LABEL = {
    "roll-up-out": "Roll away & out",
    "costless": "Costless · near · max cap",
    "roll-for-credit": "Roll for credit",
    "defend-cut-delta": "Cut Δ",
    "extend-duration": "Extend duration",
    "max-premium": "Max premium",
    "add-hedge": "Add hedge",
}
_OBJ_ORDER = ["roll-up-out", "costless", "roll-for-credit", "defend-cut-delta",
              "extend-duration", "max-premium", "add-hedge"]

# The recommender's action -> the default objective token (action-level; the rule_id
# sub-splits stay a later refinement). An unmapped / neutral label (CLOSE,
# HARVEST_THETA, TRIM, ADD, MONITOR) opens on the first present objective.
_SEED = {
    "ROLL_OUT": "roll-for-credit",
    "ROLL_OUT_AND_DOWN": "defend-cut-delta",
    "ROLL_UP_AND_OUT": "defend-cut-delta",
    "ADD_OVERLAY": "max-premium",
    "ADD_HEDGE": "defend-cut-delta",
}

_HONESTY = ("Ranked by objective-fit and client-fit — advisory, not an order. Economics "
            "and PoP use the live American pricer; P(assign) is the short leg's own |Δ|, "
            "a proxy; IV+pp is richness vs this chain's own fitted smile (shape) and "
            "IV-rank the 52-week level; the chain is on-demand (see the as-of), not "
            "streaming.")

_TOP_N = 5              # the ranked answer set above the collapsed chain
_CHARCOAL = "#2B2B2B"
_GREY5 = "#6E6E6E"
_GRID = "#E8E8E8"
_HOLLOW = "#B5B5B5"
_AMBER = "#B7791F"
_FONT = {"family": "Segoe UI, Helvetica Neue, Arial, sans-serif",
         "size": 11, "color": _CHARCOAL}


# ---------------------------------------------------------------------------
# Small cell formatters
# ---------------------------------------------------------------------------

def _money(v) -> str:
    return _fmt_money(v) if v is not None else "—"


def _sign(v) -> str:
    if v is None or v == 0:
        return ""
    return "scanner-pos" if v > 0 else "scanner-neg"


def _maxup(e) -> str:
    return "∞" if e.get("unbounded_gain") else _money(e.get("max_profit"))


def _maxdn(e) -> str:
    return "−∞" if e.get("unbounded_loss") else _money(e.get("max_loss"))


def _pop(v) -> str:
    return f"{v * 100:.0f}%" if v is not None else "—"


def _delta(v) -> str:
    return f"{v:+.2f}" if v is not None else "—"


def _px(v) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _iv(v) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _int(v) -> str:
    return f"{int(v):,}" if v is not None else "—"


def _strike(v) -> str:
    return f"{v:g}" if v is not None else "—"


def _exp(d) -> str:
    try:
        return d.strftime("%d-%b-%y")
    except Exception:
        return "—"


def _primary_leg(c):
    """The candidate's contract — the short option leg the transaction OPENS (the
    roll/write target), else its first opened option leg. Kept sibling legs of an
    enclosing structure never key the row. None for a stock-only candidate."""
    opts = [lg for lg in (getattr(c, "legs", None) or []) if lg.get("opt_type") in ("Call", "Put")]
    pool = [lg for lg in opts if lg.get("opened")] or opts
    shorts = [lg for lg in pool if (lg.get("qty") or 0) < 0]
    return (shorts or pool or [None])[0]


def _is_costless(c) -> bool:
    """True when the candidate's TRANSACTION is (near-)costless — reads the
    roll/overlay's own net, never the resulting position's entry cost."""
    nc = getattr(c, "net_credit", None)
    if nc is None:
        return False
    opts = [lg for lg in (getattr(c, "legs", None) or [])
            if lg.get("opt_type") in ("Call", "Put")]
    pool = [lg for lg in opts if lg.get("opened")] or opts
    contracts = sum(abs(int(lg.get("qty") or 0)) for lg in pool) or 1
    return abs(nc) <= _COSTLESS_PER_SHARE * 100 * max(contracts, 1)


def _stamp(pulled_at, kind, spot_asof) -> str:
    if pulled_at is None:
        return "spot from morning snapshot" if kind != "option" else "—"
    mins = max(int((datetime.now() - pulled_at).total_seconds() // 60), 0)
    ago = "just now" if mins == 0 else f"{mins} min ago"
    tail = {"live": " · spot live", "snapshot": " · spot from morning snapshot"}.get(spot_asof, "")
    return f"pulled {ago} · this name only{tail}"


def _seed_objective(account, position_id, objectives) -> str:
    """The default token: the held option's moneyness (an ITM short leads with the
    away roll, OTM with premium), then the recommender's action, then the first
    present objective. Best-effort — any gap falls back cleanly."""
    try:
        state = sa.get_state()
        acc = state.accounts.get(account) if state else None
        pos = sa.position_by_id(state, account, position_id) if state else None
        if acc is not None and pos is not None:
            right = (pos.right or "").upper()
            if (getattr(pos, "asset_class", None) == "option" and pos.strike is not None
                    and right in ("CALL", "PUT") and (pos.quantity or 0) < 0):
                spot = sa._spot_from_snapshot(acc, getattr(pos, "underlying_bbg_ticker", None))
                if spot is not None:
                    itm = spot > pos.strike if right == "CALL" else spot < pos.strike
                    lead = "roll-up-out" if itm else "max-premium"
                    if lead in objectives:
                        return lead
            tickers = {t for t in (getattr(pos, "bbg_ticker", None),
                                   getattr(pos, "underlying_bbg_ticker", None)) if t}
            for rec in (getattr(acc, "recommendations", None) or []):
                if getattr(rec, "position_id", None) in tickers:
                    seed = _SEED.get(getattr(rec, "action", None))
                    return seed if seed in objectives else objectives[0]
    except Exception:
        pass
    return objectives[0]


def _ordered_objectives(ranked) -> list:
    present = [o for o in _OBJ_ORDER if ranked.get(o)]
    present += [o for o in ranked if o not in _OBJ_ORDER and ranked.get(o)]
    return present


def _tokens(ranked, objectives, active) -> list:
    out = []
    for o in objectives:
        n = len(ranked.get(o) or [])
        cls = "scanner-tok" + (" scanner-tok-on" if o == active else "")
        out.append(html.Button(f"{_OBJ_LABEL.get(o, o)} · {n}",
                               id={"type": "scanner-obj", "obj": o}, n_clicks=0,
                               className=cls))
    return out


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

_GRID_OPTS = {"rowHeight": 28, "headerHeight": 32, "suppressCellFocus": True,
              "enableCellTextSelection": True, "ensureDomOrder": True,
              "domLayout": "autoHeight"}

_ROSTER_COLS = [
    {"field": "role", "headerName": "Leg", "width": 110,
     "cellClass": "scanner-roster-role"},
    {"field": "contract", "headerName": "Contract", "flex": 2, "minWidth": 130,
     "tooltipField": "contract"},
    {"field": "qty", "headerName": "Qty", "width": 70, "type": "rightAligned"},
    {"field": "dte", "headerName": "DTE", "width": 70, "type": "rightAligned"},
    {"field": "delta", "headerName": "Δ", "width": 80, "type": "rightAligned"},
    {"field": "mid", "headerName": "Mid", "width": 80, "type": "rightAligned",
     "headerTooltip": "morning-snapshot mid"},
    {"field": "p_assign", "headerName": "P(assign)", "width": 96, "type": "rightAligned",
     "headerTooltip": "the short leg's own |Δ| — a proxy, not a model probability",
     "checkboxSelection": False},
    {"field": "close", "headerName": "Close", "width": 62,
     "cellClass": "scanner-close-cell",
     "headerTooltip": "mark a leg to capture/close in the ticket — priced at its "
                      "contemporaneous mid, run/decay shown vs entry basis; a leg "
                      "already ticked to roll is closed by the roll itself"},
]


def _roster_grid():
    cols = [{"headerName": "Roll", "checkboxSelection": True, "width": 62,
             "headerTooltip": "legs ticked here roll together — one is the "
                              "single-leg scan, two or more roll jointly to one "
                              "shared new expiry"}] + _ROSTER_COLS
    return dag.AgGrid(
        id="scanner-roster-grid", className="ag-theme-balham blotter-grid",
        columnDefs=cols, rowData=[],
        getRowId="params.data.position_id",
        dashGridOptions={**_GRID_OPTS, "rowSelection": "multiple",
                         "suppressRowClickSelection": True},
        style={"width": "100%"},
    )


def _cand_grid():
    return dag.AgGrid(
        id="scanner-cand-grid", className="ag-theme-balham blotter-grid",
        columnDefs=_cand_cols("New legs", "BE"), rowData=[],
        getRowId="params.data.row_id",
        dashGridOptions={**_GRID_OPTS, "rowSelection": "single"},
        style={"width": "100%"},
    )


def _cand_cols(move_hdr: str, be_hdr: str) -> list:
    return [
        {"field": "rank", "headerName": "#", "width": 58,
         "tooltipField": "flags",
         "cellClass": "scanner-rank-cell"},
        {"field": "move", "headerName": move_hdr, "flex": 2, "minWidth": 170,
         "tooltipField": "reasons"},
        {"field": "dte", "headerName": "DTE", "width": 66, "type": "rightAligned"},
        {"field": "net", "headerName": "Net", "width": 92, "type": "rightAligned",
         "headerTooltip": "the transaction's own net cash — credit positive",
         "cellClass": {"function": "params.data.net_sign"}},
        {"field": "tag", "headerName": "", "width": 84,
         "cellClass": "scanner-tag-cell"},
        {"field": "ivpp", "headerName": "IV+pp", "width": 78, "type": "rightAligned",
         "headerTooltip": "the opened short leg's IV minus the fitted smile, vol points"},
        {"field": "maxp", "headerName": "Max profit", "width": 130, "type": "rightAligned",
         "headerTooltip": "resulting structure · $ and % of NAV"},
        {"field": "maxl", "headerName": "Max loss", "width": 100, "type": "rightAligned",
         "cellClass": {"function": "params.data.maxl_cls"}},
        {"field": "be", "headerName": be_hdr, "width": 88, "type": "rightAligned",
         "headerTooltip": "the breakeven on the rolled side — upper for call rolls, "
                          "lower for put rolls, both when mixed"},
        {"field": "passign", "headerName": "P(assign)", "width": 92, "type": "rightAligned",
         "headerTooltip": "per opened short leg, its own |Δ| — two shorts show both"},
        {"field": "ndelta", "headerName": "Δ", "width": 104, "type": "rightAligned",
         "headerTooltip": "resulting structure net delta, share-equivalents"},
    ]


_CHAIN_COLS = [
    {"field": "strike", "headerName": "Strike", "width": 84, "type": "rightAligned"},
    {"field": "expiry", "headerName": "Exp", "width": 100},
    {"field": "right", "headerName": "C/P", "width": 56},
    {"field": "bid", "headerName": "Bid", "width": 76, "type": "rightAligned"},
    {"field": "ask", "headerName": "Ask", "width": 76, "type": "rightAligned"},
    {"field": "mid", "headerName": "Mid", "width": 76, "type": "rightAligned"},
    {"field": "iv", "headerName": "IV", "width": 70, "type": "rightAligned"},
    {"field": "delta", "headerName": "Δ", "width": 76, "type": "rightAligned"},
    {"field": "oi", "headerName": "OI", "width": 84, "type": "rightAligned"},
    {"field": "fit", "headerName": "Fit", "flex": 2, "minWidth": 90,
     "headerTooltip": "in the smile regression, or the exclusion reason class"},
]


def _chain_grid():
    return dag.AgGrid(
        id="scanner-chain-grid", className="ag-theme-balham blotter-grid",
        columnDefs=_CHAIN_COLS, rowData=[], dashGridOptions=dict(_GRID_OPTS),
        style={"width": "100%"},
    )


# ---------------------------------------------------------------------------
# Figures (house-token Plotly; explicit inline heights — the responsive:True
# inline height:100% race against a CSS pin is closed by pinning the STYLE)
# ---------------------------------------------------------------------------

def _fig_base(height):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=_FONT, margin=dict(l=44, r=14, t=8, b=34),
                      showlegend=False, hovermode="closest", height=height)
    return fig


def _msg_fig(text, height=230):
    fig = _fig_base(height)
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text=text, showarrow=False,
                                        font={**_FONT, "color": _GREY5})])
    return fig


def _smile_fig(view, expiry_iso, selected) -> "object":
    """The Richness chart: dots per snapshotted contract at the shown expiry
    (filled = in the fit), the fitted line DASHED, spot + each rolled strike as
    dotted verticals, the selected candidate ringed amber with its label. A
    degraded fit draws dots and says so — never a fake line."""
    import plotly.graph_objects as go
    contracts = view.get("contracts") or []
    pts = [c for c in contracts
           if str(c.get("expiry")) == str(expiry_iso)
           and c.get("iv") is not None and c.get("strike")]
    if len(pts) < 4:
        return _msg_fig("Too few listed strikes to draw a smile for this expiry — "
                        "see the chain below.")
    fig = _fig_base(230)
    filled = [c for c in pts if c.get("in_fit")]
    hollow = [c for c in pts if not c.get("in_fit")]
    if filled:
        fig.add_scatter(x=[c["strike"] for c in filled], y=[c["iv"] for c in filled],
                        mode="markers", marker=dict(color=_CHARCOAL, size=6),
                        hovertemplate="%{x} · IV %{y:.1f}%<extra></extra>")
    if hollow:
        fig.add_scatter(x=[c["strike"] for c in hollow], y=[c["iv"] for c in hollow],
                        mode="markers",
                        marker=dict(color="rgba(0,0,0,0)", size=6,
                                    line=dict(color=_HOLLOW, width=1)),
                        hovertemplate="%{x} · filtered from fit<extra></extra>")
    surface, spot = view.get("surface"), view.get("spot")
    degraded = surface is None or getattr(surface, "degraded", True)
    if not degraded and spot:
        import math
        ks = sorted(c["strike"] for c in pts)
        exp_d = pts[0].get("expiry")
        try:
            dte = max((exp_d - date.today()).days, 1)
        except Exception:
            dte = 30
        t = dte / 365.0
        lo, hi = ks[0], ks[-1]
        xs = [lo + (hi - lo) * i / 60.0 for i in range(61)]
        ys = [surface.evaluate(math.log(x / spot), t) for x in xs]
        fig.add_scatter(x=xs, y=ys, mode="lines",
                        line=dict(color=_CHARCOAL, width=1.4, dash="dash"),
                        hoverinfo="skip")
    shapes, annotations = [], []
    if spot:
        shapes.append(dict(type="line", x0=spot, x1=spot, yref="paper", y0=0, y1=1,
                           line=dict(color=_GREY5, width=1, dash="dot")))
        annotations.append(dict(x=spot, y=1, yref="paper", text="spot",
                                showarrow=False, font={**_FONT, "color": _GREY5},
                                yshift=8))
    for mark in (view.get("rolled_marks") or []):
        shapes.append(dict(type="line", x0=mark["strike"], x1=mark["strike"],
                           yref="paper", y0=0, y1=1,
                           line=dict(color=_AMBER, width=1, dash="dot")))
        annotations.append(dict(x=mark["strike"], y=1, yref="paper",
                                text=mark["label"], showarrow=False,
                                font={**_FONT, "color": _AMBER}, yshift=8))
    if selected and selected.get("strike") is not None and selected.get("iv") is not None:
        fig.add_scatter(x=[selected["strike"]], y=[selected["iv"]],
                        mode="markers+text",
                        marker=dict(color="rgba(0,0,0,0)", size=14,
                                    line=dict(color=_AMBER, width=2)),
                        text=[selected.get("label") or ""],
                        textposition="top center",
                        textfont={**_FONT, "color": _AMBER}, hoverinfo="skip")
    fig.update_layout(
        xaxis=dict(title={"text": "strike", "font": _FONT}, gridcolor=_GRID,
                   zeroline=False, tickfont=_FONT),
        yaxis=dict(title={"text": "IV", "font": _FONT}, ticksuffix="%",
                   gridcolor=_GRID, zeroline=False, tickfont=_FONT),
        shapes=shapes, annotations=annotations)
    if degraded:
        fig.add_annotation(text="fit degraded — no surface line",
                           xref="paper", yref="paper", x=0.99, y=0.02,
                           showarrow=False, font={**_FONT, "color": _AMBER})
    return fig


def _payoff_fig(res) -> "object":
    """The adjusted structure at expiry (solid) and today (dashed), breakevens as
    dotted verticals labeled at the axis, the current spot dotted on P&L = 0."""
    if not res:
        return _msg_fig("Select a candidate above to draw the adjusted payoff.", 240)
    grid = res.get("grid")
    exp_curve = res.get("expiry_curve")
    if grid is None or len(grid) == 0 or exp_curve is None:
        return _msg_fig("Payoff unavailable for this candidate (pricing degraded).", 240)
    # The candidate result is the raw engine dict — its arrays are numpy; Dash's
    # JSON layer wants plain floats.
    xs = [float(v) for v in grid]
    fig = _fig_base(240)
    fig.add_scatter(x=xs, y=[float(v) for v in exp_curve], mode="lines",
                    line=dict(color=_CHARCOAL, width=2),
                    hovertemplate="%{x:.2f} · $%{y:,.0f}<extra>at expiry</extra>")
    hz = res.get("horizon_curve")
    if hz is not None:
        fig.add_scatter(x=xs, y=[float(v) for v in hz], mode="lines",
                        line=dict(color=_GREY5, width=1.3, dash="dash"),
                        hoverinfo="skip")
    spot = res.get("spot")
    if spot:
        fig.add_scatter(x=[float(spot)], y=[0], mode="markers+text",
                        marker=dict(color=_CHARCOAL, size=7), text=["now"],
                        textposition="top center", textfont=_FONT, hoverinfo="skip")
    shapes, annotations = [], []
    for b in (res.get("breakevens") or []):
        b = float(b)
        shapes.append(dict(type="line", x0=b, x1=b, yref="paper", y0=0, y1=1,
                           line=dict(color=_HOLLOW, width=1, dash="dot")))
        annotations.append(dict(x=b, y=0, text=f"{b:,.0f} BE", showarrow=False,
                                font={**_FONT, "color": _GREY5}, yshift=-12))
    fig.update_layout(
        xaxis=dict(title={"text": "underlying", "font": _FONT}, gridcolor=_GRID,
                   zeroline=False, tickfont=_FONT),
        yaxis=dict(title={"text": "P&L $", "font": _FONT}, gridcolor=_GRID,
                   zeroline=True, zerolinecolor=_CHARCOAL, zerolinewidth=1,
                   tickfont=_FONT),
        shapes=shapes, annotations=annotations)
    return fig


# ---------------------------------------------------------------------------
# Row + block builders
# ---------------------------------------------------------------------------

def _roster_rows(roster, captures=None) -> list:
    marked = set(captures or [])
    rows = []
    for r in (roster or {}).get("rows", []):
        rows.append({
            "position_id": r["position_id"],
            "role": (r.get("role") or "").replace("_", " "),
            "contract": (r.get("contract") or "—") + (" ◂" if r.get("anchor") else ""),
            "qty": f"{r['qty']:g}" if r.get("qty") is not None else "—",
            "dte": _int(r.get("dte")) if r.get("dte") is not None else "—",
            "delta": _delta(r.get("delta")),
            "mid": _px(r.get("mid")),
            "p_assign": f"{r['p_assign']:.2f}" if r.get("p_assign") is not None else "—",
            "close": "✓" if r["position_id"] in marked else "·",
        })
    return rows


def _roster_cap(roster, rolled_pids) -> list:
    econ = (roster or {}).get("econ") or {}
    t2 = econ.get("tier2") or {}
    bits = []
    stype = (econ.get("structure_type") or "").replace("_", " ")
    if stype:
        status = econ.get("status") or ""
        bits.append(html.Span(f"{stype} · {status}", className="scanner-cap-k"))
    nd = econ.get("net_delta")
    bits.append(_cap_pair("Δ", f"{nd:+,.0f}" if nd is not None else "—"))
    ndc = t2.get("net_debit_credit")
    if ndc is not None:
        lbl = "net credit" if ndc < 0 else "net debit"
        bits.append(_cap_pair(lbl, _money(abs(ndc))))
    mp, ml = t2.get("max_profit"), t2.get("max_loss")
    bits.append(_cap_pair("max profit", "∞" if t2.get("unbounded_gain") else _money(mp)))
    bits.append(_cap_pair("max loss", "−∞" if t2.get("unbounded_loss") else _money(ml)))
    bes = t2.get("breakevens")
    bits.append(_cap_pair("BE", " / ".join(f"{b:,.0f}" for b in bes) if bes else "—"))
    n = len(rolled_pids or [])
    bits.append(_cap_pair("rolling", f"{n} leg{'s' if n != 1 else ''}"))
    out = []
    for i, b in enumerate(bits):
        if i:
            out.append(html.Span(" · ", className="scanner-cap-sep"))
        out.append(b)
    return out


def _cap_pair(label, value):
    return html.Span([html.Span(f"{label} ", className="scanner-cap-k"),
                      html.Span(value, className="scanner-cap-v")])


def _smile_cap(view) -> list:
    ivr = (view.get("iv_rank") or {})
    pct = ivr.get("percentile")
    ivr_txt = f"{round(pct * 100)}" if pct is not None else "—"
    ratio = view.get("iv_rv_ratio")
    r2 = view.get("fit_r2")
    return [
        _cap_pair("IVR 1y", ivr_txt), html.Span(" · ", className="scanner-cap-sep"),
        _cap_pair("IV / 30d RV", f"{ratio:.2f}" if ratio is not None else "—"),
        html.Span(" · ", className="scanner-cap-sep"),
        _cap_pair("fit R²", f"{r2:.2f}" if r2 is not None else "—"),
    ]


def _opened_legs(c):
    return [lg for lg in (getattr(c, "legs", None) or [])
            if lg.get("opened") and lg.get("opt_type") in ("Call", "Put")]


def _move_label(c) -> str:
    d = getattr(c, "description", "") or ""
    return d.split(" @ ")[0].replace("joint roll ", "").replace("roll ", "")


def _be_for_side(c, rights) -> str:
    bes = getattr(c, "breakevens", None) or []
    if not bes:
        e = getattr(c, "economics", None) or {}
        if e.get("always_profitable"):
            return "always +"
        if e.get("always_loss"):
            return "always −"
        return "—"
    if rights == {"CALL"}:
        return f"{bes[-1]:,.1f}"
    if rights == {"PUT"}:
        return f"{bes[0]:,.1f}"
    return " / ".join(f"{b:,.0f}" for b in bes[:2])


def _cand_rows(ranked_list, view, nav) -> list:
    excess = {r.get("ticker"): r.get("iv_excess") for r in (view.get("iv_pp") or [])}
    rows = []
    for rc in (ranked_list or []):
        c = rc.candidate
        e = getattr(c, "economics", None) or {}
        opened = _opened_legs(c)
        shorts = [lg for lg in opened if (lg.get("qty") or 0) < 0]
        pa = " · ".join(f"{abs(lg['delta']):.2f}" for lg in shorts
                        if lg.get("delta") is not None) or "—"
        prim = _primary_leg(c)
        ipp = excess.get(prim.get("position_id")) if prim else None
        g = getattr(c, "greeks", None) or {}
        mp = "∞" if e.get("unbounded_gain") else _money(e.get("max_profit"))
        if nav and e.get("max_profit") is not None and not e.get("unbounded_gain"):
            mp += f" · {abs(e['max_profit']) / nav * 100:.1f}%"
        flags = list(getattr(rc, "flags", None) or [])
        rows.append({
            "row_id": f"{c.objective}::{rc.rank}",
            "obj": c.objective, "rank_n": rc.rank,
            "rank": ("★" if rc.rank == 1 else str(rc.rank)) + (" ⚠" if flags else ""),
            "move": _move_label(c),
            "dte": _int(getattr(c, "new_leg_dte", None)),
            "net": _money(c.net_credit),
            "net_sign": ("scanner-pos" if (c.net_credit or 0) > 0
                         else "scanner-neg" if (c.net_credit or 0) < 0 else ""),
            "tag": "costless" if _is_costless(c) else "",
            "ivpp": f"{ipp:+.1f}" if ipp is not None else "—",
            "maxp": mp, "maxl": _maxdn(e),
            # Red only for a genuinely negative worst case — an always-profitable
            # structure's positive "max loss" must not wear the loss colour.
            "maxl_cls": ("scanner-neg" if (e.get("unbounded_loss")
                                           or (e.get("max_loss") or 0) < 0)
                         else "scanner-pos" if (e.get("max_loss") or 0) > 0 else ""),
            "be": _be_for_side(c, {"CALL" if lg["opt_type"] == "Call" else "PUT"
                                   for lg in opened}),
            "passign": pa,
            "ndelta": f"{g.get('delta'):+,.0f}" if g.get("delta") is not None else "—",
            "flags": "\n".join(flags) or "",
            "reasons": "\n".join(getattr(rc, "reasons", None) or []) or "",
        })
    return rows


def _chain_rows(view) -> list:
    out = []
    for c in (view.get("contracts") or []):
        fit = "in fit" if c.get("in_fit") else "filtered"
        out.append({"strike": _strike(c.get("strike")), "expiry": str(c.get("expiry") or "—"),
                    "right": {"CALL": "C", "PUT": "P"}.get(c.get("right"), c.get("right") or "—"),
                    "bid": _px(c.get("bid")), "ask": _px(c.get("ask")),
                    "mid": _px(c.get("mid")), "iv": _iv(c.get("iv")),
                    "delta": _delta(c.get("delta")), "oi": _int(c.get("oi")),
                    "fit": fit})
    return out


def _field(res, key):
    if isinstance(res, dict):
        return res.get(key)
    return getattr(res, key, None)


def _cmp_row(label, cur, adj, adj_cls=""):
    return html.Tr([html.Td(label, className="scanner-cmp-k"),
                    html.Td(cur, className="scanner-cmp-cur"),
                    html.Td(adj, className=f"scanner-num {adj_cls}".strip())])


def _cmp_table(current, candidate, rc, nav) -> html.Table:
    ce = _field(current, "economics") or {}
    ne = _field(candidate, "economics") or {}
    cg = _field(current, "greeks_now") or {}
    ng = _field(candidate, "greeks_now") or {}
    c = rc.candidate
    opened = _opened_legs(c)
    shorts = [lg for lg in opened if (lg.get("qty") or 0) < 0]
    pa_new = " · ".join(f"{abs(lg['delta']):.2f}" for lg in shorts
                        if lg.get("delta") is not None) or "—"
    cur_bes = _field(current, "breakevens") or []
    new_bes = _field(candidate, "breakevens") or []
    mp_new = "∞" if ne.get("unbounded_gain") else _money(ne.get("max_profit"))
    if nav and ne.get("max_profit") is not None and not ne.get("unbounded_gain"):
        mp_new += f" · {abs(ne['max_profit']) / nav * 100:.1f}% NAV"
    nc = getattr(c, "net_credit", None)
    head = html.Tr([html.Th(getattr(c, "description", ""), className="scanner-cmp-k"),
                    html.Th("Current"), html.Th("Adjusted")])
    body = [
        _cmp_row("Net Δ",
                 f"{cg.get('delta'):+,.0f}" if cg.get("delta") is not None else "—",
                 f"{ng.get('delta'):+,.0f}" if ng.get("delta") is not None else "—"),
        _cmp_row("Max profit",
                 "∞" if ce.get("unbounded_gain") else _money(ce.get("max_profit")), mp_new),
        _cmp_row("Max loss",
                 "−∞" if ce.get("unbounded_loss") else _money(ce.get("max_loss")),
                 "−∞" if ne.get("unbounded_loss") else _money(ne.get("max_loss")),
                 adj_cls="scanner-neg"),
        _cmp_row("Breakevens",
                 " / ".join(f"{b:,.0f}" for b in cur_bes) or "—",
                 " / ".join(f"{b:,.0f}" for b in new_bes) or "—"),
        _cmp_row("PoP", _pop(ce.get("pop")), _pop(ne.get("pop"))),
        _cmp_row("P(assign) new shorts", "—", pa_new),
        # Adjusted tenor = the ROLLED legs' new expiry (the roll's own clock) — the
        # resulting structure's economics dte is its nearest expiry, often a KEPT
        # sibling's, and would read as "nothing changed" on a PMCC-style roll.
        _cmp_row("Days to expiry", _int(ce.get("dte")),
                 _int(getattr(c, "new_leg_dte", None) or ne.get("dte"))),
        _cmp_row("Net cash to adjust", "—", _money(nc),
                 adj_cls=_sign(nc)),
    ]
    return html.Table(className="scanner-tbl cmp-tbl",
                      children=[html.Thead(head), html.Tbody(body)])


_TICKET_EMPTY = ("Select a candidate — or mark a roster leg Close — to build the "
                 "ticket here.")


def _ticket_row(lg, flag=None):
    """One executable line: action · signed trade qty · contract @ mid · cash.
    Captures carry their run/decay vs entry; ``flag`` is the factual coverage
    conversion attached to the line that causes it."""
    cash_cls = _sign(lg.cash)
    desc = [html.Span(lg.description),
            html.Span(f" @ {_px(lg.mid)}", className="scanner-ticket-mid")]
    if lg.is_capture and lg.pnl_vs_entry is not None:
        desc.append(html.Span(f"{_tcash(lg.pnl_vs_entry)} vs entry",
                              className=f"scanner-ticket-pnl {_sign(lg.pnl_vs_entry)}".strip()))
    if flag:
        desc.append(html.Span(f"→ {flag}", className="scanner-ticket-flag"))
    if lg.note:
        desc.append(html.Span("◆", className="scanner-ticket-notemark", title=lg.note))
    return html.Tr([
        html.Td(lg.action, className=f"scanner-ticket-act {cash_cls}".strip()),
        html.Td(f"{lg.trade_qty:+g}", className="scanner-ticket-qty"),
        html.Td(desc, className="scanner-ticket-desc"),
        html.Td(_tcash(lg.cash), className=f"scanner-num {cash_cls}".strip()),
    ])


def _tcash(v) -> str:
    from pm.candidates.ticket import _cash_str
    return _cash_str(v, dash="—")


def _ticket_block(t) -> html.Div:
    """The ticket band: close lines, open lines, the NET row, then the resulting
    line (label + priced economics + the mids' as-of) as a cap line. Warnings
    render as the standing amber notes."""
    legs = list(t.close_set) + list(t.open_set)
    cap_idx = [i for i, lg in enumerate(legs) if lg.is_capture]
    flag_idx = cap_idx[-1] if (t.conversion and cap_idx) else None
    body = [_ticket_row(lg, flag=(t.conversion if i == flag_idx else None))
            for i, lg in enumerate(legs)]
    net = _tcash(t.net_cash)
    if t.net_cash is not None:
        net += " cr" if t.net_cash >= 0 else " dr"
    body.append(html.Tr(className="scanner-ticket-net", children=[
        html.Td(f"Net · {t.net_label}", colSpan=3, className="scanner-ticket-netlbl"),
        html.Td(net, className=f"scanner-num {_sign(t.net_cash)}".strip()),
    ]))
    res = t.resulting or {}
    cap = [_cap_pair("resulting", (res.get("label") or "—"))]
    e = res.get("economics") or {}
    if e:
        cap += [html.Span(" · ", className="scanner-cap-sep"),
                _cap_pair("max profit", _maxup(e)),
                html.Span(" · ", className="scanner-cap-sep"),
                _cap_pair("max loss", _maxdn(e))]
        bes = e.get("breakevens")
        be_txt = (" / ".join(f"{b:,.1f}" for b in bes) if bes
                  else "always +" if e.get("always_profitable")
                  else "always −" if e.get("always_loss") else "—")
        cap += [html.Span(" · ", className="scanner-cap-sep"), _cap_pair("BE", be_txt)]
    if t.conversion and flag_idx is None:
        cap += [html.Span(" · ", className="scanner-cap-sep"),
                html.Span(t.conversion, className="scanner-ticket-flag")]
    asof = t.as_of.strftime("%H:%M") if t.as_of else "—"
    cap += [html.Span(" · ", className="scanner-cap-sep"),
            _cap_pair("mids as of", asof)]
    kids = [html.Table(className="scanner-tbl scanner-ticket-tbl",
                       children=[html.Tbody(body)]),
            html.Div(cap, className="scanner-cap")]
    kids += [html.Div(w, className="scanner-note") for w in (t.warnings or [])]
    return html.Div(kids)


def _notes_block(view, ranked_list) -> list:
    notes = []
    if view.get("note"):
        notes.append(view["note"])
    seen = set()
    for rc in (ranked_list or []):
        for w in (getattr(rc.candidate, "warnings", None) or []):
            if ("truncated" in w or "slice of a" in w or "not maintained" in w) and w not in seen:
                seen.add(w)
                notes.append(w)
    if not notes:
        return []
    return [html.Div(n, className="scanner-note") for n in notes]


# ---------------------------------------------------------------------------
# The dial pairs (shared dial_sync seam — typed entry commits like a drag)
# ---------------------------------------------------------------------------

def _range_dial(label, sid, lo, hi, step, value):
    # Range bounds commit on Enter/blur (debounce=True): a per-keystroke commit
    # would cross-clamp a half-typed number against the other bound and write
    # the box back mid-edit, eating the user's typing.
    return html.Div(className="scanner-ctrl", children=[
        html.Label(label, className="scanner-ctrl-lbl"),
        dcc.Input(id=f"{sid}-lo", type="number", value=value[0], step=step,
                  debounce=True, className="scanner-ctrl-num"),
        html.Div(dcc.RangeSlider(id=sid, min=lo, max=hi, step=step, value=list(value),
                                 marks=None, allowCross=False,
                                 tooltip={"placement": "bottom", "always_visible": False},
                                 allow_direct_input=False),
                 className="scanner-ctrl-slider"),
        dcc.Input(id=f"{sid}-hi", type="number", value=value[1], step=step,
                  debounce=True, className="scanner-ctrl-num"),
    ])


def _shock_dial(label, sid, lo, hi, step, value=0):
    return html.Div(className="scanner-ctrl", children=[
        html.Label(label, className="scanner-ctrl-lbl"),
        html.Div(dcc.Slider(id=sid, min=lo, max=hi, step=step, value=value,
                            marks={int(lo): str(int(lo)), 0: "0", int(hi): str(int(hi))},
                            tooltip={"placement": "bottom", "always_visible": False},
                            allow_direct_input=False),
                 className="scanner-ctrl-slider"),
        dcc.Input(id=f"{sid}-num", type="number", value=value, step=step,
                  debounce=False, className="scanner-ctrl-num"),
    ])


def _sec(title, ctx_id=None, ctx_text=""):
    kids = [html.Span(title)]
    kids.append(html.Span(ctx_text, id=ctx_id, className="scanner-sec-ctx")
                if ctx_id else html.Span(ctx_text, className="scanner-sec-ctx"))
    return html.Div(className="scanner-sec", children=kids)


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

def render_scanner(account: str, *, position_id: str, structure_id=None) -> html.Div:
    """The drawer body for ``view='scanner'``. Opens immediately; the one-shot
    ``scanner-load`` interval fills the roster, bounds the dials from the chain's
    LISTED expiries, and runs the default scan. ``structure_id`` is drawer-state's,
    read verbatim by every scan so the roster, the candidates and the compare's
    current side all describe ONE structure (None = the leg standalone).

    ``position_id=None`` (a structure with no scannable anchor) renders an explicit
    no-roll-target state instead of a dead tab."""
    if position_id is None:
        return html.Div(className="drawer-content scanner-content", children=[
            html.Div(className="scanner-hd", children=[
                html.Span("Scan", className="scanner-hd-tk")]),
            html.Div("No roll target in this structure — a scan anchors on a "
                     "short option leg (or held stock, for an overlay write); "
                     "this structure has neither. Open a leg position directly "
                     "to scan its chain.", className="scanner-empty"),
        ])
    state = sa.get_state()
    pos = sa.position_by_id(state, account, position_id) if state else None
    name = (getattr(pos, "underlying_symbol", None) or getattr(pos, "symbol", None) or "—")
    return html.Div(className="drawer-content scanner-content", children=[
        # Identity: ticker · spot · day % · account · as-of
        html.Div(className="scanner-hd", children=[
            html.Span(name, className="scanner-hd-tk"),
            html.Span("—", id="scanner-spot", className="scanner-hd-px"),
            html.Span("", id="scanner-day", className="scanner-hd-day"),
            html.Span(account, className="scanner-hd-acct"),
            html.Span("scanning…", id="scanner-stamp", className="scanner-hd-asof"),
            html.Button("Refresh", id="scanner-refresh", n_clicks=0,
                        className="scanner-refresh-btn",
                        title="Re-pull this name's chain slice and re-rank."),
        ]),

        _sec("Managing", "scanner-managing-ctx"),
        _roster_grid(),
        html.Div(id="scanner-roster-cap", className="scanner-cap"),

        _sec("Scan", "scanner-scan-ctx"),
        html.Div(id="scanner-pills", className="scanner-toks"),
        html.Div(className="scanner-ctrls", children=[
            _range_dial("DTE", "scanner-dte", 1, 730, 1, (30, 180)),
            _range_dial("|Δ| band", "scanner-band", 0.02, 0.98, 0.01, (0.02, 0.98)),
            html.Button("Scan", id="scanner-scan", n_clicks=0, className="scanner-scan-btn",
                        title="Apply the DTE / |Δ| controls — pulls only expiries "
                              "not already fetched."),
        ]),

        _sec("Richness", ctx_text="IV+pp vs fitted smile · filled = in fit"),
        html.Div(className="scanner-smile-head", children=[
            dcc.Dropdown(id="scanner-smile-expiry", options=[], value=None,
                         clearable=False, searchable=False,
                         className="scanner-expiry-dd"),
        ]),
        dcc.Graph(id="scanner-smile", figure=_msg_fig("scanning…"),
                  config={"displayModeBar": False},
                  className="scanner-smile-graph", style={"height": "230px"}),
        html.Div(id="scanner-smile-cap", className="scanner-cap"),

        _sec("Candidates", "scanner-cand-ctx"),
        html.Div(id="scanner-notes"),
        _cand_grid(),
        html.Div(className="scanner-more", children=[
            html.Button("", id="scanner-chain-toggle", n_clicks=0,
                        className="scanner-more-btn"),
            html.Button("widen window ▸", id="scanner-widen", n_clicks=0,
                        className="scanner-more-btn",
                        title="Extend the DTE range to the next listed expiry and "
                              "pull just that expiry."),
        ]),
        html.Div(id="scanner-chain-wrap", style={"display": "none"},
                 children=_chain_grid()),

        _sec("Current vs adjusted", ctx_text="kept legs @ entry · new legs @ mid"),
        html.Div(id="scanner-cmp-body",
                 children=html.Div("Select a candidate row above to compare it here.",
                                   className="scanner-empty")),

        _sec("Payoff", ctx_text="adjusted structure · at expiry & today"),
        dcc.Graph(id="scanner-payoff",
                  figure=_msg_fig("Select a candidate above to draw the adjusted payoff.", 240),
                  config={"displayModeBar": False},
                  className="scanner-payoff-graph", style={"height": "240px"}),
        html.Div(className="scanner-ctrls", children=[
            _shock_dial("Underlying move %", "scanner-cmp-spot", -30, 30, 1),
            _shock_dial("Vol shift (pts)", "scanner-cmp-vol", -10, 10, 0.5),
            _shock_dial("Rate shift (bps)", "scanner-cmp-rate", -50, 50, 5),
            _shock_dial("Time fwd (days)", "scanner-cmp-time", 0, 60, 1),
        ]),

        _sec("Ticket", "scanner-ticket-ctx"),
        html.Div(id="scanner-ticket-body",
                 children=html.Div(_TICKET_EMPTY, className="scanner-empty")),
        html.Div(className="scanner-more scanner-ticket-copyrow", children=[
            dcc.Clipboard(id="scanner-ticket-copy", content="",
                          title="Copy the ticket as plain text — carries the as-of; "
                                "a proposal, not an order.",
                          className="scanner-ticket-clip"),
            html.Span("copy ticket", className="scanner-ticket-copylbl"),
        ]),

        dcc.Store(id="scanner-controls", data=None),
        dcc.Store(id="scanner-active", data=None),
        dcc.Store(id="scanner-cmp-sel", data=None),
        dcc.Store(id="scanner-captures", data=[]),
        dcc.Interval(id="scanner-load", interval=60, max_intervals=1),
        html.Div(_HONESTY, className="scanner-honesty"),
    ])


# ---------------------------------------------------------------------------
# The one scan -> render packager
# ---------------------------------------------------------------------------

def _controls_from(dte_value, band_value):
    dte = [int(dte_value[0]), int(dte_value[1])] if dte_value else None
    band = None
    if band_value and not (band_value[0] <= 0.021 and band_value[1] >= 0.979):
        band = [round(float(band_value[0]), 2), round(float(band_value[1]), 2)]
    return dte, band


def _default_dte(listed, pos) -> tuple:
    """The default DTE window: the same three forward monthlies the scanner always
    pulled (held-expiry-forward for a roll; ~30d out for an overlay) — so the
    opening scan costs exactly the historical pull."""
    today = date.today()
    if not listed:
        return (30, 180)
    anchor = getattr(pos, "expiry", None)
    floor_d = anchor if (anchor and getattr(pos, "asset_class", "") == "option") \
        else today + timedelta(days=30)
    fwd = [e for e in listed if e >= floor_d] or listed
    chosen = fwd[:3]
    lo = max((chosen[0] - today).days - 1, 1)
    hi = (chosen[-1] - today).days + 1
    return (lo, hi)


def _scan_view(account, position_id, *, structure_id, rolled_pids, dte_range,
               delta_band, active_hint=None, expiry_hint=None, refresh=False):
    data = sa.scanner_view_data(account, position_id, structure_id=structure_id,
                                rolled_pids=rolled_pids, dte_range=dte_range,
                                delta_band=delta_band, refresh=refresh)
    if data is None:
        return {"unavailable": True}
    state = sa.get_state()
    acc = state.accounts.get(account) if state else None
    data["nav"] = getattr(acc, "nav", None)
    ranked = data.get("ranked") or {}
    objectives = _ordered_objectives(ranked)
    active = (active_hint if active_hint in ranked and ranked.get(active_hint)
              else (_seed_objective(account, position_id, objectives) if objectives else None))
    data["objectives"], data["active"] = objectives, active
    # Rolled-strike references for the smile.
    roster = sa.scanner_roster(account, position_id, structure_id=structure_id) or {}
    data["roster"] = roster
    marks = []
    for r in roster.get("rows", []):
        if r["position_id"] in set(rolled_pids or []) and r.get("is_option"):
            try:
                marks.append({"strike": float((r.get("contract") or "0").split()[0]),
                              "label": r.get("contract", "").replace(" ◂", "")})
            except Exception:
                pass
    data["rolled_marks"] = marks
    return data


def _render_pack(view, account, position_id, rolled_pids, expiry_hint=None,
                 selected=None):
    """Everything the fill callbacks output, from one view read."""
    ranked = view.get("ranked") or {}
    active = view.get("active")
    ranked_list = ranked.get(active) or []
    nav = view.get("nav")
    contracts = view.get("contracts") or []
    exps = sorted({str(c.get("expiry")) for c in contracts if c.get("expiry")})
    exp_opts = [{"label": e, "value": e} for e in exps]
    exp_val = (expiry_hint if expiry_hint in exps
               else (exps[0] if exps else None))
    n_cand = len(ranked_list)
    n_chain = len(contracts)
    top_rows = _cand_rows(ranked_list[:_TOP_N], view, nav)
    joint = view.get("joint")
    rights = {r.get("right") for r in (view.get("roster") or {}).get("rows", [])
              if r["position_id"] in set(rolled_pids or [])} if joint else set()
    spot = view.get("spot")
    day = view.get("day_pct")
    return {
        "spot": f"{spot:,.2f}" if spot is not None else "—",
        "day": (f"{day:+.2f}%" if day is not None else ""),
        "day_cls": ("scanner-pos" if (day or 0) > 0 else "scanner-neg" if (day or 0) < 0 else ""),
        "stamp": f"as of {_stamp(view.get('pulled_at'), view.get('kind'), view.get('spot_asof'))}",
        "managing_ctx": _managing_ctx(view),
        "roster_rows": _roster_rows(view.get("roster")),
        "roster_cap": _roster_cap(view.get("roster"), rolled_pids),
        "scan_ctx": (f"roll targets for the {len(rolled_pids)} selected leg"
                     f"{'s' if len(rolled_pids) != 1 else ''}"
                     + (" · shared expiry" if len(rolled_pids) > 1 else "")),
        "pills": _tokens(ranked, view.get("objectives") or [], active),
        "active": active,
        "smile": _smile_fig(view, exp_val, selected),
        "smile_cap": _smile_cap(view),
        "cand_ctx": _cand_ctx(view, active),
        "notes": _notes_block(view, ranked_list),
        "cand_rows": top_rows,
        "more": f"+ {max(n_chain - n_cand, 0)} in chain · show ▸",
        "chain_rows": _chain_rows(view),
        "exp_opts": exp_opts, "exp_val": exp_val,
    }


def _managing_ctx(view) -> str:
    econ = (view.get("roster") or {}).get("econ") or {}
    stype = (econ.get("structure_type") or "").replace("_", " ")
    return stype if stype else "standalone position"


def _cand_ctx(view, active) -> str:
    econ = (view.get("roster") or {}).get("econ") or {}
    stype = (econ.get("structure_type") or "").replace("_", " ") or "position"
    lbl = _OBJ_LABEL.get(active, active or "—")
    return f"resulting {stype} · {lbl}"


_FILL_OUTPUTS = None  # documented in register_scanner_callbacks


def register_scanner_callbacks(app) -> None:
    """Wire the scanner surface: the one-shot load (roster + dial bounds + the
    default scan), the re-scan triggers (Scan, roster toggles, objective tokens,
    Refresh, widen), the chain expand, and the smile expiry override. The range
    dials and the shock dials register through the shared dial_sync seam — typed
    entry commits exactly like a drag."""
    register_range_dial_sync(app, ("scanner-dte", "scanner-band"))
    register_dial_sync(app, ("scanner-cmp-spot", "scanner-cmp-vol",
                             "scanner-cmp-rate", "scanner-cmp-time"))

    fill_outputs = [
        Output("scanner-spot", "children"),
        Output("scanner-day", "children"),
        Output("scanner-day", "className"),
        Output("scanner-stamp", "children"),
        Output("scanner-managing-ctx", "children"),
        Output("scanner-roster-cap", "children"),
        Output("scanner-scan-ctx", "children"),
        Output("scanner-pills", "children"),
        Output("scanner-active", "data"),
        Output("scanner-smile", "figure"),
        Output("scanner-smile-cap", "children"),
        Output("scanner-cand-ctx", "children"),
        Output("scanner-notes", "children"),
        Output("scanner-cand-grid", "rowData"),
        Output("scanner-chain-toggle", "children"),
        Output("scanner-chain-grid", "rowData"),
        Output("scanner-smile-expiry", "options"),
        Output("scanner-smile-expiry", "value"),
        Output("scanner-controls", "data"),
    ]

    def _pack_tuple(pack, controls):
        return (pack["spot"], pack["day"], f"scanner-hd-day {pack['day_cls']}".strip(),
                pack["stamp"], pack["managing_ctx"], pack["roster_cap"],
                pack["scan_ctx"], pack["pills"], pack["active"], pack["smile"],
                pack["smile_cap"], pack["cand_ctx"], pack["notes"], pack["cand_rows"],
                pack["more"], pack["chain_rows"], pack["exp_opts"], pack["exp_val"],
                controls)

    @app.callback(
        *fill_outputs,
        Output("scanner-roster-grid", "rowData"),
        Output("scanner-roster-grid", "selectedRows"),
        Output("scanner-dte", "min"), Output("scanner-dte", "max"),
        Output("scanner-dte", "value"),
        Output("scanner-dte-lo", "value"), Output("scanner-dte-hi", "value"),
        Input("scanner-load", "n_intervals"),
        State("drawer-state", "data"),
        prevent_initial_call=True,
    )
    def _load(_n, ds):
        if not ds or ds.get("view") != "scanner":
            return (no_update,) * (len(fill_outputs) + 7)
        account, pid = ds.get("account"), ds.get("position_id")
        sid = ds.get("structure_id")
        state = sa.get_state()
        pos = sa.position_by_id(state, account, pid) if state else None
        listed = sa.chain_expiries(account, pid)
        dte = _default_dte(listed, pos)
        rolled = [pid]
        view = _scan_view(account, pid, structure_id=sid, rolled_pids=rolled,
                          dte_range=dte, delta_band=None)
        if view.get("unavailable"):
            empty = html.Div("Scanner unavailable — market data required "
                             "(Bloomberg off) or no priceable slice for this "
                             "position.", className="scanner-empty")
            return (no_update, no_update, no_update, "—", no_update, no_update,
                    no_update, [], None, _msg_fig("no market data"), [], no_update,
                    empty, [], "", [], [], None,
                    {"dte": list(dte), "band": None, "rolled": rolled},
                    [], {"ids": []}, no_update, no_update, no_update,
                    no_update, no_update)
        pack = _render_pack(view, account, pid, rolled)
        controls = {"dte": list(dte), "band": None, "rolled": rolled}
        today = date.today()
        dmax = max(((listed[-1] - today).days + 1) if listed else 730, dte[1])
        return _pack_tuple(pack, controls) + (
            pack["roster_rows"], {"ids": [pid]},
            1, dmax, list(dte), dte[0], dte[1])

    @app.callback(
        *[Output(o.component_id, o.component_property, allow_duplicate=True)
          for o in fill_outputs],
        Output("scanner-cmp-body", "children", allow_duplicate=True),
        Output("scanner-payoff", "figure", allow_duplicate=True),
        Output("scanner-cmp-sel", "data", allow_duplicate=True),
        Input("scanner-scan", "n_clicks"),
        Input("scanner-refresh", "n_clicks"),
        Input("scanner-widen", "n_clicks"),
        Input("scanner-roster-grid", "selectedRows"),
        Input({"type": "scanner-obj", "obj": ALL}, "n_clicks"),
        State("drawer-state", "data"),
        State("scanner-controls", "data"),
        State("scanner-active", "data"),
        State("scanner-dte", "value"),
        State("scanner-band", "value"),
        State("scanner-smile-expiry", "value"),
        prevent_initial_call=True,
    )
    def _rescan(_s, _r, _w, sel_rows, _obj_clicks, ds, controls, active,
                dte_value, band_value, exp_hint):
        if not ds or ds.get("view") != "scanner" or controls is None:
            return (no_update,) * (len(fill_outputs) + 3)
        trig = ctx.triggered_id
        account, pid = ds.get("account"), ds.get("position_id")
        sid = ds.get("structure_id")
        refresh = trig == "scanner-refresh"
        active_hint = active
        rolled = list(controls.get("rolled") or [pid])
        dte, band = controls.get("dte"), controls.get("band")

        if isinstance(trig, dict) and trig.get("type") == "scanner-obj":
            if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
                return (no_update,) * (len(fill_outputs) + 3)
            active_hint = trig.get("obj")
        elif trig == "scanner-roster-grid":
            new_rolled = sorted(r.get("position_id") for r in (sel_rows or []))
            if not new_rolled or new_rolled == sorted(rolled):
                return (no_update,) * (len(fill_outputs) + 3)
            rolled = new_rolled
        elif trig == "scanner-scan":
            if not _s:
                return (no_update,) * (len(fill_outputs) + 3)
            dte, band = _controls_from(dte_value, band_value)
        elif trig == "scanner-widen":
            if not _w:
                return (no_update,) * (len(fill_outputs) + 3)
            listed = sa.chain_expiries(account, pid)
            today = date.today()
            beyond = [e for e in listed if (e - today).days > (dte[1] if dte else 0)]
            if not beyond:
                return (no_update,) * (len(fill_outputs) + 3)
            dte = [dte[0] if dte else 1, (beyond[0] - today).days + 1]
        elif trig == "scanner-refresh":
            if not _r:
                return (no_update,) * (len(fill_outputs) + 3)

        view = _scan_view(account, pid, structure_id=sid, rolled_pids=rolled,
                          dte_range=dte, delta_band=band, active_hint=active_hint,
                          expiry_hint=exp_hint, refresh=refresh)
        if view.get("unavailable"):
            return (no_update,) * (len(fill_outputs) + 3)
        pack = _render_pack(view, account, pid, rolled, expiry_hint=exp_hint)
        controls = {"dte": list(dte) if dte else None,
                    "band": list(band) if band else None, "rolled": rolled}
        cleared = html.Div("Select a candidate row above to compare it here.",
                           className="scanner-empty")
        return _pack_tuple(pack, controls) + (
            cleared, _msg_fig("Select a candidate above to draw the adjusted payoff.", 240),
            None)

    @app.callback(
        Output("scanner-captures", "data"),
        Output("scanner-roster-grid", "rowData", allow_duplicate=True),
        Input("scanner-roster-grid", "cellClicked"),
        State("scanner-captures", "data"),
        State("drawer-state", "data"),
        prevent_initial_call=True,
    )
    def _toggle_capture(cell, captures, ds):
        # The roster's Close column is a click toggle (the house cellClicked
        # pattern) — marks feed the ticket's capture lines; the checkbox column
        # (the rolled set) is untouched by these clicks.
        if not ds or ds.get("view") != "scanner" or not cell:
            return no_update, no_update
        if cell.get("colId") != "close":
            return no_update, no_update
        pid = cell.get("rowId")
        if not pid:
            return no_update, no_update
        captures = list(captures or [])
        if pid in captures:
            captures.remove(pid)
        else:
            captures.append(pid)
        roster = sa.scanner_roster(ds.get("account"), ds.get("position_id"),
                                   structure_id=ds.get("structure_id"))
        return captures, _roster_rows(roster, captures)

    @app.callback(
        Output("scanner-chain-wrap", "style"),
        Input("scanner-chain-toggle", "n_clicks"),
        State("scanner-chain-wrap", "style"),
        prevent_initial_call=True,
    )
    def _chain_toggle(n, style):
        if not n:
            return no_update
        hidden = (style or {}).get("display") == "none"
        return {"display": "block"} if hidden else {"display": "none"}

    @app.callback(
        Output("scanner-smile", "figure", allow_duplicate=True),
        Input("scanner-smile-expiry", "value"),
        State("drawer-state", "data"),
        State("scanner-controls", "data"),
        State("scanner-active", "data"),
        prevent_initial_call=True,
    )
    def _expiry(exp_val, ds, controls, active):
        if not ds or ds.get("view") != "scanner" or not exp_val or not controls:
            return no_update
        view = _scan_view(ds.get("account"), ds.get("position_id"),
                          structure_id=ds.get("structure_id"),
                          rolled_pids=controls.get("rolled") or [ds.get("position_id")],
                          dte_range=controls.get("dte"), delta_band=controls.get("band"),
                          active_hint=active)
        return no_update if view.get("unavailable") else _smile_fig(view, exp_val, None)


# ---------------------------------------------------------------------------
# Current-vs-adjusted + payoff (candidate selection and the shock dials)
# ---------------------------------------------------------------------------

def _cmp_pair(ds, controls, obj, rank, shock):
    account, pid = ds.get("account"), ds.get("position_id")
    sid = ds.get("structure_id")
    rolled = (controls or {}).get("rolled") or [pid]
    kw = dict(structure_id=sid, dte_range=(controls or {}).get("dte"),
              delta_band=(controls or {}).get("band"),
              rolled_pids=rolled if len(rolled) > 1 else None)
    rc = sa.scanner_candidate(account, pid, obj, rank, **kw)
    current = sa.price_payoff(account, structure_id=sid, position_id=pid, shock=shock)
    candidate = sa.price_candidate(account, pid, obj, rank, shock=shock, **kw)
    return rc, current, candidate


def register_comparison_callbacks(app) -> None:
    """Candidate row selection -> the Current-vs-adjusted table, the payoff pair
    and the smile marker; the shock dials reprice both sides at one shock."""

    @app.callback(
        Output("scanner-cmp-body", "children", allow_duplicate=True),
        Output("scanner-payoff", "figure", allow_duplicate=True),
        Output("scanner-smile", "figure", allow_duplicate=True),
        Output("scanner-cmp-sel", "data", allow_duplicate=True),
        Input("scanner-cand-grid", "selectedRows"),
        State("drawer-state", "data"),
        State("scanner-controls", "data"),
        State("scanner-smile-expiry", "value"),
        State("scanner-cmp-spot", "value"),
        State("scanner-cmp-vol", "value"),
        State("scanner-cmp-rate", "value"),
        State("scanner-cmp-time", "value"),
        prevent_initial_call=True,
    )
    def _select(rows, ds, controls, exp_hint, spot_pct, vol_pts, rate_bps, time_days):
        if not ds or ds.get("view") != "scanner" or not rows:
            return (no_update,) * 4
        row = rows[0]
        obj, rank = row.get("obj"), row.get("rank_n")
        shock = {"spot_pct": spot_pct or 0.0, "vol_pts": vol_pts or 0.0,
                 "rate_bps": rate_bps or 0.0, "time_days": int(time_days or 0)}
        rc, current, candidate = _cmp_pair(ds, controls, obj, rank, shock)
        if rc is None or candidate is None:
            return (html.Div("Comparison unavailable for this candidate.",
                             className="scanner-empty"),
                    _msg_fig("Payoff unavailable for this candidate.", 240),
                    no_update, no_update)
        state = sa.get_state()
        acc = state.accounts.get(ds.get("account")) if state else None
        nav = getattr(acc, "nav", None)
        cur = current if current is not None else {}
        cmp_body = _cmp_table(cur, candidate, rc, nav)
        # The smile marker follows the selection.
        rolled = (controls or {}).get("rolled") or [ds.get("position_id")]
        view = _scan_view(ds.get("account"), ds.get("position_id"),
                          structure_id=ds.get("structure_id"), rolled_pids=rolled,
                          dte_range=(controls or {}).get("dte"),
                          delta_band=(controls or {}).get("band"))
        smile = no_update
        if not view.get("unavailable"):
            prim = _primary_leg(rc.candidate)
            sel = None
            if prim is not None:
                tk = prim.get("position_id")
                match = next((c for c in view.get("contracts") or []
                              if c.get("ticker") == tk), None)
                if match:
                    sel = {"strike": match.get("strike"), "iv": match.get("iv"),
                           "label": f"#{rank}  {row.get('move', '')}"}
                    exp_hint = str(match.get("expiry")) or exp_hint
            smile = _smile_fig(view, exp_hint, sel)
        return cmp_body, _payoff_fig(candidate), smile, {"obj": obj, "rank": rank}

    @app.callback(
        Output("scanner-ticket-ctx", "children"),
        Output("scanner-ticket-body", "children"),
        Output("scanner-ticket-copy", "content"),
        Input("scanner-cmp-sel", "data"),
        Input("scanner-captures", "data"),
        State("scanner-controls", "data"),
        State("drawer-state", "data"),
        prevent_initial_call=True,
    )
    def _ticket(sel, captures, controls, ds):
        # The ticket band: the selected candidate's close/open transaction plus
        # the roster's capture marks, at contemporaneous mids — a PROPOSAL with
        # copyable text; never routes an order. The shock dials do not touch it
        # (mids are market marks, not hypotheticals). ``scanner-controls`` is
        # deliberately a STATE: every rescan writes ``scanner-cmp-sel`` (to
        # None), which re-fires this callback with the fresh controls — taking
        # controls as an Input raced the rescan's own selection clear and could
        # rebuild a ticket from the OLD selection against the NEW rolled set.
        if not ds or ds.get("view") != "scanner":
            return no_update, no_update, no_update
        account, pid = ds.get("account"), ds.get("position_id")
        empty = ("one net transaction",
                 html.Div(_TICKET_EMPTY, className="scanner-empty"), "")
        if pid is None:
            return empty
        rolled = (controls or {}).get("rolled") or [pid]
        t = sa.build_adjustment_ticket(
            account, pid, objective=(sel or {}).get("obj"),
            rank=(sel or {}).get("rank"), structure_id=ds.get("structure_id"),
            dte_range=(controls or {}).get("dte"),
            delta_band=(controls or {}).get("band"),
            rolled_pids=rolled, capture_pids=captures or [])
        if t is None:
            return empty
        from pm.candidates.ticket import ticket_text
        n = len(t.close_set) + len(t.open_set)
        return (f"one net transaction · {n} leg{'s' if n != 1 else ''}",
                _ticket_block(t), ticket_text(t))

    @app.callback(
        Output("scanner-cmp-body", "children", allow_duplicate=True),
        Output("scanner-payoff", "figure", allow_duplicate=True),
        Input("scanner-cmp-spot", "value"),
        Input("scanner-cmp-vol", "value"),
        Input("scanner-cmp-rate", "value"),
        Input("scanner-cmp-time", "value"),
        State("drawer-state", "data"),
        State("scanner-controls", "data"),
        State("scanner-cmp-sel", "data"),
        prevent_initial_call=True,
    )
    def _redial(spot_pct, vol_pts, rate_bps, time_days, ds, controls, sel):
        if not ds or ds.get("view") != "scanner" or not sel:
            return no_update, no_update
        shock = {"spot_pct": spot_pct or 0.0, "vol_pts": vol_pts or 0.0,
                 "rate_bps": rate_bps or 0.0, "time_days": int(time_days or 0)}
        rc, current, candidate = _cmp_pair(ds, controls, sel.get("obj"),
                                           sel.get("rank"), shock)
        if rc is None or candidate is None:
            return no_update, no_update
        state = sa.get_state()
        acc = state.accounts.get(ds.get("account")) if state else None
        return (_cmp_table(current if current is not None else {}, candidate, rc,
                           getattr(acc, "nav", None)),
                _payoff_fig(candidate))
