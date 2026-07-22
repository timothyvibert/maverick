"""Scanner drawer — the ranked roll/overlay candidates for a held position.

Renders the candidate layer the compute side already produced (``pull_slice`` ->
``generate_slice_candidates`` -> ``rank_slice_candidates`` on the cached slice) as a new
``view='scanner'`` in the shared drawer: an identity header + the "pulled N min ago"
stamp, the objective pills, and a dense ranked-candidate table with each row's plain
reasons/flags. The rank-1 row is the recommended default.

The drawer opens immediately with a "scanning…" placeholder; a one-shot loader then
runs the on-demand chain pull (the sanctioned owned-state write path) and fills the
table — so a slow first pull never blocks the drawer from appearing. Objective-pill
switches read the cached ranking with no recompute; Refresh re-pulls the slice.

Render-only: the view reads through ``state_access.scanner_view_data`` and never prices
or ranks anything itself. All colour is the shared --pm-*/--pos/--neg tokens.
"""
from __future__ import annotations

from datetime import datetime

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from pm.candidates.generate import _COSTLESS_PER_SHARE
from pm.ui import state_access as sa
from pm.ui.deepdive.aggregations import _fmt_money

# Objective labels + the order the pills read in (the recommender seed picks the default,
# the pills override). Any objective the ranker emits that is not listed still shows.
_OBJ_LABEL = {
    "roll-up-out": "Roll away & out",
    "costless": "Costless",
    "roll-for-credit": "Roll for credit",
    "defend-cut-delta": "Defend / cut Δ",
    "extend-duration": "Extend duration",
    "max-premium": "Max premium",
    "add-hedge": "Add hedge",
}
_OBJ_ORDER = ["roll-up-out", "costless", "roll-for-credit", "defend-cut-delta",
              "extend-duration", "max-premium", "add-hedge"]

# Slice window: base expiries pulled, and the Expand ceiling (bounded — never the full
# chain). Each Expand click adds one expiry, reusing the already-enumerated chain.
_BASE_EXPIRIES = 3
_MAX_EXPIRIES = 6

# The recommender's action -> the default objective pill (action-level; the rule_id
# sub-splits stay a later refinement). An unmapped / neutral label (CLOSE, HARVEST_THETA,
# TRIM, ADD, MONITOR) opens on the first present objective.
_SEED = {
    "ROLL_OUT": "roll-for-credit",
    "ROLL_OUT_AND_DOWN": "defend-cut-delta",
    "ROLL_UP_AND_OUT": "defend-cut-delta",
    "ADD_OVERLAY": "max-premium",
    "ADD_HEDGE": "defend-cut-delta",
}

_HONESTY = ("Ranked by objective-fit and client-fit — advisory, not an order. Economics and "
            "PoP use the live American pricer; IV+pp is the within-set short-leg richness "
            "percentile (the raw ±pp is in each row's reason); the chain is on-demand (see the "
            "stamp), not streaming.")


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
    """Profit-only max-upside cell: '∞' for an unbounded-gain candidate (whose
    max_profit is the None sentinel — a bare _money would show the missing-data
    dash where infinity is meant)."""
    return "∞" if e.get("unbounded_gain") else _money(e.get("max_profit"))


def _maxpl(e) -> str:
    ml = "−∞" if e.get("unbounded_loss") else _money(e.get("max_loss"))
    return f"{_maxup(e)} / {ml}"


def _pop(v) -> str:
    return f"{v * 100:.0f}%" if v is not None else "—"


def _delta(v) -> str:
    return f"{v:+.2f}" if v is not None else "—"


def _ivpp(v) -> str:
    return f"{round(v * 100)}%ile" if v is not None else "—"


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


def _breakevens(c) -> str:
    return " / ".join(f"{b:,.2f}" for b in (getattr(c, "breakevens", None) or [])) or "—"


def _primary_leg(c):
    """The candidate's contract for the Contract columns — the short option leg the
    transaction OPENS (the roll/write target), else its first opened option leg. A
    structure-anchored roll carries the enclosing structure's kept legs too; those
    never key the row (a put-side roll must not key on the kept short call).
    Candidates without the marker fall back over all option legs. None for a
    stock-only candidate."""
    opts = [lg for lg in (getattr(c, "legs", None) or []) if lg.get("opt_type") in ("Call", "Put")]
    pool = [lg for lg in opts if lg.get("opened")] or opts
    shorts = [lg for lg in pool if (lg.get("qty") or 0) < 0]
    return (shorts or pool or [None])[0]


def _identity(pos) -> str:
    if pos is None:
        return "Scan candidates"
    name = pos.underlying_symbol or pos.symbol or "—"
    if pos.asset_class == "option":
        right = (pos.right or "")[:1]
        strike = f"{pos.strike:g}" if pos.strike is not None else "?"
        expiry = pos.expiry.strftime("%b-%y") if pos.expiry else ""
        return f"{name} {strike}{right} {expiry}".strip()
    return f"{name} · {pos.asset_class}"


def _stamp(pulled_at, kind, spot_asof=None) -> str:
    if pulled_at is None:
        # The overlay path prices off the morning snapshot, not a stamped chain pull.
        return "spot from snapshot" if (kind and kind != "option") else "—"
    mins = int((datetime.now() - pulled_at).total_seconds() // 60)
    rel = "just now" if mins <= 0 else ("1 min ago" if mins == 1 else f"{mins} min ago")
    # The spot's own as-of, disclosed beside the pull time: "live" means the
    # underlier rode in the same snapshot request as the candidate quotes.
    spot_note = {"live": " · spot live", "snapshot": " · spot from morning snapshot"}.get(spot_asof, "")
    return f"pulled {rel} · this name only{spot_note}"


# ---------------------------------------------------------------------------
# Pills + table
# ---------------------------------------------------------------------------

def _ordered_objectives(ranked) -> list:
    present = [o for o in _OBJ_ORDER if ranked.get(o)]
    present += [o for o in ranked if o not in _OBJ_ORDER and ranked.get(o)]
    return present


def _seed_objective(account, position_id, objectives) -> str:
    """The default pill, in priority order: the held option's moneyness (an ITM short
    call leads with Roll away & out, an OTM one with Max premium), then the recommender's
    action, then the first present objective. Best-effort — any gap falls back cleanly."""
    try:
        state = sa.get_state()
        acc = state.accounts.get(account) if state else None
        pos = sa.position_by_id(state, account, position_id) if state else None
        if acc is not None and pos is not None:
            # Moneyness lead — a held SHORT option: ITM -> roll away & out (calls:
            # spot above strike; puts: spot below); OTM -> collect premium.
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


def _pills(ranked, objectives, active) -> list:
    out = []
    for o in objectives:
        n = len(ranked.get(o) or [])
        cls = "scanner-pill" + (" scanner-pill-active" if o == active else "")
        out.append(html.Button(f"{_OBJ_LABEL.get(o, o)} · {n}",
                               id={"type": "scanner-obj", "obj": o}, n_clicks=0, className=cls))
    return out


# Costless band: |the ROLL TRANSACTION's net| within _COSTLESS_PER_SHARE (× 100 ×
# contracts) reads "costless" — the constant is single-sourced from the generation
# layer (imported above) so the tag and the COSTLESS objective share one band.


def _is_costless(c) -> bool:
    """True when the candidate's TRANSACTION is (near-)costless.

    Reads ``Candidate.net_credit`` — the roll/overlay's own net, the number the
    row's Net column shows — NOT ``economics.net_debit_credit``, which is the
    RESULTING position's entry cost (stock basis included): that number tagged a
    deeply-debit roll "costless" (tiny resulting entry) while genuinely costless
    rolls never tagged (the new leg's whole premium)."""
    nc = getattr(c, "net_credit", None)
    if nc is None:
        return False
    leg = _primary_leg(c)
    contracts = abs(int(leg.get("qty") or 1)) if leg else 1
    return abs(nc) <= _COSTLESS_PER_SHARE * 100 * max(contracts, 1)


def _right(r) -> str:
    return {"CALL": "C", "PUT": "P"}.get(r, r or "")


def _exp_ord(d) -> int:
    try:
        return d.toordinal()
    except Exception:
        return 0


def _fit_indicator(rc):
    """A compact strength mark for a ranked candidate — the rank (★ on rank 1) beside a
    small fit bar from the combined score. The ranking's computed reasons and flags
    surface HERE: an amber ⚠ when any flag is present (over-extends, degraded
    pricing) with the full flag + reason text in the title — a computed suitability
    warning must never stay invisible. The full inline render belongs to the
    compare-step redesign."""
    score = rc.score if rc.score is not None else 0.0
    pct = max(0, min(100, round(score * 100)))
    label = "★" if rc.rank == 1 else str(rc.rank)
    flags = list(getattr(rc, "flags", None) or [])
    reasons = list(getattr(rc, "reasons", None) or [])
    tip = f"rank {rc.rank} · fit {pct}"
    detail = "\n".join(flags + reasons)
    if detail:
        tip += "\n" + detail
    children = [
        html.Span(label, className="scanner-fit-rank"),
        html.Div(className="scanner-fit-bar",
                 children=html.Div(className="scanner-fit-fill", style={"width": f"{pct}%"})),
    ]
    if flags:
        children.append(html.Span("⚠", className="scanner-fit-flag"))
    return html.Div(className="scanner-fit", title=tip, children=children)


def _cand_by_ticker(ranked_active) -> dict:
    """{contract ticker -> RankedCandidate} for the active objective, so a browse row can
    tell whether it is an actionable candidate (its short leg's contract)."""
    out: dict = {}
    for rc in (ranked_active or []):
        leg = _primary_leg(rc.candidate)
        tk = leg.get("position_id") if leg else None
        if tk and tk not in out:
            out[tk] = rc
    return out


def _browse_row(ct, rc, held_strike) -> html.Tr:
    is_cand = rc is not None
    cls = "scanner-row"
    if is_cand and rc.rank == 1:
        cls += " scanner-row-rec"
    elif is_cand:
        cls += " scanner-row-cand"
    if held_strike is not None and ct.get("strike") == held_strike:
        cls += " scanner-row-held"
    e = (rc.candidate.economics or {}) if is_cand else {}
    nc = rc.candidate.net_credit if is_cand else None
    cells = [
        html.Td(_fit_indicator(rc) if is_cand else "", className="scanner-rank"),
        # Contract (every snapshotted contract)
        html.Td(_strike(ct.get("strike")), className="scanner-num scanner-grp"),
        html.Td(_exp(ct.get("expiry")), className="scanner-mono"),
        html.Td(_right(ct.get("right")), className="scanner-num"),
        html.Td(_px(ct.get("bid")), className="scanner-num"),
        html.Td(_px(ct.get("ask")), className="scanner-num"),
        html.Td(_px(ct.get("mid")), className="scanner-num"),
        html.Td(_iv(ct.get("iv")), className="scanner-num"),
        html.Td(_delta(ct.get("delta")), className="scanner-num"),
        html.Td(_int(ct.get("oi")), className="scanner-num"),
        # Roll · resulting position (candidates only)
        html.Td(_money(nc) if is_cand else "",
                className=f"scanner-num scanner-grp {_sign(nc) if is_cand else ''}".strip()),
        html.Td(_pop(e.get("pop")) if is_cand else "", className="scanner-num"),
        html.Td(_maxup(e) if is_cand else "", className="scanner-num"),
        html.Td(html.Span("costless", className="scanner-tag-costless")
                if (is_cand and _is_costless(rc.candidate)) else "", className="scanner-tag"),
    ]
    if is_cand:
        return html.Tr(id={"type": "scanner-cand", "obj": rc.candidate.objective, "rank": rc.rank},
                       className=cls + " scanner-clickable", children=cells)
    return html.Tr(className=cls, children=cells)


def _browse_table(contracts, ranked_active, held_strike) -> html.Div:
    """Every snapshotted contract in the slice as a browse row; the active objective's
    ranked candidates float to the top and carry the roll economics + a compact fit mark."""
    if not contracts:
        return html.Div("No contracts in the slice.", className="scanner-empty")
    cbt = _cand_by_ticker(ranked_active)

    def key(ct):
        rc = cbt.get(ct.get("ticker"))
        if rc is not None:
            return (0, rc.rank, 0.0)
        return (1, _exp_ord(ct.get("expiry")), float(ct.get("strike") or 0))

    rows = sorted(contracts, key=key)
    group_head = html.Tr(className="scanner-grouprow", children=[
        html.Th("Fit", className="scanner-th-rank"),
        html.Th("Contract", colSpan=9, className="scanner-th scanner-th-grp"),
        html.Th("Roll · resulting", colSpan=4, className="scanner-th scanner-th-grp"),
    ])
    cols = [("", "scanner-th-rank"), ("Strike", "scanner-th-num scanner-grp"), ("Exp", ""),
            ("C/P", "scanner-th-num"), ("Bid", "scanner-th-num"), ("Ask", "scanner-th-num"),
            ("Mid", "scanner-th-num"), ("IV", "scanner-th-num"), ("Δ", "scanner-th-num"),
            ("OI", "scanner-th-num"), ("Net", "scanner-th-num scanner-grp"),
            ("PoP", "scanner-th-num"), ("Max upside", "scanner-th-num"), ("", "")]
    col_head = html.Tr([html.Th(h, className=f"scanner-th {cls}".strip()) for h, cls in cols])
    return html.Table(className="scanner-tbl",
                      children=[html.Thead([group_head, col_head]),
                                html.Tbody([_browse_row(ct, cbt.get(ct.get("ticker")), held_strike)
                                            for ct in rows])])


# ---------------------------------------------------------------------------
# Vol smile — single expiry, reusing the fitted vol surface + per-contract IV+pp
# ---------------------------------------------------------------------------

def _expiry_options(contracts) -> list:
    exps = sorted({c.get("expiry") for c in contracts if c.get("expiry") is not None}, key=_exp_ord)
    return [{"label": e.strftime("%d-%b-%y"), "value": e.isoformat()} for e in exps]


# Fewer than this many listed strikes on an expiry can't support a smile — show the honest
# message rather than a near-empty plot.
_SMILE_MIN_POINTS = 4


def _smile_message(text):
    import plotly.graph_objects as go

    from pm.ui.drawers.payoff import _FONT, _MUTED
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                       align="center", font=dict(family=_FONT, color=_MUTED, size=12))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      autosize=True, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _smile_figure(contracts, surface, expiry_iso, spot, held_strike):
    import math
    from datetime import date

    import plotly.graph_objects as go

    from pm.ui.drawers.payoff import _AMBER, _CHARCOAL, _FONT, _GREY3, _GRID, _MUTED
    pts = [c for c in contracts if c.get("expiry") is not None
           and c["expiry"].isoformat() == expiry_iso
           and c.get("iv") is not None and c.get("strike")]
    if len(pts) < _SMILE_MIN_POINTS:
        return _smile_message("Too few listed strikes to fit a smile for this expiry — "
                              "see the chain below.")
    fig = go.Figure()
    hollow = [c for c in pts if not c.get("in_fit")]
    filled = [c for c in pts if c.get("in_fit")]
    if hollow:
        fig.add_trace(go.Scatter(x=[c["strike"] for c in hollow], y=[c["iv"] for c in hollow],
            mode="markers", name="excluded", marker=dict(color=_GREY3, size=7, symbol="circle-open"),
            hovertemplate="K %{x:g}<br>IV %{y:.1f}<extra></extra>"))
    if filled:
        fig.add_trace(go.Scatter(x=[c["strike"] for c in filled], y=[c["iv"] for c in filled],
            mode="markers", name="in fit", marker=dict(color=_CHARCOAL, size=8),
            hovertemplate="K %{x:g}<br>IV %{y:.1f}<extra></extra>"))
    if surface is not None and not getattr(surface, "degraded", True) and spot and pts:
        dte = max((pts[0]["expiry"] - date.today()).days, 1)
        T = dte / 365.0
        xy = []
        for k in sorted(c["strike"] for c in pts):
            try:
                v = surface.evaluate(math.log(k / spot), T)
            except Exception:
                v = None
            if v is not None:
                xy.append((k, v))
        if xy:
            fig.add_trace(go.Scatter(x=[p[0] for p in xy], y=[p[1] for p in xy], mode="lines",
                name="fitted", line=dict(color=_AMBER, width=2)))
    if held_strike is not None:
        fig.add_vline(x=held_strike, line=dict(color=_MUTED, width=1.5, dash="dash"))
    if spot is not None:
        fig.add_vline(x=spot, line=dict(color=_MUTED, width=1))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=_CHARCOAL, size=11),
        margin=dict(l=48, r=12, t=22, b=36), height=270,
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="strike", gridcolor=_GRID, zeroline=False),
        yaxis=dict(title="IV %", gridcolor=_GRID, zeroline=False))
    return fig


# ---------------------------------------------------------------------------
# Body + fill
# ---------------------------------------------------------------------------

def render_scanner(account: str, *, position_id: str, structure_id=None) -> html.Div:
    """The drawer body for ``view='scanner'``. Opens immediately with a placeholder; the
    one-shot ``scanner-load`` interval then pulls + fills the ranked candidates.
    ``structure_id`` is carried for the compare's current side; the drawer itself does
    not read it — candidate generation resolves the enclosing structure in the service
    layer, so a leg inside a structure already prices as the resulting structure.

    ``position_id=None`` (a structure with no scannable anchor — no short option
    leg and no stock leg, e.g. a long straddle) renders an explicit no-roll-target
    state instead of a dead tab: the scan never starts, so no misleading
    market-data message can appear."""
    if position_id is None:
        return html.Div(className="drawer-content scanner-content", children=[
            html.Div(className="scanner-head", children=[
                html.Div("Scan candidates", className="scanner-identity")]),
            html.Div("No roll target in this structure — a scan anchors on a "
                     "short option leg (or held stock, for an overlay write); "
                     "this structure has neither. Open a leg position directly "
                     "to scan its chain.", className="scanner-empty"),
        ])
    state = sa.get_state()
    pos = sa.position_by_id(state, account, position_id) if state else None
    # Expand exists only for option rolls: it widens the slice by one expiry
    # around the HELD contract, and the slice cache is keyed by that window. An
    # overlay scan (held stock) is spot-anchored at a fixed near-dated window —
    # its caches ignore the window entirely, so an Expand there used to force a
    # Bloomberg re-snapshot that returned identical data. No window, no button.
    head_btns = ([html.Button("Expand ▸", id="scanner-expand", n_clicks=0,
                              className="scanner-refresh-btn",
                              title="Pull one more expiry into the slice — bounded, reuses the "
                                    "enumerated chain (no full-chain pull).")]
                 if getattr(pos, "asset_class", None) == "option" else []) + [
        html.Button("Refresh", id="scanner-refresh", n_clicks=0, className="scanner-refresh-btn",
                    title="Re-pull this name's chain slice and re-rank."),
    ]
    return html.Div(className="drawer-content scanner-content", children=[
        html.Div(className="scanner-head", children=[
            html.Div(className="scanner-head-left", children=[
                html.Div(_identity(pos), className="scanner-identity"),
                html.Span("scanning…", id="scanner-stamp", className="scanner-stamp"),
            ]),
            html.Div(className="scanner-head-btns", children=head_btns),
        ]),
        html.Div(id="scanner-pills", className="scanner-pills"),
        # Vol smile over the cached slice — a selectable expiry, the fitted line, and a
        # dot per snapshotted contract (offset above the line = IV+pp).
        html.Div(className="scanner-smile-wrap", children=[
            html.Div(className="scanner-smile-head", children=[
                html.Span("Vol smile", className="scanner-smile-title"),
                html.Div(className="scanner-smile-ctrl", children=[
                    html.Label("Expiry", className="scanner-smile-lbl"),
                    dcc.Dropdown(id="scanner-smile-expiry", options=[], clearable=False,
                                 className="scanner-expiry-dd"),
                ]),
            ]),
            dcc.Graph(id="scanner-smile", figure=_empty_fig(),
                      config={"displayModeBar": False, "responsive": True},
                      className="scanner-smile-graph"),
        ]),
        html.Div(id="scanner-table", className="scanner-table-wrap",
                 children=html.Div("Scanning the chain…", className="scanner-loading")),
        # Current-vs-candidate comparison — the sliders + body live here from the start, always
        # rendered (stable ids so the recompute wires; never inside display:none, where a
        # dcc.Slider fails to initialise and won't drag). The body fills when a row is selected.
        html.Div(id="scanner-compare", className="scanner-compare", children=[
            html.Div("Compare a candidate vs the current position", className="scanner-cmp-header"),
            _comparison_sliders(),
            html.Div(id="scanner-cmp-body",
                     children=html.Div("Select a candidate row above to compare it here.",
                                       className="scanner-empty")),
        ]),
        dcc.Store(id="scanner-cmp-sel"),
        dcc.Store(id="scanner-active"),
        # The active slice window (number of expiries pulled); Expand widens it.
        dcc.Store(id="scanner-window", data=_BASE_EXPIRIES),
        dcc.Interval(id="scanner-load", interval=60, max_intervals=1, n_intervals=0),
        html.Div(_HONESTY, className="scanner-footer"),
    ])


def _empty_fig():
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=8, r=8, t=8, b=8), height=270,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _scan_view(account, position_id, *, active_hint=None, expiry_hint=None, refresh=False,
               n_expiries=_BASE_EXPIRIES, structure_id=sa.STRUCTURE_AUTO):
    """A fresh scanner read → all rendered pieces. Cheap on a cache hit (the ranking is
    cached at pull time), so pill/expiry switches don't re-price; a refresh re-pulls.
    ``active_hint`` keeps the objective across a refresh, ``expiry_hint`` the smile expiry,
    ``n_expiries`` the active (base or expanded) slice window. ``structure_id`` is
    drawer-state's, passed verbatim so the scan anchors on the SAME structure the
    compare's current side prices (None = the leg standalone)."""
    data = sa.scanner_view_data(account, position_id, refresh=refresh, n_expiries=n_expiries,
                                structure_id=structure_id)
    if data is None:
        return {"unavailable": True}
    ranked = data.get("ranked") or {}
    contracts = data.get("contracts") or []
    surface, spot, held_strike = data.get("surface"), data.get("spot"), data.get("held_strike")
    objectives = _ordered_objectives(ranked)
    stamp = _stamp(data.get("pulled_at"), data.get("kind"), data.get("spot_asof"))
    exp_opts = _expiry_options(contracts)
    exp_val = (expiry_hint if any(o["value"] == expiry_hint for o in exp_opts)
               else (exp_opts[0]["value"] if exp_opts else None))
    active = (active_hint if active_hint in ranked
              else (_seed_objective(account, position_id, objectives) if objectives else None))
    return {
        "unavailable": False, "empty": (not objectives and not contracts),
        "pills": _pills(ranked, objectives, active) if objectives else [],
        "stamp": stamp, "active": active,
        "table": _browse_table(contracts, ranked.get(active, []) if active else [], held_strike),
        "smile": _smile_figure(contracts, surface, exp_val, spot, held_strike),
        "exp_opts": exp_opts, "exp_val": exp_val,
    }


# ---------------------------------------------------------------------------
# Callbacks — load / pill / refresh (all guarded on view == 'scanner')
# ---------------------------------------------------------------------------

def register_scanner_callbacks(app) -> None:
    """Wire the scanner drawer: the one-shot load pull, the objective pills (cache read),
    and Refresh (re-pull). Each reads the target from ``drawer-state`` and outputs to the
    scanner's own inner ids — never ``drawer-body`` (owned by the open-site handler)."""

    @app.callback(
        Output("scanner-pills", "children"),
        Output("scanner-stamp", "children"),
        Output("scanner-table", "children"),
        Output("scanner-active", "data"),
        Output("scanner-smile", "figure"),
        Output("scanner-smile-expiry", "options"),
        Output("scanner-smile-expiry", "value"),
        Input("scanner-load", "n_intervals"),
        State("drawer-state", "data"),
        prevent_initial_call=True,
    )
    def _load(_n, ds):
        if not ds or ds.get("view") != "scanner":
            return (no_update,) * 7
        v = _scan_view(ds.get("account"), ds.get("position_id"),
                       structure_id=ds.get("structure_id"))
        if v.get("unavailable"):
            empty = html.Div("Scanner unavailable — market data required (Bloomberg off) or no "
                             "priceable slice for this position.", className="scanner-empty")
            return [], "—", empty, None, _empty_fig(), [], None
        return (v["pills"], v["stamp"], v["table"], v["active"], v["smile"],
                v["exp_opts"], v["exp_val"])

    # Objective pill → re-filter the browse table + pills (cache read, no re-price); the
    # smile is objective-independent, so it is left as-is.
    @app.callback(
        Output("scanner-pills", "children", allow_duplicate=True),
        Output("scanner-table", "children", allow_duplicate=True),
        Output("scanner-active", "data", allow_duplicate=True),
        Input({"type": "scanner-obj", "obj": ALL}, "n_clicks"),
        State("drawer-state", "data"),
        State("scanner-window", "data"),
        prevent_initial_call=True,
    )
    def _pick(_clicks, ds, window):
        trig = ctx.triggered_id
        if not ds or ds.get("view") != "scanner" or not isinstance(trig, dict):
            return no_update, no_update, no_update
        if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
            return no_update, no_update, no_update
        v = _scan_view(ds.get("account"), ds.get("position_id"), active_hint=trig.get("obj"),
                       n_expiries=window or _BASE_EXPIRIES,
                       structure_id=ds.get("structure_id"))
        if v.get("unavailable") or v.get("empty"):
            return no_update, no_update, no_update
        return v["pills"], v["table"], v["active"]

    # Expiry selector → redraw the smile for that expiry (cache read).
    @app.callback(
        Output("scanner-smile", "figure", allow_duplicate=True),
        Input("scanner-smile-expiry", "value"),
        State("drawer-state", "data"),
        State("scanner-window", "data"),
        prevent_initial_call=True,
    )
    def _expiry(exp_val, ds, window):
        if not ds or ds.get("view") != "scanner" or not exp_val:
            return no_update
        v = _scan_view(ds.get("account"), ds.get("position_id"), expiry_hint=exp_val,
                       n_expiries=window or _BASE_EXPIRIES,
                       structure_id=ds.get("structure_id"))
        return no_update if v.get("unavailable") else v["smile"]

    @app.callback(
        Output("scanner-stamp", "children", allow_duplicate=True),
        Output("scanner-table", "children", allow_duplicate=True),
        Output("scanner-pills", "children", allow_duplicate=True),
        Output("scanner-smile", "figure", allow_duplicate=True),
        Output("scanner-smile-expiry", "options", allow_duplicate=True),
        Output("scanner-smile-expiry", "value", allow_duplicate=True),
        Input("scanner-refresh", "n_clicks"),
        State("drawer-state", "data"),
        State("scanner-active", "data"),
        State("scanner-window", "data"),
        prevent_initial_call=True,
    )
    def _refresh(n, ds, active, window):
        if not ds or ds.get("view") != "scanner" or not n:
            return (no_update,) * 6
        v = _scan_view(ds.get("account"), ds.get("position_id"), active_hint=active,
                       refresh=True, n_expiries=window or _BASE_EXPIRIES,
                       structure_id=ds.get("structure_id"))
        if v.get("unavailable"):
            return (no_update,) * 6
        return v["stamp"], v["table"], v["pills"], v["smile"], v["exp_opts"], v["exp_val"]

    # Expand → widen the slice by one expiry (bounded), re-pull, and re-fill
    # everything. Option rolls only — the button is not rendered for overlay
    # scans (their spot-anchored slice ignores the window; see render_scanner),
    # and the guard below covers a stale layout that still carries it.
    @app.callback(
        Output("scanner-window", "data"),
        Output("scanner-stamp", "children", allow_duplicate=True),
        Output("scanner-table", "children", allow_duplicate=True),
        Output("scanner-pills", "children", allow_duplicate=True),
        Output("scanner-smile", "figure", allow_duplicate=True),
        Output("scanner-smile-expiry", "options", allow_duplicate=True),
        Output("scanner-smile-expiry", "value", allow_duplicate=True),
        Input("scanner-expand", "n_clicks"),
        State("drawer-state", "data"),
        State("scanner-active", "data"),
        State("scanner-window", "data"),
        prevent_initial_call=True,
    )
    def _expand(n, ds, active, window):
        nop = (no_update,) * 7
        if not ds or ds.get("view") != "scanner" or not n:
            return nop
        state = sa.get_state()
        pos = sa.position_by_id(state, ds.get("account"), ds.get("position_id")) if state else None
        if getattr(pos, "asset_class", None) != "option":
            return nop      # no held-expiry window to widen on an overlay scan
        wider = min((window or _BASE_EXPIRIES) + 1, _MAX_EXPIRIES)
        if wider == (window or _BASE_EXPIRIES):
            return nop                      # already at the ceiling
        v = _scan_view(ds.get("account"), ds.get("position_id"), active_hint=active,
                       refresh=True, n_expiries=wider,
                       structure_id=ds.get("structure_id"))
        if v.get("unavailable"):
            return nop
        return (wider, v["stamp"], v["table"], v["pills"], v["smile"], v["exp_opts"], v["exp_val"])


# ---------------------------------------------------------------------------
# Current-vs-candidate comparison (the payoff popup's Scanner tab)
# ---------------------------------------------------------------------------

def _field(res, key):
    """Read a curve/economics field from either a PayoffResult (current) or the
    compute_payoff dict (candidate) — the two comparison sides."""
    if isinstance(res, dict):
        return res.get(key)
    return getattr(res, key, None)


def _comparison_figure(current, candidate):
    """Current (dashed) vs candidate (solid) at-expiry payoff on one set of axes, with
    the candidate's breakevens and the shared spot marked. Token-styled, lazy plotly."""
    import plotly.graph_objects as go
    from pm.ui.drawers.payoff import _AMBER, _CHARCOAL, _FONT, _GRID, _MUTED
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, width=1))
    fig.add_trace(go.Scatter(
        x=_field(current, "grid"), y=_field(current, "expiry_curve"), mode="lines",
        name="Current", line=dict(color=_MUTED, width=1.8, dash="dash"),
        hovertemplate="px %{x:,.2f}<br>current %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=_field(candidate, "grid"), y=_field(candidate, "expiry_curve"), mode="lines",
        name="Candidate", line=dict(color=_CHARCOAL, width=2.2),
        hovertemplate="px %{x:,.2f}<br>candidate %{y:$,.0f}<extra></extra>"))
    for be in (_field(candidate, "breakevens") or []):
        fig.add_vline(x=be, line=dict(color=_AMBER, width=1, dash="dot"))
    spot = _field(current, "spot")
    if spot is not None:
        fig.add_vline(x=spot, line=dict(color=_MUTED, width=1))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=_CHARCOAL, size=11),
        margin=dict(l=54, r=12, t=26, b=36), height=300,
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="underlying price", gridcolor=_GRID, zeroline=False),
        yaxis=dict(title="P&L $", gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID))
    return fig


def _cmp_money_row(label, cur, new, cur_txt=None, new_txt=None):
    """One economics comparison row. ``cur_txt`` / ``new_txt`` override the money
    formatting for non-numeric truths (∞ / −∞ / 'unbounded'); with an unbounded
    side the Δ is undefined and renders '—' (the None arithmetic already does)."""
    d = (new - cur) if (cur is not None and new is not None) else None
    return html.Tr([
        html.Td(label, className="cmp-lbl"),
        html.Td(cur_txt if cur_txt is not None else _money(cur), className="scanner-num"),
        html.Td(new_txt if new_txt is not None else _money(new), className="scanner-num"),
        html.Td(_money(d) if d is not None else "—", className=f"scanner-num {_sign(d)}".strip()),
    ])


def _side_by_side_econ(current, candidate, net_to_roll):
    ce = _field(current, "economics") or {}
    ne = _field(candidate, "economics") or {}
    pc, pn = ce.get("pop"), ne.get("pop")
    pop_d = (pn - pc) if (pc is not None and pn is not None) else None
    pop_row = html.Tr([
        html.Td("PoP", className="cmp-lbl"),
        html.Td(_pop(pc), className="scanner-num"), html.Td(_pop(pn), className="scanner-num"),
        html.Td((f"{pop_d * 100:+.0f} pp" if pop_d is not None else "—"), className="scanner-num")])
    def _up_txt(e):
        return "∞" if e.get("unbounded_gain") else None

    def _dn_txt(e):
        return "−∞" if e.get("unbounded_loss") else None

    def _car_txt(e):
        return ("unbounded" if (e.get("unbounded_loss")
                                and e.get("capital_at_risk") is None) else None)

    head = html.Tr([html.Th("Economics", className="cmp-lbl"), html.Th("Current"),
                    html.Th("Candidate"), html.Th("Δ")])
    body = [_cmp_money_row("Max profit", ce.get("max_profit"), ne.get("max_profit"),
                           _up_txt(ce), _up_txt(ne)),
            _cmp_money_row("Max loss", ce.get("max_loss"), ne.get("max_loss"),
                           _dn_txt(ce), _dn_txt(ne)),
            _cmp_money_row("Cap at risk", ce.get("capital_at_risk"), ne.get("capital_at_risk"),
                           _car_txt(ce), _car_txt(ne)),
            pop_row,
            _cmp_money_row("Net premium", ce.get("net_premium"), ne.get("net_premium"))]
    foot = html.Tr(className="cmp-foot", children=[
        html.Td("Net to roll", className="cmp-lbl"), html.Td("", colSpan=2),
        html.Td(_money(net_to_roll), className=f"scanner-num {_sign(net_to_roll)}".strip())])
    return html.Table(className="scanner-tbl cmp-tbl",
                      children=[html.Thead(head), html.Tbody(body), html.Tfoot(foot)])


def _side_by_side_greeks(current, candidate):
    cg = _field(current, "greeks_now") or {}
    ng = _field(candidate, "greeks_now") or {}
    head = html.Tr([html.Th("Greeks (engine)", className="cmp-lbl"), html.Th("Current"),
                    html.Th("Candidate"), html.Th("Δ")])
    body = [_cmp_money_row("Δ (per $1)", cg.get("delta"), ng.get("delta")),
            _cmp_money_row("Γ (per $1²)", cg.get("gamma"), ng.get("gamma")),
            _cmp_money_row("ν (per vol pt)", cg.get("vega"), ng.get("vega")),
            _cmp_money_row("Θ (per day)", cg.get("theta"), ng.get("theta"))]
    return html.Table(className="scanner-tbl cmp-tbl", children=[html.Thead(head), html.Tbody(body)])


def _comparison_sliders(shock=None):
    from pm.ui.drawers.payoff import _slider
    sp = shock or {}
    return html.Div(className="scanner-cmp-dial", children=[
        html.Div(className="payoff-ctrl", children=[
            html.Label("Underlying move %", className="payoff-ctrl-lbl"),
            _slider("scanner-cmp-spot", -30, 30, 1, sp.get("spot_pct", 0))]),
        html.Div(className="payoff-ctrl", children=[
            html.Label("Vol shift (pts)", className="payoff-ctrl-lbl"),
            _slider("scanner-cmp-vol", -10, 10, 0.5, sp.get("vol_pts", 0))]),
        html.Div(className="payoff-ctrl", children=[
            html.Label("Rate shift (bps)", className="payoff-ctrl-lbl"),
            _slider("scanner-cmp-rate", -50, 50, 5, sp.get("rate_bps", 0))]),
        html.Div(className="payoff-ctrl", children=[
            html.Label("Time fwd (days)", className="payoff-ctrl-lbl"),
            _slider("scanner-cmp-time", 0, 60, 1, sp.get("time_days", 0))]),
    ])


def _comparison_body(ds, obj, rank, shock, n_expiries=_BASE_EXPIRIES):
    """Figure + side-by-side economics/greeks for one selected candidate at the given
    shock. Both sides price at the slice's spot so the curves share a grid.
    ``n_expiries`` is the scanner's ACTIVE window: the slice cache (and its attached
    ranking) is keyed by window, so the comparison must resolve (obj, rank) against
    the same window the clicked table rendered from — not the base default."""
    account, sid, pid = ds.get("account"), ds.get("structure_id"), ds.get("position_id")
    rc = sa.scanner_candidate(account, pid, obj, rank, n_expiries=n_expiries,
                              structure_id=sid)
    current = sa.price_payoff(account, structure_id=sid, position_id=pid, shock=shock)
    candidate = sa.price_candidate(account, pid, obj, rank, shock=shock,
                                   n_expiries=n_expiries, structure_id=sid)
    if rc is None or current is None or candidate is None:
        return html.Div("Comparison unavailable for this candidate.", className="scanner-empty")
    net_to_roll = getattr(rc.candidate, "net_credit", None)
    desc = getattr(rc.candidate, "description", "")
    return html.Div(children=[
        html.Div(f"Current vs candidate — {desc}", className="scanner-cmp-title"),
        dcc.Graph(figure=_comparison_figure(current, candidate),
                  config={"displayModeBar": False, "responsive": True}, className="scanner-cmp-graph"),
        html.Div(className="scanner-cmp-cols", children=[
            _side_by_side_econ(current, candidate, net_to_roll),
            _side_by_side_greeks(current, candidate),
        ]),
    ])


def register_comparison_callbacks(app) -> None:
    """Wire the comparison: selecting a candidate row draws current vs candidate and
    reveals the sliders; the sliders (stable ids in the scanner body) reprice both at one
    shock. Independent of the payoff dial. Guarded on the scanner view."""

    @app.callback(
        Output("scanner-cmp-body", "children"),
        Output("scanner-cmp-sel", "data"),
        Input({"type": "scanner-cand", "obj": ALL, "rank": ALL}, "n_clicks"),
        State("drawer-state", "data"),
        State("scanner-window", "data"),
        State("scanner-cmp-spot", "value"),
        State("scanner-cmp-vol", "value"),
        State("scanner-cmp-rate", "value"),
        State("scanner-cmp-time", "value"),
        prevent_initial_call=True,
    )
    def _select(_clicks, ds, window, spot_pct, vol_pts, rate_bps, time_days):
        trig = ctx.triggered_id
        # The comparison works on any scanner tab; the "current" side prices the enclosing
        # structure when the anchor carries a structure_id, else the standalone leg. Renders
        # at the sliders' current shock so it agrees with what the dials show. The window
        # store rides along so the clicked row resolves against the ranking it came from
        # (after Expand, the base-window ranking is a DIFFERENT candidate list).
        if not ds or ds.get("view") != "scanner" or not isinstance(trig, dict):
            return no_update, no_update
        if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
            return no_update, no_update
        obj, rank = trig.get("obj"), trig.get("rank")
        shock = {"spot_pct": spot_pct or 0.0, "vol_pts": vol_pts or 0.0,
                 "rate_bps": rate_bps or 0.0, "time_days": int(time_days or 0)}
        n_exp = int(window or _BASE_EXPIRIES)
        return (_comparison_body(ds, obj, rank, shock, n_expiries=n_exp),
                {"obj": obj, "rank": rank})

    @app.callback(
        Output("scanner-cmp-body", "children", allow_duplicate=True),
        Input("scanner-cmp-spot", "value"),
        Input("scanner-cmp-vol", "value"),
        Input("scanner-cmp-rate", "value"),
        Input("scanner-cmp-time", "value"),
        State("drawer-state", "data"),
        State("scanner-cmp-sel", "data"),
        State("scanner-window", "data"),
        prevent_initial_call=True,
    )
    def _redial(spot_pct, vol_pts, rate_bps, time_days, ds, sel, window):
        if not ds or ds.get("view") != "scanner" or not sel:
            return no_update
        shock = {"spot_pct": spot_pct or 0.0, "vol_pts": vol_pts or 0.0,
                 "rate_bps": rate_bps or 0.0, "time_days": int(time_days or 0)}
        return _comparison_body(ds, sel.get("obj"), sel.get("rank"), shock,
                                n_expiries=int(window or _BASE_EXPIRIES))
