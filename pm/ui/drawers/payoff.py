"""Payoff drawer — the structure/position economics + scenario surface.

Renders ``pm.risk.payoff.structure_payoff`` (a ``PayoffResult``) into the shared drawer
as a new ``view='payoff'``: a named identity header, a token-styled payoff figure
(at-expiry + horizon curves with current/shocked-spot, strike, and breakeven markers),
an economics block, greeks-now-vs-shocked, and an own dial (underlying-move % / vol pts /
rate bps / time) that recomputes live via the read-only ``state_access.price_payoff`` —
no Bloomberg, no reload, no state write-back.

The x-axis is the UNDERLYING's own price (beta = 1), distinct from the Scenario section's
SPX-beta axis — so the dial's spot move is THIS name's move and is not seeded from the
Scenario shock (fork 3 = A). plotly is imported lazily in the figure builder; all colours
are the shared --pm-*/--pos/--neg tokens (never default plotly).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from dash import Input, Output, State, dcc, html, no_update

from pm.ui import state_access as sa
from pm.ui.deepdive.aggregations import _fmt_money
from pm.ui.deepdive.structures_panel import _TYPE_LABEL as _STRUCT_LABEL

# token hexes mirroring assets/style.css (:root) — plotly needs explicit colours.
_CHARCOAL = "#2B2B2B"
_POS = "#1E7E34"
_NEG = "#C62828"
_GRID = "#E8E8E8"
_AMBER = "#B7791F"
_MUTED = "#6E6E6E"
_GREY3 = "#B5B5B5"
_FONT = '"Frutiger 45 Light","Frutiger","Helvetica Neue","Segoe UI",Arial,sans-serif'


def _sign_cls(v) -> str:
    if v is None or v == 0:
        return ""
    return "payoff-pos" if v > 0 else "payoff-neg"


# ---------------------------------------------------------------------------
# Identity header — named, with a generic fallback (no structure renders nameless)
# ---------------------------------------------------------------------------

def _mon(d) -> str:
    try:
        return d.strftime("%b-%y")
    except Exception:
        return ""


def _mon_iso(iso_str) -> str:
    """'2026-09-18' -> 'Sep-26' (falls back to the raw string)."""
    try:
        from datetime import date as _date
        return _date.fromisoformat(iso_str).strftime("%b-%y")
    except Exception:
        return iso_str or ""


def _leg_phrase(leg) -> str:
    qty = leg.get("qty") or 0.0
    side = "long" if qty >= 0 else "short"
    n = abs(qty)
    if leg.get("is_stock"):
        return f"{side} {n:,.0f} sh"
    k = leg.get("K")
    k_s = f"{k:g}" if k is not None else "?"
    init = (leg.get("opt_type") or "?")[0]
    return f"{side} {n:,.0f}× {init}{k_s} {_mon(leg.get('expiry'))}".rstrip()


def payoff_identity(result) -> str:
    """e.g. 'BABA — Covered call · long 50,000 sh + short 500× C180 Jan-27', or a generic
    'BABA — 3-leg · …' for an un-templated leg-set, or 'BABA — short 5× P80 Jul-26 · no
    stock held' for a standalone. Never nameless."""
    legs = result.legs or []
    phrases = " + ".join(_leg_phrase(l) for l in legs) if legs else "—"
    has_stock = any(l.get("is_stock") for l in legs)
    if result.structure_id:
        label = _STRUCT_LABEL.get(result.structure_type) or f"{len(legs)}-leg"
        desc = f"{label} · {phrases}"
    else:
        desc = phrases
    if not has_stock:
        desc += " · no stock held"
    return f"{result.underlying or '—'} — {desc}"


# ---------------------------------------------------------------------------
# Figure (lazy plotly, token-styled)
# ---------------------------------------------------------------------------

def _y_on(curve, x, s):
    if curve is None or s is None:
        return None
    try:
        return float(np.interp(float(s), np.asarray(x, dtype=float), np.asarray(curve, dtype=float)))
    except Exception:
        return None


def payoff_figure(result, show_components=False):
    import plotly.graph_objects as go          # lazy

    x = result.grid
    horizon = result.horizon_curve
    econ = result.economics or {}
    multi = econ.get("eval_mode") == "nearest_expiry"
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=_GRID, width=1))
    # Per-leg strike markers, labelled (and expiry-tagged for a multi-expiry
    # structure, so a ladder's strike levels read per leg). Falls back to plain
    # lines when the result predates the marker field.
    markers = getattr(result, "strike_markers", None)
    if markers:
        for m in markers:
            label = f"{(m.get('opt_type') or '?')[0]}{m.get('K'):g}"
            if multi and m.get("expiry"):
                label += f" · {_mon_iso(m['expiry'])}"
            fig.add_vline(x=m["K"], line=dict(color=_MUTED, width=1, dash="dot"),
                          annotation_text=label, annotation_position="top",
                          annotation_font=dict(size=9, color=_MUTED))
    else:
        for k in (result.strikes or []):
            fig.add_vline(x=k, line=dict(color=_MUTED, width=1, dash="dot"))
    # Component (standalone-vs-net) curves, faint and UNDER the combined line: the
    # stock leg alone vs the option overlay alone (combined = their sum at every point).
    # On by default for stock+option overlays (see render_payoff / the dial toggle).
    if show_components:
        if result.expiry_curve_stock is not None:
            fig.add_trace(go.Scatter(
                x=x, y=result.expiry_curve_stock, mode="lines", name="Stock alone",
                line=dict(color=_MUTED, width=1.3, dash="dot"),
                hovertemplate="px %{x:,.2f}<br>stock-alone P&L %{y:$,.0f}<extra></extra>"))
        if result.expiry_curve_options is not None:
            fig.add_trace(go.Scatter(
                x=x, y=result.expiry_curve_options, mode="lines", name="Options alone",
                line=dict(color=_GREY3, width=1.6, dash="dash"),
                hovertemplate="px %{x:,.2f}<br>options-alone P&L %{y:$,.0f}<extra></extra>"))
    expiry_name = "At expiry"
    if multi and econ.get("eval_date"):
        expiry_name = f"At nearest expiry ({_mon_iso(econ['eval_date'])})"
    fig.add_trace(go.Scatter(
        x=x, y=result.expiry_curve, mode="lines", name=expiry_name,
        line=dict(color=_CHARCOAL, width=2),
        hovertemplate="px %{x:,.2f}<br>expiry P&L %{y:$,.0f}<extra></extra>"))
    if horizon is not None:
        fig.add_trace(go.Scatter(
            x=x, y=horizon, mode="lines", name="Horizon",
            line=dict(color=_AMBER, width=2),
            hovertemplate="px %{x:,.2f}<br>horizon P&L %{y:$,.0f}<extra></extra>"))
    if result.breakevens:
        fig.add_trace(go.Scatter(
            x=result.breakevens, y=[0] * len(result.breakevens), mode="markers", name="Breakeven",
            marker=dict(symbol="x", size=9, color=_MUTED),
            hovertemplate="breakeven %{x:,.2f}<extra></extra>"))
    ref = horizon if horizon is not None else result.expiry_curve
    cy = _y_on(ref, x, result.spot)
    if cy is not None:
        fig.add_trace(go.Scatter(
            x=[result.spot], y=[cy], mode="markers", name="Spot",
            marker=dict(symbol="circle", size=12, color=_CHARCOAL, line=dict(color="white", width=1.5)),
            hovertemplate="spot %{x:,.2f}<extra></extra>"))
    sy = _y_on(ref, x, result.shocked_spot)
    if sy is not None:
        fig.add_trace(go.Scatter(
            x=[result.shocked_spot], y=[sy], mode="markers", name="Shocked",
            marker=dict(symbol="diamond", size=12, color=_AMBER, line=dict(color="white", width=1.5)),
            hovertemplate="shocked %{x:,.2f}<extra></extra>"))
    fig.update_layout(
        # No fixed height — the container (.payoff-graph, height:100%) governs the size, so
        # the initial render and every slider recompute produce the same stretched chart.
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        font=dict(family=_FONT, color=_CHARCOAL, size=11),
        margin=dict(l=58, r=12, t=14, b=40),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=f"{result.underlying or 'underlying'} price", gridcolor=_GRID,
                   zeroline=False),
        yaxis=dict(title="P&L $", gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID))
    return fig


# ---------------------------------------------------------------------------
# Economics + greeks blocks
# ---------------------------------------------------------------------------

def _stat(label, value, cls="") -> html.Div:
    return html.Div(className="payoff-stat", children=[
        html.Span(label, className="payoff-stat-lbl"),
        html.Span(value, className=f"payoff-stat-val {cls}".strip())])


def _mpl(val, unbounded, symbol) -> str:
    if unbounded:
        return symbol
    return _fmt_money(val)


def economics_block(result) -> html.Div:
    e = result.economics
    if result.breakevens:
        be = " / ".join(f"{b:,.2f}" for b in result.breakevens)
    elif e.get("always_profitable"):
        be = "always profitable"          # the analytic list is complete: no crossing
    elif e.get("always_loss"):
        be = "always loss"
    else:
        be = "—"
    car = (_fmt_money(e["capital_at_risk"]) if e["capital_at_risk"] is not None
           else ("unbounded" if e["unbounded_loss"] else "—"))
    pop = f"{e['pop'] * 100:.0f}%" if e["pop"] is not None else "—"

    def _with_bound(txt, bound):
        # "8,200 (S ≥ 110.00)" — the attainment region beside the bounded extreme.
        return f"{txt} ({bound})" if bound else txt

    multi = (e.get("n_expiries") or 0) > 1
    dte = (str(e["dte"]) if e["dte"] is not None else "—")
    if multi and e["dte"] is not None:
        dte += f" (nearest of {e['n_expiries']} expiries)"
    pop_label = "PoP (at nearest expiry)" if e.get("eval_mode") == "nearest_expiry" \
        else "PoP (at expiry)"
    rows = [
        _stat("Breakeven(s)", be),
        _stat("Max profit",
              _with_bound(_mpl(e["max_profit"], e["unbounded_gain"], "∞"),
                          None if e["unbounded_gain"] else e.get("max_profit_bound")),
              _sign_cls(e["max_profit"])),
        _stat("Max loss",
              _with_bound(_mpl(e["max_loss"], e["unbounded_loss"], "−∞"),
                          None if e["unbounded_loss"] else e.get("max_loss_bound")),
              _sign_cls(e["max_loss"])),
        _stat("Capital at risk", car),
        _stat(pop_label, pop),
        _stat("Net premium", _fmt_money(e["net_premium"])),
        _stat("Current P&L", _fmt_money(e["current_pnl"]), _sign_cls(e["current_pnl"])),
        _stat("DTE", dte),
    ]
    children = [html.H4("Economics", className="payoff-block-title"),
                html.Div(rows, className="payoff-stats")]
    if e.get("econ_caveat"):
        children.append(html.Div(e["econ_caveat"], className="payoff-caveat"))
    if e.get("pop_caveat"):
        children.append(html.Div(e["pop_caveat"], className="payoff-caveat"))
    return html.Div(children)


def greeks_block(result) -> html.Div:
    now = result.greeks_now or {}
    sh = result.greeks_shocked

    def _row(lbl, key):
        n = now.get(key)
        s = (sh or {}).get(key)
        return html.Tr(children=[
            html.Td(lbl, className="payoff-greek-lbl"),
            html.Td(_fmt_money(n), className=f"payoff-greek-num {_sign_cls(n)}".strip()),
            html.Td(_fmt_money(s) if sh else "—", className=f"payoff-greek-num {_sign_cls(s)}".strip())])

    head = html.Tr(children=[html.Th("Greek"), html.Th("Now"), html.Th("Shocked")])
    body = [_row("Δ (per $1)", "delta"), _row("Γ (per $1²)", "gamma"),
            _row("ν (per vol pt)", "vega"), _row("Θ (per day)", "theta")]
    return html.Div(children=[
        html.H4("Greeks", className="payoff-block-title"),
        html.Table(className="payoff-greeks-table",
                   children=[html.Thead(head), html.Tbody(body)]),
        html.Div("engine per-$1² basis — distinct from the Exposure section's BBG "
                 "per-1% γ; do not compare.", className="payoff-caveat")])


# ---------------------------------------------------------------------------
# Dial (own controls; fork 3 = A — spot is the underlying's own move)
# ---------------------------------------------------------------------------

def _slider(_id, lo, hi, step, val):
    marks = {int(lo): str(int(lo)), 0: "0", int(hi): str(int(hi))}
    # allow_direct_input=False: the slider's built-in entry box moves the thumb on
    # Enter without committing the value (see pm.ui.dial_sync) — at drawer widths
    # the component auto-hides it anyway, but a wider layout would surface it.
    # The paired dcc.Input in _dial_row is the committed typed gesture.
    return dcc.Slider(id=_id, min=lo, max=hi, step=step, value=val or 0, marks=marks,
                      tooltip={"placement": "bottom", "always_visible": False},
                      allow_direct_input=False, className="payoff-slider")


def _dial_row(label, _id, lo, hi, step, val):
    # Slider + explicit numeric entry kept in lockstep by dial_sync. debounce=False
    # is load-bearing: every keystroke commits and the slider tracks it, so no
    # uncommitted text can survive to re-commit later (see pm.ui.dial_sync).
    return html.Div(className="payoff-ctrl", children=[
        html.Label(label, className="payoff-ctrl-lbl"),
        html.Div(className="payoff-dialrow", children=[
            _slider(_id, lo, hi, step, val),
            dcc.Input(id=f"{_id}-num", type="number", value=val or 0, step=step,
                      debounce=False, className="payoff-num"),
        ]),
    ])


def _is_overlay(result) -> bool:
    """True for a stock + option structure (covered call / collar) — where the
    standalone-vs-net comparison is the point. Drives the component-curve default."""
    legs = result.legs or []
    return (any(l.get("is_stock") for l in legs)
            and any(not l.get("is_stock") for l in legs))


def _dial(result, shock) -> html.Div:
    sp = shock or {}
    dte = result.economics.get("dte")
    tmax = int(dte) if (dte and dte > 0) else 30   # the drawer can't decay past expiry
    overlay = _is_overlay(result)
    return html.Div(className="payoff-dial", children=[
        _dial_row(f"{result.underlying or 'spot'} move %", "payoff-spot",
                  -30, 30, 1, sp.get("spot_pct", 0)),
        _dial_row("Vol shift (pts)", "payoff-vol", -10, 10, 0.5, sp.get("vol_pts", 0)),
        _dial_row("Rate shift (bps)", "payoff-rate", -50, 50, 5, sp.get("rate_bps", 0)),
        _dial_row("Time to expiry (days)", "payoff-time", 0, tmax, 1, sp.get("time_days", 0)),
        html.Div(className="payoff-ctrl payoff-ctrl-narrow", children=[
            html.Label("Compare", className="payoff-ctrl-lbl"),
            dcc.Checklist(id="payoff-components",
                          options=[{"label": " components", "value": "components",
                                    "disabled": not overlay}],
                          value=(["components"] if overlay else []),
                          className="payoff-check")]),
    ])


# ---------------------------------------------------------------------------
# Body + live recompute callback
# ---------------------------------------------------------------------------

def render_payoff(account: str, *, structure_id: Optional[str] = None,
                  position_id: Optional[str] = None, shock: Optional[dict] = None) -> html.Div:
    """The drawer body for ``view='payoff'``. Reads the read-only ``price_payoff``
    recompute; renders the named header, figure, economics, greeks, and the live dial."""
    result = sa.price_payoff(account, structure_id=structure_id,
                             position_id=position_id, shock=shock)
    if result is None:
        return html.Div("Payoff unavailable for this selection (no priceable legs / spot).",
                        className="drawer-content payoff-empty")
    return html.Div(className="drawer-content payoff-content", children=[
        html.Div(payoff_identity(result), className="payoff-identity"),
        _dial(result, shock),
        html.Div(className="payoff-body", children=[
            html.Div(className="payoff-graph-wrap", children=[
                dcc.Graph(id="payoff-graph",
                          figure=payoff_figure(result, show_components=_is_overlay(result)),
                          config={"displayModeBar": False, "responsive": True},
                          className="payoff-graph")]),
            html.Div(className="payoff-side", children=[
                html.Div(id="payoff-econ", children=economics_block(result)),
                html.Div(id="payoff-greeks", children=greeks_block(result)),
            ]),
        ]),
        html.Div("At-expiry = piecewise-linear intrinsic net of entry premium; horizon = "
                 "fast BS2002 engine reprice at the dialled state. Read-only — no Bloomberg, "
                 "no reload.", className="payoff-caption"),
    ])


def register_payoff_callbacks(app) -> None:
    """Wire the dial -> live repaint of the figure + economics + greeks. Reads the target
    (account + structure_id/position_id) from ``drawer-state``; recomputes read-only."""
    from pm.ui.dial_sync import register_dial_sync

    # Typed values commit via the paired entry boxes (see dial_sync) and chain
    # into _recompute through the slider write, exactly like a drag.
    register_dial_sync(app, ("payoff-spot", "payoff-vol", "payoff-rate", "payoff-time"))

    @app.callback(
        Output("payoff-graph", "figure"),
        Output("payoff-econ", "children"),
        Output("payoff-greeks", "children"),
        Input("payoff-spot", "value"),
        Input("payoff-vol", "value"),
        Input("payoff-rate", "value"),
        Input("payoff-time", "value"),
        Input("payoff-components", "value"),
        State("drawer-state", "data"),
        prevent_initial_call=True,
    )
    def _recompute(spot_pct, vol_pts, rate_bps, time_days, components, ds):
        if not ds or ds.get("view") != "payoff":
            return no_update, no_update, no_update
        shock = {"spot_pct": spot_pct or 0.0, "vol_pts": vol_pts or 0.0,
                 "rate_bps": rate_bps or 0.0, "time_days": int(time_days or 0)}
        res = sa.price_payoff(ds.get("account"), structure_id=ds.get("structure_id"),
                              position_id=ds.get("position_id"), shock=shock)
        if res is None:
            return no_update, no_update, no_update
        return (payoff_figure(res, show_components=bool(components)),
                economics_block(res), greeks_block(res))
