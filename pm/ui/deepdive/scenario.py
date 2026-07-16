"""Section — Scenario / stress, dense + interactive.

A shock-control row + preset chips drive the one sanctioned ``price_scenario``
recompute (fast vectorized BS2002, read-only over loaded state). The centerpiece is
the spot×vol P&L **heatmap** (token diverging scale around 0, never default plotly;
its zero-vol row is the P&L-vs-SPX line, and the 2-D surface shows the crash corner);
beside it a per-position/structure **impact table** (P&L@shock + dollar greeks,
sign-colored, click-a-row to drill the heatmap to that position's surface).

Layout: controls row over [heatmap | impact-table] — dense, no vertical sprawl. The
initial (zero-shock) view is computed at render so first paint is correct; dialing /
presets / drill repaint via the callbacks in ``callbacks.py``. plotly is imported
lazily in the figure builder. Token-styled throughout (no new palette).
"""
from __future__ import annotations

from typing import Optional

from dash import dcc, html

from pm.risk.scenario import (GRID_VOL_PTS, ShockSpec, shock_reprice, spot_vol_grid)
from pm.ui.deepdive.aggregations import _fmt_money

# token hex mirroring assets/style.css (:root) — plotly needs explicit colors.
_CHARCOAL = "#2B2B2B"
_POS = "#1E7E34"
_NEG = "#C62828"
_NEUTRAL = "#F5F5F5"
_GRID = "#E8E8E8"
_AMBER = "#B7791F"
_MUTED = "#6E6E6E"
_FONT = '"Frutiger 45 Light","Frutiger","Helvetica Neue","Segoe UI",Arial,sans-serif'

# preset chips -> (spx %, vol pts, rate bps, time days). Spot/vol-plane presets also
# render as diamond markers on the heatmap (see _PLANE_PRESETS).
PRESETS = [
    ("crash", "Crash", -20.0, 10.0, 0.0, 0),
    ("meltup", "Melt-up", 15.0, -5.0, 0.0, 0),
    ("spx_dn", "SPX -10%", -10.0, 0.0, 0.0, 0),
    ("spx_up", "SPX +10%", 10.0, 0.0, 0.0, 0),
    ("vol_up", "Vol +10", 0.0, 10.0, 0.0, 0),
    ("vol_dn", "Vol -10", 0.0, -10.0, 0.0, 0),
    ("rates_up", "Rates +50", 0.0, 0.0, 50.0, 0),
    ("rates_dn", "Rates -50", 0.0, 0.0, -50.0, 0),
    ("reset", "Reset", 0.0, 0.0, 0.0, 0),
]
PRESET_AXES = {name: (sp, vp, rb, td) for name, _, sp, vp, rb, td in PRESETS}
# (spot%, vol pts) of the presets that live on the spot×vol plane -> heatmap diamonds.
_PLANE_PRESETS = [(-20.0, 0.0), (-10.0, 0.0), (-5.0, 0.0), (5.0, 0.0), (10.0, 0.0),
                  (20.0, 0.0), (0.0, 10.0), (0.0, 5.0), (0.0, -5.0), (0.0, -10.0),
                  (-20.0, 10.0), (15.0, -5.0)]


def _sign_cls(v: Optional[float]) -> str:
    if v is None or v == 0:
        return ""
    return "scenario-pos" if v > 0 else "scenario-neg"


# The section's provenance line. Every rendered number — heatmap, impact table,
# preset chips (they only set the sliders) — comes from the fast pricer; the
# caption must never attribute rendered output to a tier that isn't used.
_CAPTION = (
    "All numbers here — heatmap, impact table, presets — are fast vectorized "
    "BS2002 (β-mapped SPX × vol shift, P&L vs current); ● current shock, "
    "◆ preset points. Γ$ is engine dollar-gamma per $1 spot move — a different "
    "basis from the Exposure section's per-1% γ; do not compare. θ is engine "
    "per-business-day (diverges from the BBG snapshot θ). Dial recomputes live, "
    "no Bloomberg.")


def render_scenario_section(account_state, state) -> html.Div:
    head = html.Div(className="dd-section-head", children=[
        html.H2("Scenario & stress", className="dd-section-title"),
        html.Span("hypothetical-state · engine-priced · dial live", className="dd-section-meta"),
    ])
    if account_state is None or state is None:
        return html.Div(className="dd-section", children=[
            head, html.Div("Scenario views unavailable (Bloomberg off or no priceable options).",
                           className="dd-empty")])

    # initial zero-shock view (pure functions; first paint correct without a callback)
    zero = ShockSpec("base", "base")
    impact = shock_reprice(state, account_state, zero, mode="fast")
    grid = spot_vol_grid(state, account_state)

    return html.Div(className="dd-section", children=[
        head,
        _controls(account_state, impact["rows"]),
        html.Div(className="scn-body", children=[
            html.Div(className="scn-heatmap-wrap", children=[
                dcc.Graph(id="scn-heatmap", figure=_heatmap_fig(grid, 0.0, 0.0),
                          config={"displayModeBar": False, "responsive": True},
                          className="scenario-graph")]),
            html.Div(className="scn-impact-wrap", children=[
                html.Div(id="scn-impact", children=_impact_table(impact["rows"], "account")),
                html.Div(id="scn-total", children=_total_line(impact)),
            ]),
        ]),
        html.Div(className="scenario-caption", children=[_CAPTION]),
    ])


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
def _controls(account_state, rows) -> html.Div:
    targets = [{"label": "Account", "value": "account"}]
    for st in getattr(account_state, "structures", []) or []:
        sid = getattr(st, "structure_id", None)
        if sid:
            targets.append({"label": f"⋯ {getattr(st, 'type', 'structure')}", "value": f"structure:{sid}"})
    for r in rows:
        targets.append({"label": r["label"], "value": r["id"]})

    def _slider(_id, lo, hi, step, suffix):
        marks = {int(v): f"{int(v)}{suffix}" for v in (lo, lo / 2, 0, hi / 2, hi)}
        # allow_direct_input=False: the slider's built-in entry box moves the thumb
        # on Enter WITHOUT committing the value to the server — a typed shock would
        # render as applied while the book never repriced, and text left in the box
        # survives a preset reset and re-commits on the next blur (a phantom shock).
        # The paired dcc.Input in _dial below is the committed typed gesture instead.
        return dcc.Slider(id=_id, min=lo, max=hi, step=step, value=0, marks=marks,
                          tooltip={"placement": "bottom", "always_visible": False},
                          allow_direct_input=False, className="scn-slider")

    def _dial(label, _id, lo, hi, step, suffix=""):
        # Slider + explicit numeric entry, kept in lockstep by the per-dial sync
        # callback. debounce=False is load-bearing: every keystroke commits, the
        # slider visibly tracks the typed value, and NO uncommitted text can ever
        # sit in the box — which is what made a blur after a preset reset silently
        # reprice a shock the user believed cleared.
        return html.Div(className="scn-ctrl", children=[
            html.Label(label, className="scn-ctrl-lbl"),
            html.Div(className="scn-dial", children=[
                _slider(_id, lo, hi, step, suffix),
                dcc.Input(id=f"{_id}-num", type="number", value=0, step=step,
                          debounce=False, className="scn-num"),
            ]),
        ])

    return html.Div(className="scn-controls", children=[
        _dial("SPX / spot %", "scn-spx", -20, 20, 1),
        _dial("Vol shift (pts)", "scn-vol", -10, 10, 0.5),
        _dial("Rate shift (bps)", "scn-rate", -50, 50, 5),
        _dial("Time (days fwd)", "scn-time", 0, 90, 1, "d"),
        html.Div(className="scn-ctrl scn-ctrl-narrow", children=[
            html.Label("Target", className="scn-ctrl-lbl"),
            dcc.Dropdown(id="scn-target", options=targets, value="account", clearable=False,
                         className="scn-target")]),
        html.Div(className="scn-presets", children=[
            html.Span("Presets", className="scn-ctrl-lbl"),
            *[html.Button(lbl, id={"type": "scn-preset", "name": name}, n_clicks=0,
                          className="scn-chip" + (" scn-chip-reset" if name == "reset" else ""))
              for name, lbl, *_ in PRESETS]]),
    ])


# --------------------------------------------------------------------------
# Figure + table builders (shared by render + the callbacks)
# --------------------------------------------------------------------------
def _heatmap_fig(grid, spot_pct, vol_pts, target_label=None):
    import plotly.graph_objects as go          # lazy

    z, x, y = grid["pnl_matrix"], grid["spot_axis"], grid["vol_axis"]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, zmid=0,
        colorscale=[[0.0, _NEG], [0.5, _NEUTRAL], [1.0, _POS]],
        colorbar=dict(title=dict(text="P&L $", font=dict(size=10)), thickness=10,
                      tickfont=dict(size=9), outlinewidth=0),
        hovertemplate="SPX %{x:.0f}%<br>vol %{y:+.1f}pt<br>P&L %{z:$,.0f}<extra></extra>"))
    # preset diamonds on the spot×vol plane
    fig.add_trace(go.Scatter(
        x=[p[0] for p in _PLANE_PRESETS], y=[p[1] for p in _PLANE_PRESETS], mode="markers",
        marker=dict(symbol="diamond-open", size=8, color=_AMBER, line=dict(width=1)),
        name="presets", hoverinfo="skip"))
    # current shock point
    fig.add_trace(go.Scatter(
        x=[spot_pct], y=[vol_pts], mode="markers",
        marker=dict(symbol="circle", size=13, color=_CHARCOAL, line=dict(color="white", width=1.5)),
        name="current", hovertemplate="current shock<br>SPX %{x:.1f}%, vol %{y:+.1f}pt<extra></extra>"))
    title = "P&L surface — " + (target_label or "Account")
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=_CHARCOAL), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=_CHARCOAL, size=11),
        margin=dict(l=52, r=10, t=30, b=40), height=360, showlegend=False,
        xaxis=dict(title="SPX move %", gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID),
        yaxis=dict(title="Vol shift (pts)", gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID))
    return fig


def _impact_table(rows, target):
    head = html.Tr(className="scn-impact-head", children=[
        html.Th("Position / structure"), html.Th("P&L"), html.Th("Δ$"),
        html.Th("Γ$", title="engine dollar-gamma per $1 spot move — different basis "
                            "from the Exposure section's per-1% γ"),
        html.Th("ν$"), html.Th("θ$")])
    body = []
    for r in rows:                               # already ranked worst-first
        active = " scn-impact-active" if (target and target == r["id"]) else ""
        body.append(html.Tr(
            id={"type": "scn-drill", "id": r["id"]}, n_clicks=0,
            className="scn-impact-row" + active, children=[
                html.Td(r["label"], className="scn-impact-name", title="click to drill the surface"),
                html.Td(_fmt_money(r["pnl"]), className=f"scn-impact-num {_sign_cls(r['pnl'])}"),
                html.Td(_fmt_money(r["dd"]), className="scn-impact-num"),
                html.Td(_fmt_money(r["dg"]), className="scn-impact-num"),
                html.Td(_fmt_money(r["dv"]), className="scn-impact-num"),
                html.Td(_fmt_money(r["dt"]), className="scn-impact-num"),
            ]))
    return html.Table(className="scn-impact-table", children=[html.Thead(head), html.Tbody(body)])


def _total_line(impact) -> html.Div:
    pnl = impact["account_pnl"]
    pct = impact["account_pnl_pct"]
    pct_s = "" if pct is None else f"  ({pct * 100:+.2f}% NAV)"
    # Coverage honesty: the total covers only the priceable book. With nothing
    # priceable the $0 total is vacuous — say so instead of showing it; with a
    # partial book, disclose the n-of-m coverage beside the number.
    n_priced = impact.get("n_priced")
    n_skipped = impact.get("n_skipped") or 0
    children = [html.Span("Account P&L @ shock", className="scn-total-lbl")]
    if n_skipped and not n_priced:
        children.append(html.Span("— no priceable legs (market data missing)",
                                  className="scn-total-val"))
    else:
        children.append(html.Span(_fmt_money(pnl) + pct_s,
                                  className=f"scn-total-val {_sign_cls(pnl)}"))
        if n_skipped and n_priced:
            children.append(html.Span(
                f"{n_priced} of {n_priced + n_skipped} legs priced — "
                f"{n_skipped} skipped (unpriceable)",
                className="scn-total-coverage"))
    return html.Div(className="scn-total", children=children)
