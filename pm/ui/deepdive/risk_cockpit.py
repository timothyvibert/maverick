"""Section — Risk. The account's one risk destination.

Consolidates what used to live across the Exposure, Scenario and Analytics
sections into a single full-width section whose top-to-bottom order is the
client conversation: pick the scenario → the stressed P&L → how every exposure
changes (the reshape table) → the standing book facts it draws on (current
exposures, carry, obligations, concentration) → the drill layer (heatmap +
impact table) and the structure→account rollup.

Numbers only. Every caption here is definitional (what a basis or a denominator
IS) — no interpretation, no recommendations, no generated summaries. The reshape
table shows the current and stressed exposure profiles side by side; the reader
draws the conclusion.

Reads state and calls pure aggregations at render; the only recompute behind the
dials is the sanctioned read-only ``price_scenario`` (no Bloomberg, no reload).
Each block renders independently: one block's failure degrades to an honest
error panel, never a blank Tab 2.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.risk.concentration import concentration_lenses
from pm.risk.obligations import assignment_obligations
from pm.risk.scenario import ShockSpec, shock_reprice, spot_vol_grid
from pm.ui.deepdive.aggregations import _fmt_money, long_short_premium_split
from pm.ui.deepdive.bars import bar_row
from pm.ui.deepdive.exposure import (
    _beta_panel,
    _headline_panel,
    _mv_vs_econ_panel,
    _rollup_table,
    _vega_tenor_row,
)
from pm.ui.deepdive.formatters import pct_of_nav
from pm.ui.deepdive.scenario import (
    _CAPTION,
    _controls,
    _heatmap_fig,
    _impact_table,
    _sign_cls,
    _total_line,
)

# The reshape table's metric rows: (label, exposures key, basis sub-label,
# %-of-NAV decimal places — the second-order greeks read in hundredths).
_RESHAPE_ROWS = [
    ("Net Δ$", "dd", "economic (delta-$)", 1),
    ("Net market exposure (β-$)", "dbeta", "SPX beta-mapped", 1),
    ("Net Γ$", "dg_1pct", "Δ$ per 1% spot move", 2),
    ("Net ν$", "dv", "per 1 vol pt", 2),
    ("Net θ$", "dt_bd", "per business day", 2),
]

_RESHAPE_CAPTION = (
    "Both columns are engine-priced through the same repricer (fast BS2002) — the "
    "zero shock is the current state, so Change is the shock's effect, never a "
    "live greek differenced against a recomputed one. Γ$ per 1% spot move; θ per "
    "business day; a name with no SPX beta prices at β = 1. Account scope — the "
    "Target drill moves the heatmap, not this table. The snapshot panels below "
    "read the Bloomberg greeks and differ by the engine reconciliation (~1–2% "
    "on Δ/ν).")


def _stat(label: str, value: str, sub: Optional[str] = None, cls: str = "") -> html.Div:
    children = [html.Div(label, className="dd-stat-label"),
                html.Div(value, className="dd-stat-value")]
    if sub:
        children.append(html.Div(sub, className="dd-stat-sub"))
    return html.Div(className=f"dd-stat {cls}".strip(), children=children)


def _num_cls(v: Optional[float]) -> str:
    base = "exposure-num"
    if v is None or v == 0:
        return base
    return f"{base} {'exposure-pos' if v > 0 else 'exposure-neg'}"


def _safe(build, label: str):
    """Per-block isolation: a failed block renders an honest error panel instead
    of freezing the whole populate callback (Tab 2 is one all-or-nothing repaint)."""
    try:
        return build()
    except Exception as exc:  # noqa: BLE001
        return html.Div(className="dd-panel risk-block-error", children=[
            html.H3(label, className="dd-panel-title"),
            html.Div(f"{label} failed to render — see Load notes. ({type(exc).__name__})",
                     className="dd-empty"),
        ])


# ---------------------------------------------------------------------------
# The reshape table (block 3) — also rebuilt by the dial callback
# ---------------------------------------------------------------------------

def _reshape_cell(v: Optional[float], nav, dp: int, signed_cls: bool = False) -> html.Td:
    pct = pct_of_nav(v, nav, dp=dp)
    children = [html.Span(_fmt_money(v),
                          className=("risk-reshape-num " + _sign_cls(v)) if signed_cls
                          else "risk-reshape-num")]
    if pct is not None:
        children.append(html.Span(f" ({pct})", className="risk-reshape-pct"))
    return html.Td(children, className="risk-reshape-cell")


def _reshape_table(exposures: Optional[dict], nav) -> html.Div:
    """Current vs stressed exposure profile, side by side. Data only — the gap
    between the columns is the reader's to act on."""
    title = html.H3("Exposure profile — current vs stressed", className="dd-panel-title")
    if not exposures or not exposures.get("n_legs"):
        return html.Div(className="dd-panel risk-reshape", children=[
            title,
            html.Div("Reshape unavailable — no priceable legs (market data missing).",
                     className="dd-empty")])
    now, stressed = exposures["now"], exposures["stressed"]
    head = html.Tr([html.Th("", className="am-th"),
                    html.Th("Now", className="am-th risk-reshape-colhead"),
                    html.Th("Stressed", className="am-th risk-reshape-colhead"),
                    html.Th("Change", className="am-th risk-reshape-colhead")])
    body = []
    for label, key, basis, dp in _RESHAPE_ROWS:
        v0, v1 = now.get(key), stressed.get(key)
        change = (v1 - v0) if (v0 is not None and v1 is not None) else None
        body.append(html.Tr(className="am-row", children=[
            html.Td([html.Span(label), html.Span(basis, className="risk-reshape-basis")],
                    className="risk-reshape-metric"),
            _reshape_cell(v0, nav, dp),
            _reshape_cell(v1, nav, dp),
            _reshape_cell(change, nav, dp, signed_cls=True),
        ]))
    n_legs = exposures.get("n_legs") or 0
    n_skipped = exposures.get("n_skipped") or 0
    coverage = (f"{n_legs} of {n_legs + n_skipped} legs priced — "
                f"{n_skipped} skipped (unpriceable)" if n_skipped
                else f"{n_legs} legs priced")
    return html.Div(className="dd-panel risk-reshape", children=[
        title,
        html.Table(className="am-table risk-reshape-table",
                   children=[html.Thead(head), html.Tbody(body)]),
        html.Div(coverage, className="dd-panel-note risk-reshape-coverage"),
        html.Div(_RESHAPE_CAPTION, className="dd-panel-note"),
    ])


# ---------------------------------------------------------------------------
# Carry (block 4, beside the moved exposure panels)
# ---------------------------------------------------------------------------

def _premium_panel(account_state) -> html.Div:
    s = long_short_premium_split(account_state)
    short_pct = s["short_share"]
    bar = html.Div(className="dd-split-bar", children=[
        html.Div(className="dd-split-collected",
                 style={"width": f"{(short_pct or 0) * 100:.1f}%"}),
        html.Div(className="dd-split-paid",
                 style={"width": f"{(1 - (short_pct or 0)) * 100:.1f}%"}),
    ]) if s["total"] else None
    return html.Div(className="dd-panel", children=[
        html.H3("Options premium — collected vs paid", className="dd-panel-title"),
        html.Div(className="dd-stat-row", children=[
            _stat("Collected (short)", _fmt_money(s["collected"]),
                  f"{s['n_short']} legs", cls="dd-stat-pos"),
            _stat("Paid (long)", _fmt_money(s["paid"]),
                  f"{s['n_long']} legs", cls="dd-stat-neg"),
            _stat("Net", _fmt_money(s["net"]), s["posture"]),
        ]),
        bar,
        html.Div(s["interpretation"], className="dd-panel-note"),
    ])


# ---------------------------------------------------------------------------
# Obligations (block 5)
# ---------------------------------------------------------------------------

def _oblig_row(label: str, side, nav, sub: Optional[html.Div] = None) -> html.Tr:
    itm_d = side.itm_dollars
    has = bool(side.n_positions)
    pct = pct_of_nav(side.dollars, nav) if has else None
    cells = [
        html.Td([html.Span(label)] + ([sub] if sub is not None else []),
                className="risk-oblig-side"),
        html.Td(f"{side.contracts:,.0f}" if has else "—", className="exposure-num"),
        html.Td(_fmt_money(side.dollars) if has else "—", className="exposure-num"),
        html.Td(pct if pct is not None else "—", className="exposure-num"),
        html.Td("—" if itm_d is None else f"{side.itm_contracts:,.0f}",
                className="exposure-num"),
        html.Td("—" if itm_d is None else _fmt_money(itm_d), className="exposure-num"),
    ]
    return html.Tr(className="am-row", children=cells)


def _oblig_windows(ob) -> html.Div:
    header = html.Div(className="dd-ladder-row dd-ladder-head", children=[
        html.Span("Window"),
        *[html.Span(w["label"]) for w in ob.puts.by_window]])
    puts = html.Div(className="dd-ladder-row", children=[
        html.Span("Puts — cash", className="dd-ladder-bucket"),
        *[html.Span(_fmt_money(w["dollars"]) if w["dollars"] else "—",
                    className="dd-ladder-count") for w in ob.puts.by_window]])
    calls = html.Div(className="dd-ladder-row", children=[
        html.Span("Calls — at strike", className="dd-ladder-bucket"),
        *[html.Span(_fmt_money(w["dollars"]) if w["dollars"] else "—",
                    className="dd-ladder-count") for w in ob.calls.by_window]])
    return html.Div(className="dd-ladder risk-oblig-ladder", children=[header, puts, calls])


def _obligations_panel(account_state, ob, nav) -> html.Div:
    title = html.H3("Assignment obligations — if assigned", className="dd-panel-title")
    if not ob.puts.n_positions and not ob.calls.n_positions:
        children = [title,
                    html.Div("No short options on the book — nothing to assign.",
                             className="dd-empty")]
        if ob.n_expired:
            children.append(html.Div(
                f"Expired ({ob.n_expired}) — dead contract(s) still on the book "
                "(stale extract); excluded from every figure here.",
                className="dd-panel-note"))
        return html.Div(className="dd-panel risk-oblig", children=children)

    covered_sub = None
    if ob.calls.n_positions:
        covered_sub = html.Div(
            f"covered {ob.covered_call_contracts:,.0f} · "
            f"uncovered {ob.uncovered_call_contracts:,.0f} (no covering stock)",
            className="risk-oblig-sub")
    head = html.Tr([html.Th(h, className="am-th") for h in
                    ("", "Contracts", "Obligation", "% NAV", "ITM", "ITM obligation")])
    table = html.Table(className="am-table risk-oblig-table", children=[
        html.Thead(head),
        html.Tbody([
            _oblig_row("Short puts — cash to fund", ob.puts, nav),
            _oblig_row("Short calls — deliver at strike", ob.calls, nav, covered_sub),
        ])])
    children = [
        title,
        html.Div("Strike value of every short option if assigned — short legs only; "
                 "a long option is a right, not an obligation.",
                 className="dd-panel-subtitle"),
        table,
        _oblig_windows(ob),
    ]
    n_unknown = ob.puts.n_unknown_moneyness + ob.calls.n_unknown_moneyness
    if n_unknown:
        children.append(html.Div(
            f"Spot missing on {n_unknown} short option(s) — the ITM subtotal "
            "excludes them.", className="dd-panel-note"))
    if ob.n_expired:
        children.append(html.Div(
            f"Expired ({ob.n_expired}) — dead contract(s) still on the book "
            "(stale extract); excluded from every figure here.",
            className="dd-panel-note"))
    return html.Div(className="dd-panel risk-oblig", children=children)


# ---------------------------------------------------------------------------
# Concentration (block 6)
# ---------------------------------------------------------------------------

_CONC_TOP_N = 10
_CONC_COLS = ("Name", "Net Δ$", "% NAV", "Gross |Δ$|", "Γ$ (1%)", "ν$",
              "Put oblig.", "Call oblig.", "MV")


def _conc_cells(r, nav) -> list:
    return [
        html.Td(_fmt_money(r["net_dollar_delta"]), className=_num_cls(r["net_dollar_delta"])),
        html.Td(pct_of_nav(r["net_dollar_delta"], nav) or "—", className="exposure-num"),
        html.Td(_fmt_money(r["gross_dollar_delta"]), className="exposure-num"),
        html.Td(_fmt_money(r["dollar_gamma"]), className=_num_cls(r["dollar_gamma"])),
        html.Td(_fmt_money(r["dollar_vega"]), className=_num_cls(r["dollar_vega"])),
        html.Td(_fmt_money(r["put_obligation"]) if r["put_obligation"] else "—",
                className="exposure-num"),
        html.Td(_fmt_money(r["call_obligation"]) if r["call_obligation"] else "—",
                className="exposure-num"),
        html.Td(_fmt_money(r["market_value"]), className=_num_cls(r["market_value"])),
    ]


def _others_row(rest: list, nav) -> Optional[html.Tr]:
    if not rest:
        return None

    def _tot(key):
        vals = [r[key] for r in rest if r[key] is not None]
        return sum(vals) if vals else None

    agg = {k: _tot(k) for k in ("net_dollar_delta", "gross_dollar_delta", "dollar_gamma",
                                "dollar_vega", "put_obligation", "call_obligation",
                                "market_value")}
    return html.Tr(className="am-row risk-conc-others", children=[
        html.Td(f"Other ({len(rest)})", className="risk-conc-name")] + _conc_cells(agg, nav))


def _concentration_table(lenses: dict) -> html.Div:
    nav = lenses["nav"]
    rows = lenses["rows"]
    head = html.Tr([html.Th(c, className="am-th") for c in _CONC_COLS])
    body = []
    for r in rows[:_CONC_TOP_N]:
        body.append(html.Tr(className="am-row", children=[
            html.Td(r["symbol"], className="risk-conc-name")] + _conc_cells(r, nav)))
    other = _others_row(rows[_CONC_TOP_N:], nav)
    if other is not None:
        body.append(other)
    acc = dict(lenses["account"])
    body.append(html.Tr(className="am-row exposure-row-total", children=[
        html.Td("Account", className="risk-conc-name")] + _conc_cells(acc, nav)))
    children = [
        html.H3("Concentration — by name, every basis", className="dd-panel-title"),
        html.Div("Sorted by gross |Δ$| — the lens a hedge cannot empty. Net nets "
                 "options against stock; obligations are the short-strike dollars.",
                 className="dd-panel-subtitle"),
        html.Table(className="am-table risk-conc-table",
                   children=[html.Thead(head), html.Tbody(body)]),
    ]
    missing = lenses["missing"]
    if missing["n_rows"]:
        names = missing["names"]
        shown = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
        children.append(html.Div(
            f"Delta missing on {missing['n_rows']} position(s) across {len(names)} "
            f"name(s) — their Δ columns dash and the totals understate: {shown}.",
            className="dd-panel-note"))
    if not rows:
        children.append(html.Div("No name exposure to show.", className="dd-empty"))
    return html.Div(className="dd-panel risk-conc", children=children)


def _asset_class_table(lenses: dict) -> html.Div:
    nav = lenses["nav"]
    head = html.Tr([html.Th(c, className="am-th") for c in
                    ("Class", "MV", "% NAV", "Net Δ$", "Positions")])
    label = {"equity": "Equity", "fund_etf": "Fund / ETF", "option": "Options",
             "cash": "Cash", "other": "Other"}
    body = [html.Tr(className="am-row", children=[
        html.Td(label.get(s["asset_class"], s["asset_class"]), className="risk-conc-name"),
        html.Td(_fmt_money(s["market_value"]), className=_num_cls(s["market_value"])),
        html.Td(pct_of_nav(s["market_value"], nav) or "—", className="exposure-num"),
        html.Td(_fmt_money(s["net_dollar_delta"]), className=_num_cls(s["net_dollar_delta"])),
        html.Td(str(s["n_positions"]), className="exposure-num"),
    ]) for s in lenses["asset_class_split"]]
    return html.Div(className="dd-panel risk-split", children=[
        html.H3("Asset-class split", className="dd-panel-title"),
        html.Div("Fund/ETF is the closest held-instrument read on index exposure; "
                 "options group under their own class here and under their "
                 "underlying in the name table.", className="dd-panel-subtitle"),
        html.Table(className="am-table risk-split-table",
                   children=[html.Thead(head), html.Tbody(body)]),
    ])


def _missing_delta_note(account_state) -> Optional[html.Div]:
    from pm.risk.exposure import economic_exposure_missing
    missing = economic_exposure_missing(account_state)
    if not missing["n_rows"]:
        return None
    names = missing["names"]
    shown = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
    return html.Div(
        f"Delta missing on {missing['n_rows']} position(s) across "
        f"{len(names)} name(s) — excluded from these bars: {shown}.",
        className="dd-panel-note")


def _sector_panel(account_state) -> html.Div:
    from pm.risk.exposure import economic_exposure_by_sector
    items = economic_exposure_by_sector(account_state)  # sorted by |delta-$| desc
    max_w = max((abs(r["pct_nav"] or 0) for r in items), default=0)
    bars = [bar_row(r["sector"], r["pct_nav"], max_w) for r in items]
    if not bars:
        bars = [html.Div("No economic exposure to show.", className="dd-empty")]
    children = [
        html.H3("Sector breakdown", className="dd-panel-title"),
        html.Div("Economic exposure (delta-$) by sector, signed % NAV — options "
                 "included and netted against stock.", className="dd-panel-subtitle"),
        html.Div(className="dd-bars", children=bars),
    ]
    note = _missing_delta_note(account_state)
    if note is not None:
        children.append(note)
    return html.Div(className="dd-panel", children=children)


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------

def render_risk_section(account_state, state) -> html.Div:
    head = html.Div(className="dd-section-head", children=[
        html.H2("Risk", className="dd-section-title"),
        html.Span("scenario · exposures · obligations · concentration",
                  className="dd-section-meta"),
    ])
    if account_state is None or state is None:
        return html.Div(className="dd-section", children=[
            head, html.Div("Risk views unavailable.", className="dd-empty")])

    nav = getattr(account_state, "nav", None)
    e = getattr(account_state, "exposure", None)

    # initial zero-shock view (pure functions; first paint correct without a callback)
    zero = ShockSpec("base", "base")
    try:
        impact = shock_reprice(state, account_state, zero, mode="fast")
        grid = spot_vol_grid(state, account_state)
    except Exception:  # noqa: BLE001
        impact, grid = None, None

    ob = assignment_obligations(account_state)
    lenses = concentration_lenses(account_state, obligations=ob)

    # 1-3: the scenario row, the answer strip, the reshape table
    if impact is not None:
        scenario_blocks = [
            _controls(account_state, impact["rows"]),
            html.Div(className="risk-answer", children=[
                html.Div(id="scn-total", children=_total_line(impact))]),
            html.Div(id="risk-reshape",
                     children=_reshape_table(impact.get("exposures"), nav)),
        ]
        drill = html.Div(className="scn-body", children=[
            html.Div(className="scn-heatmap-wrap", children=[
                _lazy_graph(grid)]),
            html.Div(className="scn-impact-wrap", children=[
                html.Div(id="scn-impact", children=_impact_table(impact["rows"], "account")),
            ]),
        ])
        drill_caption = html.Div(className="scenario-caption", children=[_CAPTION])
    else:
        scenario_blocks = [html.Div(
            "Scenario views unavailable (Bloomberg off or no priceable options).",
            className="dd-empty")]
        drill = None
        drill_caption = None

    # 4: the current book (snapshot basis) + carry
    if e is not None:
        current_book = html.Div(className="dd-analytics-grid", children=[
            _safe(lambda: _headline_panel(e), "Net market exposure"),
            _safe(lambda: _mv_vs_econ_panel(e), "Market value vs economic exposure"),
            _safe(lambda: _beta_panel(e), "Beta"),
            _safe(lambda: _vega_tenor_row(e), "Vega by tenor"),
            _safe(lambda: _premium_panel(account_state), "Options premium"),
        ])
        rollup = _safe(lambda: _rollup_table(e), "Exposure rollup")
    else:
        current_book = html.Div("Exposure unavailable for this account.",
                                className="dd-empty")
        rollup = None

    children = [head, *scenario_blocks,
                current_book,
                _safe(lambda: _obligations_panel(account_state, ob, nav),
                      "Assignment obligations"),
                html.Div(className="dd-analytics-grid", children=[
                    _safe(lambda: _concentration_table(lenses), "Concentration"),
                    _safe(lambda: _asset_class_table(lenses), "Asset-class split"),
                    _safe(lambda: _sector_panel(account_state), "Sector breakdown"),
                ])]
    if drill is not None:
        children.extend([drill, drill_caption])
    if rollup is not None:
        children.append(rollup)
    children.append(html.Div(
        "% of NAV = value ÷ |net asset value|. Current-book panels read the "
        "Bloomberg snapshot greeks; the reshape table and the drill are "
        "engine-priced (fast BS2002).", className="dd-panel-note risk-footer"))
    return html.Div(className="dd-section risk-cockpit", children=children)


def _lazy_graph(grid):
    from dash import dcc
    return dcc.Graph(id="scn-heatmap", figure=_heatmap_fig(grid, 0.0, 0.0),
                     config={"displayModeBar": False, "responsive": True},
                     className="scenario-graph")
