"""Section — Risk. The account's one risk destination, in the terminal idiom.

Presentation follows the Morning Blotter's system, not a card dashboard: a
block is an 11px uppercase band label over a 1px rule with content sitting
directly on the page — no bordered containers. Four type sizes only (19px
headline numbers, 15px secondary numbers, 13px grid data, 11px labels).
Sentences don't render: every qualifier is a badge whose full text lives in a
native ``title`` tooltip; the only visible words are labels, values and
inline coverage cues.

Page order: posture band → scenario dials (fenced) → the P&L ribbon → the
change-first convexity strip → heatmap | per-position impact → the
what's-coming event timeline (Plotly, Python-stacked) | event detail grid →
concentration | standing obligations.

Reads state and calls pure aggregations at render; the only recompute behind
the dials is the sanctioned read-only ``price_scenario`` (no Bloomberg, no
reload). Each block renders independently: one block's failure degrades to an
honest error line, never a blank Tab 2.
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Optional

import dash_ag_grid as dag
from dash import html

from pm.risk.concentration import concentration_lenses
from pm.risk.obligations import assignment_obligations
from pm.risk.scenario import ShockSpec, shock_reprice, spot_vol_grid
from pm.risk.upcoming import upcoming_events
from pm.ui.deepdive.aggregations import _fmt_money
from pm.ui.deepdive.exposure import _BETA_NOTE, _bucket_cell, _provenance
from pm.ui.deepdive.formatters import (
    MONEY_FULL_FMT,
    PCT_ABS_FMT,
    PCT_SIGNED_1DP_FMT,
    SIGNED_COLOR_STYLE,
    pct_of_nav,
)
from pm.ui.deepdive.scenario import (
    _AMBER,
    _CAPTION,
    _CHARCOAL,
    _FONT,
    _GRID,
    _MUTED,
    _controls,
    _heatmap_fig,
    _impact_table,
    _sign_cls,
    _total_line,
)

# The convexity strip's cells: (label, exposures key, basis sub-label).
_RESHAPE_ROWS = [
    ("Net Δ$", "dd", "economic (delta-$)"),
    ("Net Γ$", "dg_1pct", "Δ$ per 1% spot move"),
    ("Net ν$", "dv", "per 1 vol pt"),
    ("Net θ$", "dt_bd", "per business day"),
]

_RESHAPE_CAPTION = (
    "Change @ shock, then the current and stressed levels it moves between. "
    "Both states are engine-priced through the same repricer (fast BS2002) — "
    "the zero shock is the current state, so Change is the shock's effect, "
    "never a live greek differenced against a recomputed one. Γ$ per 1% spot "
    "move; θ per business day; a name with no SPX beta is excluded from spot "
    "shocks (counted and named on the badge, never priced at a default). "
    "Account scope — the Target drill moves the heatmap, not this strip. The "
    "posture band reads the Bloomberg snapshot greeks and differs by the "
    "engine reconciliation (~1–2% on Δ/ν).")

_NAV_TIP = ("% of NAV = value ÷ |net asset value| — one denominator on every "
            "risk surface, the account's signed net asset value from the "
            "extract, taken absolute.")


# ---------------------------------------------------------------------------
# The band idiom: label + ⓘ + right-aligned badges over a 1px rule
# ---------------------------------------------------------------------------

def _info(tip: str) -> html.Span:
    """The ⓘ affordance: a plain span with a native title tooltip."""
    return html.Span("ⓘ", className="risk-info", title=tip)


def _badge(text: str, tip: str, warn: bool = False) -> html.Span:
    """A qualifier as a badge — the sentence lives in the title, not the page."""
    cls = "risk-badge risk-badge-warn" if warn else "risk-badge"
    return html.Span(text, className=cls, title=tip)


def _band(label: str, tip: Optional[str] = None, badges: Optional[list] = None) -> html.Div:
    children: list = [html.Span(label, className="risk-band-label")]
    if tip:
        children.append(_info(tip))
    if badges:
        children.append(html.Div(className="risk-badges", children=badges))
    return html.Div(className="risk-band", children=children)


def _cols(left: list, right: list) -> html.Div:
    """One two-column row — the section's rhythm below the strips."""
    return html.Div(className="risk-cols", children=[
        html.Div(className="risk-col", children=left),
        html.Div(className="risk-col", children=right),
    ])


def _quiet(build):
    """A pre-compute that degrades to None on failure; the consuming _safe
    blocks then say so per block."""
    try:
        return build()
    except Exception:  # noqa: BLE001
        return None


def _safe(build, label: str):
    """Per-block isolation: a failed block renders an honest error line instead
    of freezing the whole populate callback (Tab 2 is one all-or-nothing repaint)."""
    try:
        return build()
    except Exception as exc:  # noqa: BLE001
        return html.Div(
            f"{label} failed to render — see Load notes. ({type(exc).__name__})",
            className="dd-empty risk-block-error")


# ---------------------------------------------------------------------------
# 1 — the posture band (standing book in four numbers)
# ---------------------------------------------------------------------------

_DIRECTION_TIP = (
    "Net market exposure: Σ position dollar-delta × the name's SPX beta — the "
    "book's SPX-equivalent dollars. β shown = net β-$ ÷ net Δ$, the delta "
    "book's exposure-weighted SPX beta (adjusted). " + _BETA_NOTE)
_LEVERAGE_TIP = (
    "Economic exposure ÷ market value of the greek-bearing book (equities + "
    "options at their marks; cash and unpriced holdings excluded). Economic "
    "exposure is the delta-equivalent exposure to the underlyings — they "
    "diverge where an option's premium understates its directional exposure.")
_VOL_TIP = (
    "Net dollar vega per 1 vol point (Bloomberg snapshot greeks), with its term "
    "structure by days to expiry. A dashed bucket holds no options — or only "
    "options missing vega.")
_DECAY_TIP = (
    "Net dollar theta per calendar day (Bloomberg snapshot greeks); the monthly "
    "figure is 30 calendar days as % of |NAV|.")


def _posture_cell(label: str, value: str, subs: list, tip: str,
                  cls: str = "") -> html.Div:
    children = [html.Div([html.Span(label), _info(tip)], className="dd-stat-label"),
                html.Div(value, className="dd-stat-value")]
    children += [html.Div(s, className="dd-stat-sub") for s in subs if s]
    return html.Div(className=f"dd-stat {cls}".strip(), children=children)


def _stat_sign_cls(v: Optional[float]) -> str:
    if v is None or v == 0:
        return ""
    return "dd-stat-pos" if v > 0 else "dd-stat-neg"


def _posture_strip(e, nav) -> html.Div:
    t = e.total
    nme = e.net_market_exposure
    econ = e.economic_exposure
    mv = t.market_value

    direction = ("—" if nme is None
                 else "net long" if nme > 0 else "net short" if nme < 0 else "flat")
    beta = (nme / econ) if (nme is not None and econ) else None
    dir_sub = direction if beta is None else f"{direction} · β {beta:.2f}"

    ratio = (econ / mv) if (econ is not None and mv) else None
    lev_val = f"{ratio:.2f}×" if ratio is not None else "—"

    tenor = " · ".join(f"{b.label} {_bucket_cell(b)}" for b in e.vega_by_tenor)

    theta_mo = pct_of_nav(t.dollar_theta * 30 if t.dollar_theta is not None else None,
                          nav, dp=2)

    return html.Div(className="risk-posture", children=[
        _posture_cell("Direction", _fmt_money(nme),
                      [pct_of_nav(nme, nav), dir_sub, "β SPX 2y wkly · adjusted"],
                      _DIRECTION_TIP, cls=_stat_sign_cls(nme)),
        _posture_cell("Leverage", lev_val,
                      [f"{_fmt_money(econ)} economic vs {_fmt_money(mv)} MV"],
                      _LEVERAGE_TIP),
        _posture_cell("Volatility", _fmt_money(t.dollar_vega),
                      ["per 1 vol pt", tenor],
                      _VOL_TIP, cls=_stat_sign_cls(t.dollar_vega)),
        _posture_cell("Decay", _fmt_money(t.dollar_theta),
                      ["per calendar day",
                       f"{theta_mo}/mo" if theta_mo else None],
                      _DECAY_TIP, cls=_stat_sign_cls(t.dollar_theta)),
    ])


def _posture_badges(e) -> list:
    """The posture band's qualifiers: provenance always, missing-data marks
    only when something is actually missing. Sentences live in the titles."""
    badges = [_badge("source", _provenance(e) + " " + _NAV_TIP)]
    inputs = (e.trace or {}).get("inputs", {})
    missing_beta = inputs.get("names_missing_beta", []) or []
    if missing_beta:
        n_eligible = inputs.get("n_names_beta_eligible")
        mapped = (f"{n_eligible - len(missing_beta)} of {n_eligible} names beta-mapped; "
                  if isinstance(n_eligible, int) and n_eligible else "")
        badges.append(_badge(
            f"⚠ {len(missing_beta)} no β",
            f"{mapped}{len(missing_beta)} name(s) had no SPX beta and are excluded "
            f"from dollar-beta: {', '.join(missing_beta)}", warn=True))
    missing_greeks = getattr(e, "missing_greeks", []) or []
    if missing_greeks:
        badges.append(_badge(
            f"⚠ {len(missing_greeks)} no greeks",
            f"Greeks missing on {len(missing_greeks)} name(s) — totals "
            f"understate: {', '.join(missing_greeks)}", warn=True))
    n_missing_v = sum(getattr(b, "n_missing_vega", 0) or 0 for b in e.vega_by_tenor)
    n_opts = sum(b.n_options for b in e.vega_by_tenor)
    if n_missing_v:
        badges.append(_badge(
            f"⚠ vega {n_opts - n_missing_v}/{n_opts}",
            f"Vega missing on {n_missing_v} of {n_opts} option(s) — the tenor "
            "buckets understate by those positions.", warn=True))
    n_expired = getattr(e, "n_expired_options", 0) or 0
    if n_expired:
        badges.append(_badge(
            f"{n_expired} expired",
            f"Expired ({n_expired}) — dead contract(s) still on the book "
            "(stale extract); excluded from every figure here."))
    return badges


# ---------------------------------------------------------------------------
# 4 — the convexity strip (change-first; rebuilt by the dial callback)
# ---------------------------------------------------------------------------

def _reshape_table(exposures: Optional[dict], nav, shocked: bool = True) -> html.Div:
    """The convexity strip: four cells, the CHANGE @ shock leading, the
    current → stressed levels beneath. ``shocked=False`` (the resting page)
    dashes the change — a wall of $0 reads as a bug, not a zero shock. Keeps
    its historical name: it is the ``risk-reshape`` callback target."""
    if not exposures or not exposures.get("n_legs"):
        return html.Div("Convexity unavailable — no priceable legs "
                        "(market data missing).", className="dd-empty")
    now, stressed = exposures["now"], exposures["stressed"]
    cells = []
    for label, key, basis in _RESHAPE_ROWS:
        v0, v1 = now.get(key), stressed.get(key)
        change = (v1 - v0) if (v0 is not None and v1 is not None) else None
        if shocked:
            value = _fmt_money(change)
            val_cls = _sign_cls(change)
            pct = pct_of_nav(change, nav, dp=2)
            subs = [pct,
                    f"{_fmt_money(v0)} → {_fmt_money(v1)}"]
        else:
            value = "—"
            val_cls = ""
            subs = [f"now {_fmt_money(v0)}"]
        cells.append(html.Div(className="dd-stat", children=[
            html.Div([html.Span(label),
                      html.Span(f" · {basis}", className="risk-convexity-basis")],
                     className="dd-stat-label"),
            html.Div(value, className=f"dd-stat-value {val_cls}".strip()),
            *[html.Div(s, className="dd-stat-sub") for s in subs if s],
        ]))
    return html.Div(className="risk-convexity", children=cells)


def _coverage_badge(exposures: Optional[dict]) -> Optional[html.Span]:
    if not exposures:
        return None
    n_legs = exposures.get("n_legs") or 0
    n_skipped = exposures.get("n_skipped") or 0
    if not n_legs and not n_skipped:
        return None
    if n_skipped:
        return _badge(f"{n_legs}/{n_legs + n_skipped} priced",
                      f"{n_legs} of {n_legs + n_skipped} legs priced — "
                      f"{n_skipped} skipped (unpriceable)", warn=True)
    return _badge(f"{n_legs} legs priced", f"All {n_legs} legs priced.")


def _beta_excluded_badge(exposures: Optional[dict]) -> Optional[html.Span]:
    """The missing-beta exclusion, visible where the shocked numbers live: a
    count badge with the names in the title."""
    names = (exposures or {}).get("beta_excluded_names") or []
    if not names:
        return None
    return _badge(f"⚠ {len(names)} no β",
                  f"{len(names)} name(s) have no SPX beta and are excluded from "
                  f"spot-shocked pricing (never priced at a default): "
                  + ", ".join(names), warn=True)


# ---------------------------------------------------------------------------
# 6 — what's coming: the event timeline (Plotly, Python-stacked) + the grid
# ---------------------------------------------------------------------------

_CAL_TIP = (
    "Every dated event in the next 60 days: short-option expiries with the "
    "strike value if assigned and the risk-neutral P(assignment), ex-dividend "
    "dates on optioned names (amber ⚑ when an ITM short call runs into one — "
    "early-assignment economics), and expected earnings dates where the book "
    "is net short vol. Bubble size = % of |NAV| at stake; height is a "
    "collision row only. Standing obligations (all expiries) sit in their own "
    "block below.")

_CAL_KIND_LABEL = {"expiry": "Expiry", "ex_div": "Ex-div", "earnings": "Earnings"}
_CAL_SYMBOL = {"expiry": "circle", "ex_div": "diamond", "earnings": "square"}

# Stacking geometry (server-side): the chart renders at roughly half the page,
# so the plot area is ~620px over the 60-day window. Labels sit middle-right
# at 12px ≈ 6.8px/char.
_CHART_PLOT_W = 620.0
_CHART_CHAR_W = 6.8
_CHART_PAD = 6.0


def _at_stake(r: dict) -> Optional[float]:
    if r["kind"] == "expiry":
        return r.get("obligation")
    if r["kind"] == "earnings":
        return r.get("vega")
    return None


def _at_stake_str(r: dict) -> str:
    v = _at_stake(r)
    if r["kind"] == "earnings" and v is not None:
        return f"{_fmt_money(v)}/pt"
    if r["kind"] == "ex_div":
        dps = r.get("dps")
        return f"${dps:,.2f}/sh" if dps is not None else ""
    return _fmt_money(v) if v is not None else ""


def _event_points(cal: dict, nav) -> list[dict]:
    pts = []
    for r in cal["rows"]:
        at = _at_stake(r)
        pct = (abs(at) / abs(nav) * 100.0) if (at is not None and nav) else None
        detail, tip = _cal_detail(r)
        line1 = r["underlying"] + (" ⚑" if r.get("urgent") else "")
        line2 = _at_stake_str(r)
        p_str = None
        if r["kind"] == "expiry":
            p = r.get("p_assign")
            p_str = f"P(assign) {p * 100:.1f}%" if p is not None else "P(assign) unpriced"
        hover = "<br>".join(s for s in (
            f"{r['date'].isoformat()} · in {r['days']}d",
            f"{_CAL_KIND_LABEL.get(r['kind'])} — {r['underlying']}",
            detail,
            (f"at stake {line2} ({pct:.1f}% NAV)" if pct is not None
             else (f"at stake {line2}" if line2 else None)),
            p_str,
            r.get("flag_reason"),
        ) if s)
        pts.append({
            "date": r["date"], "days": r["days"], "kind": r["kind"],
            "size": 16.0 if pct is None else max(16.0, 16.0 + math.sqrt(pct) * 5.8),
            "color": _AMBER if r.get("urgent") else (
                _CHARCOAL if r["kind"] == "expiry" else _MUTED),
            "text": f"{line1}<br>{line2}" if line2 else line1,
            "label_w": _CHART_CHAR_W * max(len(line1), len(line2)),
            # A label near the window's right edge would run off the plot —
            # those flip to the marker's left (mirrored in the collision model).
            "flip": r["days"] > 48,
            "hover": hover,
        })
    return pts


def _stack_points(pts: list[dict], horizon_days: int) -> int:
    """Assign each event the lowest collision-free row (marker radii + label
    width + padding, on whichever side the label sits), computed server-side so
    Plotly only renders. Returns the row count."""
    # The rendered axis spans [t0-2d, t0+horizon+2d] — scale the collision
    # model to that span, not the bare horizon.
    ppd = _CHART_PLOT_W / max(horizon_days + 4, 1)
    rows: list[list[tuple[float, float]]] = []
    for p in sorted(pts, key=lambda q: (q["days"], -q["size"])):
        x = (p["days"] + 2) * ppd
        if p.get("flip"):
            start = x - p["size"] / 2.0 - p["label_w"] - _CHART_PAD
            end = x + p["size"] / 2.0
        else:
            start = x - p["size"] / 2.0
            end = x + p["size"] / 2.0 + p["label_w"] + _CHART_PAD
        for i, intervals in enumerate(rows):
            if all(end <= s or start >= e for s, e in intervals):
                intervals.append((start, end))
                p["row"] = i
                break
        else:
            rows.append([(start, end)])
            p["row"] = len(rows) - 1
    return max(len(rows), 1)


def _event_chart(cal: dict, nav):
    from dash import dcc
    import plotly.graph_objects as go          # lazy

    pts = _event_points(cal, nav)
    n_rows = _stack_points(pts, cal["horizon_days"])
    t0 = cal["as_of"]

    fig = go.Figure()
    for kind in ("expiry", "ex_div", "earnings"):
        # Non-urgent first: plotly styles the legend swatch off the first
        # point, so an urgent lead event must not turn the category key amber.
        kp = sorted((p for p in pts if p["kind"] == kind),
                    key=lambda p: p["color"] == _AMBER)
        if not kp:
            continue
        fig.add_trace(go.Scatter(
            x=[p["date"] for p in kp], y=[p["row"] for p in kp],
            mode="markers+text",
            name=_CAL_KIND_LABEL[kind].lower(),
            marker=dict(symbol=_CAL_SYMBOL[kind],
                        size=[p["size"] for p in kp],
                        color=[p["color"] for p in kp],
                        line=dict(width=1, color="white")),
            text=[p["text"] for p in kp],
            textposition=["middle left" if p.get("flip") else "middle right"
                          for p in kp],
            textfont=dict(size=12, color=_CHARCOAL),
            customdata=[[p["hover"]] for p in kp],
            hovertemplate="%{customdata[0]}<extra></extra>"))

    ticks = [t0 + timedelta(days=d) for d in (0, 15, 30, 45, 60)]
    fig.update_layout(
        height=max(460, 40 + n_rows * 66 + 46),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=_CHARCOAL, size=11),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom",
                    font=dict(size=11)),
        margin=dict(l=8, r=8, t=46, b=36),
        xaxis=dict(
            range=[t0 - timedelta(days=2), t0 + timedelta(days=62)],
            tickmode="array", tickvals=ticks,
            ticktext=["today", "+15d", "+30d", "+45d", "+60d"],
            showgrid=False,
            minor=dict(dtick=7 * 24 * 3600 * 1000, showgrid=True,
                       gridcolor=_GRID),
        ),
        # y is a collision row only — no magnitude meaning; row 0 reads from
        # the top.
        yaxis=dict(visible=False, range=[n_rows - 0.4, -0.7]),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False,
                                         "responsive": True},
                     className="scenario-graph")


def build_calendar_columns() -> list[dict]:
    return [
        {"field": "date", "headerName": "Date", "width": 116,
         "filter": "agDateColumnFilter",
         "filterParams": {"comparator": {"function": "dagfuncs.ISODateComparator"},
                          "browserDatePicker": True},
         "sort": "asc", "sortIndex": 0},
        {"field": "event", "headerName": "Event", "width": 88,
         "filter": "agTextColumnFilter"},
        {"field": "underlying", "headerName": "Ticker", "width": 94,
         "filter": "agTextColumnFilter", "cellClass": "blotter-ticker-cell"},
        {"field": "detail", "headerName": "Detail", "flex": 2, "minWidth": 170,
         "filter": "agTextColumnFilter", "tooltipField": "detail_tip",
         "cellClass": "dd-cell-ellipsis"},
        {"field": "at_stake", "headerName": "At stake", "width": 108,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": MONEY_FULL_FMT,
         "headerTooltip": "Expiry rows: strike value if assigned (cash to fund "
                          "a short put, delivery at strike on a short call). "
                          "Earnings rows: the name's net dollar vega per 1 vol "
                          "pt — negative = short vol into the print."},
        {"field": "at_stake_pct", "headerName": "% NAV", "width": 88,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_ABS_FMT},
        {"field": "p_assign", "headerName": "P(assign)", "width": 108,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_ABS_FMT,
         "headerTooltip": "Risk-neutral probability the short leg finishes in "
                          "the money at expiry (N(d2) at the leg's own IV). "
                          "Dash when the leg has no usable IV."},
    ]


def _cal_detail(r: dict) -> tuple[str, str]:
    """(visible detail, hover detail) per event kind. Data qualifiers only."""
    kind = r["kind"]
    if kind == "expiry":
        right = "Call" if r.get("right") == "CALL" else "Put"
        detail = f"-{r['contracts']:,.0f} × ${r['strike']:g} {right} short"
        if r.get("itm"):
            detail += " · ITM"
        tip = r.get("flag_reason") or r.get("p_assign_reason") or ""
        return detail, tip
    if kind == "ex_div":
        dps = r.get("dps")
        detail = f"${dps:,.2f}/sh" if dps is not None else "ex-dividend"
        if r.get("urgent"):
            detail += " · ITM short call — early assignment risk"
        return detail, r.get("flag_reason") or ""
    return "expected report", ""


def build_calendar_rows(cal: dict, nav) -> list[dict]:
    rows = []
    for r in cal["rows"]:
        detail, tip = _cal_detail(r)
        iso = r["date"].isoformat()
        at = _at_stake(r)
        rows.append({
            "_row_id": f"{r['kind']}::{r.get('underlying_bbg') or r['underlying']}"
                       f"::{iso}::{r.get('position_id') or ''}",
            "_urgent": bool(r.get("urgent")),
            "date": iso,
            "event": _CAL_KIND_LABEL.get(r["kind"], r["kind"]),
            "underlying": r["underlying"],
            "detail": detail,
            "detail_tip": tip or detail,
            "at_stake": at,
            "at_stake_pct": (abs(at) / abs(nav)) if (at is not None and nav) else None,
            "p_assign": r.get("p_assign"),
        })
    return rows


def _calendar_grid(rows: list[dict]) -> dag.AgGrid:
    return dag.AgGrid(
        id="risk-calendar-grid",
        columnDefs=build_calendar_columns(),
        rowData=rows,
        dashGridOptions={
            "rowHeight": 28,
            "headerHeight": 32,
            "animateRows": False,
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
            "domLayout": "autoHeight",
            "rowClassRules": {
                "blotter-row-t1": "params.data && params.data._urgent",
            },
            "defaultColDef": {"sortable": True, "resizable": True,
                              "suppressMovable": False},
        },
        className="ag-theme-balham blotter-grid",
        getRowId={"function": "params.data._row_id"},
        style={"width": "100%"},
    )


def _events_badges(cal: dict) -> list:
    badges = []
    n_total = cal.get("n_assign_total") or 0
    n_priced = cal.get("n_assign_priced") or 0
    if n_total and n_priced < n_total:
        badges.append(_badge(
            f"P(assign) {n_priced}/{n_total}",
            f"P(assign) priced on {n_priced} of {n_total} short leg(s) in the "
            "window — the rest dash (no usable IV).", warn=True))
    for w in cal.get("warnings") or []:
        badges.append(_badge("⚠ coverage", w, warn=True))
    return badges


# ---------------------------------------------------------------------------
# 7 — standing obligations (all expiries; beside concentration)
# ---------------------------------------------------------------------------

_OBLIG_TIP = (
    "Strike value of every short option if assigned, across ALL expiries — "
    "short puts: cash to fund; short calls: shares delivered at strike. A "
    "long option is a right, not an obligation. Totals are extract-only, so "
    "they render with Bloomberg off.")


def _oblig_total_row(label: str, side, nav, sub=None) -> html.Tr:
    itm_d = side.itm_dollars
    has = bool(side.n_positions)
    pct = pct_of_nav(side.dollars, nav) if has else None
    return html.Tr(className="risk-oblig-row", children=[
        html.Td([html.Span(label)] + ([sub] if sub is not None else []),
                className="risk-oblig-side"),
        html.Td(f"{side.contracts:,.0f}" if has else "—", className="exposure-num"),
        html.Td(_fmt_money(side.dollars) if has else "—", className="exposure-num"),
        html.Td(pct if pct is not None else "—", className="exposure-num"),
        html.Td("—" if itm_d is None else _fmt_money(itm_d), className="exposure-num"),
    ])


def _oblig_block(ob, nav) -> html.Div:
    if not ob.puts.n_positions and not ob.calls.n_positions:
        return html.Div("No short options on the book — nothing to assign.",
                        className="dd-empty")
    covered_sub = None
    if ob.calls.n_positions:
        covered_sub = html.Div(
            f"covered {ob.covered_call_contracts:,.0f} · "
            f"uncovered {ob.uncovered_call_contracts:,.0f} (no covering stock)",
            className="risk-oblig-sub")
    head = html.Tr([html.Th(h) for h in
                    ("", "Contracts", "Obligation", "% NAV", "ITM oblig.")])
    return html.Table(className="risk-oblig-table", children=[
        html.Thead(head),
        html.Tbody([
            _oblig_total_row("Short puts — cash to fund", ob.puts, nav),
            _oblig_total_row("Short calls — deliver at strike", ob.calls, nav,
                             covered_sub),
        ])])


def _oblig_badges(ob) -> list:
    badges = []
    n_unknown = ob.puts.n_unknown_moneyness + ob.calls.n_unknown_moneyness
    if n_unknown:
        badges.append(_badge(
            f"⚠ {n_unknown} no spot",
            f"Spot missing on {n_unknown} short option(s) — the ITM subtotal "
            "excludes them.", warn=True))
    if ob.n_expired:
        badges.append(_badge(
            f"{ob.n_expired} expired",
            f"Expired ({ob.n_expired}) — dead contract(s) still on the book "
            "(stale extract); excluded from every figure here."))
    return badges


# ---------------------------------------------------------------------------
# 7 — concentration (every basis as % of |NAV|)
# ---------------------------------------------------------------------------

_CONC_TOP_N = 10

_CONC_TIP = (
    "Per-name exposure on every basis, as % of |NAV|. Net nets options against "
    "stock; gross sums per-position absolute deltas — the lens a hedge cannot "
    "empty (rows sort by it). Obligations are the short-strike dollars. Market "
    "value by name lives on Holdings.")


def build_concentration_columns() -> list[dict]:
    return [
        {"field": "name", "headerName": "Name", "flex": 2, "minWidth": 104,
         "filter": "agTextColumnFilter", "cellClass": "blotter-ticker-cell"},
        {"field": "net_delta_pct", "headerName": "Net Δ", "width": 90,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_SIGNED_1DP_FMT, "cellStyle": SIGNED_COLOR_STYLE,
         "headerTooltip": "Net dollar delta as % of |NAV| — options netted "
                          "against stock."},
        {"field": "gross_delta_pct", "headerName": "Gross", "width": 90,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_ABS_FMT,
         "headerTooltip": "Gross |Δ$| as % of |NAV| — per-position absolute "
                          "deltas summed; the sort basis."},
        {"field": "gamma_pct", "headerName": "Γ (1%)", "width": 88,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_SIGNED_1DP_FMT, "cellStyle": SIGNED_COLOR_STYLE,
         "headerTooltip": "Δ$ change per 1% spot move as % of |NAV| (Bloomberg "
                          "per-1% basis — not the engine per-$1 Γ)."},
        {"field": "vega_pct", "headerName": "Vega", "width": 84,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_SIGNED_1DP_FMT, "cellStyle": SIGNED_COLOR_STYLE,
         "headerTooltip": "Dollar vega per 1 vol pt as % of |NAV|."},
        {"field": "oblig_pct", "headerName": "Oblig.", "width": 86,
         "type": "rightAligned", "filter": "agNumberColumnFilter",
         "valueFormatter": PCT_ABS_FMT,
         "headerTooltip": "Short-strike dollars if assigned (puts cash + calls "
                          "delivery) as % of |NAV|."},
    ]


def _conc_pcts(r: dict, nav) -> dict:
    def _pct(v):
        return (v / nav) if (v is not None and nav) else None

    oblig = (r.get("put_obligation") or 0.0) + (r.get("call_obligation") or 0.0)
    return {
        "net_delta_pct": _pct(r.get("net_dollar_delta")),
        "gross_delta_pct": _pct(r.get("gross_dollar_delta")),
        "gamma_pct": _pct(r.get("dollar_gamma")),
        "vega_pct": _pct(r.get("dollar_vega")),
        "oblig_pct": _pct(oblig) if oblig else None,
    }


def build_concentration_rows(lenses: dict) -> tuple[list[dict], list[dict]]:
    """(rows, pinned Account row). Values are fractions of |NAV|; None dashes."""
    nav = lenses["nav"]
    rows = []
    for r in lenses["rows"][:_CONC_TOP_N]:
        rows.append({"_row_id": r["symbol"], "name": r["symbol"],
                     **_conc_pcts(r, nav)})
    rest = lenses["rows"][_CONC_TOP_N:]
    if rest:
        def _tot(key):
            vals = [r[key] for r in rest if r.get(key) is not None]
            return sum(vals) if vals else None

        agg = {k: _tot(k) for k in ("net_dollar_delta", "gross_dollar_delta",
                                    "dollar_gamma", "dollar_vega",
                                    "put_obligation", "call_obligation")}
        rows.append({"_row_id": "__others__", "name": f"Other ({len(rest)})",
                     **_conc_pcts(agg, nav)})
    pinned = [{"_row_id": "__account__", "name": "Account",
               **_conc_pcts(dict(lenses["account"]), nav)}]
    return rows, pinned


def _concentration_grid(lenses: dict):
    rows, pinned = build_concentration_rows(lenses)
    if not rows:
        return html.Div("No name exposure to show.", className="dd-empty")
    return dag.AgGrid(
        id="risk-conc-grid",
        columnDefs=build_concentration_columns(),
        rowData=rows,
        dashGridOptions={
            "rowHeight": 28,
            "headerHeight": 32,
            "animateRows": False,
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
            "domLayout": "autoHeight",
            "pinnedBottomRowData": pinned,
            "defaultColDef": {"sortable": True, "resizable": True,
                              "suppressMovable": False},
        },
        className="ag-theme-balham blotter-grid",
        getRowId={"function": "params.data._row_id"},
        style={"width": "100%"},
    )


def _conc_badges(lenses: dict) -> list:
    missing = lenses["missing"]
    if not missing["n_rows"]:
        return []
    names = missing["names"]
    return [_badge(
        f"⚠ {missing['n_rows']} unpriced",
        f"Delta missing on {missing['n_rows']} position(s) across "
        f"{len(names)} name(s) — their Δ columns dash and the totals "
        f"understate: {', '.join(names)}.", warn=True)]


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------

def render_risk_section(account_state, state) -> html.Div:
    head = html.Div(className="dd-section-head", children=[
        html.H2("Risk", className="dd-section-title"),
        html.Span("posture · scenario · what's coming · concentration",
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

    # Each pre-compute degrades alone: a calendar or obligations failure must
    # cost its own block, not the section (the blocks below are _safe-wrapped
    # and render an honest error line on a None/raise).
    ob = _quiet(lambda: assignment_obligations(account_state))
    lenses = _quiet(lambda: concentration_lenses(account_state, obligations=ob))
    cal = _quiet(lambda: upcoming_events(state, account_state, ob, lenses))

    # 1 — posture
    if e is not None:
        posture = [_band("Posture", badges=_safe(lambda: _posture_badges(e),
                                                 "Posture badges")),
                   _safe(lambda: _posture_strip(e, nav), "Posture")]
    else:
        posture = [_band("Posture"),
                   html.Div("Exposure unavailable for this account.",
                            className="dd-empty")]

    # 2-5 — dials, ribbon, convexity strip, heatmap | impact
    if impact is not None:
        exposures = impact.get("exposures")
        scenario_blocks = [
            _band("Scenario"),
            _controls(account_state, impact["rows"]),
            html.Div(id="scn-total",
                     children=_total_line(impact, shocked=False)),
            _band("Convexity — how the book reshapes", tip=_RESHAPE_CAPTION,
                  badges=[b for b in (_coverage_badge(exposures),
                                      _beta_excluded_badge(exposures)) if b]),
            html.Div(id="risk-reshape",
                     children=_reshape_table(exposures, nav, shocked=False)),
            html.Div(className="scn-body", children=[
                html.Div(className="scn-heatmap-wrap", children=[
                    _lazy_graph(grid)]),
                html.Div(className="scn-impact-wrap", children=[
                    _band("Per-position impact", tip=_CAPTION),
                    html.Div(id="scn-impact",
                             children=_impact_table(impact["rows"], "account",
                                                    shocked=False)),
                ]),
            ]),
        ]
    else:
        scenario_blocks = [
            _band("Scenario"),
            html.Div("Scenario views unavailable (Bloomberg off or no "
                     "priceable options).", className="dd-empty")]

    # 6 — what's coming: chart | event grid
    cal_rows = _quiet(lambda: build_calendar_rows(cal, nav)) or []
    left6 = [_band("What's coming — next 60 days", tip=_CAL_TIP,
                   badges=_safe(lambda: _events_badges(cal), "Event badges"))]
    right6 = [_band("Event detail")]
    if cal_rows:
        left6.append(_safe(lambda: _event_chart(cal, nav), "Event timeline"))
        right6.append(_safe(lambda: _calendar_grid(cal_rows), "Event detail"))
    else:
        left6.append(html.Div("No dated events in the next 60 days.",
                              className="dd-empty"))
        right6.append(html.Div("—", className="dd-empty"))

    # 7 — concentration | standing obligations
    left7 = [_band("Concentration", tip=_CONC_TIP,
                   badges=_safe(lambda: _conc_badges(lenses), "Concentration badges")),
             _safe(lambda: _concentration_grid(lenses), "Concentration")]
    right7 = [_band("Standing obligations", tip=_OBLIG_TIP,
                    badges=_safe(lambda: _oblig_badges(ob), "Obligation badges")),
              _safe(lambda: _oblig_block(ob, nav), "Standing obligations")]

    children = [head, *posture, *scenario_blocks,
                _cols(left6, right6),
                _cols(left7, right7)]
    return html.Div(className="dd-section risk-cockpit", children=children)


def _lazy_graph(grid):
    from dash import dcc
    return dcc.Graph(id="scn-heatmap", figure=_heatmap_fig(grid, 0.0, 0.0, shocked=False),
                     config={"displayModeBar": False, "responsive": True},
                     className="scenario-graph")
