"""Current-book exposure panels for the Risk section (the cockpit composes them).

Renders the precomputed account exposure (``acc.exposure`` from pm.risk.exposure):
the net market exposure (beta-adjusted dollar delta) headline, net dollar greeks,
market value vs economic exposure, both SPX betas, vega by tenor, and the
position -> structure -> account rollup. A pure read of ``acc.exposure`` — no
compute, no Bloomberg, no recompute. ``risk_cockpit.render_risk_section`` owns the
section shell — this module is the panel library; the one-glance "Net Greeks" KPI
in the header stays as the separate glance (and the click-through into the section).
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.ui.deepdive.aggregations import _fmt_money

_BETA_NOTE = (
    "Adjusted (Blume) is the default — name betas converge toward 1.0 in a selloff "
    "(crash correlation), so it is the steadier downside proxy; raw is the current "
    "empirical sensitivity. Both vs SPX, 2y weekly, per name."
)


def _stat(label: str, value: str, sub: Optional[str] = None, cls: str = "",
          title: Optional[str] = None) -> html.Div:
    children = [html.Div(label, className="dd-stat-label"),
                html.Div(value, className="dd-stat-value")]
    if sub:
        children.append(html.Div(sub, className="dd-stat-sub"))
    div = html.Div(className=f"dd-stat {cls}".strip(), children=children)
    if title:
        div.title = title
    return div


def _sign_cls(v: Optional[float]) -> str:
    """dd-stat sign colouring (green/red for +/-), neutral at zero/None."""
    if v is None or v == 0:
        return ""
    return "dd-stat-pos" if v > 0 else "dd-stat-neg"


def _num_cls(v: Optional[float]) -> str:
    """Rollup-cell sign colouring on the shared --pos/--neg tokens."""
    base = "exposure-num"
    if v is None or v == 0:
        return base
    return f"{base} {'exposure-pos' if v > 0 else 'exposure-neg'}"


def _struct_label(node) -> str:
    st = getattr(node, "structure_type", None)
    label = st.replace("_", " ").capitalize() if st else getattr(node, "label", "—")
    if getattr(node, "contention_group", None):
        label += "  (alt)"
    return label


# ---- panels ---------------------------------------------------------------

def _headline_panel(e) -> html.Div:
    t = e.total
    nme = e.net_market_exposure
    return html.Div(className="dd-panel", children=[
        html.H3("Net market exposure", className="dd-panel-title"),
        html.Div(className="dd-stat-row", children=[
            _stat("Net market exposure", _fmt_money(nme),
                  "β SPX 2y wkly · adjusted", cls=_sign_cls(nme),
                  title="Σ position dollar-delta × the name's SPX beta "
                        "(net beta-adjusted market exposure)"),
            _stat("Net $ Delta", _fmt_money(t.dollar_delta), "economic (delta-$)",
                  cls=_sign_cls(t.dollar_delta)),
            _stat("Net $ Gamma", _fmt_money(t.dollar_gamma), "Δ$ per 1% spot move",
                  cls=_sign_cls(t.dollar_gamma),
                  title="Bloomberg GAMMA is dDelta per 1% underlying move — a "
                        "different basis from the Scenario section's engine "
                        "per-$1 Γ$; do not compare."),
            _stat("Net $ Vega", _fmt_money(t.dollar_vega), "per 1 vol pt",
                  cls=_sign_cls(t.dollar_vega)),
            _stat("Net $ Theta", _fmt_money(t.dollar_theta), "per calendar day",
                  cls=_sign_cls(t.dollar_theta)),
        ]),
        html.Div(_provenance(e), className="dd-panel-note"),
    ])


def _mv_vs_econ_panel(e) -> html.Div:
    t = e.total
    return html.Div(className="dd-panel", children=[
        html.H3("Market value vs economic exposure", className="dd-panel-title"),
        html.Div(className="dd-stat-row", children=[
            _stat("Market value", _fmt_money(t.market_value), "marked book value",
                  cls=_sign_cls(t.market_value)),
            _stat("Economic exposure", _fmt_money(e.economic_exposure),
                  "delta-equivalent (delta-$)", cls=_sign_cls(e.economic_exposure)),
        ]),
        html.Div("Market value is what the book is marked at; economic exposure is its "
                 "delta-equivalent exposure to the underlyings — they diverge where an "
                 "option's premium understates its directional exposure.",
                 className="dd-panel-note"),
    ])


def _beta_panel(e) -> html.Div:
    t = e.total
    return html.Div(className="dd-panel", children=[
        html.Div(className="dd-panel-headrow", children=[
            html.H3("Beta", className="dd-panel-title"),
            html.Span("β SPX 2y wkly", className="dd-beta-chip"),
        ]),
        html.Div(className="dd-stat-row", children=[
            _stat("Adjusted (β-$)", _fmt_money(t.dollar_beta_adjusted), "default",
                  cls=_sign_cls(t.dollar_beta_adjusted)),
            _stat("Raw (β-$)", _fmt_money(t.dollar_beta_raw),
                  cls=_sign_cls(t.dollar_beta_raw)),
        ]),
        html.Div(_BETA_NOTE, className="dd-panel-note"),
    ])


def _bucket_cell(b) -> str:
    """A bucket's display value. Dash when it holds no options — or when every
    option in it is missing vega (that $0 would be pure missing data, not zero
    exposure)."""
    n_missing = getattr(b, "n_missing_vega", 0) or 0
    if not b.n_options or n_missing >= b.n_options:
        return "—"
    return _fmt_money(b.dollar_vega)


def _vega_tenor_row(e) -> html.Div:
    header = html.Div(className="dd-ladder-row dd-ladder-head", children=[
        html.Span("Tenor"),
        *[html.Span(b.label) for b in e.vega_by_tenor],
    ])
    values = html.Div(className="dd-ladder-row", children=[
        html.Span("Net $ Vega", className="dd-ladder-bucket"),
        *[html.Span(_bucket_cell(b),
                    className="dd-ladder-count") for b in e.vega_by_tenor],
    ])
    children = [
        html.H3("Vega by tenor", className="dd-panel-title"),
        html.Div("Vega's term structure — dollar vega by days to expiry.",
                 className="dd-panel-subtitle"),
        html.Div(className="dd-ladder dd-vega-ladder", children=[header, values]),
    ]
    n_missing = sum(getattr(b, "n_missing_vega", 0) or 0 for b in e.vega_by_tenor)
    n_options = sum(b.n_options for b in e.vega_by_tenor)
    if n_missing:
        children.append(html.Div(
            f"Vega missing on {n_missing} of {n_options} option(s) — buckets "
            "understate by those positions.", className="dd-panel-note"))
    n_expired = getattr(e, "n_expired_options", 0) or 0
    if n_expired:
        children.append(html.Div(
            f"Expired ({n_expired}) — dead contract(s) still on the book "
            "(stale extract); excluded from every tenor bucket.",
            className="dd-panel-note"))
    return html.Div(className="dd-panel", children=children)


# ---- rollup table ---------------------------------------------------------

_ROLLUP_COLS = [
    ("Structure", "left"),
    ("$ Delta", "right"), ("β-$ (SPX)", "right"), ("$ Gamma (1%)", "right"),
    ("$ Vega", "right"), ("$ Theta", "right"), ("Net MV", "right"),
]


def _rollup_row(node, row_cls: str = "") -> html.Tr:
    degraded = getattr(node, "degraded", False)
    label_children = [_struct_label(node)]
    if degraded:
        label_children.append(html.Span(" ⚠", className="exposure-degraded-mark",
                                         title="some legs could not be cleanly allocated"))
    cells = [html.Td(label_children, className="exposure-label")]
    for v in (node.dollar_delta, node.dollar_beta_adjusted, node.dollar_gamma,
              node.dollar_vega, node.dollar_theta, node.market_value):
        cells.append(html.Td(_fmt_money(v), className=_num_cls(v)))
    cls = f"am-row {row_cls}".strip()
    if getattr(node, "contention_group", None):
        cls += " exposure-row-contention"
    if degraded:
        cls += " exposure-row-degraded"
    return html.Tr(className=cls, children=cells)


def _rollup_summary(e) -> html.Summary:
    """The always-visible accordion header: the bold Account total + the expander.
    Collapsed, the reader sees the account's net exposure; expanding reveals the
    per-structure breakdown. Native <details>/<summary> — no callback, no JS."""
    t = e.total
    n = len(e.structures)

    def _stat(label, v):
        return html.Span(className="exposure-rollup-stat", children=[
            html.Span(label, className="exposure-rollup-stat-lbl"),
            html.Span(_fmt_money(v), className=_num_cls(v)),
        ])

    return html.Summary(className="exposure-rollup-summary", children=[
        html.Span("Account", className="exposure-rollup-acct"),
        _stat("Net $Δ", t.dollar_delta),
        _stat("β-$", t.dollar_beta_adjusted),
        _stat("Net MV", t.market_value),
        html.Span(className="exposure-rollup-toggle", children=[
            f"Structure → Account ({n}) ",
            html.Span("▾", className="exposure-rollup-caret"),
        ]),
    ])


def _rollup_table(e) -> html.Div:
    head = html.Tr(children=[
        html.Th(name, className="am-th",
                style={"textAlign": align}) for name, align in _ROLLUP_COLS
    ])
    body: list = [_rollup_row(s) for s in e.structures]
    body.append(_rollup_row(e.structured, "exposure-row-subtotal"))
    body.append(_rollup_row(e.unstructured, "exposure-row-subtotal"))
    body.append(_rollup_row(e.total, "exposure-row-total"))
    note = ("Structured + Unstructured = Account. Contention alternatives (alt) are "
            "mutually-exclusive readings of the same legs — shown for context, not "
            "added into the totals.") if any(
        getattr(s, "contention_group", None) for s in e.structures) else \
        "Structured + Unstructured = Account."
    return html.Div(className="dd-panel dd-exposure-rollup", children=[
        html.H3("Exposure rollup — structure → account", className="dd-panel-title"),
        html.Details(open=False, className="exposure-rollup-details", children=[
            _rollup_summary(e),
            html.Table(className="am-table exposure-table", children=[
                html.Thead(children=[head]),
                html.Tbody(children=body),
            ]),
            html.Div(note, className="dd-panel-note"),
        ]),
    ])


def _provenance(e) -> str:
    src = (e.trace or {}).get("inputs", {}).get("greek_source", "snapshot greeks")
    missing = (e.trace or {}).get("inputs", {}).get("names_missing_beta", []) or []
    base = f"From {src}; beta vs SPX (2y weekly)."
    if missing:
        shown = ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
        base += f" {len(missing)} name(s) had no SPX beta and are excluded from " \
                f"dollar-beta: {shown}."
    missing_greeks = getattr(e, "missing_greeks", []) or []
    if missing_greeks:
        shown = ", ".join(missing_greeks[:3]) + ("…" if len(missing_greeks) > 3 else "")
        base += f" Greeks missing on {len(missing_greeks)} name(s) — " \
                f"totals understate: {shown}."
    return base


