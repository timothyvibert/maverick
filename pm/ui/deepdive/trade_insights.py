"""Section — Client profile (the trade-history behavioural read).

Renders the precomputed per-account ``acc.client_profile`` (from
pm.insight.client_profile) as a dense, distribution-forward glance at *how this
client trades*: a full-width fingerprint strip over a responsive grid of
dimension tiles (strategy posture, tenor + direction, sector lean, name lean,
sizing, cadence). A pure read of the stored profile — no compute, no Bloomberg,
no recompute. The tiles draw fields the engine already produced (the weight maps,
the tenor distribution, the skews); they never re-bin or recompute from the raw
trades in the view.

This surface is behavioural, not P&L, so it stays in neutral chrome. Sign-colour
is earned in exactly one place — the net-delta lean — where the desk reads a real
direction. Magnitudes and the coverage / confidence signals stay quiet, and a
suppressed or low-confidence dimension says so rather than showing a precise-
looking false number.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.ui.deepdive.bars import bar_row, bin_histogram, diverging_gauge, magnitude_gauge
from pm.ui.deepdive.formatters import money_compact, pct

# Display labels (render-only — the engine keys are the source of truth).
_POSTURE_LABELS = {
    "long_call": "Long calls",
    "short_call": "Short calls",
    "long_put": "Long puts",
    "short_put": "Short puts",
    "long_stock": "Long stock",
    "unclassified_open": "Unclassified",
}
_TENOR_LABELS = {"short": "Short-dated", "swing": "Swing", "leaps": "LEAPS"}
_BAND_HELP = ("Coverage band from trade count and round-trip evidence — thin "
              "history reads low, so dimensions degrade rather than mislead.")

# Three across on the full-width deep-dive, collapsing to two then one on narrower
# viewports. Inline structural geometry only — palette/chrome stay on the tokens.
_GRID_STYLE = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(400px, 1fr))",
               "gap": "12px", "marginTop": "12px"}


# ---------------------------------------------------------------------------
# Small render helpers (mirror the exposure/analytics section primitives)
# ---------------------------------------------------------------------------

def _stat(label: str, value: str, sub: Optional[str] = None, cls: str = "") -> html.Div:
    children = [html.Div(label, className="dd-stat-label"),
                html.Div(value, className="dd-stat-value")]
    if sub:
        children.append(html.Div(sub, className="dd-stat-sub"))
    return html.Div(className=f"dd-stat {cls}".strip(), children=children)


def _sign_cls(v: Optional[float]) -> str:
    """Sign colouring — used ONLY for a genuine directional ± (net delta lean)."""
    if v is None or v == 0:
        return ""
    return "dd-stat-pos" if v > 0 else "dd-stat-neg"


def _conf_suffix(confidence: Optional[str]) -> str:
    """A quiet caveat appended to a tile subtitle when the read is thin."""
    return " · low confidence" if confidence == "low" else ""


def _window_text(days: Optional[int]) -> str:
    return "—" if days is None else f"{days}d"


def _skew_text(v: Optional[float], pos_word: str, neg_word: str) -> str:
    """A signed lean as a magnitude + side, e.g. +0.64 -> '64% long'."""
    if v is None:
        return "—"
    return f"{pct(abs(v), 0)} {pos_word if v >= 0 else neg_word}"


def _tile(title: str, subtitle: Optional[str], *body) -> html.Div:
    children = [html.H3(title, className="dd-panel-title")]
    if subtitle:
        children.append(html.Div(subtitle, className="dd-panel-subtitle"))
    children.extend([b for b in body if b is not None])
    return html.Div(className="dd-panel", children=children)


# ---------------------------------------------------------------------------
# Pure card-builders (ClientProfile -> display rows; unit-testable, no browser)
# ---------------------------------------------------------------------------

def build_strategy_rows(profile) -> list[dict]:
    """Strategy-posture bars, largest weight first."""
    sb = profile.strategy_bias
    ranked = sorted(sb.weights.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"key": k, "label": _POSTURE_LABELS.get(k, k), "weight": w} for k, w in ranked]


def build_sector_rows(profile) -> dict:
    """The ranked GICS-sector lean (the names that resolved to a sector)."""
    sl = profile.sector_lean
    return {"rows": [{"label": s, "weight": w} for s, w in sl.top],
            "classified": sl.classified_fraction}


def build_name_rows(profile) -> list[dict]:
    """The ranked underlying lean — the names they actually trade."""
    return [{"label": n, "weight": w} for n, w in profile.sector_lean.by_name]


def build_tenor_bins(profile) -> list[tuple]:
    """The at-open tenor distribution as ordered (label, fraction) bins."""
    d = profile.tenor_pref.distribution or {}
    return [("Short", d.get("short", 0.0)), ("Swing", d.get("swing", 0.0)),
            ("LEAPS", d.get("leaps", 0.0))]


def build_coverage_stats(profile) -> dict:
    """The history-depth numbers, with the two fractions kept distinct: the
    round-trip rate and the held-positions-with-an-opening-trade rate (which
    carries the opens-predate-window caveat)."""
    c = profile.coverage
    return {
        "band": c.band,
        "n_trades": c.n_trades,
        "window_days": c.window_days,
        "paired_fraction": c.paired_fraction,
        "positions_with_derivable_open_fraction": c.positions_with_derivable_open_fraction,
    }


# ---------------------------------------------------------------------------
# Fingerprint strip + tiles
# ---------------------------------------------------------------------------

def _fingerprint_strip(profile) -> html.Div:
    c = profile.coverage
    pos_open = c.positions_with_derivable_open_fraction
    depth = (f"{_window_text(c.window_days)} window · {c.n_trades} trades · "
             f"{pct(c.paired_fraction, 0)} round-tripped · "
             f"{pct(pos_open, 0) if pos_open is not None else '—'} with entry in window")
    return html.Div(className="dd-panel", children=[
        html.Div(className="dd-panel-headrow", children=[
            html.Div(children=[
                html.Div("How this client trades", className="dd-stat-label"),
                html.Div(profile.headline or "—", className="dd-stat-value"),
            ]),
            html.Span(f"coverage · {c.band}", className="dd-beta-chip", title=_BAND_HELP),
        ]),
        html.Div(depth, className="dd-panel-subtitle"),
    ])


def _ranked_bars(rows: list[dict]):
    if not rows:
        return html.Div("Insufficient history.", className="dd-empty")
    max_w = max(r["weight"] for r in rows)
    return html.Div(className="dd-bars", children=[
        bar_row(r["label"], r["weight"], max_w, sign_color=False) for r in rows
    ])


def _strategy_tile(profile) -> html.Div:
    return _tile(
        "Strategy posture",
        "What the account opens, by posture — from trade flow"
        + _conf_suffix(profile.strategy_bias.confidence) + ".",
        _ranked_bars(build_strategy_rows(profile)),
    )


def _tenor_direction_tile(profile) -> html.Div:
    t = profile.tenor_pref
    d = profile.direction_bias
    if t.bucket:
        tenor_body = [
            bin_histogram(build_tenor_bins(profile)),
            html.Div(f"median {t.median_dte_at_open:.0f}d · {_TENOR_LABELS.get(t.bucket, t.bucket)}",
                     className="dd-panel-note"),
        ]
    else:
        tenor_body = [html.Div("Tenor — insufficient history.", className="dd-empty")]
    # Net delta: the lone sign-coloured element (a genuine bullish/bearish lean).
    net_delta = html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "12px"},
        children=[
            html.Span("Net Δ", className="dd-stat-label", style={"minWidth": "44px"}),
            diverging_gauge(d.long_short_skew),
            html.Span(_skew_text(d.long_short_skew, "long", "short"),
                      className=f"dd-bar-val {_sign_cls(d.long_short_skew)}".strip(),
                      style={"minWidth": "64px"}),
        ],
    )
    # Call/put is a composition, not a gain/loss — kept neutral.
    call_put = html.Div(
        className="dd-panel-note",
        children=f"Call / put: {_skew_text(d.call_put_skew, 'calls', 'puts')}",
    )
    return _tile("Tenor & direction",
                 "At-open days-to-expiry and directional lean" + _conf_suffix(d.confidence) + ".",
                 *tenor_body, net_delta, call_put)


def _sector_tile(profile) -> html.Div:
    info = build_sector_rows(profile)
    if info["rows"]:
        sub = f"By GICS sector · {pct(info['classified'], 0)} of flow classified"
        body = _ranked_bars(info["rows"])
    else:
        sub = "By GICS sector"
        body = html.Div("Sector unavailable for the traded names.", className="dd-empty")
    return _tile("Sector lean", sub, body)


def _name_tile(profile) -> html.Div:
    return _tile("Name lean", "Most-traded underlyings, by trade count",
                 _ranked_bars(build_name_rows(profile)))


def _sizing_tile(profile) -> html.Div:
    s = profile.sizing
    body = [_stat("Median trade",
                  money_compact(s.median_principal) if s.median_principal is not None else "—")]
    if s.concentration_hhi is not None:
        body.append(html.Div("Name concentration", className="dd-stat-label",
                             style={"marginTop": "12px"}))
        body.append(magnitude_gauge(s.concentration_hhi))
        body.append(html.Div(f"HHI {s.concentration_hhi:.2f}", className="dd-panel-note"))
    return _tile("Sizing", "Per-trade size and how concentrated the names are.", *body)


def _cadence_tile(profile) -> html.Div:
    cad = profile.cadence
    if cad.trades_per_month is None:
        return _tile("Cadence", "Trading frequency.",
                     _stat("Cadence", "—", sub="insufficient history"))
    # clustering is shown only once derived — never a bare "n/a".
    extra = None
    if cad.clustering and cad.clustering != "n/a":
        extra = html.Div(f"clusters around {cad.clustering}", className="dd-panel-note")
    return _tile("Cadence", "Trading frequency.",
                 _stat("Trades / month", f"{cad.trades_per_month:.1f}"), extra)


_ROLL_TENDENCY_LABELS = {"rolls": "Rolls positions", "closes_early": "Closes early", "mixed": "Mixed"}


def _fragile_holds(hp) -> html.Div:
    if hp is None or hp.median_days_held is None:
        value = html.Div("Insufficient history.", className="dd-empty")
    else:
        value = html.Div(children=[
            html.Div(f"{hp.median_days_held:.0f}d", className="dd-stat-value"),
            html.Div(f"median held · {hp.n_positions} positions" + _conf_suffix(hp.confidence),
                     className="dd-stat-sub"),
        ])
    return html.Div(children=[
        html.Div("Holding period", className="dd-stat-label"),
        value,
        # The load-bearing caveat — both limitations, stated plainly, not a footnote.
        html.Div("Current-book proxy: days since each held position's contract was first opened in "
                 "the book (matched by contract across accounts — a deliberate ingest behaviour for "
                 "book transfers; only positions with such an open are counted). Survivorship-biased "
                 "toward longer holds — short positions already closed are absent. Not a realised "
                 "holding period.", className="dd-panel-note"),
    ])


def _fragile_rolls(rb) -> html.Div:
    if rb is None:
        value = html.Div("Insufficient history.", className="dd-empty")
    elif rb.tendency == "unknown":
        if rb.n_events > 0:
            plural = "s" if rb.n_events != 1 else ""
            value = html.Div(f"{rb.n_events} roll-like event{plural} observed · too few closes to "
                             "characterise the tendency", className="dd-empty")
        else:
            value = html.Div("Insufficient history.", className="dd-empty")
    else:
        value = html.Div(children=[
            html.Div(_ROLL_TENDENCY_LABELS.get(rb.tendency, rb.tendency), className="dd-stat-value"),
            html.Div(f"{rb.n_events} of {rb.n_closes} closes roll-like" + _conf_suffix(rb.confidence),
                     className="dd-stat-sub"),
        ])
    return html.Div(style={"marginTop": "14px"}, children=[
        html.Div("Roll behaviour", className="dd-stat-label"),
        value,
        html.Div("Heuristic over roll-like clustering — a close plus a same-name, same-right reopen "
                 "on a different contract within a day. Not a verified FIFO/LIFO pairing.",
                 className="dd-panel-note"),
    ])


def _fragile_income() -> html.Div:
    return html.Div(style={"marginTop": "14px"}, children=[
        html.Div("Realised income", className="dd-stat-label"),
        html.Div("Deferred — needs open↔close trade pairing and a per-trade price derivation "
                 "(no price column today).", className="dd-panel-note"),
    ])


def _fragile_panel(profile) -> html.Div:
    """The fragile lifecycle tier — holds proxy, roll heuristic, deferred income —
    behind a collapsed disclosure, each dimension carrying its own caveat."""
    return html.Div(className="dd-panel", style={"marginTop": "12px"}, children=[
        html.Details(open=False, children=[
            html.Summary("Holding period & rolls", className="dd-panel-title"),
            html.Div(style={"marginTop": "10px"}, children=[
                _fragile_holds(getattr(profile, "holding_period", None)),
                _fragile_rolls(getattr(profile, "roll_behavior", None)),
                _fragile_income(),
            ]),
        ]),
    ])


def render_trade_insights_section(account_state) -> html.Div:
    """The #5 deep-dive section — a pure read of ``acc.client_profile``."""
    profile = getattr(account_state, "client_profile", None)
    head = html.Div(className="dd-section-head", children=[
        html.H2("Client profile", className="dd-section-title"),
        html.Span("trade-history behaviour · pre-call read", className="dd-section-meta"),
    ])
    if profile is None:
        return html.Div(className="dd-section", children=[
            head,
            html.Div("Trade-history profile unavailable for this account.", className="dd-empty"),
        ])
    grid = html.Div(style=_GRID_STYLE, children=[
        _strategy_tile(profile),
        _tenor_direction_tile(profile),
        _sector_tile(profile),
        _name_tile(profile),
        _sizing_tile(profile),
        _cadence_tile(profile),
    ])
    return html.Div(className="dd-section", children=[head, _fingerprint_strip(profile), grid,
                                                      _fragile_panel(profile)])
