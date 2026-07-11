"""Shared CSS bar primitive for the deep-dive sections — one source for the
``dd-bar`` look used by the Analytics sector / concentration panels and the
client-profile cards. No charting dependency (constraint).

The exposure default sign-colours the value (green/red for +/-) and tints a
negative fill red. The behavioural client profile renders magnitudes that carry
no gain/loss meaning, so it passes ``sign_color=False`` for plain neutral bars —
colour stays reserved for the few places a genuine direction is read.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.ui.deepdive.formatters import pct


def bar_row(label: str, value: Optional[float], max_w: float,
            sign_color: bool = True, value_text: Optional[str] = None) -> html.Div:
    """One horizontal bar; width is |value| / max_w (magnitude). With
    ``sign_color`` a negative value tints the fill and value red and a positive
    greens the value; without it the bar stays neutral charcoal. ``value_text``
    overrides the default percent label."""
    neg = sign_color and value is not None and value < 0
    pos = sign_color and value is not None and value > 0
    width = (abs(value or 0) / max_w * 100) if max_w else 0
    fill_cls = "dd-bar-fill dd-bar-fill-neg" if neg else "dd-bar-fill"
    val_cls = "dd-bar-val"
    if neg:
        val_cls += " exposure-neg"
    elif pos:
        val_cls += " exposure-pos"
    return html.Div(className="dd-bar-row", children=[
        html.Span(label, className="dd-bar-label"),
        html.Div(className="dd-bar-track", children=[
            html.Div(className=fill_cls, style={"width": f"{width:.1f}%"}),
        ]),
        html.Span(value_text if value_text is not None else pct(value), className=val_cls),
    ])


# ---------------------------------------------------------------------------
# Small static data-viz primitives (CSS divs, token-styled — no charting dep).
# Geometry is inline; every colour is a design token, so they stay on-palette.
# ---------------------------------------------------------------------------

def bin_histogram(bins: list, track_h: int = 40) -> html.Div:
    """A small vertical histogram of a few labelled fractions (0..1), neutral
    charcoal. ``bins`` is a list of (label, fraction)."""
    cols = []
    for label, frac in bins:
        f = max(0.0, min(1.0, frac or 0.0))
        h = max(3, round(f * 100))
        cols.append(html.Div(
            style={"display": "flex", "flexDirection": "column",
                   "alignItems": "center", "flex": "1", "gap": "3px"},
            children=[
                html.Div(pct(frac, 0), style={"fontSize": "10px", "color": "var(--pm-grey-700)",
                                              "fontFamily": "Consolas, monospace"}),
                html.Div(
                    style={"width": "100%", "height": f"{track_h}px", "background": "var(--pm-grey-100)",
                           "borderRadius": "2px", "display": "flex", "alignItems": "flex-end"},
                    children=[html.Div(style={"width": "100%", "height": f"{h}%",
                                              "background": "var(--pm-charcoal)", "borderRadius": "2px"})],
                ),
                html.Div(label, style={"fontSize": "10px", "color": "var(--pm-grey-500)"}),
            ],
        ))
    return html.Div(style={"display": "flex", "gap": "6px", "alignItems": "flex-end"}, children=cols)


def diverging_gauge(value) -> html.Div:
    """A centered diverging gauge for a lean in [-1, 1]: the fill runs from the
    centre to the value, green to the right (positive), red to the left
    (negative). The one place this surface earns colour — a net direction."""
    v = value if value is not None else 0.0
    v = max(-1.0, min(1.0, v))
    half = abs(v) * 50.0
    fill = {"position": "absolute", "top": "0", "bottom": "0", "borderRadius": "2px",
            "background": "var(--pos)" if v >= 0 else "var(--neg)"}
    if v >= 0:
        fill.update({"left": "50%", "width": f"{half:.1f}%"})
    else:
        fill.update({"right": "50%", "width": f"{half:.1f}%"})
    return html.Div(
        style={"position": "relative", "flex": "1", "height": "8px",
               "background": "var(--pm-grey-100)", "borderRadius": "2px"},
        children=[
            html.Div(style={"position": "absolute", "left": "50%", "top": "-2px", "bottom": "-2px",
                            "width": "1px", "background": "var(--pm-grey-500)"}),
            html.Div(style=fill),
        ],
    )


def magnitude_gauge(value, lo: str = "diffuse", hi: str = "concentrated") -> html.Div:
    """A neutral 0..1 magnitude gauge with end labels (no sign colour)."""
    v = max(0.0, min(1.0, value if value is not None else 0.0))
    w = round(v * 100)
    return html.Div(children=[
        html.Div(style={"position": "relative", "width": "100%", "height": "8px",
                        "background": "var(--pm-grey-100)", "borderRadius": "2px"},
                 children=[html.Div(style={"width": f"{w}%", "height": "100%",
                                           "background": "var(--pm-charcoal)", "borderRadius": "2px"})]),
        html.Div(style={"display": "flex", "justifyContent": "space-between", "fontSize": "10px",
                        "color": "var(--pm-grey-500)", "marginTop": "2px"},
                 children=[html.Span(lo), html.Span(hi)]),
    ])
