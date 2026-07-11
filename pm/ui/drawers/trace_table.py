"""Shared helper: render a signal/fire ``trace`` dict as a structured,
complete HTML table.

This is the auditability surface. It must show EVERYTHING in the trace —
every input, its value, its source, and its stale flag; the computation
string; every threshold; the result. Nothing summarized away.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

from dash import html


def _fmt_value(v: Any) -> str:
    """Readable scalar formatting. Dates as ISO, floats trimmed, None as —."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:,.4g}"
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return str(v)


def _render_nested(v: Any) -> Any:
    """Render a result/value that may be a dict or list as a compact block."""
    if isinstance(v, dict):
        return html.Div(className="trace-nested", children=[
            html.Div(className="trace-nested-row", children=[
                html.Span(f"{k}", className="trace-nested-key"),
                html.Span(_fmt_value(val), className="trace-nested-val"),
            ])
            for k, val in v.items()
        ])
    if isinstance(v, (list, tuple)):
        return html.Div(", ".join(_fmt_value(x) for x in v),
                        className="trace-nested-val")
    return html.Span(_fmt_value(v), className="trace-nested-val")


def _inputs_section(inputs: dict) -> html.Div:
    header = html.Div(className="trace-inputs-header", children=[
        html.Div("Input"), html.Div("Value"), html.Div("Source"), html.Div("Stale"),
    ])
    rows = [header]
    for name, entry in inputs.items():
        entry = entry if isinstance(entry, dict) else {"value": entry}
        value = entry.get("value")
        source = entry.get("source", "—")
        stale = bool(entry.get("stale", False))
        rows.append(html.Div(
            className="trace-input-row" + (" trace-stale" if stale else ""),
            children=[
                html.Div(name, className="trace-input-name"),
                html.Div(_render_nested(value) if isinstance(value, (dict, list))
                         else _fmt_value(value), className="trace-input-value"),
                html.Div(str(source), className="trace-input-source"),
                html.Div("STALE" if stale else "ok",
                         className="trace-input-stale"),
            ],
        ))
    return html.Div(className="trace-inputs", children=rows)


def render_trace(trace: dict) -> html.Div:
    """Render a trace dict as nested tables (inputs / computation /
    thresholds / result). Defensive against a missing/empty trace."""
    if not trace:
        return html.Div("No trace available.", className="trace-empty")

    inputs = trace.get("inputs", {}) or {}
    computation = trace.get("computation", "")
    thresholds = trace.get("thresholds", {}) or {}
    result = trace.get("result", None)

    blocks = []

    # Inputs
    blocks.append(html.Div(className="trace-block", children=[
        html.Div("Inputs", className="trace-block-label"),
        _inputs_section(inputs) if inputs
        else html.Div("(none)", className="trace-muted"),
    ]))

    # Computation
    blocks.append(html.Div(className="trace-block", children=[
        html.Div("Computation", className="trace-block-label"),
        html.Code(computation or "—", className="trace-computation"),
    ]))

    # Thresholds
    if thresholds:
        threshold_rows = [
            html.Div(className="trace-kv", children=[
                html.Span(k, className="trace-kv-key"),
                html.Span(_fmt_value(v), className="trace-kv-val"),
            ])
            for k, v in thresholds.items()
        ]
    else:
        threshold_rows = [html.Div("(none)", className="trace-muted")]
    blocks.append(html.Div(className="trace-block", children=[
        html.Div("Thresholds", className="trace-block-label"),
        html.Div(threshold_rows, className="trace-thresholds"),
    ]))

    # Result
    blocks.append(html.Div(className="trace-block", children=[
        html.Div("Result", className="trace-block-label"),
        _render_nested(result),
    ]))

    return html.Div(className="trace-table", children=blocks)
