"""Shared number formatters for Tab 2 — Account Deep Dive.

One place for both halves of the formatting story so it is consistent, not
ad-hoc per cell:

  * **AG-Grid value formatters** (``*_FMT``) — function-strings evaluated
    client-side by dash-ag-grid, used by the positions / trades grids.
  * **Python formatters** (``money_compact`` / ``pct``) — used by the
    analytics panels and KPI strip, which build plain ``html`` (no grid).

Conventions:
  * Large dollar aggregates (KPI strip, analytics) → compact ``$49.6M``.
  * Dollar line-items (positions / trades cells) → full thousands
    separators, no decimals: ``$49,634,466`` / ``-$1,739,888``.
  * Percentages → signed for returns (``+304%``), absolute for weights
    (``42.4%``).
  * Quantities → thousands separators, sign preserved (``-53,783``).
Missing values render as an em dash.
"""
from __future__ import annotations

# Re-export the compact Python money formatter (single definition lives in
# aggregations, where the interpretation strings also use it).
from pm.ui.deepdive.aggregations import _fmt_money as money_compact


# ---------------------------------------------------------------------------
# Python formatters (for html-building panels)
# ---------------------------------------------------------------------------

def pct(value, dp: int = 1, signed: bool = False) -> str:
    """Percent string from a fraction (0.424 → '42.4%'); '—' when None."""
    if value is None:
        return "—"
    sign = "+" if (signed and value >= 0) else ""
    return f"{sign}{value * 100:.{dp}f}%"


def pct_of_nav(value, nav, dp: int = 1, signed: bool = True):
    """'+1.8% NAV' from a dollar value and the account NAV. ONE %-of-NAV
    convention on the risk surfaces: the denominator is |signed net asset value|
    (``AccountState.nav``), named in the section footer. None when either side is
    missing or NAV is zero — the caller renders a dash, never a fake 0%."""
    if value is None or not nav:
        return None
    v = value / abs(nav) * 100.0
    sign = "+" if (signed and v >= 0) else ""
    return f"{sign}{v:.{dp}f}% NAV"


# ---------------------------------------------------------------------------
# AG-Grid client-side value formatters (function-strings)
# ---------------------------------------------------------------------------

# Full dollars, thousands-separated, no decimals: $49,634,466 / -$1,739,888.
#
# NOTE: dash-ag-grid's function-string evaluator breaks on a curly-brace object
# literal in the body (e.g. ``toLocaleString(undefined, {maximumFractionDigits:
# 0})`` silently fails → the cell renders the raw number). So we round first and
# use the brace-free ``toLocaleString('en-US')`` for the thousands separators.
# Keep every function string below brace-free for the same reason.
MONEY_FULL_FMT = {
    "function": "params.value == null ? '—' : "
                "(params.value < 0 ? '-$' : '$') + "
                "Math.round(Math.abs(params.value)).toLocaleString('en-US')"
}

# Signed quantity, thousands-separated: 53,783 / -500.
QTY_FMT = {
    "function": "params.value == null ? '—' : "
                "Math.round(params.value).toLocaleString('en-US')"
}

# Signed percent from a fraction: +304% / -12%.
PCT_SIGNED_FMT = {
    "function": "params.value == null ? '—' : "
                "(params.value >= 0 ? '+' : '') + (params.value * 100).toFixed(0) + '%'"
}

# Absolute percent from a fraction, one decimal: 42.4%.
PCT_ABS_FMT = {
    "function": "params.value == null ? '—' : (params.value * 100).toFixed(1) + '%'"
}

# Signed percent from a fraction, one decimal: +1.8% / -0.4% (the %NAV
# convention at grid density — pct_of_nav's signed 1dp, client-side).
PCT_SIGNED_1DP_FMT = {
    "function": "params.value == null ? '—' : "
                "(params.value >= 0 ? '+' : '') + (params.value * 100).toFixed(1) + '%'"
}


# ---------------------------------------------------------------------------
# Conditional cell styles (data-meaning colour only)
# ---------------------------------------------------------------------------

SIGNED_COLOR_STYLE = {
    "styleConditions": [
        {"condition": "params.value < 0", "style": {"color": "var(--neg)", "fontWeight": "600"}},
        {"condition": "params.value > 0", "style": {"color": "var(--pos)", "fontWeight": "600"}},
    ],
    "defaultStyle": {"color": "var(--pm-charcoal)"},
}

BUYSELL_STYLE = {
    "styleConditions": [
        {"condition": "params.value && params.value.toLowerCase().indexOf('buy') === 0",
         "style": {"color": "var(--pos)", "fontWeight": "600"}},
        {"condition": "params.value && params.value.toLowerCase().indexOf('sell') === 0",
         "style": {"color": "var(--neg)", "fontWeight": "600"}},
    ],
}
