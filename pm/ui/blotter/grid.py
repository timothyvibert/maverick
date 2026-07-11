"""AG Grid column definitions + row-data shaping for the morning blotter.

Pure functions: fires -> grid rows, the position descriptor, the group
re-sort, tier -> style. No Dash imports here so these are unit-testable
without a browser. Display metrics are pulled from each fire's Position via
the already-loaded state — nothing is recomputed.

Grouping note: AG Grid row grouping is an Enterprise feature, so the
"Account | Pattern" toggle is implemented as a server-side re-sort
(``sort_rows``) rather than native collapsible groups — Community-only, no
licence. The account is always a visible, sortable, filterable column.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pm.ingest.position_builder import Position
from pm.insight.pattern_groups import GROUP_ORDER, PATTERN_GROUP
from pm.insight.patterns import Fire
from pm.store.portfolio_state import PortfolioState
from pm.store.suppression_store import is_active
from pm.ui import state_access as sa


# ---------------------------------------------------------------------------
# Position descriptor (pure)
# ---------------------------------------------------------------------------

def _direction_word(qty: Optional[float]) -> str:
    if qty is None:
        return ""
    return "Short" if qty < 0 else "Long"


def _abs_qty_str(qty: Optional[float]) -> str:
    if qty is None:
        return ""
    try:
        return f"{abs(int(qty)):,}"
    except (TypeError, ValueError):
        return ""


def _fmt_expiry(expiry) -> str:
    if expiry is None:
        return ""
    try:
        return expiry.strftime("%b-%y")  # e.g. "Jun-26"
    except Exception:
        return str(expiry)


def format_position_descriptor(position: Optional[Position]) -> str:
    """Account-specific, complete one-line descriptor of a position.

    Option   -> "Short 500 × Put $285 Jun-26"
    Equity   -> "Long 53,783 sh"
    Fund/ETF -> "Long 217,600 sh"
    Cash     -> "Cash USD" (the product label held on Position.symbol)
    Reads only fields already on the Position — no recompute.
    """
    if position is None:
        return "—"
    ac = position.asset_class
    qty = position.quantity

    if ac == "option":
        direction = _direction_word(qty)
        n = _abs_qty_str(qty)
        right = (position.right or position.option_type or "").title() or "Option"
        strike = f"${position.strike:g}" if position.strike is not None else ""
        expiry = _fmt_expiry(position.expiry)
        parts = [direction, n, "×", right, strike, expiry]
        return " ".join(p for p in parts if p).strip()

    if ac in ("equity", "fund_etf"):
        direction = _direction_word(qty)
        n = _abs_qty_str(qty)
        body = " ".join(p for p in (direction, n) if p)
        return (f"{body} sh").strip() if body else "—"

    if ac == "cash":
        return position.symbol or "Cash"

    # other
    return position.symbol or (ac or "—")


# ---------------------------------------------------------------------------
# Tier styling (pure) — neutral palette. Colour carries data
# meaning only; tier urgency is weight + a CSS left-accent, not red.
# ---------------------------------------------------------------------------

def cell_click_target(col_id: Optional[str]) -> str:
    """Route a grid cell click to a modal view: clicking the ticker column
    opens the Tearsheet; any other column opens the Alert view."""
    return "tearsheet" if col_id == "underlying" else "alert"


# ---------------------------------------------------------------------------
# Prev/Next navigation (pure) — operates on the grid's currently-visible
# CONSOLIDATED rows (post filter+sort, in display order; virtualRowData).
# Each visible row is one (account, position_id); navigation steps positions.
# ---------------------------------------------------------------------------

def visible_row_index(visible_rows, account, position_id) -> int:
    """Index of the (account, position_id) row within the visible rows, or -1."""
    for i, r in enumerate(visible_rows or []):
        if r.get("_account") == account and r.get("_position_id") == position_id:
            return i
    return -1


def step_row(visible_rows, account, position_id, direction):
    """Return the adjacent row dict in the requested direction ('prev'/'next'),
    or None at a boundary / if the current row isn't found. No wraparound."""
    rows = visible_rows or []
    idx = visible_row_index(rows, account, position_id)
    if idx < 0:
        return None
    new_idx = idx - 1 if direction == "prev" else idx + 1
    if 0 <= new_idx < len(rows):
        return rows[new_idx]
    return None


def nav_display(visible_rows, account, position_id):
    """Return (indicator_text, prev_disabled, next_disabled) for the modal's
    prev/next controls given the visible order and the current position row."""
    rows = visible_rows or []
    idx = visible_row_index(rows, account, position_id)
    total = len(rows)
    if idx < 0 or total == 0:
        return "", True, True
    return f"{idx + 1} of {total}", idx <= 0, idx >= total - 1


def tier_cell_style(tier: int) -> dict:
    if tier == 1:
        return {"fontWeight": "700", "color": "var(--pm-charcoal)"}
    if tier == 2:
        return {"fontWeight": "600", "color": "var(--pm-grey-700)"}
    return {"fontWeight": "400", "color": "var(--pm-grey-500)"}


# AG Grid function-string formatters (evaluated client-side by dash-ag-grid).
_PCT_FORMATTER = {
    "function": "params.value == null ? '—' : "
                "(params.value >= 0 ? '+' : '') + (params.value * 100).toFixed(0) + '%'"
}
_PCT_ABS_FORMATTER = {
    "function": "params.value == null ? '—' : (params.value * 100).toFixed(1) + '%'"
}
_INT_FORMATTER = {"function": "params.value == null ? '—' : params.value"}
_SIGNED_COLOR_STYLE = {
    "styleConditions": [
        {"condition": "params.value < 0", "style": {"color": "var(--neg)", "fontWeight": "600"}},
        {"condition": "params.value > 0", "style": {"color": "var(--pos)", "fontWeight": "600"}},
    ],
    "defaultStyle": {"color": "var(--pm-charcoal)"},
}
_TIER_STYLE = {
    "styleConditions": [
        {"condition": "params.value == 1", "style": tier_cell_style(1)},
        {"condition": "params.value == 2", "style": tier_cell_style(2)},
        {"condition": "params.value == 3", "style": tier_cell_style(3)},
    ],
}
_TIER_FORMATTER = {"function": "params.value == null ? '' : 'T' + params.value"}


def build_blotter_columns() -> list[dict]:
    """AG Grid column definitions. Account is the always-visible, sortable,
    filterable leftmost column (Community has no row grouping). Grouping is a
    re-sort handled in ``sort_rows``; the columns themselves never change."""
    return [
        {
            "field": "account", "headerName": "Account",
            "filter": "agTextColumnFilter", "width": 120, "pinned": "left",
            "cellClass": "blotter-account-cell",
        },
        {
            "field": "underlying", "headerName": "Ticker",
            "filter": "agTextColumnFilter", "width": 100,
            "cellClass": "blotter-ticker-cell",
            "valueFormatter": {"function": "params.value ? params.value : '—'"},
        },
        {"field": "position_label", "headerName": "Position", "width": 190,
         "filter": "agTextColumnFilter", "tooltipField": "position_label"},
        {
            "field": "tier", "headerName": "Tier", "width": 80,
            "filter": "agNumberColumnFilter", "type": "rightAligned",
            "valueFormatter": _TIER_FORMATTER, "cellStyle": _TIER_STYLE,
        },
        {
            # One row per position; distinct alerts comma-joined on one
            # logical line. Compact fixed-height row: the cell stays on one
            # line and truncates with an ellipsis (full text in the tooltip
            # and the modal), so row height never grows with alert count.
            "field": "alerts", "headerName": "Alerts",
            "flex": 2, "minWidth": 300, "tooltipField": "alerts",
            "cellClass": "blotter-alerts-cell",
            "cellStyle": {"whiteSpace": "nowrap", "overflow": "hidden",
                          "textOverflow": "ellipsis"},
        },
        {
            "field": "pnl_pct", "headerName": "P&L %", "width": 100,
            "type": "rightAligned", "filter": "agNumberColumnFilter",
            "valueFormatter": _PCT_FORMATTER, "cellStyle": _SIGNED_COLOR_STYLE,
        },
        {
            "field": "position_size_pct", "headerName": "% NAV", "width": 95,
            "type": "rightAligned", "filter": "agNumberColumnFilter",
            "valueFormatter": _PCT_ABS_FORMATTER,
        },
        {
            "field": "dte", "headerName": "DTE", "width": 80,
            "type": "rightAligned", "filter": "agNumberColumnFilter",
            "valueFormatter": _INT_FORMATTER,
        },
    ]


# ---------------------------------------------------------------------------
# Row data
# ---------------------------------------------------------------------------

def _dte_for(position) -> Optional[int]:
    if position is None or position.asset_class != "option" or position.expiry is None:
        return None
    try:
        return (position.expiry - date.today()).days
    except Exception:
        return None


def _alert_entries(group: list[Fire]) -> list[dict]:
    """Structured alert list for one position's fires (already tier-sorted):
    one entry per distinct pattern, severity order preserved. Each entry is
    ``{pattern_id, group, name, tier}`` — the unit the group/type filters act on.
    ``group`` is None for an unmapped id (then never hidden by the group filter)."""
    entries: list[dict] = []
    seen: set[str] = set()
    for f in group:
        if f.pattern_id in seen:
            continue
        seen.add(f.pattern_id)
        entries.append({
            "pattern_id": f.pattern_id,
            "group": PATTERN_GROUP.get(f.pattern_id),
            "name": f.pattern_name,
            "tier": f.tier,
            # Material-change re-surfacing: this alert was muted but its condition moved materially, so it is
            # active again — tag it so the desk sees it returned by a move, not afresh.
            "resurfaced": getattr(f.suppression, "kind", None) == "resurfaced",
        })
    return entries


def _row_alert_fields(entries: list[dict]) -> dict:
    """Derive a row's alert-driven fields from its (severity-ordered, deduped)
    entries. Shared by the row builder and the alert-level filter so the two can
    never diverge — the default (nothing hidden) render is byte-identical to a
    rebuild over the same entries. ``entries`` must be non-empty."""
    # A discreet, plain-English tag when any alert here re-surfaced — quieter
    # than the tier badge; empty (so byte-identical) when nothing has re-surfaced.
    resurfaced = any(e.get("resurfaced") for e in entries)
    return {
        "alerts": ", ".join(e["name"] for e in entries) + (" · re-surfaced" if resurfaced else ""),
        "tier": min(e["tier"] for e in entries),       # row tier = max severity
        "pattern_name": entries[0]["name"],             # primary, for grouping/sort
        "_primary_pattern_id": entries[0]["pattern_id"],
        "_alerts": entries,
    }


def consolidate_fires_to_rows(fires: list[Fire], state: PortfolioState) -> list[dict]:
    """One row per (account, position_id). All of a position's fires (alerts)
    are stacked in the ``alerts`` cell; the row's ``tier`` is the max severity
    (lowest tier number); ``pattern_name`` is the primary (most-severe) fire's
    pattern (used for Pattern grouping/sort). Hidden ``_fire_ids`` lists the
    fires' pattern_ids (most-severe first) so the modal can render all of them.
    Unsorted — ``sort_rows`` orders the result per the active grouping.

    Suppressed/snoozed fires are dropped up front, so cells, counts, tier
    and the per-row alert list are all derived from the active set — a position whose
    every alert is muted produces no blotter row. When nothing is suppressed this is a
    no-op and the output is byte-identical to before."""
    fires = [f for f in fires if is_active(f)]
    groups: dict[tuple, list[Fire]] = {}
    order: list[tuple] = []
    for f in fires:
        key = (f.account, f.position_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(f)

    rows: list[dict] = []
    for account, position_id in order:
        group = sorted(groups[(account, position_id)], key=lambda f: f.tier)
        primary = group[0]
        position = sa.position_by_id(state, account, position_id)
        acc = state.accounts.get(account)
        nav = abs(acc.nav) if (acc and acc.nav) else 0.0

        pnl_pct = None
        position_size_pct = None
        if position is not None:
            pnl_pct = sa.coerce_float(position.unrealized_pnl_pct)
            mv = sa.coerce_float(position.market_value)
            if mv is not None and nav:
                position_size_pct = abs(mv) / nav

        label_text = (format_position_descriptor(position)
                      if position is not None else "Account")

        # Structured per-alert list (hidden) — drives both the displayed alerts
        # cell and the group/type slicing, deduped on pattern in severity order.
        # The displayed fields come from _row_alert_fields so a filter rebuild
        # can never diverge from this initial render.
        entries = _alert_entries(group)

        rows.append({
            # Hidden — one row per position; click/nav resolve via these.
            "_account": account,
            "_position_id": position_id,
            "_underlying": primary.underlying,
            # Non-deduped, one id per fire (kept for the existing row contract).
            "_fire_ids": [f.pattern_id for f in group],
            # Displayed metrics (alert-derived fields added by _row_alert_fields:
            # alerts, tier, pattern_name, _primary_pattern_id, _alerts).
            "account": account,
            "underlying": primary.underlying or "",
            "position_label": label_text,
            "pnl_pct": pnl_pct,
            "position_size_pct": position_size_pct,
            "dte": _dte_for(position),
            **_row_alert_fields(entries),
        })
    return rows


# ---------------------------------------------------------------------------
# Server-side slicing (pure): account (row-level) + group (alert-level).
# These feed the existing rowData path — Community only, NOT an Enterprise Set
# Filter — and compose with the tier filter + sort in the blotter callback.
# ---------------------------------------------------------------------------

def apply_account_filter(rows: list[dict], selected_accounts) -> list[dict]:
    """Row-level: keep rows whose account is selected. Empty / None selection =
    all accounts (the default — no narrowing)."""
    sel = set(selected_accounts or [])
    if not sel:
        return list(rows)
    return [r for r in rows if r.get("account") in sel]


def apply_alert_filters(rows: list[dict], shown_groups=None) -> list[dict]:
    """Alert-level: drop each row's alerts whose group is not shown, rebuild the
    row's alerts cell / tier / primary from the survivors, and drop a row only
    when nothing visible is left. ``shown_groups=None`` → all groups shown, so
    the default call reproduces every row unchanged. An alert with an unmapped
    (None) group is never hidden."""
    shown = set(GROUP_ORDER if shown_groups is None else shown_groups)
    out: list[dict] = []
    for r in rows:
        survivors = [
            a for a in r.get("_alerts", [])
            if a.get("group") is None or a["group"] in shown
        ]
        if not survivors:
            continue                       # no visible alert → drop the row
        nr = dict(r)
        nr.update(_row_alert_fields(survivors))
        nr["_fire_ids"] = [a["pattern_id"] for a in survivors]
        out.append(nr)
    return out


# ---------------------------------------------------------------------------
# Grouping-as-sort: re-sort rows by the active grouping, T1 first.
# ---------------------------------------------------------------------------

def _group_value(row: dict, group_by: str):
    return row.get("account") if group_by == "account" else row.get("pattern_name")


def _sort_key(row: dict, group_by: str):
    primary = _group_value(row, group_by)
    tier = row.get("tier")
    tier = tier if tier is not None else 99
    pnl = row.get("pnl_pct")
    pnl_key = pnl if pnl is not None else float("inf")  # nulls last within tier
    return (str(primary or "~"), tier, pnl_key)


def sort_rows(rows: list[dict], group_by: str = "account") -> list[dict]:
    """Return a re-sorted copy. group_by='account' → account, then tier (T1
    first), then most-negative P&L; group_by='pattern' → pattern, then tier.
    Each returned row gets ``_group_first`` = True on the first row of each
    group block, so the grid can draw a subtle separator without collapsing."""
    group_by = group_by if group_by in ("account", "pattern") else "account"
    ordered = sorted(rows, key=lambda r: _sort_key(r, group_by))
    out: list[dict] = []
    prev = object()
    for r in ordered:
        gv = _group_value(r, group_by)
        nr = dict(r)
        nr["_group_first"] = gv != prev
        prev = gv
        out.append(nr)
    return out


# ---------------------------------------------------------------------------
# Grid options — density + sort + the group-block separator.
# ---------------------------------------------------------------------------

def default_grid_options() -> dict:
    return {
        # Compact, uniform rows. Alerts are comma-joined on one line and
        # truncate with an ellipsis (no autoHeight), so every row is 28px.
        "rowHeight": 28,
        "headerHeight": 32,
        "animateRows": False,
        # Highlight-to-copy: native text selection + Ctrl-C copies the selected
        # cells in visual order. Community only — not Enterprise range copy.
        "enableCellTextSelection": True,
        "ensureDomOrder": True,
        "rowClassRules": {
            # Top-border separator above the first row of each group block.
            "blotter-group-start": "params.data && params.data._group_first",
            # Subtle (non-red) urgency accent on T1 rows.
            "blotter-row-t1": "params.data && params.data.tier == 1",
        },
        "defaultColDef": {
            "sortable": True,
            "resizable": True,
            "suppressMovable": False,
        },
    }
