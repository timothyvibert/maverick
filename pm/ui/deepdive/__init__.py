"""Tab 2 — Account Deep Dive.

Per-account drill-down: account picker + KPI strip + five stacked sections
(Positions, Actionables, Analytics, Recent Trades). Reuses the shared modal,
the AG-Grid conventions, the palette and ``state_access`` from Tab 1; the only
genuinely new logic is three pure presentation aggregations in
``aggregations.py``. The UI reads from ``PortfolioState`` and never recomputes.
"""
