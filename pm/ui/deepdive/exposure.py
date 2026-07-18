"""Current-book exposure read helpers for the Risk section.

The posture strip (``risk_cockpit._posture_strip``) is the one consumer: it
reads the precomputed account exposure (``acc.exposure`` from
``pm.risk.exposure``) and renders direction, leverage, volatility and decay
in four cells. This module keeps the shared pieces that outlived the old
panel layout — the beta definitional note (now a tooltip), the vega tenor
bucket's honest display value, and the provenance/missing-data line. A pure
read of ``acc.exposure`` — no compute, no Bloomberg, no recompute.
"""
from __future__ import annotations

_BETA_NOTE = (
    "Adjusted (Blume) is the default — name betas converge toward 1.0 in a selloff "
    "(crash correlation), so it is the steadier downside proxy; raw is the current "
    "empirical sensitivity. Both vs SPX, 2y weekly, per name."
)


def _bucket_cell(b) -> str:
    """A bucket's display value. Dash when it holds no options — or when every
    option in it is missing vega (that $0 would be pure missing data, not zero
    exposure)."""
    from pm.ui.deepdive.aggregations import _fmt_money
    n_missing = getattr(b, "n_missing_vega", 0) or 0
    if not b.n_options or n_missing >= b.n_options:
        return "—"
    return _fmt_money(b.dollar_vega)


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
