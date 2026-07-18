"""Portfolio-level diagnostics: sector exposure, weighted beta,
earnings calendar.

V1 takes ``list[Position]`` and drops the style_mix component (the
holdings extract has no ``Style`` column). Funds / ETFs
contribute alongside equities; cash and other are excluded from
sector / beta math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from pm.core.portfolio_signals import FIELDS
from pm.ingest.position_builder import Position
from pm.core import clock


_EQUITY_LIKE = ("equity", "fund_etf")


@dataclass
class PortfolioDiagnostics:
    sector_exposure: dict[str, float]    # GICS sector → fraction of equity-like NAV
    weighted_beta: float | None          # NAV-weighted β across equity-like book
    earnings_calendar: list[dict]        # [{ticker, symbol, name, date, days_to_earnings}]
    warnings: list[str] = field(default_factory=list)


def compute_portfolio_diagnostics(
    positions: list[Position],
    underlying_snapshot: pd.DataFrame,
) -> PortfolioDiagnostics:
    equity_like = [p for p in positions if p.asset_class in _EQUITY_LIKE]
    if not equity_like:
        return PortfolioDiagnostics({}, None, [], ["No equity-like positions."])

    eq = pd.DataFrame([{
        "bbg_ticker": p.bbg_ticker,
        "symbol": p.symbol,
        "market_value": p.market_value if p.market_value is not None else 0.0,
    } for p in equity_like])

    wanted = ["GICS_SECTOR_NAME", "security_name",
              FIELDS["beta"], FIELDS["earn_dt"]]
    cols = [c for c in wanted if c in underlying_snapshot.columns]
    snap_subset = (
        underlying_snapshot[cols].copy()
        if cols else pd.DataFrame(index=underlying_snapshot.index)
    )
    snap_subset = snap_subset.rename_axis("bbg_ticker").reset_index()
    eq = eq.merge(snap_subset, on="bbg_ticker", how="left")

    eq_nav = eq["market_value"].abs().sum()
    if eq_nav == 0:
        return PortfolioDiagnostics({}, None, [], ["Zero equity-like NAV."])

    # ---- Sector exposure --------------------------------------------------
    sector_exp: dict[str, float] = {}
    if "GICS_SECTOR_NAME" in eq.columns:
        grouped = (
            eq.groupby("GICS_SECTOR_NAME", dropna=False)["market_value"].sum()
            / eq_nav
        ).fillna(0)
        for k, v in grouped.items():
            key = "Unclassified" if (k is None or pd.isna(k)) else str(k)
            sector_exp[key] = sector_exp.get(key, 0.0) + float(v)
    else:
        sector_exp = {"Unclassified": 1.0}

    # ---- Weighted beta ----------------------------------------------------
    beta_col = FIELDS["beta"]
    weighted_beta: float | None = None
    if beta_col in eq.columns:
        eligible = eq[eq[beta_col].notna() & (eq["market_value"].abs() > 0)]
        if not eligible.empty:
            num = (eligible[beta_col].astype(float) * eligible["market_value"]).sum()
            den = eligible["market_value"].sum()
            if den != 0:
                weighted_beta = float(num / den)

    # ---- Earnings calendar (next 30 days) ---------------------------------
    today = clock.today()
    horizon = today + timedelta(days=30)
    earnings: list[dict] = []
    earn_col = FIELDS["earn_dt"]
    if earn_col in eq.columns:
        for _, r in eq.iterrows():
            ed = r.get(earn_col)
            if ed is None:
                continue
            try:
                if pd.isna(ed):
                    continue
            except (TypeError, ValueError):
                continue
            try:
                ed_date = pd.to_datetime(ed).date()
            except Exception:
                continue
            if today <= ed_date <= horizon:
                name_val = r.get("security_name", "")
                earnings.append({
                    "ticker": r["bbg_ticker"],
                    "symbol": r["symbol"],
                    "name": name_val if isinstance(name_val, str) else "",
                    "date": ed_date.isoformat(),
                    "days_to_earnings": (ed_date - today).days,
                })
    earnings.sort(key=lambda x: x["days_to_earnings"])

    warnings: list[str] = []
    if weighted_beta is None:
        warnings.append("Beta missing for all equity-like positions — weighted β skipped.")

    return PortfolioDiagnostics(
        sector_exposure=sector_exp,
        weighted_beta=weighted_beta,
        earnings_calendar=earnings,
        warnings=warnings,
    )
