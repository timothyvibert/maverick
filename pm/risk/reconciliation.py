"""Engine-vs-BBG snapshot-greeks reconciliation -- the standing calibration diagnostic / regression.

Per-share comparison at *current* spot: ``pm.pricing`` truth greeks (via the
``pricing_adapter``) vs the BBG snapshot greeks the exposure layer reads. This is
the divergence the scenario layer must SURFACE (snapshot=BBG vs scenario=engine),
so it is retained as a standing diagnostic rather than a one-off.

Conventions are normalized per the calibrated convention map (measured against the live Terminal):

  * delta -- identity (both decimal).
  * vega  -- identity (both per 1 vol point).
  * gamma -- engine is per $1^2; **BBG GAMMA is per 1% move**. Compared as
             ``gamma_engine * S / 100`` so both sit in the BBG per-1% basis.
  * theta -- engine is per business day; BBG is ~calendar-day at short tenors and
             diverges with maturity (no single factor). **Reported with a caveat,
             never fake-reconciled.**
  * price -- engine vs ``PX_MID`` with ``iv_mid`` fed back is a *circular* anchor:
             it confirms the input pipeline, not the greeks.

Pure given a loaded ``PortfolioState`` (no Bloomberg). Run live for the real
divergence; the comparison/bucketing logic is unit-tested on a synthetic state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pm.risk.pricing_adapter import build_engine_legs

_GREEKS = ("delta", "gamma", "vega", "theta")


def reconcile_account(state, account_state, today=None) -> pd.DataFrame:
    """One row per option leg: engine truth greeks (conventions normalized) vs the
    BBG snapshot greeks, plus the circular price anchor and the bucketing tags.
    """
    legs = build_engine_legs(state, account_state, today=today)
    opt_snap = getattr(getattr(account_state, "snapshot", None), "options", None)
    rows: list[dict] = []
    for leg in legs:
        bbg = _bbg_greeks(opt_snap, leg.bbg_ticker)
        eng = leg.greeks(mode="truth") if leg.priceable else None
        row: dict = {
            "account": leg.account,
            "ticker": leg.bbg_ticker,
            "underlying": leg.underlying_bbg,
            "ot": leg.opt_type,
            "style": leg.style,
            "div_mode": leg.div_mode,
            "near_ex": _near_ex(leg),
            "sigma_src": leg.sigma_source,
            "S": leg.spot,
            "K": leg.K,
            "dte": int(round(leg.T * 252)),
            "moneyness": (leg.spot / leg.K) if (_finite(leg.spot) and leg.K) else None,
            "itm": _itm(leg),
            "priceable": leg.priceable,
            "p_eng": leg.price(mode="truth") if leg.priceable else None,
            "p_mid": leg.mid,
            "warnings": "; ".join(leg.warnings),
        }
        row["p_rel"] = _rel(row["p_eng"], row["p_mid"])
        if eng is not None and bbg is not None:
            normalized = {
                "delta": eng["delta"],
                "gamma": eng["gamma"] * leg.spot / 100.0,   # per $1^2 -> per 1% (BBG basis)
                "vega": eng["vega"],
                "theta": eng["theta"],
            }
            for g in _GREEKS:
                row[f"{g}_eng"] = normalized[g]
                row[f"{g}_bbg"] = bbg.get(g)
                row[f"{g}_rel"] = _rel(normalized[g], bbg.get(g))
            row["gamma_eng_native"] = eng["gamma"]          # per $1^2, pre-normalization
        rows.append(row)
    return pd.DataFrame(rows)


def reconcile_state(state, today=None) -> pd.DataFrame:
    """``reconcile_account`` across every account on the state."""
    frames = [
        reconcile_account(state, acc, today=today)
        for acc in getattr(state, "accounts", {}).values()
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate median / max |rel| per greek over the comparable rows, plus the
    theta divergence by tenor band (the headline). Returns a plain dict.
    """
    out: dict = {"n_legs": int(len(df))}
    if df.empty:
        return out
    comparable = df[df.get("delta_rel").notna()] if "delta_rel" in df else df.iloc[0:0]
    out["n_comparable"] = int(len(comparable))
    out["n_unpriceable"] = int((~df["priceable"]).sum()) if "priceable" in df else 0
    out["div_modes"] = df["div_mode"].value_counts().to_dict() if "div_mode" in df else {}
    for g in _GREEKS:
        col = f"{g}_rel"
        if col in comparable and not comparable[col].dropna().empty:
            ar = comparable[col].dropna().abs()
            out[g] = {"median_abs_rel": round(float(ar.median()), 4),
                      "max_abs_rel": round(float(ar.max()), 4)}
    if "p_rel" in comparable and not comparable["p_rel"].dropna().empty:
        out["price"] = {"median_abs_rel": round(float(comparable["p_rel"].dropna().abs().median()), 4)}
    # theta divergence by tenor band -- shows the maturity drift the gate flagged
    if "theta_rel" in comparable and "dte" in comparable:
        bands = {"<=45d": comparable["dte"] <= 45,
                 "46-365d": (comparable["dte"] > 45) & (comparable["dte"] <= 365),
                 ">365d": comparable["dte"] > 365}
        tb = {}
        for label, mask in bands.items():
            sub = comparable.loc[mask, "theta_rel"].dropna()
            if not sub.empty:
                tb[label] = {"n": int(len(sub)), "median_rel": round(float(sub.median()), 3)}
        out["theta_by_tenor"] = tb
    return out


def format_report(df: pd.DataFrame) -> str:
    """Human-readable per-leg table + the aggregate summary."""
    if df.empty:
        return "reconciliation: no option legs."
    s = summarize(df)
    show = [c for c in (
        "ticker", "ot", "style", "div_mode", "near_ex", "sigma_src", "S", "K",
        "dte", "moneyness", "itm", "p_eng", "p_mid", "p_rel",
        "delta_eng", "delta_bbg", "delta_rel",
        "gamma_eng", "gamma_bbg", "gamma_rel",
        "vega_eng", "vega_bbg", "vega_rel",
        "theta_eng", "theta_bbg", "theta_rel", "priceable", "warnings",
    ) if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 320,
                           "display.float_format", lambda x: f"{x:.4f}"):
        table = df[show].to_string(index=False)
    lines = [
        "=== ENGINE vs BBG snapshot greeks (per share, conventions normalized) ===",
        table, "",
        "=== SUMMARY ===",
        f"legs={s.get('n_legs')} comparable={s.get('n_comparable')} "
        f"unpriceable={s.get('n_unpriceable')} div_modes={s.get('div_modes')}",
        f"delta : {s.get('delta')}",
        f"vega  : {s.get('vega')}",
        f"gamma : {s.get('gamma')}  (engine x S/100 -> BBG per-1% basis)",
        f"theta : {s.get('theta')}  (engine per business-day; BBG diverges -- caveat, not reconciled)",
        f"theta by tenor: {s.get('theta_by_tenor')}",
        f"price : {s.get('price')}  (circular anchor: iv_mid fed back)",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def _bbg_greeks(opt_snap, ticker) -> dict | None:
    if opt_snap is None or ticker is None:
        return None
    try:
        if ticker not in opt_snap.index:
            return None
        row = opt_snap.loc[ticker]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    out = {
        "delta": _num(row.get("delta_mid")),
        "gamma": _num(row.get("gamma")),
        "vega": _num(row.get("vega")),
        "theta": _num(row.get("theta")),
    }
    return out if any(v is not None for v in out.values()) else None


def _near_ex(leg, window_days: int = 21) -> bool:
    """True when a discrete ex-date falls within ``window_days`` of today."""
    if leg.div_mode != "discrete" or leg.divs_df is None or leg.divs_df.empty:
        return False
    nearest = leg.divs_df["EX_DATE"].min()
    return (pd.Timestamp(nearest) - leg.today).days <= window_days


def _itm(leg) -> bool | None:
    if not _finite(leg.spot) or not leg.K:
        return None
    return (leg.spot > leg.K) if leg.opt_type == "Call" else (leg.spot < leg.K)


def _rel(a, b):
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return (a - b) / abs(b)


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and np.isfinite(v)


def main() -> int:
    """Run the standing diagnostic against the live book: load the latest
    extract with Bloomberg up, reconcile every account, print the per-leg
    report + the summary. Exit non-zero when nothing was comparable.

    Invocation: ``python -m pm.risk.reconciliation``
    """
    from pm.config import EXTRACT_DATA_DIR
    from pm.store.portfolio_state import load_portfolio_state

    state = load_portfolio_state(EXTRACT_DATA_DIR)
    if not state.bloomberg_ok:
        print("Bloomberg unavailable — the engine-vs-snapshot comparison needs "
              "live snapshot greeks.")
        return 2
    any_rows = False
    for account_id, account_state in state.accounts.items():
        df = reconcile_account(state, account_state)
        print(f"\n=== {account_id} ===")
        print(format_report(df))
        any_rows = any_rows or (df is not None and not df.empty)
    return 0 if any_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
