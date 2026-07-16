"""Deterministic stress / scenario engine.

Two entry points, one shared shocked-input / price path so a dialed point and a
precomputed preset are the *same* reprice, never an interpolation:

  * interactive (the live consumer) -- ``price_scenario`` (in state_access) calls
    ``shock_reprice`` (per position/structure) + ``spot_vol_grid`` (the heatmap
    mesh) live, read-only over already-loaded state. Everything the scenario
    section renders comes through here, at fast BS2002.
  * explicit truth tier -- ``run_account_scenario`` computes the co-moving preset
    table (truth-CRR n=200) + the fast P&L curve onto ``AccountState.scenario``.
    NOT in the load path (it cost seconds per account and nothing rendered it);
    retained as the API for a future commit-at-truth view.

Pricer tiers (enforced): every SWEEP / grid is **fast vectorized BS2002**; **truth-CRR**
is used only for discrete scenario *points* (preset table) and a committed point. A
sweep is never priced at truth. Greeks / prices come from the engine (the scenario
boundary); the exposure layer's outputs are untouched.

Shock library (blueprint 5.2) -- co-moving axis shocks applied *together*: market
+-5/10/20 % (beta-mapped via the exposure layer's SPX ``EQY_BETA``), vol +-5/10 pts, crash,
melt-up, rates +-50 bps (parallel curve shift), time +1w/+1m (full reprice at the
shorter tenor, discrete divs re-anchored to the shifted date). A custom dialed shock
is just a ``ShockSpec`` with arbitrary axes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from pm.core.bloomberg_client import pick_rate_for_dte
from pm.pricing import american_crr, european, strategy
from pm.pricing.american_crr import DEFAULT_CRR_STEPS, FAST_CRR_STEPS
from pm.pricing.conventions import year_frac
from pm.risk.pricing_adapter import EngineLeg, build_engine_legs

logger = logging.getLogger(__name__)

BETA_FIELD = "EQY_BETA"          # SPX-adjusted beta (snapshot.underlyings)
DEFAULT_BETA = 1.0               # full market participation when a name has no beta
SIGMA_FLOOR = 1e-4
_T_FLOOR = 1e-6
# A beta-mapped move past -100% (|beta| x shock, e.g. beta 6 at SPX -20%) would drive
# the shocked spot negative — NaN through the pricers. A stock cannot be worth less
# than nothing: option inputs clamp to one cent (the kernels take logs, and the fast
# greeks' spot bump has an absolute floor of 1e-4 — the clamp must stay above it),
# equity legs clamp to exactly zero (full loss of value, never more).
_SPOT_FLOOR = 0.01
CURVE_SPAN_PCT = 25.0            # +- SPX move spanned by the (v1) P&L curve
CURVE_POINTS = 101
PRESET_STEPS = FAST_CRR_STEPS    # leaned preset-table tier (n=200 truth)

# spot x vol heatmap mesh (fast vectorized BS2002).
GRID_SPOT_SPAN = 20.0
GRID_SPOT_N = 21                 # -20..+20 in 2% steps
GRID_VOL_PTS = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]


# --------------------------------------------------------------------------
# Shock library
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ShockSpec:
    """One co-moving scenario. ``spot_pct`` is an SPX move (beta-mapped per name);
    ``vol_pts`` an absolute vol-point shift; ``rate_bps`` a parallel curve shift;
    ``time_days`` a forward calendar-day shift."""
    name: str
    label: str
    spot_pct: float = 0.0
    vol_pts: float = 0.0
    rate_bps: float = 0.0
    time_days: int = 0

    def axes(self) -> dict:
        d = {}
        if self.spot_pct:
            d["spot_pct"] = self.spot_pct
        if self.vol_pts:
            d["vol_pts"] = self.vol_pts
        if self.rate_bps:
            d["rate_bps"] = self.rate_bps
        if self.time_days:
            d["time_days"] = self.time_days
        return d


SHOCK_LIBRARY: list[ShockSpec] = [
    ShockSpec("mkt_dn_20", "SPX -20%", spot_pct=-20.0),
    ShockSpec("mkt_dn_10", "SPX -10%", spot_pct=-10.0),
    ShockSpec("mkt_dn_5", "SPX -5%", spot_pct=-5.0),
    ShockSpec("mkt_up_5", "SPX +5%", spot_pct=5.0),
    ShockSpec("mkt_up_10", "SPX +10%", spot_pct=10.0),
    ShockSpec("mkt_up_20", "SPX +20%", spot_pct=20.0),
    ShockSpec("vol_up_10", "Vol +10 pts", vol_pts=10.0),
    ShockSpec("vol_up_5", "Vol +5 pts", vol_pts=5.0),
    ShockSpec("vol_dn_5", "Vol -5 pts", vol_pts=-5.0),
    ShockSpec("vol_dn_10", "Vol -10 pts", vol_pts=-10.0),
    ShockSpec("crash", "Crash (-20% spot, +10 vol)", spot_pct=-20.0, vol_pts=10.0),
    ShockSpec("meltup", "Melt-up (+15% spot, -5 vol)", spot_pct=15.0, vol_pts=-5.0),
    ShockSpec("rates_up_50", "Rates +50 bps", rate_bps=50.0),
    ShockSpec("rates_dn_50", "Rates -50 bps", rate_bps=-50.0),
    ShockSpec("time_1w", "Time +1 week", time_days=7),
    ShockSpec("time_1m", "Time +1 month", time_days=30),
]

_MARKET_BAND_SHOCKS = ("mkt_dn_20", "mkt_dn_10", "mkt_dn_5", "mkt_up_5", "mkt_up_10", "mkt_up_20")


# --------------------------------------------------------------------------
# Result types (load-path precompute)
# --------------------------------------------------------------------------
@dataclass
class ScenarioPoint:
    name: str
    label: str
    axes: dict
    pnl: float                       # account P&L under the shock (truth-CRR n=200)
    pnl_pct: Optional[float]
    attribution: dict                # empty on the leaned load path (computed on-expand)
    trace: dict


@dataclass
class AccountScenario:
    account: str
    as_of: date
    nav: Optional[float]
    scenarios: list[ScenarioPoint]   # ranked worst-loss first
    curve: dict
    n_priceable: int
    n_unpriceable: int
    div_modes: dict
    warnings: list
    trace: dict


# --------------------------------------------------------------------------
# Explicit truth-tier entry point (presets at n=200, no eager attribution).
# Not called by the load path — see the module docstring.
# --------------------------------------------------------------------------
def run_account_scenario(state, today=None) -> None:
    for acc in getattr(state, "accounts", {}).values():
        try:
            acc.scenario = compute_account_scenario(state, acc, today=today)
        except Exception as exc:  # noqa: BLE001
            logger.warning("scenario compute failed for %s: %s",
                           getattr(acc, "account", "?"), exc)
            acc.scenario = None


def compute_account_scenario(state, account_state, today=None) -> AccountScenario:
    today_ts = _normalize_today(today)
    all_legs = build_engine_legs(state, account_state, today=today_ts)
    legs = [lg for lg in all_legs if lg.priceable]
    beta_map = _beta_map(account_state)
    equities = _equity_legs(account_state, beta_map)
    curve_pts = getattr(state, "risk_free_curve", None) or []
    nav = _num(getattr(account_state, "nav", None))

    # leaned truth baseline at n=200 (no eager greeks/attribution)
    base_price = {lg.position_id: _truth_price(lg, lg.spot, lg.sigma, lg.r, lg.T,
                                               lg.today, n_steps=PRESET_STEPS)
                  for lg in legs}

    scenarios: list[ScenarioPoint] = []
    for shock in SHOCK_LIBRARY:
        pnl = _account_pnl(legs, equities, beta_map, base_price, curve_pts, today_ts,
                           shock, mode="truth", n_steps=PRESET_STEPS)
        scenarios.append(ScenarioPoint(
            name=shock.name, label=shock.label, axes=shock.axes(),
            pnl=pnl, pnl_pct=(pnl / nav if nav else None), attribution={},
            trace={"pricer": "truth-CRR n=200 (per point)", "shock": shock.axes()}))
    scenarios.sort(key=lambda s: s.pnl)

    curve = _portfolio_curve(legs, equities, beta_map, scenarios, today_ts)
    div_modes: dict = {}
    for lg in legs:
        div_modes[lg.div_mode] = div_modes.get(lg.div_mode, 0) + 1

    return AccountScenario(
        account=getattr(account_state, "account", ""), as_of=today_ts.date(), nav=nav,
        scenarios=scenarios, curve=curve,
        n_priceable=len(legs), n_unpriceable=len(all_legs) - len(legs), div_modes=div_modes,
        warnings=[w for lg in legs for w in lg.warnings],
        trace={"pricer_table": "truth-CRR n=200", "pricer_curve": "fast vectorized BS2002",
               "attribution": "on-expand only", "as_of": today_ts.date().isoformat()})


# --------------------------------------------------------------------------
# Interactive reprice — per position/structure impact (the price_scenario table)
# --------------------------------------------------------------------------
def shock_reprice(state, account_state, shock: ShockSpec, today=None, target=None,
                  mode="fast") -> dict:
    """Per-position / structure P&L (+ shocked-state dollar greeks) under one shock,
    plus the account total. The per-position rows SUM to the account total — the
    total covers the PRICEABLE book only, so the return also carries n_priced /
    n_skipped (whole-book counts) for the coverage cue. Fast on the dial;
    ``mode='truth'`` for a committed point. Pure, read-only.

    The return also carries ``exposures`` — the priced scope's exposure totals at
    the CURRENT state and at the SHOCKED state, engine-priced on both sides through
    the same shocked-input path (the zero shock IS the current state), so the
    difference between the two columns is pure shock effect, never a live greek
    differenced against a recomputed one. Dollar gamma is reported on both bases
    (engine per-$1, and per-1% via x spot/100 — the exposure section's basis);
    theta is engine per business day."""
    today_ts = _normalize_today(today)
    legs, equities, skips = _select(state, account_state, target, today_ts)
    beta_map = _beta_map(account_state)
    curve = getattr(state, "risk_free_curve", None) or []
    shifted = _shifted_curve(curve, shock.rate_bps)
    nav = _num(getattr(account_state, "nav", None))
    zero = ShockSpec("base", "base")

    rows: list[dict] = []
    total = 0.0
    exp_now = _exposure_acc()
    exp_str = _exposure_acc()
    for lg in legs:
        beta = beta_map.get(lg.underlying_bbg, DEFAULT_BETA)
        S, sigma, r, T, t2 = _shocked_inputs(lg, beta, shock, shifted, today_ts)
        px0 = _price_at(lg, lg.spot, lg.sigma, lg.r, lg.T, lg.today, mode)
        px1 = _price_at(lg, S, sigma, r, T, t2, mode)
        mult = lg.qty * lg.multiplier
        pnl = mult * (px1 - px0)
        total += pnl
        g = _greeks_at(lg, S, sigma, r, T, t2, mode)
        # Current-state greeks through the SAME input path at the zero shock —
        # identical tuples at zero shock, so Now == Stressed exactly there.
        S0, sig0, r0, T0, t0 = _shocked_inputs(lg, beta, zero, curve, today_ts)
        g0 = _greeks_at(lg, S0, sig0, r0, T0, t0, mode)
        _accrue_exposure(exp_now, mult, g0, S0, beta)
        _accrue_exposure(exp_str, mult, g, S, beta)
        rows.append({
            "id": lg.position_id, "label": _leg_label(lg), "kind": "option",
            "underlying": lg.underlying_bbg, "structure_id": _structure_of(account_state, lg.position_id),
            "pnl": pnl, "dd": mult * g.get("delta", 0.0) * S, "dg": mult * g.get("gamma", 0.0) * S,
            "dv": mult * g.get("vega", 0.0), "dt": mult * g.get("theta", 0.0)})
    for eq in equities:
        s1 = max(eq["spot"] * (1.0 + eq["beta"] * shock.spot_pct / 100.0), 0.0)
        pnl = eq["qty"] * (s1 - eq["spot"])
        total += pnl
        _accrue_equity_exposure(exp_now, eq["qty"] * eq["spot"], eq["beta"])
        _accrue_equity_exposure(exp_str, eq["qty"] * s1, eq["beta"])
        rows.append({
            "id": eq["bbg"], "label": eq["bbg"], "kind": "equity", "underlying": eq["bbg"],
            "structure_id": None, "pnl": pnl, "dd": eq["qty"] * s1,
            "dg": 0.0, "dv": 0.0, "dt": 0.0})

    rows.sort(key=lambda r: r["pnl"])
    n_skipped = skips["n_options_skipped"] + skips["n_equities_skipped"]
    return {"account_pnl": total, "account_pnl_pct": (total / nav if nav else None),
            "rows": rows, "n_priced": len(legs) + len(equities), "n_skipped": n_skipped,
            "exposures": {
                "now": exp_now, "stressed": exp_str,
                "n_legs": len(legs) + len(equities), "n_skipped": n_skipped,
                "basis": {
                    "gamma": "dollar gamma per 1% spot move (engine gamma x spot/100); "
                             "dg_native is the engine per-$1 form",
                    "theta": "engine, per business day",
                    "beta": f"{BETA_FIELD} (SPX 2y wkly); missing beta prices at "
                            f"{DEFAULT_BETA:g} (full market participation)",
                    "pricer": ("truth-CRR" if mode == "truth"
                               else "fast vectorized BS2002"),
                }}}


def _exposure_acc() -> dict:
    """One state's exposure totals over the priced scope: dollar delta, beta-$
    (net market exposure), dollar gamma (per-1% and engine-native per-$1),
    dollar vega (per vol pt), dollar theta (per business day)."""
    return {"dd": 0.0, "dbeta": 0.0, "dg_1pct": 0.0, "dg_native": 0.0,
            "dv": 0.0, "dt_bd": 0.0}


def _accrue_exposure(acc: dict, mult, g: dict, S, beta) -> None:
    dd = mult * g.get("delta", 0.0) * S
    acc["dd"] += dd
    acc["dbeta"] += dd * beta
    dg_native = mult * g.get("gamma", 0.0) * S
    acc["dg_native"] += dg_native
    acc["dg_1pct"] += dg_native * S / 100.0
    acc["dv"] += mult * g.get("vega", 0.0)
    acc["dt_bd"] += mult * g.get("theta", 0.0)


def _accrue_equity_exposure(acc: dict, dd, beta) -> None:
    # Stock is delta-one and carries no vol/rate/time sensitivity in this model —
    # its gamma/vega/theta contributions are genuine zeros, not missing data.
    acc["dd"] += dd
    acc["dbeta"] += dd * beta


def spot_vol_grid(state, account_state, *, rate_bps=0.0, time_days=0, target=None,
                  today=None) -> dict:
    """The spot x vol P&L mesh for the heatmap, fast vectorized BS2002 (never truth).
    Cells are P&L vs the *current* (unshocked) state; rate/time shocks shift the
    per-leg r / T / today before the sweep. Pure, read-only."""
    today_ts = _normalize_today(today)
    legs, equities, _skips = _select(state, account_state, target, today_ts)
    beta_map = _beta_map(account_state)
    shifted = _shifted_curve(getattr(state, "risk_free_curve", None) or [], rate_bps)
    spot_axis = np.round(np.linspace(-GRID_SPOT_SPAN, GRID_SPOT_SPAN, GRID_SPOT_N), 4)
    vol_axis = np.array(GRID_VOL_PTS, dtype=float)
    matrix = np.zeros((len(vol_axis), len(spot_axis)))

    # Same business-day roll as _shocked_inputs: a weekend-landing time shock must
    # re-tenor the grid from the SAME shifted date as the impact table, or the two
    # surfaces differ by one business day of decay.
    t2 = _bd(today_ts + pd.Timedelta(days=time_days)) if time_days else None
    for lg in legs:
        beta = beta_map.get(lg.underlying_bbg, DEFAULT_BETA)
        if time_days:
            T = max(year_frac(t2.date(), lg.expiry), _T_FLOOR)
            r = _shocked_rate(lg, shifted, t2, rate_bps)
        else:
            T = lg.T
            r = _shocked_rate(lg, shifted, lg.today, rate_bps)
        px0 = float(strategy.price_leg(lg.spot, lg.K, lg.T, lg.r, lg.q, lg.sigma,
                                       lg.opt_type, style=lg.style, mode="fast"))  # current state
        s_arr = np.maximum(lg.spot * (1.0 + beta * spot_axis / 100.0), _SPOT_FLOOR)
        mult = lg.qty * lg.multiplier
        for vi, vp in enumerate(vol_axis):
            sig = max(lg.sigma + vp / 100.0, SIGMA_FLOOR)
            px = np.asarray(strategy.price_leg(s_arr, lg.K, T, r, lg.q, sig, lg.opt_type,
                                               style=lg.style, mode="fast"), dtype=float)
            matrix[vi, :] += mult * (px - px0)
    for eq in equities:                                   # linear, vol-independent
        s_arr = np.maximum(eq["spot"] * (1.0 + eq["beta"] * spot_axis / 100.0), 0.0)
        matrix += (eq["qty"] * (s_arr - eq["spot"]))[None, :]

    return {"spot_axis": spot_axis.tolist(), "vol_axis": vol_axis.tolist(),
            "pnl_matrix": matrix.tolist(), "pricer": "fast vectorized BS2002"}


# --------------------------------------------------------------------------
# Shared shocked-input / price / greeks core (the gate's "same reprice" guarantee)
# --------------------------------------------------------------------------
def _bd(ts) -> pd.Timestamp:
    """Roll a date to the most recent business day — the truth engine's CRR theta
    step (busday_offset) requires a business-day as-of, and a weekend load date or a
    time-shock landing on a weekend would otherwise raise."""
    return pd.Timestamp(np.busday_offset(pd.Timestamp(ts).date(), 0, roll="backward"))


def _shocked_inputs(leg: EngineLeg, beta, shock: ShockSpec, shifted_curve, today_ts):
    S = max(leg.spot * (1.0 + beta * shock.spot_pct / 100.0), _SPOT_FLOOR)
    sigma = max(leg.sigma + shock.vol_pts / 100.0, SIGMA_FLOOR)
    if shock.time_days:
        t2 = _bd(today_ts + pd.Timedelta(days=shock.time_days))
        T = max(year_frac(t2.date(), leg.expiry), _T_FLOOR)
    else:
        t2 = leg.today
        T = leg.T
    r = _shocked_rate(leg, shifted_curve, t2, shock.rate_bps)
    return S, sigma, r, T, t2


def _price_at(leg, S, sigma, r, T, today, mode, n_steps=None):
    if mode == "truth":
        return _truth_price(leg, S, sigma, r, T, today, n_steps=n_steps)
    return float(strategy.price_leg(S, leg.K, T, r, leg.q, sigma, leg.opt_type,
                                    style=leg.style, mode="fast"))


def _greeks_at(leg, S, sigma, r, T, today, mode):
    engine = strategy.REGISTRY[(leg.style, mode)]
    return engine.greeks(S, leg.K, T, r, leg.q, sigma, leg.opt_type,
                         divs=leg.divs_df, today=_bd(today))


def _account_pnl(legs, equities, beta_map, base_price, curve_pts, today_ts, shock,
                 mode="truth", n_steps=None) -> float:
    shifted = _shifted_curve(curve_pts, shock.rate_bps)
    total = 0.0
    for lg in legs:
        beta = beta_map.get(lg.underlying_bbg, DEFAULT_BETA)
        S, sigma, r, T, t2 = _shocked_inputs(lg, beta, shock, shifted, today_ts)
        px = _price_at(lg, S, sigma, r, T, t2, mode, n_steps=n_steps)
        total += lg.qty * lg.multiplier * (px - base_price[lg.position_id])
    for eq in equities:
        s1 = max(eq["spot"] * (1.0 + eq["beta"] * shock.spot_pct / 100.0), 0.0)
        total += eq["qty"] * (s1 - eq["spot"])
    return total


def _truth_price(leg, S, sigma, r, T, today, n_steps=None):
    n = n_steps or DEFAULT_CRR_STEPS
    if leg.style == "American":
        if leg.divs_df is not None and len(leg.divs_df) > 0:
            return float(american_crr.crr_price(S, leg.K, T, r, sigma, leg.divs_df,
                                                leg.opt_type, today=_bd(today), n_steps=n))
        return float(american_crr.crr_price_continuous_q(S, leg.K, T, r, leg.q, sigma,
                                                         leg.opt_type, n_steps=n))
    return float(european.price(S, leg.K, T, r, leg.q, sigma, leg.opt_type))


def _shocked_rate(leg, shifted_curve, today2, rate_bps) -> float:
    dte = (pd.Timestamp(leg.expiry) - pd.Timestamp(today2)).days
    pick = pick_rate_for_dte(shifted_curve, dte)
    if pick and pick.get("rate") is not None:
        return pick["rate"]
    return leg.r + rate_bps / 10000.0


def _shifted_curve(curve, rate_bps):
    if not rate_bps or not curve:
        return curve
    d = rate_bps / 10000.0
    return [{**pt, "rate": (pt["rate"] + d) if pt.get("rate") is not None else None}
            for pt in curve]


# --------------------------------------------------------------------------
# Target selection (account / position / structure)
# --------------------------------------------------------------------------
def _select(state, account_state, target, today_ts):
    """Priceable option + equity legs for the target scope, plus the full-book
    skip counts (unpriceable options; spot-/qty-less equities). The counts are
    pre-target (whole book) — they describe the book's pricing coverage, which
    the impact/total surfaces disclose regardless of any drill."""
    all_legs = build_engine_legs(state, account_state, today=today_ts)
    legs = [lg for lg in all_legs if lg.priceable]
    equities, n_eq_skipped = _equity_legs_with_skips(account_state, _beta_map(account_state))
    skips = {"n_options_skipped": len(all_legs) - len(legs),
             "n_equities_skipped": n_eq_skipped}
    pids = _target_position_ids(account_state, target)
    if pids is None:
        return legs, equities, skips
    # Equities match on EITHER key: a structure target's pid set holds
    # StructureLeg.position_id values (the bare ticker for stock legs), while the
    # impact table's equity rows drill back by their row id, the bbg ticker.
    return ([lg for lg in legs if lg.position_id in pids],
            [eq for eq in equities if eq["pid"] in pids or eq["bbg"] in pids],
            skips)


def _target_position_ids(account_state, target):
    if not target:
        return None
    if isinstance(target, str):
        return {target}
    kind, tid = target.get("kind"), target.get("id")
    if kind in (None, "account") or tid is None:
        return None
    if kind == "structure":
        for st in getattr(account_state, "structures", []) or []:
            if getattr(st, "structure_id", None) == tid:
                return {getattr(lg, "position_id", None) for lg in getattr(st, "legs", [])}
        return set()
    return {tid}


def _structure_of(account_state, pid):
    for st in getattr(account_state, "structures", []) or []:
        for lg in getattr(st, "legs", []):
            if getattr(lg, "position_id", None) == pid:
                return getattr(st, "structure_id", None)
    return None


def _leg_label(leg) -> str:
    und = (leg.underlying_bbg or "").split(" ")[0] or leg.underlying_bbg or "?"
    k = f"{leg.K:g}" if leg.K is not None else "?"
    exp = leg.expiry.strftime("%b-%y") if leg.expiry else ""
    return f"{und} {leg.opt_type[0] if leg.opt_type else '?'}{k} {exp}".strip()


# --------------------------------------------------------------------------
# v1 P&L curve (kept; the heatmap replaces it in the section)
# --------------------------------------------------------------------------
def _portfolio_curve(legs, equities, beta_map, scenarios, today_ts) -> dict:
    grid = np.linspace(-CURVE_SPAN_PCT, CURVE_SPAN_PCT, CURVE_POINTS)
    pnl = np.zeros_like(grid)
    for lg in legs:
        beta = beta_map.get(lg.underlying_bbg, DEFAULT_BETA)
        s_arr = np.maximum(lg.spot * (1.0 + beta * grid / 100.0), _SPOT_FLOOR)
        p = np.asarray(strategy.price_leg(s_arr, lg.K, lg.T, lg.r, lg.q, lg.sigma,
                                          lg.opt_type, style=lg.style, mode="fast"), dtype=float)
        p0 = float(strategy.price_leg(lg.spot, lg.K, lg.T, lg.r, lg.q, lg.sigma,
                                      lg.opt_type, style=lg.style, mode="fast"))
        pnl += lg.qty * lg.multiplier * (p - p0)
    for eq in equities:
        s_arr = np.maximum(eq["spot"] * (1.0 + eq["beta"] * grid / 100.0), 0.0)
        pnl += eq["qty"] * (s_arr - eq["spot"])

    truth_x, truth_pnl = [], []
    by_name = {s.name: s for s in scenarios}
    for nm in _MARKET_BAND_SHOCKS:
        s = by_name.get(nm)
        if s is not None:
            truth_x.append(s.axes.get("spot_pct", 0.0))
            truth_pnl.append(s.pnl)
    band = _band_halfwidth(grid, pnl, truth_x, truth_pnl)
    return {"x_pct": grid.tolist(), "pnl": pnl.tolist(),
            "band_lo": (pnl - band).tolist(), "band_hi": (pnl + band).tolist(),
            "truth_x": truth_x, "truth_pnl": truth_pnl,
            "breakevens": _zero_crossings(grid, pnl),
            "x_label": "SPX move %", "pricer": "fast vectorized BS2002"}


def _band_halfwidth(grid, pnl_fast, truth_x, truth_pnl) -> np.ndarray:
    if not truth_x:
        return np.zeros_like(grid)
    fast_at = np.interp(truth_x, grid, pnl_fast)
    gaps = np.abs(np.asarray(truth_pnl) - fast_at)
    order = np.argsort(truth_x)
    return np.interp(grid, np.asarray(truth_x)[order], gaps[order])


def _zero_crossings(grid, pnl) -> list:
    out: list = []
    for i in range(1, len(pnl)):
        if (pnl[i - 1] <= 0 <= pnl[i]) or (pnl[i - 1] >= 0 >= pnl[i]):
            if pnl[i] != pnl[i - 1]:
                t = -pnl[i - 1] / (pnl[i] - pnl[i - 1])
                x = float(grid[i - 1] + t * (grid[i] - grid[i - 1]))
                if not out or abs(x - out[-1]) > 1e-6:
                    out.append(x)
    return out


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def _beta_map(account_state) -> dict:
    snap = getattr(getattr(account_state, "snapshot", None), "underlyings", None)
    out: dict = {}
    if snap is None or BETA_FIELD not in getattr(snap, "columns", []):
        return out
    for idx, val in snap[BETA_FIELD].items():
        b = _num(val)
        if b is not None:
            out[idx] = b
    return out


def _equity_legs(account_state, beta_map) -> list:
    out, _ = _equity_legs_with_skips(account_state, beta_map)
    return out


def _equity_legs_with_skips(account_state, beta_map) -> tuple:
    """Equity/fund legs plus the count of ones dropped for missing spot/quantity —
    the drop must be countable so the scenario surfaces can disclose partial
    coverage instead of silently repricing a smaller book."""
    snap = getattr(getattr(account_state, "snapshot", None), "underlyings", None)
    out = []
    n_skipped = 0
    for p in getattr(account_state, "positions", []) or []:
        if getattr(p, "asset_class", None) not in ("equity", "fund_etf"):
            continue
        bbg = getattr(p, "bbg_ticker", None)
        spot = _num(_snap_val(snap, bbg, "PX_LAST"))
        qty = _num(getattr(p, "quantity", None))
        if spot is None or qty is None:
            n_skipped += 1
            continue
        out.append({"bbg": bbg, "pid": getattr(p, "position_id", None),
                    "spot": spot, "qty": qty,
                    "beta": beta_map.get(bbg, DEFAULT_BETA)})
    return out, n_skipped


def _snap_val(snap, idx, col):
    if snap is None or idx is None:
        return None
    try:
        if idx not in snap.index or col not in getattr(snap, "columns", []):
            return None
        v = snap.loc[idx, col]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else None
    return v


def _normalize_today(today) -> pd.Timestamp:
    if today is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(today).normalize()


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None
