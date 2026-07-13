"""Structure / position payoff assembler — combined leg curves, economics, greeks.

The first live consumer of the oracle-validated, previously-unwired combined-payoff
toolkit in :mod:`pm.pricing.payoff_risk`. Pure and read-only — no Bloomberg, no
reload, no ``_RUNTIME`` write-back: the structure-level analogue of
``state_access.price_scenario`` (a hypothetical must never mutate owned state).

It assembles a detected ``Structure`` (or a standalone ``Position``) into the combined
leg list the toolkit consumes — long stock + option legs as ONE position on the
UNDERLYING's own price axis (beta = 1) — honouring each leg's signed ``allocated_qty``
SLICE and deriving every premium / cost from ENTRY ``cost_basis`` (never the current
mark), then orchestrates:

* the at-expiry NET P&L hockey-stick (``payoff_net_at_expiry``),
* the engine-priced HORIZON curve at the (optionally shocked) state — fast BS2002,
  priced per leg so multi-expiry legs keep their own r/q/T,
* breakevens, max profit / loss + capital-at-risk, probability-of-profit — computed
  ANALYTICALLY over the piecewise-linear payoff's kink set
  (:mod:`pm.pricing.payoff_analytic`), never read off a spot grid, so no strike or
  breakeven is ever outside the window; the grid exists only to render the chart,
* greeks now vs under the shock.

Two unit conversions are load-bearing and are the conservation oracle's job to prove (both fall
straight out of the slice algebra — the ``allocated_qty`` cancels):

* stock leg per-share cost basis  = ``position.cost_basis / position.quantity``
  (total-$ cost over total qty),
* option leg entry premium / share = ``position.cost_basis / (position.quantity * 100)``
  (a positive magnitude; the long/short sign is carried by ``qty``).

Consequently ``Σ baked premium across legs == net_debit_credit`` exactly, mark-free —
the primary conservation gate. The Tier-1 slice sums are recomputed here (mirroring
``pm.ui.deepdive.structure_economics``) deliberately, so the risk layer never imports
the UI layer; the test cross-checks them against that canonical function.

Greek basis: per-$1² *position* greeks — delta = ∂($ value)/∂S (share-equivalent,
stock contributes its share count), gamma = ∂delta/∂S, vega per +1 vol point, theta
per business day. Option legs are scaled ×100 (the vectorized kernel returns
per-share-per-contract). This is the engine per-$1² basis — distinct from, and not to
be silently compared with, the exposure section's BBG per-1% gamma.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from pm.pricing import payoff_analytic, payoff_risk
from pm.pricing.conventions import year_frac
from pm.pricing.strategy import avg_iv, price_leg
from pm.risk.pricing_adapter import build_engine_legs

DEFAULT_N_POINTS = 200
DEFAULT_RANGE_PCT = 0.5
_MIN_SIGMA = 0.01

_GREEK_BASIS = (
    "engine per-$1² position greeks: delta = ∂($ value)/∂S (share-equivalent; stock "
    "contributes its share count), gamma = ∂delta/∂S, vega per +1 vol pt, theta per "
    "business day. Distinct from the exposure section's BBG per-1% gamma — do not compare."
)


@dataclass
class PayoffResult:
    account: str
    underlying: str
    structure_id: Optional[str]
    position_id: Optional[str]
    structure_type: Optional[str]
    spot: float
    shocked_spot: Optional[float]
    grid: list
    expiry_curve: list
    horizon_curve: Optional[list]
    expiry_curve_stock: Optional[list]      # stock legs alone (None if no stock leg)
    expiry_curve_options: Optional[list]    # option legs alone (None if no option legs)
    strikes: list
    breakevens: list
    economics: dict
    greeks_now: dict
    greeks_shocked: Optional[dict]
    legs: list
    degraded: bool
    warnings: list
    trace: dict
    # Expiry/role-tagged chart markers, one per option leg (added with the analytic
    # statistics; defaulted so existing constructions stay valid).
    strike_markers: Optional[list] = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f   # NaN -> None
    except (TypeError, ValueError):
        return None


def _frac(allocated_qty, quantity) -> Optional[float]:
    """``allocated_qty / quantity`` (the slice fraction), guarding None/0."""
    q = _num(quantity)
    a = _num(allocated_qty)
    if q is None or q == 0.0 or a is None:
        return None
    return a / q


def _opt_type_of(pos) -> str:
    r = (getattr(pos, "right", None) or getattr(pos, "option_type", None) or "").upper()
    return "Call" if r.startswith("C") else "Put"


def _today_ts(today) -> pd.Timestamp:
    return pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)


def _underlying_spot(account_state, bbg) -> Optional[float]:
    """Underlying PX_LAST from the snapshot (the BBG spot we anchor the curve on)."""
    snap = getattr(account_state, "snapshot", None)
    und = getattr(snap, "underlyings", None)
    if und is None or not bbg:
        return None
    try:
        if bbg in und.index and "PX_LAST" in und.columns:
            return _num(und.loc[bbg, "PX_LAST"])
    except Exception:
        return None
    return None


def _dte(expiries, today_ts) -> Optional[int]:
    if not expiries:
        return None
    try:
        return int((pd.Timestamp(min(expiries)) - today_ts).days)
    except Exception:
        return None


def _normalized_legs(target):
    """Normalise a Structure OR a standalone Position to a common leg list.

    Returns (is_structure, underlying, structure_id, structure_type,
    [(position_id, allocated_qty, role), ...])."""
    legs = getattr(target, "legs", None)
    if legs is not None:                      # a Structure
        norm = [(lg.position_id, lg.allocated_qty, lg.role) for lg in legs]
        return (True, getattr(target, "underlying", None),
                getattr(target, "structure_id", None), getattr(target, "type", None), norm)
    pos = target                              # a standalone Position
    qty = getattr(pos, "quantity", None) or 0.0
    ac = getattr(pos, "asset_class", None)
    if ac == "option":
        role = f"{'long' if qty >= 0 else 'short'}_{_opt_type_of(pos).lower()}"
    elif ac in ("equity", "fund_etf"):
        role = "long_stock" if qty >= 0 else "short_stock"
    else:
        role = ac or "other"
    underlying = getattr(pos, "underlying_symbol", None) or getattr(pos, "symbol", None)
    return (False, underlying, None, None, [(pos.position_id, getattr(pos, "quantity", None), role)])


# ---------------------------------------------------------------------------
# Assembler — Structure/Position -> combined payoff-leg dicts (the keystone)
# ---------------------------------------------------------------------------

def _assemble_legs(by_id, elegs, norm, account_state, today_ts) -> dict:
    """The pure per-leg assembly: slice each leg, synthesise the stock leg, build the
    toolkit-shaped dicts + Tier-1 slice economics. Separated from the engine-leg
    production so it is unit-testable with a hand ``by_id`` + an empty engine-leg map
    (no snapshot needed). ``norm`` is ``[(position_id, allocated_qty, role), ...]``."""
    leg_dicts: list = []
    summaries: list = []
    costs, pnls, prems = [], [], []
    strikes, expiries = [], []
    degraded = False
    warnings: list = []
    spot_candidates: list = []

    for pid, alloc, role in norm:
        pos = by_id.get(pid)
        if pos is None:
            degraded = True
            warnings.append(f"{pid}: position not loaded — leg dropped.")
            costs.append(None); pnls.append(None); prems.append(None)
            continue

        # ---- Tier-1 slice (mirror structure_economics; risk layer stays UI-free) ----
        frac = _frac(alloc, pos.quantity)
        if frac is None:
            degraded = True
            costs.append(None); pnls.append(None); prems.append(None)
        else:
            cb, mv = _num(pos.cost_basis), _num(pos.market_value)
            cost = cb * frac if cb is not None else None
            mval = mv * frac if mv is not None else None
            costs.append(cost)
            pnls.append((mval - cost) if (cost is not None and mval is not None) else None)
            prems.append(cost if pos.asset_class == "option" else 0.0)

        ac = pos.asset_class
        if ac == "option":
            qf, cb = _num(pos.quantity), _num(pos.cost_basis)
            mid = (cb / (qf * 100.0)) if (cb is not None and qf) else 0.0   # ENTRY premium/share
            opt_type = _opt_type_of(pos)
            K = _num(pos.strike)
            expiry = pos.expiry
            eleg = elegs.get(pid)
            sigma = eleg.sigma if eleg else None
            style = (eleg.style if eleg else None) or "American"
            T = (eleg.T if eleg else (year_frac(today_ts, expiry) if expiry else None))
            r = (eleg.r if eleg else None)
            q = (eleg.q if eleg else None)
            priceable = bool(eleg and eleg.priceable and sigma is not None)
            if eleg is not None and _num(eleg.spot):
                spot_candidates.append(float(eleg.spot))
            if K is not None:
                strikes.append(K)
            if expiry is not None:
                expiries.append(expiry)
            if not priceable:
                warnings.append(f"{pid}: option not priceable (no σ) — at-expiry intrinsic only.")
            d = {"opt_type": opt_type, "K": K, "expiry": expiry, "T": T, "sigma": sigma,
                 "style": style, "qty": _num(alloc) or 0.0, "mid": mid, "r": r, "q": q,
                 "priceable": priceable, "position_id": pid, "role": role}
        elif ac in ("equity", "fund_etf"):
            qf, cb = _num(pos.quantity), _num(pos.cost_basis)
            cps = (cb / qf) if (cb is not None and qf) else None        # per-share ENTRY basis
            if cps is None:
                mv = _num(pos.market_value)
                cps = (mv / qf) if (mv is not None and qf) else 0.0
                warnings.append(f"{pid}: stock cost basis unavailable — using current mark.")
            d = {"opt_type": "Stock", "K": None, "expiry": None, "T": None, "sigma": None,
                 "style": None, "qty": _num(alloc) or 0.0, "mid": None, "cost_basis": cps,
                 "r": None, "q": None, "priceable": True, "position_id": pid, "role": role}
            s2 = _underlying_spot(account_state, getattr(pos, "bbg_ticker", None))
            if s2:
                spot_candidates.append(s2)
            else:
                qf2, mv = _num(pos.quantity), _num(pos.market_value)
                if qf2 and mv is not None:
                    spot_candidates.append(mv / qf2)
        else:
            degraded = True
            warnings.append(f"{pid}: asset_class {ac!r} has no payoff — skipped.")
            continue

        leg_dicts.append(d)
        summaries.append({"role": role, "opt_type": d["opt_type"], "K": d.get("K"),
                          "expiry": d.get("expiry"), "qty": d["qty"],
                          "is_stock": d["opt_type"] == "Stock",
                          "priceable": d.get("priceable", True)})

    def _sum(vals):
        return None if any(v is None for v in vals) else float(sum(vals))

    tier1 = {
        "net_debit_credit": _sum(costs),
        "net_pnl": _sum(pnls),
        "net_premium": _sum(prems),
        "strikes": sorted(set(strikes)),
        "expiries": sorted(set(expiries)),
        "degraded": degraded,
    }
    return {"leg_dicts": leg_dicts, "summaries": summaries, "tier1": tier1,
            "spot": (spot_candidates[0] if spot_candidates else None),
            "warnings": warnings, "degraded": degraded}


def build_structure_payoff_legs(state, account_state, target, today=None, elegs=None) -> dict:
    """Assemble a combined leg list for the payoff toolkit, sliced and entry-based.

    Option legs reuse the resolved engine inputs (σ/style/T/r/q via ``EngineLeg``) but
    override ``qty`` -> the structure's ``allocated_qty`` slice and ``mid`` -> the ENTRY
    premium per share (``cost_basis / (quantity*100)``). The long-stock leg — which no
    producer emits — is synthesised ``{opt_type:'Stock', qty:allocated_shares,
    cost_basis:per_share}`` with per-share = ``cost_basis / quantity``.

    Returns a dict with ``leg_dicts`` (toolkit-shaped), ``summaries`` (for the drawer header),
    the Tier-1 slice economics, the anchoring ``spot``, and warnings.
    """
    is_struct, underlying, sid, stype, norm = _normalized_legs(target)
    by_id = {p.position_id: p for p in (getattr(account_state, "positions", None) or [])}
    if elegs is None:
        elegs = {e.position_id: e for e in build_engine_legs(state, account_state, today=today)}
    asm = _assemble_legs(by_id, elegs, norm, account_state, _today_ts(today))
    asm.update({"underlying": underlying, "structure_id": sid, "structure_type": stype,
                "is_structure": is_struct})
    return asm


# ---------------------------------------------------------------------------
# Assembly driver — combined leg dicts -> curves + economics + greeks (pure)
# ---------------------------------------------------------------------------

def _chart_grid(spot, kink_spots, breakevens, shocked_spot, *, n_points, range_pct):
    """The CHART-rendering grid only — the statistics are analytic and never read it.

    The default ±``range_pct`` window is widened to cover every strike and breakeven
    (10%/15% padding so a plateau renders visibly flat), floored at 0, and the exact
    kink spots (strikes, breakevens, spot, shocked spot) are injected so the plotted
    hockey-stick's vertices are exact. Length is ``n_points`` plus the few inserts."""
    lo = spot * (1.0 - range_pct)
    hi = spot * (1.0 + range_pct)
    marks = [float(k) for k in kink_spots if k and k > 0.0] + [float(b) for b in breakevens]
    if marks:
        lo = min(lo, 0.90 * min(marks))
        hi = max(hi, 1.10 * max(marks))
    lo = max(lo, 0.0)
    pts = np.linspace(lo, hi, int(n_points))
    inserts = [x for x in ([spot, shocked_spot] + marks)
               if x is not None and lo <= x <= hi]
    if inserts:
        return np.unique(np.concatenate([pts, np.asarray(inserts, dtype=float)]))
    return pts


def _bound_str(region) -> Optional[str]:
    """Human-readable attainment region for a bounded extremum: 'S = 103.00',
    'S ≥ 110.00' (plateau to +inf), 'S ≤ 100.00' (plateau from 0), 'S = 0',
    or 'S ∈ [100.00, 110.00]' (interior plateau). None for unbounded."""
    if region is None:
        return None
    lo, hi = region
    if hi == float("inf"):
        return f"S ≥ {lo:,.2f}"
    if lo == hi:
        return "S = 0" if lo == 0.0 else f"S = {lo:,.2f}"
    if lo == 0.0:
        return f"S ≤ {hi:,.2f}"
    return f"S ∈ [{lo:,.2f}, {hi:,.2f}]"


def _iso(d) -> Optional[str]:
    try:
        return None if d is None else pd.Timestamp(d).date().isoformat()
    except Exception:
        return None


def _strike_markers(leg_dicts) -> list:
    """One expiry/role-tagged marker per option leg, for the chart's strike lines."""
    out = []
    for l in leg_dicts:
        if l.get("opt_type") in ("Call", "Put") and l.get("K") is not None:
            out.append({"K": float(l["K"]), "expiry": _iso(l.get("expiry")),
                        "role": l.get("role"), "qty": l.get("qty"),
                        "opt_type": l["opt_type"]})
    return sorted(out, key=lambda m: (m["K"], m["expiry"] or ""))


def _horizon_curve(leg_dicts, grid, shocked_today, dvol, dr):
    """Engine-priced P&L *value* over the grid at the (shocked) state — options priced
    per leg (fast BS2002) so each keeps its own r/q/T, plus the linear stock term. The
    caller subtracts net_debit_credit to turn value into P&L."""
    grid = np.asarray(grid, dtype=float)
    total = np.zeros_like(grid)
    for l in leg_dicts:
        if l["opt_type"] == "Stock":
            total = total + l["qty"] * grid
            continue
        K = l.get("K")
        if K is None:
            continue
        if not l.get("priceable") or l.get("sigma") is None:
            intr = (np.maximum(grid - K, 0.0) if l["opt_type"] == "Call"
                    else np.maximum(K - grid, 0.0))
            total = total + l["qty"] * intr * 100.0
            continue
        T_h = year_frac(shocked_today, l["expiry"])
        if T_h <= 0:
            px = (np.maximum(grid - K, 0.0) if l["opt_type"] == "Call"
                  else np.maximum(K - grid, 0.0))
        else:
            sigma_h = max(float(l["sigma"]) + dvol, _MIN_SIGMA)
            r_h = (l["r"] if l.get("r") is not None else 0.04) + dr
            q_h = l["q"] if l.get("q") is not None else 0.0
            px = price_leg(grid, K, T_h, r_h, q_h, sigma_h, l["opt_type"],
                           style=l.get("style") or "American", mode="fast")
        total = total + l["qty"] * np.asarray(px, dtype=float) * 100.0
    return total


def _greeks(leg_dicts, spot, today_ts, dvol, r, q) -> Optional[dict]:
    """Position greeks at one spot. Option legs ×100 (the kernel returns
    per-share-per-contract); stock adds its share count to delta only."""
    if spot is None:
        return None
    opts = [l for l in leg_dicts
            if l["opt_type"] != "Stock" and l.get("priceable") and l.get("sigma") is not None]
    stock_shares = sum(l["qty"] for l in leg_dicts if l["opt_type"] == "Stock")
    if not opts:
        return {"delta": float(stock_shares), "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    legs_g = [{"opt_type": l["opt_type"], "K": l["K"], "expiry": l["expiry"],
               "sigma": max(float(l["sigma"]) + dvol, _MIN_SIGMA), "qty": l["qty"],
               "style": l.get("style") or "American"} for l in opts]
    og = payoff_risk.strategy_greeks_vectorized(
        np.array([spot], dtype=float), legs_g, r, q, today=today_ts)
    return {
        "delta": 100.0 * float(og["delta"][0]) + float(stock_shares),
        "gamma": 100.0 * float(og["gamma"][0]),
        "vega": 100.0 * float(og["vega"][0]),
        "theta": 100.0 * float(og["theta"][0]),
    }


def compute_payoff(leg_dicts, spot, tier1, *, shock=None, n_points=DEFAULT_N_POINTS,
                   range_pct=DEFAULT_RANGE_PCT, today=None) -> dict:
    """Pure orchestration over assembled leg dicts (no state) — testable with synthetic
    structures. Returns the curves, markers, economics, greeks, and a trace carrying the
    conservation cross-check (baked premium vs net_debit_credit)."""
    today_ts = _today_ts(today)
    spot = float(spot)
    warnings: list = []

    # ---- at-expiry statistics: analytic over the PWL kink set (grid-free, so no
    # strike/breakeven/plateau is ever outside a window) ----
    pwl = payoff_analytic.pwl_from_legs(leg_dicts)
    breakevens = payoff_analytic.pwl_breakevens(pwl)
    maxpl = payoff_analytic.pwl_max_profit_loss(pwl)
    intervals = payoff_analytic.profit_intervals(pwl)

    # ---- shock parse (up front: the shocked spot is a chart-grid insert) ----
    sp = shock or {}
    spot_pct = float(sp.get("spot_pct", 0.0))
    dvol = float(sp.get("vol_pts", 0.0)) / 100.0
    dr = float(sp.get("rate_bps", 0.0)) / 1e4
    dt_days = int(sp.get("time_days", 0))
    shocked_today = today_ts + pd.Timedelta(days=dt_days)
    shocked_spot = (spot * (1.0 + spot_pct / 100.0)) if shock is not None else None

    # ---- chart grid (rendering substrate only — statistics never read it) ----
    grid = _chart_grid(spot, pwl.kinks[1:], breakevens, shocked_spot,
                       n_points=n_points, range_pct=range_pct)
    expiry_curve = payoff_risk.payoff_net_at_expiry(leg_dicts, grid)

    # Component (standalone-vs-net) curves: the same at-expiry kernel over leg subsets.
    # Linear in legs, so combined == stock-alone + options-alone at every grid point
    # (the conservation oracle). None when that subset is empty.
    stock_legs = [l for l in leg_dicts if l["opt_type"] == "Stock"]
    option_legs = [l for l in leg_dicts if l["opt_type"] != "Stock"]
    expiry_curve_stock = (payoff_risk.payoff_net_at_expiry(stock_legs, grid)
                          if stock_legs else None)
    expiry_curve_options = (payoff_risk.payoff_net_at_expiry(option_legs, grid)
                            if option_legs else None)

    # Baked premium — the constant the at-expiry NET curve subtracts. The conservation
    # identity is: baked == net_debit_credit (mark-free; proves slice + per-share basis).
    baked = 0.0
    for l in leg_dicts:
        if l["opt_type"] == "Stock":
            baked += l["qty"] * (l.get("cost_basis") or 0.0)
        else:
            baked += l["qty"] * (l.get("mid") or 0.0) * 100.0

    # Representative σ/T/r/q for the single-distribution PoP (nearest expiry, |qty|-IV).
    opts = [l for l in leg_dicts
            if l["opt_type"] != "Stock" and l.get("priceable") and l.get("sigma") is not None]
    expiry_set = {round(l["T"], 4) for l in opts if l.get("T") is not None}
    multi_expiry = len(expiry_set) > 1
    if opts:
        nearest = min(opts, key=lambda l: l["T"] if l.get("T") is not None else 1e9)
        r_repr = _num(nearest.get("r"))
        q_repr = _num(nearest.get("q")) or 0.0
        T_repr = nearest.get("T")
        sigma_repr = avg_iv([{"opt_type": l["opt_type"], "sigma": l["sigma"], "qty": l["qty"]}
                             for l in opts])
    else:
        r_repr, q_repr, T_repr, sigma_repr = None, 0.0, None, float("nan")
    if r_repr is None:
        r_repr = 0.04

    pop, pop_caveat = None, None
    if opts and T_repr and T_repr > 0 and sigma_repr == sigma_repr and sigma_repr > 0:
        val = payoff_analytic.pop_lognormal_intervals(
            spot, sigma_repr, T_repr, r_repr, q_repr, intervals)
        pop = None if val != val else float(val)
        if multi_expiry and pop is not None:
            pop_caveat = ("multi-expiry: PoP uses the nearest expiry + |qty|-weighted IV "
                          "(single-σ/T approximation).")

    # Horizon P&L = engine value at the (shocked) state − entry cost (net_debit_credit).
    nd = tier1.get("net_debit_credit")
    horizon_value = _horizon_curve(leg_dicts, grid, shocked_today, dvol, dr)
    if nd is not None:
        horizon_curve = horizon_value - nd
    else:
        horizon_curve = None
        warnings.append("net debit/credit unavailable (degraded slice) — horizon P&L not anchored.")

    greeks_now = _greeks(leg_dicts, spot, today_ts, 0.0, r_repr, q_repr)
    greeks_shocked = (_greeks(leg_dicts, shocked_spot, shocked_today, dvol, r_repr + dr, q_repr)
                      if shock is not None else None)

    expiries = tier1.get("expiries") or []
    eval_date = _iso(min(expiries)) if expiries else None
    economics = {
        "max_profit": maxpl["max_profit"],
        "max_loss": maxpl["max_loss"],
        "capital_at_risk": (abs(maxpl["max_loss"]) if maxpl["max_loss"] is not None else None),
        "unbounded_gain": maxpl["unbounded_gain"],
        "unbounded_loss": maxpl["unbounded_loss"],
        "pop": pop,
        "pop_caveat": pop_caveat,
        "net_premium": tier1.get("net_premium"),
        "net_debit_credit": nd,
        "current_pnl": tier1.get("net_pnl"),
        "dte": _dte(tier1.get("expiries"), today_ts),
        # Analytic-statistics additions (all additive to the original contract):
        "always_profitable": maxpl["always_profitable"],
        "always_loss": maxpl["always_loss"],
        "max_profit_bound": _bound_str(maxpl.get("max_profit_region")),
        "max_loss_bound": _bound_str(maxpl.get("max_loss_region")),
        "eval_mode": "expiry",
        "eval_date": eval_date,
        "econ_caveat": None,
        "n_expiries": len(expiries),
    }
    conservation_ok = (nd is not None and abs(baked - nd) <= 1e-6 * max(1.0, abs(nd)))
    trace = {
        "baked_premium": baked,
        "net_debit_credit": nd,
        "conservation_ok": conservation_ok,
        "spot": spot,
        "sigma_repr": (None if sigma_repr != sigma_repr else sigma_repr),
        "T_repr": T_repr, "r_repr": r_repr, "q_repr": q_repr,
        "multi_expiry": multi_expiry,
        "greek_basis": _GREEK_BASIS,
        "evaluation": {"mode": economics["eval_mode"], "eval_date": eval_date,
                       "far_leg_count": 0},
        "statistics": ("analytic: kink-exact over {0} ∪ strikes ∪ tail slopes "
                       "(no grid window); grid is chart-rendering only."),
        "pricer": ("fast BS2002 (horizon sweep + greeks); at-expiry intrinsic; "
                   "truth-CRR reserved for committed points."),
    }
    return {
        "grid": grid, "expiry_curve": expiry_curve, "horizon_curve": horizon_curve,
        "expiry_curve_stock": expiry_curve_stock, "expiry_curve_options": expiry_curve_options,
        "breakevens": breakevens, "strikes": tier1.get("strikes") or [],
        "strike_markers": _strike_markers(leg_dicts),
        "economics": economics, "greeks_now": greeks_now, "greeks_shocked": greeks_shocked,
        "shocked_spot": shocked_spot, "spot": spot, "warnings": warnings, "trace": trace,
    }


# ---------------------------------------------------------------------------
# Public entry — the structure-level read-only recompute (analog of price_scenario)
# ---------------------------------------------------------------------------

def structure_payoff(state, account_state, target, *, shock=None,
                     n_points=DEFAULT_N_POINTS, range_pct=DEFAULT_RANGE_PCT,
                     today=None, elegs=None) -> Optional[PayoffResult]:
    """Assemble ``target`` (a Structure or a standalone Position) and compute its payoff
    panel. Read-only: no Bloomberg, no reload, no state write-back. Returns None when no
    priceable leg / spot is available. ``elegs`` (the pre-built engine-leg map) lets a
    load-path pass over many structures build the legs ONCE instead of per call."""
    asm = build_structure_payoff_legs(state, account_state, target, today=today, elegs=elegs)
    leg_dicts, spot = asm["leg_dicts"], asm["spot"]
    if not leg_dicts or spot is None or not (spot > 0):
        return None
    res = compute_payoff(leg_dicts, spot, asm["tier1"], shock=shock,
                         n_points=n_points, range_pct=range_pct, today=today)

    def _flist(v):
        return None if v is None else [float(x) for x in v]

    return PayoffResult(
        account=getattr(account_state, "account", ""),
        underlying=asm["underlying"] or "",
        structure_id=asm["structure_id"],
        position_id=(None if asm["is_structure"] else leg_dicts[0]["position_id"]),
        structure_type=asm["structure_type"],
        spot=res["spot"], shocked_spot=res["shocked_spot"],
        grid=[float(x) for x in res["grid"]],
        expiry_curve=[float(x) for x in res["expiry_curve"]],
        horizon_curve=_flist(res["horizon_curve"]),
        expiry_curve_stock=_flist(res["expiry_curve_stock"]),
        expiry_curve_options=_flist(res["expiry_curve_options"]),
        strikes=res["strikes"], breakevens=res["breakevens"],
        economics=res["economics"], greeks_now=res["greeks_now"],
        greeks_shocked=res["greeks_shocked"], legs=asm["summaries"],
        degraded=asm["degraded"], warnings=asm["warnings"] + res["warnings"], trace=res["trace"],
        strike_markers=res.get("strike_markers"),
    )


def run_structure_tier2(state, today=None) -> None:
    """Load-path pass: per account, build the option engine legs ONCE, then
    compute each structure's zero-shock payoff and store its Tier-2 economics on
    ``account_state.structure_tier2[structure_id]`` — the By-Structure grid reads the
    breakeven(s) from it (killing the 'pending pricing' stub). Pure / read-only; runs
    after ``run_account_scenario``. A structure with no priceable leg / spot is skipped
    (its grid cell falls back to '—'). ``today`` (default: wall clock, as the live load
    intends) threads through the engine legs and the payoff so the pass is
    deterministic under test."""
    for acc in state.accounts.values():
        tier2: dict = {}
        try:
            elegs = {e.position_id: e for e in build_engine_legs(state, acc, today=today)}
        except Exception:
            elegs = {}
        for s in (getattr(acc, "structures", None) or []):
            if getattr(s, "status", None) == "rejected":
                continue
            try:
                res = structure_payoff(state, acc, s, elegs=elegs, today=today)
            except Exception:
                res = None
            if res is None:
                continue
            tier2[s.structure_id] = {
                "breakevens": list(res.breakevens or []),
                "max_profit": res.economics.get("max_profit"),
                "max_loss": res.economics.get("max_loss"),
                "pop": res.economics.get("pop"),
                "multi_expiry": bool(res.trace.get("multi_expiry")),
                "eval_date": res.economics.get("eval_date"),
                "always_profitable": bool(res.economics.get("always_profitable")),
                "always_loss": bool(res.economics.get("always_loss")),
            }
        acc.structure_tier2 = tier2
