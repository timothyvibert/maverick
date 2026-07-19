"""Published / oracle-free validation targets for the pricing engines.

  - Hull Example 21.1 American put: CRR-500 reproduces the published 4.283021.
  - CRR convergence: monotone in n toward the true American value.
  - European put-call parity: machine-precision identity (the BS engine is exact).
  - American (BS2002) >= European (BS) across a grid.

  - BS2002 paper tables: every two-step-boundary value published in
    Bjerksund-Stensland (2002) Tables 1-4 (October 21, 2002 version, the one
    with the corrected Table 2), 135 cells, reproduced to the print precision
    (+/-0.005; measured worst |diff| 0.0050). Parameters map via q = r - b.
    One transcription note: the paper's page-7 text fixes sigma = 0.20 for ALL
    of Table 4 (the panels vary only the cost of carry b) - the third panel is
    encoded accordingly.
"""
import math

from pm.pricing.american_bs2002 import bs2002_price
from pm.pricing.american_crr import crr_price_continuous_q
from pm.pricing.european import bs_price

HULL_21_1_TARGET = 4.283021
HULL_21_1_TOL = 0.005


def hull_21_1():
    """Hull Example 21.1 American put (S=K=50, T=5/12, r=10%, q=0, sigma=40%)."""
    value = crr_price_continuous_q(50.0, 50.0, 5.0 / 12.0, 0.10, 0.0, 0.40,
                                   "Put", n_steps=500)
    return {"value": value, "target": HULL_21_1_TARGET,
            "ok": abs(value - HULL_21_1_TARGET) <= HULL_21_1_TOL}


def crr_convergence(ns=(50, 100, 200, 500, 1000, 2000)):
    """CRR-continuous-q ladder for a standard ATM American put; monotone in n."""
    vals = [crr_price_continuous_q(100.0, 100.0, 1.0, 0.05, 0.0, 0.30, "Put", n_steps=n)
            for n in ns]
    monotone = all(b >= a for a, b in zip(vals, vals[1:]))
    return {"ns": list(ns), "values": vals, "monotone": monotone,
            "tail_gap": abs(vals[-1] - vals[-2])}


def put_call_parity(S=100.0, K=95.0, T=0.5, r=0.04, q=0.02, sigma=0.25):
    """European put-call parity on S_eff: C - P == S_eff - K e^{-rT}."""
    S_eff = S * math.exp(-q * T)
    c = float(bs_price(S_eff, K, T, r, sigma, "Call"))
    p = float(bs_price(S_eff, K, T, r, sigma, "Put"))
    resid = (c - p) - (S_eff - K * math.exp(-r * T))
    return {"call": c, "put": p, "residual": resid, "ok": abs(resid) < 1e-10}


def american_ge_european():
    """American (BS2002) >= European (BS) across a grid; count violations."""
    viol = 0
    n = 0
    for sk in (0.8, 0.9, 1.0, 1.1, 1.2):
        for T in (0.1, 0.5, 1.0):
            for v in (0.15, 0.3, 0.5):
                for opt in ("Call", "Put"):
                    S = 100.0 * sk
                    am = float(bs2002_price(S, 100.0, T, 0.06, 0.03, v, opt))
                    eu = float(bs_price(S * math.exp(-0.03 * T), 100.0, T, 0.06, v, opt))
                    n += 1
                    if am < eu - 1e-6:
                        viol += 1
    return {"points": n, "violations": viol, "ok": viol == 0}


# Bjerksund-Stensland (2002) Tables 1-4, two-step boundary column, K = 100.
# Panel key: (r, b, sigma, T); cells: {S: value} per side. q = r - b.
BS2002_PAPER_TOL = 0.0055   # 2dp print rounding + float wiggle
BS2002_PAPER_PANELS = [
    # Table 1 (b = -0.04)
    (0.08, -0.04, 0.20, 0.25,
     {80: 0.03, 90: 0.58, 100: 3.51, 110: 10.34, 120: 20.00},
     {80: 20.41, 90: 11.25, 100: 4.40, 110: 1.12, 120: 0.18}),
    (0.12, -0.04, 0.20, 0.25,
     {80: 0.03, 90: 0.57, 100: 3.49, 110: 10.31, 120: 20.00},
     {80: 20.23, 90: 11.14, 100: 4.35, 110: 1.11, 120: 0.18}),
    (0.08, -0.04, 0.40, 0.25,
     {80: 1.05, 90: 3.26, 100: 7.39, 110: 13.51, 120: 21.26},
     {80: 21.44, 90: 13.91, 100: 8.27, 110: 4.52, 120: 2.29}),
    (0.08, -0.04, 0.20, 0.5,
     {80: 0.21, 90: 1.35, 100: 4.69, 110: 10.98, 120: 20.00},
     {80: 20.96, 90: 12.63, 100: 6.37, 110: 2.65, 120: 0.92}),
    # Table 2 (b = +0.04)
    (0.08, 0.04, 0.20, 0.25,
     {80: 0.05, 90: 0.85, 100: 4.44, 110: 11.66, 120: 20.90},
     {80: 20.00, 90: 10.21, 100: 3.53, 110: 0.79, 120: 0.11}),
    (0.12, 0.04, 0.20, 0.25,
     {80: 0.05, 90: 0.84, 100: 4.40, 110: 11.55, 120: 20.69},
     {80: 20.00, 90: 10.19, 100: 3.51, 110: 0.78, 120: 0.11}),
    (0.08, 0.04, 0.40, 0.25,
     {80: 1.29, 90: 3.82, 100: 8.35, 110: 14.80, 120: 22.71},
     {80: 20.55, 90: 12.94, 100: 7.45, 110: 3.94, 120: 1.94}),
    (0.08, 0.04, 0.20, 0.5,
     {80: 0.41, 90: 2.18, 100: 6.50, 110: 13.42, 120: 22.06},
     {80: 20.00, 90: 10.73, 100: 4.74, 110: 1.72, 120: 0.52}),
    # Table 3 (b = r, non-dividend stock: puts only)
    (0.08, 0.08, 0.20, 0.25, None,
     {80: 20.00, 90: 10.02, 100: 3.20, 110: 0.66, 120: 0.09}),
    (0.12, 0.12, 0.20, 0.25, None,
     {80: 20.00, 90: 10.00, 100: 2.90, 110: 0.55, 120: 0.07}),
    (0.08, 0.08, 0.40, 0.25, None,
     {80: 20.30, 90: 12.54, 100: 7.09, 110: 3.69, 120: 1.78}),
    (0.08, 0.08, 0.20, 0.5, None,
     {80: 20.00, 90: 10.27, 100: 4.15, 110: 1.39, 120: 0.39}),
    # Table 4 (r = 0.08, sigma = 0.20 throughout, T = 3; b varies per panel)
    (0.08, -0.04, 0.20, 3.0,
     {80: 2.32, 90: 4.74, 100: 8.47, 110: 13.77, 120: 20.86},
     {80: 25.64, 90: 20.07, 100: 15.49, 110: 11.80, 120: 8.88}),
    (0.08, 0.00, 0.20, 3.0,
     {80: 3.97, 90: 7.23, 100: 11.68, 110: 17.28, 120: 23.95},
     {80: 22.14, 90: 16.17, 100: 11.68, 110: 8.35, 120: 5.91}),
    (0.08, 0.04, 0.20, 3.0,
     {80: 6.88, 90: 11.49, 100: 17.21, 110: 23.84, 120: 31.16},
     {80: 20.33, 90: 13.47, 100: 8.91, 110: 5.88, 120: 3.87}),
    (0.08, 0.08, 0.20, 3.0, None,
     {80: 20.00, 90: 11.68, 100: 6.91, 110: 4.13, 120: 2.49}),
]


def bs2002_paper_tables(tol=BS2002_PAPER_TOL):
    """Engine BS2002 vs every published two-step value (Tables 1-4). Returns
    per-cell failures (empty when all reproduce to print precision)."""
    fails = []
    n = 0
    worst = 0.0
    for r, b, sigma, T, calls, puts in BS2002_PAPER_PANELS:
        q = r - b
        for opt_type, cells in (("Call", calls), ("Put", puts)):
            for S, target in (cells or {}).items():
                got = float(bs2002_price(float(S), 100.0, T, r, q, sigma, opt_type))
                d = abs(got - target)
                n += 1
                worst = max(worst, d)
                if d > tol:
                    fails.append({"r": r, "b": b, "sigma": sigma, "T": T,
                                  "S": S, "opt_type": opt_type,
                                  "target": target, "value": got, "diff": d})
    return {"cells": n, "worst": worst, "failures": fails, "ok": not fails}


def run_all(verbose=True):
    results = {
        "hull_21_1": hull_21_1(),
        "crr_convergence": crr_convergence(),
        "put_call_parity": put_call_parity(),
        "american_ge_european": american_ge_european(),
        "bs2002_paper_tables": bs2002_paper_tables(),
    }
    if verbose:
        h = results["hull_21_1"]
        print(f"Hull 21.1 American put = {h['value']:.6f} (target {h['target']}) "
              f"-> {'PASS' if h['ok'] else 'FAIL'}")
        cc = results["crr_convergence"]
        print(f"CRR convergence monotone={cc['monotone']} tail_gap={cc['tail_gap']:.2e}")
        pp = results["put_call_parity"]
        print(f"Put-call parity residual={pp['residual']:.2e} "
              f"-> {'PASS' if pp['ok'] else 'FAIL'}")
        ae = results["american_ge_european"]
        print(f"American>=European violations={ae['violations']}/{ae['points']} "
              f"-> {'PASS' if ae['ok'] else 'FAIL'}")
        bs = results["bs2002_paper_tables"]
        print(f"BS2002 paper tables {bs['cells']} cells worst={bs['worst']:.4f} "
              f"-> {'PASS' if bs['ok'] else 'FAIL'}")
    return results


if __name__ == "__main__":
    run_all()
