"""Published / oracle-free validation targets for the pricing engines.

  - Hull Example 21.1 American put: CRR-500 reproduces the published 4.283021.
  - CRR convergence: monotone in n toward the true American value.
  - European put-call parity: machine-precision identity (the BS engine is exact).
  - American (BS2002) >= European (BS) across a grid.

The BS2002 paper-table fixtures (Bjerksund-Stensland 2002, Tables 1-2, +/-0.005)
are a follow-on: the original single-point cell that paired Hull 4.283021 with a
BS2002 value was not preserved with this code, so those parameters must be recovered
from the paper before they are pinned as encoded fixtures. The BS2002 path is covered
in the meantime by the consistency sweep (validation.sweep) and the checks here.
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


def run_all(verbose=True):
    results = {
        "hull_21_1": hull_21_1(),
        "crr_convergence": crr_convergence(),
        "put_call_parity": put_call_parity(),
        "american_ge_european": american_ge_european(),
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
    return results


if __name__ == "__main__":
    run_all()
