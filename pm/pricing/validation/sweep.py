"""BS2002-vs-CRR consistency sweep (two-tier).

Grades the BS2002 closed-form American approximation against the convergent CRR-500
lattice (the truth oracle) across two parameter tiers. The tolerances ABSORB the
inherent accuracy limit of the two-step boundary approximation while still catching
structural bugs (which would be orders of magnitude larger than any approximation
error):

  - The paper's own tested grid (sigma <= 40%) reports a maximum of about 30 bps.
    Extending to sigma <= 60% (production-realistic for high-vol single names) yields
    up to ~150-200 bps in deep-ITM corners; at sigma >= 80% errors approach 500 bps.
    These are documented properties of the boundary, not implementation bugs.
  - A point fails only if BOTH the absolute (> 1 cent) AND relative (> tolerance bps)
    thresholds are exceeded. The absolute floor prevents thin-OTM amplification (a
    sub-cent disagreement reads as a large relative error but is just lattice noise).
"""
from pm.pricing.american_bs2002 import bs2002_price
from pm.pricing.american_crr import DEFAULT_CRR_STEPS, crr_price_continuous_q

TOL_TIER1_BPS = 200.0   # production-realistic (HARD assert, relative)
TOL_TIER2_BPS = 500.0   # extended grid (warn only, relative)
TOL_ABS_CENTS = 0.01    # 1 cent absolute floor (any tier, compound)

# Descriptive: the region where BS2002 is most reliable (not used as gate bounds).
TIER1_VOL_MAX = 0.60
TIER1_SK_MIN = 0.70
TIER1_SK_MAX = 1.40
TIER1_T_MAX = 1.0       # years


def run_sweep(verbose=True):
    """Validate BS2002 vs CRR-500 across two tiers.

    Tier 1 (production-realistic, HARD assertion): sigma <= 0.60, sk in [0.70, 1.40],
        T in [3d, 1y]. Tolerance: < 200 bps relative (compound with the 1c floor).
    Tier 2 (extended, WARN only): sigma <= 1.00, sk in [0.50, 1.50], T in [7d, 2y].
        Tolerance: < 500 bps relative.

    Returns (tier1_max_rel_bps, tier1_max_abs, tier2_max_rel_bps, tier2_max_abs).
    Raises AssertionError if Tier 1 fails.
    """
    def _crr_truth(S, K, T, r, q, v, opt):
        return crr_price_continuous_q(S, K, T, r, q, v, opt, n_steps=DEFAULT_CRR_STEPS)

    def _run(sk_grid, T_grid, r_grid, q_grid, v_grid, tol_rel_bps):
        worst_rel_bps = 0.0
        worst_rel_case = None
        worst_abs = 0.0
        failures = []
        n_pts = 0
        for sk in sk_grid:
            for T in T_grid:
                for r in r_grid:
                    for q in q_grid:
                        for v in v_grid:
                            for opt in ['Call', 'Put']:
                                S = 100.0 * sk
                                bs = float(bs2002_price(S, 100.0, T, r, q, v, opt))
                                crr_v = _crr_truth(S, 100.0, T, r, q, v, opt)
                                if crr_v <= 0.05:
                                    continue  # skip near-zero refs (CRR can't resolve)
                                abs_err = abs(bs - crr_v)
                                rel_bps = abs_err / crr_v * 1e4
                                n_pts += 1
                                if rel_bps > worst_rel_bps:
                                    worst_rel_bps = rel_bps
                                    worst_rel_case = (S, T * 365, r, q, v, opt, bs, crr_v)
                                if abs_err > worst_abs:
                                    worst_abs = abs_err
                                # Compound failure: BOTH thresholds exceeded.
                                if abs_err > TOL_ABS_CENTS and rel_bps > tol_rel_bps:
                                    failures.append(
                                        (S, T * 365, r, q, v, opt, bs, crr_v, abs_err, rel_bps))
        return worst_rel_bps, worst_rel_case, worst_abs, failures, n_pts

    t1_rel, t1_case, t1_abs, t1_fails, t1_n = _run(
        sk_grid=[0.70, 0.85, 1.00, 1.15, 1.30, 1.40],
        T_grid=[3 / 365, 14 / 365, 30 / 365, 90 / 365, 180 / 365, 365 / 365],
        r_grid=[0.0, 0.025, 0.05, 0.10],
        q_grid=[0.0, 0.02, 0.05],
        v_grid=[0.10, 0.20, 0.30, 0.40, 0.60],
        tol_rel_bps=TOL_TIER1_BPS,
    )
    t2_rel, t2_case, t2_abs, t2_fails, t2_n = _run(
        sk_grid=[0.50, 0.75, 1.00, 1.25, 1.50],
        T_grid=[7 / 365, 30 / 365, 90 / 365, 180 / 365, 365 / 365, 730 / 365],
        r_grid=[0.0, 0.025, 0.05, 0.10],
        q_grid=[0.0, 0.02, 0.05],
        v_grid=[0.10, 0.25, 0.50, 0.80, 1.00],
        tol_rel_bps=TOL_TIER2_BPS,
    )

    if verbose:
        print(f'[BS2002 sweep] Tier 1 ({t1_n} pts): worst rel {t1_rel:.1f} bps, '
              f'worst abs {t1_abs*100:.2f}c, compound failures {len(t1_fails)}  '
              f'{"PASS" if not t1_fails else "FAIL"}')
        print(f'[BS2002 sweep] Tier 2 ({t2_n} pts): worst rel {t2_rel:.1f} bps, '
              f'worst abs {t2_abs*100:.2f}c, compound failures {len(t2_fails)}  '
              f'{"PASS" if not t2_fails else "WARN"}')

    assert not t1_fails, (
        f'BS2002 Tier-1 compound sweep failed at {len(t1_fails)} points '
        f'(both abs > {TOL_ABS_CENTS*100:.0f}c AND rel > {TOL_TIER1_BPS:.0f}bps). '
        f'First failure: {t1_fails[0]}')
    return t1_rel, t1_abs, t2_rel, t2_abs


if __name__ == "__main__":
    run_sweep()
