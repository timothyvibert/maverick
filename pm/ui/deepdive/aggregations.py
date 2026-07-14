"""Pure presentation aggregations for Tab 2 — Account Deep Dive.

These are NOT engine computations: they read fields already computed on
``AccountState`` (``.greeks.totals``, ``.positions``) and reframe them for the
analytics panel. No Bloomberg calls, no signal/pattern recomputation — fully
consistent with the no-recompute-in-UI invariant (they aggregate the same way
``consolidate_fires_to_rows`` aggregates ``% NAV`` from positions).

No Dash imports here, so every function is unit-testable without a browser.
Each takes an ``account_state`` (anything exposing the attributes it reads, so
tests can pass a lightweight stand-in) and returns plain dicts/lists.

Decisions (confirmed with the desk):
- Premium split uses **entry premium** (``cost_basis``), not current mark — the
  truest "collected vs paid" income-posture framing. Falls back to
  ``market_value`` when ``cost_basis`` is missing.
- Expiry-ladder notional is **strike notional** (``|qty| × multiplier × strike``)
  — the strike obligation, the meaningful exposure for an options book. Labeled
  "Notional (strike)" in the UI so the number is never ambiguous.
"""
from __future__ import annotations

from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Compact money / direction formatting (used inside the interpretation strings)
# ---------------------------------------------------------------------------

def _fmt_money(v: Optional[float]) -> str:
    """Compact dollar string: $1.2M / -$340k / $0."""
    if v is None:
        return "—"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:.0f}k"
    return f"{sign}${a:.0f}"


def _signed_money(v: Optional[float]) -> str:
    """Signed compact dollar string for the header one-liner: +$1.2M / −$84k."""
    if v is None:
        return "—"
    sign = "+" if v >= 0 else "−"  # unicode minus
    return f"{sign}{_fmt_money(abs(v))}"


def _dir_phrase(label: str, v: Optional[float]) -> Optional[str]:
    """'net long $1.2M delta' / 'net short $84k vega' / 'flat theta'."""
    if v is None:
        return None
    if abs(v) < 1:
        return f"flat {label}"
    side = "long" if v > 0 else "short"
    return f"net {side} {_fmt_money(abs(v))} {label}"


def _is_option(p) -> bool:
    return getattr(p, "asset_class", None) == "option"


def _coerce_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1) Net dollar Greeks + directional interpretation
# ---------------------------------------------------------------------------

def net_greeks_summary(account_state) -> dict:
    """Net dollar delta/gamma/vega/theta for the account, with directional
    interpretation strings. Reads the already-aggregated ``greeks.totals`` —
    does NOT re-sum positions (the engine already did that with skipna).

    Returns keys: dollar_delta, dollar_gamma, dollar_vega, dollar_theta,
    delta_pct_of_nav, headline (signed compact, for the KPI strip),
    interpretation (worded, for Analytics).

    Honesty rule: a total the engine set to None (every eligible row missing
    that greek) renders "—", never $0, and the interpretation says the data is
    missing; a partially-missing total keeps its sum but carries an explicit
    "n/m missing" cue — a zero from missing data must never look like a zero
    exposure.
    """
    greeks = getattr(account_state, "greeks", None)
    totals = getattr(greeks, "totals", None) or {}

    dd = _coerce_float(totals.get("dollar_delta"))
    dg = _coerce_float(totals.get("dollar_gamma"))
    dv = _coerce_float(totals.get("dollar_vega"))
    dt = _coerce_float(totals.get("dollar_theta"))
    delta_pct = _coerce_float(totals.get("delta_pct_of_nav"))

    cov = totals.get("greeks_coverage") or {}

    def _missing(col: str) -> tuple[int, int]:
        c = cov.get(col) or {}
        try:
            return int(c.get("n_missing") or 0), int(c.get("n_total") or 0)
        except (TypeError, ValueError):
            return 0, 0

    miss_d, tot_d = _missing("dollar_delta")
    miss_v, tot_v = _missing("dollar_vega")

    phrases = [p for p in (_dir_phrase("delta", dd), _dir_phrase("vega", dv)) if p]
    if dd is None and dv is None and (tot_d or tot_v):
        interpretation = (f"greeks unavailable — market data missing on all "
                          f"{tot_d} position(s)")
    else:
        interpretation = " · ".join(phrases) if phrases else "no greeks"
        cues = []
        if dd is not None and miss_d:
            cues.append(f"Δ missing {miss_d}/{tot_d}")
        if dv is not None and miss_v:
            cues.append(f"vega missing {miss_v}/{tot_v}")
        if cues:
            interpretation += " · " + ", ".join(cues)
    headline = (f"net Δ {_signed_money(dd)} · "
                f"net vega {_signed_money(dv)}")

    return {
        "dollar_delta": dd,
        "dollar_gamma": dg,
        "dollar_vega": dv,
        "dollar_theta": dt,
        "delta_pct_of_nav": delta_pct,
        "greeks_coverage": cov,
        "headline": headline,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 2) Long/short option premium split (income posture)
# ---------------------------------------------------------------------------

def long_short_premium_split(account_state) -> dict:
    """For the options sleeve only: premium **collected** (short options) vs
    **paid** (long options), in dollars and as a short share.

    Premium magnitude = ``|cost_basis|`` (entry premium) per option leg, with
    ``|market_value|`` as a fallback. Side is taken from the sign of quantity
    (negative = short = collected; positive = long = paid). This frames the
    account's income posture vs directional posture.

    Edge: an account with no options returns zeros / None shares cleanly.
    """
    collected = 0.0   # short premium $
    paid = 0.0        # long premium $
    n_short = 0
    n_long = 0

    for p in getattr(account_state, "positions", []) or []:
        if not _is_option(p):
            continue
        qty = _coerce_float(getattr(p, "quantity", None))
        if not qty:  # None or 0 — no directional premium
            continue
        prem = _coerce_float(getattr(p, "cost_basis", None))
        if prem is None:
            prem = _coerce_float(getattr(p, "market_value", None))
        if prem is None:
            continue
        prem = abs(prem)
        if qty < 0:
            collected += prem
            n_short += 1
        else:
            paid += prem
            n_long += 1

    total = collected + paid
    short_share = (collected / total) if total else None
    long_share = (paid / total) if total else None
    net = collected - paid

    if total == 0:
        posture = "no options"
    elif net > 0:
        posture = "net short premium"
    elif net < 0:
        posture = "net long premium"
    else:
        posture = "balanced premium"

    if total == 0:
        interpretation = "No options held."
    else:
        pct = round((short_share or 0.0) * 100)
        interpretation = (
            f"Collected {_fmt_money(collected)} / Paid {_fmt_money(paid)} · "
            f"{posture} {_fmt_money(abs(net))} ({pct}% of options premium is short)"
        )

    return {
        "collected": collected,
        "paid": paid,
        "net": net,
        "total": total,
        "short_share": short_share,
        "long_share": long_share,
        "n_short": n_short,
        "n_long": n_long,
        "posture": posture,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 3) Expiry ladder (options bucketed by time to expiry)
# ---------------------------------------------------------------------------

_LADDER_BUCKETS = [
    ("≤30d", lambda d: d <= 30),
    ("31–60d", lambda d: 31 <= d <= 60),
    ("61–90d", lambda d: 61 <= d <= 90),
    (">90d", lambda d: d > 90),
]


def expiry_ladder(account_state, as_of: Optional[date] = None) -> tuple[list[dict], int]:
    """Option positions bucketed by days-to-expiry window with the count and
    summed **strike** notional (``|qty| × multiplier × strike``) per bucket,
    plus the count of EXPIRED options excluded from the ladder.

    Returns ``(buckets, n_expired)`` — always the four buckets in order (zeros
    if empty). An expired option (negative DTE, e.g. from a stale extract — a
    designed-for operating mode) is a dead obligation, not an imminent one: it
    must never inflate the ≤30d window or its strike notional, so it is counted
    separately for the panel's "expired (n)" cue. DTE is computed from
    ``position.expiry`` against ``as_of`` (default today) — matching the
    blotter's DTE column. Positions without an expiry or strike are skipped.

    ``as_of`` is exposed only so tests can pin a reference date; runtime uses
    today, identical to the DTE shown in the positions grid.
    """
    ref = as_of or date.today()
    buckets = [{"label": lbl, "count": 0, "notional": 0.0} for lbl, _ in _LADDER_BUCKETS]
    n_expired = 0

    for p in getattr(account_state, "positions", []) or []:
        if not _is_option(p):
            continue
        expiry = getattr(p, "expiry", None)
        strike = _coerce_float(getattr(p, "strike", None))
        qty = _coerce_float(getattr(p, "quantity", None))
        mult = _coerce_float(getattr(p, "multiplier", None)) or 100.0
        if expiry is None or strike is None or qty is None:
            continue
        try:
            dte = (expiry - ref).days
        except Exception:
            continue
        if dte < 0:
            n_expired += 1
            continue
        notional = abs(qty) * mult * strike
        for i, (_lbl, pred) in enumerate(_LADDER_BUCKETS):
            if pred(dte):
                buckets[i]["count"] += 1
                buckets[i]["notional"] += notional
                break

    return buckets, n_expired


# ---------------------------------------------------------------------------
# Supporting aggregation (book glance) — pure/tested. Top-N concentration now lives
# in pm.risk.exposure on the economic (delta-$) basis (option-aware), not stock MV.
# ---------------------------------------------------------------------------

def book_summary(account_state) -> dict:
    """One-glance book counts for the KPI strip: nav, cash %, # positions,
    # options, # equity-like. Pure read over positions."""
    positions = getattr(account_state, "positions", []) or []
    nav = abs(_coerce_float(getattr(account_state, "nav", None)) or 0.0)
    n_options = sum(1 for p in positions if _is_option(p))
    n_equity = sum(1 for p in positions
                   if getattr(p, "asset_class", None) in ("equity", "fund_etf"))
    cash_mv = 0.0
    for p in positions:
        if getattr(p, "asset_class", None) == "cash":
            cash_mv += abs(_coerce_float(getattr(p, "market_value", None)) or 0.0)
    cash_pct = (cash_mv / nav) if nav else None
    return {
        "nav": _coerce_float(getattr(account_state, "nav", None)),
        "cash_pct": cash_pct,
        "n_positions": len(positions),
        "n_options": n_options,
        "n_equity": n_equity,
    }
