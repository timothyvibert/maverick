"""The adjustment ticket — one close/open transaction, priced at contemporaneous mids.

A scanner candidate is a RESULTING position plus one net-cash scalar; the desk's
order line is the TRANSACTION that gets there. This module carries that transaction
as its own small type: a close-set plus an open-set, each leg with the signed trade
quantity it executes, the contemporaneous mid it was priced at, its own contract
multiplier, and its signed cash — plus the transaction's net cash, the resulting
position's economics, and a factual coverage flag when the transaction leaves a
short uncovered. A ticket with an empty open-set (plainly closing or capturing a
leg) and one with an empty close-set (an overlay write) are both first-class.

Sign law (one formula, no per-case arithmetic):

    cash = -trade_qty x multiplier x mid        (received positive)

where a CLOSE trades against the held position (``trade_qty = -held_qty``) and an
OPEN trades the new position's own sign (``trade_qty = +opened_qty``). Buying to
close a short pays; selling to open receives; selling to close a long receives;
buying to open pays — all four fall out of the one formula. Quantities are REAL
contracts (the ledger slice), multipliers per leg via ``Position.multiplier``
(stock legs carry multiplier 1) — never a hardcoded 100.

Everything here is pure and Dash-free: builders take numbers, the coverage
projection runs the existing pure structure detector over a COPIED position list
(no mutation of the live book), and the text formatter emits the copyable order
line with its as-of — a proposal, never an order.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import date, datetime
from types import SimpleNamespace
from typing import Optional

_STD_MULT = 100.0


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


@dataclass
class TicketLeg:
    """One executable line of the transaction."""
    action: str                      # BTC | STC | STO | BTO | BUY | SELL
    description: str                 # e.g. "NVDA 2026-09-18 190 C" / "NVDA stock"
    trade_qty: float                 # signed TRADE direction: + buy, - sell (real contracts/shares)
    mid: Optional[float]             # contemporaneous per-share mark this line priced at
    multiplier: float                # the leg's own contract multiplier (1 for stock)
    cash: Optional[float]            # -trade_qty x multiplier x mid; None when the mid is missing
    position_id: Optional[str] = None
    right: Optional[str] = None      # 'CALL' | 'PUT' | None (stock)
    strike: Optional[float] = None
    expiry: Optional[date] = None
    is_capture: bool = False         # a capture/close line (outside the rolled set)
    pnl_vs_entry: Optional[float] = None   # captures: run/decay vs ENTRY basis, $
    note: Optional[str] = None       # partial-slice remainder / fetched-mid disclosure


@dataclass
class AdjustmentTicket:
    """The whole transaction: what closes, what opens, what it nets, what remains."""
    close_set: list
    open_set: list
    net_cash: Optional[float]        # None if any leg's mid is missing — a dash, never $0
    net_label: str                   # 'roll only' | 'roll + close' | 'open + close' | 'close only' | 'open only'
    account: str
    underlier: str
    as_of: Optional[datetime]        # when the mids were pulled — every quotable number's clock
    resulting: Optional[dict] = None # label + the resulting legs' priced economics (or flat)
    conversion: Optional[str] = None # factual coverage flag ("leaves N naked short calls")
    warnings: list = field(default_factory=list)
    # Client-context blocks (each None -> its lines simply don't render). Three
    # clocks, each labeled where it renders: the spot's own as-of, the position
    # block's extract date, and the per-leg mids' pull time.
    analyst: Optional[dict] = None       # {provider, rating, target, reason}
    spot_info: Optional[dict] = None     # {spot, kind: 'live'|'snapshot'|None, asof}
    position_block: Optional[dict] = None  # {asof, legs: [{label, contract, basis, mv, pnl, pct}], coverage}


# ---------------------------------------------------------------------------
# Leg builders (the one place the sign law lives)
# ---------------------------------------------------------------------------

def _leg_cash(trade_qty, multiplier, mid) -> Optional[float]:
    m = _num(mid)
    if m is None:
        return None
    return -float(trade_qty) * float(multiplier) * m


def _close_action(held_qty: float, right: Optional[str]) -> str:
    if right is None:                       # stock
        return "SELL" if held_qty > 0 else "BUY"
    return "STC" if held_qty > 0 else "BTC"


def _open_action(qty: float, right: Optional[str]) -> str:
    if right is None:
        return "BUY" if qty > 0 else "SELL"
    return "BTO" if qty > 0 else "STO"


def close_leg(*, description, held_qty, mid, multiplier, position_id=None,
              right=None, strike=None, expiry=None, is_capture=False,
              entry_per_share=None, note=None) -> TicketLeg:
    """A line closing ``held_qty`` (signed as held) at ``mid``. Captures carry
    the run/decay vs ENTRY basis: ``(mid - entry) x held_qty x multiplier`` —
    positive is a gain for longs and shorts alike (the qty sign carries it)."""
    trade_qty = -float(held_qty)
    pnl = None
    if is_capture:
        m, e = _num(mid), _num(entry_per_share)
        if m is not None and e is not None:
            pnl = (m - e) * float(held_qty) * float(multiplier)
    return TicketLeg(action=_close_action(float(held_qty), right),
                     description=description, trade_qty=trade_qty, mid=_num(mid),
                     multiplier=float(multiplier),
                     cash=_leg_cash(trade_qty, multiplier, mid),
                     position_id=position_id, right=right, strike=strike,
                     expiry=expiry, is_capture=is_capture, pnl_vs_entry=pnl,
                     note=note)


def open_leg(*, description, qty, mid, multiplier=_STD_MULT, position_id=None,
             right=None, strike=None, expiry=None, note=None) -> TicketLeg:
    """A line opening ``qty`` (signed as the new position) at ``mid``."""
    trade_qty = float(qty)
    return TicketLeg(action=_open_action(trade_qty, right), description=description,
                     trade_qty=trade_qty, mid=_num(mid), multiplier=float(multiplier),
                     cash=_leg_cash(trade_qty, multiplier, mid),
                     position_id=position_id, right=right, strike=strike,
                     expiry=expiry, note=note)


def assemble(close_set, open_set, *, account, underlier, as_of,
             resulting=None, conversion=None, warnings=None,
             analyst=None, spot_info=None, position_block=None) -> AdjustmentTicket:
    """The whole ticket. ``net_cash`` is None the moment any leg's cash is —
    a partially-priced net would read as a real number. The context blocks
    (``analyst`` / ``spot_info`` / ``position_block``) are optional; a ticket
    without them renders exactly the transaction, as before."""
    close_set, open_set = list(close_set or []), list(open_set or [])
    cashes = [lg.cash for lg in close_set + open_set]
    net = None if (not cashes or any(c is None for c in cashes)) else float(sum(cashes))
    has_roll = any(not lg.is_capture for lg in close_set)
    has_cap = any(lg.is_capture for lg in close_set)
    if open_set and close_set:
        label = ("roll + close" if (has_roll and has_cap)
                 else "open + close" if has_cap else "roll only")
    elif open_set:
        label = "open only"
    else:
        label = "close only"
    return AdjustmentTicket(close_set=close_set, open_set=open_set, net_cash=net,
                            net_label=label, account=account, underlier=underlier,
                            as_of=as_of, resulting=resulting, conversion=conversion,
                            warnings=list(warnings or []), analyst=analyst,
                            spot_info=spot_info, position_block=position_block)


# ---------------------------------------------------------------------------
# Coverage projection — the existing pure detector over a COPIED book
# ---------------------------------------------------------------------------

def apply_transaction(positions, closes, opens, *, underlying_symbol,
                      pid_prefix="ticket-new") -> list:
    """The projected position list after the transaction. ``positions`` is the
    account's live list (never mutated — reduced legs are dataclass copies);
    ``closes`` is ``[(position_id, closed_signed_qty)]`` (same sign as held);
    ``opens`` is ``[{right, strike, expiry, qty}]`` in signed STANDARD contracts
    (new legs are standard listed contracts)."""
    closed = {}
    for pid, q in (closes or []):
        closed[pid] = closed.get(pid, 0.0) + float(q)
    out = []
    for p in positions:
        dq = closed.get(getattr(p, "position_id", None))
        if dq is None:
            out.append(p)
            continue
        remaining = (_num(p.quantity) or 0.0) - dq
        if abs(remaining) > 1e-9:
            if is_dataclass(p):
                out.append(replace(p, quantity=remaining))
            else:                        # duck-typed (tests, projections)
                q = copy.copy(p)
                q.quantity = remaining
                out.append(q)
    for i, o in enumerate(opens or [], start=1):
        out.append(SimpleNamespace(
            position_id=f"{pid_prefix}-{i}", asset_class="option",
            symbol=underlying_symbol, underlying_symbol=underlying_symbol,
            right=str(o["right"]).upper(), option_type=str(o["right"]).upper(),
            strike=_num(o.get("strike")), expiry=o.get("expiry"),
            quantity=float(o["qty"]), multiplier=100,
            option_contract_key=None))
    return out


def project_structures(account, positions, trades_by_underlying, underlying_symbol) -> list:
    """Run the existing pure structure detector over ONE underlying's positions
    (a book that may be a projected copy). No mutation, no I/O, no resolutions —
    the raw detection is the name-level coverage truth the flag reads."""
    from pm.insight.structures import detect_account_structures
    subset = [p for p in positions
              if (getattr(p, "asset_class", None) == "option"
                  and getattr(p, "underlying_symbol", None) == underlying_symbol)
              or (getattr(p, "asset_class", None) in ("equity", "fund_etf")
                  and getattr(p, "symbol", None) == underlying_symbol)]
    return detect_account_structures(SimpleNamespace(
        account=account, positions=subset,
        trades_by_underlying=trades_by_underlying or {}))


def uncovered_counts(structures) -> dict:
    """Uncovered short contracts per right: legs marked with the naked-excess
    roles, counted once per contention group (the obligations rule —
    alternatives are readings of the same legs, never summed). Stock cover
    stays senior by construction: the detector's reservation and name-level
    reconcile already downgrade any short that deliverable stock (or a
    same-right later-dated long) still covers."""
    from pm.insight.structures import NAKED_EXCESS_SHORT_CALL, NAKED_EXCESS_SHORT_PUT
    counts = {"CALL": 0.0, "PUT": 0.0}
    per_group: dict = {}
    for st in structures:
        for leg in getattr(st, "legs", []) or []:
            right = {NAKED_EXCESS_SHORT_CALL: "CALL",
                     NAKED_EXCESS_SHORT_PUT: "PUT"}.get(getattr(leg, "role", None))
            if right is None:
                continue
            n = abs(_num(leg.allocated_qty) or 0.0)
            group = getattr(st, "contention_group", None)
            if group:
                g = per_group.setdefault((group, right), {})
                pid = getattr(leg, "position_id", None)
                g[pid] = max(g.get(pid, 0.0), n)
            else:
                counts[right] += n
    for (_grp, right), g in per_group.items():
        counts[right] += sum(g.values())
    return counts


def legs_as_positions(leg_dicts, underlying_symbol, *, account="ticket") -> list:
    """Payoff-engine leg dicts -> minimal duck-typed positions, so the pure
    detector can NAME the exact leg set the ticket's economics price (the same
    scope, never the whole name — name-level detection stays with the coverage
    flag, where stock seniority needs the full book)."""
    out = []
    for i, d in enumerate(leg_dicts or [], start=1):
        pid = d.get("position_id") or f"leg-{i}"
        if d.get("opt_type") in ("Call", "Put"):
            right = "CALL" if d["opt_type"] == "Call" else "PUT"
            out.append(SimpleNamespace(
                account=account, position_id=pid, asset_class="option",
                symbol=underlying_symbol, underlying_symbol=underlying_symbol,
                right=right, option_type=right, strike=_num(d.get("K")),
                expiry=d.get("expiry"), quantity=_num(d.get("qty")) or 0.0,
                multiplier=100, option_contract_key=None))
        elif d.get("opt_type") == "Stock":
            out.append(SimpleNamespace(
                account=account, position_id=pid, asset_class="equity",
                symbol=underlying_symbol, underlying_symbol=None, right=None,
                option_type=None, strike=None, expiry=None,
                quantity=_num(d.get("qty")) or 0.0, multiplier=1,
                option_contract_key=None))
    return out


def _role_name(d) -> str:
    qty = _num(d.get("qty")) or 0.0
    side = "long" if qty >= 0 else "short"
    kind = {"Call": "call", "Put": "put", "Stock": "stock"}.get(d.get("opt_type"), "leg")
    return f"{side} {kind}"


def resulting_label(leg_dicts, underlying_symbol) -> Optional[str]:
    """A display label for the RESULTING leg set — the same legs the ticket's
    economics price. One leg names itself by role; a multi-leg set is named by
    the pure detector over exactly those legs; zero or ambiguous readings
    degrade to None (the caller falls back to a leg-count label, no guessing)."""
    legs = list(leg_dicts or [])
    if not legs:
        return None
    if len(legs) == 1:
        return _role_name(legs[0])
    structures = project_structures(
        "ticket", legs_as_positions(legs, underlying_symbol), None,
        underlying_symbol)
    types = {getattr(st, "type", None) for st in structures
             if not getattr(st, "contention_group", None)}
    types = {t for t in types if t}
    if len(types) == 1:
        return next(iter(types)).replace("_", " ")
    return None


def legs_summary(leg_dicts) -> Optional[str]:
    """A compact leg roster for the RESULTING heading — '217,600 sh + 2,176
    short 2026-09-18 15 C'. Payoff-engine leg dicts in (standard contracts);
    None when there are no legs."""
    parts = []
    for d in leg_dicts or []:
        q = _num(d.get("qty"))
        if not q:
            continue
        if d.get("opt_type") == "Stock":
            parts.append(f"{q:,.0f} sh")
        elif d.get("opt_type") in ("Call", "Put"):
            side = "short" if q < 0 else "long"
            exp = d.get("expiry")
            exp_s = f"{exp:%Y-%m-%d}" if exp is not None else "?"
            k = d.get("K")
            k_s = f"{k:g}" if k is not None else "?"
            r = "C" if d["opt_type"] == "Call" else "P"
            parts.append(f"{abs(q):,.0f} {side} {exp_s} {k_s} {r}")
    return " + ".join(parts) if parts else None


def _econ_bits(e) -> list:
    """The priced-economics fragments shared by ``resulting_line`` (the drawer
    band) and the copy text's RESULTING section — one composition, two renders.
    PoP / capital-at-risk render only when the economics dict carries them."""
    e = e or {}
    bits = []
    mp = "unbounded" if e.get("unbounded_gain") else (
        _cash_str(e.get("max_profit")) if e.get("max_profit") is not None else None)
    ml = "unbounded" if e.get("unbounded_loss") else (
        _cash_str(e.get("max_loss")) if e.get("max_loss") is not None else None)
    if mp is not None:
        bits.append(f"max profit {mp}")
    if ml is not None:
        bits.append(f"max loss {ml}")
    bes = e.get("breakevens")
    if bes:
        bits.append("BE " + " / ".join(f"{b:,.2f}" for b in bes))
    elif e.get("always_profitable"):
        bits.append("always profitable at expiry")
    elif e.get("always_loss"):
        bits.append("always a loss at expiry")
    pop = e.get("pop")
    if pop is not None:
        bits.append(f"PoP {pop * 100:.0f}%")
    car = e.get("capital_at_risk")
    if car is not None:
        bits.append(f"capital at risk ${abs(car):,.0f}")
    return bits


def resulting_line(label, economics) -> Optional[str]:
    """The one-line resulting summary: label + the priced economics that
    survived. ``economics=None`` with a label still renders the label; an
    all-closed transaction should pass label='flat'."""
    if not label and not economics:
        return None
    return " - ".join([label or "adjusted position"] + _econ_bits(economics))


def coverage_conversion(before: dict, after: dict) -> Optional[str]:
    """A factual label when the transaction INCREASES the uncovered short count
    on either right — numbers only, never advice. None when coverage is
    preserved (including when deliverable stock still covers the short)."""
    parts = []
    for right, noun in (("CALL", "short call"), ("PUT", "short put")):
        delta = (after.get(right) or 0.0) - (before.get(right) or 0.0)
        if delta > 1e-9:
            n = int(round(delta))
            parts.append(f"leaves {n} naked {noun}{'s' if n != 1 else ''}")
    return " · ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# The copyable text (design law: every client-quotable number states its as-of)
# ---------------------------------------------------------------------------

def _cash_str(v, *, dash="-") -> str:
    if v is None:
        return dash
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"


def _money_str(v, *, dash="-") -> str:
    """Unsigned-positive money ('$2,420,753' / '-$54,925') for basis/MV columns;
    P&L stays explicitly signed via ``_cash_str``."""
    if v is None:
        return dash
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def _pct_str(v, *, dash="-") -> str:
    if v is None:
        return dash
    return f"{v * 100:+.1f}%"


def _header_lines(t: "AdjustmentTicket") -> list:
    """The client-context header: the spot with ITS OWN as-of (a live slice
    pull shares the mids' clock; the morning snapshot says so; missing market
    data dashes), and the research line naming the provider AS DATA — the
    template stays provider-neutral, falling back to 'analyst' when no
    provider rides the record."""
    lines = []
    if t.spot_info is not None:
        s = t.spot_info or {}
        spot, kind, asof = s.get("spot"), s.get("kind"), s.get("asof")
        if spot is None:
            lines.append("spot - (no market data)")
        elif kind == "live":
            ts = f" ({asof:%Y-%m-%d %H:%M})" if asof else ""
            lines.append(f"spot ${spot:,.2f} - live, same pull as the mids below{ts}")
        else:
            lines.append(f"spot ${spot:,.2f} - morning snapshot")
    if t.analyst is not None:
        a = t.analyst or {}
        name = a.get("provider") or "analyst"
        rating = a.get("rating")
        rating_s = str(rating).strip().title() if rating else "-"
        if not rating and a.get("reason"):
            rating_s += f" ({a['reason']})"
        target = a.get("target")
        target_s = f"${target:,.2f}" if target is not None else "-"
        lines.append(f"{name} rating {rating_s} - price target {target_s}")
    return lines


def _position_lines(t: "AdjustmentTicket") -> list:
    """The CURRENT POSITION block — extract marks, labeled with the extract's
    own date (the third clock): per-leg basis / MV / P&L / P&L%, then the
    coverage line carrying the detector's covered-vs-remaining share split."""
    pb = t.position_block
    if not pb:
        return []
    asof = pb.get("asof")
    lines = ["", f"CURRENT POSITION - marks from the {asof} extract" if asof
             else "CURRENT POSITION"]
    for lg in pb.get("legs") or []:
        left = f"{lg.get('label') or 'leg':<13} {lg.get('contract') or '':<36}"
        lines.append(f"{left} basis {_money_str(lg.get('basis')):>12}   "
                     f"MV {_money_str(lg.get('mv')):>12}   "
                     f"P&L {_cash_str(lg.get('pnl'))} ({_pct_str(lg.get('pct'))})")
    cov = pb.get("coverage")
    if cov:
        lines.append(cov)
    return lines


def coverage_line(sh_alloc, sh_total, n_short, noun) -> Optional[str]:
    """The covered-vs-remaining share split, the same n-of-m discipline the
    By-Structure quantity columns use — never a cross-class sum. None when any
    component is missing (no line beats a guessed one)."""
    a, tot, n = _num(sh_alloc), _num(sh_total), _num(n_short)
    if a is None or n is None or not n:
        return None
    if tot is None or abs(tot - a) < 1e-9:
        return (f"coverage: {a:,.0f} sh cover the {abs(n):,.0f} {noun} - "
                "fully covered, 0 sh outside")
    return (f"coverage: {a:,.0f}/{tot:,.0f} sh cover the {abs(n):,.0f} {noun} - "
            f"{tot - a:,.0f} sh outside this structure")


def _qty_str(v) -> str:
    f = float(v)
    s = f"{f:+g}"
    return s


def _line(lg: TicketLeg) -> str:
    mid = f"@ {lg.mid:.2f}" if lg.mid is not None else "@ -"
    bits = [f"{lg.action:<5}", f"{_qty_str(lg.trade_qty):>5}",
            f"{lg.description} {mid}", _cash_str(lg.cash)]
    if lg.is_capture and lg.pnl_vs_entry is not None:
        bits.append(f"({_cash_str(lg.pnl_vs_entry)} vs entry)")
    if lg.note:
        bits.append(f"[{lg.note}]")
    return "  ".join(bits)


def ticket_text(t: AdjustmentTicket) -> str:
    """Plain text the principal can paste — header (spot + research, each with
    its clock), the CURRENT POSITION extract marks, the PROPOSAL transaction at
    contemporaneous mids, and the RESULTING position. Every quotable number
    states its as-of; a proposal, never an order. Context blocks the caller
    didn't supply simply don't render (the bare transaction reads as before)."""
    asof = t.as_of.strftime("%Y-%m-%d %H:%M") if t.as_of else "unknown"
    lines = [
        f"ADJUSTMENT TICKET - {t.underlier} - account {t.account} - proposal, not an order",
    ]
    lines += _header_lines(t)
    lines += _position_lines(t)
    lines += ["", "PROPOSAL - one net transaction"]
    lines += [_line(lg) for lg in t.close_set]
    lines += [_line(lg) for lg in t.open_set]
    net = _cash_str(t.net_cash)
    if t.net_cash is not None:
        net += " credit" if t.net_cash >= 0 else " debit"
    lines.append(f"NET ({t.net_label})  {net}")
    lines.append(f"as of {asof} - per-leg mids are contemporaneous marks from "
                 "that pull; indicative, not executable quotes")
    res = t.resulting or {}
    if res.get("line") or res.get("label"):
        label = res.get("label") or "adjusted position"
        summary = res.get("legs_summary")
        head = f"RESULTING - {label}" + (f" ({summary})" if summary else "")
        bits = _econ_bits(res.get("economics"))
        lines += ["", head, " - ".join(bits) if bits else "-"]
        if t.open_set:
            n_new = len(t.open_set)
            when = f"the {t.as_of:%H:%M}" if t.as_of else "the pulled"
            lines.append("kept legs at entry basis - "
                         f"new leg{'s' if n_new != 1 else ''} at {when} "
                         f"mid{'s' if n_new != 1 else ''}")
    if t.conversion:
        lines.append(t.conversion)
    for w in t.warnings:
        lines.append(f"note: {w}")
    return "\n".join(lines)
