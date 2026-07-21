"""NYSE full-day holiday calendar, generated from the exchange's published rules.

Consumed by the tenor day-count (``conventions.year_frac``): the live probe
(four non-dividend names, ATM calls and puts, expiries bracketing Labor Day
and Thanksgiving) showed the terminal's option clock excludes exchange
holidays, while a plain weekday count overstates the tenor by one day per
holiday spanned. ``year_frac`` is pinned in the byte-identical gate, so wiring
this calendar in was a baseline-recapture boundary: the date-bearing baseline
entries were re-pinned when the wiring landed (2026-07-20), every other entry
kept its prior pinned value, and the outgoing baseline is archived beside the
live fixture.

Why a rules generator, not a static list or an external source: NYSE holidays
are deterministic functions of the year (fixed dates, nth-weekday rules, and
Good Friday via the Gregorian Easter computus) plus the exchange's observation
rules, so a generator needs no annual upkeep. A static list decays every
January; a calendar library adds a dependency for ten dates a year; a terminal
pull would make pricing depend on a live Bloomberg session. Known limitation,
accepted: unscheduled closures (mourning days, disasters) cannot be generated
— they shift a forward window by one day only from the day they are announced.
Rule changes are rare (Juneteenth, added 2022, was the first new full-day
holiday in decades) and are one-line edits here.

Stdlib-only (``datetime``): keeps the pricing package's import isolation.
"""
from __future__ import annotations

import datetime as dt

# Juneteenth became an NYSE full-day holiday in 2022.
_JUNETEENTH_FROM = 2022


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter Sunday (Anonymous Gregorian / Meeus-Jones-Butcher
    computus). Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th (1-based) *weekday* (Mon=0) of *month*."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """The last *weekday* (Mon=0) of *month*."""
    if month == 12:
        nxt = dt.date(year + 1, 1, 1)
    else:
        nxt = dt.date(year, month + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(holiday: dt.date) -> dt.date | None:
    """NYSE observation rule for a fixed-date holiday: Saturday is observed the
    Friday before, Sunday the Monday after — except a Saturday New Year's Day,
    which the exchange does not observe at all (Rule 7.2 precedent: markets
    were open Friday 2021-12-31 ahead of Sat 2022-01-01)."""
    if holiday.weekday() == 5:  # Saturday
        if holiday.month == 1 and holiday.day == 1:
            return None
        return holiday - dt.timedelta(days=1)
    if holiday.weekday() == 6:  # Sunday
        return holiday + dt.timedelta(days=1)
    return holiday


def nyse_holidays(year: int) -> tuple[dt.date, ...]:
    """All NYSE full-day holidays observed within calendar *year*, ascending.

    Note a Friday observance of a Saturday Jan 1 would belong to the PRIOR
    year; under the no-observance exception above it never occurs, so every
    date returned falls inside *year*.
    """
    days: list[dt.date] = []
    for fixed in (dt.date(year, 1, 1),                       # New Year's Day
                  dt.date(year, 7, 4),                       # Independence Day
                  dt.date(year, 12, 25)):                    # Christmas
        obs = _observed(fixed)
        if obs is not None:
            days.append(obs)
    if year >= _JUNETEENTH_FROM:
        obs = _observed(dt.date(year, 6, 19))
        if obs is not None:
            days.append(obs)
    days.append(_nth_weekday(year, 1, 0, 3))                 # MLK Day
    days.append(_nth_weekday(year, 2, 0, 3))                 # Washington's Birthday
    days.append(easter_sunday(year) - dt.timedelta(days=2))  # Good Friday
    days.append(_last_weekday(year, 5, 0))                   # Memorial Day
    days.append(_nth_weekday(year, 9, 0, 1))                 # Labor Day
    days.append(_nth_weekday(year, 11, 3, 4))                # Thanksgiving
    return tuple(sorted(days))


def holidays_between(start: dt.date, end: dt.date) -> tuple[dt.date, ...]:
    """NYSE holidays in the half-open window [start, end) — the same window
    convention as ``np.busday_count(start, end)``; ``conventions.year_frac``
    consumes this as
    ``np.busday_count(start, end, holidays=holidays_between(start, end))``."""
    if end <= start:
        return ()
    out = []
    for year in range(start.year, end.year + 1):
        for h in nyse_holidays(year):
            if start <= h < end:
                out.append(h)
    return tuple(out)
