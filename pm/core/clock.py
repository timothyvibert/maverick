"""The one calendar clock behind every 'today' read on the review path.

Every module that needs the current date for calendar semantics — days held,
DTE, business-day windows, expiry buckets, event calendars — reads
``clock.today()`` (or ``clock.now()``) instead of calling ``date.today()``
directly. At runtime the two are identical: nothing in the live app pins the
clock, so ``today()`` is the wall clock.

The indirection exists so a dated book can be read against its own calendar:
``pinned(as_of)`` fixes the clock inside the ``with`` block, and every
downstream read — position builder, signal library, pattern detectors,
structure fires, risk calendars, grid DTE columns — sees the same date. A
review of an old extract then produces the same fires, the same windows and
the same day counts it produced on the day the extract was cut, instead of
drifting as wall time passes.

Freshness *stamps* (``loaded_at``, ``pulled_at``, "pulled N min ago") stay on
the real wall clock deliberately — they describe when data was fetched, not
which calendar the book is read against.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from typing import Iterator, Optional

_PINNED: Optional[date] = None


def today() -> date:
    """The calendar date the book is read against (wall clock unless pinned)."""
    return _PINNED if _PINNED is not None else date.today()


def now() -> datetime:
    """Datetime companion to :func:`today` (midnight of the pinned date when
    pinned, else the real wall clock)."""
    if _PINNED is not None:
        return datetime.combine(_PINNED, datetime.min.time())
    return datetime.now()


@contextmanager
def pinned(as_of: date) -> Iterator[None]:
    """Fix the clock at ``as_of`` for the duration of the block.

    Not used by the live app; it exists so a dated extract can be loaded and
    asserted against the calendar it was cut under.
    """
    global _PINNED
    prev = _PINNED
    _PINNED = as_of
    try:
        yield
    finally:
        _PINNED = prev
