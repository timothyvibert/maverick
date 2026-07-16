"""Slider + numeric-entry dial pairs — the sync that makes typed shocks commit.

The Dash slider's built-in direct-entry box (this release) updates its thumb on
Enter without committing the value to the server: a typed shock renders as
applied while nothing reprices, and text left in the box survives a programmatic
reset and re-commits on the next blur — a phantom shock on a stress surface.
Every dial therefore disables the built-in box (``allow_direct_input=False``)
and pairs the slider with an explicit ``dcc.Input``; this module registers the
two-way sync per dial:

- a typed value (commits on Enter/blur) is snapped to the dial's own step and
  clamped to the slider's own min/max, then written onto the slider — which
  chains into the surface's recompute callback exactly like a drag;
- a drag / keyboard move on the slider is written back into the box, so the
  pair never disagrees.

A write that changes BOTH halves in one update cycle (a preset / reset chip
setting slider and box together) is left alone: the pair is already consistent,
and echoing it back would fire a duplicate recompute.
"""
from __future__ import annotations

from typing import Iterable

from dash import Input, Output, State, ctx, no_update


def register_dial_sync(app, slider_ids: Iterable[str]) -> None:
    """Register the two-way sync for each ``<sid>`` slider / ``<sid>-num`` input pair."""
    for sid in slider_ids:
        _register_one(app, sid)


def resolve_dial_pair(triggered_ids, sid, slider_v, num_v, lo, hi, step):
    """The pure sync decision (unit-tested): ``(slider_out, num_out)`` where None
    means leave-as-is. ``triggered_ids`` is the set of component ids that changed
    this update cycle."""
    if len(triggered_ids) != 1:
        return None, None                  # both halves written together (preset)
    if f"{sid}-num" in triggered_ids:
        if num_v is None:
            # Mid-edit empty / lone minus sign (the box commits per keystroke,
            # debounce=False): not a value yet — never coerce it to zero, or a
            # user could not type a negative or multi-digit number.
            return None, None
        v = num_v
        if step:
            v = round(round(v / step) * step, 10)
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        # Write the slider (chains the recompute like a drag); snap the box back
        # only where step-snap / clamping changed the typed value.
        return v, (v if v != num_v else None)
    return None, slider_v


def _register_one(app, sid: str) -> None:
    @app.callback(
        Output(sid, "value", allow_duplicate=True),
        Output(f"{sid}-num", "value", allow_duplicate=True),
        Input(sid, "value"),
        Input(f"{sid}-num", "value"),
        State(sid, "min"),
        State(sid, "max"),
        State(sid, "step"),
        prevent_initial_call=True,
    )
    def _sync(slider_v, num_v, lo, hi, step, _sid=sid):
        trigs = {t["prop_id"].rsplit(".", 1)[0] for t in (ctx.triggered or [])}
        s_out, n_out = resolve_dial_pair(trigs, _sid, slider_v, num_v, lo, hi, step)
        return (no_update if s_out is None else s_out,
                no_update if n_out is None else n_out)
