"""Dash app for Portfolio Manager — v1.0 UI shell.

Thin entry point: build the two-tab shell (``pm.ui.shell``) with an *empty*
state so the server is reachable immediately, register callbacks, run. The
PortfolioState is loaded *after* first paint by a one-shot ``initial-load``
callback (see ``pm.ui.blotter.callbacks``), which writes the runtime singleton
and populates the tabs — so the slow Bloomberg prefetch never blocks startup.

The runtime PortfolioState singleton is owned by ``pm.ui.state_access`` (not
here): ``python -m pm.app`` runs this file as ``__main__``, a *different* module
object from the ``pm.app`` that callbacks import, so a global stored here would
be invisible to them. The UI reads/writes state exclusively through
``state_access`` so there is one canonical instance.
"""
from __future__ import annotations

import dash

from pm.config import HOST, PORT
from pm.ui import state_access as sa

# Back-compat alias: any reader of ``pm.app._DASHBOARD_STATE`` sees the same
# dict state_access owns (single instance, regardless of __main__).
_DASHBOARD_STATE = sa._RUNTIME


def build_app() -> dash.Dash:
    """Build the shell (empty state) + register callbacks. Returns the app.
    Data loads after first paint via the ``initial-load`` callback — no
    Bloomberg I/O happens here, so the server binds immediately."""
    from pm.ui.shell import build_shell
    from pm.ui.blotter.callbacks import register_callbacks
    from pm.ui.deepdive.callbacks import register_deepdive_callbacks
    from pm.ui.drawers.payoff import register_payoff_callbacks
    from pm.ui.drawers.scanner import register_comparison_callbacks, register_scanner_callbacks

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Portfolio Manager"
    app.layout = build_shell(sa.get_state())  # None at cold start
    register_callbacks(app)
    register_deepdive_callbacks(app)
    register_payoff_callbacks(app)
    register_scanner_callbacks(app)
    register_comparison_callbacks(app)
    return app


if __name__ == "__main__":
    # threaded=True so a long Refresh BBG reload on one request thread does not
    # freeze the UI on others (the spinner shows; old data stays interactive).
    build_app().run(host=HOST, port=PORT, debug=False, threaded=True)
