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
    _harden(app.server)
    return app


# Host values a browser may legitimately present for the loopback bind.
_TRUSTED_HOSTS = ("127.0.0.1", "localhost")


def _harden(server) -> None:
    """Loopback hardening. The app binds 127.0.0.1 only, but the state-mutating
    callbacks (mute/restore, structure resolve, threshold apply, scanner pulls)
    are plain POSTs — a DNS-rebinding page could reach them through the
    victim's own browser. Rejecting foreign Host headers closes that route;
    the response headers stop framing and MIME sniffing.

    No Content-Security-Policy header here: Dash's renderer boots from inline
    scripts, so a useful CSP needs 'unsafe-inline' script-src — near-zero value
    against the rebinding threat the Host check already closes, at real risk of
    breaking the grid/plotly renderers. Deliberate omission, not an oversight."""
    from flask import abort, request

    @server.before_request
    def _reject_foreign_hosts():
        host = (request.host or "").rsplit(":", 1)[0]
        if host not in _TRUSTED_HOSTS:
            abort(403)

    @server.after_request
    def _static_headers(resp):
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp


if __name__ == "__main__":
    # threaded=True so a long Refresh BBG reload on one request thread does not
    # freeze the UI on others (the spinner shows; old data stays interactive).
    build_app().run(host=HOST, port=PORT, debug=False, threaded=True)
