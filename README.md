# Maverick

A portfolio review and idea-generation tool for options and equity sales
trading and advisory. It turns a morning holdings-and-trades extract into a
cross-account signal blotter and a per-account deep dive, so a desk can triage
what needs attention and build client conversations from it — fast.

## Overview

Maverick runs as a local single-page [Dash](https://dash.plotly.com/)
application. Each morning it ingests a holdings + trades extract, enriches it
with live Bloomberg data, and runs a **deterministic insight engine** that emits
pattern-based alerts with a complete audit trail. The result is presented across
two surfaces:

- **Morning Blotter** — a dense, cross-account grid of every alert, triaged by
  urgency tier, one row per position. The desk's first stop: what fired
  overnight and where to act.
- **Account Deep Dive** — a per-account view: a KPI strip, the full position
  book (viewable by position or grouped into recognized multi-leg structures)
  with merged alerts, portfolio analytics, and recent trades. The follow-up
  stop: full context on a single account before a client call.

Both surfaces open the same evidence drawer, so any alert or position is one
click from its full signal tearsheet and the math behind it.

## Features

**Insight engine**
- A library of signals across trend & momentum, volatility, catalysts,
  sentiment/ratings, position-specifics, and a blended composite score.
- Pattern detectors turn those signals into tiered alerts — *act today*,
  *worth raising*, *FYI* — each carrying a complete audit `trace` (every input,
  its source and as-of, the computation, the thresholds, and the result).
- Stale or missing inputs are surfaced explicitly; a pattern that depends on a
  stale signal does not fire.

**Morning Blotter**
- One row per position with its alerts consolidated; filter by tier, group by
  account or pattern.
- Click through to a per-position alert view (every alert stacked, with its
  rationale and audit trace) or a per-underlying signal tearsheet.

**Account Deep Dive**
- KPI strip: NAV, cash %, position/option counts, a net-Greeks one-liner, and
  per-tier alert counts.
- Institutional position table across every asset class, with alerts merged in —
  viewable **by position** or **by structure** (detected covered calls, collars,
  verticals, and put structures with their net economics), ordered by attention.
  Confirming a structure unlocks the management alerts specific to it.
- Analytics: net dollar Greeks, options premium collected-vs-paid split, an
  expiry ladder (strike-obligation exposure by window), sector breakdown with
  weighted beta, and top concentrations.
- The account's full recent trade book, most-recent first.

**Risk & pricing**
- A self-contained options pricing engine (Black-Scholes, Bjerksund-Stensland
  2002, CRR binomial with discrete dividends, implied vol, payoff/breakeven/PoP
  kernels), validated against published targets.
- Per-account exposure aggregation (net dollar Greeks, SPX dollar-beta, vega by
  tenor, a conserved structure→account rollup) and an interactive scenario
  section (co-moving spot/vol/rate/time shocks over a P&L heatmap).
- A payoff drawer for any structure or position: at-expiry and horizon curves,
  breakevens, max P&L, probability of profit, Greeks now-vs-shocked.
- An option scanner: on-demand chain slices, a fitted vol smile, and
  replacement/roll candidates ranked by objective fit and account fit.

## Architecture

A layered, read-only-at-the-edge pipeline:

```
Holdings/trades extract  →  ingest (typed Position model)
                         →  Bloomberg enrichment + insight engine
                         →  PortfolioState (single source of truth)
                         →  UI (Dash + AG Grid) — reads only
```

The engine computes everything upstream while building `PortfolioState`; the UI
layer only reads from it and never recomputes. Data is fetched once per load in
a single Bloomberg pass and sliced per account.

- **UI:** Dash with [dash-ag-grid](https://dash.plotly.com/dash-ag-grid) for the
  dense grids.
- **Market data:** Bloomberg via
  [polars-bloomberg](https://pypi.org/project/polars-bloomberg/) (BLPAPI).
- **Data handling:** pandas / numpy; Excel via openpyxl.

The app launches immediately and loads market data after first paint, with a
loading indicator next to the refresh controls. Two refresh actions are
available — re-read the latest extract file, or re-pull market data on the
current extract — and both are non-blocking, so the current view stays
interactive while new data loads.

## Requirements

- Python 3.12+
- A Bloomberg Terminal with BLPAPI available (for live enrichment). Without it,
  the app still runs and renders everything derivable from the extract;
  Bloomberg-dependent signals are marked unavailable.

## Setup

```bash
pip install -r requirements.txt
```

A dedicated environment matching the pins is recommended (e.g. a conda env with
Python 3.12+); the app is developed and tested against exactly the versions in
`requirements.txt`.

## Running

```bash
python -m pm.app
```

Then open <http://127.0.0.1:8062/>. On Windows, `launch.bat` activates the
environment, frees the port, starts the server, and opens the browser for you.

## Data

Place the morning extract in `pm/data/` (kept local — never committed), named
`<source>_extract_YYYYMMDD_HHMMSS.xlsx` (any lowercase-alphanumeric source
prefix). It is an `.xlsx` workbook with two sheets:

- **Holdings** — one row per position: account, asset class, instrument details
  (ticker, option type/strike/expiry where applicable), quantity, valuation,
  market value, cost basis, and unrealized P&L.
- **Trades** — the trade journal: account, trade date, buy/sell, lifecycle
  action, instrument details, quantity, and principal.

Column headers are normalized on load; the most recent extract in the folder is
selected automatically.

## License

Released under the MIT License. See [LICENSE](LICENSE).
