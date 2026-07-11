# Data directory

The application reads its morning **holdings + trades extract** from this
folder. Extracts are **not** committed — this directory ships empty apart from
this note.

Place an `.xlsx` workbook here with two sheets:

- **Holdings** — one row per position (account, asset class, instrument details,
  quantity, valuation, market value, cost basis, unrealized P&L).
- **Trades** — the trade journal (account, trade date, buy/sell, lifecycle
  action, instrument details, quantity, principal).

The most recent extract in this folder is selected automatically on load.
