"""pm.risk — the portfolio risk-analytics engines that consume the pricing spine.

Home for the layered risk views (risk blueprint §4): the **exposure** engine lives
here today; the **scenario** (deterministic shocks) and **distribution** (correlated
Monte-Carlo) engines follow as they land. Each is a self-contained module
behind a pure interface — it reads an already-loaded ``PortfolioState`` and never
calls Bloomberg or recomputes in the UI.
"""
