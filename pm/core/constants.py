"""Legacy constants retained from the retired v0.10 engine, for reference.

No live module imports these; the values document the old engine's grid and
tolerance conventions only.
"""

GRID_SIZE = 300            # Number of price-grid points for payoff computation
GRID_RANGE = 3.0           # Price grid extends to GRID_RANGE × spot
BE_TOLERANCE = 0.50        # P&L values within ±$0.50 of zero count as "at zero"
EPSILON = 1e-6             # General floating-point comparison tolerance
FLATNESS_THRESHOLD = 0.01  # Slope / zone-width threshold for flatness checks
DEFAULT_MULTIPLIER = 100   # Standard option contract multiplier

ENGINE_MISSING = "--"      # Produced by analysis_pack.py, consumed by report_model.py
DISPLAY_MISSING = "\u2014"     # Em-dash shown to users in PDF reports (view_model.py)
