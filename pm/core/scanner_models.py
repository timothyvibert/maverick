"""Data models for the unified scanner engine.

Pure Python dataclasses — no BQL, Dash, or Bloomberg dependencies.
Used by signal_detector.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Signal:
    """A detected vol or technical signal on a security."""
    signal_type: str        # e.g. 'rich_vol', 'trend_bullish', 'breakout'
    strength: float         # 0.0 - 1.0 (normalized)
    detail: str             # Human-readable: "94th percentile", "Trend 2 (S>L>M)"
    direction: str          # 'bullish', 'bearish', 'neutral'
    # --- Derivation metadata (for audit chain) ---
    metric_name: str = ""               # e.g. "IV Percentile", "Z-Score"
    metric_value: float | None = None   # raw metric value (e.g. 94.0, 2.5)
    threshold_value: float | None = None  # threshold used for gate check
    strength_formula: str = ""          # human-readable formula string


@dataclass
class StrategyRec:
    """A strategy recommendation with fitness score and reasons."""
    strategy_name: str      # 'Covered Call', 'Iron Condor', etc.
    strategy_id: int        # Maps to strategy_map.csv for Options Builder handoff
    fit_score: float        # 0-100 (normalized)
    reasons: list[str]      # ["Rich vol (94th pctl)", "Confirmed uptrend"]
    signal_contributions: list = field(default_factory=list)
    raw_score: float = 0.0  # Pre-sigmoid sum for auditability


@dataclass
class SecurityProfile:
    """Complete scan result for one security."""
    ticker: str
    name: str
    # --- Vol metrics (from Query A) ---
    iv: float | None = None
    iv_pctl: float | None = None
    rv: float | None = None
    spread: float | None = None         # iv - rv
    skew: float | None = None
    skew_pctl: float | None = None
    z_score: float | None = None
    iv_1m: float | None = None
    iv_6m: float | None = None
    term: float | None = None           # iv_1m - iv_6m
    rr_pctl: float | None = None
    bf_pctl: float | None = None
    px: float | None = None
    # --- Technical metrics (from Query B) ---
    sma_s: float | None = None
    sma_m: float | None = None
    sma_l: float | None = None
    trend_state: int | None = None      # 1-6
    breakouts: dict = field(default_factory=dict)
    # --- Computed results ---
    signals: list[Signal] = field(default_factory=list)
    strategy_recs: list[StrategyRec] = field(default_factory=list)
    composite_score: float = 0.0
    composite_breakdown: dict = field(default_factory=dict)
