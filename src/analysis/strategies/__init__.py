"""
Trading strategies module.

Implements different trading strategies: Support/Resistance, EMA Crossover, and Combined.
"""

from .base_strategy import BaseStrategy
from .combined import CombinedStrategy
from .ema_crossover import EMACrossoverStrategy
from .support_resistance import SupportResistanceStrategy

__all__ = [
    "BaseStrategy",
    "SupportResistanceStrategy",
    "EMACrossoverStrategy",
    "CombinedStrategy",
]
