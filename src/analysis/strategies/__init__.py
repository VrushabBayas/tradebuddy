"""
Trading strategies module.

Implements different trading strategies: Support/Resistance, EMA Crossover, and Combined.
"""

from .base_strategy import BaseStrategy
from .support_resistance import SupportResistanceStrategy
from .ema_crossover import EMACrossoverStrategy
from .combined import CombinedStrategy

__all__ = [
    "BaseStrategy",
    "SupportResistanceStrategy", 
    "EMACrossoverStrategy",
    "CombinedStrategy"
]