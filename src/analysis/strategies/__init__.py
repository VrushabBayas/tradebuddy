"""
Trading strategies module.

Implements different trading strategies: Support/Resistance, EMA Crossover V2, and Combined.
"""

from .base_strategy import BaseStrategy
from .combined import CombinedStrategy
from .ema_crossover_v2 import EMACrossoverV2Strategy
from .support_resistance import SupportResistanceStrategy

__all__ = [
    "BaseStrategy",
    "SupportResistanceStrategy", 
    "EMACrossoverV2Strategy",
    "CombinedStrategy",
]
