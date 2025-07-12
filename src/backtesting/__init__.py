"""
TradeBuddy Backtesting Module.

Comprehensive backtesting system for validating trading strategies against
historical data with professional-grade reporting and analysis.
"""

from src.backtesting.engine import BacktestEngine
from src.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
    PerformanceMetrics,
)
from src.backtesting.portfolio import Portfolio

__all__ = [
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResult",
    "BacktestTrade",
    "EquityPoint",
    "PerformanceMetrics",
    "Portfolio",
]