"""
Abstract interface for AI models in TradeBuddy.

Defines the contract that all AI models must implement following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import structlog

from src.core.models import (
    MarketData,
    AnalysisResult,
    StrategyType,
    SessionConfig,
)
from src.core.exceptions import DataValidationError

logger = structlog.get_logger(__name__)


class AIModelInterface(ABC):
    """
    Abstract interface for AI models.
    
    Following SOLID principles:
    - Single Responsibility: Define AI model contract
    - Open/Closed: Open for extension, closed for modification
    - Liskov Substitution: All implementations must be substitutable
    - Interface Segregation: Specific contract for AI analysis
    - Dependency Inversion: High-level modules depend on this abstraction
    """

    @abstractmethod
    async def analyze_market(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy: StrategyType,
        session_config: Optional[SessionConfig] = None,
    ) -> AnalysisResult:
        """
        Analyze market data and generate trading signals.

        Args:
            market_data: Market data to analyze
            technical_analysis: Technical analysis results
            strategy: Trading strategy to apply
            session_config: Optional session configuration

        Returns:
            AnalysisResult with AI analysis and signals

        Raises:
            DataValidationError: Invalid input data
            APIConnectionError: AI service connection error
            APITimeoutError: Request timeout
        """
        pass

    @abstractmethod
    async def check_model_health(self) -> bool:
        """
        Check if the AI model is available and healthy.

        Returns:
            True if model is available and healthy
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close AI model connection and cleanup resources."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the AI model name/identifier."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Get the AI model version."""
        pass


def validate_ai_inputs(
    market_data: MarketData,
    technical_analysis: Dict[str, Any],
    strategy: StrategyType,
) -> None:
    """
    Validate inputs for AI analysis (Fail Fast principle).

    Args:
        market_data: Market data to validate
        technical_analysis: Technical analysis data
        strategy: Trading strategy

    Raises:
        DataValidationError: If inputs are invalid
    """
    if market_data is None:
        raise DataValidationError("Market data cannot be None")

    if not market_data.ohlcv_data:
        raise DataValidationError("No OHLCV data provided for analysis")

    if market_data.current_price <= 0:
        raise DataValidationError("Invalid current price in market data")

    if technical_analysis is None:
        raise DataValidationError("Technical analysis cannot be None")

    if strategy is None:
        raise DataValidationError("Strategy cannot be None")

    logger.debug(
        "AI input validation passed",
        symbol=market_data.symbol,
        strategy=strategy.value,
        ohlcv_count=len(market_data.ohlcv_data),
        current_price=market_data.current_price,
    )


async def handle_ai_errors(func):
    """
    Generic error handler for AI operations.

    Args:
        func: Async function to execute

    Raises:
        Original exceptions for proper error handling
    """
    try:
        return await func()
    except Exception as e:
        logger.error("AI operation failed", error=str(e), function=func.__name__)
        raise