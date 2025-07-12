"""
Base strategy class for TradeBuddy trading strategies.

Provides common functionality and interface for all trading strategies.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import structlog

from src.analysis.indicators import TechnicalIndicators
from src.analysis.ollama_client import OllamaClient
from src.core.exceptions import DataValidationError, StrategyError
from src.core.models import AnalysisResult, MarketData, SessionConfig, StrategyType

logger = structlog.get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Provides common functionality including:
    - Technical indicators calculation
    - AI analysis integration
    - Data validation
    - Error handling
    - Performance monitoring
    """

    def __init__(self):
        """Initialize base strategy with required components."""
        self.indicators = TechnicalIndicators()
        self.ollama_client = OllamaClient()
        self.strategy_type = StrategyType.COMBINED  # Default, override in subclasses

        logger.debug("Strategy initialized", strategy_type=self.strategy_type.value)

    @abstractmethod
    async def analyze(
        self, market_data: MarketData, session_config: SessionConfig
    ) -> AnalysisResult:
        """
        Analyze market data and generate trading signals.

        Args:
            market_data: Market data to analyze
            session_config: Session configuration parameters

        Returns:
            AnalysisResult with signals and analysis

        Raises:
            DataValidationError: Invalid input data
            StrategyError: Strategy execution error
        """
        raise NotImplementedError("Subclasses must implement analyze method")

    async def analyze_for_backtesting(
        self, market_data: MarketData, session_config: SessionConfig
    ) -> AnalysisResult:
        """
        Analyze market data for backtesting without AI analysis.

        This method provides fast technical analysis for backtesting by
        skipping expensive AI calls and using pure technical indicators
        for signal generation.

        Args:
            market_data: Market data to analyze
            session_config: Session configuration parameters

        Returns:
            AnalysisResult with signals based on technical analysis

        Raises:
            DataValidationError: Invalid input data
            StrategyError: Strategy execution error
        """
        logger.debug(
            "Starting backtesting analysis",
            strategy=self.strategy_type.value,
            symbol=market_data.symbol,
            data_points=len(market_data.ohlcv_data),
        )

        try:
            # Validate inputs
            self._validate_market_data(market_data)
            self._validate_session_config(session_config)

            # Calculate technical analysis
            technical_analysis = await self._calculate_technical_analysis(market_data)

            # Generate signals using strategy-specific logic (without AI)
            analysis_result = await self._generate_backtesting_signals(
                market_data, technical_analysis, session_config
            )

            # Filter signals by confidence threshold
            filtered_result = self._filter_signals_by_confidence(
                analysis_result, session_config.confidence_threshold
            )

            logger.debug(
                "Backtesting analysis completed",
                strategy=self.strategy_type.value,
                symbol=market_data.symbol,
                signals_count=len(filtered_result.signals),
            )

            return filtered_result

        except Exception as e:
            logger.error(
                "Backtesting analysis failed",
                strategy=self.strategy_type.value,
                symbol=market_data.symbol,
                error=str(e),
            )
            raise StrategyError(f"Backtesting analysis failed: {str(e)}")

    @abstractmethod
    async def _generate_backtesting_signals(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        session_config: SessionConfig,
    ) -> AnalysisResult:
        """
        Generate trading signals for backtesting using technical analysis.

        This method should be implemented by each strategy to provide
        fast signal generation without AI analysis.

        Args:
            market_data: Market data context
            technical_analysis: Technical analysis results
            session_config: Session configuration

        Returns:
            AnalysisResult with strategy-specific signals
        """
        raise NotImplementedError("Subclasses must implement _generate_backtesting_signals")

    def _validate_market_data(self, market_data: MarketData) -> None:
        """
        Validate market data for analysis.

        Args:
            market_data: Market data to validate

        Raises:
            DataValidationError: If market data is invalid
        """
        if market_data is None:
            raise DataValidationError("Market data cannot be None")

        if not market_data.ohlcv_data:
            raise DataValidationError("No OHLCV data provided for analysis")

        if market_data.current_price <= 0:
            raise DataValidationError("Invalid current price in market data")

        # Check for minimum data requirements
        min_periods = self._get_minimum_periods()
        if len(market_data.ohlcv_data) < min_periods:
            raise DataValidationError(
                f"Insufficient data: {len(market_data.ohlcv_data)} periods provided, "
                f"minimum {min_periods} required for {self.strategy_type.value} strategy"
            )

    def _validate_session_config(self, session_config: SessionConfig) -> None:
        """
        Validate session configuration.

        Args:
            session_config: Session configuration to validate

        Raises:
            DataValidationError: If session config is invalid
        """
        if session_config is None:
            raise DataValidationError("Session config cannot be None")

        if session_config.strategy != self.strategy_type:
            logger.warning(
                "Strategy type mismatch",
                expected=self.strategy_type.value,
                provided=session_config.strategy.value,
            )

    def _get_minimum_periods(self) -> int:
        """
        Get minimum number of periods required for this strategy.

        Returns:
            Minimum number of OHLCV periods needed
        """
        # Base requirement for EMA calculations
        return 20

    async def _calculate_technical_analysis(
        self, market_data: MarketData
    ) -> Dict[str, Any]:
        """
        Calculate technical analysis for the strategy.

        Args:
            market_data: Market data to analyze

        Returns:
            Dictionary with technical analysis results
        """
        logger.debug(
            "Calculating technical analysis",
            strategy=self.strategy_type.value,
            data_points=len(market_data.ohlcv_data),
        )

        start_time = time.time()

        try:
            # Perform comprehensive technical analysis
            analysis = self.indicators.comprehensive_analysis(market_data.ohlcv_data)

            calculation_time = time.time() - start_time

            logger.debug(
                "Technical analysis completed",
                strategy=self.strategy_type.value,
                calculation_time=calculation_time,
                sentiment=analysis.get("overall_sentiment"),
                confidence=analysis.get("confidence_score"),
            )

            return analysis

        except Exception as e:
            logger.error(
                "Technical analysis failed",
                strategy=self.strategy_type.value,
                error=str(e),
            )
            raise StrategyError(f"Technical analysis failed: {str(e)}")

    async def _generate_ai_analysis(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        session_config: SessionConfig = None,
    ) -> AnalysisResult:
        """
        Generate AI-powered analysis using Ollama.

        Args:
            market_data: Market data context
            technical_analysis: Technical analysis results

        Returns:
            AnalysisResult from AI analysis
        """
        logger.debug(
            "Generating AI analysis",
            strategy=self.strategy_type.value,
            symbol=market_data.symbol,
        )

        start_time = time.time()

        try:
            # Use Ollama client for AI analysis
            analysis_result = await self.ollama_client.analyze_market(
                market_data=market_data,
                technical_analysis=technical_analysis,
                strategy=self.strategy_type,
                session_config=session_config,
            )

            ai_time = time.time() - start_time

            logger.info(
                "AI analysis completed",
                strategy=self.strategy_type.value,
                symbol=market_data.symbol,
                ai_time=ai_time,
                signals_count=len(analysis_result.signals),
                primary_action=analysis_result.primary_signal.action.value
                if analysis_result.primary_signal
                else None,
                confidence=analysis_result.primary_signal.confidence
                if analysis_result.primary_signal
                else None,
            )

            return analysis_result

        except Exception as e:
            logger.error(
                "AI analysis failed",
                strategy=self.strategy_type.value,
                symbol=market_data.symbol,
                error=str(e),
            )
            raise StrategyError(f"AI analysis failed: {str(e)}")

    def _filter_signals_by_confidence(
        self, analysis_result: AnalysisResult, min_confidence: int
    ) -> AnalysisResult:
        """
        Filter signals based on minimum confidence threshold.

        Args:
            analysis_result: Original analysis result
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered AnalysisResult
        """
        original_count = len(analysis_result.signals)

        # Filter signals by confidence
        filtered_signals = [
            signal
            for signal in analysis_result.signals
            if signal.confidence >= min_confidence
        ]

        # Update analysis result
        analysis_result.signals = filtered_signals

        logger.debug(
            "Signals filtered by confidence",
            strategy=self.strategy_type.value,
            min_confidence=min_confidence,
            original_count=original_count,
            filtered_count=len(filtered_signals),
        )

        return analysis_result

    async def close(self) -> None:
        """Close strategy resources."""
        try:
            await self.ollama_client.close()
            logger.debug("Strategy resources closed", strategy=self.strategy_type.value)
        except Exception as e:
            logger.error("Error closing strategy resources", error=str(e))

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
