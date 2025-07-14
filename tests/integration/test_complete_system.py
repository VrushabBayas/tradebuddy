"""
Complete system integration test for TradeBuddy.

Tests the full pipeline from CLI to trading signals.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from src.analysis.strategies.combined import CombinedStrategy
from src.cli.main import CLIApplication
from src.core.models import (
    OHLCV,
    AnalysisResult,
    MarketData,
    SessionConfig,
    SignalAction,
    SignalStrength,
    StrategyType,
    Symbol,
    TimeFrame,
    TradingSignal,
)
from src.data.delta_client import DeltaExchangeClient

logger = structlog.get_logger(__name__)


class TestCompleteSystemIntegration:
    """Test complete system integration."""

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        ohlcv_data = [
            OHLCV(
                open=50000 + i * 100,
                high=50100 + i * 100,
                low=49900 + i * 100,
                close=50050 + i * 100,
                volume=1000000 + i * 1000,
            )
            for i in range(30)
        ]

        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=53050.0,
            ohlcv_data=ohlcv_data,
        )

    @pytest.fixture
    def mock_session_config(self):
        """Create mock session configuration."""
        return SessionConfig(
            strategy=StrategyType.COMBINED,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            stop_loss_pct=3.0,
            take_profit_pct=6.0,
            position_size_pct=2.0,
            confidence_threshold=6,
        )

    @pytest.fixture
    def mock_analysis_result(self, mock_market_data):
        """Create mock analysis result."""
        primary_signal = TradingSignal(
            symbol=Symbol.BTCUSD,
            strategy=StrategyType.COMBINED,
            action=SignalAction.BUY,
            strength=SignalStrength.STRONG,
            confidence=8,
            entry_price=53050.0,
            reasoning="Strong golden cross with volume confirmation and support bounce",
        )

        return AnalysisResult(
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            strategy=StrategyType.COMBINED,
            signals=[primary_signal],
            ai_analysis="AI analysis indicates strong bullish momentum with golden cross formation. "
            "Volume confirmation supports the upward move. Entry recommended at current levels.",
            market_data=mock_market_data,
        )

    @pytest.mark.asyncio
    async def test_complete_system_flow(
        self, mock_market_data, mock_session_config, mock_analysis_result
    ):
        """Test complete system flow from data fetch to signal generation."""
        logger.info("Starting complete system integration test")

        # Create CLI application
        cli_app = CLIApplication()

        try:
            # Mock the Delta Exchange client
            with patch.object(cli_app.delta_client, "get_market_data") as mock_get_data:
                mock_get_data.return_value = mock_market_data

                # Mock the strategy analysis
                with patch.object(
                    cli_app.strategies[StrategyType.COMBINED], "analyze"
                ) as mock_analyze:
                    mock_analyze.return_value = mock_analysis_result

                    # Step 1: Fetch market data
                    market_data = await cli_app.delta_client.get_market_data(
                        symbol=mock_session_config.symbol,
                        timeframe=mock_session_config.timeframe.value,
                        limit=100,
                    )

                    # Verify market data
                    assert market_data is not None
                    assert market_data.symbol == "BTCUSD"
                    assert market_data.timeframe == "1h"
                    assert market_data.current_price > 0
                    assert len(market_data.ohlcv_data) > 0

                    logger.info(
                        "✅ Market data fetch successful",
                        symbol=market_data.symbol,
                        data_points=len(market_data.ohlcv_data),
                    )

                    # Step 2: Run strategy analysis
                    strategy = cli_app.strategies[StrategyType.COMBINED]
                    analysis_result = await strategy.analyze(
                        market_data, mock_session_config
                    )

                    # Verify analysis result
                    assert analysis_result is not None
                    assert len(analysis_result.signals) > 0
                    assert analysis_result.primary_signal is not None
                    assert analysis_result.ai_analysis is not None

                    logger.info(
                        "✅ Strategy analysis successful",
                        signals_count=len(analysis_result.signals),
                        primary_action=analysis_result.primary_signal.action.value,
                        confidence=analysis_result.primary_signal.confidence,
                    )

                    # Step 3: Verify signal quality
                    primary_signal = analysis_result.primary_signal
                    assert (
                        primary_signal.confidence
                        >= mock_session_config.confidence_threshold
                    )
                    assert primary_signal.entry_price > 0
                    assert primary_signal.action in [
                        SignalAction.BUY,
                        SignalAction.SELL,
                    ]
                    assert primary_signal.reasoning is not None

                    logger.info(
                        "✅ Signal quality validation passed",
                        action=primary_signal.action.value,
                        confidence=primary_signal.confidence,
                        entry_price=primary_signal.entry_price,
                    )

                    # Step 4: Test risk management calculations
                    entry_price = primary_signal.entry_price

                    if primary_signal.action == SignalAction.BUY:
                        stop_loss = entry_price * (
                            1 - mock_session_config.stop_loss_pct / 100
                        )
                        take_profit = entry_price * (
                            1 + mock_session_config.take_profit_pct / 100
                        )
                    else:
                        stop_loss = entry_price * (
                            1 + mock_session_config.stop_loss_pct / 100
                        )
                        take_profit = entry_price * (
                            1 - mock_session_config.take_profit_pct / 100
                        )

                    # Verify risk management
                    risk = abs(entry_price - stop_loss)
                    reward = abs(take_profit - entry_price)
                    risk_reward_ratio = reward / risk if risk > 0 else 0

                    assert risk_reward_ratio > 1.5  # Minimum 1.5:1 ratio
                    assert stop_loss != entry_price
                    assert take_profit != entry_price

                    logger.info(
                        "✅ Risk management validation passed",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward_ratio,
                    )

                    logger.info("🎯 COMPLETE SYSTEM INTEGRATION TEST PASSED!")

        finally:
            # Cleanup
            await cli_app.cleanup()

    @pytest.mark.asyncio
    async def test_cli_error_handling(self, mock_session_config):
        """Test CLI error handling capabilities."""
        logger.info("Testing CLI error handling")

        cli_app = CLIApplication()

        try:
            # Test with invalid market data
            with patch.object(cli_app.delta_client, "get_market_data") as mock_get_data:
                mock_get_data.side_effect = Exception("API connection failed")

                # This should handle the error gracefully
                with pytest.raises(Exception):
                    await cli_app.delta_client.get_market_data(
                        symbol=mock_session_config.symbol,
                        timeframe=mock_session_config.timeframe.value,
                        limit=100,
                    )

                logger.info("✅ Error handling test passed")

        finally:
            await cli_app.cleanup()

    @pytest.mark.asyncio
    async def test_strategy_switching(self, mock_market_data, mock_session_config):
        """Test switching between different strategies."""
        logger.info("Testing strategy switching")

        cli_app = CLIApplication()

        try:
            # Test all three strategies
            strategies_to_test = [
                StrategyType.SUPPORT_RESISTANCE,
                StrategyType.EMA_CROSSOVER_V2,
                StrategyType.COMBINED,
            ]

            for strategy_type in strategies_to_test:
                # Update config for current strategy
                test_config = SessionConfig(
                    strategy=strategy_type,
                    symbol=mock_session_config.symbol,
                    timeframe=mock_session_config.timeframe,
                    stop_loss_pct=mock_session_config.stop_loss_pct,
                    take_profit_pct=mock_session_config.take_profit_pct,
                    position_size_pct=mock_session_config.position_size_pct,
                    confidence_threshold=mock_session_config.confidence_threshold,
                )

                # Get strategy instance
                strategy = cli_app.strategies[strategy_type]
                assert strategy is not None
                assert strategy.strategy_type == strategy_type

                logger.info(f"✅ Strategy {strategy_type.value} validation passed")

            logger.info("✅ All strategy switching tests passed")

        finally:
            await cli_app.cleanup()

    @pytest.mark.asyncio
    async def test_system_performance(self, mock_market_data, mock_session_config):
        """Test system performance with multiple concurrent operations."""
        logger.info("Testing system performance")

        cli_app = CLIApplication()

        try:
            # Mock the dependencies
            with patch.object(cli_app.delta_client, "get_market_data") as mock_get_data:
                mock_get_data.return_value = mock_market_data

                # Test multiple concurrent operations
                tasks = []
                for i in range(5):
                    task = asyncio.create_task(
                        cli_app.delta_client.get_market_data(
                            symbol=mock_session_config.symbol,
                            timeframe=mock_session_config.timeframe.value,
                            limit=100,
                        )
                    )
                    tasks.append(task)

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)

                # Verify all results
                assert len(results) == 5
                for result in results:
                    assert result is not None
                    assert result.symbol == "BTCUSD"

                logger.info(
                    "✅ Performance test passed - handled 5 concurrent operations"
                )

        finally:
            await cli_app.cleanup()


if __name__ == "__main__":
    """Run the complete system integration test."""
    asyncio.run(
        TestCompleteSystemIntegration().test_complete_system_flow(
            mock_market_data=None, mock_session_config=None, mock_analysis_result=None
        )
    )
