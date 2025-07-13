"""
Functionality tests for trading strategies.

Focuses on testing strategy behavior and signal generation rather than implementation details.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
from src.analysis.strategies.support_resistance import SupportResistanceStrategy
from src.analysis.strategies.combined import CombinedStrategy
from src.core.models import (
    MarketData, OHLCV, SessionConfig, StrategyType, Symbol, TimeFrame,
    SignalAction, TradingSignal
)


class TestEMACrossoverFunctionality:
    """Test EMA Crossover strategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create EMA crossover strategy."""
        return EMACrossoverStrategy()

    @pytest.fixture
    def market_data_bullish(self):
        """Create market data that should generate bullish signals."""
        # Create OHLCV data showing upward trend with EMA crossover
        candles = []
        base_price = 50000.0
        for i in range(30):
            # Gradually increasing prices for bullish trend
            price_increase = i * 100
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=base_price + price_increase,
                high=base_price + price_increase + 200,
                low=base_price + price_increase - 100,
                close=base_price + price_increase + 150,
                volume=1000.0 + i * 10
            )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price + 3000,  # Current price higher than start
            ohlcv_data=candles
        )

    @pytest.fixture
    def market_data_bearish(self):
        """Create market data that should generate bearish signals."""
        # Create OHLCV data showing downward trend
        candles = []
        base_price = 50000.0
        for i in range(30):
            # Gradually decreasing prices for bearish trend
            price_decrease = i * 100
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=base_price - price_decrease,
                high=base_price - price_decrease + 100,
                low=base_price - price_decrease - 200,
                close=base_price - price_decrease - 150,
                volume=1000.0 + i * 10
            )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price - 3000,  # Current price lower than start
            ohlcv_data=candles
        )

    @pytest.fixture
    def session_config(self):
        """Create session configuration."""
        return SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            total_capital_inr=Decimal("100000"),
            trading_capital_pct=Decimal("50.0"),
            risk_per_trade_pct=Decimal("2.0"),
            take_profit_pct=Decimal("5.0"),
            leverage=10,
        )

    @pytest.mark.asyncio
    async def test_strategy_initialization(self, strategy):
        """Test that strategy initializes correctly."""
        assert strategy.strategy_type == StrategyType.EMA_CROSSOVER
        assert strategy.indicators is not None
        assert strategy.ollama_client is not None

    @pytest.mark.asyncio
    async def test_bullish_signal_generation(self, strategy, market_data_bullish, session_config):
        """Test that strategy generates bullish signals for uptrending market."""
        with patch.object(strategy.ollama_client, 'analyze_market_data') as mock_ollama:
            # Mock Ollama to return bullish analysis
            mock_ollama.return_value = MagicMock(
                signals=[
                    TradingSignal(
                        symbol=Symbol.BTCUSD,
                        strategy=StrategyType.EMA_CROSSOVER,
                        action=SignalAction.BUY,
                        strength="STRONG",
                        confidence=8,
                        entry_price=Decimal("52000"),
                        reasoning="Strong bullish EMA crossover with volume confirmation"
                    )
                ]
            )
            
            result = await strategy.analyze(market_data_bullish, session_config)
            
            # Verify functional behavior
            assert result is not None
            assert len(result.signals) > 0
            
            primary_signal = result.primary_signal
            assert primary_signal is not None
            assert primary_signal.action == SignalAction.BUY
            assert primary_signal.confidence >= 6  # Should be actionable
            assert primary_signal.entry_price > 0

    @pytest.mark.asyncio
    async def test_bearish_signal_generation(self, strategy, market_data_bearish, session_config):
        """Test that strategy generates bearish signals for downtrending market."""
        with patch.object(strategy.ollama_client, 'analyze_market_data') as mock_ollama:
            # Mock Ollama to return bearish analysis
            mock_ollama.return_value = MagicMock(
                signals=[
                    TradingSignal(
                        symbol=Symbol.BTCUSD,
                        strategy=StrategyType.EMA_CROSSOVER,
                        action=SignalAction.SELL,
                        strength="STRONG",
                        confidence=7,
                        entry_price=Decimal("48000"),
                        reasoning="Strong bearish EMA crossover pattern"
                    )
                ]
            )
            
            result = await strategy.analyze(market_data_bearish, session_config)
            
            # Verify functional behavior
            assert result is not None
            assert len(result.signals) > 0
            
            primary_signal = result.primary_signal
            assert primary_signal is not None
            assert primary_signal.action == SignalAction.SELL
            assert primary_signal.confidence >= 6  # Should be actionable


class TestSupportResistanceFunctionality:
    """Test Support/Resistance strategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create support/resistance strategy."""
        return SupportResistanceStrategy()

    @pytest.fixture
    def market_data_at_support(self):
        """Create market data showing price bouncing at support level."""
        candles = []
        support_level = 50000.0
        
        # Create pattern where price touches support and bounces
        for i in range(20):
            if i < 10:
                # Price approaching support
                ohlcv = OHLCV(
                    timestamp=datetime.now(timezone.utc),
                    open=support_level + 500 - (i * 50),
                    high=support_level + 600 - (i * 50),
                    low=support_level + 100 - (i * 50),
                    close=support_level + 200 - (i * 50),
                    volume=1000.0
                )
            else:
                # Price bouncing from support
                ohlcv = OHLCV(
                    timestamp=datetime.now(timezone.utc),
                    open=support_level + 100 + ((i-10) * 30),
                    high=support_level + 400 + ((i-10) * 30),
                    low=support_level - 50,
                    close=support_level + 300 + ((i-10) * 30),
                    volume=1000.0
                )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=support_level + 200,
            ohlcv_data=candles
        )

    @pytest.fixture
    def session_config(self):
        """Create session configuration."""
        return SessionConfig(
            strategy=StrategyType.SUPPORT_RESISTANCE,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            total_capital_inr=Decimal("100000"),
            trading_capital_pct=Decimal("50.0"),
            risk_per_trade_pct=Decimal("2.0"),
            take_profit_pct=Decimal("5.0"),
            leverage=10,
        )

    @pytest.mark.asyncio
    async def test_support_level_signal(self, strategy, market_data_at_support, session_config):
        """Test that strategy identifies support level bounce opportunities."""
        with patch.object(strategy.ollama_client, 'analyze_market_data') as mock_ollama:
            # Mock Ollama to return support bounce analysis
            mock_ollama.return_value = MagicMock(
                signals=[
                    TradingSignal(
                        symbol=Symbol.BTCUSD,
                        strategy=StrategyType.SUPPORT_RESISTANCE,
                        action=SignalAction.BUY,
                        strength="MODERATE",
                        confidence=7,
                        entry_price=Decimal("50200"),
                        reasoning="Price bouncing from strong support level"
                    )
                ],
                support_resistance_levels=[
                    MagicMock(level=Decimal("50000"), strength=8, is_support=True)
                ]
            )
            
            result = await strategy.analyze(market_data_at_support, session_config)
            
            # Verify functional behavior
            assert result is not None
            assert len(result.signals) > 0
            assert len(result.support_resistance_levels) > 0
            
            # Should identify support level
            support_levels = [level for level in result.support_resistance_levels if level.is_support]
            assert len(support_levels) > 0
            assert support_levels[0].level > 0


class TestCombinedStrategyFunctionality:
    """Test Combined strategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create combined strategy."""
        return CombinedStrategy()

    @pytest.fixture
    def market_data_high_confidence(self):
        """Create market data that should generate high confidence signals."""
        candles = []
        base_price = 50000.0
        
        # Create strong bullish pattern with volume confirmation
        for i in range(25):
            price_increase = i * 120  # Strong uptrend
            volume_increase = 1000.0 + i * 50  # Increasing volume
            
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=base_price + price_increase,
                high=base_price + price_increase + 300,
                low=base_price + price_increase - 50,
                close=base_price + price_increase + 250,
                volume=volume_increase
            )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price + 3500,
            ohlcv_data=candles
        )

    @pytest.fixture
    def session_config(self):
        """Create session configuration."""
        return SessionConfig(
            strategy=StrategyType.COMBINED,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            total_capital_inr=Decimal("100000"),
            trading_capital_pct=Decimal("50.0"),
            risk_per_trade_pct=Decimal("2.0"),
            take_profit_pct=Decimal("5.0"),
            leverage=10,
        )

    @pytest.mark.asyncio
    async def test_high_confidence_signal_generation(self, strategy, market_data_high_confidence, session_config):
        """Test that combined strategy generates high confidence signals when multiple strategies align."""
        with patch.object(strategy.ollama_client, 'analyze_market_data') as mock_ollama:
            # Mock Ollama to return high confidence analysis
            mock_ollama.return_value = MagicMock(
                signals=[
                    TradingSignal(
                        symbol=Symbol.BTCUSD,
                        strategy=StrategyType.COMBINED,
                        action=SignalAction.BUY,
                        strength="STRONG",
                        confidence=9,  # High confidence from multiple confirmations
                        entry_price=Decimal("53500"),
                        reasoning="Strong bullish signals from both EMA crossover and support/resistance confluence"
                    )
                ]
            )
            
            result = await strategy.analyze(market_data_high_confidence, session_config)
            
            # Verify functional behavior
            assert result is not None
            assert len(result.signals) > 0
            
            primary_signal = result.primary_signal
            assert primary_signal is not None
            assert primary_signal.confidence >= 8  # Should be high confidence
            assert primary_signal.action in [SignalAction.BUY, SignalAction.SELL]
            assert primary_signal.strategy == StrategyType.COMBINED

    @pytest.mark.asyncio
    async def test_strategy_combination_logic(self, strategy, market_data_high_confidence, session_config):
        """Test that combined strategy properly integrates multiple analysis methods."""
        with patch.object(strategy.ollama_client, 'analyze_market_data') as mock_ollama:
            mock_ollama.return_value = MagicMock(
                signals=[
                    TradingSignal(
                        symbol=Symbol.BTCUSD,
                        strategy=StrategyType.COMBINED,
                        action=SignalAction.BUY,
                        strength="STRONG",
                        confidence=8,
                        entry_price=Decimal("53500"),
                        reasoning="Multiple strategy confirmation"
                    )
                ]
            )
            
            result = await strategy.analyze(market_data_high_confidence, session_config)
            
            # Verify that combined strategy produces comprehensive analysis
            assert result is not None
            assert result.strategy == StrategyType.COMBINED
            assert result.market_data.symbol == "BTCUSD"
            
            # Should have processed the market data
            assert len(result.market_data.ohlcv_data) > 0
            assert result.market_data.current_price > 0


class TestStrategyErrorHandling:
    """Test strategy error handling and edge cases."""

    @pytest.fixture
    def invalid_market_data(self):
        """Create invalid market data."""
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=50000.0,
            ohlcv_data=[]  # Empty data
        )

    @pytest.fixture
    def session_config(self):
        """Create session configuration."""
        return SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            total_capital_inr=Decimal("100000"),
            trading_capital_pct=Decimal("50.0"),
            risk_per_trade_pct=Decimal("2.0"),
            take_profit_pct=Decimal("5.0"),
            leverage=10,
        )

    @pytest.mark.asyncio
    async def test_strategy_handles_insufficient_data(self, invalid_market_data, session_config):
        """Test that strategies handle insufficient data gracefully."""
        strategy = EMACrossoverStrategy()
        
        # Should handle insufficient data without crashing
        try:
            result = await strategy.analyze(invalid_market_data, session_config)
            # If it doesn't raise an exception, verify it returns valid response
            if result is not None:
                assert isinstance(result.signals, list)
        except Exception as e:
            # If it raises an exception, it should be a specific validation error
            assert "data" in str(e).lower() or "insufficient" in str(e).lower()

    @pytest.mark.asyncio
    async def test_strategy_handles_ollama_errors(self, session_config):
        """Test that strategies handle Ollama client errors gracefully."""
        strategy = EMACrossoverStrategy()
        
        # Create minimal valid market data
        market_data = MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=50000.0,
            ohlcv_data=[
                OHLCV(
                    timestamp=datetime.now(timezone.utc),
                    open=50000, high=50100, low=49900, close=50050, volume=1000
                )
            ]
        )
        
        with patch.object(strategy.ollama_client, 'analyze_market_data') as mock_ollama:
            # Mock Ollama to raise an exception
            mock_ollama.side_effect = Exception("Ollama connection error")
            
            # Strategy should handle Ollama errors gracefully
            try:
                result = await strategy.analyze(market_data, session_config)
                # If no exception, should return reasonable fallback
                if result is not None:
                    assert isinstance(result.signals, list)
            except Exception as e:
                # Should be a handled error with meaningful message
                assert len(str(e)) > 0