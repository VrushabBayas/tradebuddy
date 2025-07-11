"""
Unit tests for trading strategies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.analysis.strategies import (
    BaseStrategy,
    SupportResistanceStrategy,
    EMACrossoverStrategy,
    CombinedStrategy
)
from src.core.models import (
    OHLCV, 
    MarketData, 
    AnalysisResult, 
    TradingSignal,
    StrategyType,
    SignalAction,
    SessionConfig
)
from src.core.exceptions import DataValidationError, StrategyError


class TestBaseStrategy:
    """Test cases for base strategy class."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        for i in range(20):
            timestamp = base_time.replace(hour=i)
            base_price = 50000 + (i * 100)
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=base_price,
                high=base_price + 50,
                low=base_price - 30,
                close=base_price + 25,
                volume=1000 + (i * 10)
            )
            ohlcv_data.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=52000.0,
            ohlcv_data=ohlcv_data
        )

    @pytest.fixture
    def session_config(self):
        """Create sample session configuration."""
        return SessionConfig(
            strategy=StrategyType.SUPPORT_RESISTANCE,
            symbol="BTCUSDT",
            timeframe="1h",
            stop_loss_pct=2.5,
            take_profit_pct=5.0,
            position_size_pct=2.0,
            confidence_threshold=6
        )

    def test_base_strategy_initialization(self):
        """Test base strategy initialization."""
        strategy = BaseStrategy()
        
        assert strategy.indicators is not None
        assert strategy.ollama_client is not None
        assert strategy.strategy_type == StrategyType.COMBINED  # Default

    @pytest.mark.asyncio
    async def test_base_strategy_analyze_not_implemented(self, sample_market_data, session_config):
        """Test that base strategy analyze method raises NotImplementedError."""
        strategy = BaseStrategy()
        
        with pytest.raises(NotImplementedError):
            await strategy.analyze(sample_market_data, session_config)

    def test_validate_market_data_valid(self, sample_market_data):
        """Test validation of valid market data."""
        strategy = BaseStrategy()
        
        # Should not raise any exception
        strategy._validate_market_data(sample_market_data)

    def test_validate_market_data_none(self):
        """Test validation of None market data."""
        strategy = BaseStrategy()
        
        with pytest.raises(DataValidationError, match="Market data cannot be None"):
            strategy._validate_market_data(None)

    def test_validate_market_data_no_ohlcv(self):
        """Test validation of market data with no OHLCV data."""
        strategy = BaseStrategy()
        
        invalid_data = MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=50000.0,
            ohlcv_data=[]
        )
        
        with pytest.raises(DataValidationError, match="No OHLCV data provided"):
            strategy._validate_market_data(invalid_data)

    def test_validate_session_config_valid(self, session_config):
        """Test validation of valid session config."""
        strategy = BaseStrategy()
        
        # Should not raise any exception
        strategy._validate_session_config(session_config)

    def test_validate_session_config_none(self):
        """Test validation of None session config."""
        strategy = BaseStrategy()
        
        with pytest.raises(DataValidationError, match="Session config cannot be None"):
            strategy._validate_session_config(None)


class TestSupportResistanceStrategy:
    """Test cases for Support/Resistance strategy."""

    @pytest.fixture
    def strategy(self):
        """Create Support/Resistance strategy instance."""
        return SupportResistanceStrategy()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data with clear S/R levels."""
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        # Create data with support around 49500 and resistance around 52000
        prices = [50000, 49800, 49500, 49600, 50200, 51000, 51800, 52000, 51900, 51200,
                  50800, 50200, 49800, 49500, 49700, 50500, 51200, 51800, 52000, 51800]
        
        for i, price in enumerate(prices):
            timestamp = base_time.replace(hour=i)
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=price,
                high=price + 100,
                low=price - 100,
                close=price + 50,
                volume=1000 + (i * 10)
            )
            ohlcv_data.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=51000.0,
            ohlcv_data=ohlcv_data
        )

    @pytest.fixture
    def session_config(self):
        """Create session config for S/R strategy."""
        return SessionConfig(
            strategy=StrategyType.SUPPORT_RESISTANCE,
            symbol="BTCUSDT",
            timeframe="1h",
            confidence_threshold=6
        )

    def test_strategy_initialization(self, strategy):
        """Test S/R strategy initialization."""
        assert strategy.strategy_type == StrategyType.SUPPORT_RESISTANCE
        assert strategy.indicators is not None
        assert strategy.ollama_client is not None

    @pytest.mark.asyncio
    async def test_analyze_with_mock_ollama(self, strategy, sample_market_data, session_config):
        """Test analysis with mocked Ollama response."""
        # Mock Ollama client response
        mock_analysis_result = AnalysisResult(
            symbol="BTCUSDT",
            timeframe="1h",
            strategy=StrategyType.SUPPORT_RESISTANCE,
            market_data=sample_market_data,
            signals=[
                TradingSignal(
                    symbol="BTCUSDT",
                    strategy=StrategyType.SUPPORT_RESISTANCE,
                    action=SignalAction.BUY,
                    strength="MODERATE",
                    confidence=7,
                    entry_price=51000.0,
                    stop_loss=49500.0,
                    take_profit=52500.0,
                    reasoning="Price bouncing off strong support level"
                )
            ],
            ai_analysis="Strong support at 49500, resistance at 52000. BUY signal.",
            execution_time=2.5
        )
        
        strategy.ollama_client.analyze_market = AsyncMock(return_value=mock_analysis_result)
        
        # Run analysis
        result = await strategy.analyze(sample_market_data, session_config)
        
        # Verify results
        assert isinstance(result, AnalysisResult)
        assert result.strategy == StrategyType.SUPPORT_RESISTANCE
        assert len(result.signals) > 0
        assert result.primary_signal.action == SignalAction.BUY
        assert result.primary_signal.confidence == 7

    @pytest.mark.asyncio
    async def test_analyze_with_invalid_data(self, strategy, session_config):
        """Test analysis with invalid market data."""
        invalid_data = MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=50000.0,
            ohlcv_data=[]
        )
        
        with pytest.raises(DataValidationError):
            await strategy.analyze(invalid_data, session_config)


class TestEMACrossoverStrategy:
    """Test cases for EMA Crossover strategy."""

    @pytest.fixture
    def strategy(self):
        """Create EMA Crossover strategy instance."""
        return EMACrossoverStrategy()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data with clear EMA crossover."""
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        # Create uptrending data for golden cross
        for i in range(20):
            timestamp = base_time.replace(hour=i)
            base_price = 50000 + (i * 150)  # Strong uptrend
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=base_price,
                high=base_price + 75,
                low=base_price - 50,
                close=base_price + 50,
                volume=1500 + (i * 25)  # Increasing volume
            )
            ohlcv_data.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=53000.0,
            ohlcv_data=ohlcv_data
        )

    @pytest.fixture
    def session_config(self):
        """Create session config for EMA strategy."""
        return SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER,
            symbol="BTCUSDT",
            timeframe="1h",
            confidence_threshold=6
        )

    def test_strategy_initialization(self, strategy):
        """Test EMA strategy initialization."""
        assert strategy.strategy_type == StrategyType.EMA_CROSSOVER
        assert strategy.indicators is not None
        assert strategy.ollama_client is not None

    @pytest.mark.asyncio
    async def test_analyze_golden_cross(self, strategy, sample_market_data, session_config):
        """Test analysis with golden cross scenario."""
        # Mock Ollama response for golden cross
        mock_analysis_result = AnalysisResult(
            symbol="BTCUSDT",
            timeframe="1h",
            strategy=StrategyType.EMA_CROSSOVER,
            market_data=sample_market_data,
            signals=[
                TradingSignal(
                    symbol="BTCUSDT",
                    strategy=StrategyType.EMA_CROSSOVER,
                    action=SignalAction.BUY,
                    strength="STRONG",
                    confidence=8,
                    entry_price=53000.0,
                    stop_loss=51500.0,
                    take_profit=55000.0,
                    reasoning="Golden cross with volume confirmation"
                )
            ],
            ai_analysis="9 EMA crossed above 15 EMA with strong volume. BUY signal.",
            execution_time=2.1
        )
        
        strategy.ollama_client.analyze_market = AsyncMock(return_value=mock_analysis_result)
        
        # Run analysis
        result = await strategy.analyze(sample_market_data, session_config)
        
        # Verify results
        assert isinstance(result, AnalysisResult)
        assert result.strategy == StrategyType.EMA_CROSSOVER
        assert result.primary_signal.action == SignalAction.BUY
        assert result.primary_signal.confidence == 8

    @pytest.mark.asyncio
    async def test_analyze_death_cross(self, strategy, session_config):
        """Test analysis with death cross scenario."""
        # Create downtrending data
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        for i in range(20):
            timestamp = base_time.replace(hour=i)
            base_price = 52000 - (i * 100)  # Downtrend
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=base_price,
                high=base_price + 50,
                low=base_price - 75,
                close=base_price - 25,
                volume=1000 - (i * 10)  # Decreasing volume
            )
            ohlcv_data.append(ohlcv)
        
        bearish_data = MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=50000.0,
            ohlcv_data=ohlcv_data
        )
        
        # Mock Ollama response for death cross
        mock_analysis_result = AnalysisResult(
            symbol="BTCUSDT",
            timeframe="1h",
            strategy=StrategyType.EMA_CROSSOVER,
            market_data=bearish_data,
            signals=[
                TradingSignal(
                    symbol="BTCUSDT",
                    strategy=StrategyType.EMA_CROSSOVER,
                    action=SignalAction.SELL,
                    strength="MODERATE",
                    confidence=7,
                    entry_price=50000.0,
                    stop_loss=51000.0,
                    take_profit=48500.0,
                    reasoning="Death cross formation indicates bearish trend"
                )
            ],
            ai_analysis="9 EMA crossed below 15 EMA. SELL signal.",
            execution_time=1.9
        )
        
        strategy.ollama_client.analyze_market = AsyncMock(return_value=mock_analysis_result)
        
        # Run analysis
        result = await strategy.analyze(bearish_data, session_config)
        
        # Verify results
        assert result.primary_signal.action == SignalAction.SELL
        assert result.primary_signal.confidence == 7


class TestCombinedStrategy:
    """Test cases for Combined strategy."""

    @pytest.fixture
    def strategy(self):
        """Create Combined strategy instance."""
        return CombinedStrategy()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data suitable for combined analysis."""
        ohlcv_data = []
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        # Create data with both EMA crossover and S/R confirmation
        for i in range(25):  # More data for combined analysis
            timestamp = base_time.replace(hour=i)
            
            if i < 10:
                base_price = 50000 + (i * 50)  # Gradual rise
            elif i < 15:
                base_price = 50500 - ((i-10) * 30)  # Small pullback to support
            else:
                base_price = 50200 + ((i-15) * 120)  # Strong breakout
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=base_price,
                high=base_price + 75,
                low=base_price - 50,
                close=base_price + 25,
                volume=1200 + (i * 15)
            )
            ohlcv_data.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=51400.0,
            ohlcv_data=ohlcv_data
        )

    @pytest.fixture
    def session_config(self):
        """Create session config for Combined strategy."""
        return SessionConfig(
            strategy=StrategyType.COMBINED,
            symbol="BTCUSDT",
            timeframe="1h",
            confidence_threshold=7  # Higher threshold for combined
        )

    def test_strategy_initialization(self, strategy):
        """Test Combined strategy initialization."""
        assert strategy.strategy_type == StrategyType.COMBINED
        assert strategy.indicators is not None
        assert strategy.ollama_client is not None

    @pytest.mark.asyncio
    async def test_analyze_high_confidence_signal(self, strategy, sample_market_data, session_config):
        """Test analysis with high confidence signal from both strategies."""
        # Mock Ollama response for combined signal
        mock_analysis_result = AnalysisResult(
            symbol="BTCUSDT",
            timeframe="1h",
            strategy=StrategyType.COMBINED,
            market_data=sample_market_data,
            signals=[
                TradingSignal(
                    symbol="BTCUSDT",
                    strategy=StrategyType.COMBINED,
                    action=SignalAction.BUY,
                    strength="STRONG",
                    confidence=9,
                    entry_price=51400.0,
                    stop_loss=50200.0,
                    take_profit=53200.0,
                    reasoning="Both EMA crossover and S/R breakout confirm bullish signal"
                )
            ],
            ai_analysis="Golden cross confirmed by support breakout. Strong BUY signal.",
            execution_time=3.2
        )
        
        strategy.ollama_client.analyze_market = AsyncMock(return_value=mock_analysis_result)
        
        # Run analysis
        result = await strategy.analyze(sample_market_data, session_config)
        
        # Verify results
        assert isinstance(result, AnalysisResult)
        assert result.strategy == StrategyType.COMBINED
        assert result.primary_signal.action == SignalAction.BUY
        assert result.primary_signal.confidence == 9

    @pytest.mark.asyncio
    async def test_analyze_conflicting_signals(self, strategy, sample_market_data, session_config):
        """Test analysis when strategies provide conflicting signals."""
        # Mock Ollama response for conflicting signals
        mock_analysis_result = AnalysisResult(
            symbol="BTCUSDT",
            timeframe="1h",
            strategy=StrategyType.COMBINED,
            market_data=sample_market_data,
            signals=[
                TradingSignal(
                    symbol="BTCUSDT",
                    strategy=StrategyType.COMBINED,
                    action=SignalAction.NEUTRAL,
                    strength="WEAK",
                    confidence=4,
                    entry_price=51400.0,
                    reasoning="EMA suggests bullish but resistance level holds - conflicting signals"
                )
            ],
            ai_analysis="EMA crossover bullish but strong resistance. NEUTRAL signal.",
            execution_time=2.8
        )
        
        strategy.ollama_client.analyze_market = AsyncMock(return_value=mock_analysis_result)
        
        # Run analysis
        result = await strategy.analyze(sample_market_data, session_config)
        
        # Verify results
        assert result.primary_signal.action == SignalAction.NEUTRAL
        assert result.primary_signal.confidence < session_config.confidence_threshold

    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self, strategy, session_config):
        """Test analysis with insufficient data for combined strategy."""
        # Create minimal data (not enough for reliable combined analysis)
        minimal_data = MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=50000.0,
            ohlcv_data=[
                OHLCV(
                    timestamp=datetime.now(timezone.utc),
                    open=50000.0,
                    high=50100.0,
                    low=49900.0,
                    close=50050.0,
                    volume=1000.0
                )
            ]
        )
        
        with pytest.raises(DataValidationError):
            await strategy.analyze(minimal_data, session_config)