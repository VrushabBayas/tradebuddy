"""
Unit tests for Ollama API client.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.analysis.ollama_client import OllamaClient
from src.core.exceptions import APIConnectionError, APITimeoutError, DataValidationError
from src.core.models import (
    OHLCV,
    AnalysisResult,
    MarketData,
    StrategyType,
    TradingSignal,
)


class TestOllamaClient:
    """Test cases for Ollama API client."""

    @pytest.fixture
    def client(self):
        """Create an Ollama client instance."""
        return OllamaClient(
            base_url="http://localhost:11434", model="qwen2.5:14b", timeout=30
        )

    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        return session

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        ohlcv = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50500.0,
            low=49800.0,
            close=50300.0,
            volume=1500.0,
        )

        market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[ohlcv]
        )

        return market_data

    @pytest.fixture
    def sample_technical_analysis(self):
        """Create sample technical analysis data."""
        return {
            "ema_crossover": {
                "ema_9": 50250.0,
                "ema_15": 50100.0,
                "is_golden_cross": True,
                "crossover_strength": 7,
            },
            "support_resistance": [
                {"level": 49800.0, "strength": 6, "is_support": True},
                {"level": 50800.0, "strength": 5, "is_support": False},
            ],
            "volume_analysis": {
                "current_volume": 1500.0,
                "average_volume": 1200.0,
                "volume_ratio": 1.25,
                "volume_trend": "increasing",
            },
            "price_action": {
                "trend_direction": "bullish",
                "trend_strength": 7,
                "momentum": 2.5,
                "volatility": 1.8,
            },
            "overall_sentiment": "bullish",
            "confidence_score": 8,
        }

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization with default parameters."""
        client = OllamaClient()

        assert client.base_url == "http://localhost:11434"
        assert client.model == "qwen2.5:14b"
        assert client.timeout == 30
        assert client._session is None

    @pytest.mark.asyncio
    async def test_model_health_check_success(self, client, mock_session):
        """Test successful model health check."""
        # Mock response for model list
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "qwen2.5:14b",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 8000000000,
                    "digest": "abc123",
                }
            ]
        }
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client._session = mock_session

        # Call method
        is_healthy = await client.check_model_health()

        # Verify
        assert is_healthy is True
        mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_health_check_model_not_found(self, client, mock_session):
        """Test health check when model is not found."""
        # Mock response with different model
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "different:model",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 8000000000,
                    "digest": "abc123",
                }
            ]
        }
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client._session = mock_session

        # Call method
        is_healthy = await client.check_model_health()

        # Verify
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_analyze_market_success(
        self, client, mock_session, sample_market_data, sample_technical_analysis
    ):
        """Test successful market analysis."""
        # Mock successful Ollama response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "response": """Based on the technical analysis provided:

TRADING SIGNAL: BUY
CONFIDENCE: 8/10
ENTRY PRICE: $50,300
STOP LOSS: $49,500
TAKE PROFIT: $51,500

ANALYSIS:
The golden cross formation (9 EMA above 15 EMA) with strong volume confirmation suggests bullish momentum. The current price is holding above support at $49,800. Risk-reward ratio of 1:1.5 is acceptable.

REASONING:
- Strong EMA crossover signal (strength 7/10)
- Volume 25% above average confirming the move
- Clear support level identified at $49,800
- Bullish price action with good momentum"""
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response

        client._session = mock_session

        # Call method
        result = await client.analyze_market(
            market_data=sample_market_data,
            technical_analysis=sample_technical_analysis,
            strategy=StrategyType.EMA_CROSSOVER,
        )

        # Verify
        assert isinstance(result, AnalysisResult)
        assert result.symbol == "BTCUSD"
        assert result.strategy == StrategyType.EMA_CROSSOVER
        assert len(result.signals) > 0

        # Check primary signal
        primary_signal = result.primary_signal
        assert primary_signal is not None
        assert primary_signal.action.value == "BUY"
        assert primary_signal.confidence == 8
        assert primary_signal.entry_price == 50300.0

    @pytest.mark.asyncio
    async def test_analyze_market_with_support_resistance_strategy(
        self, client, mock_session, sample_market_data, sample_technical_analysis
    ):
        """Test market analysis with support/resistance strategy."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "response": """SUPPORT/RESISTANCE ANALYSIS:

TRADING SIGNAL: BUY
CONFIDENCE: 7/10
ENTRY PRICE: $50,300
STOP LOSS: $49,700
TAKE PROFIT: $50,900

ANALYSIS:
Price is testing resistance at $50,800 but strong support exists at $49,800. The bounce from support level shows buyer interest. Volume increase supports potential breakout.

REASONING:
- Strong support level at $49,800 (strength 6/10)
- Price action shows bullish reversal from support
- Volume confirmation indicates institutional interest"""
        }
        mock_session.post.return_value.__aenter__.return_value = mock_response

        client._session = mock_session

        result = await client.analyze_market(
            market_data=sample_market_data,
            technical_analysis=sample_technical_analysis,
            strategy=StrategyType.SUPPORT_RESISTANCE,
        )

        assert isinstance(result, AnalysisResult)
        assert result.strategy == StrategyType.SUPPORT_RESISTANCE
        assert result.primary_signal.confidence == 7

    @pytest.mark.asyncio
    async def test_parse_trading_signal_success(self, client):
        """Test successful signal parsing from AI response."""
        ai_response = """Based on the analysis:

TRADING SIGNAL: SELL
CONFIDENCE: 9/10
ENTRY PRICE: $50,300
STOP LOSS: $51,000
TAKE PROFIT: $49,000

The bearish divergence and high RSI suggest a reversal is imminent."""

        market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[]
        )

        signals = client._parse_trading_signals(
            ai_response, market_data, StrategyType.EMA_CROSSOVER
        )

        assert len(signals) == 1
        signal = signals[0]
        assert signal.action.value == "SELL"
        assert signal.confidence == 9
        assert signal.entry_price == 50300.0
        assert signal.stop_loss == 51000.0
        assert signal.take_profit == 49000.0

    @pytest.mark.asyncio
    async def test_parse_neutral_signal(self, client):
        """Test parsing neutral signal."""
        ai_response = """Market Analysis:

TRADING SIGNAL: NEUTRAL
CONFIDENCE: 5/10

The market is consolidating with no clear direction. Wait for better setup."""

        market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[]
        )

        signals = client._parse_trading_signals(
            ai_response, market_data, StrategyType.COMBINED
        )

        assert len(signals) == 1
        signal = signals[0]
        assert signal.action.value == "NEUTRAL"
        assert signal.confidence == 5

    @pytest.mark.asyncio
    async def test_api_error_handling(self, client, mock_session):
        """Test API error handling."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal server error"
        mock_session.post.return_value.__aenter__.return_value = mock_response

        client._session = mock_session

        market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[]
        )

        with pytest.raises(APIConnectionError):
            await client.analyze_market(market_data, {}, StrategyType.EMA_CROSSOVER)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, client, mock_session):
        """Test timeout error handling."""
        # Mock timeout error
        mock_session.post.side_effect = aiohttp.ClientTimeout()

        client._session = mock_session

        market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[]
        )

        with pytest.raises(APITimeoutError):
            await client.analyze_market(market_data, {}, StrategyType.EMA_CROSSOVER)

    @pytest.mark.asyncio
    async def test_invalid_market_data_validation(self, client):
        """Test validation of invalid market data."""
        # Empty market data
        with pytest.raises(DataValidationError, match="Market data cannot be None"):
            await client.analyze_market(None, {}, StrategyType.EMA_CROSSOVER)

        # Market data with no OHLCV data
        empty_market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[]
        )

        with pytest.raises(DataValidationError, match="No OHLCV data provided"):
            await client.analyze_market(
                empty_market_data, {}, StrategyType.EMA_CROSSOVER
            )

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        async with OllamaClient() as client:
            assert client._session is not None

        # Session should be closed after exiting context
        assert client._session.closed

    @pytest.mark.asyncio
    async def test_prompt_generation_ema_strategy(
        self, client, sample_market_data, sample_technical_analysis
    ):
        """Test prompt generation for EMA crossover strategy."""
        prompt = client._generate_prompt(
            market_data=sample_market_data,
            technical_analysis=sample_technical_analysis,
            strategy=StrategyType.EMA_CROSSOVER,
        )

        assert "EMA Crossover" in prompt
        assert "BTCUSD" in prompt
        assert "$50,300" in prompt
        assert "9 EMA: $50,250" in prompt
        assert "15 EMA: $50,100" in prompt
        assert "Golden Cross: TRUE" in prompt

    @pytest.mark.asyncio
    async def test_prompt_generation_support_resistance_strategy(
        self, client, sample_market_data, sample_technical_analysis
    ):
        """Test prompt generation for support/resistance strategy."""
        prompt = client._generate_prompt(
            market_data=sample_market_data,
            technical_analysis=sample_technical_analysis,
            strategy=StrategyType.SUPPORT_RESISTANCE,
        )

        assert "Support and Resistance" in prompt
        assert "Support Level: $49,800" in prompt
        assert "Resistance Level: $50,800" in prompt

    @pytest.mark.asyncio
    async def test_signal_parsing_edge_cases(self, client):
        """Test signal parsing with edge cases."""
        market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[]
        )

        # Test with malformed response
        malformed_response = "This is not a proper trading signal format"
        signals = client._parse_trading_signals(
            malformed_response, market_data, StrategyType.EMA_CROSSOVER
        )

        # Should create a default WAIT signal when parsing fails
        assert len(signals) == 1
        assert signals[0].action.value == "WAIT"
        assert signals[0].confidence == 1

    @pytest.mark.asyncio
    async def test_multiple_signals_parsing(self, client):
        """Test parsing multiple signals from AI response."""
        ai_response = """Analysis shows multiple opportunities:

PRIMARY SIGNAL:
TRADING SIGNAL: BUY
CONFIDENCE: 8/10
ENTRY PRICE: $50,300
STOP LOSS: $49,500
TAKE PROFIT: $51,500

SECONDARY SIGNAL:
TRADING SIGNAL: SELL
CONFIDENCE: 6/10
ENTRY PRICE: $50,800
STOP LOSS: $51,200
TAKE PROFIT: $50,200

The primary signal has higher confidence due to volume confirmation."""

        market_data = MarketData(
            symbol="BTCUSD", timeframe="1h", current_price=50300.0, ohlcv_data=[]
        )

        signals = client._parse_trading_signals(
            ai_response, market_data, StrategyType.COMBINED
        )

        # Should extract the primary signal (highest confidence)
        assert len(signals) >= 1
        primary_signal = max(signals, key=lambda s: s.confidence)
        assert primary_signal.action.value == "BUY"
        assert primary_signal.confidence == 8
