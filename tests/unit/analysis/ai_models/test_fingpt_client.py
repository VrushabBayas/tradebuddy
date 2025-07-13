"""
Test FinGPT client implementation.

Tests FinGPT integration following TDD approach.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.analysis.ai_models.fingpt_client import FinGPTClient
from src.core.models import (
    MarketData,
    AnalysisResult,
    StrategyType,
    Symbol,
    OHLCV,
    TradingSignal,
    SignalAction,
)
from src.core.exceptions import APIConnectionError, APITimeoutError, DataValidationError


class TestFinGPTClient:
    """Test FinGPT client functionality."""

    @pytest.fixture
    def fingpt_client(self):
        """Create FinGPT client for testing."""
        return FinGPTClient(model_variant="v3.2")

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return MarketData(
            symbol=Symbol.BTCUSD,
            timeframe="1m",
            current_price=45000.0,
            ohlcv_data=[
                OHLCV(open=44900, high=45100, low=44800, close=45000, volume=1000),
                OHLCV(open=45000, high=45200, low=44950, close=45100, volume=1200),
            ]
        )

    @pytest.fixture
    def sample_technical_analysis(self):
        """Create sample technical analysis data."""
        return {
            "ema_crossover": {
                "ema_9": 45050.0,
                "ema_15": 44950.0,
                "is_golden_cross": True,
                "crossover_strength": 7
            },
            "volume_analysis": {
                "current_volume": 1100,
                "volume_ratio": 1.2
            }
        }

    def test_fingpt_client_initialization(self, fingpt_client):
        """Test FinGPT client initialization."""
        assert fingpt_client.model_variant == "v3.2"
        assert fingpt_client.model_name == "fingpt:v3.2"
        assert fingpt_client.model_version == "v3.2"

    def test_fingpt_client_model_variants(self):
        """Test different FinGPT model variants."""
        variants = ["v3.1", "v3.2", "v3.3"]
        
        for variant in variants:
            client = FinGPTClient(model_variant=variant)
            assert client.model_variant == variant
            assert client.model_name == f"fingpt:{variant}"

    @pytest.mark.asyncio
    async def test_fingpt_analyze_market_input_validation(self, fingpt_client):
        """Test input validation for analyze_market method."""
        # Test with None market data
        with pytest.raises(DataValidationError):
            await fingpt_client.analyze_market(
                market_data=None,
                technical_analysis={},
                strategy=StrategyType.EMA_CROSSOVER
            )

        # Test with empty OHLCV data
        invalid_market_data = MarketData(
            symbol=Symbol.BTCUSD,
            timeframe="1m",
            current_price=45000.0,
            ohlcv_data=[]
        )
        with pytest.raises(DataValidationError):
            await fingpt_client.analyze_market(
                market_data=invalid_market_data,
                technical_analysis={},
                strategy=StrategyType.EMA_CROSSOVER
            )

    @pytest.mark.asyncio
    async def test_fingpt_generate_prompt(self, fingpt_client, sample_market_data, sample_technical_analysis):
        """Test FinGPT prompt generation."""
        prompt = fingpt_client._generate_financial_prompt(
            market_data=sample_market_data,
            technical_analysis=sample_technical_analysis,
            strategy=StrategyType.EMA_CROSSOVER
        )
        
        # Verify prompt contains key elements
        assert "BTCUSD" in prompt
        assert "45000" in prompt
        assert "FinGPT Financial Analysis" in prompt
        assert "EMA Crossover" in prompt
        assert "Golden Cross: True" in prompt

    @pytest.mark.asyncio
    async def test_fingpt_parse_response(self, fingpt_client, sample_market_data):
        """Test FinGPT response parsing."""
        mock_response = """
        Financial Analysis: Based on the EMA crossover pattern and current market conditions.
        
        TRADING SIGNAL: BUY
        CONFIDENCE: 8/10
        ENTRY PRICE: $45000
        STOP LOSS: $44500
        TAKE PROFIT: $46000
        
        REASONING: Strong golden cross with volume confirmation.
        """
        
        signals = fingpt_client._parse_fingpt_response(
            response=mock_response,
            market_data=sample_market_data,
            strategy=StrategyType.EMA_CROSSOVER
        )
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.action == SignalAction.BUY
        assert signal.confidence == 8
        assert signal.entry_price == 45000.0
        assert signal.stop_loss == 44500.0
        assert signal.take_profit == 46000.0

    @pytest.mark.asyncio
    async def test_fingpt_sentiment_analysis(self, fingpt_client):
        """Test FinGPT sentiment analysis capability."""
        news_text = "Bitcoin reaches new highs as institutional adoption increases"
        
        sentiment = await fingpt_client.analyze_sentiment(news_text)
        
        assert "sentiment" in sentiment
        assert "confidence" in sentiment
        assert sentiment["sentiment"] in ["bullish", "bearish", "neutral"]
        assert 0 <= sentiment["confidence"] <= 10

    @pytest.mark.asyncio
    @patch('src.analysis.ai_models.fingpt_client.requests.post')
    async def test_fingpt_api_connection_error(self, mock_post, fingpt_client, sample_market_data, sample_technical_analysis):
        """Test API connection error handling."""
        mock_post.side_effect = ConnectionError("Connection failed")
        
        with pytest.raises(APIConnectionError):
            await fingpt_client.analyze_market(
                market_data=sample_market_data,
                technical_analysis=sample_technical_analysis,
                strategy=StrategyType.EMA_CROSSOVER
            )

    @pytest.mark.asyncio
    @patch('src.analysis.ai_models.fingpt_client.requests.post')
    async def test_fingpt_api_timeout_error(self, mock_post, fingpt_client, sample_market_data, sample_technical_analysis):
        """Test API timeout error handling."""
        import requests
        mock_post.side_effect = requests.Timeout("Request timeout")
        
        with pytest.raises(APITimeoutError):
            await fingpt_client.analyze_market(
                market_data=sample_market_data,
                technical_analysis=sample_technical_analysis,
                strategy=StrategyType.EMA_CROSSOVER
            )

    @pytest.mark.asyncio
    async def test_fingpt_health_check(self, fingpt_client):
        """Test FinGPT model health check."""
        with patch.object(fingpt_client, '_check_model_availability', return_value=True):
            health = await fingpt_client.check_model_health()
            assert health is True

        with patch.object(fingpt_client, '_check_model_availability', return_value=False):
            health = await fingpt_client.check_model_health()
            assert health is False

    @pytest.mark.asyncio
    async def test_fingpt_close_cleanup(self, fingpt_client):
        """Test FinGPT client cleanup."""
        # Should not raise exception
        await fingpt_client.close()

    @pytest.mark.asyncio
    async def test_fingpt_context_manager(self, sample_market_data, sample_technical_analysis):
        """Test FinGPT client as async context manager."""
        async with FinGPTClient(model_variant="v3.2") as client:
            assert client is not None
            assert isinstance(client, FinGPTClient)

    def test_fingpt_error_handling_functionality(self, fingpt_client):
        """Test error handling follows expected patterns."""
        # Test that client handles various error scenarios gracefully
        assert hasattr(fingpt_client, '_handle_api_error')
        assert hasattr(fingpt_client, '_validate_response')
        assert hasattr(fingpt_client, '_create_fallback_signal')

    @pytest.mark.asyncio
    async def test_fingpt_financial_specific_features(self, fingpt_client, sample_market_data):
        """Test FinGPT's financial-specific capabilities."""
        # Test financial prompt generation
        prompt = fingpt_client._generate_financial_prompt(
            market_data=sample_market_data,
            technical_analysis={"test": "data"},
            strategy=StrategyType.EMA_CROSSOVER
        )
        
        # Should contain financial-specific elements
        assert "Financial Analysis" in prompt
        assert "Market Data" in prompt
        assert "Technical Indicators" in prompt

    @pytest.mark.asyncio
    async def test_fingpt_response_validation(self, fingpt_client, sample_market_data):
        """Test FinGPT response validation."""
        # Test valid response
        valid_response = "TRADING SIGNAL: BUY\nCONFIDENCE: 7/10"
        signals = fingpt_client._parse_fingpt_response(
            response=valid_response,
            market_data=sample_market_data,
            strategy=StrategyType.EMA_CROSSOVER
        )
        assert len(signals) >= 1

        # Test invalid response creates fallback signal
        invalid_response = "Invalid response format"
        signals = fingpt_client._parse_fingpt_response(
            response=invalid_response,
            market_data=sample_market_data,
            strategy=StrategyType.EMA_CROSSOVER
        )
        assert len(signals) == 1
        assert signals[0].action == SignalAction.WAIT


class TestFinGPTIntegration:
    """Test FinGPT integration functionality."""

    @pytest.mark.asyncio
    async def test_fingpt_with_different_strategies(self):
        """Test FinGPT with different trading strategies."""
        client = FinGPTClient(model_variant="v3.2")
        
        market_data = MarketData(
            symbol=Symbol.BTCUSD,
            timeframe="1m",
            current_price=45000.0,
            ohlcv_data=[OHLCV(open=44900, high=45100, low=44800, close=45000, volume=1000)]
        )
        
        for strategy in StrategyType:
            prompt = client._generate_financial_prompt(
                market_data=market_data,
                technical_analysis={},
                strategy=strategy
            )
            assert strategy.value.replace('_', ' ').title() in prompt

    @pytest.mark.asyncio
    async def test_fingpt_model_variants_consistency(self):
        """Test that all FinGPT variants maintain consistent interface."""
        variants = ["v3.1", "v3.2", "v3.3"]
        
        for variant in variants:
            client = FinGPTClient(model_variant=variant)
            assert client.model_variant == variant
            assert hasattr(client, 'analyze_market')
            assert hasattr(client, 'check_model_health')
            assert hasattr(client, 'close')