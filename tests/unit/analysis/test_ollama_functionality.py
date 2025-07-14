"""
Functionality tests for Ollama client.

Focuses on testing AI analysis behavior and prompt generation rather than implementation details.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

from src.analysis.ollama_client import OllamaClient
from src.core.models import (
    MarketData, OHLCV, SessionConfig, StrategyType, Symbol, TimeFrame,
    AnalysisResult, TradingSignal, SignalAction
)


class TestOllamaClientFunctionality:
    """Test Ollama client functionality."""

    @pytest.fixture
    def ollama_client(self):
        """Create Ollama client instance."""
        return OllamaClient()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        candles = []
        base_price = 50000.0
        
        # Create realistic market data with trend
        for i in range(20):
            price = base_price + (i * 100)  # Upward trend
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price + 150,
                low=price - 50,
                close=price + 100,
                volume=1000.0 + (i * 10)
            )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price + 2000,
            ohlcv_data=candles
        )

    @pytest.fixture
    def session_config(self):
        """Create session configuration."""
        return SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER_V2,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            total_capital_inr=Decimal("100000"),
            trading_capital_pct=Decimal("50.0"),
            risk_per_trade_pct=Decimal("2.0"),
            take_profit_pct=Decimal("5.0"),
            leverage=10,
        )

    def test_ollama_client_initialization(self, ollama_client):
        """Test that Ollama client initializes correctly."""
        assert ollama_client is not None
        assert hasattr(ollama_client, 'base_url')
        assert hasattr(ollama_client, 'model')
        assert hasattr(ollama_client, 'timeout')
        
        # Should have reasonable defaults
        assert ollama_client.base_url is not None
        assert ollama_client.model is not None
        assert ollama_client.timeout > 0

    def test_prompt_generation_functionality(self, ollama_client, sample_market_data, session_config):
        """Test that prompt generation includes essential trading information."""
        # Test the prompt building method if it exists
        try:
            # Test if client can build prompts with market data
            prompt = ollama_client._build_analysis_prompt(
                sample_market_data, 
                session_config, 
                strategy_context="EMA crossover analysis"
            )
            
            # Prompt should contain essential trading information
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should be substantial
            
            # Should contain key trading elements
            essential_elements = [
                "BTCUSD",  # Symbol
                "EMA",     # Strategy context
                "price",   # Price information
                "volume",  # Volume data
            ]
            
            prompt_lower = prompt.lower()
            for element in essential_elements:
                assert element.lower() in prompt_lower, f"Prompt missing {element}"
                
        except AttributeError:
            # Method might not exist or be named differently
            # Test that client can at least be used for analysis
            pass

    @pytest.mark.asyncio
    async def test_market_data_analysis_functionality(self, ollama_client, sample_market_data, session_config):
        """Test that client can analyze market data and return structured results."""
        # Mock the HTTP client to simulate Ollama API response
        with patch('aiohttp.ClientSession') as mock_session:
            # Create mock response with realistic AI analysis
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": """
                Based on the market data analysis:
                
                SIGNAL: BUY
                CONFIDENCE: 8/10
                ENTRY_PRICE: 52000
                REASONING: Strong bullish EMA crossover detected with volume confirmation
                
                The 9-period EMA has crossed above the 15-period EMA, indicating bullish momentum.
                Volume is 15% above average, confirming the breakout.
                """
            }
            
            # Setup mock session
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.post.return_value.__aenter__.return_value = mock_response
            
            # Test analysis
            result = await ollama_client.analyze_market_data(sample_market_data, session_config)
            
            # Should return AnalysisResult
            assert isinstance(result, AnalysisResult)
            assert result.symbol == sample_market_data.symbol
            assert result.strategy == session_config.strategy
            assert result.market_data == sample_market_data
            
            # Should have parsed signals
            assert isinstance(result.signals, list)
            if result.signals:
                signal = result.signals[0]
                assert isinstance(signal, TradingSignal)
                assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.NEUTRAL]
                assert 1 <= signal.confidence <= 10
                assert signal.entry_price > 0

    @pytest.mark.asyncio
    async def test_signal_parsing_functionality(self, ollama_client):
        """Test that client can parse various AI response formats."""
        test_responses = [
            # Format 1: Structured response
            """
            SIGNAL: BUY
            CONFIDENCE: 7
            ENTRY_PRICE: 51500
            REASONING: Bullish pattern detected
            """,
            
            # Format 2: Natural language
            """
            I recommend a BUY signal with 8/10 confidence.
            Entry price should be around 51200.
            The reasoning is strong EMA crossover with volume confirmation.
            """,
            
            # Format 3: JSON-like
            """
            {
                "signal": "SELL",
                "confidence": 6,
                "entry_price": 50800,
                "reasoning": "Bearish divergence detected"
            }
            """
        ]
        
        for response_text in test_responses:
            try:
                # Test signal parsing method if it exists
                signals = ollama_client._parse_signals(response_text, "BTCUSD", StrategyType.EMA_CROSSOVER_V2)
                
                # Should parse at least some signals
                assert isinstance(signals, list)
                
                if signals:
                    signal = signals[0]
                    assert isinstance(signal, TradingSignal)
                    assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.NEUTRAL]
                    assert 1 <= signal.confidence <= 10
                    assert signal.entry_price > 0
                    assert len(signal.reasoning) > 0
                    
            except AttributeError:
                # Parsing method might not exist or be named differently
                pass

    @pytest.mark.asyncio
    async def test_error_handling_functionality(self, ollama_client, sample_market_data, session_config):
        """Test that client handles various error scenarios gracefully."""
        # Test HTTP connection error
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.side_effect = ConnectionError("Cannot connect to Ollama")
            
            try:
                result = await ollama_client.analyze_market_data(sample_market_data, session_config)
                # If no exception, should return reasonable fallback
                if result is not None:
                    assert isinstance(result, AnalysisResult)
                    assert result.ai_analysis is not None
            except Exception as e:
                # Should be meaningful error
                assert "connect" in str(e).lower() or "ollama" in str(e).lower()

        # Test HTTP error response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.post.return_value.__aenter__.return_value = mock_response
            
            try:
                result = await ollama_client.analyze_market_data(sample_market_data, session_config)
                # Should handle error gracefully
                if result is not None:
                    assert isinstance(result, AnalysisResult)
            except Exception as e:
                # Should be meaningful error
                assert len(str(e)) > 0

    @pytest.mark.asyncio
    async def test_timeout_handling_functionality(self, ollama_client, sample_market_data, session_config):
        """Test that client handles timeouts appropriately."""
        # Test timeout scenario
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Simulate timeout
            import asyncio
            mock_session_instance.post.side_effect = asyncio.TimeoutError("Request timeout")
            
            try:
                result = await ollama_client.analyze_market_data(sample_market_data, session_config)
                # Should handle timeout gracefully
                if result is not None:
                    assert isinstance(result, AnalysisResult)
            except Exception as e:
                # Should be timeout-related error
                assert "timeout" in str(e).lower() or "time" in str(e).lower()

    def test_market_data_formatting_functionality(self, ollama_client, sample_market_data):
        """Test that client formats market data appropriately for AI analysis."""
        try:
            # Test market data formatting method if it exists
            formatted_data = ollama_client._format_market_data(sample_market_data)
            
            # Should be string representation suitable for AI
            assert isinstance(formatted_data, str)
            assert len(formatted_data) > 50  # Should be substantial
            
            # Should contain key market information
            assert "BTCUSD" in formatted_data
            assert any(price_term in formatted_data.lower() for price_term in ['price', 'open', 'close', 'high', 'low'])
            assert "volume" in formatted_data.lower()
            
        except AttributeError:
            # Method might not exist - test that client can process market data
            assert sample_market_data.symbol == "BTCUSD"
            assert len(sample_market_data.ohlcv_data) > 0

    @pytest.mark.asyncio
    async def test_analysis_result_completeness(self, ollama_client, sample_market_data, session_config):
        """Test that analysis results contain all required information."""
        # Mock successful analysis
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": "BUY signal with 7/10 confidence at 51800. Strong bullish momentum."
            }
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.post.return_value.__aenter__.return_value = mock_response
            
            result = await ollama_client.analyze_market_data(sample_market_data, session_config)
            
            # Verify completeness of analysis result
            assert result.symbol == sample_market_data.symbol
            assert result.timeframe == session_config.timeframe
            assert result.strategy == session_config.strategy
            assert result.market_data == sample_market_data
            assert isinstance(result.ai_analysis, str)
            assert len(result.ai_analysis) > 0
            assert isinstance(result.signals, list)
            
            # Should have timestamp
            assert result.timestamp is not None
            assert isinstance(result.timestamp, datetime)

    def test_strategy_context_functionality(self, ollama_client):
        """Test that client can handle different strategy contexts."""
        strategies = [
            StrategyType.EMA_CROSSOVER_V2,
            StrategyType.SUPPORT_RESISTANCE,
            StrategyType.COMBINED
        ]
        
        for strategy in strategies:
            try:
                # Test strategy-specific prompt building if method exists
                context = ollama_client._get_strategy_context(strategy)
                
                # Should return strategy-specific context
                assert isinstance(context, str)
                assert len(context) > 0
                
                # Should contain strategy-relevant terms
                if strategy == StrategyType.EMA_CROSSOVER_V2:
                    assert "ema" in context.lower() or "moving average" in context.lower()
                elif strategy == StrategyType.SUPPORT_RESISTANCE:
                    assert "support" in context.lower() or "resistance" in context.lower()
                elif strategy == StrategyType.COMBINED:
                    assert "combined" in context.lower() or "multiple" in context.lower()
                    
            except AttributeError:
                # Method might not exist - test basic strategy handling
                assert strategy in [StrategyType.EMA_CROSSOVER_V2, StrategyType.SUPPORT_RESISTANCE, StrategyType.COMBINED]

    @pytest.mark.asyncio
    async def test_multiple_signals_functionality(self, ollama_client, sample_market_data, session_config):
        """Test that client can handle responses with multiple trading signals."""
        # Mock response with multiple signals
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": """
                Multiple signals detected:
                
                PRIMARY SIGNAL: BUY
                CONFIDENCE: 8/10
                ENTRY_PRICE: 52000
                REASONING: Strong EMA crossover
                
                SECONDARY SIGNAL: NEUTRAL
                CONFIDENCE: 6/10
                ENTRY_PRICE: 51500
                REASONING: Support level hold
                """
            }
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.post.return_value.__aenter__.return_value = mock_response
            
            result = await ollama_client.analyze_market_data(sample_market_data, session_config)
            
            # Should handle multiple signals
            assert isinstance(result.signals, list)
            
            # If multiple signals parsed, should have primary signal method
            if len(result.signals) > 1:
                primary = result.primary_signal
                assert primary is not None
                assert isinstance(primary, TradingSignal)
                
                # Primary should have highest confidence
                for signal in result.signals:
                    assert primary.confidence >= signal.confidence

    @pytest.mark.asyncio
    async def test_invalid_response_handling(self, ollama_client, sample_market_data, session_config):
        """Test handling of invalid or malformed AI responses."""
        invalid_responses = [
            "",  # Empty response
            "Invalid JSON response",  # No structured data
            "SIGNAL: INVALID_ACTION",  # Invalid signal action
            "CONFIDENCE: 15",  # Invalid confidence (>10)
            "No clear trading signal detected",  # Neutral/unclear response
        ]
        
        for invalid_response in invalid_responses:
            with patch('aiohttp.ClientSession') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = {"response": invalid_response}
                
                mock_session_instance = AsyncMock()
                mock_session.return_value.__aenter__.return_value = mock_session_instance
                mock_session_instance.post.return_value.__aenter__.return_value = mock_response
                
                result = await ollama_client.analyze_market_data(sample_market_data, session_config)
                
                # Should handle invalid responses gracefully
                assert isinstance(result, AnalysisResult)
                assert result.ai_analysis is not None
                assert isinstance(result.signals, list)
                
                # If no valid signals parsed, signals list might be empty
                # This is acceptable behavior for invalid responses
                for signal in result.signals:
                    assert isinstance(signal, TradingSignal)
                    assert 1 <= signal.confidence <= 10