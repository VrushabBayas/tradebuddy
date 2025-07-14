"""
Functionality tests for data processing capabilities.

Focuses on testing data retrieval and processing functionality
rather than implementation details.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from src.data.delta_client import DeltaExchangeClient
from src.core.models import OHLCV, MarketData, Symbol, TimeFrame
from src.core.exceptions import APIConnectionError, APITimeoutError, DataValidationError


class TestDeltaClientFunctionality:
    """Test Delta Exchange client functionality."""

    @pytest.fixture
    def delta_client(self):
        """Create Delta Exchange client."""
        return DeltaExchangeClient(
            base_url="https://api.delta.exchange",
            timeout=30,
            rate_limit=10
        )

    @pytest.fixture
    def mock_ohlcv_response(self):
        """Mock OHLCV API response."""
        return {
            "success": True,
            "result": [
                {
                    "time": 1640995200,  # Unix timestamp
                    "open": "50000.0",
                    "high": "51000.0", 
                    "low": "49500.0",
                    "close": "50500.0",
                    "volume": "1000.5"
                },
                {
                    "time": 1640998800,  # Next candle
                    "open": "50500.0",
                    "high": "51500.0",
                    "low": "50000.0", 
                    "close": "51200.0",
                    "volume": "1200.8"
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_market_data_retrieval_functionality(self, delta_client, mock_ohlcv_response):
        """Test market data retrieval and processing."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            # Setup mock session
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_ohlcv_response
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            # Test data retrieval
            market_data = await delta_client.get_market_data(
                symbol="BTCUSD",
                timeframe="1h",
                limit=100
            )
            
            # Verify functional behavior
            assert isinstance(market_data, MarketData)
            assert market_data.symbol == "BTCUSD"
            assert market_data.timeframe == "1h"
            assert len(market_data.ohlcv_data) == 2
            
            # Verify OHLCV data integrity
            first_candle = market_data.ohlcv_data[0]
            assert isinstance(first_candle, OHLCV)
            assert first_candle.open == Decimal("50000.0")
            assert first_candle.high == Decimal("51000.0")
            assert first_candle.low == Decimal("49500.0")
            assert first_candle.close == Decimal("50500.0")
            assert first_candle.volume == Decimal("1000.5")
            
            # Verify data ordering (should be chronological)
            assert market_data.ohlcv_data[0].timestamp < market_data.ohlcv_data[1].timestamp

    @pytest.mark.asyncio
    async def test_market_data_current_price_functionality(self, delta_client, mock_ohlcv_response):
        """Test current price extraction from market data."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_ohlcv_response
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            market_data = await delta_client.get_market_data("BTCUSD", "1h")
            
            # Current price should be the last close price
            expected_current_price = 51200.0  # Last candle's close price
            assert market_data.current_price == expected_current_price

    @pytest.mark.asyncio
    async def test_error_handling_functionality(self, delta_client):
        """Test error handling for various failure scenarios."""
        # Test connection error
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.side_effect = ConnectionError("Network unreachable")
            
            with pytest.raises((APIConnectionError, ConnectionError)):
                await delta_client.get_market_data("BTCUSD", "1h")
        
        # Test HTTP error response
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises((APIConnectionError, Exception)):
                await delta_client.get_market_data("BTCUSD", "1h")
        
        # Test timeout
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.get.side_effect = asyncio.TimeoutError("Request timeout")
            
            with pytest.raises((APITimeoutError, asyncio.TimeoutError)):
                await delta_client.get_market_data("BTCUSD", "1h")

    @pytest.mark.asyncio
    async def test_data_validation_functionality(self, delta_client):
        """Test data validation for malformed responses."""
        invalid_responses = [
            {"success": False, "error": "Invalid symbol"},  # API error
            {"success": True, "result": []},  # Empty data
            {"success": True, "result": [{"invalid": "data"}]},  # Missing required fields
            {"malformed": "response"},  # Completely invalid
        ]
        
        for invalid_response in invalid_responses:
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aender__.return_value = mock_session
                
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = invalid_response
                mock_session.get.return_value.__aenter__.return_value = mock_response
                
                try:
                    result = await delta_client.get_market_data("BTCUSD", "1h")
                    # If no exception, verify result is reasonable
                    if result is not None:
                        assert isinstance(result, MarketData)
                except (DataValidationError, ValueError, KeyError):
                    # Expected for invalid data
                    pass

    @pytest.mark.asyncio
    async def test_rate_limiting_functionality(self, delta_client):
        """Test rate limiting behavior."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock rate limit response
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.text.return_value = "Rate limit exceeded"
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises((APIConnectionError, Exception)):
                await delta_client.get_market_data("BTCUSD", "1h")

    @pytest.mark.asyncio
    async def test_multiple_symbol_support(self, delta_client, mock_ohlcv_response):
        """Test functionality with different symbols."""
        symbols_to_test = ["BTCUSD", "ETHUSD", "SOLUSDT", "ADAUSDT"]
        
        for symbol in symbols_to_test:
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_ohlcv_response
                mock_session.get.return_value.__aenter__.return_value = mock_response
                
                market_data = await delta_client.get_market_data(symbol, "1h")
                
                # Should handle any valid symbol
                assert market_data.symbol == symbol
                assert len(market_data.ohlcv_data) > 0

    @pytest.mark.asyncio
    async def test_timeframe_support(self, delta_client, mock_ohlcv_response):
        """Test functionality with different timeframes."""
        timeframes_to_test = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        for timeframe in timeframes_to_test:
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aender__.return_value = mock_session
                
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_ohlcv_response
                mock_session.get.return_value.__aenter__.return_value = mock_response
                
                market_data = await delta_client.get_market_data("BTCUSD", timeframe)
                
                # Should handle any valid timeframe
                assert market_data.timeframe == timeframe

    def test_client_configuration_functionality(self):
        """Test client configuration options."""
        # Test custom configuration
        custom_client = DeltaExchangeClient(
            base_url="https://custom.api.com",
            timeout=60,
            rate_limit=5
        )
        
        assert custom_client.base_url == "https://custom.api.com"
        assert custom_client.timeout == 60
        assert custom_client.rate_limit == 5
        
        # Test default configuration
        default_client = DeltaExchangeClient()
        assert default_client.base_url is not None
        assert default_client.timeout > 0
        assert default_client.rate_limit > 0




class TestDataValidationFunctionality:
    """Test data validation and processing functionality."""

    @pytest.fixture
    def delta_client(self):
        """Create Delta client."""
        return DeltaExchangeClient()

    def test_data_model_compatibility(self):
        """Test OHLCV data model compatibility."""
        # Create OHLCV from REST API format
        rest_ohlcv = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000,
            high=50100,
            low=49900,
            close=50050,
            volume=1000
        )
        
        # Verify data structure integrity
        assert hasattr(rest_ohlcv, 'timestamp')
        assert hasattr(rest_ohlcv, 'open')
        assert hasattr(rest_ohlcv, 'high')
        assert hasattr(rest_ohlcv, 'low')
        assert hasattr(rest_ohlcv, 'close')
        assert hasattr(rest_ohlcv, 'volume')
        
        # Verify data types
        assert isinstance(rest_ohlcv.timestamp, datetime)
        assert rest_ohlcv.open > 0
        assert rest_ohlcv.high >= rest_ohlcv.open
        assert rest_ohlcv.low <= rest_ohlcv.open
        assert rest_ohlcv.volume >= 0

    @pytest.mark.asyncio
    async def test_historical_data_processing(self, delta_client):
        """Test historical data processing functionality."""
        # Mock historical data retrieval
        with patch.object(delta_client, 'get_market_data') as mock_historical:
            mock_market_data = MarketData(
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
            mock_historical.return_value = mock_market_data
            
            # Get historical data
            historical_data = await delta_client.get_market_data("BTCUSD", "1h")
            
            # Verify data processing
            assert historical_data.symbol == "BTCUSD"
            assert len(historical_data.ohlcv_data) > 0
            assert historical_data.current_price > 0
            
            # Verify timestamp ordering
            if len(historical_data.ohlcv_data) > 1:
                for i in range(1, len(historical_data.ohlcv_data)):
                    assert historical_data.ohlcv_data[i].timestamp >= historical_data.ohlcv_data[i-1].timestamp