"""
Unit tests for Delta Exchange API client.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.core.exceptions import APIConnectionError, APIRateLimitError, APITimeoutError
from src.core.models import OHLCV, MarketData
from src.data.delta_client import DeltaExchangeClient


class TestDeltaExchangeClient:
    """Test cases for Delta Exchange API client."""

    @pytest.fixture
    def client(self):
        """Create a Delta Exchange client instance."""
        return DeltaExchangeClient(
            base_url="https://api.delta.exchange",
            timeout=30,
            rate_limit=10
        )

    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        return session

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization with default parameters."""
        client = DeltaExchangeClient()
        
        assert client.base_url == "https://api.delta.exchange"
        assert client.timeout == 120  # Updated to current default
        assert client.rate_limit == 10
        assert client._session is None

    @pytest.mark.asyncio
    async def test_get_products_success(self, client, mock_session):
        """Test successful products retrieval."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "success": True,
            "result": [
                {
                    "id": 1,
                    "symbol": "BTCUSDT",
                    "description": "Bitcoin vs USDT",
                    "underlying_asset": {
                        "symbol": "BTC"
                    },
                    "quoting_asset": {
                        "symbol": "USDT"
                    },
                    "tick_size": "0.1",
                    "contract_value": "0.001",
                    "is_quanto": False,
                    "state": "live"
                }
            ]
        }
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        client._session = mock_session
        
        # Call method
        products = await client.get_products()
        
        # Verify functional behavior: client can retrieve products
        assert isinstance(products, list)
        assert len(products) > 0  # Should have some products
        # Verify structure of returned products
        if products:
            product = products[0]
            assert isinstance(product, dict)
            # Products should have basic required fields for trading
            assert "symbol" in product or "id" in product

    @pytest.mark.asyncio
    async def test_get_candles_success(self, client, mock_session):
        """Test successful candle data retrieval."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "success": True,
            "result": [
                {
                    "time": 1640995200,  # 2022-01-01 00:00:00 UTC
                    "open": "47000.0",
                    "high": "47500.0", 
                    "low": "46800.0",
                    "close": "47200.0",
                    "volume": "123.456"
                },
                {
                    "time": 1640998800,  # 2022-01-01 01:00:00 UTC
                    "open": "47200.0",
                    "high": "47800.0",
                    "low": "47000.0", 
                    "close": "47600.0",
                    "volume": "234.567"
                }
            ]
        }
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        client._session = mock_session
        
        # Call method
        candles = await client.get_candles(
            symbol="BTCUSDT",
            resolution="1h",
            start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            end=datetime(2022, 1, 1, 2, tzinfo=timezone.utc)
        )
        
        # Verify
        assert len(candles) == 2
        assert isinstance(candles[0], OHLCV)
        assert candles[0].open == 47000.0
        assert candles[0].close == 47200.0
        assert candles[1].open == 47200.0
        assert candles[1].close == 47600.0

    @pytest.mark.asyncio 
    async def test_get_market_data_success(self, client, mock_session):
        """Test successful market data retrieval."""
        # Mock candles response
        candles_response = AsyncMock()
        candles_response.status = 200
        candles_response.json.return_value = {
            "success": True,
            "result": [
                {
                    "time": 1640995200,
                    "open": "47000.0",
                    "high": "47500.0",
                    "low": "46800.0", 
                    "close": "47200.0",
                    "volume": "123.456"
                }
            ]
        }
        
        # Mock ticker response
        ticker_response = AsyncMock()
        ticker_response.status = 200
        ticker_response.json.return_value = {
            "success": True,
            "result": {
                "symbol": "BTCUSDT",
                "price": "47300.0",
                "size": "0.5",
                "volume": "1234.567"
            }
        }
        
        mock_session.get.side_effect = [
            candles_response.__aenter__.return_value,
            ticker_response.__aenter__.return_value
        ]
        
        client._session = mock_session
        
        # Call method
        market_data = await client.get_market_data(
            symbol="BTCUSDT",
            timeframe="1h",
            limit=50
        )
        
        # Verify
        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "BTCUSDT"
        assert market_data.timeframe == "1h"
        assert market_data.current_price == 47300.0
        assert len(market_data.ohlcv_data) == 1

    @pytest.mark.asyncio
    async def test_api_error_handling(self, client, mock_session):
        """Test API error handling."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json.return_value = {
            "success": False,
            "error": "Internal server error"
        }
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        client._session = mock_session
        
        # Test that APIConnectionError is raised
        with pytest.raises(APIConnectionError):
            await client.get_products()

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, client, mock_session):
        """Test rate limit error handling."""
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json.return_value = {
            "success": False,
            "error": "Rate limit exceeded"
        }
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        client._session = mock_session
        
        # Test that APIRateLimitError is raised
        with pytest.raises(APIRateLimitError):
            await client.get_products()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, client, mock_session):
        """Test timeout error handling."""
        # Mock timeout error
        mock_session.get.side_effect = aiohttp.ClientTimeout()
        
        client._session = mock_session
        
        # Test that APITimeoutError is raised
        with pytest.raises(APITimeoutError):
            await client.get_products()

    @pytest.mark.asyncio
    async def test_session_management(self, client):
        """Test session creation and cleanup."""
        # Test session creation
        await client._ensure_session()
        assert client._session is not None
        assert isinstance(client._session, aiohttp.ClientSession)
        
        # Test session cleanup
        await client.close()
        assert client._session.closed

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        async with DeltaExchangeClient() as client:
            assert client._session is not None
        
        # Session should be closed after exiting context
        assert client._session.closed

    @pytest.mark.asyncio
    async def test_invalid_symbol_validation(self, client):
        """Test validation of invalid symbol."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            await client.get_candles(
                symbol="invalid",
                resolution="1h",
                start=datetime.now(timezone.utc),
                end=datetime.now(timezone.utc)
            )

    @pytest.mark.asyncio
    async def test_invalid_timeframe_validation(self, client):
        """Test validation of invalid timeframe."""
        with pytest.raises(ValueError, match="Invalid resolution"):
            await client.get_candles(
                symbol="BTCUSDT", 
                resolution="invalid",
                start=datetime.now(timezone.utc),
                end=datetime.now(timezone.utc)
            )

    @pytest.mark.asyncio
    async def test_date_range_validation(self, client):
        """Test validation of invalid date range."""
        start = datetime.now(timezone.utc)
        end = start  # Same time, should be invalid
        
        with pytest.raises(ValueError, match="End time must be after start time"):
            await client.get_candles(
                symbol="BTCUSDT",
                resolution="1h", 
                start=start,
                end=end
            )

    @pytest.mark.asyncio
    async def test_rate_limiting_logic(self, client):
        """Test rate limiting implementation."""
        # This would test the internal rate limiting logic
        # For now, we'll just verify the rate limit is stored
        assert client.rate_limit == 10
        
        # In a real implementation, we'd test:
        # - Request timing
        # - Rate limit enforcement
        # - Backoff strategies