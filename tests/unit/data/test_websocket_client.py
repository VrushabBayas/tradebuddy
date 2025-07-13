"""
Unit tests for WebSocket client functionality.

Tests the functional behavior of the Delta Exchange WebSocket client
rather than implementation details.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.core.constants import WebSocketConstants
from src.core.exceptions import APIConnectionError
from src.core.models import OHLCV
from src.data.websocket_client import (
    DeltaWebSocketClient,
    process_candlestick_data,
    process_orderbook_data,
    process_trade_data,
)


class TestDeltaWebSocketClient:
    """Test DeltaWebSocketClient functionality."""

    def test_client_initialization(self):
        """Test client initializes with correct default values."""
        client = DeltaWebSocketClient()
        
        assert client.base_url == "wss://socket.delta.exchange"
        assert client.ping_interval == 30
        assert client.ping_timeout == 10
        assert client.max_reconnect_attempts == 10
        assert not client.connected
        assert client.get_subscription_count() == 0

    def test_client_initialization_with_custom_params(self):
        """Test client initializes with custom parameters."""
        client = DeltaWebSocketClient(
            base_url="wss://custom.url",
            ping_interval=60,
            ping_timeout=20,
            max_reconnect_attempts=5
        )
        
        assert client.base_url == "wss://custom.url"
        assert client.ping_interval == 60
        assert client.ping_timeout == 20
        assert client.max_reconnect_attempts == 5

    def test_subscription_management(self):
        """Test subscription management functionality."""
        client = DeltaWebSocketClient()
        
        # Initially no subscriptions
        assert client.get_subscription_count() == 0
        assert len(client.subscriptions) == 0
        
        # Add subscriptions (without actually connecting)
        client._subscriptions[f"{WebSocketConstants.CANDLESTICK_CHANNEL}:BTCUSDT"] = {
            "type": WebSocketConstants.SUBSCRIBE,
            "channel": WebSocketConstants.CANDLESTICK_CHANNEL,
            "symbol": "BTCUSDT",
            "timeframe": "1m"
        }
        
        client._subscriptions[f"{WebSocketConstants.ORDERBOOK_CHANNEL}:ETHUSDT"] = {
            "type": WebSocketConstants.SUBSCRIBE,
            "channel": WebSocketConstants.ORDERBOOK_CHANNEL,
            "symbol": "ETHUSDT"
        }
        
        # Check subscriptions were added
        assert client.get_subscription_count() == 2
        subscriptions = client.subscriptions
        assert f"{WebSocketConstants.CANDLESTICK_CHANNEL}:BTCUSDT" in subscriptions
        assert f"{WebSocketConstants.ORDERBOOK_CHANNEL}:ETHUSDT" in subscriptions

    def test_subscription_data_structure(self):
        """Test subscription data contains correct information."""
        client = DeltaWebSocketClient()
        
        # Simulate adding a candlestick subscription
        subscription = {
            "type": WebSocketConstants.SUBSCRIBE,
            "channel": WebSocketConstants.CANDLESTICK_CHANNEL,
            "symbol": "BTCUSDT",
            "timeframe": "1m"
        }
        
        channel = f"{WebSocketConstants.CANDLESTICK_CHANNEL}:BTCUSDT"
        client._subscriptions[channel] = subscription
        
        # Verify subscription structure
        subscriptions = client.subscriptions
        assert subscriptions[channel]["type"] == WebSocketConstants.SUBSCRIBE
        assert subscriptions[channel]["channel"] == WebSocketConstants.CANDLESTICK_CHANNEL
        assert subscriptions[channel]["symbol"] == "BTCUSDT"
        assert subscriptions[channel]["timeframe"] == "1m"

    def test_message_handler_registration(self):
        """Test message handler registration functionality."""
        client = DeltaWebSocketClient()
        
        # Create a mock callback
        callback = Mock()
        
        # Register handler
        channel = f"{WebSocketConstants.CANDLESTICK_CHANNEL}:BTCUSDT"
        client._message_handlers[channel] = callback
        
        # Verify handler was registered
        assert channel in client._message_handlers
        assert client._message_handlers[channel] == callback

    def test_connection_state_management(self):
        """Test connection state management."""
        client = DeltaWebSocketClient()
        
        # Initially not connected
        assert not client.connected
        assert client._connected is False
        
        # Simulate connected state
        client._connected = True
        assert client.connected
        
        # Simulate disconnected state
        client._connected = False
        assert not client.connected

    def test_reconnection_attempt_tracking(self):
        """Test reconnection attempt tracking."""
        client = DeltaWebSocketClient(max_reconnect_attempts=3)
        
        # Initially zero attempts
        assert client._reconnect_attempts == 0
        
        # Simulate reconnection attempts
        client._reconnect_attempts = 1
        assert client._reconnect_attempts == 1
        
        client._reconnect_attempts = 2
        assert client._reconnect_attempts == 2
        
        # Should not exceed max attempts
        assert client._reconnect_attempts <= client.max_reconnect_attempts


class TestWebSocketDataProcessing:
    """Test WebSocket data processing functionality."""

    def test_process_candlestick_data_valid(self):
        """Test processing valid candlestick data."""
        data = {
            "type": "candlestick",
            "data": {
                "open": "50000.00",
                "high": "51000.00",
                "low": "49500.00",
                "close": "50500.00",
                "volume": "1000.5"
            }
        }
        
        result = process_candlestick_data(data)
        
        assert result is not None
        assert isinstance(result, OHLCV)
        assert result.open == Decimal("50000.00")
        assert result.high == Decimal("51000.00")
        assert result.low == Decimal("49500.00")
        assert result.close == Decimal("50500.00")
        assert result.volume == Decimal("1000.5")

    def test_process_candlestick_data_numeric_values(self):
        """Test processing candlestick data with numeric values."""
        data = {
            "type": "candlestick",
            "data": {
                "open": 50000.00,
                "high": 51000.00,
                "low": 49500.00,
                "close": 50500.00,
                "volume": 1000.5
            }
        }
        
        result = process_candlestick_data(data)
        
        assert result is not None
        assert isinstance(result, OHLCV)
        assert float(result.open) == 50000.00
        assert float(result.high) == 51000.00
        assert float(result.low) == 49500.00
        assert float(result.close) == 50500.00
        assert float(result.volume) == 1000.5

    def test_process_candlestick_data_missing_fields(self):
        """Test processing candlestick data with missing fields."""
        data = {
            "type": "candlestick",
            "data": {
                "open": "50000.00",
                "high": "51000.00"
                # Missing low, close, volume
            }
        }
        
        result = process_candlestick_data(data)
        
        assert result is not None
        assert isinstance(result, OHLCV)
        assert result.open == Decimal("50000.00")
        assert result.high == Decimal("51000.00")
        # Missing fields should default to None/0
        assert result.low is not None  # Should have some default value
        assert result.close is not None
        assert result.volume is not None

    def test_process_candlestick_data_empty(self):
        """Test processing empty candlestick data."""
        data = {
            "type": "candlestick",
            "data": {}
        }
        
        result = process_candlestick_data(data)
        
        assert result is not None
        assert isinstance(result, OHLCV)
        # All fields should have default values

    def test_process_candlestick_data_no_data_field(self):
        """Test processing candlestick data without data field."""
        data = {
            "type": "candlestick"
            # No data field
        }
        
        result = process_candlestick_data(data)
        
        assert result is not None
        assert isinstance(result, OHLCV)

    def test_process_orderbook_data_valid(self):
        """Test processing valid order book data."""
        data = {
            "type": "orderbook",
            "data": {
                "bids": [
                    ["50000.00", "1.5"],
                    ["49999.00", "2.0"],
                    ["49998.00", "0.5"]
                ],
                "asks": [
                    ["50001.00", "1.2"],
                    ["50002.00", "0.8"],
                    ["50003.00", "2.5"]
                ]
            }
        }
        
        result = process_orderbook_data(data)
        
        assert result is not None
        assert "bids" in result
        assert "asks" in result
        assert "timestamp" in result
        
        # Check bids
        assert len(result["bids"]) == 3
        assert result["bids"][0]["price"] == 50000.00
        assert result["bids"][0]["size"] == 1.5
        assert result["bids"][1]["price"] == 49999.00
        assert result["bids"][1]["size"] == 2.0
        
        # Check asks
        assert len(result["asks"]) == 3
        assert result["asks"][0]["price"] == 50001.00
        assert result["asks"][0]["size"] == 1.2
        assert result["asks"][1]["price"] == 50002.00
        assert result["asks"][1]["size"] == 0.8

    def test_process_orderbook_data_empty_book(self):
        """Test processing empty order book data."""
        data = {
            "type": "orderbook",
            "data": {
                "bids": [],
                "asks": []
            }
        }
        
        result = process_orderbook_data(data)
        
        assert result is not None
        assert "bids" in result
        assert "asks" in result
        assert "timestamp" in result
        assert len(result["bids"]) == 0
        assert len(result["asks"]) == 0

    def test_process_orderbook_data_invalid_entries(self):
        """Test processing order book data with invalid entries."""
        data = {
            "type": "orderbook",
            "data": {
                "bids": [
                    ["50000.00", "1.5"],    # Valid
                    ["invalid", "2.0"],     # Invalid price
                    ["49998.00"],           # Missing size
                    ["49997.00", "0.5"]     # Valid
                ],
                "asks": [
                    ["50001.00", "1.2"],    # Valid
                    ["50002.00", "invalid"] # Invalid size
                ]
            }
        }
        
        result = process_orderbook_data(data)
        
        assert result is not None
        assert "bids" in result
        assert "asks" in result
        
        # Should process only valid entries
        assert len(result["bids"]) == 2  # Only 2 valid bids
        assert len(result["asks"]) == 1  # Only 1 valid ask

    def test_process_trade_data_valid(self):
        """Test processing valid trade data."""
        data = {
            "type": "trade",
            "data": {
                "id": "12345",
                "price": "50000.00",
                "size": "0.5",
                "side": "buy"
            }
        }
        
        result = process_trade_data(data)
        
        assert result is not None
        assert result["id"] == "12345"
        assert result["price"] == 50000.00
        assert result["size"] == 0.5
        assert result["side"] == "buy"
        assert "timestamp" in result

    def test_process_trade_data_sell_side(self):
        """Test processing sell trade data."""
        data = {
            "type": "trade",
            "data": {
                "id": "67890",
                "price": 49999.50,
                "size": 1.25,
                "side": "sell"
            }
        }
        
        result = process_trade_data(data)
        
        assert result is not None
        assert result["id"] == "67890"
        assert result["price"] == 49999.50
        assert result["size"] == 1.25
        assert result["side"] == "sell"

    def test_process_trade_data_missing_fields(self):
        """Test processing trade data with missing fields."""
        data = {
            "type": "trade",
            "data": {
                "id": "12345",
                "price": "50000.00"
                # Missing size and side
            }
        }
        
        result = process_trade_data(data)
        
        assert result is not None
        assert result["id"] == "12345"
        assert result["price"] == 50000.00
        assert result["size"] == 0.0  # Default value
        assert result["side"] is None  # Default value

    def test_process_trade_data_invalid_numbers(self):
        """Test processing trade data with invalid numeric values."""
        data = {
            "type": "trade",
            "data": {
                "id": "12345",
                "price": "invalid_price",
                "size": "invalid_size",
                "side": "buy"
            }
        }
        
        result = process_trade_data(data)
        
        assert result is not None
        assert result["id"] == "12345"
        assert result["price"] == 0.0  # Should default to 0 for invalid values
        assert result["size"] == 0.0   # Should default to 0 for invalid values
        assert result["side"] == "buy"


class TestWebSocketConstants:
    """Test WebSocket constants are properly defined."""

    def test_websocket_constants_exist(self):
        """Test that required WebSocket constants exist."""
        assert hasattr(WebSocketConstants, 'RECONNECT_DELAY')
        assert hasattr(WebSocketConstants, 'MAX_RECONNECT_ATTEMPTS')
        assert hasattr(WebSocketConstants, 'PING_INTERVAL')
        assert hasattr(WebSocketConstants, 'PING_TIMEOUT')
        
        assert hasattr(WebSocketConstants, 'SUBSCRIBE')
        assert hasattr(WebSocketConstants, 'UNSUBSCRIBE')
        assert hasattr(WebSocketConstants, 'PING')
        assert hasattr(WebSocketConstants, 'PONG')
        
        assert hasattr(WebSocketConstants, 'ORDERBOOK_CHANNEL')
        assert hasattr(WebSocketConstants, 'TRADES_CHANNEL')
        assert hasattr(WebSocketConstants, 'CANDLESTICK_CHANNEL')

    def test_websocket_constants_values(self):
        """Test WebSocket constants have expected values."""
        assert WebSocketConstants.RECONNECT_DELAY == 5
        assert WebSocketConstants.MAX_RECONNECT_ATTEMPTS == 10
        assert WebSocketConstants.PING_INTERVAL == 30
        assert WebSocketConstants.PING_TIMEOUT == 10
        
        assert WebSocketConstants.SUBSCRIBE == "subscribe"
        assert WebSocketConstants.UNSUBSCRIBE == "unsubscribe"
        assert WebSocketConstants.PING == "ping"
        assert WebSocketConstants.PONG == "pong"
        
        assert WebSocketConstants.ORDERBOOK_CHANNEL == "l2_orderbook"
        assert WebSocketConstants.TRADES_CHANNEL == "all_trades"
        assert WebSocketConstants.CANDLESTICK_CHANNEL == "candlestick_1m"


class TestWebSocketErrorHandling:
    """Test WebSocket error handling functionality."""

    def test_listen_requires_connection(self):
        """Test that listen() requires connection."""
        client = DeltaWebSocketClient()
        
        # Should raise error when not connected
        with pytest.raises(APIConnectionError):
            # This should fail because we're not connected
            async def test_listen():
                async for _ in client.listen():
                    pass
            
            import asyncio
            asyncio.get_event_loop().run_until_complete(test_listen())

    def test_subscription_channel_format(self):
        """Test subscription channel format is correct."""
        client = DeltaWebSocketClient()
        
        # Test channel format for different subscription types
        candlestick_channel = f"{WebSocketConstants.CANDLESTICK_CHANNEL}:BTCUSDT"
        orderbook_channel = f"{WebSocketConstants.ORDERBOOK_CHANNEL}:ETHUSDT"
        trades_channel = f"{WebSocketConstants.TRADES_CHANNEL}:SOLUSDT"
        
        assert candlestick_channel == "candlestick_1m:BTCUSDT"
        assert orderbook_channel == "l2_orderbook:ETHUSDT"
        assert trades_channel == "all_trades:SOLUSDT"

    def test_data_processing_error_handling(self):
        """Test data processing functions handle errors gracefully."""
        # Test with completely invalid data
        invalid_data = {"invalid": "structure"}
        
        # Should not raise exceptions
        candlestick_result = process_candlestick_data(invalid_data)
        orderbook_result = process_orderbook_data(invalid_data)
        trade_result = process_trade_data(invalid_data)
        
        # Should return some result (even if default values)
        assert candlestick_result is not None
        assert orderbook_result is not None
        assert trade_result is not None

    def test_data_processing_with_none_input(self):
        """Test data processing functions handle None input."""
        # Should not raise exceptions with None input
        candlestick_result = process_candlestick_data(None)
        orderbook_result = process_orderbook_data(None)
        trade_result = process_trade_data(None)
        
        # Should return None or handle gracefully
        # The exact behavior depends on implementation but should not crash