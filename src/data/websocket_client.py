"""
Delta Exchange WebSocket client for real-time market data streaming.

Provides async WebSocket client for live market data with automatic reconnection,
subscription management, and proper error handling.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from urllib.parse import urljoin

import structlog
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from src.core.config import settings
from src.core.constants import (
    APIEndpoints,
    ErrorCodes,
    RateLimitConstants,
    WebSocketConstants,
)
from src.core.exceptions import APIConnectionError, APITimeoutError, DataValidationError
from src.core.models import OHLCV, MarketData, Symbol, TimeFrame
from src.utils.helpers import get_value, safe_float_conversion, to_decimal

logger = structlog.get_logger(__name__)


class DeltaWebSocketClient:
    """
    Async WebSocket client for Delta Exchange real-time market data.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Subscription management for multiple channels
    - Real-time OHLCV data streaming
    - Order book updates
    - Trade stream processing
    - Connection health monitoring
    - Message queuing during disconnections
    """

    def __init__(
        self,
        base_url: str = None,
        ping_interval: int = None,
        ping_timeout: int = None,
        max_reconnect_attempts: int = None
    ):
        """
        Initialize Delta WebSocket client.
        
        Args:
            base_url: WebSocket base URL
            ping_interval: Ping interval in seconds
            ping_timeout: Ping timeout in seconds
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.base_url = base_url or APIEndpoints.WEBSOCKET_URL
        self.ping_interval = ping_interval or WebSocketConstants.PING_INTERVAL
        self.ping_timeout = ping_timeout or WebSocketConstants.PING_TIMEOUT
        self.max_reconnect_attempts = max_reconnect_attempts or WebSocketConstants.MAX_RECONNECT_ATTEMPTS
        
        # Connection state
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._last_ping_time = 0.0
        
        # Subscription management
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._message_handlers: Dict[str, Callable] = {}
        self._message_queue: List[Dict[str, Any]] = []
        
        # Background tasks
        self._ping_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Delta WebSocket client initialized",
            base_url=self.base_url,
            ping_interval=self.ping_interval,
            max_reconnect_attempts=self.max_reconnect_attempts
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """
        Connect to Delta Exchange WebSocket.
        
        Raises:
            APIConnectionError: Connection failed
            APITimeoutError: Connection timeout
        """
        if self._connected:
            logger.warning("Already connected to WebSocket")
            return
            
        try:
            logger.info("Connecting to Delta Exchange WebSocket", url=self.base_url)
            
            # Create WebSocket connection
            self._websocket = await websockets.connect(
                self.base_url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=10,
                max_size=2**20,  # 1MB max message size
                compression=None  # Disable compression for better performance
            )
            
            self._connected = True
            self._reconnect_attempts = 0
            
            # Start background tasks
            self._ping_task = asyncio.create_task(self._ping_loop())
            
            # Resubscribe to channels if reconnecting
            if self._subscriptions:
                await self._resubscribe_all()
            
            logger.info("Connected to Delta Exchange WebSocket")
            
        except Exception as e:
            logger.error("Failed to connect to WebSocket", error=str(e))
            self._connected = False
            
            if isinstance(e, (ConnectionClosed, InvalidStatusCode)):
                raise APIConnectionError(
                    f"WebSocket connection failed: {str(e)}",
                    error_code=ErrorCodes.API_CONNECTION_FAILED
                )
            else:
                raise APITimeoutError(
                    f"WebSocket connection timeout: {str(e)}",
                    error_code=ErrorCodes.API_TIMEOUT
                )

    async def disconnect(self) -> None:
        """Disconnect from WebSocket and cleanup resources."""
        if not self._connected:
            return
            
        logger.info("Disconnecting from Delta Exchange WebSocket")
        
        # Cancel background tasks
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning("Error closing WebSocket", error=str(e))
            finally:
                self._websocket = None
        
        self._connected = False
        logger.info("Disconnected from Delta Exchange WebSocket")

    async def subscribe_candlestick(
        self, 
        symbol: str, 
        timeframe: str = "1m",
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscribe to candlestick data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            callback: Optional callback function for processing updates
        """
        channel = f"{WebSocketConstants.CANDLESTICK_CHANNEL}:{symbol}"
        
        subscription = {
            "type": WebSocketConstants.SUBSCRIBE,
            "payload": {
                "channels": [
                    {
                        "name": "candlestick_1m",
                        "symbols": [symbol]
                    }
                ]
            }
        }
        
        self._subscriptions[channel] = subscription
        
        if callback:
            self._message_handlers[channel] = callback
        
        if self._connected:
            await self._send_message(subscription)
            
        logger.info(
            "Subscribed to candlestick data",
            symbol=symbol,
            timeframe=timeframe,
            channel=channel
        )

    async def subscribe_orderbook(
        self, 
        symbol: str,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscribe to order book updates for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            callback: Optional callback function for processing updates
        """
        channel = f"{WebSocketConstants.ORDERBOOK_CHANNEL}:{symbol}"
        
        subscription = {
            "type": WebSocketConstants.SUBSCRIBE,
            "payload": {
                "channels": [
                    {
                        "name": WebSocketConstants.ORDERBOOK_CHANNEL,
                        "symbols": [symbol]
                    }
                ]
            }
        }
        
        self._subscriptions[channel] = subscription
        
        if callback:
            self._message_handlers[channel] = callback
        
        if self._connected:
            await self._send_message(subscription)
            
        logger.info(
            "Subscribed to order book data",
            symbol=symbol,
            channel=channel
        )

    async def subscribe_trades(
        self, 
        symbol: str,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscribe to trade stream for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            callback: Optional callback function for processing updates
        """
        channel = f"{WebSocketConstants.TRADES_CHANNEL}:{symbol}"
        
        subscription = {
            "type": WebSocketConstants.SUBSCRIBE,
            "payload": {
                "channels": [
                    {
                        "name": WebSocketConstants.TRADES_CHANNEL,
                        "symbols": [symbol]
                    }
                ]
            }
        }
        
        self._subscriptions[channel] = subscription
        
        if callback:
            self._message_handlers[channel] = callback
        
        if self._connected:
            await self._send_message(subscription)
            
        logger.info(
            "Subscribed to trade stream",
            symbol=symbol,
            channel=channel
        )

    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel to unsubscribe from
        """
        if channel not in self._subscriptions:
            logger.warning("Not subscribed to channel", channel=channel)
            return
            
        subscription = self._subscriptions[channel].copy()
        subscription["type"] = WebSocketConstants.UNSUBSCRIBE
        
        if self._connected:
            await self._send_message(subscription)
        
        # Remove from subscriptions and handlers
        del self._subscriptions[channel]
        if channel in self._message_handlers:
            del self._message_handlers[channel]
            
        logger.info("Unsubscribed from channel", channel=channel)

    async def listen(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Listen for incoming WebSocket messages.
        
        Yields:
            Parsed message data
            
        Raises:
            APIConnectionError: Connection lost
        """
        if not self._connected:
            raise APIConnectionError(
                "WebSocket not connected",
                error_code=ErrorCodes.API_CONNECTION_FAILED
            )
        
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    
                    # Process message
                    await self._process_message(data)
                    
                    yield data
                    
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON message received", error=str(e))
                    continue
                except Exception as e:
                    logger.error("Error processing message", error=str(e))
                    continue
                    
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._connected = False
            
            # Attempt reconnection
            if self._reconnect_attempts < self.max_reconnect_attempts:
                self._reconnect_task = asyncio.create_task(self._reconnect())
            
            raise APIConnectionError(
                "WebSocket connection lost",
                error_code=ErrorCodes.API_CONNECTION_FAILED
            )

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send message to WebSocket."""
        if not self._connected or not self._websocket:
            logger.warning("Cannot send message: not connected")
            return
            
        try:
            await self._websocket.send(json.dumps(message))
            logger.debug("Sent WebSocket message", message=message)
        except Exception as e:
            logger.error("Failed to send WebSocket message", error=str(e))

    async def _process_message(self, data: Dict[str, Any]) -> None:
        """Process incoming WebSocket message."""
        try:
            message_type = get_value(data, 'type')
            channel = get_value(data, 'channel')
            symbol = get_value(data, 'symbol')
            
            if message_type == WebSocketConstants.PONG:
                self._last_ping_time = time.time()
                return
            
            # Find matching handler based on message type and symbol
            handler_key = None
            
            # For candlestick messages, use type + symbol
            if message_type == 'candlestick_1m' and symbol:
                handler_key = f"{WebSocketConstants.CANDLESTICK_CHANNEL}:{symbol}"
            # For other messages, use channel + symbol
            elif channel and symbol:
                handler_key = f"{channel}:{symbol}"
            # Fallback: try to match any handler with the symbol
            elif symbol:
                for key in self._message_handlers:
                    if key.endswith(f":{symbol}"):
                        handler_key = key
                        break
            
            if handler_key and handler_key in self._message_handlers:
                callback = self._message_handlers[handler_key]
                await callback(data)
            else:
                logger.debug("No handler for message", type=message_type, channel=channel, symbol=symbol)
                
        except Exception as e:
            logger.error("Error processing WebSocket message", error=str(e))

    async def _ping_loop(self) -> None:
        """Background task for sending ping messages."""
        while self._connected:
            try:
                await asyncio.sleep(self.ping_interval)
                
                if self._connected and self._websocket:
                    await self._websocket.ping()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in ping loop", error=str(e))

    async def _reconnect(self) -> None:
        """Attempt to reconnect to WebSocket."""
        while self._reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = min(
                WebSocketConstants.RECONNECT_DELAY * (2 ** self._reconnect_attempts),
                60  # Max 60 seconds delay
            )
            
            logger.info(
                "Attempting WebSocket reconnection",
                attempt=self._reconnect_attempts,
                delay=delay
            )
            
            await asyncio.sleep(delay)
            
            try:
                await self.connect()
                logger.info("WebSocket reconnected successfully")
                return
            except Exception as e:
                logger.warning(
                    "WebSocket reconnection failed",
                    attempt=self._reconnect_attempts,
                    error=str(e)
                )
        
        logger.error("Max reconnection attempts reached")

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all channels after reconnection."""
        for channel, subscription in self._subscriptions.items():
            try:
                await self._send_message(subscription)
                logger.debug("Resubscribed to channel", channel=channel)
            except Exception as e:
                logger.error("Failed to resubscribe to channel", channel=channel, error=str(e))

    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    @property
    def subscriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get current subscriptions."""
        return self._subscriptions.copy()

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self._subscriptions)


# Helper functions for processing WebSocket data

def process_candlestick_data(data: Dict[str, Any]) -> Optional[OHLCV]:
    """
    Process candlestick data from WebSocket message.
    
    Args:
        data: WebSocket message data in Delta Exchange format
        
    Returns:
        OHLCV object or None if processing fails
    """
    try:
        if not data:
            return None
            
        # Check if this is Delta Exchange candlestick data
        if data.get('type') != 'candlestick_1m':
            return None
        
        # Extract OHLCV values directly from the data (not nested in 'data' field)
        open_val = get_value(data, 'open')
        high_val = get_value(data, 'high')
        low_val = get_value(data, 'low')
        close_val = get_value(data, 'close')
        volume_val = get_value(data, 'volume')
        
        # Validate we have the required fields
        if not all([open_val is not None, high_val is not None, 
                   low_val is not None, close_val is not None]):
            logger.warning("Missing required OHLCV fields in candlestick data", data=data)
            return None
        
        # Extract timestamp with proper error handling
        timestamp = datetime.now(timezone.utc)  # Default fallback
        timestamp_raw = data.get('timestamp')
        
        if timestamp_raw:
            try:
                # Convert to numeric value
                if isinstance(timestamp_raw, str):
                    timestamp_value = float(timestamp_raw)
                else:
                    timestamp_value = float(timestamp_raw)
                
                # Determine format based on magnitude and convert
                if 1000000000 <= timestamp_value <= 9999999999:
                    # Seconds (10 digits)
                    timestamp = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
                elif 1000000000000 <= timestamp_value <= 9999999999999:
                    # Milliseconds (13 digits)
                    timestamp = datetime.fromtimestamp(timestamp_value / 1000, tz=timezone.utc)
                elif 1000000000000000 <= timestamp_value <= 9999999999999999:
                    # Microseconds (16 digits)
                    timestamp = datetime.fromtimestamp(timestamp_value / 1000000, tz=timezone.utc)
                elif timestamp_value >= 1000000000000000000:
                    # Nanoseconds (19+ digits)
                    timestamp = datetime.fromtimestamp(timestamp_value / 1000000000, tz=timezone.utc)
                else:
                    # Unknown format - treat as seconds
                    timestamp = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
                    
                # Validate reasonable year range
                if not (1970 <= timestamp.year <= 2100):
                    logger.warning(f"Invalid timestamp year: {timestamp.year}, using current time")
                    timestamp = datetime.now(timezone.utc)
                    
            except (ValueError, OSError, OverflowError) as e:
                logger.warning(f"Failed to parse timestamp {timestamp_raw}: {e}, using current time")
                timestamp = datetime.now(timezone.utc)
        
        # Convert to Decimal and create OHLCV object with proper timestamp
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=to_decimal(open_val),
            high=to_decimal(high_val),
            low=to_decimal(low_val),
            close=to_decimal(close_val),
            volume=to_decimal(volume_val or 0.1)  # Small default if volume missing
        )
        
        return ohlcv
        
    except Exception as e:
        logger.error("Error processing candlestick data", error=str(e))
        return None

def process_orderbook_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process order book data from WebSocket message.
    
    Args:
        data: WebSocket message data
        
    Returns:
        Processed order book data or None if processing fails
    """
    try:
        orderbook = get_value(data, 'data', {})
        
        processed = {
            'bids': [],
            'asks': [],
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Process bids
        bids = get_value(orderbook, 'bids', [])
        for bid in bids:
            if len(bid) >= 2:
                price = safe_float_conversion(bid[0])
                size = safe_float_conversion(bid[1])
                # Only add valid entries
                if price > 0 and size > 0:
                    processed['bids'].append({
                        'price': price,
                        'size': size
                    })
        
        # Process asks
        asks = get_value(orderbook, 'asks', [])
        for ask in asks:
            if len(ask) >= 2:
                price = safe_float_conversion(ask[0])
                size = safe_float_conversion(ask[1])
                # Only add valid entries
                if price > 0 and size > 0:
                    processed['asks'].append({
                        'price': price,
                        'size': size
                    })
        
        return processed
        
    except Exception as e:
        logger.error("Error processing order book data", error=str(e))
        return None

def process_trade_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process trade data from WebSocket message.
    
    Args:
        data: WebSocket message data
        
    Returns:
        Processed trade data or None if processing fails
    """
    try:
        trade = get_value(data, 'data', {})
        
        processed = {
            'id': get_value(trade, 'id'),
            'price': safe_float_conversion(get_value(trade, 'price', 0)),
            'size': safe_float_conversion(get_value(trade, 'size', 0)),
            'side': get_value(trade, 'side'),
            'timestamp': datetime.now(timezone.utc)
        }
        
        return processed
        
    except Exception as e:
        logger.error("Error processing trade data", error=str(e))
        return None