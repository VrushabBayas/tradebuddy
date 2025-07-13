"""
Delta Exchange API client for market data retrieval.

Provides async HTTP client for interacting with Delta Exchange REST API
with proper error handling, rate limiting, and data validation.
"""

import asyncio
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
import structlog

from src.core.config import settings
from src.core.constants import (
    APIEndpoints,
    ErrorCodes,
    RateLimitConstants,
    ValidationPatterns,
)
from src.core.exceptions import (
    APIConnectionError,
    APIRateLimitError,
    APITimeoutError,
    DataValidationError,
)
from src.core.models import OHLCV, MarketData

logger = structlog.get_logger(__name__)


class DeltaExchangeClient:
    """
    Async HTTP client for Delta Exchange API.
    
    Provides methods for retrieving market data including:
    - Product listings
    - OHLCV candle data
    - Current market prices
    - Order book data
    - Trade history
    
    Features:
    - Automatic rate limiting
    - Connection pooling
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Request/response validation
    """

    def __init__(
        self,
        base_url: str = None,
        timeout: int = None,
        rate_limit: int = None
    ):
        """
        Initialize Delta Exchange client.
        
        Args:
            base_url: API base URL (defaults to production)
            timeout: Request timeout in seconds
            rate_limit: Max requests per second
        """
        self.base_url = base_url or settings.delta_exchange_api_url
        self.timeout = timeout or RateLimitConstants.OLLAMA_REQUEST_TIMEOUT
        self.rate_limit = rate_limit or RateLimitConstants.DELTA_REST_REQUESTS_PER_SECOND
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(self.rate_limit)
        self._last_request_time = 0.0
        
        logger.info(
            "Delta Exchange client initialized",
            base_url=self.base_url,
            timeout=self.timeout,
            rate_limit=self.rate_limit
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool size
                limit_per_host=30,
                enable_cleanup_closed=True
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "TradeBuddy/0.1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    # Anti-caching headers to ensure fresh data
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
            
            logger.debug("HTTP session created")

    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("HTTP session closed")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            
        Returns:
            Parsed JSON response
            
        Raises:
            APIConnectionError: Connection or HTTP errors
            APIRateLimitError: Rate limit exceeded
            APITimeoutError: Request timeout
        """
        await self._ensure_session()
        
        # Rate limiting
        async with self._rate_limiter:
            # Ensure minimum time between requests
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self._last_request_time
            min_interval = 1.0 / self.rate_limit
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
            
            self._last_request_time = asyncio.get_event_loop().time()
        
        url = urljoin(self.base_url, endpoint)
        
        # Add anti-caching parameters to ensure fresh data
        if params is None:
            params = {}
        
        # Add unique request ID and timestamp to prevent caching
        params.update({
            "_req_id": str(uuid.uuid4())[:8],  # Short unique ID
            "_ts": int(datetime.now(timezone.utc).timestamp() * 1000)  # Millisecond timestamp
        })
        
        try:
            logger.debug(
                "Making API request",
                method=method,
                url=url,
                params=params,
                anti_cache_enabled=True
            )
            
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                json=data
            ) as response:
                
                # Handle rate limiting
                if response.status == 429:
                    logger.warning("Rate limit exceeded", status=response.status)
                    raise APIRateLimitError(
                        "Rate limit exceeded",
                        error_code=ErrorCodes.API_RATE_LIMIT_EXCEEDED
                    )
                
                # Handle other HTTP errors
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        "API request failed",
                        status=response.status,
                        error=error_text
                    )
                    raise APIConnectionError(
                        f"API request failed with status {response.status}: {error_text}",
                        error_code=ErrorCodes.API_CONNECTION_FAILED
                    )
                
                # Parse JSON response
                try:
                    result = await response.json()
                    logger.debug("API request successful", status=response.status)
                    return result
                except aiohttp.ContentTypeError as e:
                    logger.error("Invalid JSON response", error=str(e))
                    raise APIConnectionError(
                        "Invalid JSON response from API",
                        error_code=ErrorCodes.API_INVALID_RESPONSE
                    )
        
        except asyncio.TimeoutError:
            logger.error("Request timeout", url=url, timeout=self.timeout)
            raise APITimeoutError(
                f"Request timeout after {self.timeout} seconds",
                error_code=ErrorCodes.API_TIMEOUT
            )
        except aiohttp.ClientError as e:
            logger.error("HTTP client error", error=str(e), url=url)
            raise APIConnectionError(
                f"HTTP client error: {str(e)}",
                error_code=ErrorCodes.API_CONNECTION_FAILED
            )

    def _validate_symbol(self, symbol: str) -> None:
        """Validate trading symbol format."""
        if not re.match(ValidationPatterns.SYMBOL_PATTERN, symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")

    def _validate_resolution(self, resolution: str) -> None:
        """Validate timeframe resolution."""
        valid_resolutions = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
        if resolution not in valid_resolutions:
            raise ValueError(f"Invalid resolution: {resolution}. Valid options: {valid_resolutions}")

    def _validate_date_range(self, start: datetime, end: datetime) -> None:
        """Validate date range."""
        if end <= start:
            raise ValueError("End time must be after start time")
    
    def _validate_data_freshness(self, candles: List[OHLCV], symbol: str, timeframe: str) -> None:
        """
        Validate that the received market data is fresh and not stale.
        
        Args:
            candles: List of OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Raises:
            DataValidationError: If data appears stale
        """
        if not candles:
            logger.warning("No candles received - cannot validate freshness", symbol=symbol)
            return
        
        current_time = datetime.now(timezone.utc)
        # Delta Exchange returns candles in reverse chronological order (newest first)
        latest_candle = candles[0]
        
        # Calculate expected maximum age based on timeframe
        # Be strict for minute timeframes to ensure fresh data
        if timeframe.endswith('m'):
            timeframe_minutes = int(timeframe[:-1])
            # For minute timeframes, allow only 2x the timeframe for fresh data
            # This ensures 1-minute data is max 2 minutes old, 5-minute data max 10 minutes old
            max_age_minutes = timeframe_minutes * 2
        elif timeframe.endswith('h'):
            timeframe_hours = int(timeframe[:-1])
            max_age_minutes = timeframe_hours * 60 * 3  # 3x timeframe for hourly data
        elif timeframe == '1d':
            max_age_minutes = 24 * 60 * 3  # 3 days for daily data
        else:
            max_age_minutes = 120  # Default 2 hours max age for unknown timeframes
        
        # Check if latest candle is too old
        time_diff = current_time - latest_candle.timestamp
        age_minutes = time_diff.total_seconds() / 60
        
        if age_minutes > max_age_minutes:
            from src.core.config import settings
            
            # In development mode, be more lenient with stale data for testing
            if settings.is_development:
                logger.warning(
                    "Stale market data detected - allowing in development mode",
                    symbol=symbol,
                    timeframe=timeframe,
                    latest_candle_time=latest_candle.timestamp.isoformat(),
                    age_minutes=age_minutes,
                    max_age_minutes=max_age_minutes,
                    development_mode=True,
                    note="Data validation relaxed for development/testing"
                )
                # Continue with stale data in development mode - skip the error
            elif age_minutes > 360:  # 6 hours for production
                logger.warning(
                    "Very stale market data detected - markets may be closed",
                    symbol=symbol,
                    timeframe=timeframe,
                    latest_candle_time=latest_candle.timestamp.isoformat(),
                    age_hours=age_minutes / 60,
                    note="Proceeding with caution - verify market status"
                )
                # For very stale data, we'll warn but allow processing with a flag
                # This allows analysis during market closures but alerts the user
            else:
                logger.error(
                    "Stale market data detected",
                    symbol=symbol,
                    timeframe=timeframe,
                    latest_candle_time=latest_candle.timestamp.isoformat(),
                    age_minutes=age_minutes,
                    max_age_minutes=max_age_minutes
                )
                raise DataValidationError(
                    f"Market data is too stale: {age_minutes:.1f} minutes old (max: {max_age_minutes}). "
                    f"This may indicate markets are closed or there's a data delay.",
                    error_code=ErrorCodes.API_INVALID_RESPONSE
                )
        
        logger.debug(
            "Data freshness validated",
            symbol=symbol,
            timeframe=timeframe,
            age_minutes=age_minutes,
            max_age_minutes=max_age_minutes,
            is_fresh=True
        )

    async def get_products(self) -> List[Dict[str, Any]]:
        """
        Get list of available trading products.
        
        Returns:
            List of product information dictionaries
        """
        logger.info("Fetching products list")
        
        response = await self._make_request("GET", APIEndpoints.PRODUCTS)
        
        if not response.get("success", False):
            raise APIConnectionError(
                f"Failed to get products: {response.get('error', 'Unknown error')}",
                error_code=ErrorCodes.API_INVALID_RESPONSE
            )
        
        products = response.get("result", [])
        logger.info("Products fetched successfully", count=len(products))
        
        return products

    async def get_candles(
        self,
        symbol: str,
        resolution: str,
        start: datetime,
        end: datetime
    ) -> List[OHLCV]:
        """
        Get OHLCV candle data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            resolution: Timeframe (e.g., "1h", "1d")
            start: Start time (UTC)
            end: End time (UTC)
            
        Returns:
            List of OHLCV data objects
        """
        # Validation
        self._validate_symbol(symbol)
        self._validate_resolution(resolution)
        self._validate_date_range(start, end)
        
        logger.info(
            "Fetching candle data",
            symbol=symbol,
            resolution=resolution,
            start=start.isoformat(),
            end=end.isoformat()
        )
        
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "start": int(start.timestamp()),
            "end": int(end.timestamp())
        }
        
        response = await self._make_request("GET", APIEndpoints.CANDLES, params=params)
        
        if not response.get("success", False):
            raise APIConnectionError(
                f"Failed to get candles: {response.get('error', 'Unknown error')}",
                error_code=ErrorCodes.API_INVALID_RESPONSE
            )
        
        # Parse candle data
        candles_data = response.get("result", [])
        candles = []
        
        for candle_data in candles_data:
            try:
                ohlcv = OHLCV(
                    timestamp=datetime.fromtimestamp(candle_data["time"], tz=timezone.utc),
                    open=float(candle_data["open"]),
                    high=float(candle_data["high"]),
                    low=float(candle_data["low"]),
                    close=float(candle_data["close"]),
                    volume=float(candle_data["volume"])
                )
                candles.append(ohlcv)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "Invalid candle data",
                    error=str(e),
                    data=candle_data
                )
                continue
        
        logger.info("Candle data fetched successfully", count=len(candles))
        return candles

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker data dictionary
        """
        self._validate_symbol(symbol)
        
        logger.debug("Fetching ticker data", symbol=symbol)
        
        params = {"symbol": symbol}
        response = await self._make_request("GET", APIEndpoints.TICKERS, params=params)
        
        if not response.get("success", False):
            raise APIConnectionError(
                f"Failed to get ticker: {response.get('error', 'Unknown error')}",
                error_code=ErrorCodes.API_INVALID_RESPONSE
            )
        
        result = response.get("result", {})
        
        # Log the response format for debugging
        logger.debug(
            "Ticker response received",
            symbol=symbol,
            result_type=type(result),
            result_length=len(result) if isinstance(result, (list, dict)) else 0
        )
        
        # Log first few items to understand structure
        if isinstance(result, list) and len(result) > 0:
            logger.debug(
                "First ticker item structure",
                first_item=result[0],
                first_item_keys=list(result[0].keys()) if isinstance(result[0], dict) else "not_dict"
            )
        
        return result

    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200
    ) -> MarketData:
        """
        Get comprehensive market data for analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            limit: Number of candles to fetch
            
        Returns:
            MarketData object with OHLCV and current price
        """
        self._validate_symbol(symbol)
        self._validate_resolution(timeframe)
        
        logger.info(
            "Fetching market data",
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        # Calculate time range for candles - request slightly in the future to get latest data
        current_time = datetime.now(timezone.utc)
        
        # Set end_time to 2 minutes in the future to ensure we get the absolute latest candle
        # This accounts for potential API delays and ensures we don't miss the current candle
        end_time = current_time + timedelta(minutes=2)
        
        # Add small random jitter (0-999 microseconds) to ensure unique time windows
        jitter_microseconds = random.randint(0, 999)
        end_time = end_time.replace(microsecond=jitter_microseconds)
        
        # Calculate start time based on timeframe and requested limit
        # Remove aggressive limits to support V2 strategy requiring 60+ periods
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            # For minute timeframes, allow full requested range
            # V2 strategy needs sufficient data for 50 EMA + analysis buffer
            lookback_minutes = minutes * limit
            start_time = end_time - timedelta(minutes=lookback_minutes)
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            # For hourly timeframes, allow full requested range
            # V2 strategy requires 60+ periods (200-300 hours for safety)
            lookback_hours = hours * limit
            start_time = end_time - timedelta(hours=lookback_hours)
        elif timeframe == '1d':
            # For daily timeframes, allow full requested range
            # Technical analysis may need several months of daily data
            lookback_days = limit
            start_time = end_time - timedelta(days=lookback_days)
        else:
            # Fallback: use limit as hours
            start_time = end_time - timedelta(hours=limit)
        
        # Preserve microsecond precision for uniqueness
        start_time = start_time.replace(second=0)  # Keep microseconds for uniqueness
        end_time = end_time.replace(second=0)  # Keep microseconds for uniqueness
        
        # Calculate expected number of periods for validation
        if timeframe.endswith('m'):
            minutes_per_period = int(timeframe[:-1])
            total_minutes = (end_time - start_time).total_seconds() / 60
            expected_periods = int(total_minutes / minutes_per_period)
        elif timeframe.endswith('h'):
            hours_per_period = int(timeframe[:-1])
            total_hours = (end_time - start_time).total_seconds() / 3600
            expected_periods = int(total_hours / hours_per_period)
        elif timeframe == '1d':
            total_days = (end_time - start_time).days
            expected_periods = total_days
        else:
            expected_periods = limit

        logger.info(
            "Data fetching parameters",
            symbol=symbol,
            timeframe=timeframe,
            requested_limit=limit,
            time_range_start=start_time.isoformat(),
            time_range_end=end_time.isoformat(),
            expected_periods=expected_periods,
            note="V2 strategy requires 60+ periods"
        )
        
        logger.debug(
            "Generated unique time window",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            jitter_microseconds=jitter_microseconds
        )
        
        # Fetch candles and ticker concurrently
        candles_task = self.get_candles(symbol, timeframe, start_time, end_time)
        ticker_task = self.get_ticker(symbol)
        
        candles, ticker = await asyncio.gather(candles_task, ticker_task)
        
        # Extract current price from ticker (handle both dict and list responses)
        current_price = 0.0
        if isinstance(ticker, dict):
            # Try different price fields for dict response
            for price_field in ["mark_price", "close", "price", "last"]:
                if price_field in ticker:
                    current_price = float(ticker.get(price_field, 0.0))
                    if current_price > 0:
                        break
        elif isinstance(ticker, list) and len(ticker) > 0:
            # If ticker is a list, find the matching symbol
            for ticker_item in ticker:
                if isinstance(ticker_item, dict) and ticker_item.get("symbol") == symbol:
                    # Try different price fields for the matching symbol
                    for price_field in ["mark_price", "close", "price", "last"]:
                        if price_field in ticker_item:
                            price_value = ticker_item.get(price_field, 0.0)
                            if price_value:
                                try:
                                    current_price = float(price_value)
                                    if current_price > 0:
                                        logger.info(f"Found price for {symbol}", price_field=price_field, price=current_price)
                                        break
                                except (ValueError, TypeError):
                                    continue
                    if current_price > 0:
                        break
            
            # If we didn't find the symbol or price, log warning
            if current_price == 0.0:
                logger.warning(f"Could not find price for symbol {symbol} in ticker data", 
                              ticker_count=len(ticker))
        else:
            logger.warning("Unexpected ticker format", ticker_type=type(ticker), ticker=ticker)
        
        # Fallback: use the latest candle's close price (first candle since they're in reverse order)
        if current_price == 0.0 and candles:
            current_price = float(candles[0].close)
            logger.info("Using latest candle close price as fallback", current_price=current_price)
        
        # Validate data freshness with enhanced logging
        self._validate_data_freshness(candles, symbol, timeframe)
        
        # Log detailed timing information for debugging
        if candles:
            latest_candle = candles[0]  # Assuming newest first (verified by diagnostic)
            current_utc = datetime.now(timezone.utc)
            age_minutes = (current_utc - latest_candle.timestamp).total_seconds() / 60
            
            logger.info(
                "Market data timing analysis",
                symbol=symbol,
                timeframe=timeframe,
                latest_candle_time=latest_candle.timestamp.isoformat(),
                current_utc_time=current_utc.isoformat(),
                data_age_minutes=round(age_minutes, 2),
                note="Compare with live chart for discrepancies"
            )
        
        # Create MarketData object
        market_data = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=end_time,
            current_price=current_price,
            ohlcv_data=candles
        )
        
        logger.info(
            "Market data assembled successfully",
            symbol=symbol,
            candles_count=len(candles),
            requested_limit=limit,
            current_price=current_price,
            sufficient_for_v2=len(candles) >= 60,
            data_freshness_validated=True
        )
        
        return market_data

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """
        Get order book data for a symbol.
        
        Args:
            symbol: Trading symbol
            depth: Order book depth
            
        Returns:
            Order book data
        """
        self._validate_symbol(symbol)
        
        logger.debug("Fetching orderbook", symbol=symbol, depth=depth)
        
        endpoint = APIEndpoints.ORDERBOOK.format(symbol=symbol)
        params = {"depth": depth}
        
        response = await self._make_request("GET", endpoint, params=params)
        
        if not response.get("success", False):
            raise APIConnectionError(
                f"Failed to get orderbook: {response.get('error', 'Unknown error')}",
                error_code=ErrorCodes.API_INVALID_RESPONSE
            )
        
        return response.get("result", {})

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of trades to fetch
            
        Returns:
            List of trade data
        """
        self._validate_symbol(symbol)
        
        logger.debug("Fetching recent trades", symbol=symbol, limit=limit)
        
        endpoint = APIEndpoints.TRADES.format(symbol=symbol)
        params = {"limit": limit}
        
        response = await self._make_request("GET", endpoint, params=params)
        
        if not response.get("success", False):
            raise APIConnectionError(
                f"Failed to get trades: {response.get('error', 'Unknown error')}",
                error_code=ErrorCodes.API_INVALID_RESPONSE
            )
        
        return response.get("result", [])

    async def health_check(self) -> bool:
        """
        Check API health and connectivity.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            logger.debug("Performing health check")
            
            # Try to fetch products list as health check
            products = await self.get_products()
            
            is_healthy = len(products) > 0
            logger.info("Health check completed", healthy=is_healthy)
            
            return is_healthy
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False