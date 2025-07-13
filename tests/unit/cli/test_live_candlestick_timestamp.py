"""
Test live candlestick timestamp display functionality.

Following TDD approach - tests for timestamp extraction and display in real-time candlestick output.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from rich.console import Console

from src.cli.realtime import RealTimeAnalyzer
from src.core.models import RealTimeConfig, StrategyType, Symbol, TimeFrame
from src.data.delta_client import DeltaExchangeClient
from src.data.websocket_client import DeltaWebSocketClient


class TestLiveCandlestickTimestamp:
    """Test live candlestick timestamp functionality."""

    @pytest.fixture
    def console(self):
        """Create console for testing."""
        return Console()

    @pytest.fixture
    def delta_client(self):
        """Create mock Delta Exchange client."""
        return AsyncMock(spec=DeltaExchangeClient)

    @pytest.fixture
    def websocket_client(self):
        """Create mock WebSocket client."""
        return AsyncMock(spec=DeltaWebSocketClient)

    @pytest.fixture
    def strategies(self):
        """Create mock strategies."""
        return {"ema_crossover": AsyncMock()}

    @pytest.fixture
    def realtime_analyzer(self, console, delta_client, websocket_client, strategies):
        """Create RealTimeAnalyzer instance for testing."""
        return RealTimeAnalyzer(
            console=console,
            delta_client=delta_client,
            websocket_client=websocket_client,
            strategies=strategies
        )

    @pytest.fixture
    def realtime_config(self):
        """Create real-time configuration."""
        return RealTimeConfig(
            strategy=StrategyType.EMA_CROSSOVER,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_MINUTE,
            duration_minutes=5
        )

    @pytest.fixture
    def mock_websocket_data_with_timestamp(self):
        """Mock WebSocket candlestick data with timestamp."""
        return {
            'type': 'candlestick_1m',
            'symbol': 'BTCUSD',
            'open': 117900.0,
            'high': 118000.0,
            'low': 117800.0,
            'close': 117900.0,
            'volume': 35.0,
            'timestamp': 1673632656000  # Unix timestamp in milliseconds
        }

    @pytest.fixture
    def mock_websocket_data_without_timestamp(self):
        """Mock WebSocket candlestick data without timestamp."""
        return {
            'type': 'candlestick_1m',
            'symbol': 'BTCUSD',
            'open': 117900.0,
            'high': 118000.0,
            'low': 117800.0,
            'close': 117900.0,
            'volume': 35.0
        }

    @pytest.mark.asyncio
    async def test_live_candlestick_displays_timestamp_when_available(
        self, realtime_analyzer, realtime_config, mock_websocket_data_with_timestamp
    ):
        """Test that live candlestick displays timestamp when available in WebSocket data."""
        # Arrange
        realtime_analyzer.config = realtime_config
        realtime_analyzer.analysis_count = 0
        
        with patch.object(realtime_analyzer.console, 'print') as mock_print, \
             patch('src.data.websocket_client.process_candlestick_data') as mock_process:
            
            # Mock successful candlestick processing
            from src.core.models import OHLCV
            mock_ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=117900.0, high=118000.0, low=117800.0, close=117900.0, volume=35.0
            )
            mock_process.return_value = mock_ohlcv
            
            # Act
            await realtime_analyzer.handle_candlestick(
                mock_websocket_data_with_timestamp, 
                MagicMock()
            )
            
            # Assert
            print_calls = [str(call) for call in mock_print.call_args_list]
            print_content = ' '.join(print_calls)
            
            # Should display candlestick header
            assert "LIVE CANDLESTICK #1" in print_content
            
            # Should display timestamp (this is what we're testing for)
            assert "Time:" in print_content or "IST" in print_content
            
            # Should display price and volume
            assert "Close:" in print_content
            assert "117,900.00" in print_content
            assert "Volume:" in print_content
            assert "35.0" in print_content

    @pytest.mark.asyncio
    async def test_live_candlestick_shows_fallback_timestamp_when_websocket_lacks_timestamp(
        self, realtime_analyzer, realtime_config, mock_websocket_data_without_timestamp
    ):
        """Test that live candlestick shows fallback timestamp when WebSocket data lacks timestamp."""
        # Arrange
        realtime_analyzer.config = realtime_config
        realtime_analyzer.analysis_count = 0
        
        with patch.object(realtime_analyzer.console, 'print') as mock_print, \
             patch('src.data.websocket_client.process_candlestick_data') as mock_process:
            
            # Mock successful candlestick processing
            from src.core.models import OHLCV
            mock_ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=117900.0, high=118000.0, low=117800.0, close=117900.0, volume=35.0
            )
            mock_process.return_value = mock_ohlcv
            
            # Act
            await realtime_analyzer.handle_candlestick(
                mock_websocket_data_without_timestamp,
                MagicMock()
            )
            
            # Assert
            print_calls = [str(call) for call in mock_print.call_args_list]
            print_content = ' '.join(print_calls)
            
            # Should still display a timestamp (fallback from OHLCV or current time)
            assert "Time:" in print_content or "IST" in print_content

    @pytest.mark.asyncio
    async def test_timestamp_format_is_user_friendly(
        self, realtime_analyzer, realtime_config, mock_websocket_data_with_timestamp
    ):
        """Test that timestamp is formatted in user-friendly IST format."""
        # Arrange
        realtime_analyzer.config = realtime_config
        realtime_analyzer.analysis_count = 0
        
        with patch.object(realtime_analyzer.console, 'print') as mock_print, \
             patch('src.data.websocket_client.process_candlestick_data') as mock_process, \
             patch('src.utils.helpers.format_ist_time_only') as mock_format:
            
            # Mock the IST formatting function
            mock_format.return_value = "10:27:36 PM IST"
            
            # Mock successful candlestick processing
            from src.core.models import OHLCV
            mock_ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=117900.0, high=118000.0, low=117800.0, close=117900.0, volume=35.0
            )
            mock_process.return_value = mock_ohlcv
            
            # Act
            await realtime_analyzer.handle_candlestick(
                mock_websocket_data_with_timestamp,
                MagicMock()
            )
            
            # Assert
            print_calls = [str(call) for call in mock_print.call_args_list]
            print_content = ' '.join(print_calls)
            
            # Should use IST time formatting
            assert "PM IST" in print_content or "AM IST" in print_content
            
            # Should call the formatting function
            mock_format.assert_called()

    def test_websocket_timestamp_extraction_functionality(self):
        """Test that timestamp can be extracted from WebSocket data."""
        # Arrange
        websocket_data = {
            'timestamp': 1673632656000,  # Unix timestamp in milliseconds
            'close': 117900.0,
            'volume': 35.0
        }
        
        # Act - Test timestamp extraction logic
        timestamp_ms = websocket_data.get('timestamp')
        
        # Assert
        assert timestamp_ms is not None
        assert isinstance(timestamp_ms, (int, float))
        
        # Test conversion to datetime
        if timestamp_ms:
            # Convert from milliseconds to seconds for datetime
            timestamp_seconds = timestamp_ms / 1000
            dt = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)
            assert isinstance(dt, datetime)
            assert dt.tzinfo == timezone.utc

    def test_timestamp_fallback_logic(self):
        """Test fallback logic when WebSocket data lacks timestamp."""
        # Arrange
        websocket_data_no_timestamp = {
            'close': 117900.0,
            'volume': 35.0
            # No timestamp field
        }
        
        # Act - Test fallback logic
        timestamp_ms = websocket_data_no_timestamp.get('timestamp')
        
        # Assert
        assert timestamp_ms is None
        
        # Should fall back to current time
        fallback_time = datetime.now(timezone.utc)
        assert isinstance(fallback_time, datetime)
        assert fallback_time.tzinfo == timezone.utc