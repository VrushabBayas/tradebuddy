"""
Pytest configuration and fixtures for TradeBuddy tests.

Provides common test fixtures and configuration for all test modules.
"""

import asyncio
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import Settings
from src.core.models import (
    OHLCV,
    MarketData,
    SessionConfig,
    SignalAction,
    SignalStrength,
    StrategyType,
    Symbol,
    TimeFrame,
    TradingSignal,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Test-specific settings configuration."""
    return Settings(
        python_env="testing",
        log_level="DEBUG",
        delta_exchange_api_url="http://localhost:8080",
        ollama_api_url="http://localhost:11434",
        ollama_model="test-model",
        default_symbol="BTCUSDT",
        default_timeframe="1h",
        default_strategy="combined",
        cli_refresh_rate=0.1,  # Fast refresh for tests
        debug=True,
    )


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data for testing."""
    return OHLCV(
        open=Decimal("50000.00"),
        high=Decimal("51000.00"),
        low=Decimal("49500.00"),
        close=Decimal("50500.00"),
        volume=Decimal("1000.50"),
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_market_data(sample_ohlcv):
    """Sample market data for testing."""
    return MarketData(
        symbol=Symbol.BTCUSD,
        timeframe=TimeFrame.ONE_HOUR,
        ohlcv=sample_ohlcv,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for testing."""
    return TradingSignal(
        symbol=Symbol.BTCUSD,
        strategy=StrategyType.COMBINED,
        action=SignalAction.BUY,
        strength=SignalStrength.STRONG,
        confidence=8,
        entry_price=Decimal("50500.00"),
        stop_loss=Decimal("49500.00"),
        take_profit=Decimal("52500.00"),
        reasoning="Strong support level with volume confirmation",
        risk_reward_ratio=Decimal("2.0"),
        position_size_pct=Decimal("3.0"),
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_session_config():
    """Sample session configuration for testing."""
    return SessionConfig(
        strategy=StrategyType.COMBINED,
        symbol=Symbol.BTCUSD,
        timeframe=TimeFrame.ONE_HOUR,
        total_capital_inr=Decimal("100000"),
        trading_capital_pct=Decimal("50.0"),
        risk_per_trade_pct=Decimal("2.0"),
        take_profit_pct=Decimal("5.0"),
        leverage=10,
        confidence_threshold=6,
        max_signals_per_session=10,
    )


@pytest.fixture
def mock_delta_client():
    """Mock Delta Exchange client for testing."""
    client = AsyncMock()

    # Mock market data response
    client.get_market_data.return_value = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "data": [
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "open": 50000.0,
                "high": 51000.0,
                "low": 49500.0,
                "close": 50500.0,
                "volume": 1000.5,
            }
        ],
    }

    # Mock historical data response
    client.get_historical_data.return_value = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "data": [
            {
                "timestamp": f"2024-01-01T{i:02d}:00:00Z",
                "open": 50000.0 + i * 10,
                "high": 51000.0 + i * 10,
                "low": 49500.0 + i * 10,
                "close": 50500.0 + i * 10,
                "volume": 1000.5 + i,
            }
            for i in range(24)  # 24 hours of data
        ],
    }

    return client


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    client = AsyncMock()

    # Mock analysis response
    client.generate_analysis.return_value = {
        "signal": "BUY",
        "confidence": 8,
        "reasoning": "Strong support level with volume confirmation",
        "entry_price": 50500.0,
        "stop_loss": 49500.0,
        "take_profit": 52500.0,
        "risk_reward_ratio": 2.0,
        "position_size_pct": 3.0,
    }

    # Mock health check
    client.health_check.return_value = {
        "status": "healthy",
        "model": "test-model",
        "available_models": ["test-model", "qwen2.5:14b"],
    }

    return client


@pytest.fixture
def mock_websocket_client():
    """Mock WebSocket client for testing."""
    client = AsyncMock()

    # Mock connection
    client.connect.return_value = True
    client.disconnect.return_value = True
    client.is_connected.return_value = True

    # Mock message stream
    async def mock_message_stream():
        for i in range(5):  # Simulate 5 messages
            yield {
                "type": "price_update",
                "symbol": "BTCUSDT",
                "price": 50500.0 + i * 10,
                "timestamp": f"2024-01-01T12:0{i}:00Z",
            }

    client.message_stream.return_value = mock_message_stream()

    return client


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    limiter = AsyncMock()

    # Mock acquire method
    limiter.acquire.return_value = True
    limiter.can_proceed.return_value = True
    limiter.get_wait_time.return_value = 0.0

    return limiter


@pytest.fixture
def sample_api_response():
    """Sample API response data."""
    return {
        "success": True,
        "data": {
            "symbol": "BTCUSDT",
            "price": 50500.0,
            "volume": 1000.5,
            "timestamp": "2024-01-01T12:00:00Z",
        },
        "metadata": {"request_id": "test-123", "timestamp": "2024-01-01T12:00:00Z"},
    }


@pytest.fixture
def mock_environment_validator():
    """Mock environment validator for testing."""
    validator = AsyncMock()

    # Mock successful validation
    validator.validate_environment.return_value = {
        "is_valid": True,
        "warnings": [],
        "details": {
            "python_version": {"current": "3.11.0", "valid": True},
            "packages": {"missing": []},
            "external_services": {
                "ollama": {"status": "Connected"},
                "delta_exchange": {"status": "Connected"},
            },
        },
    }

    return validator


@pytest.fixture
def clean_environment():
    """Clean environment variables for testing."""
    # Store original environment
    original_env = os.environ.copy()

    # Clear test-related environment variables
    test_vars = [
        "PYTHON_ENV",
        "LOG_LEVEL",
        "DEBUG",
        "DELTA_EXCHANGE_API_URL",
        "OLLAMA_API_URL",
        "OLLAMA_MODEL",
        "DEFAULT_SYMBOL",
    ]

    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    return logger


# Test data fixtures
@pytest.fixture
def historical_price_data():
    """Historical price data for backtesting."""
    return [
        {
            "timestamp": f"2024-01-{day:02d}T12:00:00Z",
            "open": 50000.0 + (day * 100),
            "high": 51000.0 + (day * 100),
            "low": 49500.0 + (day * 100),
            "close": 50500.0 + (day * 100),
            "volume": 1000.0 + (day * 10),
        }
        for day in range(1, 31)  # 30 days of data
    ]


@pytest.fixture
def strategy_test_cases():
    """Test cases for strategy testing."""
    return [
        {
            "name": "support_bounce",
            "market_data": {
                "price": 50000.0,
                "support_level": 49900.0,
                "volume": 1500.0,
            },
            "expected_signal": "BUY",
            "expected_confidence": 8,
        },
        {
            "name": "resistance_rejection",
            "market_data": {
                "price": 51000.0,
                "resistance_level": 51100.0,
                "volume": 1800.0,
            },
            "expected_signal": "SELL",
            "expected_confidence": 7,
        },
        {
            "name": "golden_cross",
            "market_data": {
                "price": 50500.0,
                "ema_9": 50450.0,
                "ema_15": 50400.0,
                "volume": 1200.0,
            },
            "expected_signal": "BUY",
            "expected_confidence": 7,
        },
    ]


# Parametrize fixtures
@pytest.fixture(
    params=[
        StrategyType.SUPPORT_RESISTANCE,
        StrategyType.EMA_CROSSOVER,
        StrategyType.COMBINED,
    ]
)
def strategy_type(request):
    """Parametrized strategy type fixture."""
    return request.param


@pytest.fixture(params=[Symbol.BTCUSD, Symbol.ETHUSD, Symbol.SOLUSDT])
def symbol(request):
    """Parametrized symbol fixture."""
    return request.param


@pytest.fixture(
    params=[
        TimeFrame.ONE_MINUTE,
        TimeFrame.FIVE_MINUTES,
        TimeFrame.ONE_HOUR,
        TimeFrame.ONE_DAY,
    ]
)
def timeframe(request):
    """Parametrized timeframe fixture."""
    return request.param


# Helper functions
def create_mock_response(
    status_code: int = 200, data: Dict[str, Any] = None
) -> MagicMock:
    """Create a mock HTTP response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.status = status_code
    mock_response.json.return_value = data or {}
    mock_response.text = str(data) if data else ""
    return mock_response


def assert_signal_valid(signal: TradingSignal):
    """Assert that a trading signal is valid."""
    assert signal.confidence >= 1 and signal.confidence <= 10
    assert signal.entry_price > 0
    assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.NEUTRAL]
    assert signal.strength in [
        SignalStrength.WEAK,
        SignalStrength.MODERATE,
        SignalStrength.STRONG,
    ]
    assert len(signal.reasoning) > 0

    if signal.stop_loss:
        assert signal.stop_loss > 0
    if signal.take_profit:
        assert signal.take_profit > 0
    if signal.risk_reward_ratio:
        assert signal.risk_reward_ratio > 0


# Candlestick Body Significance Test Fixtures

@pytest.fixture
def small_body_candles():
    """Create candles with small bodies for noise testing."""
    from datetime import datetime, timezone
    
    return [
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50050, low=49950, close=50010,  # 10 points
            volume=1000.0
        ),
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50060, low=49940, close=50025,  # 25 points
            volume=1000.0
        ),
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50080, low=49920, close=50015,  # 15 points
            volume=1000.0
        ),
    ]


@pytest.fixture 
def significant_body_candles():
    """Create candles with significant bodies for signal testing."""
    from datetime import datetime, timezone
    
    return [
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50500, low=49900, close=50450,  # 450 points (good for 1h)
            volume=1000.0
        ),
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50100, low=49520, close=49550,  # 450 points bearish
            volume=1000.0
        ),
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50350, low=49950, close=50320,  # 320 points
            volume=1000.0
        ),
    ]


@pytest.fixture
def exceptional_body_candles():
    """Create candles with exceptional bodies for confidence testing."""
    from datetime import datetime, timezone
    
    return [
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50800, low=49700, close=50750,  # 750 points (exceptional for 1h)
            volume=1500.0
        ),
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50150, low=49200, close=49300,  # 700 points bearish
            volume=1500.0
        ),
    ]


@pytest.fixture
def doji_candles():
    """Create doji candles for neutral signal testing."""
    from datetime import datetime, timezone
    
    return [
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50100, low=49900, close=50002,  # 2 points (perfect doji)
            volume=1000.0
        ),
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50150, low=49850, close=49999,  # 1 point
            volume=1000.0
        ),
        OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=50000, high=50080, low=49920, close=50004,  # 4 points
            volume=1000.0
        ),
    ]


@pytest.fixture
def timeframe_test_data():
    """Create test data for different timeframe scenarios."""
    from datetime import datetime, timezone
    
    # 100-point movement that behaves differently across timeframes
    base_candle = OHLCV(
        timestamp=datetime.now(timezone.utc),
        open=50000, high=50150, low=49950, close=50100,  # 100 points
        volume=1000.0
    )
    
    return {
        "candle": base_candle,
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "expected_1m": "exceptional",  # 100 points is large for 1m  
        "expected_1h": "insufficient", # 100 points is small for 1h
        "expected_1d": "insufficient", # 100 points is tiny for 1d
    }


def create_candle_with_body_size(body_size: float, base_price: float = 50000) -> OHLCV:
    """Helper function to create candles with specific body sizes."""
    from datetime import datetime, timezone
    
    close_price = base_price + body_size
    high_price = max(base_price, close_price) + 50
    low_price = min(base_price, close_price) - 50
    
    return OHLCV(
        timestamp=datetime.now(timezone.utc),
        open=base_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=1000.0
    )
