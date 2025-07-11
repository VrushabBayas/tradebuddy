"""
Pytest configuration and fixtures for TradeBuddy tests.

Provides common test fixtures and configuration for all test modules.
"""

import pytest
import asyncio
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from src.core.config import Settings
from src.core.models import (
    OHLCV, MarketData, TradingSignal, SessionConfig,
    Symbol, TimeFrame, StrategyType, SignalAction, SignalStrength
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
        debug=True
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
        timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )


@pytest.fixture
def sample_market_data(sample_ohlcv):
    """Sample market data for testing."""
    return MarketData(
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.ONE_HOUR,
        ohlcv=sample_ohlcv,
        timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for testing."""
    return TradingSignal(
        symbol=Symbol.BTCUSDT,
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
        timestamp=datetime(2024, 1, 1, 12, 0, 0)
    )


@pytest.fixture
def sample_session_config():
    """Sample session configuration for testing."""
    return SessionConfig(
        strategy=StrategyType.COMBINED,
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.ONE_HOUR,
        stop_loss_pct=Decimal("2.5"),
        take_profit_pct=Decimal("5.0"),
        position_size_pct=Decimal("2.0"),
        confidence_threshold=6,
        max_signals_per_session=10
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
                "volume": 1000.5
            }
        ]
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
                "volume": 1000.5 + i
            }
            for i in range(24)  # 24 hours of data
        ]
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
        "position_size_pct": 3.0
    }
    
    # Mock health check
    client.health_check.return_value = {
        "status": "healthy",
        "model": "test-model",
        "available_models": ["test-model", "qwen2.5:14b"]
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
                "timestamp": f"2024-01-01T12:0{i}:00Z"
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
            "timestamp": "2024-01-01T12:00:00Z"
        },
        "metadata": {
            "request_id": "test-123",
            "timestamp": "2024-01-01T12:00:00Z"
        }
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
                "delta_exchange": {"status": "Connected"}
            }
        }
    }
    
    return validator


@pytest.fixture
def clean_environment():
    """Clean environment variables for testing."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Clear test-related environment variables
    test_vars = [
        "PYTHON_ENV", "LOG_LEVEL", "DEBUG",
        "DELTA_EXCHANGE_API_URL", "OLLAMA_API_URL",
        "OLLAMA_MODEL", "DEFAULT_SYMBOL"
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
            "volume": 1000.0 + (day * 10)
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
                "volume": 1500.0
            },
            "expected_signal": "BUY",
            "expected_confidence": 8
        },
        {
            "name": "resistance_rejection",
            "market_data": {
                "price": 51000.0,
                "resistance_level": 51100.0,
                "volume": 1800.0
            },
            "expected_signal": "SELL",
            "expected_confidence": 7
        },
        {
            "name": "golden_cross",
            "market_data": {
                "price": 50500.0,
                "ema_9": 50450.0,
                "ema_15": 50400.0,
                "volume": 1200.0
            },
            "expected_signal": "BUY",
            "expected_confidence": 7
        }
    ]


# Parametrize fixtures
@pytest.fixture(params=[
    StrategyType.SUPPORT_RESISTANCE,
    StrategyType.EMA_CROSSOVER,
    StrategyType.COMBINED
])
def strategy_type(request):
    """Parametrized strategy type fixture."""
    return request.param


@pytest.fixture(params=[
    Symbol.BTCUSDT,
    Symbol.ETHUSDT,
    Symbol.SOLUSDT
])
def symbol(request):
    """Parametrized symbol fixture."""
    return request.param


@pytest.fixture(params=[
    TimeFrame.ONE_MINUTE,
    TimeFrame.FIVE_MINUTES,
    TimeFrame.ONE_HOUR,
    TimeFrame.ONE_DAY
])
def timeframe(request):
    """Parametrized timeframe fixture."""
    return request.param


# Helper functions
def create_mock_response(status_code: int = 200, data: Dict[str, Any] = None) -> MagicMock:
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
    assert signal.strength in [SignalStrength.WEAK, SignalStrength.MODERATE, SignalStrength.STRONG]
    assert len(signal.reasoning) > 0
    
    if signal.stop_loss:
        assert signal.stop_loss > 0
    if signal.take_profit:
        assert signal.take_profit > 0
    if signal.risk_reward_ratio:
        assert signal.risk_reward_ratio > 0