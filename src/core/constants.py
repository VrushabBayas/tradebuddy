"""
Constants for TradeBuddy application.

Defines application-wide constants and enums.
"""

from enum import Enum
from typing import Dict, List

# Application Information
APP_NAME = "TradeBuddy"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "AI-powered trading signal analysis using Delta Exchange and Ollama"

# API Endpoints
class APIEndpoints:
    """Delta Exchange API endpoints."""
    
    BASE_URL = "https://api.delta.exchange"
    
    # Market Data Endpoints
    PRODUCTS = "/v2/products"
    ORDERBOOK = "/v2/l2orderbook/{symbol}"
    TRADES = "/v2/trades/{symbol}"
    CANDLES = "/v2/history/candles"
    TICKERS = "/v2/tickers"
    
    # WebSocket Endpoints
    WEBSOCKET_URL = "wss://socket.delta.exchange"


# Trading Constants
class TradingConstants:
    """Trading-related constants."""
    
    # Risk Management
    MIN_STOP_LOSS_PCT = 0.1
    MAX_STOP_LOSS_PCT = 20.0
    MIN_TAKE_PROFIT_PCT = 0.1
    MAX_TAKE_PROFIT_PCT = 50.0
    MIN_POSITION_SIZE_PCT = 0.1
    MAX_POSITION_SIZE_PCT = 10.0
    
    # Signal Confidence
    MIN_CONFIDENCE = 1
    MAX_CONFIDENCE = 10
    ACTIONABLE_CONFIDENCE_THRESHOLD = 6
    
    # Technical Analysis
    EMA_SHORT_PERIOD = 9
    EMA_LONG_PERIOD = 15
    VOLUME_LOOKBACK_PERIOD = 20
    SR_LEVEL_MIN_TOUCHES = 2
    SR_LEVEL_MAX_DISTANCE_PCT = 0.5
    
    # Risk-Reward Ratios
    MIN_RISK_REWARD_RATIO = 1.0
    RECOMMENDED_RISK_REWARD_RATIO = 2.0
    
    # Strategy Analysis Thresholds
    PRICE_TOLERANCE_PCT = 0.5  # 0.5% tolerance for price level comparisons
    SUPPORT_RESISTANCE_TOLERANCE_PCT = 0.5  # 0.5% tolerance for S/R levels
    VOLUME_CONFIRMATION_THRESHOLD = 1.2  # 20% above average volume
    STRONG_VOLUME_THRESHOLD = 1.3  # 30% above average for strong confirmation
    VERY_STRONG_VOLUME_THRESHOLD = 1.5  # 50% above average for very strong confirmation
    
    # EMA Analysis Constants
    EMA_SEPARATION_WEAK = 1.0  # 1% separation for weak signals
    EMA_SEPARATION_MODERATE = 1.5  # 1.5% separation for moderate signals  
    EMA_SEPARATION_STRONG = 2.0  # 2% separation for strong signals
    
    # Volume Analysis Periods
    VOLUME_ANALYSIS_SHORT_PERIOD = 3  # 3 periods for short-term volume analysis
    VOLUME_ANALYSIS_MEDIUM_PERIOD = 6  # 6 periods for medium-term volume analysis
    VOLUME_ANALYSIS_LONG_PERIOD = 20  # 20 periods for long-term volume analysis
    
    # Confidence Bonuses
    VOLUME_CONFIDENCE_BONUS = 2  # Bonus points for volume confirmation
    ALIGNMENT_CONFIDENCE_BONUS = 2  # Bonus points for strategy alignment
    
    # Combined Strategy Weights
    STRATEGY_ALIGNMENT_WEIGHT = 0.4  # 40% weight for strategy alignment
    SIGNAL_STRENGTH_WEIGHT = 0.3  # 30% weight for signal strength
    VOLUME_CONFIRMATION_WEIGHT = 0.2  # 20% weight for volume confirmation
    PRICE_ACTION_WEIGHT = 0.1  # 10% weight for price action consistency
    
    # Confirmation Levels
    STRONG_CONFIRMATION_LEVEL = 8  # Level 8+ for strong confirmation
    MODERATE_CONFIRMATION_LEVEL = 6  # Level 6+ for moderate confirmation
    WEAK_CONFIRMATION_LEVEL = 4  # Level 4+ for weak confirmation
    
    # Position Risk Constants
    POSITION_RISK_TOLERANCE_MULTIPLIER = 1.2  # 20% tolerance for position risk
    LEVERAGE_SAFETY_MULTIPLIER = 1.5  # 50% safety margin for leverage calculations


# Ollama Constants
class OllamaConstants:
    """Ollama-related constants."""
    
    DEFAULT_MODEL = "qwen2.5:14b"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    CONTEXT_WINDOW = 32000
    
    # Model Requirements
    MIN_RAM_GB = 8
    RECOMMENDED_RAM_GB = 16
    
    # Prompt Templates
    ANALYSIS_PROMPT_TEMPLATE = """
You are a professional cryptocurrency trading analyst using Delta Exchange market data.
Analyze the following live market data using {strategy} strategy:

DELTA EXCHANGE DATA:
Symbol: {symbol}
Current Live Price: ${current_price}
Timeframe: {timeframe}
Data Source: Delta Exchange API
Timestamp: {timestamp}

{technical_data}

Provide clear, actionable trading signals with specific entry/exit levels.
"""


# WebSocket Constants
class WebSocketConstants:
    """WebSocket-related constants."""
    
    RECONNECT_DELAY = 5  # seconds
    MAX_RECONNECT_ATTEMPTS = 10
    PING_INTERVAL = 30  # seconds
    PING_TIMEOUT = 10   # seconds
    
    # Message Types
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    
    # Channels
    ORDERBOOK_CHANNEL = "l2_orderbook"
    TRADES_CHANNEL = "all_trades"
    CANDLESTICK_CHANNEL = "candlestick_1m"


# Rate Limiting Constants
class RateLimitConstants:
    """Rate limiting constants."""
    
    # Delta Exchange Rate Limits
    DELTA_REST_REQUESTS_PER_SECOND = 10
    DELTA_WEIGHT_WINDOW_SECONDS = 300  # 5 minutes
    DELTA_MAX_WEIGHT_PER_WINDOW = 1000
    
    # Request Weights (Delta Exchange)
    WEIGHT_GET_PRODUCTS = 1
    WEIGHT_GET_ORDERBOOK = 2
    WEIGHT_GET_TRADES = 1
    WEIGHT_GET_CANDLES = 3
    WEIGHT_GET_TICKERS = 1
    
    # Ollama Rate Limits
    OLLAMA_MAX_CONCURRENT_REQUESTS = 3
    OLLAMA_REQUEST_TIMEOUT = 30


# Error Codes
class ErrorCodes:
    """Application error codes."""
    
    # Configuration Errors
    CONFIG_INVALID_ENV = "CONFIG_001"
    CONFIG_MISSING_REQUIRED = "CONFIG_002"
    CONFIG_INVALID_VALUE = "CONFIG_003"
    
    # API Errors
    API_CONNECTION_FAILED = "API_001"
    API_RATE_LIMIT_EXCEEDED = "API_002"
    API_INVALID_RESPONSE = "API_003"
    API_TIMEOUT = "API_004"
    
    # Data Errors
    DATA_VALIDATION_FAILED = "DATA_001"
    DATA_MISSING_REQUIRED = "DATA_002"
    DATA_INVALID_FORMAT = "DATA_003"
    
    # Strategy Errors
    STRATEGY_EXECUTION_FAILED = "STRATEGY_001"
    STRATEGY_INVALID_CONFIG = "STRATEGY_002"
    STRATEGY_INSUFFICIENT_DATA = "STRATEGY_003"
    
    # Signal Errors
    SIGNAL_GENERATION_FAILED = "SIGNAL_001"
    SIGNAL_INVALID_CONFIDENCE = "SIGNAL_002"
    SIGNAL_MISSING_LEVELS = "SIGNAL_003"
    
    # CLI Errors
    CLI_INVALID_INPUT = "CLI_001"
    CLI_USER_INTERRUPT = "CLI_002"
    CLI_DISPLAY_ERROR = "CLI_003"


# File Patterns
class FilePatterns:
    """File patterns and extensions."""
    
    # Log Files
    LOG_FILE_PATTERN = "tradebuddy_*.log"
    ERROR_LOG_PATTERN = "tradebuddy_errors_*.log"
    
    # Data Files
    MARKET_DATA_PATTERN = "market_data_*.json"
    SIGNAL_DATA_PATTERN = "signals_*.json"
    SESSION_DATA_PATTERN = "session_*.json"
    
    # Config Files
    CONFIG_EXTENSIONS = [".yaml", ".yml", ".json", ".toml"]


# Performance Thresholds
class PerformanceThresholds:
    """Performance benchmarks and thresholds."""
    
    # Response Times (seconds)
    MAX_API_RESPONSE_TIME = 5.0
    MAX_AI_ANALYSIS_TIME = 10.0
    MAX_SIGNAL_GENERATION_TIME = 2.0
    MAX_CLI_RESPONSE_TIME = 0.1
    
    # Success Rates (percentages)
    MIN_SIGNAL_ACCURACY = 60.0
    MIN_API_SUCCESS_RATE = 95.0
    MIN_SYSTEM_UPTIME = 99.0
    
    # Resource Usage
    MAX_MEMORY_USAGE_MB = 1000
    MAX_CPU_USAGE_PCT = 80.0
    MAX_DISK_USAGE_PCT = 85.0


# Display Constants
class DisplayConstants:
    """Display and formatting constants."""
    
    # CLI Colors
    CLI_PRIMARY_COLOR = "cyan"
    CLI_SUCCESS_COLOR = "green"
    CLI_WARNING_COLOR = "yellow"
    CLI_ERROR_COLOR = "red"
    CLI_INFO_COLOR = "blue"
    
    # Table Formatting
    MAX_TABLE_WIDTH = 120
    MAX_COLUMN_WIDTH = 50
    TABLE_PADDING = 2
    
    # Number Formatting
    PRICE_DECIMAL_PLACES = 2
    PERCENTAGE_DECIMAL_PLACES = 2
    VOLUME_DECIMAL_PLACES = 1
    
    # Date/Time Formatting
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:%S"


# Signal Display Emojis
class SignalEmojis:
    """Emojis for signal display."""
    
    # Signal Actions
    BUY = "🟢"
    SELL = "🔴"
    NEUTRAL = "⚪"
    WAIT = "🟡"
    
    # Signal Strength
    STRONG = "💪"
    MODERATE = "👍"
    WEAK = "👎"
    
    # General
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"
    ROCKET = "🚀"
    CHART = "📊"
    ROBOT = "🤖"
    BRAIN = "🧠"
    TARGET = "🎯"


# Validation Patterns
class ValidationPatterns:
    """Regular expression patterns for validation."""
    
    # Symbol Pattern (e.g., BTCUSDT)
    SYMBOL_PATTERN = r"^[A-Z]{3,10}USDT?$"
    
    # Timeframe Pattern (e.g., 1m, 5m, 1h, 1d)
    TIMEFRAME_PATTERN = r"^(\d+[mhd]|1d)$"
    
    # Price Pattern (decimal number)
    PRICE_PATTERN = r"^\d+(\.\d{1,8})?$"
    
    # Percentage Pattern (0-100 with optional decimal)
    PERCENTAGE_PATTERN = r"^(\d{1,2}(\.\d{1,2})?|100(\.0{1,2})?)$"
    
    # Confidence Pattern (1-10)
    CONFIDENCE_PATTERN = r"^([1-9]|10)$"


# Environment Specific Settings
ENVIRONMENT_CONFIGS: Dict[str, Dict[str, any]] = {
    "development": {
        "log_level": "DEBUG",
        "debug": True,
        "cli_refresh_rate": 1.0,
        "show_traceback": True,
        "enable_performance_logging": True,
    },
    "testing": {
        "log_level": "WARNING",
        "debug": False,
        "cli_refresh_rate": 0.1,
        "show_traceback": False,
        "enable_performance_logging": False,
    },
    "production": {
        "log_level": "INFO",
        "debug": False,
        "cli_refresh_rate": 2.0,
        "show_traceback": False,
        "enable_performance_logging": True,
    }
}

# Default Configuration Values
DEFAULT_CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"],
    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "strategies": ["support_resistance", "ema_crossover", "combined"],
    "risk_management": {
        "default_stop_loss": 2.5,
        "default_take_profit": 5.0,
        "default_position_size": 2.0,
        "max_position_size": 5.0,
    },
    "analysis": {
        "confidence_threshold": 6,
        "max_signals_per_session": 10,
        "signal_timeout": 300,  # 5 minutes
    }
}