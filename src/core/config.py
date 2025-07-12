"""
Configuration management for TradeBuddy application.

Uses Pydantic Settings for environment-based configuration with validation.
"""

import os
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden using environment variables.
    """

    # Environment Configuration
    python_env: str = Field(default="development", env="PYTHON_ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")

    # API Configuration
    delta_exchange_api_url: str = Field(
        default="https://api.delta.exchange", env="DELTA_EXCHANGE_API_URL"
    )

    # Delta Exchange API Keys (Optional - only needed for trading operations)
    delta_exchange_api_key: Optional[str] = Field(
        default=None, env="DELTA_EXCHANGE_API_KEY"
    )
    delta_exchange_api_secret: Optional[str] = Field(
        default=None, env="DELTA_EXCHANGE_API_SECRET"
    )

    ollama_api_url: str = Field(default="http://localhost:11434", env="OLLAMA_API_URL")
    ollama_model: str = Field(default="qwen2.5:14b", env="OLLAMA_MODEL")

    # Database Configuration
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    database_url: str = Field(default="sqlite:///tradebuddy.db", env="DATABASE_URL")

    # Trading Configuration
    default_symbol: str = Field(default="BTCUSDT", env="DEFAULT_SYMBOL")
    default_timeframe: str = Field(default="1h", env="DEFAULT_TIMEFRAME")
    default_strategy: str = Field(default="combined", env="DEFAULT_STRATEGY")

    # Risk Management (Delta Exchange India)
    max_position_size: float = Field(default=5.0, env="MAX_POSITION_SIZE")
    default_stop_loss: float = Field(default=2.5, env="DEFAULT_STOP_LOSS")
    default_take_profit: float = Field(default=5.0, env="DEFAULT_TAKE_PROFIT")
    default_commission: float = Field(default=0.05, env="DEFAULT_COMMISSION")
    default_slippage: float = Field(default=0.05, env="DEFAULT_SLIPPAGE")
    
    # Currency Configuration
    base_currency: str = Field(default="INR", env="BASE_CURRENCY")
    usd_to_inr_rate: float = Field(default=85.0, env="USD_TO_INR_RATE")

    # API Rate Limiting
    delta_api_rate_limit: int = Field(default=10, env="DELTA_API_RATE_LIMIT")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")
    websocket_reconnect_attempts: int = Field(
        default=3, env="WEBSOCKET_RECONNECT_ATTEMPTS"
    )

    # CLI Configuration
    cli_refresh_rate: float = Field(default=1.0, env="CLI_REFRESH_RATE")
    cli_max_signals_display: int = Field(default=5, env="CLI_MAX_SIGNALS_DISPLAY")
    cli_compact_mode: bool = Field(default=False, env="CLI_COMPACT_MODE")

    # Supported symbols and timeframes
    supported_symbols: List[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
    )
    
    # Indian market symbols (INR pairs)
    indian_symbols: List[str] = Field(
        default=["BTCINR", "ETHINR", "SOLINR", "ADAINR", "DOGEINR"]
    )
    supported_timeframes: List[str] = Field(
        default=["1m", "5m", "15m", "1h", "4h", "1d"]
    )
    supported_strategies: List[str] = Field(
        default=["support_resistance", "ema_crossover", "combined"]
    )

    @field_validator("python_env")
    @classmethod
    def validate_python_env(cls, v):
        """Validate Python environment setting."""
        valid_envs = ["development", "testing", "production"]
        if v not in valid_envs:
            raise ValueError(f"python_env must be one of {valid_envs}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level setting."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @field_validator("max_position_size", "default_stop_loss", "default_take_profit", "default_commission", "default_slippage")
    @classmethod
    def validate_percentages(cls, v):
        """Validate percentage values are reasonable."""
        if v <= 0:
            raise ValueError(f"Value must be positive")
        if v > 100:
            raise ValueError(f"Value cannot exceed 100%")
        return v
    
    @field_validator("usd_to_inr_rate")
    @classmethod
    def validate_exchange_rate(cls, v):
        """Validate USD to INR exchange rate."""
        if v < 50 or v > 150:
            raise ValueError(f"Exchange rate must be between 50 and 150")
        return v

    @field_validator("cli_refresh_rate")
    @classmethod
    def validate_refresh_rate(cls, v):
        """Validate CLI refresh rate is reasonable."""
        if v < 0.1:
            raise ValueError("cli_refresh_rate cannot be less than 0.1 seconds")
        if v > 60:
            raise ValueError("cli_refresh_rate cannot exceed 60 seconds")
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.python_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.python_env == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.python_env == "testing"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "validate_assignment": True,
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global settings
    settings = Settings()
    return settings
