"""
Unit tests for configuration management.

Tests the Settings class and configuration validation.
"""

import pytest
import os
from decimal import Decimal
from pydantic import ValidationError

from src.core.config import Settings, get_settings, reload_settings
from src.core.exceptions import ConfigurationError


class TestSettings:
    """Test Settings class."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.python_env == "development"
        assert settings.log_level == "INFO"
        assert settings.delta_exchange_api_url == "https://api.delta.exchange"
        assert settings.ollama_api_url == "http://localhost:11434"
        assert settings.ollama_model == "qwen2.5:14b"
        assert settings.default_symbol == "BTCUSDT"
        assert settings.default_timeframe == "1h"
        assert settings.default_strategy == "combined"
    
    def test_environment_variable_override(self, clean_environment):
        """Test environment variable override."""
        # Set environment variables
        os.environ["PYTHON_ENV"] = "production"
        os.environ["LOG_LEVEL"] = "ERROR"
        os.environ["DEFAULT_SYMBOL"] = "ETHUSDT"
        os.environ["OLLAMA_MODEL"] = "custom-model"
        
        settings = Settings()
        
        assert settings.python_env == "production"
        assert settings.log_level == "ERROR"
        assert settings.default_symbol == "ETHUSDT"
        assert settings.ollama_model == "custom-model"
    
    def test_python_env_validation(self):
        """Test Python environment validation."""
        # Valid environment
        settings = Settings(python_env="development")
        assert settings.python_env == "development"
        
        settings = Settings(python_env="production")
        assert settings.python_env == "production"
        
        # Invalid environment
        with pytest.raises(ValidationError) as exc_info:
            Settings(python_env="invalid")
        
        assert "python_env must be one of" in str(exc_info.value)
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level
        
        # Case insensitive
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"
        
        # Invalid log level
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="INVALID")
        
        assert "log_level must be one of" in str(exc_info.value)
    
    def test_symbol_validation(self):
        """Test symbol validation."""
        # Valid default symbol
        settings = Settings(default_symbol="BTCUSDT")
        assert settings.default_symbol == "BTCUSDT"
        
        # Invalid default symbol
        with pytest.raises(ValidationError) as exc_info:
            Settings(default_symbol="INVALID")
        
        assert "default_symbol must be one of" in str(exc_info.value)
    
    def test_timeframe_validation(self):
        """Test timeframe validation."""
        # Valid timeframes
        for timeframe in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            settings = Settings(default_timeframe=timeframe)
            assert settings.default_timeframe == timeframe
        
        # Invalid timeframe
        with pytest.raises(ValidationError) as exc_info:
            Settings(default_timeframe="invalid")
        
        assert "default_timeframe must be one of" in str(exc_info.value)
    
    def test_strategy_validation(self):
        """Test strategy validation."""
        # Valid strategies
        for strategy in ["support_resistance", "ema_crossover", "combined"]:
            settings = Settings(default_strategy=strategy)
            assert settings.default_strategy == strategy
        
        # Invalid strategy
        with pytest.raises(ValidationError) as exc_info:
            Settings(default_strategy="invalid")
        
        assert "default_strategy must be one of" in str(exc_info.value)
    
    def test_percentage_validation(self):
        """Test percentage field validation."""
        # Valid percentages
        settings = Settings(
            max_position_size=5.0,
            default_stop_loss=2.5,
            default_take_profit=5.0
        )
        assert settings.max_position_size == 5.0
        assert settings.default_stop_loss == 2.5
        assert settings.default_take_profit == 5.0
        
        # Invalid percentages (negative)
        with pytest.raises(ValidationError):
            Settings(max_position_size=-1.0)
        
        with pytest.raises(ValidationError):
            Settings(default_stop_loss=-2.5)
        
        # Invalid percentages (too high)
        with pytest.raises(ValidationError):
            Settings(max_position_size=150.0)
    
    def test_cli_refresh_rate_validation(self):
        """Test CLI refresh rate validation."""
        # Valid refresh rates
        settings = Settings(cli_refresh_rate=1.0)
        assert settings.cli_refresh_rate == 1.0
        
        # Too low
        with pytest.raises(ValidationError) as exc_info:
            Settings(cli_refresh_rate=0.05)
        
        assert "cli_refresh_rate cannot be less than" in str(exc_info.value)
        
        # Too high
        with pytest.raises(ValidationError) as exc_info:
            Settings(cli_refresh_rate=120.0)
        
        assert "cli_refresh_rate cannot exceed" in str(exc_info.value)
    
    def test_environment_properties(self):
        """Test environment detection properties."""
        # Development environment
        settings = Settings(python_env="development")
        assert settings.is_development is True
        assert settings.is_production is False
        assert settings.is_testing is False
        
        # Production environment
        settings = Settings(python_env="production")
        assert settings.is_development is False
        assert settings.is_production is True
        assert settings.is_testing is False
        
        # Testing environment
        settings = Settings(python_env="testing")
        assert settings.is_development is False
        assert settings.is_production is False
        assert settings.is_testing is True
    
    def test_settings_immutability(self):
        """Test that settings maintain validation on assignment."""
        settings = Settings()
        
        # Valid assignment
        settings.python_env = "production"
        assert settings.python_env == "production"
        
        # Invalid assignment should raise validation error
        with pytest.raises(ValidationError):
            settings.python_env = "invalid"
    
    def test_supported_lists(self):
        """Test supported symbols, timeframes, and strategies lists."""
        settings = Settings()
        
        assert "BTCUSDT" in settings.supported_symbols
        assert "ETHUSDT" in settings.supported_symbols
        assert "SOLUSDT" in settings.supported_symbols
        
        assert "1m" in settings.supported_timeframes
        assert "1h" in settings.supported_timeframes
        assert "1d" in settings.supported_timeframes
        
        assert "support_resistance" in settings.supported_strategies
        assert "ema_crossover" in settings.supported_strategies
        assert "combined" in settings.supported_strategies


class TestSettingsGlobalFunctions:
    """Test global settings functions."""
    
    def test_get_settings(self):
        """Test get_settings function."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.python_env in ["development", "testing", "production"]
    
    def test_reload_settings(self, clean_environment):
        """Test reload_settings function."""
        # Set environment variable
        os.environ["PYTHON_ENV"] = "production"
        
        # Reload settings
        settings = reload_settings()
        
        assert settings.python_env == "production"
        assert isinstance(settings, Settings)
    
    def test_settings_singleton_behavior(self):
        """Test that settings behave like a singleton."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same instance
        assert settings1 is settings2


class TestSettingsIntegration:
    """Integration tests for settings."""
    
    def test_env_file_loading(self, tmp_path, clean_environment):
        """Test loading settings from .env file."""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            "PYTHON_ENV=production\n"
            "LOG_LEVEL=ERROR\n"
            "DEFAULT_SYMBOL=ETHUSDT\n"
            "OLLAMA_MODEL=custom-model\n"
        )
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            settings = Settings()
            
            assert settings.python_env == "production"
            assert settings.log_level == "ERROR"
            assert settings.default_symbol == "ETHUSDT"
            assert settings.ollama_model == "custom-model"
        finally:
            os.chdir(original_cwd)
    
    def test_case_insensitive_env_vars(self, clean_environment):
        """Test case insensitive environment variable handling."""
        # Set lowercase environment variables
        os.environ["python_env"] = "production"
        os.environ["log_level"] = "debug"
        
        settings = Settings()
        
        # Should work regardless of case
        assert settings.python_env == "production"
        assert settings.log_level == "DEBUG"  # Should be normalized to uppercase
    
    def test_missing_optional_settings(self, clean_environment):
        """Test behavior with missing optional settings."""
        # Only set required settings
        os.environ["DELTA_EXCHANGE_API_URL"] = "https://api.delta.exchange"
        os.environ["OLLAMA_API_URL"] = "http://localhost:11434"
        
        settings = Settings()
        
        # Should use defaults for optional settings
        assert settings.redis_url is None
        assert settings.database_url == "sqlite:///tradebuddy.db"
        assert settings.cli_refresh_rate == 1.0