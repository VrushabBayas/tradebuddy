"""
Custom exceptions for TradeBuddy application.

Provides structured error handling across the application.
"""

from typing import Any, Dict, Optional


class TradeBuddyException(Exception):
    """Base exception class for TradeBuddy application."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        """String representation of the exception."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        return base_msg


class ConfigurationError(TradeBuddyException):
    """Raised when there's a configuration error."""

    pass


class APIError(TradeBuddyException):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_data = response_data or {}


class DeltaExchangeAPIError(APIError):
    """Raised when Delta Exchange API returns an error."""

    pass


class OllamaAPIError(APIError):
    """Raised when Ollama API returns an error."""

    pass


class RateLimitExceededError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class DataValidationError(TradeBuddyException):
    """Raised when data validation fails."""

    pass


class StrategyError(TradeBuddyException):
    """Raised when there's an error in strategy execution."""

    pass


class SignalGenerationError(TradeBuddyException):
    """Raised when signal generation fails."""

    pass



class EnvironmentError(TradeBuddyException):
    """Raised when environment validation fails."""

    pass


class CLIError(TradeBuddyException):
    """Raised when CLI operations fail."""

    pass


class RiskManagementError(TradeBuddyException):
    """Raised when risk management rules are violated."""

    pass


class APIConnectionError(APIError):
    """Raised when API connection fails."""

    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class APITimeoutError(APIError):
    """Raised when API request times out."""

    pass


# Exception mappings for specific error codes
EXCEPTION_MAPPINGS = {
    "CONFIG_ERROR": ConfigurationError,
    "API_ERROR": APIError,
    "DELTA_API_ERROR": DeltaExchangeAPIError,
    "OLLAMA_API_ERROR": OllamaAPIError,
    "RATE_LIMIT_ERROR": RateLimitExceededError,
    "DATA_VALIDATION_ERROR": DataValidationError,
    "STRATEGY_ERROR": StrategyError,
    "SIGNAL_ERROR": SignalGenerationError,
    "ENVIRONMENT_ERROR": EnvironmentError,
    "CLI_ERROR": CLIError,
    "RISK_ERROR": RiskManagementError,
}


def create_exception(error_code: str, message: str, **kwargs) -> TradeBuddyException:
    """Create an exception instance based on error code."""
    exception_class = EXCEPTION_MAPPINGS.get(error_code, TradeBuddyException)
    return exception_class(message, error_code=error_code, **kwargs)
