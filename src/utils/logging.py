"""
Logging configuration for TradeBuddy application.

Provides structured logging with different levels and formatters.
"""

import logging
import logging.config
import sys
from typing import Dict, Any
from pathlib import Path

import structlog
from rich.logging import RichHandler

from src.core.config import settings


def setup_logging(log_level: str = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Log level to use (overrides settings)
    """
    if log_level is None:
        log_level = settings.log_level
    
    # Configure standard logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=False),
                "foreign_pre_chain": [
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                ],
            },
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler",
                "level": log_level,
                "formatter": "standard",
                "markup": True,
                "rich_tracebacks": True,
                "show_path": settings.is_development,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/tradebuddy.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/tradebuddy_errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "tradebuddy": {
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "src": {
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            # Third-party loggers
            "aiohttp": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "websockets": {
                "level": "WARNING", 
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (defaults to caller's module)
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get caller's module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    return structlog.get_logger(name)


def log_performance(func):
    """
    Decorator to log function performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(
                "Function executed successfully",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s",
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "Function execution failed",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s",
                error=str(e),
                error_type=type(e).__name__
            )
            
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(
                "Function executed successfully",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s",
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                "Function execution failed",
                function=func.__name__,
                execution_time=f"{execution_time:.3f}s",
                error=str(e),
                error_type=type(e).__name__
            )
            
            raise
    
    # Return appropriate wrapper based on function type
    if hasattr(func, '__call__') and hasattr(func, '__await__'):
        return async_wrapper
    else:
        return sync_wrapper


def log_api_call(endpoint: str, method: str = "GET"):
    """
    Decorator to log API calls.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            logger.info(
                "API call started",
                endpoint=endpoint,
                method=method,
                function=func.__name__
            )
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    "API call completed successfully",
                    endpoint=endpoint,
                    method=method,
                    execution_time=f"{execution_time:.3f}s",
                    response_type=type(result).__name__
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    "API call failed",
                    endpoint=endpoint,
                    method=method,
                    execution_time=f"{execution_time:.3f}s",
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                raise
        
        return wrapper
    
    return decorator


class ContextualLogger:
    """
    Logger with contextual information.
    
    Maintains context across log calls within a scope.
    """
    
    def __init__(self, logger: structlog.stdlib.BoundLogger, **context):
        self.logger = logger.bind(**context)
        self.context = context
    
    def bind(self, **new_context) -> 'ContextualLogger':
        """Add new context to the logger."""
        combined_context = {**self.context, **new_context}
        return ContextualLogger(self.logger, **combined_context)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(message, **kwargs)


def get_contextual_logger(name: str = None, **context) -> ContextualLogger:
    """
    Get a contextual logger with initial context.
    
    Args:
        name: Logger name
        **context: Initial context
        
    Returns:
        Contextual logger instance
    """
    logger = get_logger(name)
    return ContextualLogger(logger, **context)


# Initialize logging on module import
if not logging.getLogger().handlers:
    setup_logging()