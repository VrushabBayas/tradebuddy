"""
Logging utilities for TradeBuddy.

Provides context managers and utilities for managing log levels during operations.
"""

import logging
from contextlib import contextmanager
from typing import Optional

import structlog


@contextmanager
def quiet_analysis():
    """
    Context manager to reduce log noise during analysis operations.
    
    Temporarily raises log level to WARNING to suppress verbose DEBUG/INFO logs
    while keeping ERROR logs visible.
    """
    # Get all relevant loggers
    loggers_to_quiet = [
        'src.analysis',
        'src.data', 
        'src.core',
        'src.utils'
    ]
    
    # Store original levels
    original_levels = {}
    
    try:
        # Set higher log levels
        for logger_name in loggers_to_quiet:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)
        
        # Also quiet the root logger if it's set to DEBUG
        root_logger = logging.getLogger()
        original_levels['root'] = root_logger.level
        if root_logger.level <= logging.DEBUG:
            root_logger.setLevel(logging.WARNING)
            
        yield
        
    finally:
        # Restore original levels
        for logger_name, original_level in original_levels.items():
            if logger_name == 'root':
                logging.getLogger().setLevel(original_level)
            else:
                logging.getLogger(logger_name).setLevel(original_level)


@contextmanager
def analysis_progress(console, description: str):
    """
    Context manager that combines quiet logging with progress display.
    
    Args:
        console: Rich console instance
        description: Description for the progress spinner
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    with quiet_analysis():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True  # Hide after completion
        ) as progress:
            task = progress.add_task(description, total=None)
            yield progress, task


def setup_cli_logging():
    """
    Setup optimized logging for CLI usage.
    
    Reduces noise while keeping important information visible.
    """
    # Configure specific loggers to be less verbose during CLI usage
    verbose_loggers = [
        'src.analysis.indicators',
        'src.analysis.strategies.base_strategy', 
        'src.analysis.strategies.support_resistance',
        'src.analysis.strategies.ema_crossover',
        'src.analysis.strategies.ema_crossover_v2',
        'src.analysis.strategies.combined',
        'src.data.delta_client',
        'src.analysis.ollama_client',
        'src.analysis.ai_models.ollama_wrapper',
        'src.analysis.ai_models.fingpt_client',
        'src.analysis.ai_models.model_factory',
        'src.core.config',
        'src.utils.helpers'
    ]
    
    for logger_name in verbose_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
    
    # Keep these loggers at INFO level for important updates
    important_loggers = [
        'src.cli.main',
        'src.analysis.ai_models.comparative_analyzer'
    ]
    
    for logger_name in important_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
    
    # For development, we can still see debug logs if needed by setting environment variable
    if logging.getLogger().level <= logging.DEBUG:
        # If root logger is DEBUG, allow some debug info from key components
        key_debug_loggers = [
            'src.cli.main'
        ]
        for logger_name in key_debug_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)