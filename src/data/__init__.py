"""
Data layer module for TradeBuddy.

Handles data acquisition, processing, and integration with external APIs.
"""

from .delta_client import DeltaExchangeClient

__all__ = ["DeltaExchangeClient"]