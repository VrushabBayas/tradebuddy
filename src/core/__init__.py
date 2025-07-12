"""
Core module for TradeBuddy application.

Contains fundamental components like configuration, models, events, and exceptions.
"""

from .config import settings
from .exceptions import *
from .models import *

__all__ = ["settings"]
