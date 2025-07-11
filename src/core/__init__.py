"""
Core module for TradeBuddy application.

Contains fundamental components like configuration, models, events, and exceptions.
"""

from .config import settings
from .models import *
from .exceptions import *

__all__ = ["settings"]