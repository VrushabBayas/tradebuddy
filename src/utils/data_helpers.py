"""
Data manipulation utilities for TradeBuddy.

Provides common functions for safely accessing and manipulating data
from various sources including Pydantic models, dictionaries, and API responses.
"""

from typing import Any, Optional, Union, Dict, List
from decimal import Decimal, InvalidOperation
import structlog

logger = structlog.get_logger(__name__)


def get_value(obj: Any, key: Union[str, int], default: Any = None) -> Any:
    """
    Safely get value from either Pydantic model or dictionary.
    
    This function handles the common pattern of accessing attributes from
    objects that could be either Pydantic models (with attributes) or
    dictionaries (with keys).
    
    Args:
        obj: Object to extract value from (Pydantic model, dict, or other)
        key: Key/attribute name to access
        default: Default value to return if key is not found
        
    Returns:
        Value from the object or default if not found
        
    Examples:
        >>> from pydantic import BaseModel
        >>> class TestModel(BaseModel):
        ...     name: str = "test"
        >>> model = TestModel()
        >>> get_value(model, "name", "default")
        'test'
        >>> get_value({"name": "test"}, "name", "default")
        'test'
        >>> get_value({}, "missing", "default")
        'default'
    """
    try:
        # Try index access first (for lists/tuples with numeric key)
        if isinstance(obj, (list, tuple)) and isinstance(key, int):
            return obj[key] if 0 <= key < len(obj) else default
        # Try dictionary access (for dicts)
        elif isinstance(obj, dict):
            return obj.get(key, default)
        # Try attribute access (for Pydantic models)
        elif isinstance(key, str) and hasattr(obj, key):
            return getattr(obj, key)
        else:
            return default
    except (AttributeError, KeyError, IndexError, TypeError):
        return default


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with fallback.
    
    Args:
        value: Value to convert to float
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if value is None:
            return default
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, str):
            return float(value)
        else:
            return default
    except (ValueError, TypeError, OverflowError):
        return default


def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """
    Safely convert value to Decimal with fallback.
    
    Args:
        value: Value to convert to Decimal
        default: Default value if conversion fails
        
    Returns:
        Decimal value or default
    """
    try:
        if value is None:
            return default
        elif isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, str):
            return Decimal(value)
        else:
            return default
    except (ValueError, TypeError, OverflowError, AttributeError, InvalidOperation):
        return default


def extract_nested_value(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Extract value from nested dictionary using dot notation.
    
    Args:
        data: Dictionary to extract from
        key_path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if path not found
        
    Returns:
        Value at the path or default
        
    Example:
        >>> data = {"user": {"profile": {"name": "John"}}}
        >>> extract_nested_value(data, "user.profile.name")
        'John'
        >>> extract_nested_value(data, "user.missing.key", "default")
        'default'
    """
    try:
        keys = key_path.split(".")
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    except (AttributeError, KeyError, TypeError):
        return default


def normalize_price_data(price_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize price data from various API formats to consistent float values.
    
    Args:
        price_data: Dictionary with price information
        
    Returns:
        Normalized price data with float values
    """
    normalized = {}
    
    # Common price field mappings
    price_fields = {
        "open": ["open", "openPrice", "open_price"],
        "high": ["high", "highPrice", "high_price"],
        "low": ["low", "lowPrice", "low_price"],
        "close": ["close", "closePrice", "close_price", "price", "last"],
        "volume": ["volume", "vol", "baseVolume", "base_volume"]
    }
    
    for standard_field, possible_fields in price_fields.items():
        for field in possible_fields:
            value = get_value(price_data, field)
            if value is not None:
                normalized[standard_field] = safe_float_conversion(value)
                break
        else:
            # If no field found, set to 0.0
            normalized[standard_field] = 0.0
    
    return normalized


def filter_non_zero_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out zero, null, and empty values from dictionary.
    
    Args:
        data: Dictionary to filter
        
    Returns:
        Filtered dictionary with non-zero values
    """
    filtered = {}
    
    for key, value in data.items():
        if value is not None and value != 0 and value != "":
            if isinstance(value, (int, float)) and value != 0.0:
                filtered[key] = value
            elif isinstance(value, str) and value.strip():
                filtered[key] = value
            elif isinstance(value, (list, dict)) and len(value) > 0:
                filtered[key] = value
            elif not isinstance(value, (int, float, str, list, dict)):
                filtered[key] = value
    
    return filtered


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (positive for increase, negative for decrease)
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else 100.0
    
    return ((new_value - old_value) / old_value) * 100.0


def find_closest_value(target: float, values: List[float]) -> Optional[float]:
    """
    Find the closest value to target in a list of values.
    
    Args:
        target: Target value to find closest to
        values: List of values to search
        
    Returns:
        Closest value or None if list is empty
    """
    if not values:
        return None
    
    return min(values, key=lambda x: abs(x - target))


def group_by_key(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group list of dictionaries by a specific key.
    
    Args:
        data: List of dictionaries to group
        key: Key to group by
        
    Returns:
        Dictionary with groups
    """
    groups = {}
    
    for item in data:
        group_key = get_value(item, key, "unknown")
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(item)
    
    return groups


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate that required fields are present in data.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Returns:
        List of missing field names
    """
    missing_fields = []
    
    for field in required_fields:
        if get_value(data, field) is None:
            missing_fields.append(field)
    
    return missing_fields


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later ones taking precedence.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    merged = {}
    
    for d in dicts:
        if isinstance(d, dict):
            merged.update(d)
    
    return merged


def sanitize_symbol(symbol: str) -> str:
    """
    Sanitize trading symbol to standard format.
    
    Args:
        symbol: Trading symbol to sanitize
        
    Returns:
        Sanitized symbol in uppercase
    """
    if not isinstance(symbol, str):
        return ""
    
    # Remove any non-alphanumeric characters and convert to uppercase
    sanitized = ''.join(c for c in symbol if c.isalnum()).upper()
    
    # Ensure it ends with USDT if it doesn't already
    if not sanitized.endswith("USDT") and not sanitized.endswith("USD"):
        if len(sanitized) >= 3:
            sanitized += "USDT"
    
    return sanitized


def format_large_number(number: float, precision: int = 2) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B).
    
    Args:
        number: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if abs(number) >= 1e9:
        return f"{number / 1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number / 1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number / 1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def is_valid_price(price: Any) -> bool:
    """
    Check if a price value is valid (positive number).
    
    Args:
        price: Price value to validate
        
    Returns:
        True if valid price, False otherwise
    """
    try:
        price_float = safe_float_conversion(price)
        return price_float > 0
    except:
        return False


def calculate_distance_percentage(value1: float, value2: float) -> float:
    """
    Calculate the percentage distance between two values.
    
    Args:
        value1: First value
        value2: Second value
        
    Returns:
        Percentage distance (always positive)
    """
    if value1 == 0:
        return 0.0
    
    return abs((value2 - value1) / value1) * 100.0