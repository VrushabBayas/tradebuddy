"""
Unified helper utilities for TradeBuddy.

Combines type conversion, data manipulation, and common utility functions
for consistent and safe operations throughout the application.
"""

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union
import structlog

logger = structlog.get_logger(__name__)

# Type aliases for better readability
NumericType = Union[int, float, Decimal, str]
FinancialValue = Union[Decimal, float]


# ============================================================================
# Type Conversion Functions
# ============================================================================

def to_decimal(
    value: Any, precision: int = 8, default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Convert value to Decimal with specified precision.

    Args:
        value: Value to convert
        precision: Number of decimal places
        default: Default value if conversion fails

    Returns:
        Decimal value or default
    """
    if value is None:
        return default

    try:
        if isinstance(value, Decimal):
            return value.quantize(
                Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP
            )
        elif isinstance(value, (int, float)):
            # Convert to string first to avoid floating point precision issues
            decimal_val = Decimal(str(value))
            return decimal_val.quantize(
                Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP
            )
        elif isinstance(value, str):
            # Handle string representations
            decimal_val = Decimal(value)
            return decimal_val.quantize(
                Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP
            )
        else:
            return default
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        logger.warning(
            f"Failed to convert {value} to Decimal", value=value, type=type(value)
        )
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value to float with fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default

    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, str):
            return float(value)
        else:
            return default
    except (ValueError, TypeError, OverflowError):
        logger.warning(
            f"Failed to convert {value} to float", value=value, type=type(value)
        )
        return default


def to_int(value: Any, default: int = 0) -> int:
    """
    Convert value to int with fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Int value or default
    """
    if value is None:
        return default

    try:
        if isinstance(value, int):
            return value
        elif isinstance(value, (float, Decimal)):
            return int(value)
        elif isinstance(value, str):
            # Handle string representations
            return int(float(value))  # Convert to float first for decimal strings
        else:
            return default
    except (ValueError, TypeError, OverflowError):
        logger.warning(
            f"Failed to convert {value} to int", value=value, type=type(value)
        )
        return default


# Legacy aliases for backward compatibility
safe_float_conversion = to_float
safe_decimal_conversion = lambda value, default=Decimal("0.0"): to_decimal(value, default=default)


# ============================================================================
# Financial Calculation Functions
# ============================================================================

def financial_add(a: FinancialValue, b: FinancialValue) -> Decimal:
    """
    Safely add two financial values with proper precision.

    Args:
        a: First value
        b: Second value

    Returns:
        Sum as Decimal
    """
    decimal_a = to_decimal(a, default=Decimal("0"))
    decimal_b = to_decimal(b, default=Decimal("0"))

    if decimal_a is None or decimal_b is None:
        return Decimal("0")

    return decimal_a + decimal_b


def financial_subtract(a: FinancialValue, b: FinancialValue) -> Decimal:
    """
    Safely subtract two financial values with proper precision.

    Args:
        a: First value
        b: Second value

    Returns:
        Difference as Decimal
    """
    decimal_a = to_decimal(a, default=Decimal("0"))
    decimal_b = to_decimal(b, default=Decimal("0"))

    if decimal_a is None or decimal_b is None:
        return Decimal("0")

    return decimal_a - decimal_b


def financial_multiply(a: FinancialValue, b: FinancialValue) -> Decimal:
    """
    Safely multiply two financial values with proper precision.

    Args:
        a: First value
        b: Second value

    Returns:
        Product as Decimal
    """
    decimal_a = to_decimal(a, default=Decimal("0"))
    decimal_b = to_decimal(b, default=Decimal("0"))

    if decimal_a is None or decimal_b is None:
        return Decimal("0")

    return decimal_a * decimal_b


def financial_divide(a: FinancialValue, b: FinancialValue) -> Optional[Decimal]:
    """
    Safely divide two financial values with proper precision.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient as Decimal or None if division by zero
    """
    decimal_a = to_decimal(a, default=Decimal("0"))
    decimal_b = to_decimal(b, default=Decimal("0"))

    if decimal_a is None or decimal_b is None or decimal_b == 0:
        return None

    return decimal_a / decimal_b


def safe_division_with_fallback(
    numerator: NumericType, denominator: NumericType, fallback: NumericType = 0
) -> Decimal:
    """
    Perform division with fallback for zero denominator.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        fallback: Fallback value for division by zero

    Returns:
        Division result or fallback
    """
    result = financial_divide(numerator, denominator)
    return result if result is not None else to_decimal(fallback, default=Decimal("0"))


# ============================================================================
# Percentage and Ratio Functions
# ============================================================================

def percentage_to_decimal(percentage: NumericType) -> Decimal:
    """
    Convert percentage to decimal multiplier.

    Args:
        percentage: Percentage value (e.g., 5.0 for 5%)

    Returns:
        Decimal multiplier (e.g., 0.05 for 5%)
    """
    decimal_pct = to_decimal(percentage, default=Decimal("0"))
    if decimal_pct is None:
        return Decimal("0")

    return decimal_pct / Decimal("100")


def decimal_to_percentage(decimal_value: NumericType) -> Decimal:
    """
    Convert decimal multiplier to percentage.

    Args:
        decimal_value: Decimal multiplier (e.g., 0.05)

    Returns:
        Percentage value (e.g., 5.0 for 0.05)
    """
    decimal_val = to_decimal(decimal_value, default=Decimal("0"))
    if decimal_val is None:
        return Decimal("0")

    return decimal_val * Decimal("100")


def calculate_percentage_change(old_value: NumericType, new_value: NumericType) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change (positive for increase, negative for decrease)
    """
    old_float = to_float(old_value)
    new_float = to_float(new_value)
    
    if old_float == 0:
        return 0.0 if new_float == 0 else 100.0

    return ((new_float - old_float) / old_float) * 100.0


# ============================================================================
# Data Access and Manipulation Functions
# ============================================================================

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


def extract_nested_value(
    data: Dict[str, Any], key_path: str, default: Any = None
) -> Any:
    """
    Extract value from nested dictionary using dot notation.

    Args:
        data: Dictionary to extract from
        key_path: Dot-separated path (e.g., "user.profile.name")
        default: Default value if path not found

    Returns:
        Value at the path or default
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


# ============================================================================
# Normalization and Standardization Functions
# ============================================================================

def normalize_price(price: NumericType, precision: int = 2) -> Decimal:
    """
    Normalize price to standard precision.

    Args:
        price: Price value
        precision: Decimal precision for price

    Returns:
        Normalized price as Decimal
    """
    return to_decimal(price, precision=precision, default=Decimal("0"))


def normalize_volume(volume: NumericType, precision: int = 1) -> Decimal:
    """
    Normalize volume to standard precision.

    Args:
        volume: Volume value
        precision: Decimal precision for volume

    Returns:
        Normalized volume as Decimal
    """
    return to_decimal(volume, precision=precision, default=Decimal("0"))


def normalize_percentage(percentage: NumericType, precision: int = 2) -> Decimal:
    """
    Normalize percentage to standard precision.

    Args:
        percentage: Percentage value
        precision: Decimal precision for percentage

    Returns:
        Normalized percentage as Decimal
    """
    return to_decimal(percentage, precision=precision, default=Decimal("0"))


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
        "volume": ["volume", "vol", "baseVolume", "base_volume"],
    }

    for standard_field, possible_fields in price_fields.items():
        for field in possible_fields:
            value = get_value(price_data, field)
            if value is not None:
                normalized[standard_field] = to_float(value)
                break
        else:
            # If no field found, set to 0.0
            normalized[standard_field] = 0.0

    return normalized


# ============================================================================
# Validation and Safety Functions
# ============================================================================

def ensure_positive(value: NumericType, default: NumericType = 0) -> Decimal:
    """
    Ensure value is positive, return default if not.

    Args:
        value: Value to check
        default: Default value if not positive

    Returns:
        Positive Decimal value
    """
    decimal_val = to_decimal(value, default=Decimal("0"))
    default_decimal = to_decimal(default, default=Decimal("0"))

    if decimal_val is None or decimal_val <= 0:
        return default_decimal

    return decimal_val


def clamp_value(
    value: NumericType, min_value: NumericType, max_value: NumericType
) -> Decimal:
    """
    Clamp value between min and max bounds.

    Args:
        value: Value to clamp
        min_value: Minimum bound
        max_value: Maximum bound

    Returns:
        Clamped value as Decimal
    """
    decimal_val = to_decimal(value, default=Decimal("0"))
    decimal_min = to_decimal(min_value, default=Decimal("0"))
    decimal_max = to_decimal(max_value, default=Decimal("0"))

    if decimal_val is None:
        return decimal_min

    return max(decimal_min, min(decimal_val, decimal_max))


def is_valid_price(price: Any) -> bool:
    """
    Check if a price value is valid (positive number).

    Args:
        price: Price value to validate

    Returns:
        True if valid price, False otherwise
    """
    try:
        price_float = to_float(price)
        return price_float > 0
    except:
        return False


def is_approximately_equal(
    a: NumericType, b: NumericType, tolerance: float = 1e-8
) -> bool:
    """
    Check if two values are approximately equal within tolerance.

    Args:
        a: First value
        b: Second value
        tolerance: Tolerance for comparison

    Returns:
        True if values are approximately equal
    """
    float_a = to_float(a)
    float_b = to_float(b)

    return abs(float_a - float_b) < tolerance


def validate_required_fields(
    data: Dict[str, Any], required_fields: List[str]
) -> List[str]:
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


# ============================================================================
# Utility and Helper Functions
# ============================================================================

def convert_list_to_float(values: List[Any]) -> List[float]:
    """
    Convert list of values to list of floats.

    Args:
        values: List of values to convert

    Returns:
        List of float values
    """
    return [to_float(value) for value in values]


def convert_dict_values_to_float(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert dictionary values to floats where possible.

    Args:
        data: Dictionary with mixed value types

    Returns:
        Dictionary with float values
    """
    converted = {}

    for key, value in data.items():
        if isinstance(value, (int, float, Decimal, str)):
            converted[key] = to_float(value)
        else:
            # Keep non-numeric values as is
            converted[key] = value

    return converted


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


def group_by_key(
    data: List[Dict[str, Any]], key: str
) -> Dict[str, List[Dict[str, Any]]]:
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


# ============================================================================
# Formatting and Display Functions
# ============================================================================

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
    sanitized = "".join(c for c in symbol if c.isalnum()).upper()

    # Ensure it ends with USDT if it doesn't already
    if not sanitized.endswith("USDT") and not sanitized.endswith("USD"):
        if len(sanitized) >= 3:
            sanitized += "USDT"

    return sanitized


def round_to_significant_figures(value: NumericType, sig_figs: int) -> Decimal:
    """
    Round value to specified number of significant figures.

    Args:
        value: Value to round
        sig_figs: Number of significant figures

    Returns:
        Rounded value as Decimal
    """
    decimal_val = to_decimal(value, default=Decimal("0"))

    if decimal_val is None or decimal_val == 0:
        return Decimal("0")

    # Calculate the precision needed
    import math

    precision = sig_figs - int(math.floor(math.log10(abs(float(decimal_val))))) - 1

    return decimal_val.quantize(Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP)