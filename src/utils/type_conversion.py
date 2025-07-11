"""
Type conversion utilities for TradeBuddy.

Provides safe and consistent type conversions between different numeric types,
especially for financial calculations where precision is important.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Union, Optional, List, Dict
import structlog

logger = structlog.get_logger(__name__)

# Type aliases for better readability
NumericType = Union[int, float, Decimal, str]
FinancialValue = Union[Decimal, float]


def to_decimal(value: Any, precision: int = 8, default: Optional[Decimal] = None) -> Optional[Decimal]:
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
            return value.quantize(Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP)
        elif isinstance(value, (int, float)):
            # Convert to string first to avoid floating point precision issues
            decimal_val = Decimal(str(value))
            return decimal_val.quantize(Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP)
        elif isinstance(value, str):
            # Handle string representations
            decimal_val = Decimal(value)
            return decimal_val.quantize(Decimal(f"0.{'0' * precision}"), rounding=ROUND_HALF_UP)
        else:
            return default
    except (ValueError, TypeError, OverflowError):
        logger.warning(f"Failed to convert {value} to Decimal", value=value, type=type(value))
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
        logger.warning(f"Failed to convert {value} to float", value=value, type=type(value))
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
        logger.warning(f"Failed to convert {value} to int", value=value, type=type(value))
        return default


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


def calculate_percentage_change(old_value: NumericType, new_value: NumericType) -> Optional[Decimal]:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change as Decimal or None if invalid
    """
    old_decimal = to_decimal(old_value, default=Decimal("0"))
    new_decimal = to_decimal(new_value, default=Decimal("0"))
    
    if old_decimal is None or new_decimal is None or old_decimal == 0:
        return None
    
    change = new_decimal - old_decimal
    return (change / old_decimal) * Decimal("100")


def safe_division_with_fallback(numerator: NumericType, denominator: NumericType, fallback: NumericType = 0) -> Decimal:
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


def clamp_value(value: NumericType, min_value: NumericType, max_value: NumericType) -> Decimal:
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


def is_approximately_equal(a: NumericType, b: NumericType, tolerance: float = 1e-8) -> bool:
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