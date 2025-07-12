"""
Unit tests for type conversion utility module.

Tests the safe type conversion utilities used across all strategy files
for handling Decimal/float conversions in financial calculations.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.utils.type_conversion import (
    clamp_value,
    decimal_to_percentage,
    ensure_positive,
    financial_add,
    financial_divide,
    financial_multiply,
    financial_subtract,
    is_approximately_equal,
    normalize_price,
    normalize_volume,
    percentage_to_decimal,
    to_decimal,
    to_float,
    to_int,
)


class TestToFloat:
    """Test to_float function for safe float conversions."""

    def test_to_float_with_int(self):
        """Test converting integer to float."""
        assert to_float(42) == 42.0
        assert to_float(0) == 0.0
        assert to_float(-5) == -5.0

    def test_to_float_with_float(self):
        """Test converting float to float (passthrough)."""
        assert to_float(3.14) == 3.14
        assert to_float(0.0) == 0.0
        assert to_float(-2.5) == -2.5

    def test_to_float_with_decimal(self):
        """Test converting Decimal to float."""
        assert to_float(Decimal("42.5")) == 42.5
        assert to_float(Decimal("0")) == 0.0
        assert to_float(Decimal("-3.14")) == -3.14

    def test_to_float_with_string(self):
        """Test converting string to float."""
        assert to_float("42.5") == 42.5
        assert to_float("0") == 0.0
        assert to_float("-3.14") == -3.14

    def test_to_float_with_invalid_string(self):
        """Test converting invalid string returns default."""
        assert to_float("not_a_number") == 0.0
        assert to_float("not_a_number", 42.0) == 42.0
        assert to_float("") == 0.0
        assert to_float("", 10.0) == 10.0

    def test_to_float_with_none(self):
        """Test converting None returns default."""
        assert to_float(None) == 0.0
        assert to_float(None, 100.0) == 100.0

    def test_to_float_with_complex_types(self):
        """Test converting complex types returns default."""
        assert to_float([1, 2, 3]) == 0.0
        assert to_float({"key": "value"}) == 0.0
        assert to_float(object()) == 0.0


class TestToDecimal:
    """Test to_decimal function for safe Decimal conversions."""

    def test_to_decimal_with_int(self):
        """Test converting integer to Decimal."""
        result = to_decimal(42)
        assert result == Decimal("42.00000000")  # Default precision is 8

    def test_to_decimal_with_float(self):
        """Test converting float to Decimal."""
        result = to_decimal(3.14)
        assert result == Decimal("3.14000000")

    def test_to_decimal_with_decimal(self):
        """Test converting Decimal to Decimal with precision."""
        original = Decimal("42.123456789")
        result = to_decimal(original, precision=2)
        assert result == Decimal("42.12")

    def test_to_decimal_with_string(self):
        """Test converting string to Decimal."""
        result = to_decimal("42.5")
        assert result == Decimal("42.50000000")

    def test_to_decimal_with_invalid_string(self):
        """Test converting invalid string returns default."""
        result = to_decimal("not_a_number")
        assert result is None  # Default is None

    def test_to_decimal_with_none(self):
        """Test converting None returns default."""
        result = to_decimal(None)
        assert result is None

    def test_to_decimal_with_custom_default(self):
        """Test converting with custom default."""
        default = Decimal("99.9")
        result = to_decimal("invalid", default=default)
        assert result == default


class TestToInt:
    """Test to_int function for safe integer conversions."""

    def test_to_int_with_int(self):
        """Test converting integer to int (passthrough)."""
        assert to_int(42) == 42
        assert to_int(0) == 0
        assert to_int(-5) == -5

    def test_to_int_with_float(self):
        """Test converting float to int."""
        assert to_int(3.14) == 3
        assert to_int(3.9) == 3
        assert to_int(-2.5) == -2

    def test_to_int_with_decimal(self):
        """Test converting Decimal to int."""
        assert to_int(Decimal("42.5")) == 42
        assert to_int(Decimal("0")) == 0
        assert to_int(Decimal("-3.14")) == -3

    def test_to_int_with_string(self):
        """Test converting string to int."""
        assert to_int("42") == 42
        assert to_int("0") == 0
        assert to_int("-5") == -5
        assert to_int("3.14") == 3  # Should truncate

    def test_to_int_with_invalid_string(self):
        """Test converting invalid string returns default."""
        assert to_int("not_a_number") == 0
        assert to_int("not_a_number", 42) == 42
        assert to_int("") == 0

    def test_to_int_with_none(self):
        """Test converting None returns default."""
        assert to_int(None) == 0
        assert to_int(None, 100) == 100


class TestFinancialArithmetic:
    """Test financial arithmetic functions."""

    def test_financial_add(self):
        """Test safe financial addition."""
        result = financial_add(Decimal("10.5"), Decimal("20.3"))
        assert result == Decimal("30.8")

        result = financial_add(10.5, 20.3)
        assert result == Decimal("30.8")

    def test_financial_subtract(self):
        """Test safe financial subtraction."""
        result = financial_subtract(Decimal("20.5"), Decimal("10.3"))
        assert result == Decimal("10.2")

        result = financial_subtract(20.5, 10.3)
        assert result == Decimal("10.2")

    def test_financial_multiply(self):
        """Test safe financial multiplication."""
        result = financial_multiply(Decimal("10.5"), Decimal("2"))
        assert result == Decimal("21.0")

        result = financial_multiply(10.5, 2)
        assert result == Decimal("21.0")

    def test_financial_divide(self):
        """Test safe financial division."""
        result = financial_divide(Decimal("21.0"), Decimal("2"))
        assert result == Decimal("10.5")

        result = financial_divide(21.0, 2)
        assert result == Decimal("10.5")

    def test_financial_divide_by_zero(self):
        """Test division by zero returns None."""
        result = financial_divide(10.0, 0)
        assert result is None

        result = financial_divide(10.0, Decimal("0"))
        assert result is None


class TestPercentageConversions:
    """Test percentage conversion functions."""

    def test_percentage_to_decimal(self):
        """Test converting percentage to decimal."""
        result = percentage_to_decimal(50)
        assert result == Decimal("0.5")

        result = percentage_to_decimal(5.5)
        assert result == Decimal("0.055")

    def test_decimal_to_percentage(self):
        """Test converting decimal to percentage."""
        result = decimal_to_percentage(Decimal("0.5"))
        assert result == Decimal("50")

        result = decimal_to_percentage(0.055)
        assert result == Decimal("5.5")

    def test_percentage_conversions_roundtrip(self):
        """Test that percentage conversions are reversible."""
        original = 25.5
        decimal = percentage_to_decimal(original)
        back_to_percentage = decimal_to_percentage(decimal)
        assert back_to_percentage == Decimal("25.5")


class TestEnsurePositive:
    """Test ensure_positive function."""

    def test_ensure_positive_with_positive_values(self):
        """Test with already positive values."""
        assert ensure_positive(5) == Decimal("5")
        assert ensure_positive(3.14) == Decimal("3.14")
        assert ensure_positive(Decimal("42.5")) == Decimal("42.5")

    def test_ensure_positive_with_negative_values(self):
        """Test with negative values returns default."""
        assert ensure_positive(-5) == Decimal("0")
        assert ensure_positive(-3.14) == Decimal("0")
        assert ensure_positive(Decimal("-42.5")) == Decimal("0")

    def test_ensure_positive_with_zero(self):
        """Test with zero value."""
        assert ensure_positive(0) == Decimal("0")
        assert ensure_positive(0.0) == Decimal("0")
        assert ensure_positive(Decimal("0")) == Decimal("0")

    def test_ensure_positive_with_custom_default(self):
        """Test with custom default value."""
        assert ensure_positive(-5, 1) == Decimal("1")
        assert ensure_positive(-3.14, 1.0) == Decimal("1.0")
        assert ensure_positive(Decimal("-42.5"), Decimal("1")) == Decimal("1")


class TestClampValue:
    """Test clamp_value function."""

    def test_clamp_value_within_range(self):
        """Test with values within the specified range."""
        assert clamp_value(5, 0, 10) == Decimal("5")
        assert clamp_value(3.14, 0.0, 10.0) == Decimal("3.14")
        assert clamp_value(Decimal("7.5"), Decimal("0"), Decimal("10")) == Decimal(
            "7.5"
        )

    def test_clamp_value_below_minimum(self):
        """Test with values below minimum."""
        assert clamp_value(-5, 0, 10) == Decimal("0")
        assert clamp_value(-3.14, 0.0, 10.0) == Decimal("0.0")
        assert clamp_value(Decimal("-2.5"), Decimal("0"), Decimal("10")) == Decimal("0")

    def test_clamp_value_above_maximum(self):
        """Test with values above maximum."""
        assert clamp_value(15, 0, 10) == Decimal("10")
        assert clamp_value(13.14, 0.0, 10.0) == Decimal("10.0")
        assert clamp_value(Decimal("12.5"), Decimal("0"), Decimal("10")) == Decimal(
            "10"
        )

    def test_clamp_value_at_boundaries(self):
        """Test with values at the boundaries."""
        assert clamp_value(0, 0, 10) == Decimal("0")
        assert clamp_value(10, 0, 10) == Decimal("10")
        assert clamp_value(0.0, 0.0, 10.0) == Decimal("0.0")
        assert clamp_value(10.0, 0.0, 10.0) == Decimal("10.0")


class TestNormalizeFunctions:
    """Test normalize_price and normalize_volume functions."""

    def test_normalize_price(self):
        """Test price normalization."""
        result = normalize_price(123.456789)
        assert result == Decimal("123.46")  # Default precision is 2

        result = normalize_price(123.456789, precision=4)
        assert result == Decimal("123.4568")

    def test_normalize_volume(self):
        """Test volume normalization."""
        result = normalize_volume(1234.56789)
        assert result == Decimal("1234.6")  # Default precision is 1

        result = normalize_volume(1234.56789, precision=3)
        assert result == Decimal("1234.568")

    def test_normalize_with_invalid_input(self):
        """Test normalization with invalid input."""
        result = normalize_price("not_a_number")
        assert result == Decimal("0")

        result = normalize_volume(None)
        assert result == Decimal("0")


class TestIsApproximatelyEqual:
    """Test is_approximately_equal function."""

    def test_is_approximately_equal_true_cases(self):
        """Test cases where values should be approximately equal."""
        assert is_approximately_equal(1.0, 1.0000001) is True
        assert is_approximately_equal(Decimal("1.0"), Decimal("1.0000001")) is True
        assert is_approximately_equal(100, 100.0000001) is True

    def test_is_approximately_equal_false_cases(self):
        """Test cases where values should not be approximately equal."""
        assert is_approximately_equal(1.0, 1.01) is False
        assert is_approximately_equal(Decimal("1.0"), Decimal("1.01")) is False
        assert is_approximately_equal(100, 101) is False

    def test_is_approximately_equal_with_custom_tolerance(self):
        """Test with custom tolerance."""
        assert is_approximately_equal(1.0, 1.05, tolerance=0.1) is True
        assert is_approximately_equal(1.0, 1.15, tolerance=0.1) is False

    def test_is_approximately_equal_with_invalid_input(self):
        """Test with invalid input."""
        assert is_approximately_equal("not_a_number", 1.0) is False
        assert is_approximately_equal(1.0, None) is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_numbers(self):
        """Test with very large numbers."""
        large_num = 1e15
        result = to_decimal(large_num)
        assert result is not None
        assert float(result) == large_num

    def test_very_small_numbers(self):
        """Test with very small numbers."""
        small_num = 1e-10
        result = to_decimal(small_num)
        assert result is not None
        assert float(result) == small_num

    def test_infinity_handling(self):
        """Test handling of infinity values."""
        result = to_float(float("inf"))
        assert result == float("inf")

        result = to_decimal(float("inf"))
        assert result is None  # Should return None for invalid conversion

    def test_nan_handling(self):
        """Test handling of NaN values."""
        result = to_float(float("nan"))
        assert result != result  # NaN != NaN

        result = to_decimal(float("nan"))
        assert result is None  # Should return None for invalid conversion
