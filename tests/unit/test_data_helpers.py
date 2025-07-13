"""
Unit tests for data helpers utility module.

Tests the shared data manipulation utilities used across all strategy files.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.utils.helpers import (
    calculate_distance_percentage,
    calculate_percentage_change,
    extract_nested_value,
    format_large_number,
    get_value,
    is_valid_price,
    normalize_price_data,
    safe_decimal_conversion,
    safe_float_conversion,
    sanitize_symbol,
    validate_required_fields,
)


class TestGetValue:
    """Test get_value function for safe data access."""

    def test_get_value_from_pydantic_model(self):
        """Test getting value from Pydantic model."""
        # Create a mock Pydantic model
        mock_model = Mock()
        mock_model.test_attr = "test_value"

        result = get_value(mock_model, "test_attr")
        assert result == "test_value"

    def test_get_value_from_dict(self):
        """Test getting value from dictionary."""
        test_dict = {"key1": "value1", "key2": 42}

        result = get_value(test_dict, "key1")
        assert result == "value1"

        result = get_value(test_dict, "key2")
        assert result == 42

    def test_get_value_from_list_with_int_key(self):
        """Test getting value from list with integer key."""
        test_list = ["item0", "item1", "item2"]

        result = get_value(test_list, 0)
        assert result == "item0"

        result = get_value(test_list, 2)
        assert result == "item2"

    def test_get_value_from_tuple_with_int_key(self):
        """Test getting value from tuple with integer key."""
        test_tuple = ("item0", "item1", "item2")

        result = get_value(test_tuple, 1)
        assert result == "item1"

    def test_get_value_with_default(self):
        """Test getting value with default fallback."""
        test_dict = {"existing_key": "value"}

        # Non-existent key should return default
        result = get_value(test_dict, "non_existent", "default_value")
        assert result == "default_value"

        # Existing key should return actual value
        result = get_value(test_dict, "existing_key", "default_value")
        assert result == "value"

    def test_get_value_with_none_object(self):
        """Test getting value from None object."""
        result = get_value(None, "any_key", "default")
        assert result == "default"

    def test_get_value_with_invalid_index(self):
        """Test getting value with invalid list index."""
        test_list = ["item0", "item1"]

        # Out of bounds index should return default
        result = get_value(test_list, 5, "default")
        assert result == "default"

        # Negative index should return default
        result = get_value(test_list, -1, "default")
        assert result == "default"


class TestExtractNestedValue:
    """Test extract_nested_value function."""

    def test_extract_nested_value_from_dict(self):
        """Test extracting nested value from dictionary."""
        test_data = {"user": {"profile": {"name": "John"}}}

        result = extract_nested_value(test_data, "user.profile.name")
        assert result == "John"

    def test_extract_nested_value_with_missing_key(self):
        """Test extracting nested value with missing key."""
        test_data = {"user": {"profile": "value"}}

        result = extract_nested_value(test_data, "user.missing.key", "default")
        assert result == "default"

    def test_extract_nested_value_with_none_data(self):
        """Test extracting nested value from None data."""
        result = extract_nested_value(None, "any.path", "default")
        assert result == "default"

    def test_extract_nested_value_single_level(self):
        """Test extracting single level value."""
        test_data = {"key": "value"}

        result = extract_nested_value(test_data, "key")
        assert result == "value"


class TestSafeFloatConversion:
    """Test safe_float_conversion function."""

    def test_safe_float_conversion_with_valid_numbers(self):
        """Test conversion of valid numbers."""
        assert safe_float_conversion(42) == 42.0
        assert safe_float_conversion(3.14) == 3.14
        assert safe_float_conversion(Decimal("2.5")) == 2.5
        assert safe_float_conversion("1.23") == 1.23

    def test_safe_float_conversion_with_invalid_values(self):
        """Test conversion of invalid values."""
        assert safe_float_conversion("not_a_number") == 0.0
        assert safe_float_conversion(None) == 0.0
        assert safe_float_conversion([]) == 0.0
        assert safe_float_conversion({}) == 0.0

    def test_safe_float_conversion_with_custom_default(self):
        """Test conversion with custom default."""
        assert safe_float_conversion("invalid", 99.9) == 99.9
        assert safe_float_conversion(None, 10.0) == 10.0


class TestSafeDecimalConversion:
    """Test safe_decimal_conversion function."""

    def test_safe_decimal_conversion_with_valid_numbers(self):
        """Test conversion of valid numbers."""
        assert safe_decimal_conversion(42) == Decimal("42")
        assert safe_decimal_conversion(3.14) == Decimal("3.14")
        assert safe_decimal_conversion("2.5") == Decimal("2.5")

    def test_safe_decimal_conversion_with_invalid_values(self):
        """Test conversion of invalid values."""
        assert safe_decimal_conversion("not_a_number") == Decimal("0.0")
        assert safe_decimal_conversion(None) == Decimal("0.0")
        assert safe_decimal_conversion([]) == Decimal("0.0")

    def test_safe_decimal_conversion_with_custom_default(self):
        """Test conversion with custom default."""
        default = Decimal("99.9")
        assert safe_decimal_conversion("invalid", default) == default


class TestNormalizePriceData:
    """Test normalize_price_data function."""

    def test_normalize_price_data_standard_fields(self):
        """Test normalization with standard field names."""
        price_data = {
            "open": "100.5",
            "high": "105.2",
            "low": "99.8",
            "close": "103.1",
            "volume": "1500.0",
        }

        result = normalize_price_data(price_data)

        assert result["open"] == 100.5
        assert result["high"] == 105.2
        assert result["low"] == 99.8
        assert result["close"] == 103.1
        assert result["volume"] == 1500.0

    def test_normalize_price_data_alternative_fields(self):
        """Test normalization with alternative field names."""
        price_data = {
            "openPrice": "100.5",
            "highPrice": "105.2",
            "lowPrice": "99.8",
            "price": "103.1",
            "vol": "1500.0",
        }

        result = normalize_price_data(price_data)

        assert result["open"] == 100.5
        assert result["high"] == 105.2
        assert result["low"] == 99.8
        assert result["close"] == 103.1
        assert result["volume"] == 1500.0

    def test_normalize_price_data_missing_fields(self):
        """Test normalization with missing fields."""
        price_data = {
            "open": "100.5",
            "high": "105.2"
            # Missing other fields
        }

        result = normalize_price_data(price_data)

        assert result["open"] == 100.5
        assert result["high"] == 105.2
        assert result["low"] == 0.0
        assert result["close"] == 0.0
        assert result["volume"] == 0.0


class TestIsValidPrice:
    """Test is_valid_price function."""

    def test_is_valid_price_with_valid_prices(self):
        """Test validation of valid prices."""
        assert is_valid_price(100.5) is True
        assert is_valid_price("99.99") is True
        assert is_valid_price(Decimal("50.25")) is True

    def test_is_valid_price_with_invalid_prices(self):
        """Test validation of invalid prices."""
        assert is_valid_price(0) is False
        assert is_valid_price(-10.5) is False
        assert is_valid_price("not_a_number") is False
        assert is_valid_price(None) is False


class TestCalculatePercentageChange:
    """Test calculate_percentage_change function."""

    def test_calculate_percentage_change_positive(self):
        """Test percentage change calculation for positive change."""
        result = calculate_percentage_change(100.0, 110.0)
        assert result == 10.0

    def test_calculate_percentage_change_negative(self):
        """Test percentage change calculation for negative change."""
        result = calculate_percentage_change(100.0, 90.0)
        assert result == -10.0

    def test_calculate_percentage_change_zero_old_value(self):
        """Test percentage change with zero old value."""
        result = calculate_percentage_change(0.0, 50.0)
        assert result == 100.0

        result = calculate_percentage_change(0.0, 0.0)
        assert result == 0.0


class TestValidateRequiredFields:
    """Test validate_required_fields function."""

    def test_validate_required_fields_all_present(self):
        """Test validation when all fields are present."""
        data = {"name": "John", "age": 30, "email": "john@example.com"}
        required = ["name", "age", "email"]

        result = validate_required_fields(data, required)
        assert result == []

    def test_validate_required_fields_some_missing(self):
        """Test validation when some fields are missing."""
        data = {"name": "John", "age": 30}
        required = ["name", "age", "email", "phone"]

        result = validate_required_fields(data, required)
        assert "email" in result
        assert "phone" in result
        assert len(result) == 2

    def test_validate_required_fields_none_values(self):
        """Test validation with None values."""
        data = {"name": "John", "age": None, "email": ""}
        required = ["name", "age", "email"]

        result = validate_required_fields(data, required)
        assert "age" in result
        assert len(result) == 1  # email is empty string, not None


class TestSanitizeSymbol:
    """Test sanitize_symbol function."""

    def test_sanitize_symbol_standard_format(self):
        """Test sanitization of standard symbols."""
        assert sanitize_symbol("btcusdt") == "BTCUSDT"
        assert sanitize_symbol("ETHUSDT") == "ETHUSDT"
        assert sanitize_symbol("BTC-USDT") == "BTCUSDT"

    def test_sanitize_symbol_add_usdt_suffix(self):
        """Test adding USDT suffix when missing."""
        assert sanitize_symbol("BTC") == "BTCUSDT"
        assert sanitize_symbol("eth") == "ETHUSDT"

    def test_sanitize_symbol_with_special_chars(self):
        """Test sanitization with special characters."""
        assert sanitize_symbol("BTC/USDT") == "BTCUSDT"
        assert sanitize_symbol("ETH_USDT") == "ETHUSDT"

    def test_sanitize_symbol_invalid_input(self):
        """Test sanitization with invalid input."""
        assert sanitize_symbol(None) == ""
        assert sanitize_symbol(123) == ""
        assert sanitize_symbol("") == ""


class TestFormatLargeNumber:
    """Test format_large_number function."""

    def test_format_large_number_billions(self):
        """Test formatting numbers in billions."""
        assert format_large_number(1500000000) == "1.50B"
        assert format_large_number(2300000000, 1) == "2.3B"

    def test_format_large_number_millions(self):
        """Test formatting numbers in millions."""
        assert format_large_number(1500000) == "1.50M"
        assert format_large_number(2300000, 1) == "2.3M"

    def test_format_large_number_thousands(self):
        """Test formatting numbers in thousands."""
        assert format_large_number(1500) == "1.50K"
        assert format_large_number(2300, 1) == "2.3K"

    def test_format_large_number_small_numbers(self):
        """Test formatting small numbers."""
        assert format_large_number(500) == "500.00"
        assert format_large_number(42.5, 1) == "42.5"


class TestCalculateDistancePercentage:
    """Test calculate_distance_percentage function."""

    def test_calculate_distance_percentage_normal(self):
        """Test distance calculation with normal values."""
        assert calculate_distance_percentage(100.0, 110.0) == 10.0
        assert calculate_distance_percentage(100.0, 90.0) == 10.0  # Always positive

    def test_calculate_distance_percentage_zero_value(self):
        """Test distance calculation with zero value."""
        assert calculate_distance_percentage(0.0, 50.0) == 0.0
        assert calculate_distance_percentage(0.0, 0.0) == 0.0

    def test_calculate_distance_percentage_identical_values(self):
        """Test distance calculation with identical values."""
        assert calculate_distance_percentage(100.0, 100.0) == 0.0
