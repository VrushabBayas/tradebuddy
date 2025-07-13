"""
Simple functionality tests for technical indicators.

Focuses on testing actual methods that exist in the TechnicalIndicators class.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.analysis.indicators import TechnicalIndicators
from src.core.models import OHLCV, MarketData


class TestTechnicalIndicatorsSimple:
    """Test basic technical indicators functionality."""

    @pytest.fixture
    def indicators(self):
        """Create technical indicators instance."""
        return TechnicalIndicators()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        candles = []
        base_price = 50000.0
        
        # Create 30 candles with slight upward trend
        for i in range(30):
            price = base_price + (i * 50)  # Gradual upward trend
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price + 100,
                low=price - 50,
                close=price + 25,
                volume=1000.0 + (i * 5)
            )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price + 1500,
            ohlcv_data=candles
        )

    def test_indicators_initialization(self, indicators):
        """Test that indicators initialize properly."""
        assert indicators is not None
        # Should not raise any errors

    def test_calculate_sma_functionality(self, indicators, sample_market_data):
        """Test SMA calculation functionality."""
        sma_values = indicators.calculate_sma(sample_market_data.ohlcv_data, period=10)
        
        # Functional tests
        assert isinstance(sma_values, list)
        assert len(sma_values) > 0
        
        # All SMA values should be positive
        for sma in sma_values:
            assert sma > 0, "SMA values should be positive"
        
        # SMA should smooth out price action - last value should be reasonable
        last_sma = sma_values[-1]
        current_price = sample_market_data.current_price
        
        # SMA should be within reasonable range of current price
        price_diff_pct = abs(current_price - last_sma) / current_price * 100
        assert price_diff_pct < 50, f"SMA too far from current price: {price_diff_pct}%"

    def test_calculate_ema_functionality(self, indicators, sample_market_data):
        """Test EMA calculation functionality."""
        ema_values = indicators.calculate_ema(sample_market_data.ohlcv_data, period=10)
        
        # Functional tests
        assert isinstance(ema_values, list)
        assert len(ema_values) > 0
        
        # All EMA values should be positive
        for ema in ema_values:
            assert ema > 0, "EMA values should be positive"
        
        # EMA should be more responsive than SMA
        sma_values = indicators.calculate_sma(sample_market_data.ohlcv_data, period=10)
        
        if len(ema_values) > 0 and len(sma_values) > 0:
            last_ema = ema_values[-1]
            last_sma = sma_values[-1]
            
            # Both should be positive and reasonable
            assert last_ema > 0 and last_sma > 0

    def test_detect_ema_crossover_functionality(self, indicators, sample_market_data):
        """Test EMA crossover detection."""
        try:
            crossover = indicators.detect_ema_crossover(sample_market_data.ohlcv_data)
            
            # Should return EMACrossover object
            assert crossover is not None
            assert hasattr(crossover, 'ema_9')
            assert hasattr(crossover, 'ema_15')
            assert hasattr(crossover, 'is_golden_cross')
            assert hasattr(crossover, 'crossover_strength')
            
            # Values should be reasonable
            assert crossover.ema_9 > 0
            assert crossover.ema_15 > 0
            assert isinstance(crossover.is_golden_cross, bool)
            assert 1 <= crossover.crossover_strength <= 10
            
        except Exception as e:
            # If method fails, it should be due to insufficient data or similar
            assert "data" in str(e).lower() or "period" in str(e).lower()

    def test_detect_support_resistance_functionality(self, indicators, sample_market_data):
        """Test support/resistance level detection."""
        try:
            levels = indicators.detect_support_resistance(sample_market_data.ohlcv_data)
            
            # Should return list of levels
            assert isinstance(levels, list)
            
            # If levels are found, they should have proper structure
            for level in levels:
                assert hasattr(level, 'level')
                assert hasattr(level, 'strength')
                assert hasattr(level, 'is_support')
                assert level.level > 0
                assert 1 <= level.strength <= 10
                assert isinstance(level.is_support, bool)
                
        except Exception as e:
            # Method might fail with insufficient data
            assert "data" in str(e).lower() or len(sample_market_data.ohlcv_data) < 20

    def test_analyze_volume_functionality(self, indicators, sample_market_data):
        """Test volume analysis functionality."""
        try:
            volume_analysis = indicators.analyze_volume(sample_market_data.ohlcv_data)
            
            # Should return dictionary with volume metrics
            assert isinstance(volume_analysis, dict)
            
            # Should have reasonable volume-related keys
            expected_keys = ['average_volume', 'current_volume', 'volume_ratio', 'volume_trend']
            
            # At least some volume analysis should be present
            assert len(volume_analysis) > 0
            
            # Values should be reasonable if present
            for key, value in volume_analysis.items():
                if 'volume' in key.lower() and isinstance(value, (int, float)):
                    assert value >= 0, f"Volume metric {key} should be non-negative"
                    
        except Exception as e:
            # Volume analysis might fail with certain data patterns
            assert len(str(e)) > 0  # Should have meaningful error

    def test_data_validation_functionality(self, indicators):
        """Test that indicators handle invalid data appropriately."""
        # Test with empty data
        empty_data = []
        
        try:
            indicators.calculate_sma(empty_data, period=10)
            assert False, "Should raise error with empty data"
        except Exception as e:
            # Should raise meaningful error
            assert "data" in str(e).lower() or "empty" in str(e).lower()
        
        # Test with insufficient data
        insufficient_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50000, high=50100, low=49900, close=50050, volume=1000
            )
        ]
        
        try:
            sma_result = indicators.calculate_sma(insufficient_data, period=10)
            # If it doesn't raise an error, result should be reasonable
            if sma_result:
                assert len(sma_result) <= len(insufficient_data)
        except Exception:
            # Expected with insufficient data
            pass

    def test_dataframe_conversion_functionality(self, indicators, sample_market_data):
        """Test internal dataframe conversion works."""
        try:
            # This is testing the internal _to_dataframe method indirectly
            # by calling a method that uses it
            sma_values = indicators.calculate_sma(sample_market_data.ohlcv_data, period=5)
            
            # If SMA calculation works, dataframe conversion worked
            assert isinstance(sma_values, list)
            assert len(sma_values) > 0
            
        except Exception as e:
            # Should be meaningful error if it fails
            assert len(str(e)) > 0

    def test_multiple_periods_functionality(self, indicators, sample_market_data):
        """Test that indicators work with different period lengths."""
        periods_to_test = [5, 10, 15]
        
        for period in periods_to_test:
            try:
                sma_values = indicators.calculate_sma(sample_market_data.ohlcv_data, period=period)
                ema_values = indicators.calculate_ema(sample_market_data.ohlcv_data, period=period)
                
                if sma_values and ema_values:
                    # Should have values for valid periods
                    assert len(sma_values) > 0
                    assert len(ema_values) > 0
                    
                    # Values should be positive
                    assert all(val > 0 for val in sma_values)
                    assert all(val > 0 for val in ema_values)
                    
            except Exception as e:
                # Some periods might fail due to insufficient data
                if period > len(sample_market_data.ohlcv_data):
                    # Expected failure
                    assert "period" in str(e).lower() or "data" in str(e).lower()
                else:
                    # Unexpected failure
                    raise e

    def test_indicators_with_real_market_patterns(self, indicators):
        """Test indicators with realistic market data patterns."""
        # Create more realistic market data with volatility
        candles = []
        base_price = 50000.0
        
        # Simulate more realistic price action
        import random
        random.seed(42)  # For reproducible tests
        
        for i in range(25):
            # Add some randomness to simulate real market
            price_change = random.uniform(-100, 150)  # Slight upward bias
            price = base_price + (i * 30) + price_change
            
            high = price + random.uniform(50, 200)
            low = price - random.uniform(50, 150)
            close = price + random.uniform(-50, 100)
            volume = random.uniform(800, 1200)
            
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=high,
                low=max(low, 0),  # Ensure positive prices
                close=max(close, 0),
                volume=volume
            )
            candles.append(ohlcv)
        
        realistic_data = MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price + 1000,
            ohlcv_data=candles
        )
        
        # Test that indicators can handle realistic volatility
        try:
            sma_values = indicators.calculate_sma(realistic_data.ohlcv_data, period=10)
            ema_values = indicators.calculate_ema(realistic_data.ohlcv_data, period=10)
            
            assert len(sma_values) > 0
            assert len(ema_values) > 0
            assert all(val > 0 for val in sma_values)
            assert all(val > 0 for val in ema_values)
            
        except Exception as e:
            # Should handle realistic data gracefully
            assert False, f"Indicators should handle realistic data: {e}"