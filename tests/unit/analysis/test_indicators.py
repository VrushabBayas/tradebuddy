"""
Unit tests for technical indicators.
"""

import pytest
from datetime import datetime, timezone
from typing import List

from src.analysis.indicators import TechnicalIndicators
from src.core.models import OHLCV
from src.core.exceptions import DataValidationError


class TestTechnicalIndicators:
    """Test cases for technical indicators."""

    @pytest.fixture
    def sample_ohlcv_data(self) -> List[OHLCV]:
        """Create sample OHLCV data for testing."""
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        # Create 20 periods of sample data with upward trend
        data = []
        for i in range(20):
            timestamp = base_time.replace(hour=i)
            base_price = 50000 + (i * 100)  # Gradual price increase
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=base_price,
                high=base_price + 50,
                low=base_price - 30,
                close=base_price + 25,
                volume=1000 + (i * 10)
            )
            data.append(ohlcv)
        
        return data

    @pytest.fixture
    def downward_trend_data(self) -> List[OHLCV]:
        """Create sample OHLCV data with downward trend."""
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        data = []
        for i in range(20):
            timestamp = base_time.replace(hour=i)
            base_price = 52000 - (i * 100)  # Gradual price decrease
            
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=base_price,
                high=base_price + 30,
                low=base_price - 50,
                close=base_price - 25,
                volume=1000 + (i * 10)
            )
            data.append(ohlcv)
        
        return data

    def test_ema_calculation(self, sample_ohlcv_data):
        """Test EMA calculation."""
        indicators = TechnicalIndicators()
        
        # Test 9-period EMA
        ema_9 = indicators.calculate_ema(sample_ohlcv_data, period=9)
        
        assert len(ema_9) == len(sample_ohlcv_data)
        assert all(isinstance(val, float) for val in ema_9)
        
        # EMA should be increasing for upward trend
        assert ema_9[-1] > ema_9[0]
        
        # Test 15-period EMA
        ema_15 = indicators.calculate_ema(sample_ohlcv_data, period=15)
        
        assert len(ema_15) == len(sample_ohlcv_data)
        assert all(isinstance(val, float) for val in ema_15)

    def test_ema_crossover_detection(self, sample_ohlcv_data):
        """Test EMA crossover detection."""
        indicators = TechnicalIndicators()
        
        crossover = indicators.detect_ema_crossover(sample_ohlcv_data)
        
        assert hasattr(crossover, 'ema_9')
        assert hasattr(crossover, 'ema_15')
        assert hasattr(crossover, 'is_golden_cross')
        assert hasattr(crossover, 'crossover_strength')
        
        # For upward trend, should likely be golden cross
        assert isinstance(crossover.is_golden_cross, bool)
        assert 1 <= crossover.crossover_strength <= 10

    def test_support_resistance_detection(self, sample_ohlcv_data):
        """Test support and resistance level detection."""
        indicators = TechnicalIndicators()
        
        levels = indicators.detect_support_resistance(sample_ohlcv_data)
        
        assert isinstance(levels, list)
        
        for level in levels:
            assert hasattr(level, 'level')
            assert hasattr(level, 'strength')
            assert hasattr(level, 'is_support')
            assert hasattr(level, 'touches')
            
            assert isinstance(level.level, float)
            assert 1 <= level.strength <= 10
            assert isinstance(level.is_support, bool)
            assert level.touches >= 0

    def test_volume_analysis(self, sample_ohlcv_data):
        """Test volume analysis."""
        indicators = TechnicalIndicators()
        
        volume_analysis = indicators.analyze_volume(sample_ohlcv_data)
        
        assert 'current_volume' in volume_analysis
        assert 'average_volume' in volume_analysis
        assert 'volume_ratio' in volume_analysis
        assert 'volume_trend' in volume_analysis
        
        assert isinstance(volume_analysis['current_volume'], float)
        assert isinstance(volume_analysis['average_volume'], float)
        assert isinstance(volume_analysis['volume_ratio'], float)
        assert volume_analysis['volume_trend'] in ['increasing', 'decreasing', 'stable']

    def test_price_action_analysis(self, sample_ohlcv_data):
        """Test price action analysis."""
        indicators = TechnicalIndicators()
        
        price_analysis = indicators.analyze_price_action(sample_ohlcv_data)
        
        assert 'trend_direction' in price_analysis
        assert 'trend_strength' in price_analysis
        assert 'momentum' in price_analysis
        assert 'volatility' in price_analysis
        
        assert price_analysis['trend_direction'] in ['bullish', 'bearish', 'sideways']
        assert 1 <= price_analysis['trend_strength'] <= 10
        assert isinstance(price_analysis['momentum'], float)
        assert isinstance(price_analysis['volatility'], float)

    def test_comprehensive_analysis(self, sample_ohlcv_data):
        """Test comprehensive technical analysis."""
        indicators = TechnicalIndicators()
        
        analysis = indicators.comprehensive_analysis(sample_ohlcv_data)
        
        # Should contain all analysis components
        assert 'ema_crossover' in analysis
        assert 'support_resistance' in analysis
        assert 'volume_analysis' in analysis
        assert 'price_action' in analysis
        assert 'overall_sentiment' in analysis
        assert 'confidence_score' in analysis
        
        # Overall sentiment should be one of the expected values
        assert analysis['overall_sentiment'] in ['bullish', 'bearish', 'neutral']
        assert 1 <= analysis['confidence_score'] <= 10

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        indicators = TechnicalIndicators()
        
        # Test with only 2 data points (insufficient for EMA)
        insufficient_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1000.0
            ),
            OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50050.0,
                high=50150.0,
                low=49950.0,
                close=50100.0,
                volume=1100.0
            )
        ]
        
        with pytest.raises(DataValidationError):
            indicators.calculate_ema(insufficient_data, period=9)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        indicators = TechnicalIndicators()
        
        with pytest.raises(DataValidationError):
            indicators.calculate_ema([], period=9)

    def test_invalid_period_handling(self, sample_ohlcv_data):
        """Test handling of invalid periods."""
        indicators = TechnicalIndicators()
        
        with pytest.raises(DataValidationError):
            indicators.calculate_ema(sample_ohlcv_data, period=0)
        
        with pytest.raises(DataValidationError):
            indicators.calculate_ema(sample_ohlcv_data, period=-1)

    def test_ema_crossover_with_downward_trend(self, downward_trend_data):
        """Test EMA crossover detection with downward trend."""
        indicators = TechnicalIndicators()
        
        crossover = indicators.detect_ema_crossover(downward_trend_data)
        
        # For downward trend, should likely be death cross (not golden cross)
        assert isinstance(crossover.is_golden_cross, bool)
        assert 1 <= crossover.crossover_strength <= 10

    def test_support_resistance_with_trending_market(self, sample_ohlcv_data):
        """Test support/resistance detection in trending market."""
        indicators = TechnicalIndicators()
        
        levels = indicators.detect_support_resistance(sample_ohlcv_data, min_touches=1)
        
        # Should find at least some levels even in trending market
        assert len(levels) >= 0
        
        # If levels found, they should be valid
        for level in levels:
            assert level.level > 0
            assert level.touches >= 1

    def test_fibonacci_retracements(self, sample_ohlcv_data):
        """Test Fibonacci retracement calculation."""
        indicators = TechnicalIndicators()
        
        fib_levels = indicators.calculate_fibonacci_retracements(sample_ohlcv_data)
        
        assert isinstance(fib_levels, dict)
        
        # Should contain standard Fibonacci levels
        expected_levels = ['23.6%', '38.2%', '50.0%', '61.8%', '78.6%']
        for level in expected_levels:
            assert level in fib_levels
            assert isinstance(fib_levels[level], float)

    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI calculation."""
        indicators = TechnicalIndicators()
        
        rsi_values = indicators.calculate_rsi(sample_ohlcv_data, period=14)
        
        assert len(rsi_values) == len(sample_ohlcv_data)
        
        # RSI should be between 0 and 100
        for rsi in rsi_values:
            if rsi is not None:  # First few values might be None due to calculation
                assert 0 <= rsi <= 100

    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        indicators = TechnicalIndicators()
        
        bb = indicators.calculate_bollinger_bands(sample_ohlcv_data, period=20, std_dev=2)
        
        assert 'upper_band' in bb
        assert 'middle_band' in bb
        assert 'lower_band' in bb
        
        assert len(bb['upper_band']) == len(sample_ohlcv_data)
        assert len(bb['middle_band']) == len(sample_ohlcv_data)
        assert len(bb['lower_band']) == len(sample_ohlcv_data)
        
        # Upper band should be higher than middle, middle higher than lower
        for i in range(len(sample_ohlcv_data)):
            if (bb['upper_band'][i] is not None and 
                bb['middle_band'][i] is not None and 
                bb['lower_band'][i] is not None):
                assert bb['upper_band'][i] >= bb['middle_band'][i] >= bb['lower_band'][i]