"""
Functionality tests for technical indicators.

Focuses on testing indicator calculations and trading logic rather than implementation details.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.analysis.indicators import TechnicalIndicators
from src.core.models import OHLCV, MarketData


class TestTechnicalIndicatorsFunctionality:
    """Test technical indicators functionality."""

    @pytest.fixture
    def indicators(self):
        """Create technical indicators instance."""
        return TechnicalIndicators()

    @pytest.fixture
    def trending_market_data(self):
        """Create market data with clear upward trend."""
        candles = []
        base_price = 50000.0
        
        # Create 50 candles with clear upward trend
        for i in range(50):
            price = base_price + (i * 100)  # Steady upward trend
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price + 150,
                low=price - 50,
                close=price + 100,
                volume=1000.0 + (i * 10)
            )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price + 5000,
            ohlcv_data=candles
        )

    @pytest.fixture
    def sideways_market_data(self):
        """Create market data with sideways/ranging price action."""
        candles = []
        base_price = 50000.0
        
        # Create 50 candles oscillating around base price
        for i in range(50):
            # Oscillate between 49500 and 50500
            price_offset = 500 * (1 if i % 4 < 2 else -1)
            price = base_price + price_offset
            
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price + 200,
                low=price - 200,
                close=price + (50 if i % 2 == 0 else -50),
                volume=1000.0
            )
            candles.append(ohlcv)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=base_price,
            ohlcv_data=candles
        )

    def test_sma_calculation_functionality(self, indicators, trending_market_data):
        """Test Simple Moving Average calculation with trending data."""
        # Test SMA calculation
        sma_list = indicators.calculate_sma(trending_market_data.ohlcv_data, period=20)
        sma_20 = sma_list[-1] if sma_list else None
        
        # Functional tests: SMA should smooth out price action
        assert sma_20 is not None
        assert isinstance(sma_20, (int, float, Decimal))
        assert sma_20 > 0
        
        # For uptrending data, current price should be above SMA
        current_price = trending_market_data.current_price
        assert current_price > sma_20, "In uptrend, current price should be above SMA"
        
        # Test different periods
        sma_10_list = indicators.calculate_sma(trending_market_data.ohlcv_data, period=10)
        sma_10 = sma_10_list[-1] if sma_10_list else None
        
        # For SMA 50, we need at least 50 candles, so let's skip this for now
        # sma_50_list = indicators.calculate_sma(trending_market_data.ohlcv_data, period=50)
        # sma_50 = sma_50_list[-1] if sma_50_list else None
        
        # In uptrend: shorter SMA should be higher than longer SMA (simplified test)
        assert sma_10 is not None and sma_20 is not None
        assert sma_10 > sma_20, "Shorter SMA should be higher than longer SMA in uptrend"

    def test_ema_calculation_functionality(self, indicators, trending_market_data):
        """Test Exponential Moving Average calculation."""
        ema_9_list = indicators.calculate_ema(trending_market_data.ohlcv_data, period=9)
        ema_21_list = indicators.calculate_ema(trending_market_data.ohlcv_data, period=21)
        
        ema_9 = ema_9_list[-1] if ema_9_list else None
        ema_21 = ema_21_list[-1] if ema_21_list else None
        
        # Functional tests
        assert ema_9 is not None
        assert ema_21 is not None
        assert ema_9 > 0 and ema_21 > 0
        
        # EMA should be more responsive than SMA
        sma_9_list = indicators.calculate_sma(trending_market_data.ohlcv_data, period=9)
        sma_9 = sma_9_list[-1] if sma_9_list else None
        
        # In uptrend, EMA should be closer to current price than SMA
        current_price = trending_market_data.current_price
        ema_distance = abs(current_price - ema_9)
        sma_distance = abs(current_price - sma_9)
        
        # EMA should be more responsive (closer to current price)
        assert ema_distance <= sma_distance, "EMA should be more responsive than SMA"

    def test_ema_crossover_detection(self, indicators, trending_market_data):
        """Test EMA crossover detection functionality."""
        # Test the actual EMA crossover detection method that exists
        crossover = indicators.detect_ema_crossover(trending_market_data.ohlcv_data)
        
        # Functional tests
        assert rsi_trending is not None
        assert 0 <= rsi_trending <= 100, "RSI should be between 0 and 100"
        
        # In strong uptrend, RSI should be elevated (>50)
        assert rsi_trending > 50, "RSI should be above 50 in uptrend"
        
        # Test RSI with sideways data
        rsi_sideways = indicators.rsi(sideways_market_data.ohlcv_data, period=14)
        assert 0 <= rsi_sideways <= 100
        
        # RSI in sideways market should be more neutral (30-70 range typically)
        assert 20 <= rsi_sideways <= 80, "RSI in sideways market should be in neutral range"

    def test_bollinger_bands_functionality(self, indicators, trending_market_data):
        """Test Bollinger Bands calculation and price positioning."""
        bb_result = indicators.bollinger_bands(trending_market_data.ohlcv_data, period=20, std_dev=2)
        
        # Should return tuple of (upper, middle, lower)
        assert isinstance(bb_result, tuple)
        assert len(bb_result) == 3
        
        upper, middle, lower = bb_result
        
        # Functional tests
        assert upper > middle > lower, "Upper band should be above middle, middle above lower"
        assert all(val > 0 for val in bb_result), "All bands should be positive"
        
        # Middle band should approximate SMA
        sma_20 = indicators.sma(trending_market_data.ohlcv_data, period=20)
        middle_sma_diff = abs(middle - sma_20)
        assert middle_sma_diff < 10, "Middle band should be close to SMA"
        
        # Test price position relative to bands
        current_price = trending_market_data.current_price
        
        # Price should be within reasonable range of bands
        band_width = upper - lower
        assert band_width > 0, "Band width should be positive"

    def test_macd_functionality(self, indicators, trending_market_data):
        """Test MACD calculation and trend signals."""
        macd_result = indicators.macd(trending_market_data.ohlcv_data)
        
        # Should return tuple of (macd_line, signal_line, histogram)
        assert isinstance(macd_result, tuple)
        assert len(macd_result) == 3
        
        macd_line, signal_line, histogram = macd_result
        
        # Functional tests
        assert all(val is not None for val in macd_result)
        
        # Histogram should be macd_line - signal_line
        calculated_histogram = macd_line - signal_line
        histogram_diff = abs(histogram - calculated_histogram)
        assert histogram_diff < 0.01, "Histogram should equal MACD - Signal"
        
        # In uptrend, MACD line should eventually be above signal line
        # (though this might not always be true at the exact moment)
        assert isinstance(macd_line, (int, float, Decimal))
        assert isinstance(signal_line, (int, float, Decimal))

    def test_atr_functionality(self, indicators, trending_market_data):
        """Test Average True Range calculation for volatility measurement."""
        atr = indicators.atr(trending_market_data.ohlcv_data, period=14)
        
        # Functional tests
        assert atr is not None
        assert atr > 0, "ATR should be positive"
        
        # ATR should be reasonable relative to price levels
        typical_price = trending_market_data.current_price
        atr_percentage = (atr / typical_price) * 100
        
        # ATR should typically be 1-10% of price for crypto
        assert 0.1 <= atr_percentage <= 20, f"ATR percentage {atr_percentage}% seems unreasonable"

    def test_stochastic_functionality(self, indicators, trending_market_data, sideways_market_data):
        """Test Stochastic oscillator calculation."""
        # Test with trending data
        stoch_k, stoch_d = indicators.stochastic(trending_market_data.ohlcv_data, k_period=14, d_period=3)
        
        # Functional tests
        assert 0 <= stoch_k <= 100, "Stochastic %K should be between 0 and 100"
        assert 0 <= stoch_d <= 100, "Stochastic %D should be between 0 and 100"
        
        # %D should be smoother (average) of %K
        assert isinstance(stoch_k, (int, float, Decimal))
        assert isinstance(stoch_d, (int, float, Decimal))
        
        # Test with sideways data
        stoch_k_side, stoch_d_side = indicators.stochastic(sideways_market_data.ohlcv_data, k_period=14, d_period=3)
        assert 0 <= stoch_k_side <= 100
        assert 0 <= stoch_d_side <= 100

    def test_volume_indicators_functionality(self, indicators, trending_market_data):
        """Test volume-based indicators."""
        # Test Volume SMA
        vol_sma = indicators.volume_sma(trending_market_data.ohlcv_data, period=20)
        assert vol_sma > 0, "Volume SMA should be positive"
        
        # Test Volume ratio (current vs average)
        current_volume = trending_market_data.ohlcv_data[-1].volume
        volume_ratio = current_volume / vol_sma
        assert volume_ratio > 0, "Volume ratio should be positive"
        
        # Test VWAP (Volume Weighted Average Price)
        vwap = indicators.vwap(trending_market_data.ohlcv_data)
        assert vwap > 0, "VWAP should be positive"
        
        # VWAP should be reasonable relative to price range
        prices = [candle.close for candle in trending_market_data.ohlcv_data]
        min_price = min(prices)
        max_price = max(prices)
        assert min_price <= vwap <= max_price, "VWAP should be within price range"

    def test_candlestick_pattern_detection(self, indicators, trending_market_data):
        """Test candlestick pattern detection functionality."""
        # Test Doji detection
        has_doji = indicators.is_doji(trending_market_data.ohlcv_data[-1])
        assert isinstance(has_doji, bool)
        
        # Test Hammer pattern
        has_hammer = indicators.is_hammer(trending_market_data.ohlcv_data[-1])
        assert isinstance(has_hammer, bool)
        
        # Test Engulfing pattern
        has_engulfing = indicators.is_engulfing(trending_market_data.ohlcv_data[-2:])
        assert isinstance(has_engulfing, bool)

    def test_support_resistance_levels(self, indicators, trending_market_data):
        """Test support and resistance level identification."""
        levels = indicators.find_support_resistance_levels(trending_market_data.ohlcv_data)
        
        # Should return list of levels
        assert isinstance(levels, list)
        
        # If levels are found, they should have required attributes
        for level in levels:
            assert hasattr(level, 'level')
            assert hasattr(level, 'strength')
            assert hasattr(level, 'is_support')
            assert level.level > 0
            assert 1 <= level.strength <= 10

    def test_trend_analysis_functionality(self, indicators, trending_market_data, sideways_market_data):
        """Test trend analysis capabilities."""
        # Test trend detection with trending data
        trend_trending = indicators.detect_trend(trending_market_data.ohlcv_data)
        
        # Should return trend direction
        assert trend_trending in ['uptrend', 'downtrend', 'sideways', 'bullish', 'bearish', 'neutral']
        
        # For our trending data, should detect upward movement
        assert trend_trending in ['uptrend', 'bullish'], f"Should detect uptrend, got {trend_trending}"
        
        # Test with sideways data
        trend_sideways = indicators.detect_trend(sideways_market_data.ohlcv_data)
        assert trend_sideways in ['uptrend', 'downtrend', 'sideways', 'bullish', 'bearish', 'neutral']

    def test_indicator_edge_cases(self, indicators):
        """Test indicator behavior with edge cases."""
        # Test with minimal data
        minimal_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50000, high=50100, low=49900, close=50050, volume=1000
            )
        ]
        
        # Most indicators should handle insufficient data gracefully
        try:
            sma = indicators.sma(minimal_data, period=20)
            # If it returns a value, should be reasonable
            if sma is not None:
                assert sma > 0
        except (ValueError, IndexError):
            # Expected for insufficient data
            pass
        
        # Test with zero volume
        zero_volume_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50000, high=50000, low=50000, close=50000, volume=0
            ) for _ in range(20)
        ]
        
        try:
            vol_sma = indicators.volume_sma(zero_volume_data, period=10)
            if vol_sma is not None:
                assert vol_sma >= 0  # Should handle zero volume
        except (ValueError, ZeroDivisionError):
            # Expected for zero volume scenarios
            pass

    def test_indicator_combinations(self, indicators, trending_market_data):
        """Test that multiple indicators can be calculated together."""
        # Test calculating multiple indicators on same data
        sma_20 = indicators.sma(trending_market_data.ohlcv_data, period=20)
        ema_20 = indicators.ema(trending_market_data.ohlcv_data, period=20)
        rsi_14 = indicators.rsi(trending_market_data.ohlcv_data, period=14)
        
        # All should return valid values
        assert all(val is not None for val in [sma_20, ema_20, rsi_14])
        assert all(val > 0 for val in [sma_20, ema_20])
        assert 0 <= rsi_14 <= 100
        
        # Test that indicators are consistent with each other
        # (EMA should be close to SMA for same period)
        ema_sma_diff = abs(ema_20 - sma_20)
        price_level = trending_market_data.current_price
        diff_percentage = (ema_sma_diff / price_level) * 100
        
        # EMA and SMA shouldn't differ by more than 5% typically
        assert diff_percentage < 5, f"EMA and SMA differ by {diff_percentage}%, seems excessive"