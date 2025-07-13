"""
TDD Tests for Enhanced EMA Analysis Functionality.

Tests for the advanced technical analysis features that support EMA Crossover V2.
Focus on WHAT the analysis produces, not HOW it calculates.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock

from src.core.models import OHLCV, MarketData


class TestTrendStrengthCalculation:
    """Test trend strength calculation functionality."""

    @pytest.fixture
    def strong_uptrend_data(self):
        """Create data representing a strong uptrend."""
        candles = []
        base_time = datetime.now(timezone.utc)
        
        # Consistent uptrend: each candle higher than previous
        prices = [95000, 96500, 98000, 99500, 101000]
        volumes = [1200, 1400, 1800, 1600, 2000]  # Increasing volume
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 200,
                high=price + 300,
                low=price - 100,
                close=price,
                volume=volume
            )
            candles.append(candle)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=101000,
            ohlcv_data=candles
        )

    @pytest.fixture
    def weak_trend_data(self):
        """Create data representing a weak/choppy trend."""
        candles = []
        base_time = datetime.now(timezone.utc)
        
        # Choppy movement: inconsistent direction
        prices = [97000, 97200, 96800, 97100, 96900]
        volumes = [800, 900, 700, 850, 750]  # Low volume
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 50,
                high=price + 100,
                low=price - 150,
                close=price,
                volume=volume
            )
            candles.append(candle)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=96900,
            ohlcv_data=candles
        )

    def test_should_calculate_trend_strength_in_0_to_100_range(self, strong_uptrend_data):
        """Should calculate trend strength scores in 0-100 range."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Calculating trend strength for strong uptrend
        trend_strength = indicators.calculate_trend_strength(strong_uptrend_data.ohlcv_data)
        
        # Then: Should return score in valid range
        assert isinstance(trend_strength, (int, float))
        assert 0 <= trend_strength <= 100
        
        # Strong uptrend should have high trend strength
        assert trend_strength >= 60

    def test_should_distinguish_strong_vs_weak_trends(self, strong_uptrend_data, weak_trend_data):
        """Should distinguish between strong and weak trends."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Calculating trend strength for both scenarios
        strong_trend_strength = indicators.calculate_trend_strength(strong_uptrend_data.ohlcv_data)
        weak_trend_strength = indicators.calculate_trend_strength(weak_trend_data.ohlcv_data)
        
        # Then: Strong trend should have significantly higher score
        assert strong_trend_strength > weak_trend_strength
        assert strong_trend_strength >= 60  # Strong threshold
        assert weak_trend_strength <= 40   # Weak threshold

    def test_should_incorporate_volume_in_trend_strength(self, strong_uptrend_data):
        """Should incorporate volume analysis in trend strength calculation."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Comparing high volume vs low volume scenarios
        high_volume_data = strong_uptrend_data
        
        # Create low volume version
        low_volume_data = MarketData(
            symbol=strong_uptrend_data.symbol,
            timeframe=strong_uptrend_data.timeframe,
            current_price=strong_uptrend_data.current_price,
            ohlcv_data=[
                OHLCV(
                    timestamp=candle.timestamp,
                    open=candle.open, high=candle.high, low=candle.low, close=candle.close,
                    volume=candle.volume * 0.3  # Reduce volume significantly
                ) for candle in strong_uptrend_data.ohlcv_data
            ]
        )
        
        high_vol_strength = indicators.calculate_trend_strength(high_volume_data.ohlcv_data)
        low_vol_strength = indicators.calculate_trend_strength(low_volume_data.ohlcv_data)
        
        # Then: High volume should contribute to higher trend strength
        assert high_vol_strength >= low_vol_strength


class TestMarketStateDetection:
    """Test market state detection functionality."""

    def test_should_detect_trending_market_conditions(self, strong_uptrend_data):
        """Should detect when market is in trending state."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Analyzing strong trending market
        is_trending = indicators.is_trending_market(strong_uptrend_data.ohlcv_data)
        
        # Then: Should identify as trending
        assert is_trending is True

    def test_should_detect_sideways_market_conditions(self, weak_trend_data):
        """Should detect when market is in sideways/choppy state."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Analyzing sideways market
        is_trending = indicators.is_trending_market(weak_trend_data.ohlcv_data)
        
        # Then: Should identify as non-trending (sideways)
        assert is_trending is False

    def test_should_use_configurable_trend_thresholds(self, strong_uptrend_data):
        """Should use configurable thresholds for trend detection."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Using different trend strength thresholds
        is_trending_strict = indicators.is_trending_market(
            strong_uptrend_data.ohlcv_data, 
            min_trend_strength=80  # Very strict
        )
        is_trending_relaxed = indicators.is_trending_market(
            strong_uptrend_data.ohlcv_data,
            min_trend_strength=30  # More relaxed
        )
        
        # Then: Different thresholds should affect detection
        # Relaxed threshold should be more likely to detect trends
        assert is_trending_relaxed is True
        # Strict threshold may or may not detect (depends on data strength)


class TestTrendQualityAssessment:
    """Test trend quality scoring functionality."""

    def test_should_calculate_trend_quality_with_multiple_factors(self, strong_uptrend_data):
        """Should calculate trend quality scores using multiple factors."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Calculating trend quality
        quality_score = indicators.calculate_trend_quality(strong_uptrend_data.ohlcv_data)
        
        # Then: Should return comprehensive quality score
        assert isinstance(quality_score, (int, float))
        assert 0 <= quality_score <= 100
        
        # Strong uptrend should have good quality score
        assert quality_score >= 50

    def test_should_incorporate_trend_duration_in_quality(self):
        """Should incorporate trend duration in quality assessment."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create data with long trend duration
        long_trend_data = self._create_extended_trend_data(periods=20)
        short_trend_data = self._create_extended_trend_data(periods=5)
        
        long_quality = indicators.calculate_trend_quality(long_trend_data)
        short_quality = indicators.calculate_trend_quality(short_trend_data)
        
        # Longer trends should generally have higher quality scores
        assert long_quality >= short_quality

    def _create_extended_trend_data(self, periods):
        """Helper to create trend data with specified duration."""
        candles = []
        base_time = datetime.now(timezone.utc)
        base_price = 95000
        
        for i in range(periods):
            price = base_price + (i * 200)  # Steady uptrend
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 100,
                high=price + 150,
                low=price - 50,
                close=price,
                volume=1000 + (i * 50)
            )
            candles.append(candle)
        
        return candles


class TestEMAAlignmentDetection:
    """Test EMA alignment pattern detection."""

    def test_should_detect_bullish_ema_alignment(self):
        """Should detect bullish EMA alignment (9>15>50)."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create data with perfect bullish alignment
        bullish_data = self._create_aligned_ema_data(
            ema_9=99000, ema_15=98500, ema_50=98000, current_price=99200
        )
        
        # When: Detecting EMA alignment
        alignment = indicators.detect_ema_alignment(bullish_data)
        
        # Then: Should detect bullish alignment
        assert alignment == "bullish" or alignment == "strong_bullish"

    def test_should_detect_bearish_ema_alignment(self):
        """Should detect bearish EMA alignment (9<15<50)."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create data with perfect bearish alignment
        bearish_data = self._create_aligned_ema_data(
            ema_9=95000, ema_15=95500, ema_50=96000, current_price=94800
        )
        
        # When: Detecting EMA alignment
        alignment = indicators.detect_ema_alignment(bearish_data)
        
        # Then: Should detect bearish alignment
        assert alignment == "bearish" or alignment == "strong_bearish"

    def test_should_detect_mixed_ema_alignment(self):
        """Should detect mixed/unclear EMA alignment."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create data with mixed alignment
        mixed_data = self._create_aligned_ema_data(
            ema_9=97200, ema_15=97000, ema_50=97500, current_price=97100
        )
        
        # When: Detecting EMA alignment
        alignment = indicators.detect_ema_alignment(mixed_data)
        
        # Then: Should detect mixed/neutral alignment
        assert alignment in ["mixed", "neutral", "unclear"]

    def _create_aligned_ema_data(self, ema_9, ema_15, ema_50, current_price):
        """Helper to create data with specific EMA alignment."""
        # This would be populated by the indicators calculation
        return {
            "ema_9": ema_9,
            "ema_15": ema_15, 
            "ema_50": ema_50,
            "current_price": current_price
        }


class TestVolatilityContextAnalysis:
    """Test volatility context analysis functionality."""

    def test_should_calculate_volatility_percentiles(self, strong_uptrend_data):
        """Should calculate ATR-based volatility percentiles."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Calculating volatility percentile
        volatility_percentile = indicators.calculate_volatility_percentile(
            strong_uptrend_data.ohlcv_data
        )
        
        # Then: Should return percentile in 0-1 range
        assert isinstance(volatility_percentile, (int, float))
        assert 0 <= volatility_percentile <= 1

    def test_should_identify_high_volatility_periods(self):
        """Should identify high volatility periods."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create high volatility data
        high_vol_data = self._create_high_volatility_data()
        normal_vol_data = self._create_normal_volatility_data()
        
        high_vol_percentile = indicators.calculate_volatility_percentile(high_vol_data)
        normal_vol_percentile = indicators.calculate_volatility_percentile(normal_vol_data)
        
        # High volatility should have higher percentile
        assert high_vol_percentile > normal_vol_percentile
        assert high_vol_percentile >= 0.7  # High volatility threshold

    def _create_high_volatility_data(self):
        """Helper to create high volatility market data."""
        candles = []
        base_time = datetime.now(timezone.utc)
        
        # Large price swings
        prices = [95000, 98000, 92000, 99000, 91000]
        
        for i, price in enumerate(prices):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 1000,
                high=price + 2000,  # Large ranges
                low=price - 2500,
                close=price,
                volume=1500
            )
            candles.append(candle)
        
        return candles

    def _create_normal_volatility_data(self):
        """Helper to create normal volatility market data."""
        candles = []
        base_time = datetime.now(timezone.utc)
        
        # Moderate price movements
        prices = [97000, 97300, 97100, 97400, 97200]
        
        for i, price in enumerate(prices):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 100,
                high=price + 200,  # Normal ranges
                low=price - 150,
                close=price,
                volume=1200
            )
            candles.append(candle)
        
        return candles


class TestMarketStructureAnalysis:
    """Test market structure analysis functionality."""

    def test_should_detect_swing_highs_and_lows(self, strong_uptrend_data):
        """Should detect swing highs and lows in market structure."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Analyzing market structure
        structure = indicators.calculate_market_structure(strong_uptrend_data.ohlcv_data)
        
        # Then: Should identify swing points
        assert "swing_highs" in structure
        assert "swing_lows" in structure
        assert isinstance(structure["swing_highs"], list)
        assert isinstance(structure["swing_lows"], list)

    def test_should_identify_higher_highs_and_lower_lows(self):
        """Should identify higher highs and lower lows patterns."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create uptrend with higher highs
        uptrend_data = self._create_higher_highs_data()
        
        # When: Analyzing structure
        structure = indicators.calculate_market_structure(uptrend_data)
        
        # Then: Should detect higher highs pattern
        assert structure.get("higher_highs_count", 0) > 0
        assert structure.get("trend_structure") in ["bullish", "uptrend"]

    def test_should_calculate_structure_strength(self, strong_uptrend_data):
        """Should calculate market structure strength scores."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # When: Calculating structure strength
        structure = indicators.calculate_market_structure(strong_uptrend_data.ohlcv_data)
        
        # Then: Should include strength assessment
        assert "structure_strength" in structure
        assert isinstance(structure["structure_strength"], (int, float))
        assert 0 <= structure["structure_strength"] <= 100

    def _create_higher_highs_data(self):
        """Helper to create data with clear higher highs pattern."""
        candles = []
        base_time = datetime.now(timezone.utc)
        
        # Progressive higher highs: 96000, 97500, 99000
        high_points = [96000, 95500, 97500, 97000, 99000]
        
        for i, high in enumerate(high_points):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=high - 300,
                high=high,
                low=high - 800,
                close=high - 200,
                volume=1200
            )
            candles.append(candle)
        
        return candles


class TestTrendDurationTracking:
    """Test trend duration tracking functionality."""

    def test_should_track_trend_duration_in_periods(self):
        """Should track trend duration in number of periods."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create consistent trend data
        consistent_trend = self._create_consistent_trend_data(periods=8)
        
        # When: Tracking trend duration
        duration = indicators.calculate_trend_duration(consistent_trend)
        
        # Then: Should return duration count
        assert isinstance(duration, int)
        assert duration >= 5  # Should detect sustained trend

    def test_should_reset_duration_on_trend_changes(self):
        """Should reset duration count when trend changes."""
        from src.analysis.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Create trend that changes direction
        changing_trend = self._create_trend_change_data()
        
        # When: Tracking duration with trend change
        duration = indicators.calculate_trend_duration(changing_trend)
        
        # Then: Duration should reflect recent trend, not entire period
        assert duration <= 5  # Should reset on trend change

    def _create_consistent_trend_data(self, periods):
        """Helper to create consistent trend data."""
        candles = []
        base_time = datetime.now(timezone.utc)
        base_price = 95000
        
        for i in range(periods):
            price = base_price + (i * 150)  # Consistent uptrend
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 50,
                high=price + 100,
                low=price - 75,
                close=price,
                volume=1100
            )
            candles.append(candle)
        
        return candles

    def _create_trend_change_data(self):
        """Helper to create data with trend direction change."""
        candles = []
        base_time = datetime.now(timezone.utc)
        
        # Uptrend for first 4 periods, then downtrend
        prices = [95000, 95500, 96000, 96500, 96200, 95800, 95400, 95000]
        
        for i, price in enumerate(prices):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 100,
                high=price + 150,
                low=price - 200,
                close=price,
                volume=1100
            )
            candles.append(candle)
        
        return candles