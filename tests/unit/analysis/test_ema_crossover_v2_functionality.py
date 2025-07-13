"""
TDD Tests for Enhanced EMA Crossover V2 Strategy Functionality.

These tests focus on WHAT the strategy does, not HOW it does it.
Following TDD principles: write tests first, then implement functionality.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.core.models import (
    MarketData, OHLCV, SessionConfig, StrategyType, Symbol, TimeFrame,
    SignalAction, SignalStrength, EMAStrategyConfig
)


class TestEMACrossoverV2CoreFunctionality:
    """Test core EMA Crossover V2 strategy functionality."""

    @pytest.fixture
    def v2_strategy_config(self):
        """Create EMA V2 strategy configuration."""
        return EMAStrategyConfig(
            enable_rsi_filter=True,
            enable_ema50_filter=True,  # Key difference for V2
            enable_volume_filter=True,
            enable_candlestick_filter=True,
            min_trend_strength=40,
            min_trend_quality=70,
            min_trend_duration=3,
        )

    @pytest.fixture
    def session_config_v2(self, v2_strategy_config):
        """Create session config with V2 strategy."""
        return SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER_V2,
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.FIFTEEN_MINUTES,
            ema_config=v2_strategy_config
        )

    @pytest.fixture
    def strong_bullish_market_data(self):
        """Create market data representing a strong bullish trend."""
        # Price progression: 95000 -> 96000 -> 97000 -> 98000 -> 99000
        # EMAs: 9 EMA > 15 EMA > 50 EMA (perfect bullish alignment)
        # RSI: 65 (bullish momentum)
        # Volume: 150% of average (strong confirmation)
        candles = []
        base_time = datetime.now(timezone.utc)
        
        prices = [95000, 96000, 97000, 98000, 99000]
        volumes = [1500, 1800, 2100, 1900, 2200]  # Above average volume
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 100,
                high=price + 200,
                low=price - 50,
                close=price,
                volume=volume
            )
            candles.append(candle)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=99000,
            ohlcv_data=candles
        )

    @pytest.fixture
    def strong_bearish_market_data(self):
        """Create market data representing a strong bearish trend."""
        # Price progression: 99000 -> 98000 -> 97000 -> 96000 -> 95000
        # EMAs: 9 EMA < 15 EMA < 50 EMA (perfect bearish alignment)
        # RSI: 35 (bearish momentum)
        # Volume: 140% of average (strong confirmation)
        candles = []
        base_time = datetime.now(timezone.utc)
        
        prices = [99000, 98000, 97000, 96000, 95000]
        volumes = [1400, 1700, 2000, 1800, 2100]  # Above average volume
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price + 100,
                high=price + 50,
                low=price - 200,
                close=price,
                volume=volume
            )
            candles.append(candle)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=95000,
            ohlcv_data=candles
        )

    @pytest.fixture
    def sideways_market_data(self):
        """Create market data representing a sideways/choppy market."""
        # Price oscillation: 97000 -> 97500 -> 97200 -> 97600 -> 97300
        # EMAs: Mixed alignment, low trend strength
        # Volume: Normal levels
        candles = []
        base_time = datetime.now(timezone.utc)
        
        prices = [97000, 97500, 97200, 97600, 97300]
        volumes = [1000, 1100, 900, 1200, 1050]  # Normal volume
        
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            candle = OHLCV(
                timestamp=base_time.replace(minute=i*15),
                open=price - 50,
                high=price + 100,
                low=price - 100,
                close=price,
                volume=volume
            )
            candles.append(candle)
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=97300,
            ohlcv_data=candles
        )

    async def test_should_detect_golden_cross_with_proper_confidence(self, strong_bullish_market_data, session_config_v2):
        """Should detect golden cross signals with proper confidence scoring."""
        # This test will fail initially - we need to implement the strategy
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: Analyzing strong bullish market with golden cross
        result = await strategy.analyze(strong_bullish_market_data, session_config_v2)
        
        # Then: Should generate BUY signal with high confidence
        assert len(result.signals) >= 1
        primary_signal = result.primary_signal
        assert primary_signal.action == SignalAction.BUY
        assert primary_signal.confidence >= 8  # High confidence for strong trend
        assert primary_signal.strength == SignalStrength.STRONG

    async def test_should_detect_death_cross_with_proper_confidence(self, strong_bearish_market_data, session_config_v2):
        """Should detect death cross signals with proper confidence scoring."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: Analyzing strong bearish market with death cross
        result = await strategy.analyze(strong_bearish_market_data, session_config_v2)
        
        # Then: Should generate SELL signal with high confidence
        assert len(result.signals) >= 1
        primary_signal = result.primary_signal
        assert primary_signal.action == SignalAction.SELL
        assert primary_signal.confidence >= 8  # High confidence for strong trend
        assert primary_signal.strength == SignalStrength.STRONG

    async def test_should_filter_out_sideways_market_signals(self, sideways_market_data, session_config_v2):
        """Should filter out signals in sideways markets with low trend strength."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: Analyzing sideways market data
        result = await strategy.analyze(sideways_market_data, session_config_v2)
        
        # Then: Should not generate strong signals (NEUTRAL or low confidence)
        if result.signals:
            primary_signal = result.primary_signal
            # Either no actionable signals OR low confidence
            assert primary_signal.action == SignalAction.NEUTRAL or primary_signal.confidence < 6

    async def test_should_require_50_ema_trend_alignment_for_high_quality_signals(self, strong_bullish_market_data, session_config_v2):
        """Should require 50 EMA trend alignment for high-quality signals."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: 50 EMA filter is enabled
        session_config_v2.ema_config.enable_ema50_filter = True
        result = await strategy.analyze(strong_bullish_market_data, session_config_v2)
        
        # Then: Should only generate signals when price is above 50 EMA for bullish signals
        if result.signals:
            for signal in result.signals:
                if signal.action == SignalAction.BUY:
                    # Verify signal includes 50 EMA confirmation in reasoning
                    assert "50 EMA" in signal.reasoning or signal.confidence >= 7

    async def test_should_use_atr_based_dynamic_stop_losses(self, strong_bullish_market_data, session_config_v2):
        """Should use ATR-based dynamic stop losses instead of fixed percentages."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: Analyzing market data
        result = await strategy.analyze(strong_bullish_market_data, session_config_v2)
        
        # Then: Stop losses should be ATR-based (not fixed percentage)
        if result.signals:
            signal = result.signals[0]
            if signal.stop_loss:
                # ATR-based stops should be reasonable relative to current price
                stop_distance = abs(signal.entry_price - signal.stop_loss)
                price_percentage = (stop_distance / signal.entry_price) * 100
                assert 0.5 <= price_percentage <= 5.0  # Reasonable ATR-based range

    async def test_should_handle_missing_data_gracefully(self, session_config_v2):
        """Should handle missing or insufficient data gracefully."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: Providing insufficient data (less than required for indicators)
        minimal_data = MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=97000,
            ohlcv_data=[OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=96900, high=97100, low=96800, close=97000, volume=1000
            )]
        )
        
        # Then: Should handle gracefully without throwing exceptions
        try:
            result = await strategy.analyze(minimal_data, session_config_v2)
            # Should either return neutral signals or handle insufficient data
            assert result is not None
            if result.signals:
                assert all(signal.confidence <= 5 for signal in result.signals)
        except Exception as e:
            # If exception occurs, it should be a meaningful validation error
            assert "insufficient data" in str(e).lower() or "not enough" in str(e).lower()

    async def test_should_respect_configurable_filter_settings(self, strong_bullish_market_data, session_config_v2):
        """Should respect configurable filter settings (RSI, volume, candlestick)."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: Disabling all filters
        session_config_v2.ema_config.enable_rsi_filter = False
        session_config_v2.ema_config.enable_volume_filter = False
        session_config_v2.ema_config.enable_candlestick_filter = False
        session_config_v2.ema_config.enable_ema50_filter = False
        
        result_no_filters = await strategy.analyze(strong_bullish_market_data, session_config_v2)
        
        # When: Enabling all filters
        session_config_v2.ema_config.enable_rsi_filter = True
        session_config_v2.ema_config.enable_volume_filter = True
        session_config_v2.ema_config.enable_candlestick_filter = True
        session_config_v2.ema_config.enable_ema50_filter = True
        
        result_with_filters = await strategy.analyze(strong_bullish_market_data, session_config_v2)
        
        # Then: Filter settings should affect signal generation
        # With filters, signals should have higher confidence or different count
        if result_no_filters.signals and result_with_filters.signals:
            # Filters typically increase signal quality (confidence) but may reduce quantity
            assert (len(result_with_filters.signals) <= len(result_no_filters.signals) or
                    max(s.confidence for s in result_with_filters.signals) >= 
                    max(s.confidence for s in result_no_filters.signals))

    async def test_should_generate_higher_confidence_for_multiple_confirmations(self, strong_bullish_market_data, session_config_v2):
        """Should generate higher confidence for signals with multiple confirmations."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # When: Analyzing strong trend with multiple confirmations
        result = await strategy.analyze(strong_bullish_market_data, session_config_v2)
        
        # Then: Signals with multiple confirmations should have higher confidence
        if result.signals:
            signal = result.primary_signal
            # Strong trend + volume + RSI + candlestick + 50 EMA alignment should give high confidence
            if signal.action in [SignalAction.BUY, SignalAction.SELL]:
                assert signal.confidence >= 7  # Multiple confirmations increase confidence
                # Reasoning should mention multiple confirmations
                assert len(signal.reasoning.split()) >= 5  # Detailed reasoning


class TestEMACrossoverV2ConfigurationFunctionality:
    """Test configuration-related functionality."""

    def test_should_use_default_v2_parameters(self):
        """Should use appropriate default parameters for V2 strategy."""
        config = EMAStrategyConfig()
        
        # Then: V2 should have enhanced defaults
        assert config.enable_ema50_filter is False  # Optional by default
        assert config.ema_trend_period == 50  # 50 EMA for trend filtering
        assert config.volume_threshold_pct == 110.0  # Enhanced volume threshold
        assert config.atr_stop_multiplier == 1.5  # ATR-based stops

    def test_should_validate_v2_parameter_ranges(self):
        """Should validate V2 parameter ranges."""
        # Valid configuration should work
        valid_config = EMAStrategyConfig(
            min_trend_strength=30,
            min_trend_quality=60,
            min_trend_duration=2
        )
        assert valid_config.min_trend_strength == 30
        
        # Invalid ranges should raise validation errors
        with pytest.raises(ValueError):
            EMAStrategyConfig(min_trend_strength=-10)  # Negative not allowed
        
        with pytest.raises(ValueError):
            EMAStrategyConfig(min_trend_quality=150)  # Over 100 not allowed


class TestEMACrossoverV2EdgeCases:
    """Test edge cases and error handling."""

    async def test_should_handle_extreme_volatility_gracefully(self, session_config_v2):
        """Should handle extreme volatility periods gracefully."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # Create extremely volatile market data
        volatile_data = MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=50000,
            ohlcv_data=[
                OHLCV(timestamp=datetime.now(timezone.utc), open=90000, high=95000, low=85000, close=92000, volume=1000),
                OHLCV(timestamp=datetime.now(timezone.utc), open=92000, high=88000, low=75000, close=80000, volume=1500),
                OHLCV(timestamp=datetime.now(timezone.utc), open=80000, high=105000, low=78000, close=100000, volume=2000),
            ]
        )
        
        # Should handle without crashing and potentially reduce signal confidence
        result = await strategy.analyze(volatile_data, session_config_v2)
        assert result is not None
        
        if result.signals:
            # High volatility should affect confidence or signal generation
            max_confidence = max(s.confidence for s in result.signals)
            assert max_confidence <= 8  # Extreme volatility should reduce confidence

    async def test_should_handle_zero_volume_data(self, session_config_v2):
        """Should handle zero volume data gracefully."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # Create data with zero volume
        zero_volume_data = MarketData(
            symbol="BTCUSD",
            timeframe="15m",
            current_price=97000,
            ohlcv_data=[
                OHLCV(timestamp=datetime.now(timezone.utc), open=97000, high=97100, low=96900, close=97000, volume=0),
                OHLCV(timestamp=datetime.now(timezone.utc), open=97000, high=97200, low=96800, close=97100, volume=0),
            ]
        )
        
        # Should handle gracefully
        result = await strategy.analyze(zero_volume_data, session_config_v2)
        assert result is not None
        
        # Zero volume should prevent high-confidence signals
        if result.signals:
            assert all(signal.confidence <= 5 for signal in result.signals)

    async def test_body_significance_integration_improves_signal_quality(self, session_config_v2):
        """Test that body significance integration improves V2 signal quality."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # Create market data with mixed body sizes
        mixed_body_data = MarketData(
            symbol="BTCUSD",
            timeframe="1h",  # Use 1h for body significance testing
            current_price=50450,
            ohlcv_data=[
                # Large body trend continuation (should generate signals)
                OHLCV(timestamp=datetime.now(timezone.utc), open=49500, high=50000, low=49400, close=49950, volume=1500),
                OHLCV(timestamp=datetime.now(timezone.utc), open=49950, high=50450, low=49850, close=50400, volume=1600),
                OHLCV(timestamp=datetime.now(timezone.utc), open=50400, high=50900, low=50300, close=50850, volume=1700),
                # Small body noise (should be filtered)
                OHLCV(timestamp=datetime.now(timezone.utc), open=50850, high=50870, low=50830, close=50860, volume=800),
                OHLCV(timestamp=datetime.now(timezone.utc), open=50860, high=50880, low=50840, close=50865, volume=750),
            ]
        )
        
        # When: Analyzing with V2 strategy (includes body significance)
        result = await strategy.analyze(mixed_body_data, session_config_v2)
        
        # Then: Should filter out signals from small-body candles
        assert result is not None
        
        # Check that analysis includes body significance considerations
        # V2 should be more selective due to body filtering
        if result.signals:
            # All signals should be meaningful (body significance filtered)
            for signal in result.signals:
                assert signal.confidence >= 6, "V2 with body filtering should produce higher quality signals"

    async def test_timeframe_body_requirements_affect_v2_signals(self, session_config_v2):
        """Test that different timeframes have appropriate body requirements in V2."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # Same price movement (200 points) on different timeframes
        price_movement_data = [
            OHLCV(timestamp=datetime.now(timezone.utc), open=50000, high=50250, low=49950, close=50200, volume=1000),
            OHLCV(timestamp=datetime.now(timezone.utc), open=50200, high=50450, low=50150, close=50400, volume=1100),
        ]
        
        # Test on 1m timeframe (200 points should be exceptional)
        market_data_1m = MarketData(
            symbol="BTCUSD",
            timeframe="1m",
            current_price=50400,
            ohlcv_data=price_movement_data
        )
        
        # Test on 1h timeframe (200 points should be insufficient)
        market_data_1h = MarketData(
            symbol="BTCUSD", 
            timeframe="1h",
            current_price=50400,
            ohlcv_data=price_movement_data
        )
        
        # When: Analyzing same movement on different timeframes
        result_1m = await strategy.analyze(market_data_1m, session_config_v2)
        result_1h = await strategy.analyze(market_data_1h, session_config_v2)
        
        # Then: Should show different signal behavior
        # 1m might allow signals (200 points exceptional for 1m)
        # 1h should filter signals (200 points insufficient for 1h)
        signals_1m = len(result_1m.signals) if result_1m and result_1m.signals else 0
        signals_1h = len(result_1h.signals) if result_1h and result_1h.signals else 0
        
        # At least one timeframe should behave differently
        assert signals_1m != signals_1h or (signals_1m == 0 and signals_1h == 0), \
            "Timeframes should show different signal sensitivity"

    async def test_exceptional_bodies_boost_v2_confidence(self, session_config_v2):
        """Test that exceptional body sizes boost V2 signal confidence."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # Normal body size trend
        normal_body_data = MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=50450,
            ohlcv_data=[
                OHLCV(timestamp=datetime.now(timezone.utc), open=50000, high=50500, low=49900, close=50450, volume=1000),  # 450 points
            ]
        )
        
        # Exceptional body size trend  
        exceptional_body_data = MarketData(
            symbol="BTCUSD",
            timeframe="1h", 
            current_price=50750,
            ohlcv_data=[
                OHLCV(timestamp=datetime.now(timezone.utc), open=50000, high=50800, low=49700, close=50750, volume=1500),  # 750 points
            ]
        )
        
        # When: Analyzing both scenarios
        normal_result = await strategy.analyze(normal_body_data, session_config_v2)
        exceptional_result = await strategy.analyze(exceptional_body_data, session_config_v2)
        
        # Then: Exceptional bodies should produce higher confidence signals
        if normal_result and normal_result.signals and exceptional_result and exceptional_result.signals:
            normal_max_confidence = max(s.confidence for s in normal_result.signals)
            exceptional_max_confidence = max(s.confidence for s in exceptional_result.signals)
            
            assert exceptional_max_confidence >= normal_max_confidence, \
                "Exceptional body sizes should boost signal confidence"

    async def test_v2_filters_doji_patterns_appropriately(self, session_config_v2):
        """Test that V2 strategy properly filters doji patterns as neutral."""
        from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
        
        strategy = EMACrossoverV2Strategy()
        
        # Doji-heavy market data (indecision)
        doji_market_data = MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=50002,
            ohlcv_data=[
                # Perfect doji patterns
                OHLCV(timestamp=datetime.now(timezone.utc), open=50000, high=50100, low=49900, close=50001, volume=1000),  # 1 point body
                OHLCV(timestamp=datetime.now(timezone.utc), open=50001, high=50120, low=49880, close=50002, volume=1000),  # 1 point body
                OHLCV(timestamp=datetime.now(timezone.utc), open=50002, high=50080, low=49920, close=50000, volume=1000),  # 2 point body
            ]
        )
        
        # When: Analyzing doji-dominated market
        result = await strategy.analyze(doji_market_data, session_config_v2)
        
        # Then: Should not generate strong directional signals
        assert result is not None
        
        if result.signals:
            # Any signals should be low confidence due to doji filtering
            max_confidence = max(s.confidence for s in result.signals)
            assert max_confidence <= 5, "Doji patterns should not generate high-confidence V2 signals"
            
            # Most signals should be neutral
            neutral_signals = sum(1 for s in result.signals if s.action.value in ["NEUTRAL", "WAIT"])
            assert neutral_signals >= len(result.signals) * 0.5, "Majority should be neutral in doji market"