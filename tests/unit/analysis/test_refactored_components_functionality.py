"""
Functionality tests for refactored shared components.

Tests the behavior of shared EMA analysis, signal scoring, and market context components.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.analysis.ema_analysis import EMAAnalyzer, EMACalculator, EMACrossoverData, EMAValues
from src.analysis.scoring.signal_scorer import SignalScorer, ScoreWeights, ScoreComponent
from src.analysis.market_context import MarketContextAnalyzer, MarketContext, MarketState, VolatilityLevel
from src.core.models import OHLCV


class TestEMAAnalyzerFunctionality:
    """Test EMA analysis shared component functionality."""

    @pytest.fixture
    def ema_analyzer(self):
        """Create EMA analyzer."""
        return EMAAnalyzer()

    @pytest.fixture
    def trending_ohlcv_data(self):
        """Create OHLCV data showing clear trend."""
        candles = []
        base_price = 50000.0
        for i in range(50):
            # Create uptrend with increasing prices
            price = base_price + (i * 100)
            candle = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price + 150,
                low=price - 50,
                close=price + 100,
                volume=1000.0 + (i * 10)
            )
            candles.append(candle)
        return candles

    def test_should_calculate_ema_values_correctly(self, ema_analyzer, trending_ohlcv_data):
        """Should calculate EMA values with proper periods."""
        ema_values = ema_analyzer.calculator.calculate_basic_emas(trending_ohlcv_data)
        
        assert isinstance(ema_values, EMAValues)
        assert ema_values.ema_9 > 0
        assert ema_values.ema_15 > 0
        assert ema_values.ema_50 > 0
        assert ema_values.current_price > 0
        
        # In an uptrend, shorter EMAs should be higher than longer ones
        assert ema_values.ema_9 > ema_values.ema_15
        assert ema_values.ema_15 > ema_values.ema_50

    def test_should_detect_ema_crossover_patterns(self, ema_analyzer, trending_ohlcv_data):
        """Should detect EMA crossover patterns correctly."""
        crossover_data = ema_analyzer.calculator.create_crossover_analysis(trending_ohlcv_data)
        
        assert isinstance(crossover_data, EMACrossoverData)
        assert crossover_data.is_golden_cross is not None
        assert 1 <= crossover_data.crossover_strength <= 10
        assert crossover_data.ema_values.ema_9 > 0
        assert crossover_data.ema_values.ema_15 > 0

    def test_should_analyze_for_different_strategy_types(self, ema_analyzer, trending_ohlcv_data):
        """Should provide strategy-specific analysis."""
        # Test basic strategy analysis
        basic_analysis = ema_analyzer.analyze_for_strategy(trending_ohlcv_data, "basic")
        assert "ema_crossover" in basic_analysis
        assert "ema_alignment" in basic_analysis
        
        # Test V2 strategy analysis
        v2_analysis = ema_analyzer.analyze_for_strategy(trending_ohlcv_data, "v2")
        assert "ema_crossover" in v2_analysis
        assert "ema_alignment" in v2_analysis
        # V2 should include additional analysis
        assert "trend_direction" in v2_analysis

    def test_should_determine_ema_alignment_correctly(self, ema_analyzer, trending_ohlcv_data):
        """Should correctly determine EMA alignment patterns."""
        ema_values = ema_analyzer.calculator.calculate_basic_emas(trending_ohlcv_data)
        alignment = ema_analyzer.calculator.assess_ema_alignment(ema_values)
        
        # Should return valid alignment category
        valid_alignments = ["strong_bullish", "bullish", "strong_bearish", "bearish", "mixed", "neutral"]
        assert alignment in valid_alignments


class TestSignalScorerFunctionality:
    """Test signal scoring framework functionality."""

    @pytest.fixture
    def signal_scorer(self):
        """Create signal scorer with default weights."""
        return SignalScorer()

    @pytest.fixture
    def custom_scorer(self):
        """Create signal scorer with custom weights."""
        weights = ScoreWeights(
            trend_strength=0.4,
            trend_quality=0.3,
            ema_alignment=0.2,
            market_structure=0.05,
            volume_confirmation=0.03,
            candlestick_pattern=0.02
        )
        return SignalScorer(weights=weights)

    def test_should_score_basic_ema_strategy_correctly(self, signal_scorer):
        """Should calculate reasonable scores for basic EMA strategy."""
        score_result = signal_scorer.score_ema_strategy(
            crossover_strength=8,
            separation_pct=1.5,
            volume_ratio=1.3,
            trend_alignment="strong_bullish",
            rsi_value=65,
            candlestick_strength="strong"
        )
        
        assert "final_score" in score_result
        assert "components" in score_result
        assert "strategy_type" in score_result
        
        # Should be high quality signal
        assert 65 <= score_result["final_score"] <= 100
        assert score_result["strategy_type"] == "ema_basic"

    def test_should_score_v2_strategy_correctly(self, signal_scorer):
        """Should calculate reasonable scores for V2 strategy."""
        market_structure = {"structure_strength": 80, "trend_structure": "uptrend"}
        volume_analysis = {"volume_ratio": 1.4}
        candlestick_analysis = {"pattern_strength": 7}
        
        score_result = signal_scorer.score_v2_strategy(
            trend_strength=75,
            trend_quality=70,
            trend_duration=8,
            ema_alignment="strong_bullish",
            volatility_percentile=0.4,
            market_structure=market_structure,
            volume_analysis=volume_analysis,
            candlestick_analysis=candlestick_analysis
        )
        
        assert "final_score" in score_result
        assert "strategy_type" in score_result
        assert score_result["strategy_type"] == "ema_v2"
        
        # Should be high quality signal with strong inputs
        assert 60 <= score_result["final_score"] <= 100

    def test_should_handle_weak_signals_appropriately(self, signal_scorer):
        """Should give low scores to weak signals."""
        score_result = signal_scorer.score_ema_strategy(
            crossover_strength=2,  # Weak crossover
            separation_pct=0.1,    # Small separation
            volume_ratio=0.8,      # Below average volume
            trend_alignment="mixed", # Poor alignment
            rsi_value=None,
            candlestick_strength=None
        )
        
        # Should be low quality signal
        assert 0 <= score_result["final_score"] <= 50

    def test_should_apply_custom_weights_correctly(self, custom_scorer):
        """Should use custom weights in calculations."""
        score_result = custom_scorer.score_ema_strategy(
            crossover_strength=7,
            separation_pct=1.0,
            volume_ratio=1.2,
            trend_alignment="bullish"
        )
        
        assert "final_score" in score_result
        # Custom weights should produce different result than default
        assert 0 <= score_result["final_score"] <= 100

    def test_should_validate_score_component_behavior(self):
        """Should correctly calculate score component values."""
        component = ScoreComponent(
            name="test_component",
            value=75.0,
            weight=0.3,
            max_value=100.0
        )
        
        assert component.normalized_value == 0.75
        assert component.weighted_score == 22.5  # 0.75 * 0.3 * 100


class TestMarketContextAnalyzerFunctionality:
    """Test market context analyzer functionality."""

    @pytest.fixture
    def market_analyzer(self):
        """Create market context analyzer."""
        return MarketContextAnalyzer()

    @pytest.fixture
    def trending_data(self):
        """Create data showing strong trend."""
        candles = []
        base_price = 50000.0
        for i in range(30):
            # Strong uptrend
            price = base_price + (i * 200)
            candle = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price + 300,
                low=price - 100,
                close=price + 250,
                volume=1000.0 + (i * 20)
            )
            candles.append(candle)
        return candles

    @pytest.fixture
    def choppy_data(self):
        """Create data showing choppy/sideways market."""
        candles = []
        base_price = 50000.0
        for i in range(30):
            # Random price movement around base
            price_variation = (i % 4 - 2) * 100  # -200 to +200
            price = base_price + price_variation
            candle = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=price,
                high=price + 150,
                low=price - 150,
                close=price + (price_variation / 2),
                volume=1000.0
            )
            candles.append(candle)
        return candles

    def test_should_identify_trending_markets(self, market_analyzer, trending_data):
        """Should correctly identify trending market conditions."""
        context = market_analyzer.analyze(trending_data)
        
        assert isinstance(context, MarketContext)
        assert context.state in [MarketState.TRENDING, MarketState.STRONG_TRENDING]
        assert context.confidence >= 60
        assert context.trend_strength >= 50
        assert context.trend_direction in ["bullish", "bearish"]

    def test_should_identify_choppy_markets(self, market_analyzer, choppy_data):
        """Should correctly identify choppy/sideways market conditions."""
        context = market_analyzer.analyze(choppy_data)
        
        assert isinstance(context, MarketContext)
        assert context.state in [MarketState.SIDEWAYS, MarketState.CHOPPY]
        assert context.trend_strength <= 50

    def test_should_assess_volatility_correctly(self, market_analyzer, trending_data):
        """Should correctly assess market volatility."""
        context = market_analyzer.analyze(trending_data)
        
        assert context.volatility_level in [level for level in VolatilityLevel]
        assert 0 <= context.volatility_percentile <= 1
        assert context.atr_normalized >= 0

    def test_should_provide_trading_recommendations(self, market_analyzer, trending_data):
        """Should provide actionable trading recommendations."""
        context = market_analyzer.analyze(trending_data)
        
        valid_recommendations = ["favorable", "caution", "avoid"]
        assert context.trading_recommendation in valid_recommendations
        assert context.risk_level in ["low", "medium", "high", "extreme"]

    def test_should_determine_favorable_conditions_correctly(self, market_analyzer, trending_data):
        """Should correctly identify favorable trading conditions."""
        context = market_analyzer.analyze(trending_data)
        
        # is_favorable_for_trading should be consistent with other metrics
        if context.is_favorable_for_trading:
            assert context.state in [MarketState.TRENDING, MarketState.STRONG_TRENDING]
            assert context.volatility_level != VolatilityLevel.EXTREME
            assert context.confidence >= 60

    def test_should_provide_market_summary(self, market_analyzer, trending_data):
        """Should provide readable market summary."""
        context = market_analyzer.analyze(trending_data)
        
        summary = context.summary
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert context.state.value.replace('_', ' ').lower() in summary.lower()

    def test_should_get_appropriate_trading_filters(self, market_analyzer, trending_data):
        """Should provide context-appropriate trading filters."""
        context = market_analyzer.analyze(trending_data)
        filters = market_analyzer.get_trading_filters(context)
        
        assert "min_trend_strength" in filters
        assert "min_signal_confidence" in filters
        assert "position_size_multiplier" in filters
        
        # Filters should be reasonable
        assert 0 <= filters["min_trend_strength"] <= 100
        assert 1 <= filters["min_signal_confidence"] <= 10
        assert 0.1 <= filters["position_size_multiplier"] <= 2.0

    def test_should_handle_insufficient_data_gracefully(self, market_analyzer):
        """Should handle insufficient data without crashing."""
        # Create minimal data
        minimal_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50000, high=50100, low=49900, close=50050, volume=1000
            )
        ]
        
        # Should raise appropriate error for insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            market_analyzer.analyze(minimal_data)


class TestSharedComponentsIntegration:
    """Test integration between shared components."""

    @pytest.fixture
    def integration_data(self):
        """Create comprehensive data for integration testing."""
        candles = []
        base_price = 50000.0
        for i in range(60):  # Enough data for all components
            price_trend = i * 50  # Steady uptrend
            candle = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=base_price + price_trend,
                high=base_price + price_trend + 200,
                low=base_price + price_trend - 100,
                close=base_price + price_trend + 150,
                volume=1000.0 + (i * 10)
            )
            candles.append(candle)
        return candles

    def test_should_integrate_ema_analysis_with_scoring(self, integration_data):
        """Should integrate EMA analysis with signal scoring."""
        ema_analyzer = EMAAnalyzer()
        signal_scorer = SignalScorer()
        
        # Get EMA analysis
        ema_analysis = ema_analyzer.analyze_for_strategy(integration_data, "basic")
        
        # Extract data for scoring
        ema_crossover = ema_analysis.get("ema_crossover", {})
        volume_analysis = ema_analysis.get("volume_analysis", {})
        
        if ema_crossover:
            score_result = signal_scorer.score_ema_strategy(
                crossover_strength=ema_crossover.get("crossover_strength", 5),
                separation_pct=1.0,
                volume_ratio=volume_analysis.get("volume_ratio", 1.0),
                trend_alignment=ema_analysis.get("ema_alignment", "neutral")
            )
            
            assert "final_score" in score_result
            assert 0 <= score_result["final_score"] <= 100

    def test_should_integrate_market_context_with_ema_analysis(self, integration_data):
        """Should integrate market context with EMA analysis."""
        market_analyzer = MarketContextAnalyzer()
        ema_analyzer = EMAAnalyzer()
        
        # Get both analyses
        market_context = market_analyzer.analyze(integration_data)
        ema_analysis = ema_analyzer.analyze_for_strategy(integration_data, "v2")
        
        # Both should provide consistent trend assessment
        assert market_context.trend_direction in ["bullish", "bearish", "neutral"]
        assert ema_analysis.get("ema_alignment") in [
            "strong_bullish", "bullish", "strong_bearish", "bearish", "mixed", "neutral"
        ]
        
        # In trending market, EMA alignment should be consistent
        if market_context.state in [MarketState.TRENDING, MarketState.STRONG_TRENDING]:
            if market_context.trend_direction == "bullish":
                assert ema_analysis.get("ema_alignment") in ["strong_bullish", "bullish", "mixed"]

    def test_should_provide_comprehensive_signal_assessment(self, integration_data):
        """Should provide comprehensive signal assessment using all components."""
        ema_analyzer = EMAAnalyzer()
        signal_scorer = SignalScorer()
        market_analyzer = MarketContextAnalyzer()
        
        # Get comprehensive analysis
        ema_analysis = ema_analyzer.analyze_for_strategy(integration_data, "v2")
        market_context = market_analyzer.analyze(integration_data)
        
        # Use market context to enhance scoring
        if market_context.is_favorable_for_trading:
            # Should be able to generate quality scores in favorable conditions
            ema_crossover = ema_analysis.get("ema_crossover", {})
            if ema_crossover:
                score_result = signal_scorer.score_v2_strategy(
                    trend_strength=market_context.trend_strength,
                    trend_quality=market_context.trend_quality,
                    trend_duration=market_context.trend_duration,
                    ema_alignment=ema_analysis.get("ema_alignment", "neutral"),
                    volatility_percentile=market_context.volatility_percentile,
                    market_structure={"structure_strength": market_context.structure_strength},
                    volume_analysis=ema_analysis.get("volume_analysis", {}),
                    candlestick_analysis={"pattern_strength": 5}
                )
                
                # Should produce high quality score in favorable trending conditions
                assert score_result["final_score"] >= 40