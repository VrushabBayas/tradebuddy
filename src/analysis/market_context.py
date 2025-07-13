"""
Market Context Analyzer.

Provides unified market state assessment combining trend, volatility, 
and structure analysis for better trading decisions.
"""

import structlog
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.core.models import OHLCV
from src.core.constants import TradingConstants

logger = structlog.get_logger(__name__)


class MarketState(str, Enum):
    """Market state classifications."""
    STRONG_TRENDING = "strong_trending"
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    CHOPPY = "choppy"
    TRANSITIONAL = "transitional"


class VolatilityLevel(str, Enum):
    """Volatility level classifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class MarketContext:
    """Comprehensive market context analysis."""
    
    # Market state
    state: MarketState
    confidence: float  # 0-100
    
    # Trend analysis
    trend_strength: float  # 0-100
    trend_quality: float  # 0-100
    trend_duration: int  # periods
    trend_direction: str  # "bullish", "bearish", "neutral"
    
    # Volatility analysis
    volatility_level: VolatilityLevel
    volatility_percentile: float  # 0-1
    atr_normalized: float  # ATR as % of price
    
    # Market structure
    structure_type: str  # "uptrend", "downtrend", "sideways"
    structure_strength: float  # 0-100
    swing_count: int
    
    # Risk assessment
    risk_level: str  # "low", "medium", "high", "extreme"
    trading_recommendation: str  # "favorable", "caution", "avoid"
    
    @property
    def is_favorable_for_trading(self) -> bool:
        """Check if market conditions are favorable for trading."""
        return (
            self.state in [MarketState.TRENDING, MarketState.STRONG_TRENDING] and
            self.volatility_level != VolatilityLevel.EXTREME and
            self.confidence >= 60
        )
    
    @property
    def summary(self) -> str:
        """Get human-readable market summary."""
        return (
            f"{self.state.value.title()} market "
            f"({self.confidence:.0f}% confidence) - "
            f"{self.volatility_level.value} volatility - "
            f"{self.trading_recommendation}"
        )


class MarketContextAnalyzer:
    """
    Unified market context analyzer.
    
    Combines multiple analysis techniques to provide comprehensive
    market state assessment for trading strategies.
    """
    
    def __init__(self):
        """Initialize market context analyzer."""
        # Import here to avoid circular imports
        from src.analysis.indicators import TechnicalIndicators
        self.indicators = TechnicalIndicators()
        logger.debug("Market context analyzer initialized")
    
    def analyze(self, data: List[OHLCV]) -> MarketContext:
        """
        Perform comprehensive market context analysis.
        
        Args:
            data: OHLCV data points
            
        Returns:
            MarketContext with complete analysis
            
        Raises:
            ValueError: If insufficient data provided
        """
        if len(data) < 20:
            raise ValueError(f"Insufficient data: {len(data)} periods, minimum 20 required")
        
        logger.debug("Analyzing market context", data_points=len(data))
        
        # Core trend analysis
        trend_analysis = self._analyze_trend(data)
        
        # Volatility analysis
        volatility_analysis = self._analyze_volatility(data)
        
        # Market structure analysis
        structure_analysis = self._analyze_structure(data)
        
        # Determine overall market state
        market_state, confidence = self._determine_market_state(
            trend_analysis, volatility_analysis, structure_analysis
        )
        
        # Risk assessment
        risk_analysis = self._assess_risk(trend_analysis, volatility_analysis, structure_analysis)
        
        context = MarketContext(
            # Market state
            state=market_state,
            confidence=confidence,
            
            # Trend analysis
            trend_strength=trend_analysis["strength"],
            trend_quality=trend_analysis["quality"],
            trend_duration=trend_analysis["duration"],
            trend_direction=trend_analysis["direction"],
            
            # Volatility analysis
            volatility_level=volatility_analysis["level"],
            volatility_percentile=volatility_analysis["percentile"],
            atr_normalized=volatility_analysis["atr_normalized"],
            
            # Market structure
            structure_type=structure_analysis["type"],
            structure_strength=structure_analysis["strength"],
            swing_count=structure_analysis["swing_count"],
            
            # Risk assessment
            risk_level=risk_analysis["level"],
            trading_recommendation=risk_analysis["recommendation"]
        )
        
        logger.debug(
            "Market context analysis completed",
            state=context.state.value,
            confidence=context.confidence,
            trend_strength=context.trend_strength,
            volatility_level=context.volatility_level.value
        )
        
        return context
    
    def _analyze_trend(self, data: List[OHLCV]) -> Dict[str, Any]:
        """Analyze trend characteristics."""
        # Use enhanced indicators for comprehensive trend analysis
        trend_strength = self.indicators.calculate_trend_strength(data)
        trend_quality = self.indicators.calculate_trend_quality(data)
        trend_duration = self.indicators.calculate_trend_duration(data)
        is_trending = self.indicators.is_trending_market(data)
        
        # Determine trend direction
        prices = [candle.close for candle in data]
        recent_change = (prices[-1] - prices[-10]) / prices[-10] * 100 if len(prices) >= 10 else 0
        
        if recent_change > 2:
            direction = "bullish"
        elif recent_change < -2:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return {
            "strength": trend_strength,
            "quality": trend_quality,
            "duration": trend_duration,
            "direction": direction,
            "is_trending": is_trending,
            "recent_change_pct": recent_change
        }
    
    def _analyze_volatility(self, data: List[OHLCV]) -> Dict[str, Any]:
        """Analyze volatility characteristics."""
        # Calculate volatility metrics
        volatility_percentile = self.indicators.calculate_volatility_percentile(data)
        atr_values = self.indicators.calculate_atr(data, 14)
        current_atr = atr_values[-1] if atr_values else 0
        current_price = data[-1].close
        
        # Normalize ATR as percentage of price
        atr_normalized = (current_atr / current_price * 100) if current_price > 0 else 0
        
        # Classify volatility level
        if volatility_percentile >= 0.9:
            level = VolatilityLevel.EXTREME
        elif volatility_percentile >= 0.7:
            level = VolatilityLevel.HIGH
        elif volatility_percentile >= 0.3:
            level = VolatilityLevel.NORMAL
        else:
            level = VolatilityLevel.LOW
        
        return {
            "level": level,
            "percentile": volatility_percentile,
            "atr_normalized": atr_normalized,
            "current_atr": current_atr
        }
    
    def _analyze_structure(self, data: List[OHLCV]) -> Dict[str, Any]:
        """Analyze market structure characteristics."""
        market_structure = self.indicators.calculate_market_structure(data)
        
        structure_type = market_structure.get("trend_structure", "sideways")
        structure_strength = market_structure.get("structure_strength", 0)
        swing_highs = market_structure.get("swing_highs", [])
        swing_lows = market_structure.get("swing_lows", [])
        swing_count = len(swing_highs) + len(swing_lows)
        
        return {
            "type": structure_type,
            "strength": structure_strength,
            "swing_count": swing_count,
            "swing_highs": len(swing_highs),
            "swing_lows": len(swing_lows)
        }
    
    def _determine_market_state(
        self, 
        trend_analysis: Dict[str, Any],
        volatility_analysis: Dict[str, Any],
        structure_analysis: Dict[str, Any]
    ) -> tuple[MarketState, float]:
        """Determine overall market state with confidence."""
        
        trend_strength = trend_analysis["strength"]
        trend_quality = trend_analysis["quality"]
        is_trending = trend_analysis["is_trending"]
        volatility_level = volatility_analysis["level"]
        structure_strength = structure_analysis["strength"]
        
        confidence_factors = []
        
        # Determine base market state
        if trend_strength >= 70 and trend_quality >= 70 and is_trending:
            state = MarketState.STRONG_TRENDING
            confidence_factors.append(85)  # High confidence for strong trends
        elif trend_strength >= 50 and is_trending:
            state = MarketState.TRENDING
            confidence_factors.append(70)  # Good confidence for trending
        elif trend_strength < 30 or volatility_level == VolatilityLevel.EXTREME:
            state = MarketState.CHOPPY
            confidence_factors.append(80)  # High confidence in choppy identification
        elif 30 <= trend_strength < 50:
            state = MarketState.TRANSITIONAL
            confidence_factors.append(50)  # Lower confidence for transitional
        else:
            state = MarketState.SIDEWAYS
            confidence_factors.append(65)  # Moderate confidence for sideways
        
        # Adjust confidence based on supporting factors
        if structure_strength >= 70:
            confidence_factors.append(75)  # Structure supports the state
        elif structure_strength >= 50:
            confidence_factors.append(60)
        else:
            confidence_factors.append(40)
        
        # Volatility consistency factor
        if volatility_level in [VolatilityLevel.LOW, VolatilityLevel.NORMAL]:
            confidence_factors.append(70)  # Normal volatility supports analysis
        elif volatility_level == VolatilityLevel.HIGH:
            confidence_factors.append(50)  # High volatility reduces confidence
        else:
            confidence_factors.append(30)  # Extreme volatility hurts confidence
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return state, overall_confidence
    
    def _assess_risk(
        self,
        trend_analysis: Dict[str, Any],
        volatility_analysis: Dict[str, Any], 
        structure_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Assess trading risk and provide recommendations."""
        
        volatility_level = volatility_analysis["level"]
        trend_strength = trend_analysis["strength"]
        structure_strength = structure_analysis["strength"]
        
        # Risk level assessment
        if volatility_level == VolatilityLevel.EXTREME:
            risk_level = "extreme"
            recommendation = "avoid"
        elif volatility_level == VolatilityLevel.HIGH and trend_strength < 50:
            risk_level = "high"
            recommendation = "avoid"
        elif volatility_level == VolatilityLevel.HIGH:
            risk_level = "high"
            recommendation = "caution"
        elif trend_strength >= 60 and structure_strength >= 60:
            risk_level = "low"
            recommendation = "favorable"
        elif trend_strength >= 40:
            risk_level = "medium"
            recommendation = "caution"
        else:
            risk_level = "high"
            recommendation = "avoid"
        
        return {
            "level": risk_level,
            "recommendation": recommendation
        }
    
    def get_trading_filters(self, context: MarketContext) -> Dict[str, Any]:
        """
        Get recommended trading filters based on market context.
        
        Args:
            context: Market context analysis
            
        Returns:
            Dictionary with filter recommendations
        """
        filters = {
            "min_trend_strength": 40,
            "min_signal_confidence": 6,
            "enable_volatility_filter": True,
            "enable_trend_filter": True,
            "position_size_multiplier": 1.0
        }
        
        # Adjust filters based on market state
        if context.state == MarketState.STRONG_TRENDING:
            filters.update({
                "min_trend_strength": 60,
                "min_signal_confidence": 7,
                "position_size_multiplier": 1.2  # Can be more aggressive
            })
        elif context.state in [MarketState.CHOPPY, MarketState.SIDEWAYS]:
            filters.update({
                "min_trend_strength": 70,
                "min_signal_confidence": 8,
                "position_size_multiplier": 0.5  # More conservative
            })
        
        # Adjust for volatility
        if context.volatility_level == VolatilityLevel.HIGH:
            filters["position_size_multiplier"] *= 0.7
            filters["min_signal_confidence"] += 1
        elif context.volatility_level == VolatilityLevel.EXTREME:
            filters["position_size_multiplier"] *= 0.3
            filters["min_signal_confidence"] = 9
        
        return filters