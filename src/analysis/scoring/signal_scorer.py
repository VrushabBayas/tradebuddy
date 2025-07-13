"""
Configurable signal scoring framework.

Provides flexible, reusable components for assessing signal quality across
different trading strategies.
"""

import structlog
from typing import Dict, List, Optional, Protocol, Any
from dataclasses import dataclass, field
from decimal import Decimal

logger = structlog.get_logger(__name__)


@dataclass
class ScoreComponent:
    """Individual scoring component with weight and value."""
    name: str
    value: float  # Raw score value (0-100)
    weight: float  # Weight in final calculation (0.0-1.0)
    max_value: float = 100.0  # Maximum possible value
    
    @property
    def normalized_value(self) -> float:
        """Get normalized value (0.0-1.0)."""
        return min(self.value / self.max_value, 1.0) if self.max_value > 0 else 0.0
    
    @property
    def weighted_score(self) -> float:
        """Get weighted contribution to final score."""
        return self.normalized_value * self.weight * 100


@dataclass
class ScoreWeights:
    """Configuration for scoring component weights."""
    trend_strength: float = 0.3
    trend_quality: float = 0.2
    ema_alignment: float = 0.2
    market_structure: float = 0.15
    volume_confirmation: float = 0.1
    candlestick_pattern: float = 0.05
    
    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = sum([
            self.trend_strength, self.trend_quality, self.ema_alignment,
            self.market_structure, self.volume_confirmation, self.candlestick_pattern
        ])
        if abs(total - 1.0) > 0.01:  # Allow small floating point differences
            raise ValueError(f"Score weights must sum to 1.0, got {total}")


class ScoreCalculator(Protocol):
    """Protocol for score calculation strategies."""
    
    def calculate(self, components: List[ScoreComponent]) -> Dict[str, float]:
        """Calculate final score from components."""
        ...


class WeightedScoreCalculator:
    """Standard weighted score calculator."""
    
    def calculate(self, components: List[ScoreComponent]) -> Dict[str, float]:
        """
        Calculate weighted average score from components.
        
        Args:
            components: List of score components
            
        Returns:
            Dictionary with scoring details
        """
        if not components:
            return {"final_score": 0.0, "total_weight": 0.0}
        
        total_weighted_score = sum(comp.weighted_score for comp in components)
        total_weight = sum(comp.weight for comp in components)
        
        # Normalize by total weight to handle partial component sets
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        return {
            "final_score": max(0.0, min(100.0, final_score)),
            "total_weight": total_weight,
            "component_count": len(components),
            "weighted_sum": total_weighted_score
        }


class VolatilityAdjustedCalculator:
    """Score calculator with volatility adjustments."""
    
    def __init__(self, volatility_threshold: float = 0.7, volatility_penalty: float = 0.2):
        """
        Initialize volatility-adjusted calculator.
        
        Args:
            volatility_threshold: Volatility percentile threshold for penalty
            volatility_penalty: Score reduction factor for high volatility
        """
        self.volatility_threshold = volatility_threshold
        self.volatility_penalty = volatility_penalty
    
    def calculate(self, components: List[ScoreComponent]) -> Dict[str, float]:
        """Calculate score with volatility adjustments."""
        # Use base weighted calculation
        base_calculator = WeightedScoreCalculator()
        result = base_calculator.calculate(components)
        
        # Apply volatility adjustment if volatility component is present
        volatility_multiplier = 1.0
        for comp in components:
            if comp.name == "volatility_percentile" and comp.value > self.volatility_threshold:
                volatility_multiplier = 1.0 - self.volatility_penalty
                result["volatility_adjusted"] = True
                break
        
        result["final_score"] *= volatility_multiplier
        result["volatility_multiplier"] = volatility_multiplier
        
        return result


class SignalScorer:
    """
    Configurable signal quality scorer.
    
    Provides flexible framework for calculating signal quality scores
    from multiple components with different weights and calculation strategies.
    """
    
    def __init__(
        self, 
        weights: Optional[ScoreWeights] = None,
        calculator: Optional[ScoreCalculator] = None
    ):
        """
        Initialize signal scorer.
        
        Args:
            weights: Component weights configuration
            calculator: Score calculation strategy
        """
        self.weights = weights or ScoreWeights()
        self.calculator = calculator or WeightedScoreCalculator()
        logger.debug("Signal scorer initialized", weights=self.weights)
    
    def score_ema_strategy(
        self,
        crossover_strength: float,
        separation_pct: float,
        volume_ratio: float,
        trend_alignment: str,
        rsi_value: Optional[float] = None,
        candlestick_strength: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Score basic EMA crossover strategy signals.
        
        Args:
            crossover_strength: EMA crossover strength (1-10)
            separation_pct: EMA separation percentage
            volume_ratio: Volume vs average ratio
            trend_alignment: Trend alignment category
            rsi_value: RSI value for momentum assessment
            candlestick_strength: Candlestick pattern strength
            
        Returns:
            Dictionary with scoring details
        """
        components = []
        
        # Crossover strength component (40% for basic strategy)
        components.append(ScoreComponent(
            name="crossover_strength",
            value=crossover_strength * 10,  # Scale 1-10 to 0-100
            weight=0.4,
            max_value=100
        ))
        
        # EMA separation component (25%)
        separation_score = min(100, separation_pct * 20)  # Scale separation to 0-100
        components.append(ScoreComponent(
            name="ema_separation",
            value=separation_score,
            weight=0.25
        ))
        
        # Volume component (20%)
        volume_score = min(100, volume_ratio * 50)
        components.append(ScoreComponent(
            name="volume_confirmation", 
            value=volume_score,
            weight=0.2
        ))
        
        # Trend alignment component (15%)
        alignment_scores = {
            "strong_bullish": 100, "strong_bearish": 100,
            "bullish": 80, "bearish": 80,
            "mixed": 40, "neutral": 20
        }
        alignment_score = alignment_scores.get(trend_alignment, 20)
        components.append(ScoreComponent(
            name="trend_alignment",
            value=alignment_score,
            weight=0.15
        ))
        
        # Optional RSI component
        if rsi_value is not None:
            rsi_score = self._calculate_rsi_score(rsi_value)
            components.append(ScoreComponent(
                name="rsi_momentum",
                value=rsi_score,
                weight=0.05  # Small additional weight
            ))
        
        # Optional candlestick component
        if candlestick_strength:
            candle_scores = {"strong": 100, "moderate": 70, "weak": 40}
            candle_score = candle_scores.get(candlestick_strength, 40)
            components.append(ScoreComponent(
                name="candlestick_pattern",
                value=candle_score,
                weight=0.05  # Small additional weight
            ))
        
        # Calculate final score
        result = self.calculator.calculate(components)
        result["components"] = {comp.name: comp.weighted_score for comp in components}
        result["strategy_type"] = "ema_basic"
        
        logger.debug(
            "EMA strategy scored",
            final_score=result["final_score"],
            component_count=len(components)
        )
        
        return result
    
    def score_v2_strategy(
        self,
        trend_strength: float,
        trend_quality: float,
        trend_duration: int,
        ema_alignment: str,
        volatility_percentile: float,
        market_structure: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        candlestick_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score enhanced V2 strategy signals.
        
        Args:
            trend_strength: Trend strength score (0-100)
            trend_quality: Trend quality score (0-100)
            trend_duration: Trend duration in periods
            ema_alignment: EMA alignment pattern
            volatility_percentile: Volatility percentile (0-1)
            market_structure: Market structure analysis
            volume_analysis: Volume analysis results
            candlestick_analysis: Candlestick pattern analysis
            
        Returns:
            Dictionary with V2 scoring details
        """
        components = []
        
        # Trend strength component (using configured weight)
        components.append(ScoreComponent(
            name="trend_strength",
            value=trend_strength,
            weight=self.weights.trend_strength
        ))
        
        # Trend quality component
        components.append(ScoreComponent(
            name="trend_quality",
            value=trend_quality,
            weight=self.weights.trend_quality
        ))
        
        # EMA alignment component
        alignment_scores = {
            "strong_bullish": 100, "strong_bearish": 100,
            "bullish": 80, "bearish": 80,
            "mixed": 40, "neutral": 20
        }
        alignment_score = alignment_scores.get(ema_alignment, 20)
        components.append(ScoreComponent(
            name="ema_alignment",
            value=alignment_score,
            weight=self.weights.ema_alignment
        ))
        
        # Market structure component
        structure_score = market_structure.get("structure_strength", 50)
        components.append(ScoreComponent(
            name="market_structure",
            value=structure_score,
            weight=self.weights.market_structure
        ))
        
        # Volume confirmation component
        volume_ratio = volume_analysis.get("volume_ratio", 1.0)
        volume_score = min(100, volume_ratio * 50)
        components.append(ScoreComponent(
            name="volume_confirmation",
            value=volume_score,
            weight=self.weights.volume_confirmation
        ))
        
        # Candlestick pattern component
        pattern_strength = candlestick_analysis.get("pattern_strength", 5)
        candlestick_score = (pattern_strength / 10) * 100
        components.append(ScoreComponent(
            name="candlestick_pattern",
            value=candlestick_score,
            weight=self.weights.candlestick_pattern
        ))
        
        # Add volatility as special component for volatility-adjusted calculator
        components.append(ScoreComponent(
            name="volatility_percentile",
            value=volatility_percentile * 100,  # Convert to 0-100 scale
            weight=0.0  # No direct weight, used for adjustment
        ))
        
        # Calculate final score
        result = self.calculator.calculate(components)
        result["components"] = {comp.name: comp.weighted_score for comp in components}
        result["strategy_type"] = "ema_v2"
        
        # Add V2-specific metrics
        result["trend_duration"] = trend_duration
        result["ema_alignment"] = ema_alignment
        result["volatility_percentile"] = volatility_percentile
        
        logger.debug(
            "V2 strategy scored",
            final_score=result["final_score"],
            trend_strength=trend_strength,
            trend_quality=trend_quality,
            ema_alignment=ema_alignment
        )
        
        return result
    
    def _calculate_rsi_score(self, rsi_value: float) -> float:
        """Calculate RSI momentum score."""
        if 30 <= rsi_value <= 70:  # Neutral RSI range
            return 70  # Good for trend following
        elif rsi_value > 70:  # Overbought
            return 40  # Caution
        elif rsi_value < 30:  # Oversold
            return 40  # Caution  
        elif rsi_value > 50:  # Bullish momentum
            return 85
        else:  # Bearish momentum
            return 85
    
    def compare_strategies(
        self, 
        scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple strategy scores.
        
        Args:
            scores: List of scoring results from different strategies
            
        Returns:
            Comparison analysis
        """
        if not scores:
            return {"best_strategy": None, "scores": []}
        
        # Sort by final score
        sorted_scores = sorted(scores, key=lambda x: x["final_score"], reverse=True)
        
        return {
            "best_strategy": sorted_scores[0]["strategy_type"],
            "best_score": sorted_scores[0]["final_score"],
            "scores": sorted_scores,
            "score_difference": sorted_scores[0]["final_score"] - sorted_scores[-1]["final_score"] if len(sorted_scores) > 1 else 0
        }