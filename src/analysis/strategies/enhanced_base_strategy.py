"""
Enhanced base strategy with common EMA functionality.

Extends BaseStrategy with shared EMA analysis capabilities, caching,
and improved error handling for EMA-based trading strategies.
"""

import structlog
from typing import Dict, Any, Optional
from abc import abstractmethod

from src.analysis.strategies.base_strategy import BaseStrategy
from src.analysis.ema_analysis import EMAAnalyzer, EMACrossoverData
from src.analysis.scoring.signal_scorer import SignalScorer, ScoreWeights
from src.analysis.market_context import MarketContextAnalyzer, MarketContext
from src.core.models import MarketData, SessionConfig, AnalysisResult
from src.core.exceptions import StrategyError, DataValidationError

logger = structlog.get_logger(__name__)


class EnhancedEMABaseStrategy(BaseStrategy):
    """
    Enhanced base strategy with common EMA functionality.
    
    Provides shared capabilities for EMA-based strategies including:
    - Unified EMA analysis
    - Signal scoring framework
    - Market context assessment
    - Calculation caching
    - Enhanced error handling
    """
    
    def __init__(self):
        """Initialize enhanced EMA base strategy."""
        super().__init__()
        
        # Shared analysis components
        self.ema_analyzer = EMAAnalyzer()
        self.signal_scorer = SignalScorer()
        self.market_analyzer = MarketContextAnalyzer()
        
        # Calculation cache
        self._analysis_cache = {}
        self._cache_enabled = True
        
        logger.debug("Enhanced EMA base strategy initialized")
    
    def _get_minimum_periods_for_ema(self) -> int:
        """Get minimum periods required for EMA analysis."""
        return 20  # Base requirement, can be overridden
    
    async def _calculate_enhanced_ema_analysis(
        self,
        market_data: MarketData,
        session_config: SessionConfig,
        strategy_type: str = "basic"
    ) -> Dict[str, Any]:
        """
        Calculate enhanced EMA analysis with caching.
        
        Args:
            market_data: Market data to analyze
            session_config: Session configuration
            strategy_type: Strategy type for optimization ("basic", "v2", "combined")
            
        Returns:
            Enhanced technical analysis with EMA data
            
        Raises:
            DataValidationError: If insufficient data or invalid configuration
        """
        # Create cache key
        cache_key = self._create_cache_key(market_data, strategy_type)
        
        # Check cache first
        if self._cache_enabled and cache_key in self._analysis_cache:
            logger.debug("Using cached EMA analysis", cache_key=cache_key)
            return self._analysis_cache[cache_key]
        
        try:
            # Validate data sufficiency
            min_periods = self._get_minimum_periods_for_ema()
            if len(market_data.ohlcv_data) < min_periods:
                raise DataValidationError(
                    f"Insufficient data for EMA analysis: {len(market_data.ohlcv_data)} periods provided, "
                    f"minimum {min_periods} required"
                )
            
            logger.debug(
                "Calculating enhanced EMA analysis",
                symbol=market_data.symbol,
                data_points=len(market_data.ohlcv_data),
                strategy_type=strategy_type
            )
            
            # Get base technical analysis
            base_analysis = await self._calculate_technical_analysis(market_data)
            
            # Perform EMA-specific analysis
            ema_analysis = self.ema_analyzer.analyze_for_strategy(
                market_data.ohlcv_data, strategy_type
            )
            
            # Perform market context analysis
            market_context = self.market_analyzer.analyze(market_data.ohlcv_data)
            
            # Combine all analyses
            enhanced_analysis = {
                **base_analysis,
                **ema_analysis,
                "market_context": {
                    "state": market_context.state.value,
                    "confidence": market_context.confidence,
                    "trend_strength": market_context.trend_strength,
                    "trend_quality": market_context.trend_quality,
                    "volatility_level": market_context.volatility_level.value,
                    "trading_recommendation": market_context.trading_recommendation,
                    "is_favorable": market_context.is_favorable_for_trading
                },
                "strategy_type": strategy_type
            }
            
            # Cache the result
            if self._cache_enabled:
                self._analysis_cache[cache_key] = enhanced_analysis
            
            logger.debug(
                "Enhanced EMA analysis completed",
                strategy_type=strategy_type,
                market_state=market_context.state.value,
                trend_strength=market_context.trend_strength
            )
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(
                "Enhanced EMA analysis failed",
                symbol=market_data.symbol,
                strategy_type=strategy_type,
                error=str(e)
            )
            raise StrategyError(f"Enhanced EMA analysis failed: {str(e)}")
    
    def _calculate_signal_score(
        self,
        analysis: Dict[str, Any],
        strategy_type: str = "basic"
    ) -> Dict[str, Any]:
        """
        Calculate signal quality score using the scoring framework.
        
        Args:
            analysis: Technical analysis results
            strategy_type: Strategy type for score calculation
            
        Returns:
            Signal scoring results
        """
        try:
            if strategy_type == "v2":
                return self._calculate_v2_signal_score(analysis)
            else:
                return self._calculate_basic_signal_score(analysis)
                
        except Exception as e:
            logger.warning("Signal scoring failed", error=str(e))
            return {"final_score": 50, "components": {}, "error": str(e)}
    
    def _calculate_basic_signal_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate signal score for basic EMA strategy."""
        ema_crossover = analysis.get("ema_crossover", {})
        ema_context = analysis.get("ema_context", {})
        volume_analysis = analysis.get("volume_analysis", {})
        
        crossover_strength = ema_crossover.get("crossover_strength", 5)
        separation_pct = ema_context.get("separation_pct", 0)
        volume_ratio = volume_analysis.get("volume_ratio", 1.0)
        trend_alignment = ema_context.get("alignment", "neutral")
        
        # Optional components
        rsi_values = analysis.get("rsi_values", [])
        current_rsi = rsi_values[-1] if rsi_values else None
        
        candlestick_analysis = analysis.get("candlestick_analysis", {})
        candlestick_strength = candlestick_analysis.get("strength", None)
        
        return self.signal_scorer.score_ema_strategy(
            crossover_strength=crossover_strength,
            separation_pct=separation_pct,
            volume_ratio=volume_ratio,
            trend_alignment=trend_alignment,
            rsi_value=current_rsi,
            candlestick_strength=candlestick_strength
        )
    
    def _calculate_v2_signal_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate signal score for V2 EMA strategy."""
        market_context = analysis.get("market_context", {})
        volume_analysis = analysis.get("volume_analysis", {})
        candlestick_analysis = analysis.get("candlestick_analysis", {})
        
        # Extract market structure if available
        market_structure = analysis.get("market_structure", {
            "structure_strength": 50,
            "trend_structure": "sideways"
        })
        
        return self.signal_scorer.score_v2_strategy(
            trend_strength=market_context.get("trend_strength", 50),
            trend_quality=market_context.get("trend_quality", 50),
            trend_duration=analysis.get("trend_duration", 5),
            ema_alignment=analysis.get("ema_alignment", "neutral"),
            volatility_percentile=analysis.get("volatility_percentile", 0.5),
            market_structure=market_structure,
            volume_analysis=volume_analysis,
            candlestick_analysis=candlestick_analysis
        )
    
    def _create_cache_key(self, market_data: MarketData, strategy_type: str) -> str:
        """Create cache key for analysis results."""
        # Use last candle timestamp and data length for cache key
        last_timestamp = market_data.ohlcv_data[-1].timestamp if market_data.ohlcv_data else "none"
        return f"{market_data.symbol}_{strategy_type}_{len(market_data.ohlcv_data)}_{last_timestamp}"
    
    def _filter_signals_by_market_context(
        self,
        analysis_result: AnalysisResult,
        market_context: MarketContext
    ) -> AnalysisResult:
        """
        Filter signals based on market context recommendations.
        
        Args:
            analysis_result: Original analysis result
            market_context: Market context analysis
            
        Returns:
            Filtered analysis result
        """
        if not market_context.is_favorable_for_trading:
            # Remove or downgrade signals in unfavorable conditions
            filtered_signals = []
            for signal in analysis_result.signals:
                if signal.confidence >= 8:  # Only keep very high confidence signals
                    # Reduce confidence in unfavorable conditions
                    signal.confidence = max(signal.confidence - 2, 1)
                    filtered_signals.append(signal)
            
            analysis_result.signals = filtered_signals
            
            logger.debug(
                "Signals filtered by market context",
                original_count=len(analysis_result.signals),
                filtered_count=len(filtered_signals),
                market_state=market_context.state.value,
                recommendation=market_context.trading_recommendation
            )
        
        return analysis_result
    
    def clear_cache(self):
        """Clear analysis cache."""
        self._analysis_cache.clear()
        self.ema_analyzer.calculator.clear_cache()
        logger.debug("Enhanced strategy cache cleared")
    
    def enable_caching(self, enabled: bool = True):
        """Enable or disable analysis caching."""
        self._cache_enabled = enabled
        logger.debug("Caching", enabled="enabled" if enabled else "disabled")
    
    @abstractmethod
    def _get_strategy_specific_weights(self) -> Optional[ScoreWeights]:
        """Get strategy-specific scoring weights."""
        pass
    
    @abstractmethod
    def _apply_strategy_specific_filters(
        self,
        analysis_result: AnalysisResult,
        analysis: Dict[str, Any],
        session_config: SessionConfig
    ) -> AnalysisResult:
        """Apply strategy-specific signal filters."""
        pass