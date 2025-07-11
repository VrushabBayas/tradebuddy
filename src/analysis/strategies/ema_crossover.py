"""
EMA Crossover trading strategy implementation.

Uses exponential moving average crossovers to identify trend changes
and generate trading signals with volume confirmation.
"""

import structlog

from src.analysis.strategies.base_strategy import BaseStrategy
from src.core.models import (
    MarketData,
    AnalysisResult,
    SessionConfig,
    StrategyType
)
from src.core.constants import TradingConstants
from src.core.exceptions import StrategyError

logger = structlog.get_logger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover trading strategy.
    
    Strategy Logic:
    - Uses 9 EMA and 15 EMA for trend identification
    - Golden Cross (9 EMA > 15 EMA): Bullish signal
    - Death Cross (9 EMA < 15 EMA): Bearish signal
    - Requires volume confirmation for signal validation
    - Crossover strength determines signal confidence
    
    Signal Generation:
    - BUY: Golden cross with volume confirmation
    - SELL: Death cross with volume confirmation
    - NEUTRAL: EMAs converging but no clear cross
    - WAIT: Low volume or weak crossover strength
    """

    def __init__(self):
        """Initialize EMA Crossover strategy."""
        super().__init__()
        self.strategy_type = StrategyType.EMA_CROSSOVER
        
        logger.info("EMA Crossover strategy initialized")

    def _get_minimum_periods(self) -> int:
        """EMA Crossover requires enough data for reliable EMA calculation."""
        return TradingConstants.EMA_LONG_PERIOD + 5  # 20 periods minimum

    async def analyze(
        self,
        market_data: MarketData,
        session_config: SessionConfig
    ) -> AnalysisResult:
        """
        Analyze market data using EMA Crossover strategy.
        
        Args:
            market_data: Market data to analyze
            session_config: Session configuration
            
        Returns:
            AnalysisResult with EMA Crossover analysis
            
        Raises:
            DataValidationError: Invalid input data
            StrategyError: Analysis execution error
        """
        logger.info(
            "Starting EMA Crossover analysis",
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            current_price=market_data.current_price,
            data_points=len(market_data.ohlcv_data)
        )
        
        try:
            # Validate inputs
            self._validate_market_data(market_data)
            self._validate_session_config(session_config)
            
            # Calculate technical analysis with EMA focus
            technical_analysis = await self._calculate_ema_analysis(market_data)
            
            # Generate AI analysis with EMA context
            analysis_result = await self._generate_ai_analysis(
                market_data, technical_analysis, session_config
            )
            
            # Filter signals by confidence threshold
            filtered_result = self._filter_signals_by_confidence(
                analysis_result, session_config.confidence_threshold
            )
            
            # Add EMA specific context to the result
            filtered_result = self._add_ema_context(filtered_result, technical_analysis)
            
            # Handle logging with both dict and Pydantic model support
            ema_crossover = technical_analysis.get('ema_crossover')
            crossover_type = "unknown"
            crossover_strength = 0
            
            if ema_crossover:
                if hasattr(ema_crossover, 'is_golden_cross'):
                    crossover_type = "golden_cross" if ema_crossover.is_golden_cross else "death_cross"
                    crossover_strength = getattr(ema_crossover, 'crossover_strength', 0)
                elif isinstance(ema_crossover, dict):
                    crossover_type = "golden_cross" if ema_crossover.get('is_golden_cross') else "death_cross"
                    crossover_strength = ema_crossover.get('crossover_strength', 0)
            
            logger.info(
                "EMA Crossover analysis completed",
                symbol=market_data.symbol,
                signals_count=len(filtered_result.signals),
                crossover_type=crossover_type,
                crossover_strength=crossover_strength,
                primary_signal=str(filtered_result.primary_signal.action) if filtered_result.primary_signal else None
            )
            
            return filtered_result
            
        except Exception as e:
            logger.error(
                "EMA Crossover analysis failed",
                symbol=market_data.symbol,
                error=str(e)
            )
            raise StrategyError(f"EMA Crossover analysis failed: {str(e)}")

    async def _calculate_ema_analysis(self, market_data: MarketData) -> dict:
        """
        Calculate EMA Crossover specific technical analysis.
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Technical analysis with EMA emphasis
        """
        logger.debug(
            "Calculating EMA technical analysis",
            symbol=market_data.symbol,
            data_points=len(market_data.ohlcv_data)
        )
        
        # Get comprehensive technical analysis
        analysis = await self._calculate_technical_analysis(market_data)
        
        # Enhance with EMA specific calculations
        analysis = self._enhance_ema_analysis(analysis, market_data)
        
        return analysis

    def _enhance_ema_analysis(self, analysis: dict, market_data: MarketData) -> dict:
        """
        Enhance analysis with EMA specific metrics.
        
        Args:
            analysis: Base technical analysis
            market_data: Market data context
            
        Returns:
            Enhanced analysis with EMA metrics
        """
        try:
            ema_crossover = analysis.get('ema_crossover')
            volume_analysis = analysis.get('volume_analysis', {})
            
            if not ema_crossover:
                logger.warning("No EMA crossover data available")
                analysis['ema_context'] = {}
                return analysis
            
            # Calculate additional EMA metrics - handle both Pydantic models and dicts
            def get_value(obj, key, default=0):
                if hasattr(obj, key):
                    return getattr(obj, key)
                elif isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return default
            
            ema_9 = get_value(ema_crossover, 'ema_9', 0)
            ema_15 = get_value(ema_crossover, 'ema_15', 0)
            is_golden_cross = get_value(ema_crossover, 'is_golden_cross', False)
            crossover_strength = get_value(ema_crossover, 'crossover_strength', 0)
            
            current_price = market_data.current_price
            
            # Calculate EMA separation
            ema_separation = abs(ema_9 - ema_15)
            separation_pct = (ema_separation / current_price) * 100
            
            # Calculate price position relative to EMAs
            price_above_ema9 = current_price > ema_9
            price_above_ema15 = current_price > ema_15
            
            # Determine trend strength based on EMA alignment
            if price_above_ema9 and price_above_ema15 and is_golden_cross:
                trend_alignment = "strong_bullish"
            elif not price_above_ema9 and not price_above_ema15 and not is_golden_cross:
                trend_alignment = "strong_bearish"
            elif price_above_ema9 and price_above_ema15:
                trend_alignment = "bullish"
            elif not price_above_ema9 and not price_above_ema15:
                trend_alignment = "bearish"
            else:
                trend_alignment = "mixed"
            
            # Volume confirmation for crossover
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            volume_confirmed = volume_ratio > 1.2  # 20% above average
            
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(
                crossover_strength, separation_pct, volume_ratio, trend_alignment
            )
            
            # Add EMA specific context
            analysis['ema_context'] = {
                'ema_separation_pct': separation_pct,
                'price_above_ema9': price_above_ema9,
                'price_above_ema15': price_above_ema15,
                'trend_alignment': trend_alignment,
                'volume_confirmed': volume_confirmed,
                'momentum_score': momentum_score,
                'crossover_type': "golden_cross" if is_golden_cross else "death_cross",
                'signal_quality': self._assess_signal_quality(crossover_strength, volume_ratio, separation_pct)
            }
            
            logger.debug(
                "EMA analysis enhanced",
                separation_pct=separation_pct,
                trend_alignment=trend_alignment,
                momentum_score=momentum_score,
                volume_confirmed=volume_confirmed
            )
            
        except Exception as e:
            logger.warning("Failed to enhance EMA analysis", error=str(e))
            analysis['ema_context'] = {}
        
        return analysis

    def _calculate_momentum_score(
        self, 
        crossover_strength: int, 
        separation_pct: float, 
        volume_ratio: float, 
        trend_alignment: str
    ) -> int:
        """
        Calculate overall momentum score for EMA strategy.
        
        Args:
            crossover_strength: EMA crossover strength (1-10)
            separation_pct: EMA separation percentage
            volume_ratio: Volume ratio vs average
            trend_alignment: Trend alignment description
            
        Returns:
            Momentum score (1-10)
        """
        score = 0
        
        # Crossover strength contribution (40%)
        score += crossover_strength * 0.4
        
        # Separation contribution (25%)
        if separation_pct >= 2.0:
            score += 2.5
        elif separation_pct >= 1.0:
            score += 2.0
        elif separation_pct >= 0.5:
            score += 1.5
        else:
            score += 1.0
        
        # Volume contribution (20%)
        if volume_ratio >= 1.5:
            score += 2.0
        elif volume_ratio >= 1.2:
            score += 1.5
        elif volume_ratio >= 1.0:
            score += 1.0
        else:
            score += 0.5
        
        # Trend alignment contribution (15%)
        alignment_scores = {
            "strong_bullish": 1.5,
            "strong_bearish": 1.5,
            "bullish": 1.2,
            "bearish": 1.2,
            "mixed": 0.5
        }
        score += alignment_scores.get(trend_alignment, 0.5)
        
        return max(1, min(10, int(score)))

    def _assess_signal_quality(
        self, 
        crossover_strength: int, 
        volume_ratio: float, 
        separation_pct: float
    ) -> str:
        """
        Assess the overall quality of the EMA signal.
        
        Args:
            crossover_strength: EMA crossover strength
            volume_ratio: Volume ratio
            separation_pct: EMA separation percentage
            
        Returns:
            Signal quality assessment
        """
        if (crossover_strength >= 8 and volume_ratio >= 1.3 and separation_pct >= 1.5):
            return "excellent"
        elif (crossover_strength >= 6 and volume_ratio >= 1.2 and separation_pct >= 1.0):
            return "good"
        elif (crossover_strength >= 4 and volume_ratio >= 1.0 and separation_pct >= 0.5):
            return "fair"
        else:
            return "poor"

    def _add_ema_context(self, analysis_result: AnalysisResult, technical_analysis: dict) -> AnalysisResult:
        """
        Add EMA context to analysis result.
        
        Args:
            analysis_result: Original analysis result
            technical_analysis: Technical analysis with EMA data
            
        Returns:
            Enhanced analysis result
        """
        try:
            ema_context = technical_analysis.get('ema_context', {})
            ema_crossover = technical_analysis.get('ema_crossover')
            
            # Add EMA crossover data to the analysis result
            if ema_crossover:
                analysis_result.ema_crossover = ema_crossover
            
            # Enhance AI analysis text with EMA context
            if ema_context:
                context_text = self._generate_ema_context_text(ema_context, ema_crossover)
                analysis_result.ai_analysis += f"\n\nEMA CONTEXT:\n{context_text}"
            
            logger.debug("EMA context added to analysis result")
            
        except Exception as e:
            logger.warning("Failed to add EMA context", error=str(e))
        
        return analysis_result

    def _generate_ema_context_text(self, ema_context: dict, ema_crossover) -> str:
        """
        Generate human-readable EMA context text.
        
        Args:
            ema_context: EMA context data
            ema_crossover: EMA crossover data
            
        Returns:
            Formatted context text
        """
        lines = []
        
        if ema_crossover:
            # Helper function to safely get values from both Pydantic models and dicts
            def get_value(obj, key, default=0):
                if hasattr(obj, key):
                    return getattr(obj, key)
                elif isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return default
            
            ema_9 = get_value(ema_crossover, 'ema_9', 0)
            ema_15 = get_value(ema_crossover, 'ema_15', 0)
            crossover_strength = get_value(ema_crossover, 'crossover_strength', 0)
            
            lines.append(f"EMA Values: 9-EMA ${ema_9:,.2f}, 15-EMA ${ema_15:,.2f}")
            lines.append(f"Crossover: {ema_context.get('crossover_type', 'unknown').replace('_', ' ').title()}")
            lines.append(f"Crossover Strength: {crossover_strength}/10")
        
        lines.append(f"EMA Separation: {ema_context.get('ema_separation_pct', 0):.2f}%")
        lines.append(f"Trend Alignment: {ema_context.get('trend_alignment', 'unknown').replace('_', ' ').title()}")
        lines.append(f"Volume Confirmed: {'Yes' if ema_context.get('volume_confirmed') else 'No'}")
        lines.append(f"Momentum Score: {ema_context.get('momentum_score', 0)}/10")
        lines.append(f"Signal Quality: {ema_context.get('signal_quality', 'unknown').title()}")
        
        return "\n".join(lines)