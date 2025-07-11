"""
Support and Resistance trading strategy implementation.

Focuses on identifying key price levels where price historically bounces 
or gets rejected, providing clear entry and exit points.
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
from src.utils.data_helpers import get_value
from src.utils.type_conversion import to_float

logger = structlog.get_logger(__name__)


class SupportResistanceStrategy(BaseStrategy):
    """
    Support and Resistance trading strategy.
    
    Strategy Logic:
    - Identifies strong support and resistance levels
    - Looks for bounces off support levels (BUY signals)
    - Looks for rejections at resistance levels (SELL signals)
    - Requires volume confirmation for signal validation
    - Uses historical touch count to determine level strength
    
    Signal Generation:
    - BUY: Price bounces off strong support with volume
    - SELL: Price rejects at strong resistance with volume
    - NEUTRAL: Price in consolidation between levels
    - WAIT: Unclear level interaction or low volume
    """

    def __init__(self):
        """Initialize Support/Resistance strategy."""
        super().__init__()
        self.strategy_type = StrategyType.SUPPORT_RESISTANCE
        
        logger.info("Support/Resistance strategy initialized")

    def _get_minimum_periods(self) -> int:
        """Support/Resistance requires more data to identify reliable levels."""
        return 25  # Need more data to identify reliable S/R levels
    

    async def analyze(
        self,
        market_data: MarketData,
        session_config: SessionConfig
    ) -> AnalysisResult:
        """
        Analyze market data using Support/Resistance strategy.
        
        Args:
            market_data: Market data to analyze
            session_config: Session configuration
            
        Returns:
            AnalysisResult with Support/Resistance analysis
            
        Raises:
            DataValidationError: Invalid input data
            StrategyError: Analysis execution error
        """
        logger.info(
            "Starting Support/Resistance analysis",
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            current_price=market_data.current_price,
            data_points=len(market_data.ohlcv_data)
        )
        
        try:
            # Validate inputs
            self._validate_market_data(market_data)
            self._validate_session_config(session_config)
            
            # Calculate technical analysis with S/R focus
            technical_analysis = await self._calculate_sr_analysis(market_data)
            
            # Generate AI analysis with S/R context
            analysis_result = await self._generate_ai_analysis(
                market_data, technical_analysis, session_config
            )
            
            # Filter signals by confidence threshold
            filtered_result = self._filter_signals_by_confidence(
                analysis_result, session_config.confidence_threshold
            )
            
            # Add S/R specific context to the result
            filtered_result = self._add_sr_context(filtered_result, technical_analysis)
            
            logger.info(
                "Support/Resistance analysis completed",
                symbol=market_data.symbol,
                signals_count=len(filtered_result.signals),
                support_levels=len([l for l in technical_analysis.get('support_resistance', []) 
                                  if get_value(l, 'is_support', False)]),
                resistance_levels=len([l for l in technical_analysis.get('support_resistance', [])
                                     if not get_value(l, 'is_support', True)]),
                primary_signal=filtered_result.primary_signal.action.value if filtered_result.primary_signal else None
            )
            
            return filtered_result
            
        except Exception as e:
            logger.error(
                "Support/Resistance analysis failed",
                symbol=market_data.symbol,
                error=str(e)
            )
            raise StrategyError(f"Support/Resistance analysis failed: {str(e)}")

    async def _calculate_sr_analysis(self, market_data: MarketData) -> dict:
        """
        Calculate Support/Resistance specific technical analysis.
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Technical analysis with S/R emphasis
        """
        logger.debug(
            "Calculating S/R technical analysis",
            symbol=market_data.symbol,
            data_points=len(market_data.ohlcv_data)
        )
        
        # Get comprehensive technical analysis
        analysis = await self._calculate_technical_analysis(market_data)
        
        # Enhance with S/R specific calculations
        analysis = self._enhance_sr_analysis(analysis, market_data)
        
        return analysis

    def _enhance_sr_analysis(self, analysis: dict, market_data: MarketData) -> dict:
        """
        Enhance analysis with S/R specific metrics.
        
        Args:
            analysis: Base technical analysis
            market_data: Market data context
            
        Returns:
            Enhanced analysis with S/R metrics
        """
        try:
            current_price = market_data.current_price
            sr_levels = analysis.get('support_resistance', [])
            
            # Find nearest support and resistance levels
            nearest_support = None
            nearest_resistance = None
            
            for level in sr_levels:
                level_price = get_value(level, 'level', 0)
                is_support = get_value(level, 'is_support', False)
                
                if is_support and level_price < current_price:
                    if nearest_support is None or level_price > get_value(nearest_support, 'level', 0):
                        nearest_support = level
                elif not is_support and level_price > current_price:
                    if nearest_resistance is None or level_price < get_value(nearest_resistance, 'level', float('inf')):
                        nearest_resistance = level
            
            # Calculate distance to levels
            support_distance = None
            resistance_distance = None
            
            if nearest_support:
                support_price = to_float(get_value(nearest_support, 'level', 0))
                support_distance = ((current_price - support_price) / current_price) * 100
            
            if nearest_resistance:
                resistance_price = to_float(get_value(nearest_resistance, 'level', 0))
                resistance_distance = ((resistance_price - current_price) / current_price) * 100
            
            # Add S/R specific metrics
            analysis['sr_context'] = {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance_pct': support_distance,
                'resistance_distance_pct': resistance_distance,
                'total_support_levels': len([l for l in sr_levels if get_value(l, 'is_support', False)]),
                'total_resistance_levels': len([l for l in sr_levels if not get_value(l, 'is_support', True)]),
                'price_position': self._determine_price_position(current_price, sr_levels)
            }
            
            logger.debug(
                "S/R analysis enhanced",
                support_distance=support_distance,
                resistance_distance=resistance_distance,
                price_position=analysis['sr_context']['price_position']
            )
            
        except Exception as e:
            logger.warning("Failed to enhance S/R analysis", error=str(e))
            analysis['sr_context'] = {}
        
        return analysis

    def _determine_price_position(self, current_price: float, sr_levels: list) -> str:
        """
        Determine current price position relative to S/R levels.
        
        Args:
            current_price: Current market price
            sr_levels: List of support/resistance levels
            
        Returns:
            Price position description
        """
        if not sr_levels:
            return "no_levels_identified"
        
        support_levels = []
        resistance_levels = []
        
        for level in sr_levels:
            level_price = to_float(get_value(level, 'level', 0))
            is_support = get_value(level, 'is_support', False)
            
            if is_support:
                support_levels.append(level_price)
            else:
                resistance_levels.append(level_price)
        
        # Check if price is near a level (within configured tolerance)
        tolerance = current_price * (TradingConstants.SUPPORT_RESISTANCE_TOLERANCE_PCT / 100)
        
        for level_price in support_levels + resistance_levels:
            if abs(current_price - level_price) <= tolerance:
                level_type = "support" if level_price in support_levels else "resistance"
                return f"at_{level_type}_level"
        
        # Check if price is between levels
        nearby_support = max([l for l in support_levels if l < current_price], default=None)
        nearby_resistance = min([l for l in resistance_levels if l > current_price], default=None)
        
        if nearby_support and nearby_resistance:
            return "between_levels"
        elif nearby_support:
            return "above_support"
        elif nearby_resistance:
            return "below_resistance"
        else:
            return "outside_identified_levels"

    def _add_sr_context(self, analysis_result: AnalysisResult, technical_analysis: dict) -> AnalysisResult:
        """
        Add S/R context to analysis result.
        
        Args:
            analysis_result: Original analysis result
            technical_analysis: Technical analysis with S/R data
            
        Returns:
            Enhanced analysis result
        """
        try:
            sr_context = technical_analysis.get('sr_context', {})
            
            # Add S/R levels to the analysis result
            if 'support_resistance' in technical_analysis:
                analysis_result.support_resistance_levels = technical_analysis['support_resistance']
            
            # Enhance AI analysis text with S/R context
            if sr_context:
                context_text = self._generate_sr_context_text(sr_context)
                analysis_result.ai_analysis += f"\n\nS/R CONTEXT:\n{context_text}"
            
            logger.debug("S/R context added to analysis result")
            
        except Exception as e:
            logger.warning("Failed to add S/R context", error=str(e))
        
        return analysis_result

    def _generate_sr_context_text(self, sr_context: dict) -> str:
        """
        Generate human-readable S/R context text.
        
        Args:
            sr_context: S/R context data
            
        Returns:
            Formatted context text
        """
        lines = []
        
        if sr_context.get('nearest_support'):
            support = sr_context['nearest_support']
            support_price = to_float(get_value(support, 'level', 0))
            support_strength = get_value(support, 'strength', 0)
            distance = sr_context.get('support_distance_pct', 0)
            lines.append(f"Nearest Support: ${support_price:,.2f} (Strength: {support_strength}/10, Distance: {distance:.1f}%)")
        
        if sr_context.get('nearest_resistance'):
            resistance = sr_context['nearest_resistance']
            resistance_price = to_float(get_value(resistance, 'level', 0))
            resistance_strength = get_value(resistance, 'strength', 0)
            distance = sr_context.get('resistance_distance_pct', 0)
            lines.append(f"Nearest Resistance: ${resistance_price:,.2f} (Strength: {resistance_strength}/10, Distance: {distance:.1f}%)")
        
        lines.append(f"Price Position: {sr_context.get('price_position', 'unknown').replace('_', ' ').title()}")
        lines.append(f"Total Levels: {sr_context.get('total_support_levels', 0)} Support, {sr_context.get('total_resistance_levels', 0)} Resistance")
        
        return "\n".join(lines)