"""
Combined trading strategy implementation.

Merges both EMA Crossover and Support/Resistance strategies for maximum 
confidence signals by requiring confirmation from both approaches.
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


class CombinedStrategy(BaseStrategy):
    """
    Combined EMA Crossover and Support/Resistance strategy.
    
    Strategy Logic:
    - Requires confirmation from BOTH EMA and S/R strategies
    - Only generates BUY/SELL signals when both strategies align
    - Uses NEUTRAL when strategies provide conflicting signals
    - Demands higher confidence thresholds for signal generation
    - Provides maximum confidence signals for major positions
    
    Signal Generation:
    - BUY: Golden cross + support breakout/bounce with volume
    - SELL: Death cross + resistance rejection/break with volume  
    - NEUTRAL: Strategies conflict or insufficient confirmation
    - WAIT: Neither strategy provides clear signals
    
    Confidence Weighting:
    - Both strategies bullish + volume: 8-10 confidence
    - Both strategies bearish + volume: 8-10 confidence
    - Single strategy + volume: 5-7 confidence
    - Conflicting strategies: 1-4 confidence
    """

    def __init__(self):
        """Initialize Combined strategy."""
        super().__init__()
        self.strategy_type = StrategyType.COMBINED
        
        logger.info("Combined strategy initialized")

    def _get_minimum_periods(self) -> int:
        """Combined strategy requires the most data for reliable analysis."""
        return 30  # Need sufficient data for both EMA and S/R analysis

    async def analyze(
        self,
        market_data: MarketData,
        session_config: SessionConfig
    ) -> AnalysisResult:
        """
        Analyze market data using Combined strategy.
        
        Args:
            market_data: Market data to analyze
            session_config: Session configuration
            
        Returns:
            AnalysisResult with Combined strategy analysis
            
        Raises:
            DataValidationError: Invalid input data
            StrategyError: Analysis execution error
        """
        logger.info(
            "Starting Combined strategy analysis",
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            current_price=market_data.current_price,
            data_points=len(market_data.ohlcv_data)
        )
        
        try:
            # Validate inputs
            self._validate_market_data(market_data)
            self._validate_session_config(session_config)
            
            # Calculate comprehensive technical analysis
            technical_analysis = await self._calculate_combined_analysis(market_data)
            
            # Generate AI analysis with combined context
            analysis_result = await self._generate_ai_analysis(
                market_data, technical_analysis, session_config
            )
            
            # Apply combined strategy filtering (higher standards)
            filtered_result = self._apply_combined_filtering(
                analysis_result, technical_analysis, session_config
            )
            
            # Add combined strategy context
            filtered_result = self._add_combined_context(filtered_result, technical_analysis)
            
            logger.info(
                "Combined strategy analysis completed",
                symbol=market_data.symbol,
                signals_count=len(filtered_result.signals),
                strategy_alignment=technical_analysis.get('combined_context', {}).get('strategy_alignment'),
                confirmation_level=technical_analysis.get('combined_context', {}).get('confirmation_level'),
                primary_signal=filtered_result.primary_signal.action.value if filtered_result.primary_signal else None
            )
            
            return filtered_result
            
        except Exception as e:
            logger.error(
                "Combined strategy analysis failed",
                symbol=market_data.symbol,
                error=str(e)
            )
            raise StrategyError(f"Combined strategy analysis failed: {str(e)}")

    async def _calculate_combined_analysis(self, market_data: MarketData) -> dict:
        """
        Calculate comprehensive analysis for both EMA and S/R strategies.
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Technical analysis with both EMA and S/R data
        """
        logger.debug(
            "Calculating combined technical analysis",
            symbol=market_data.symbol,
            data_points=len(market_data.ohlcv_data)
        )
        
        # Get comprehensive technical analysis
        analysis = await self._calculate_technical_analysis(market_data)
        
        # Enhance with combined strategy analysis
        analysis = self._enhance_combined_analysis(analysis, market_data)
        
        return analysis

    def _enhance_combined_analysis(self, analysis: dict, market_data: MarketData) -> dict:
        """
        Enhance analysis with combined strategy metrics.
        
        Args:
            analysis: Base technical analysis
            market_data: Market data context
            
        Returns:
            Enhanced analysis with combined metrics
        """
        try:
            ema_crossover = analysis.get('ema_crossover')
            sr_levels = analysis.get('support_resistance', [])
            volume_analysis = analysis.get('volume_analysis', {})
            price_action = analysis.get('price_action', {})
            
            # Analyze EMA signals
            ema_signal = self._analyze_ema_signal(ema_crossover, volume_analysis)
            
            # Analyze S/R signals
            sr_signal = self._analyze_sr_signal(sr_levels, market_data.current_price, volume_analysis)
            
            # Determine strategy alignment
            strategy_alignment = self._determine_strategy_alignment(ema_signal, sr_signal)
            
            # Calculate confirmation level
            confirmation_level = self._calculate_confirmation_level(
                ema_signal, sr_signal, volume_analysis, price_action
            )
            
            # Generate combined signal assessment
            combined_signal = self._generate_combined_signal(ema_signal, sr_signal, confirmation_level)
            
            # Add combined context
            analysis['combined_context'] = {
                'ema_signal': ema_signal,
                'sr_signal': sr_signal,
                'strategy_alignment': strategy_alignment,
                'confirmation_level': confirmation_level,
                'combined_signal': combined_signal,
                'volume_strength': self._assess_volume_strength(volume_analysis),
                'overall_conviction': self._calculate_overall_conviction(ema_signal, sr_signal, volume_analysis)
            }
            
            logger.debug(
                "Combined analysis enhanced",
                ema_signal=ema_signal.get('direction'),
                sr_signal=sr_signal.get('direction'),
                alignment=strategy_alignment,
                confirmation=confirmation_level
            )
            
        except Exception as e:
            logger.warning("Failed to enhance combined analysis", error=str(e))
            analysis['combined_context'] = {}
        
        return analysis

    
    def _analyze_ema_signal(self, ema_crossover, volume_analysis: dict) -> dict:
        """Analyze EMA crossover signal strength and direction."""
        if not ema_crossover:
            return {'direction': 'neutral', 'strength': 0, 'confidence': 1}
        
        is_golden_cross = get_value(ema_crossover, 'is_golden_cross', False)
        crossover_strength = get_value(ema_crossover, 'crossover_strength', 0)
        volume_ratio = volume_analysis.get('volume_ratio', 1.0)
        
        # Determine direction and adjust strength based on volume
        direction = 'bullish' if is_golden_cross else 'bearish'
        adjusted_strength = crossover_strength
        
        # Volume confirmation bonus
        if volume_ratio > TradingConstants.STRONG_VOLUME_THRESHOLD:
            adjusted_strength = min(10, adjusted_strength + TradingConstants.VOLUME_CONFIDENCE_BONUS)
        elif volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD:
            adjusted_strength = min(10, adjusted_strength + 1)
        
        # Calculate confidence
        confidence = max(1, min(10, adjusted_strength))
        
        return {
            'direction': direction,
            'strength': adjusted_strength,
            'confidence': confidence,
            'volume_confirmed': volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD
        }

    def _analyze_sr_signal(self, sr_levels: list, current_price: float, volume_analysis: dict) -> dict:
        """Analyze Support/Resistance signal strength and direction."""
        if not sr_levels:
            return {'direction': 'neutral', 'strength': 0, 'confidence': 1}
        
        # Find nearest levels
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
        
        # Determine signal based on proximity to levels
        volume_ratio = volume_analysis.get('volume_ratio', 1.0)
        tolerance = current_price * (TradingConstants.PRICE_TOLERANCE_PCT / 100)
        
        direction = 'neutral'
        strength = 0
        confidence = 1
        
        if nearest_support:
            support_price = get_value(nearest_support, 'level', 0)
            support_strength = get_value(nearest_support, 'strength', 0)
            
            if abs(current_price - support_price) <= tolerance:
                # Near support - potential bounce
                direction = 'bullish'
                strength = support_strength
                confidence = min(10, strength + (TradingConstants.VOLUME_CONFIDENCE_BONUS if volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD else 0))
        
        if nearest_resistance:
            resistance_price = get_value(nearest_resistance, 'level', 0)
            resistance_strength = get_value(nearest_resistance, 'strength', 0)
            
            if abs(current_price - resistance_price) <= tolerance:
                # Near resistance - potential rejection
                if direction == 'neutral':
                    direction = 'bearish'
                    strength = resistance_strength
                    confidence = min(10, strength + (TradingConstants.VOLUME_CONFIDENCE_BONUS if volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD else 0))
                else:
                    # Conflicting signals
                    direction = 'neutral'
                    strength = min(strength, resistance_strength)
                    confidence = max(1, confidence - TradingConstants.VOLUME_CONFIDENCE_BONUS)
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }

    def _determine_strategy_alignment(self, ema_signal: dict, sr_signal: dict) -> str:
        """Determine how well the two strategies align."""
        ema_dir = ema_signal.get('direction', 'neutral')
        sr_dir = sr_signal.get('direction', 'neutral')
        
        if ema_dir == sr_dir and ema_dir != 'neutral':
            return 'strong_alignment'
        elif ema_dir == 'neutral' or sr_dir == 'neutral':
            return 'partial_alignment'
        elif ema_dir != sr_dir:
            return 'conflicting'
        else:
            return 'no_alignment'

    def _calculate_confirmation_level(
        self, 
        ema_signal: dict, 
        sr_signal: dict, 
        volume_analysis: dict, 
        price_action: dict
    ) -> int:
        """Calculate overall confirmation level (1-10)."""
        confirmation = 0
        
        # Strategy alignment contribution (40%)
        alignment = self._determine_strategy_alignment(ema_signal, sr_signal)
        if alignment == 'strong_alignment':
            confirmation += 10 * TradingConstants.STRATEGY_ALIGNMENT_WEIGHT
        elif alignment == 'partial_alignment':
            confirmation += 5 * TradingConstants.STRATEGY_ALIGNMENT_WEIGHT
        elif alignment == 'conflicting':
            confirmation += 1 * TradingConstants.STRATEGY_ALIGNMENT_WEIGHT
        
        # Signal strength contribution (30%)
        avg_strength = (ema_signal.get('strength', 0) + sr_signal.get('strength', 0)) / 2
        confirmation += (avg_strength / 10) * (10 * TradingConstants.SIGNAL_STRENGTH_WEIGHT)
        
        # Volume confirmation (20%)
        volume_ratio = volume_analysis.get('volume_ratio', 1.0)
        if volume_ratio > TradingConstants.VERY_STRONG_VOLUME_THRESHOLD:
            confirmation += 10 * TradingConstants.VOLUME_CONFIRMATION_WEIGHT
        elif volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD:
            confirmation += 7 * TradingConstants.VOLUME_CONFIRMATION_WEIGHT
        elif volume_ratio > 1.0:
            confirmation += 5 * TradingConstants.VOLUME_CONFIRMATION_WEIGHT
        else:
            confirmation += 2 * TradingConstants.VOLUME_CONFIRMATION_WEIGHT
        
        # Price action consistency (10%)
        trend_direction = price_action.get('trend_direction', 'neutral')
        ema_dir = ema_signal.get('direction', 'neutral')
        
        if (trend_direction == 'bullish' and ema_dir == 'bullish') or \
           (trend_direction == 'bearish' and ema_dir == 'bearish'):
            confirmation += 10 * TradingConstants.PRICE_ACTION_WEIGHT
        elif trend_direction != 'sideways':
            confirmation += 5 * TradingConstants.PRICE_ACTION_WEIGHT
        
        return max(1, min(10, int(confirmation)))

    def _generate_combined_signal(self, ema_signal: dict, sr_signal: dict, confirmation_level: int) -> dict:
        """Generate the overall combined signal assessment."""
        ema_dir = ema_signal.get('direction', 'neutral')
        sr_dir = sr_signal.get('direction', 'neutral')
        
        # Determine final direction
        if ema_dir == sr_dir and ema_dir != 'neutral':
            final_direction = ema_dir
            confidence_bonus = TradingConstants.ALIGNMENT_CONFIDENCE_BONUS
        elif ema_dir != 'neutral' and sr_dir == 'neutral':
            final_direction = ema_dir
            confidence_bonus = 0
        elif sr_dir != 'neutral' and ema_dir == 'neutral':
            final_direction = sr_dir
            confidence_bonus = 0
        else:
            final_direction = 'neutral'
            confidence_bonus = -1
        
        # Calculate final confidence
        base_confidence = (ema_signal.get('confidence', 1) + sr_signal.get('confidence', 1)) / 2
        final_confidence = max(1, min(10, int(base_confidence + confidence_bonus)))
        
        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'strength': 'strong' if confirmation_level >= TradingConstants.STRONG_CONFIRMATION_LEVEL else 'moderate' if confirmation_level >= TradingConstants.MODERATE_CONFIRMATION_LEVEL else 'weak'
        }

    def _assess_volume_strength(self, volume_analysis: dict) -> str:
        """Assess volume strength for confirmation."""
        volume_ratio = volume_analysis.get('volume_ratio', 1.0)
        volume_trend = volume_analysis.get('volume_trend', 'stable')
        
        if volume_ratio > TradingConstants.VERY_STRONG_VOLUME_THRESHOLD and volume_trend == 'increasing':
            return 'very_strong'
        elif volume_ratio > TradingConstants.STRONG_VOLUME_THRESHOLD:
            return 'strong'
        elif volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD:
            return 'moderate'
        elif volume_ratio > 0.9:
            return 'weak'
        else:
            return 'very_weak'

    def _calculate_overall_conviction(self, ema_signal: dict, sr_signal: dict, volume_analysis: dict) -> int:
        """Calculate overall conviction level for the combined strategy."""
        ema_conf = ema_signal.get('confidence', 1)
        sr_conf = sr_signal.get('confidence', 1)
        volume_ratio = volume_analysis.get('volume_ratio', 1.0)
        
        # Base conviction from signal confidences
        base_conviction = (ema_conf + sr_conf) / 2
        
        # Volume modifier
        if volume_ratio > TradingConstants.STRONG_VOLUME_THRESHOLD:
            volume_modifier = 1.2
        elif volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD:
            volume_modifier = 1.1
        else:
            volume_modifier = 0.9
        
        # Alignment modifier
        alignment = self._determine_strategy_alignment(ema_signal, sr_signal)
        if alignment == 'strong_alignment':
            alignment_modifier = 1.3
        elif alignment == 'partial_alignment':
            alignment_modifier = 1.0
        else:
            alignment_modifier = 0.7
        
        conviction = base_conviction * volume_modifier * alignment_modifier
        return max(1, min(10, int(conviction)))

    def _apply_combined_filtering(
        self, 
        analysis_result: AnalysisResult, 
        technical_analysis: dict, 
        session_config: SessionConfig
    ) -> AnalysisResult:
        """Apply combined strategy specific filtering with higher standards."""
        combined_context = technical_analysis.get('combined_context', {})
        confirmation_level = combined_context.get('confirmation_level', 1)
        
        # Higher confidence threshold for combined strategy
        min_confidence = max(session_config.confidence_threshold, TradingConstants.MODERATE_CONFIRMATION_LEVEL)
        
        # Filter signals based on confirmation level and confidence
        filtered_signals = []
        for signal in analysis_result.signals:
            if (signal.confidence >= min_confidence and 
                confirmation_level >= TradingConstants.MODERATE_CONFIRMATION_LEVEL):  # Require decent confirmation
                filtered_signals.append(signal)
        
        analysis_result.signals = filtered_signals
        
        logger.debug(
            "Combined strategy filtering applied",
            original_count=len(analysis_result.signals),
            filtered_count=len(filtered_signals),
            min_confidence=min_confidence,
            confirmation_level=confirmation_level
        )
        
        return analysis_result

    def _add_combined_context(self, analysis_result: AnalysisResult, technical_analysis: dict) -> AnalysisResult:
        """Add combined strategy context to analysis result."""
        try:
            combined_context = technical_analysis.get('combined_context', {})
            
            if combined_context:
                context_text = self._generate_combined_context_text(combined_context)
                analysis_result.ai_analysis += f"\n\nCOMBINED STRATEGY CONTEXT:\n{context_text}"
            
            logger.debug("Combined context added to analysis result")
            
        except Exception as e:
            logger.warning("Failed to add combined context", error=str(e))
        
        return analysis_result

    def _generate_combined_context_text(self, combined_context: dict) -> str:
        """Generate human-readable combined strategy context text."""
        lines = []
        
        ema_signal = combined_context.get('ema_signal', {})
        sr_signal = combined_context.get('sr_signal', {})
        
        lines.append(f"EMA Signal: {ema_signal.get('direction', 'neutral').title()} (Confidence: {ema_signal.get('confidence', 0)}/10)")
        lines.append(f"S/R Signal: {sr_signal.get('direction', 'neutral').title()} (Confidence: {sr_signal.get('confidence', 0)}/10)")
        lines.append(f"Strategy Alignment: {combined_context.get('strategy_alignment', 'unknown').replace('_', ' ').title()}")
        lines.append(f"Confirmation Level: {combined_context.get('confirmation_level', 0)}/10")
        lines.append(f"Volume Strength: {combined_context.get('volume_strength', 'unknown').replace('_', ' ').title()}")
        lines.append(f"Overall Conviction: {combined_context.get('overall_conviction', 0)}/10")
        
        combined_signal = combined_context.get('combined_signal', {})
        lines.append(f"Combined Signal: {combined_signal.get('direction', 'neutral').title()} ({combined_signal.get('strength', 'weak').title()})")
        
        return "\n".join(lines)