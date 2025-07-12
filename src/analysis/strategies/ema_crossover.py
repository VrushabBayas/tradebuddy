"""
EMA Crossover trading strategy implementation.

Uses exponential moving average crossovers to identify trend changes
and generate trading signals with volume confirmation.
"""

import structlog
from typing import Any, Dict

from src.analysis.strategies.base_strategy import BaseStrategy
from src.core.constants import TradingConstants
from src.core.exceptions import StrategyError
from src.core.models import AnalysisResult, MarketData, SessionConfig, StrategyType
from src.utils.helpers import get_value, to_float

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

        # Import here to avoid circular imports
        from src.analysis.indicators import TechnicalIndicators

        self.technical_indicators = TechnicalIndicators()

        logger.info("EMA Crossover strategy initialized")

    def _get_minimum_periods(self) -> int:
        """EMA Crossover requires enough data for reliable EMA calculation."""
        return TradingConstants.EMA_LONG_PERIOD + 5  # 20 periods minimum

    async def analyze(
        self, market_data: MarketData, session_config: SessionConfig
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
            data_points=len(market_data.ohlcv_data),
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
            ema_crossover = technical_analysis.get("ema_crossover")
            crossover_type = "unknown"
            crossover_strength = 0

            if ema_crossover:
                if hasattr(ema_crossover, "is_golden_cross"):
                    crossover_type = (
                        "golden_cross"
                        if ema_crossover.is_golden_cross
                        else "death_cross"
                    )
                    crossover_strength = getattr(ema_crossover, "crossover_strength", 0)
                elif isinstance(ema_crossover, dict):
                    crossover_type = (
                        "golden_cross"
                        if ema_crossover.get("is_golden_cross")
                        else "death_cross"
                    )
                    crossover_strength = ema_crossover.get("crossover_strength", 0)

            logger.info(
                "EMA Crossover analysis completed",
                symbol=market_data.symbol,
                signals_count=len(filtered_result.signals),
                crossover_type=crossover_type,
                crossover_strength=crossover_strength,
                primary_signal=str(filtered_result.primary_signal.action)
                if filtered_result.primary_signal
                else None,
            )

            return filtered_result

        except Exception as e:
            logger.error(
                "EMA Crossover analysis failed", symbol=market_data.symbol, error=str(e)
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
            data_points=len(market_data.ohlcv_data),
        )

        # Get comprehensive technical analysis
        analysis = await self._calculate_technical_analysis(market_data)

        # Add RSI calculation for enhanced strategy
        rsi_values = self.technical_indicators.calculate_rsi(market_data.ohlcv_data, 14)
        analysis["rsi_values"] = rsi_values

        # Enhance with EMA specific calculations
        analysis = self._enhance_ema_analysis(analysis, market_data)

        return analysis

    def _enhance_ema_analysis(self, analysis: dict, market_data: MarketData) -> dict:
        """
        Enhance analysis with EMA specific metrics and optional strategy filters.

        Args:
            analysis: Base technical analysis
            market_data: Market data context

        Returns:
            Enhanced analysis with EMA metrics and enhanced filters
        """
        try:
            ema_crossover = analysis.get("ema_crossover")
            volume_analysis = analysis.get("volume_analysis", {})

            if not ema_crossover:
                logger.warning("No EMA crossover data available")
                analysis["ema_context"] = {}
                return analysis

            # Calculate additional EMA metrics - handle both Pydantic models and dicts
            ema_9 = to_float(get_value(ema_crossover, "ema_9", 0))
            ema_15 = to_float(get_value(ema_crossover, "ema_15", 0))
            is_golden_cross = get_value(ema_crossover, "is_golden_cross", False)
            crossover_strength = get_value(ema_crossover, "crossover_strength", 0)

            current_price = market_data.current_price

            # Calculate EMA separation
            ema_separation = abs(ema_9 - ema_15)
            separation_pct = (ema_separation / current_price) * 100

            # Calculate price position relative to EMAs
            price_above_ema9 = current_price > ema_9
            price_above_ema15 = current_price > ema_15

            # Enhanced Strategy Filters - Calculate 50 EMA for trend filter
            ema_50_values = self.technical_indicators.calculate_ema(
                market_data.ohlcv_data, 50
            )
            ema_50 = ema_50_values[-1] if ema_50_values else current_price
            price_above_ema50 = current_price > ema_50

            # RSI Filter
            rsi_values = analysis.get("rsi_values", [])
            current_rsi = rsi_values[-1] if rsi_values else 50
            rsi_bullish = current_rsi > 50
            rsi_bearish = current_rsi < 50

            # Enhanced Volume Confirmation (110% of 20-period SMA)
            volume_confirmation_110pct = volume_analysis.get(
                "volume_confirmation_110pct", False
            )

            # Calculate ATR for dynamic stop losses
            atr_values = self.technical_indicators.calculate_atr(
                market_data.ohlcv_data, 14
            )
            current_atr = atr_values[-1] if atr_values else 0

            # Candlestick Confirmation
            candlestick_analysis = (
                self.technical_indicators.analyze_candlestick_confirmation(
                    market_data.ohlcv_data
                )
            )
            candlestick_confirmation = candlestick_analysis.get(
                "confirmation", "neutral"
            )
            candlestick_strength = candlestick_analysis.get("strength", "weak")

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

            # Enhanced Signal Validation
            long_signal_valid = (
                is_golden_cross
                and price_above_ema9
                and price_above_ema15
                and current_price > ema_9
                and current_price > ema_15  # Close above both EMAs
            )

            short_signal_valid = (
                not is_golden_cross
                and not price_above_ema9
                and not price_above_ema15
                and current_price < ema_9
                and current_price < ema_15  # Close below both EMAs
            )

            # Optional Filters (configurable) - check session config for EMA settings
            ema_config = getattr(market_data, "session_config", None)
            if hasattr(ema_config, "ema_config") and ema_config.ema_config:
                config = ema_config.ema_config
                enable_rsi = config.enable_rsi_filter
                enable_ema50 = config.enable_ema50_filter
                enable_volume = config.enable_volume_filter
                enable_candlestick = config.enable_candlestick_filter
            else:
                # Default configuration (as per strategy document)
                enable_rsi = True
                enable_ema50 = False  # As requested - optional by default
                enable_volume = True
                enable_candlestick = True

            enhanced_long_filters = {
                "rsi_filter": rsi_bullish if enable_rsi else True,  # RSI > 50
                "ema50_filter": price_above_ema50
                if enable_ema50
                else True,  # Price > 50 EMA
                "volume_filter": volume_confirmation_110pct
                if enable_volume
                else True,  # Volume >= 110% of 20-period SMA
                "candlestick_filter": (candlestick_confirmation == "bullish")
                if enable_candlestick
                else True,
            }

            enhanced_short_filters = {
                "rsi_filter": rsi_bearish if enable_rsi else True,  # RSI < 50
                "ema50_filter": (not price_above_ema50)
                if enable_ema50
                else True,  # Price < 50 EMA
                "volume_filter": volume_confirmation_110pct
                if enable_volume
                else True,  # Volume >= 110% of 20-period SMA
                "candlestick_filter": (candlestick_confirmation == "bearish")
                if enable_candlestick
                else True,
            }

            # Volume confirmation for crossover (legacy)
            volume_ratio = volume_analysis.get("volume_ratio", 1.0)
            volume_confirmed = (
                volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD
            )

            # Calculate enhanced momentum score
            momentum_score = self._calculate_enhanced_momentum_score(
                crossover_strength,
                separation_pct,
                volume_ratio,
                trend_alignment,
                current_rsi,
                candlestick_strength,
                current_atr,
            )

            # Add EMA specific context with enhanced filters
            analysis["ema_context"] = {
                "ema_separation_pct": separation_pct,
                "price_above_ema9": price_above_ema9,
                "price_above_ema15": price_above_ema15,
                "price_above_ema50": price_above_ema50,
                "ema_50": ema_50,
                "trend_alignment": trend_alignment,
                "volume_confirmed": volume_confirmed,
                "volume_confirmation_110pct": volume_confirmation_110pct,
                "momentum_score": momentum_score,
                "crossover_type": "golden_cross" if is_golden_cross else "death_cross",
                "signal_quality": self._assess_signal_quality(
                    crossover_strength, volume_ratio, separation_pct
                ),
                "current_rsi": current_rsi,
                "rsi_bullish": rsi_bullish,
                "rsi_bearish": rsi_bearish,
                "current_atr": current_atr,
                "candlestick_confirmation": candlestick_confirmation,
                "candlestick_strength": candlestick_strength,
                "long_signal_valid": long_signal_valid,
                "short_signal_valid": short_signal_valid,
                "enhanced_long_filters": enhanced_long_filters,
                "enhanced_short_filters": enhanced_short_filters,
                "all_long_filters_passed": all(enhanced_long_filters.values()),
                "all_short_filters_passed": all(enhanced_short_filters.values()),
                "filter_config": {
                    "rsi_enabled": enable_rsi,
                    "ema50_enabled": enable_ema50,
                    "volume_enabled": enable_volume,
                    "candlestick_enabled": enable_candlestick,
                },
            }

            # Store additional data for AI analysis
            analysis["rsi_values"] = rsi_values
            analysis["atr_values"] = atr_values
            analysis["candlestick_analysis"] = candlestick_analysis

            logger.debug(
                "Enhanced EMA analysis completed",
                separation_pct=separation_pct,
                trend_alignment=trend_alignment,
                momentum_score=momentum_score,
                current_rsi=current_rsi,
                current_atr=current_atr,
                candlestick_confirmation=candlestick_confirmation,
                all_long_filters_passed=all(enhanced_long_filters.values()),
                all_short_filters_passed=all(enhanced_short_filters.values()),
            )

        except Exception as e:
            logger.warning("Failed to enhance EMA analysis", error=str(e))
            analysis["ema_context"] = {}

        return analysis

    def _calculate_momentum_score(
        self,
        crossover_strength: int,
        separation_pct: float,
        volume_ratio: float,
        trend_alignment: str,
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
        if separation_pct >= TradingConstants.EMA_SEPARATION_STRONG:
            score += 2.5
        elif separation_pct >= TradingConstants.EMA_SEPARATION_MODERATE:
            score += 2.0
        elif separation_pct >= TradingConstants.EMA_SEPARATION_WEAK:
            score += 1.5
        else:
            score += 1.0

        # Volume contribution (20%)
        if volume_ratio >= TradingConstants.VERY_STRONG_VOLUME_THRESHOLD:
            score += 2.0
        elif volume_ratio >= TradingConstants.VOLUME_CONFIRMATION_THRESHOLD:
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
            "mixed": 0.5,
        }
        score += alignment_scores.get(trend_alignment, 0.5)

        return max(1, min(10, int(score)))

    def _calculate_enhanced_momentum_score(
        self,
        crossover_strength: int,
        separation_pct: float,
        volume_ratio: float,
        trend_alignment: str,
        current_rsi: float,
        candlestick_strength: str,
        current_atr: float,
    ) -> int:
        """
        Calculate enhanced momentum score including RSI, candlestick, and ATR factors.

        Args:
            crossover_strength: EMA crossover strength (1-10)
            separation_pct: EMA separation percentage
            volume_ratio: Volume ratio vs average
            trend_alignment: Trend alignment description
            current_rsi: Current RSI value
            candlestick_strength: Candlestick strength ('weak', 'moderate', 'strong')
            current_atr: Current ATR value

        Returns:
            Enhanced momentum score (1-10)
        """
        score = 0

        # Base momentum score (60% weight)
        base_score = self._calculate_momentum_score(
            crossover_strength, separation_pct, volume_ratio, trend_alignment
        )
        score += base_score * 0.6

        # RSI contribution (15% weight)
        rsi_score = 0
        if 30 <= current_rsi <= 70:  # Neutral RSI range
            rsi_score = 5
        elif current_rsi > 70:  # Overbought
            rsi_score = 3
        elif current_rsi < 30:  # Oversold
            rsi_score = 3
        elif current_rsi > 50:  # Bullish
            rsi_score = 7
        else:  # Bearish
            rsi_score = 7
        score += (rsi_score / 10) * 1.5

        # Candlestick confirmation (15% weight)
        candlestick_scores = {"strong": 1.5, "moderate": 1.0, "weak": 0.5}
        score += candlestick_scores.get(candlestick_strength, 0.5)

        # Volatility factor (10% weight) - ATR relative to price
        # Higher ATR can indicate stronger moves
        if current_atr > 0:
            atr_factor = min(1.0, current_atr / 1000)  # Normalize ATR
            score += atr_factor * 1.0
        else:
            score += 0.5

        return max(1, min(10, int(score)))

    def _assess_signal_quality(
        self, crossover_strength: int, volume_ratio: float, separation_pct: float
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
        if crossover_strength >= 8 and volume_ratio >= 1.3 and separation_pct >= 1.5:
            return "excellent"
        elif crossover_strength >= 6 and volume_ratio >= 1.2 and separation_pct >= 1.0:
            return "good"
        elif crossover_strength >= 4 and volume_ratio >= 1.0 and separation_pct >= 0.5:
            return "fair"
        else:
            return "poor"

    def _add_ema_context(
        self, analysis_result: AnalysisResult, technical_analysis: dict
    ) -> AnalysisResult:
        """
        Add EMA context to analysis result.

        Args:
            analysis_result: Original analysis result
            technical_analysis: Technical analysis with EMA data

        Returns:
            Enhanced analysis result
        """
        try:
            ema_context = technical_analysis.get("ema_context", {})
            ema_crossover = technical_analysis.get("ema_crossover")

            # Add EMA crossover data to the analysis result
            if ema_crossover:
                analysis_result.ema_crossover = ema_crossover

            # Enhance AI analysis text with EMA context
            if ema_context:
                context_text = self._generate_ema_context_text(
                    ema_context, ema_crossover
                )
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
            # Use the imported get_value function for consistency
            ema_9 = to_float(get_value(ema_crossover, "ema_9", 0))
            ema_15 = to_float(get_value(ema_crossover, "ema_15", 0))
            crossover_strength = get_value(ema_crossover, "crossover_strength", 0)

            lines.append(f"EMA Values: 9-EMA ${ema_9:,.2f}, 15-EMA ${ema_15:,.2f}")
            lines.append(
                f"Crossover: {ema_context.get('crossover_type', 'unknown').replace('_', ' ').title()}"
            )
            lines.append(f"Crossover Strength: {crossover_strength}/10")

        lines.append(f"EMA Separation: {ema_context.get('ema_separation_pct', 0):.2f}%")
        lines.append(
            f"50 EMA: ${ema_context.get('ema_50', 0):,.2f} (Price {'Above' if ema_context.get('price_above_ema50') else 'Below'})"
        )
        lines.append(
            f"Trend Alignment: {ema_context.get('trend_alignment', 'unknown').replace('_', ' ').title()}"
        )
        lines.append(
            f"RSI(14): {ema_context.get('current_rsi', 50):.1f} ({'Bullish' if ema_context.get('rsi_bullish') else 'Bearish'})"
        )
        lines.append(f"ATR(14): ${ema_context.get('current_atr', 0):.2f}")
        lines.append(
            f"Volume Confirmed: {'Yes' if ema_context.get('volume_confirmed') else 'No'}"
        )
        lines.append(
            f"Enhanced Volume (110%): {'Yes' if ema_context.get('volume_confirmation_110pct') else 'No'}"
        )
        lines.append(
            f"Candlestick: {ema_context.get('candlestick_confirmation', 'neutral').title()} ({ema_context.get('candlestick_strength', 'weak').title()})"
        )
        lines.append(
            f"Enhanced Momentum Score: {ema_context.get('momentum_score', 0)}/10"
        )
        lines.append(
            f"Signal Quality: {ema_context.get('signal_quality', 'unknown').title()}"
        )

        # Add filter status
        if ema_context.get("enhanced_long_filters"):
            long_filters = ema_context["enhanced_long_filters"]
            lines.append(
                f"Long Filters: RSI({long_filters.get('rsi_filter', False)}), EMA50({long_filters.get('ema50_filter', False)}), Volume({long_filters.get('volume_filter', False)}), Candle({long_filters.get('candlestick_filter', False)})"
            )

        if ema_context.get("enhanced_short_filters"):
            short_filters = ema_context["enhanced_short_filters"]
            lines.append(
                f"Short Filters: RSI({short_filters.get('rsi_filter', False)}), EMA50({short_filters.get('ema50_filter', False)}), Volume({short_filters.get('volume_filter', False)}), Candle({short_filters.get('candlestick_filter', False)})"
            )

        # ATR-based stop loss suggestion
        current_atr = ema_context.get("current_atr", 0)
        if current_atr > 0:
            atr_multiplier = 1.5  # As per strategy document
            lines.append(
                f"Suggested ATR Stop Loss: ±${current_atr * atr_multiplier:.2f} (1.5x ATR)"
            )

        return "\n".join(lines)

    async def _generate_backtesting_signals(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        session_config: SessionConfig,
    ) -> AnalysisResult:
        """
        Generate EMA Crossover signals for backtesting without AI analysis.

        Args:
            market_data: Market data context
            technical_analysis: Technical analysis results
            session_config: Session configuration

        Returns:
            AnalysisResult with EMA crossover signals
        """
        from src.core.models import AnalysisResult, TradingSignal, SignalAction, SignalStrength

        logger.debug(
            "Generating EMA crossover backtesting signals",
            symbol=market_data.symbol,
            current_price=market_data.current_price,
        )

        # Enhance technical analysis with EMA specific data
        enhanced_analysis = self._enhance_ema_analysis(technical_analysis, market_data)
        
        ema_crossover = enhanced_analysis.get("ema_crossover")
        ema_context = enhanced_analysis.get("ema_context", {})
        
        signals = []
        
        if not ema_crossover:
            logger.warning("No EMA crossover data available for signal generation")
            return AnalysisResult(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                strategy=self.strategy_type,
                market_data=market_data,
                signals=[],
                ai_analysis="Backtesting mode - no EMA data available"
            )

        # Extract EMA crossover data
        is_golden_cross = get_value(ema_crossover, "is_golden_cross", False)
        crossover_strength = get_value(ema_crossover, "crossover_strength", 0)
        ema_9 = to_float(get_value(ema_crossover, "ema_9", 0))
        ema_15 = to_float(get_value(ema_crossover, "ema_15", 0))

        # Get filter states
        long_signal_valid = ema_context.get("long_signal_valid", False)
        short_signal_valid = ema_context.get("short_signal_valid", False)
        all_long_filters_passed = ema_context.get("all_long_filters_passed", False)
        all_short_filters_passed = ema_context.get("all_short_filters_passed", False)
        
        # Signal generation logic based on EMA crossover and filters
        signal_action = SignalAction.NEUTRAL
        confidence = 5
        reasoning = "No clear signal"

        # Golden Cross - BUY Signal
        if is_golden_cross and long_signal_valid:
            signal_action = SignalAction.BUY
            
            # Base confidence from crossover strength
            confidence = max(6, min(10, 5 + crossover_strength))
            
            # Boost confidence if all filters pass
            if all_long_filters_passed:
                confidence = min(10, confidence + 1)
            
            # Determine reasoning based on filters
            if all_long_filters_passed:
                reasoning = f"Strong EMA golden cross with all filters confirmed (strength: {crossover_strength}/10)"
            else:
                reasoning = f"EMA golden cross detected (strength: {crossover_strength}/10)"
                
        # Death Cross - SELL Signal  
        elif not is_golden_cross and short_signal_valid:
            signal_action = SignalAction.SELL
            
            # Base confidence from crossover strength
            confidence = max(6, min(10, 5 + crossover_strength))
            
            # Boost confidence if all filters pass
            if all_short_filters_passed:
                confidence = min(10, confidence + 1)
            
            # Determine reasoning based on filters
            if all_short_filters_passed:
                reasoning = f"Strong EMA death cross with all filters confirmed (strength: {crossover_strength}/10)"
            else:
                reasoning = f"EMA death cross detected (strength: {crossover_strength}/10)"

        # Create signal if action is not neutral
        if signal_action != SignalAction.NEUTRAL:
            signal = TradingSignal(
                symbol=market_data.symbol,
                strategy=self.strategy_type,
                action=signal_action,
                strength=SignalStrength.STRONG if confidence >= 8 else SignalStrength.MODERATE if confidence >= 6 else SignalStrength.WEAK,
                confidence=confidence,
                entry_price=market_data.current_price,
                reasoning=reasoning,
            )
            signals.append(signal)

            logger.debug(
                "EMA crossover signal generated",
                action=signal_action.value,
                confidence=confidence,
                crossover_type="golden_cross" if is_golden_cross else "death_cross",
                filters_passed=all_long_filters_passed if signal_action == SignalAction.BUY else all_short_filters_passed,
            )

        # Create analysis result
        analysis_result = AnalysisResult(
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            strategy=self.strategy_type,
            market_data=market_data,
            signals=signals,
            ai_analysis="Backtesting mode - EMA crossover strategy signals generated from technical analysis"
        )

        # Add EMA crossover data to result
        analysis_result.ema_crossover = ema_crossover

        return analysis_result
