"""
Enhanced EMA Crossover V2 trading strategy implementation.

Uses advanced EMA crossover analysis with trend filtering, market structure
analysis, and enhanced signal confirmation for high-quality trading signals.
"""

import structlog
from typing import Any, Dict
from decimal import Decimal

from src.analysis.strategies.base_strategy import BaseStrategy
from src.core.constants import TradingConstants
from src.core.exceptions import StrategyError, DataValidationError
from src.core.models import (
    AnalysisResult, MarketData, SessionConfig, StrategyType,
    TradingSignal, SignalAction, SignalStrength, EMAStrategyConfig
)
from src.utils.helpers import get_value, to_float

logger = structlog.get_logger(__name__)


class EMACrossoverV2Strategy(BaseStrategy):
    """
    Enhanced EMA Crossover V2 trading strategy.

    Advanced Features:
    - 50 EMA trend filter for high-quality signals
    - Multi-factor trend strength scoring (0-100)
    - Market structure analysis (swing highs/lows)
    - Trend quality assessment with weighted scoring
    - ATR-based dynamic stop losses
    - Volatility percentile calculations
    - Enhanced EMA alignment detection (9>15>50)
    - Configurable filter system (RSI, volume, candlestick, 50 EMA)

    Signal Generation Philosophy:
    - Quality over quantity: Fewer but higher confidence signals
    - Trend filtering: Only trade in favorable market conditions
    - Multi-confirmation: Require multiple indicators to align
    - Risk-adaptive: ATR-based position sizing and stops
    """

    def __init__(self):
        """Initialize Enhanced EMA Crossover V2 strategy."""
        super().__init__()
        self.strategy_type = StrategyType.EMA_CROSSOVER_V2

        # Import here to avoid circular imports
        from src.analysis.indicators import TechnicalIndicators
        self.technical_indicators = TechnicalIndicators()

        logger.info("Enhanced EMA Crossover V2 strategy initialized")

    def _get_minimum_periods(self) -> int:
        """V2 requires more data for 50 EMA and advanced analysis."""
        return max(TradingConstants.EMA_V2_TREND_FILTER_PERIOD + 10, 60)  # 60 periods minimum

    async def analyze(
        self, market_data: MarketData, session_config: SessionConfig
    ) -> AnalysisResult:
        """
        Analyze market data using Enhanced EMA Crossover V2 strategy.

        Args:
            market_data: Market data to analyze
            session_config: Session configuration with V2 parameters

        Returns:
            AnalysisResult with V2 enhanced analysis

        Raises:
            DataValidationError: Invalid input data
            StrategyError: Analysis execution error
        """
        logger.info(
            "Starting EMA Crossover V2 analysis",
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            current_price=market_data.current_price,
            data_points=len(market_data.ohlcv_data),
        )

        try:
            # Validate inputs
            self._validate_market_data(market_data)
            self._validate_session_config(session_config)

            # Check for sufficient data
            min_periods = self._get_minimum_periods()
            if len(market_data.ohlcv_data) < min_periods:
                raise DataValidationError(
                    f"Insufficient data for V2 analysis: {len(market_data.ohlcv_data)} periods provided, "
                    f"minimum {min_periods} required"
                )

            # Calculate enhanced technical analysis with V2 features
            technical_analysis = await self._calculate_v2_analysis(market_data, session_config)

            # Generate AI analysis with V2 context
            analysis_result = await self._generate_ai_analysis(
                market_data, technical_analysis, session_config
            )

            # Filter signals by V2 quality standards
            filtered_result = self._filter_v2_signals(
                analysis_result, session_config, technical_analysis
            )

            # Add V2 specific context to the result
            filtered_result = self._add_v2_context(filtered_result, technical_analysis)

            logger.info(
                "EMA Crossover V2 analysis completed",
                symbol=market_data.symbol,
                signals_count=len(filtered_result.signals),
                trend_strength=technical_analysis.get("trend_strength", 0),
                trend_quality=technical_analysis.get("trend_quality", 0),
                is_trending=technical_analysis.get("is_trending", False),
                primary_signal=str(filtered_result.primary_signal.action)
                if filtered_result.primary_signal
                else None,
            )

            return filtered_result

        except Exception as e:
            logger.error(
                "EMA Crossover V2 analysis failed", symbol=market_data.symbol, error=str(e)
            )
            raise StrategyError(f"EMA Crossover V2 analysis failed: {str(e)}")

    async def _calculate_v2_analysis(
        self, market_data: MarketData, session_config: SessionConfig
    ) -> Dict[str, Any]:
        """
        Calculate V2 enhanced technical analysis.

        Args:
            market_data: Market data to analyze
            session_config: Session configuration with V2 settings

        Returns:
            Enhanced technical analysis with V2 features
        """
        logger.debug(
            "Calculating V2 enhanced technical analysis",
            symbol=market_data.symbol,
            data_points=len(market_data.ohlcv_data),
        )

        # Get base technical analysis (includes basic EMA crossover)
        analysis = await self._calculate_technical_analysis(market_data)

        # Add V2 enhanced indicators
        analysis = await self._enhance_v2_analysis(analysis, market_data, session_config)

        return analysis

    async def _enhance_v2_analysis(
        self, analysis: Dict[str, Any], market_data: MarketData, session_config: SessionConfig
    ) -> Dict[str, Any]:
        """
        Enhance analysis with V2 specific metrics and features.

        Args:
            analysis: Base technical analysis
            market_data: Market data context
            session_config: Session configuration

        Returns:
            Enhanced analysis with V2 features
        """
        try:
            # Get EMA configuration
            ema_config = session_config.ema_config or EMAStrategyConfig()

            # Calculate V2 enhanced indicators
            ohlcv_data = market_data.ohlcv_data

            # Core V2 metrics
            trend_strength = self.technical_indicators.calculate_trend_strength(ohlcv_data)
            is_trending = self.technical_indicators.is_trending_market(
                ohlcv_data, ema_config.min_trend_strength
            )
            trend_quality = self.technical_indicators.calculate_trend_quality(ohlcv_data)

            # Market structure analysis
            market_structure = self.technical_indicators.calculate_market_structure(ohlcv_data)
            
            # Trend duration
            trend_duration = self.technical_indicators.calculate_trend_duration(ohlcv_data)

            # Volatility context
            volatility_percentile = self.technical_indicators.calculate_volatility_percentile(
                ohlcv_data, ema_config.atr_lookback_periods
            )

            # Enhanced EMA calculations (9, 15, 50)
            ema_9_values = self.technical_indicators.calculate_ema(ohlcv_data, 9)
            ema_15_values = self.technical_indicators.calculate_ema(ohlcv_data, 15)
            ema_50_values = self.technical_indicators.calculate_ema(ohlcv_data, 50)

            current_ema_9 = ema_9_values[-1]
            current_ema_15 = ema_15_values[-1]
            current_ema_50 = ema_50_values[-1]

            # EMA alignment detection
            ema_alignment_data = {
                "ema_9": current_ema_9,
                "ema_15": current_ema_15,
                "ema_50": current_ema_50,
                "current_price": market_data.current_price
            }
            ema_alignment = self.technical_indicators.detect_ema_alignment(ema_alignment_data)

            # Enhanced RSI calculation
            rsi_values = self.technical_indicators.calculate_rsi(ohlcv_data, ema_config.rsi_period)
            current_rsi = rsi_values[-1] if rsi_values else 50

            # Enhanced ATR calculation
            atr_values = self.technical_indicators.calculate_atr(ohlcv_data, ema_config.atr_period)
            current_atr = atr_values[-1] if atr_values else 0

            # Volume analysis
            volume_analysis = self.technical_indicators.analyze_volume(ohlcv_data)

            # Advanced candlestick analysis with timeframe context
            candlestick_analysis = self.technical_indicators.detect_advanced_candlestick_patterns(
                ohlcv_data, 
                timeframe=market_data.timeframe,
                atr_value=current_atr
            )

            # Calculate V2 signal scores
            v2_scores = self._calculate_v2_signal_scores(
                trend_strength=trend_strength,
                trend_quality=trend_quality,
                trend_duration=trend_duration,
                ema_alignment=ema_alignment,
                volatility_percentile=volatility_percentile,
                market_structure=market_structure,
                volume_analysis=volume_analysis,
                candlestick_analysis=candlestick_analysis
            )

            # Add V2 data to analysis
            analysis.update({
                # Core V2 metrics
                "trend_strength": trend_strength,
                "is_trending": is_trending,
                "trend_quality": trend_quality,
                "trend_duration": trend_duration,
                "volatility_percentile": volatility_percentile,
                
                # EMA data
                "ema_9_values": ema_9_values,
                "ema_15_values": ema_15_values,
                "ema_50_values": ema_50_values,
                "ema_alignment": ema_alignment,
                
                # Market analysis
                "market_structure": market_structure,
                "rsi_values": rsi_values,
                "current_rsi": current_rsi,
                "atr_values": atr_values,
                "current_atr": current_atr,
                "candlestick_analysis": candlestick_analysis,
                
                # V2 scoring
                "v2_scores": v2_scores,
                
                # Configuration
                "ema_config": ema_config,
            })

            logger.debug(
                "V2 analysis enhancement completed",
                trend_strength=trend_strength,
                trend_quality=trend_quality,
                is_trending=is_trending,
                ema_alignment=ema_alignment,
                volatility_percentile=volatility_percentile,
                trend_duration=trend_duration,
            )

        except Exception as e:
            logger.warning("Failed to enhance V2 analysis", error=str(e))
            # Provide fallback values
            analysis.update({
                "trend_strength": 0,
                "is_trending": False,
                "trend_quality": 0,
                "ema_alignment": "neutral",
                "v2_scores": {"signal_quality": 0}
            })

        return analysis

    def _calculate_v2_signal_scores(
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
        Calculate V2 signal quality scores using weighted factors.

        Args:
            Various analysis components

        Returns:
            Dictionary with V2 scoring metrics
        """
        # Signal quality scoring (0-100)
        signal_quality = 0

        # Trend component (40% weight)
        trend_component = (trend_strength * 0.7 + trend_quality * 0.3) * 0.4
        signal_quality += trend_component

        # EMA alignment component (20% weight)
        alignment_scores = {
            "strong_bullish": 100, "strong_bearish": 100,
            "bullish": 70, "bearish": 70,
            "mixed": 30, "neutral": 20
        }
        alignment_score = alignment_scores.get(ema_alignment, 20) * 0.2
        signal_quality += alignment_score

        # Market structure component (20% weight)
        structure_score = market_structure.get("structure_strength", 50) * 0.2
        signal_quality += structure_score

        # Volume confirmation component (10% weight)
        volume_ratio = volume_analysis.get("volume_ratio", 1.0)
        volume_score = min(100, volume_ratio * 50) * 0.1
        signal_quality += volume_score

        # Candlestick component (10% weight)
        pattern_strength = candlestick_analysis.get("pattern_strength", 5)
        candlestick_score = (pattern_strength / 10) * 100 * 0.1
        signal_quality += candlestick_score

        # Volatility adjustment
        volatility_multiplier = 1.0
        if volatility_percentile > TradingConstants.V2_HIGH_VOLATILITY_THRESHOLD:
            volatility_multiplier = 0.8  # Reduce quality in high volatility

        signal_quality *= volatility_multiplier

        return {
            "signal_quality": max(0, min(100, signal_quality)),
            "trend_component": trend_component,
            "alignment_score": alignment_score,
            "structure_score": structure_score,
            "volume_score": volume_score,
            "candlestick_score": candlestick_score,
            "volatility_multiplier": volatility_multiplier
        }

    def _filter_v2_signals(
        self,
        analysis_result: AnalysisResult,
        session_config: SessionConfig,
        technical_analysis: Dict[str, Any]
    ) -> AnalysisResult:
        """
        Filter signals using V2 quality standards.

        Args:
            analysis_result: Original analysis result
            session_config: Session configuration
            technical_analysis: Technical analysis data

        Returns:
            Filtered analysis result with high-quality signals only
        """
        ema_config = session_config.ema_config or EMAStrategyConfig()
        
        # V2 filtering criteria
        trend_strength = technical_analysis.get("trend_strength", 0)
        trend_quality = technical_analysis.get("trend_quality", 0)
        trend_duration = technical_analysis.get("trend_duration", 0)
        is_trending = technical_analysis.get("is_trending", False)
        v2_scores = technical_analysis.get("v2_scores", {})

        # Filter signals based on V2 criteria
        filtered_signals = []
        
        for signal in analysis_result.signals:
            should_keep = True
            filter_reasons = []

            # Minimum trend strength filter
            if trend_strength < ema_config.min_trend_strength:
                should_keep = False
                filter_reasons.append(f"trend_strength({trend_strength:.1f}) < min({ema_config.min_trend_strength})")

            # Minimum trend quality filter
            if trend_quality < ema_config.min_trend_quality:
                should_keep = False
                filter_reasons.append(f"trend_quality({trend_quality:.1f}) < min({ema_config.min_trend_quality})")

            # Minimum trend duration filter
            if trend_duration < ema_config.min_trend_duration:
                should_keep = False
                filter_reasons.append(f"trend_duration({trend_duration}) < min({ema_config.min_trend_duration})")

            # Trending market requirement
            if not is_trending:
                should_keep = False
                filter_reasons.append("market not trending")

            # Signal quality threshold
            signal_quality = v2_scores.get("signal_quality", 0)
            if signal_quality < 60:  # V2 quality threshold
                should_keep = False
                filter_reasons.append(f"signal_quality({signal_quality:.1f}) < 60")

            if should_keep:
                filtered_signals.append(signal)
            else:
                logger.debug(
                    "V2 signal filtered out",
                    signal_action=signal.action.value,
                    confidence=signal.confidence,
                    filter_reasons=filter_reasons
                )

        # Update analysis result
        analysis_result.signals = filtered_signals

        logger.debug(
            "V2 signal filtering completed",
            original_signals=len(analysis_result.signals),
            filtered_signals=len(filtered_signals),
            trend_strength=trend_strength,
            trend_quality=trend_quality,
            is_trending=is_trending
        )

        return analysis_result

    def _add_v2_context(
        self, analysis_result: AnalysisResult, technical_analysis: Dict[str, Any]
    ) -> AnalysisResult:
        """
        Add V2 specific context to analysis result.

        Args:
            analysis_result: Original analysis result
            technical_analysis: Technical analysis with V2 data

        Returns:
            Enhanced analysis result with V2 context
        """
        try:
            # Generate V2 context text
            v2_context_text = self._generate_v2_context_text(technical_analysis)
            analysis_result.ai_analysis += f"\n\nEMA CROSSOVER V2 CONTEXT:\n{v2_context_text}"

            logger.debug("V2 context added to analysis result")

        except Exception as e:
            logger.warning("Failed to add V2 context", error=str(e))

        return analysis_result

    def _generate_v2_context_text(self, technical_analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable V2 context text.

        Args:
            technical_analysis: Technical analysis data

        Returns:
            Formatted V2 context text
        """
        lines = []

        # Core V2 metrics
        trend_strength = technical_analysis.get("trend_strength", 0)
        trend_quality = technical_analysis.get("trend_quality", 0)
        trend_duration = technical_analysis.get("trend_duration", 0)
        is_trending = technical_analysis.get("is_trending", False)

        lines.append(f"Trend Strength: {trend_strength:.1f}/100 ({'Trending' if is_trending else 'Sideways'})")
        lines.append(f"Trend Quality: {trend_quality:.1f}/100")
        lines.append(f"Trend Duration: {trend_duration} periods")

        # EMA analysis
        ema_alignment = technical_analysis.get("ema_alignment", "neutral")
        lines.append(f"EMA Alignment: {ema_alignment.replace('_', ' ').title()}")

        # Market structure
        market_structure = technical_analysis.get("market_structure", {})
        structure_strength = market_structure.get("structure_strength", 0)
        trend_structure = market_structure.get("trend_structure", "unclear")
        lines.append(f"Market Structure: {trend_structure.title()} (Strength: {structure_strength:.1f}/100)")

        # Volatility context
        volatility_percentile = technical_analysis.get("volatility_percentile", 0.5)
        volatility_level = "High" if volatility_percentile > 0.7 else "Normal" if volatility_percentile > 0.3 else "Low"
        lines.append(f"Volatility: {volatility_level} ({volatility_percentile:.1%} percentile)")

        # V2 signal quality
        v2_scores = technical_analysis.get("v2_scores", {})
        signal_quality = v2_scores.get("signal_quality", 0)
        lines.append(f"V2 Signal Quality: {signal_quality:.1f}/100")

        # ATR context
        current_atr = technical_analysis.get("current_atr", 0)
        if current_atr > 0:
            lines.append(f"ATR(14): ${current_atr:.2f}")

        return "\n".join(lines)

    async def _generate_backtesting_signals(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        session_config: SessionConfig,
    ) -> AnalysisResult:
        """
        Generate V2 signals for backtesting without AI analysis.

        Args:
            market_data: Market data context
            technical_analysis: Technical analysis results
            session_config: Session configuration

        Returns:
            AnalysisResult with V2 signals
        """
        from src.core.models import AnalysisResult, TradingSignal, SignalAction, SignalStrength

        logger.debug(
            "Generating EMA Crossover V2 backtesting signals",
            symbol=market_data.symbol,
            current_price=market_data.current_price,
        )

        # Enhance technical analysis with V2 data
        enhanced_analysis = await self._enhance_v2_analysis(
            technical_analysis, market_data, session_config
        )

        signals = []

        # Check if we have sufficient data and trending conditions
        trend_strength = enhanced_analysis.get("trend_strength", 0)
        is_trending = enhanced_analysis.get("is_trending", False)
        trend_quality = enhanced_analysis.get("trend_quality", 0)
        ema_alignment = enhanced_analysis.get("ema_alignment", "neutral")
        
        ema_config = session_config.ema_config or EMAStrategyConfig()

        # Only generate signals if V2 quality criteria are met
        if (trend_strength >= ema_config.min_trend_strength and
            trend_quality >= ema_config.min_trend_quality and
            is_trending):

            ema_crossover = enhanced_analysis.get("ema_crossover")
            v2_scores = enhanced_analysis.get("v2_scores", {})

            if ema_crossover:
                is_golden_cross = get_value(ema_crossover, "is_golden_cross", False)
                crossover_strength = get_value(ema_crossover, "crossover_strength", 0)

                # Apply 50 EMA filter if enabled
                current_price = market_data.current_price
                ema_50_values = enhanced_analysis.get("ema_50_values", [])
                current_ema_50 = ema_50_values[-1] if ema_50_values else current_price

                # V2 signal logic
                signal_action = SignalAction.NEUTRAL
                confidence = 5
                reasoning = "No V2 signal criteria met"

                # Golden Cross with V2 enhancements
                if (is_golden_cross and 
                    ema_alignment in ["strong_bullish", "bullish"] and
                    (not ema_config.enable_ema50_filter or current_price > current_ema_50)):
                    
                    signal_action = SignalAction.BUY
                    confidence = max(7, min(10, int(v2_scores.get("signal_quality", 70) / 10)))
                    reasoning = f"V2 Enhanced Golden Cross: Trend strength {trend_strength:.1f}, Quality {trend_quality:.1f}, {ema_alignment} alignment"

                # Death Cross with V2 enhancements
                elif (not is_golden_cross and 
                      ema_alignment in ["strong_bearish", "bearish"] and
                      (not ema_config.enable_ema50_filter or current_price < current_ema_50)):
                    
                    signal_action = SignalAction.SELL
                    confidence = max(7, min(10, int(v2_scores.get("signal_quality", 70) / 10)))
                    reasoning = f"V2 Enhanced Death Cross: Trend strength {trend_strength:.1f}, Quality {trend_quality:.1f}, {ema_alignment} alignment"

                # Create signal if criteria met
                if signal_action != SignalAction.NEUTRAL:
                    # Calculate ATR-based stop loss
                    current_atr = enhanced_analysis.get("current_atr", 0)
                    atr_multiplier = ema_config.atr_stop_multiplier
                    
                    if signal_action == SignalAction.BUY:
                        stop_loss = current_price - (current_atr * atr_multiplier) if current_atr > 0 else None
                        take_profit = current_price + (current_atr * atr_multiplier * 2) if current_atr > 0 else None
                    else:
                        stop_loss = current_price + (current_atr * atr_multiplier) if current_atr > 0 else None
                        take_profit = current_price - (current_atr * atr_multiplier * 2) if current_atr > 0 else None

                    signal = TradingSignal(
                        symbol=market_data.symbol,
                        strategy=self.strategy_type,
                        action=signal_action,
                        strength=SignalStrength.STRONG if confidence >= 8 else SignalStrength.MODERATE,
                        confidence=confidence,
                        entry_price=current_price,
                        stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
                        take_profit=Decimal(str(take_profit)) if take_profit else None,
                        reasoning=reasoning,
                    )
                    signals.append(signal)

                    logger.debug(
                        "V2 signal generated",
                        action=signal_action.value,
                        confidence=confidence,
                        trend_strength=trend_strength,
                        trend_quality=trend_quality,
                        ema_alignment=ema_alignment,
                        signal_quality=v2_scores.get("signal_quality", 0)
                    )

        # Create analysis result
        analysis_result = AnalysisResult(
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            strategy=self.strategy_type,
            market_data=market_data,
            signals=signals,
            ai_analysis="Backtesting mode - EMA Crossover V2 enhanced signals generated"
        )

        return analysis_result