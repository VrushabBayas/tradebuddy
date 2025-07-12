"""
Ollama API client for AI-powered market analysis.

Provides integration with Ollama's local LLM (Qwen2.5:14b) for intelligent
trading signal generation based on technical analysis data.
"""

import asyncio
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

from src.core.config import settings
from src.core.constants import ErrorCodes, OllamaConstants
from src.core.exceptions import (
    APIConnectionError,
    APITimeoutError,
    DataValidationError,
    OllamaAPIError,
)
from src.core.models import (
    AnalysisResult,
    MarketData,
    SessionConfig,
    SignalAction,
    SignalStrength,
    StrategyType,
    TechnicalIndicator,
    TradingSignal,
)
from src.utils.risk_management import (
    calculate_stop_loss_take_profit,
    optimize_position_for_delta_exchange,
)

logger = structlog.get_logger(__name__)


class OllamaClient:
    """
    Async client for Ollama API integration.

    Provides methods for:
    - Market analysis using local LLM
    - Trading signal generation
    - Strategy-specific analysis
    - Model health monitoring

    Features:
    - Automatic prompt optimization for each strategy
    - Signal parsing and validation
    - Error handling and retry logic
    - Performance monitoring
    """

    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.ollama_api_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or OllamaConstants.DEFAULT_TIMEOUT

        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0

        logger.info(
            "Ollama client initialized",
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=10, limit_per_host=5, enable_cleanup_closed=True
            )

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "TradeBuddy/0.1.0",
                    # Anti-caching headers for AI requests
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                },
            )

            logger.debug("Ollama HTTP session created")

    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Ollama HTTP session closed")

    async def _make_request(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API.

        Args:
            endpoint: API endpoint
            payload: Request payload

        Returns:
            Response data

        Raises:
            APIConnectionError: Connection or HTTP errors
            APITimeoutError: Request timeout
            OllamaAPIError: Ollama-specific errors
        """
        await self._ensure_session()

        # Rate limiting - ensure minimum time between requests
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / OllamaConstants.MAX_RETRIES  # Conservative rate limiting

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self._last_request_time = time.time()

        url = f"{self.base_url}{endpoint}"

        # Add request uniqueness to prevent any AI response caching
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()

        # Inject uniqueness into the prompt
        if "prompt" in payload:
            payload[
                "prompt"
            ] += f"\n\n[Request ID: {request_id}, Timestamp: {timestamp}]"

        try:
            logger.debug(
                "Making Ollama API request",
                url=url,
                model=payload.get("model"),
                prompt_length=len(payload.get("prompt", "")),
                request_id=request_id,
                anti_cache_enabled=True,
            )

            start_time = time.time()

            async with self._session.post(url, json=payload) as response:
                request_time = time.time() - start_time

                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        "Ollama API request failed",
                        status=response.status,
                        error=error_text,
                        request_time=request_time,
                    )

                    if response.status == 404:
                        raise OllamaAPIError(
                            f"Model '{self.model}' not found. Please ensure it's installed.",
                            error_code=ErrorCodes.API_INVALID_RESPONSE,
                            status_code=response.status,
                        )

                    raise APIConnectionError(
                        f"Ollama API request failed with status {response.status}: {error_text}",
                        error_code=ErrorCodes.API_CONNECTION_FAILED,
                    )

                result = await response.json()

                logger.debug(
                    "Ollama API request successful",
                    status=response.status,
                    request_time=request_time,
                    response_length=len(result.get("response", "")),
                )

                return result

        except asyncio.TimeoutError:
            logger.error("Ollama request timeout", url=url, timeout=self.timeout)
            raise APITimeoutError(
                f"Ollama request timeout after {self.timeout} seconds",
                error_code=ErrorCodes.API_TIMEOUT,
            )
        except aiohttp.ClientError as e:
            logger.error("Ollama HTTP client error", error=str(e), url=url)
            raise APIConnectionError(
                f"Ollama HTTP client error: {str(e)}",
                error_code=ErrorCodes.API_CONNECTION_FAILED,
            )

    def _validate_market_data(self, market_data: MarketData) -> None:
        """Validate market data for analysis."""
        if market_data is None:
            raise DataValidationError("Market data cannot be None")

        if not market_data.ohlcv_data:
            raise DataValidationError("No OHLCV data provided for analysis")

        if market_data.current_price <= 0:
            raise DataValidationError("Invalid current price in market data")

    def _generate_prompt(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy: StrategyType,
    ) -> str:
        """
        Generate strategy-specific prompt for AI analysis.

        Args:
            market_data: Market data to analyze
            technical_analysis: Technical analysis results
            strategy: Trading strategy to apply

        Returns:
            Formatted prompt string
        """
        # Base market data
        base_prompt = f"""You are a professional cryptocurrency trading analyst using Delta Exchange market data.
Analyze the following live market data using {strategy.value.replace('_', ' ').title()} strategy:

DELTA EXCHANGE DATA:
Symbol: {market_data.symbol}
Current Live Price: ${market_data.current_price:,.2f}
Timeframe: {market_data.timeframe}
Data Source: Delta Exchange API
Timestamp: {market_data.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

OHLCV DATA (Latest):"""

        # Add latest OHLCV data
        if market_data.latest_ohlcv:
            ohlcv = market_data.latest_ohlcv
            base_prompt += f"""
Open: ${ohlcv.open:,.2f}
High: ${ohlcv.high:,.2f}
Low: ${ohlcv.low:,.2f}
Close: ${ohlcv.close:,.2f}
Volume: {ohlcv.volume:,.1f}"""

        # Add technical analysis based on strategy
        if strategy == StrategyType.EMA_CROSSOVER:
            base_prompt += self._add_ema_analysis(technical_analysis)
        elif strategy == StrategyType.SUPPORT_RESISTANCE:
            base_prompt += self._add_sr_analysis(technical_analysis)
        elif strategy == StrategyType.COMBINED:
            base_prompt += self._add_combined_analysis(technical_analysis)

        # Add instruction
        instruction = """

REQUIRED OUTPUT FORMAT:
TRADING SIGNAL: [BUY/SELL/NEUTRAL/WAIT]
CONFIDENCE: [1-10]/10
ENTRY PRICE: $[price]
STOP LOSS: $[price]
TAKE PROFIT: $[price]

ANALYSIS:
[Provide clear reasoning for the signal]

REASONING:
[List 3-5 key factors supporting the decision]

Provide a clear, actionable trading signal with specific entry/exit levels and detailed reasoning."""

        return base_prompt + instruction

    def _add_ema_analysis(self, analysis: Dict[str, Any]) -> str:
        """Add Enhanced EMA-specific analysis to prompt."""
        ema_data = analysis.get("ema_crossover")
        volume_data = analysis.get("volume_analysis", {})
        price_action = analysis.get("price_action", {})
        ema_context = analysis.get("ema_context", {})
        rsi_values = analysis.get("rsi_values", [])
        atr_values = analysis.get("atr_values", [])
        candlestick_analysis = analysis.get("candlestick_analysis", {})

        # Handle EMACrossover object or dict
        if hasattr(ema_data, "ema_9"):
            ema_9 = ema_data.ema_9
            ema_15 = ema_data.ema_15
            is_golden_cross = ema_data.is_golden_cross
            crossover_strength = ema_data.crossover_strength
        else:
            ema_9 = ema_data.get("ema_9", 0) if ema_data else 0
            ema_15 = ema_data.get("ema_15", 0) if ema_data else 0
            is_golden_cross = (
                ema_data.get("is_golden_cross", False) if ema_data else False
            )
            crossover_strength = (
                ema_data.get("crossover_strength", 0) if ema_data else 0
            )

        # Get enhanced indicators
        current_rsi = rsi_values[-1] if rsi_values else 50
        current_atr = atr_values[-1] if atr_values else 0
        ema_50 = ema_context.get("ema_50", 0)

        ema_prompt = f"""

ENHANCED EMA CROSSOVER ANALYSIS:
9 EMA: ${ema_9:,.2f}
15 EMA: ${ema_15:,.2f}
50 EMA: ${ema_50:,.2f} (Trend Filter)
Golden Cross: {'TRUE' if is_golden_cross else 'FALSE'}
Crossover Strength: {crossover_strength}/10
Price Above 50 EMA: {'TRUE' if ema_context.get('price_above_ema50') else 'FALSE'}

RSI ANALYSIS:
RSI(14): {current_rsi:.1f}
RSI Signal: {'BULLISH (>50)' if current_rsi > 50 else 'BEARISH (<50)'}
RSI Overbought: {'TRUE' if current_rsi > 70 else 'FALSE'}
RSI Oversold: {'TRUE' if current_rsi < 30 else 'FALSE'}

VOLATILITY ANALYSIS (ATR):
ATR(14): ${current_atr:.2f}
Suggested Stop Loss Distance: ${current_atr * 1.5:.2f} (1.5x ATR)

ENHANCED VOLUME ANALYSIS:
Current Volume: {volume_data.get('current_volume', 0):,.1f}
Volume SMA(20): {volume_data.get('volume_sma_20', 0):,.1f}
Volume vs SMA(20): {volume_data.get('volume_vs_sma_20', 1.0):.2f}x
Volume Confirmation (110%): {'PASSED' if volume_data.get('volume_confirmation_110pct') else 'FAILED'}
Volume Trend: {volume_data.get('volume_trend', 'stable').title()}

CANDLESTICK CONFIRMATION:
Pattern: {candlestick_analysis.get('pattern', 'unknown').replace('_', ' ').title()}
Confirmation: {candlestick_analysis.get('confirmation', 'neutral').title()}
Strength: {candlestick_analysis.get('strength', 'weak').title()}
Bullish Candle: {'TRUE' if candlestick_analysis.get('is_bullish') else 'FALSE'}

STRATEGY FILTERS STATUS:"""

        # Add filter status
        if ema_context.get("enhanced_long_filters"):
            long_filters = ema_context["enhanced_long_filters"]
            ema_prompt += f"""
LONG SIGNAL FILTERS:
- RSI > 50: {'✓ PASS' if long_filters.get('rsi_filter') else '✗ FAIL'}
- Price > 50 EMA: {'✓ PASS' if long_filters.get('ema50_filter') else '✗ FAIL'}
- Volume ≥ 110% SMA: {'✓ PASS' if long_filters.get('volume_filter') else '✗ FAIL'}
- Bullish Candle: {'✓ PASS' if long_filters.get('candlestick_filter') else '✗ FAIL'}
All Long Filters: {'✓ PASSED' if ema_context.get('all_long_filters_passed') else '✗ FAILED'}"""

        if ema_context.get("enhanced_short_filters"):
            short_filters = ema_context["enhanced_short_filters"]
            ema_prompt += f"""
SHORT SIGNAL FILTERS:
- RSI < 50: {'✓ PASS' if short_filters.get('rsi_filter') else '✗ FAIL'}
- Price < 50 EMA: {'✓ PASS' if short_filters.get('ema50_filter') else '✗ FAIL'}
- Volume ≥ 110% SMA: {'✓ PASS' if short_filters.get('volume_filter') else '✗ FAIL'}
- Bearish Candle: {'✓ PASS' if short_filters.get('candlestick_filter') else '✗ FAIL'}
All Short Filters: {'✓ PASSED' if ema_context.get('all_short_filters_passed') else '✗ FAILED'}"""

        ema_prompt += f"""

PRICE ACTION:
Trend Direction: {price_action.get('trend_direction', 'neutral').title()}
Trend Strength: {price_action.get('trend_strength', 0)}/10
Momentum: {price_action.get('momentum', 0):.2f}%
Volatility: {price_action.get('volatility', 0):.2f}%

ENHANCED STRATEGY NOTES:
- This is the comprehensive 9/15 EMA crossover strategy with RSI, 50 EMA, Volume, and Candlestick filters
- Long signals require: 9 EMA > 15 EMA + Close above both EMAs + Optional filters
- Short signals require: 9 EMA < 15 EMA + Close below both EMAs + Optional filters
- Use 1.5x ATR for dynamic stop losses as shown above
- All filters must pass for high-confidence signals"""

        return ema_prompt

    def _add_sr_analysis(self, analysis: Dict[str, Any]) -> str:
        """Add Support/Resistance analysis to prompt."""
        sr_levels = analysis.get("support_resistance", [])
        volume_data = analysis.get("volume_analysis", {})
        price_action = analysis.get("price_action", {})

        sr_prompt = """

SUPPORT/RESISTANCE ANALYSIS:"""

        # Handle list of SupportResistanceLevel objects or dicts
        support_levels = []
        resistance_levels = []

        for level in sr_levels:
            if hasattr(level, "is_support"):
                # It's a SupportResistanceLevel object
                if level.is_support:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
            else:
                # It's a dictionary
                if level.get("is_support", False):
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)

        # Add support levels
        if support_levels:
            sr_prompt += "\nSupport Levels:"
            for level in support_levels[:3]:  # Top 3 support levels
                if hasattr(level, "level"):
                    sr_prompt += f"\n- Support Level: ${level.level:,.2f} (Strength: {level.strength}/10, Touches: {level.touches})"
                else:
                    sr_prompt += f"\n- Support Level: ${level.get('level', 0):,.2f} (Strength: {level.get('strength', 0)}/10, Touches: {level.get('touches', 0)})"

        # Add resistance levels
        if resistance_levels:
            sr_prompt += "\nResistance Levels:"
            for level in resistance_levels[:3]:  # Top 3 resistance levels
                if hasattr(level, "level"):
                    sr_prompt += f"\n- Resistance Level: ${level.level:,.2f} (Strength: {level.strength}/10, Touches: {level.touches})"
                else:
                    sr_prompt += f"\n- Resistance Level: ${level.get('level', 0):,.2f} (Strength: {level.get('strength', 0)}/10, Touches: {level.get('touches', 0)})"

        sr_prompt += f"""

VOLUME CONFIRMATION:
Current Volume: {volume_data.get('current_volume', 0):,.1f}
Volume vs Average: {volume_data.get('volume_ratio', 1.0):.2f}x
Volume Trend: {volume_data.get('volume_trend', 'stable').title()}

PRICE ACTION:
Trend Direction: {price_action.get('trend_direction', 'neutral').title()}
Trend Strength: {price_action.get('trend_strength', 0)}/10"""

        return sr_prompt

    def _add_combined_analysis(self, analysis: Dict[str, Any]) -> str:
        """Add combined strategy analysis to prompt."""
        # Combine both EMA and S/R analysis
        combined_prompt = self._add_ema_analysis(analysis)
        combined_prompt += self._add_sr_analysis(analysis)

        # Add overall sentiment
        overall_sentiment = analysis.get("overall_sentiment", "neutral")
        confidence_score = analysis.get("confidence_score", 5)

        combined_prompt += f"""

OVERALL TECHNICAL SENTIMENT:
Market Sentiment: {overall_sentiment.title()}
Technical Confidence: {confidence_score}/10

STRATEGY NOTE: Use BOTH EMA crossover and Support/Resistance levels for signal confirmation. 
Only provide BUY/SELL signals when both strategies align. Use NEUTRAL when strategies conflict."""

        return combined_prompt

    def _parse_trading_signals(
        self,
        ai_response: str,
        market_data: MarketData,
        strategy: StrategyType,
        session_config: SessionConfig = None,
    ) -> List[TradingSignal]:
        """
        Parse trading signals from AI response.

        Args:
            ai_response: AI response text
            market_data: Market data context
            strategy: Strategy used

        Returns:
            List of parsed trading signals
        """
        signals = []

        try:
            # Extract signal components using regex
            signal_pattern = r"TRADING SIGNAL:\s*(BUY|SELL|NEUTRAL|WAIT)"
            confidence_pattern = r"CONFIDENCE:\s*(\d+)(?:/10)?"
            entry_pattern = r"ENTRY PRICE:\s*\$?([0-9,]+\.?\d*)"
            stop_loss_pattern = r"STOP LOSS:\s*\$?([0-9,]+\.?\d*)"
            take_profit_pattern = r"TAKE PROFIT:\s*\$?([0-9,]+\.?\d*)"

            # Find all matches
            signal_matches = re.findall(signal_pattern, ai_response, re.IGNORECASE)
            confidence_matches = re.findall(
                confidence_pattern, ai_response, re.IGNORECASE
            )
            entry_matches = re.findall(entry_pattern, ai_response, re.IGNORECASE)
            stop_loss_matches = re.findall(
                stop_loss_pattern, ai_response, re.IGNORECASE
            )
            take_profit_matches = re.findall(
                take_profit_pattern, ai_response, re.IGNORECASE
            )

            # Process primary signal (first/strongest match)
            if signal_matches:
                signal_action = SignalAction(signal_matches[0].upper())
                confidence = int(confidence_matches[0]) if confidence_matches else 5

                # Parse prices
                entry_price = (
                    float(entry_matches[0].replace(",", ""))
                    if entry_matches
                    else market_data.current_price
                )

                # Use AI-provided levels or calculate using risk management
                ai_stop_loss = (
                    float(stop_loss_matches[0].replace(",", ""))
                    if stop_loss_matches
                    else None
                )
                ai_take_profit = (
                    float(take_profit_matches[0].replace(",", ""))
                    if take_profit_matches
                    else None
                )

                # Calculate optimized stop loss and take profit if session config is available
                if session_config and signal_action in [
                    SignalAction.BUY,
                    SignalAction.SELL,
                ]:
                    (
                        calc_stop_loss,
                        calc_take_profit,
                        calc_risk_reward,
                    ) = calculate_stop_loss_take_profit(
                        entry_price=entry_price,
                        signal_action=signal_action,
                        stop_loss_pct=float(session_config.stop_loss_pct),
                        take_profit_pct=float(session_config.take_profit_pct),
                        leverage=session_config.leverage,
                    )

                    # Use AI levels if provided, otherwise use calculated levels
                    stop_loss = ai_stop_loss if ai_stop_loss else calc_stop_loss
                    take_profit = ai_take_profit if ai_take_profit else calc_take_profit
                    risk_reward_ratio = calc_risk_reward

                    # Calculate position parameters
                    position_params = optimize_position_for_delta_exchange(
                        session_config=session_config,
                        current_price=market_data.current_price,
                        account_balance=10000.0,  # Default account balance
                    )
                    position_size_pct = float(session_config.position_size_pct)

                else:
                    # Fallback to AI-provided or default values
                    stop_loss = ai_stop_loss
                    take_profit = ai_take_profit
                    position_size_pct = 5.0  # Default 5% for leveraged trading

                    # Calculate simple risk-reward ratio
                    if (
                        signal_action in [SignalAction.BUY, SignalAction.SELL]
                        and stop_loss
                        and take_profit
                    ):
                        if signal_action == SignalAction.BUY:
                            risk = abs(entry_price - stop_loss)
                            reward = abs(take_profit - entry_price)
                        else:  # SELL
                            risk = abs(stop_loss - entry_price)
                            reward = abs(entry_price - take_profit)

                        risk_reward_ratio = reward / risk if risk > 0 else None
                    else:
                        risk_reward_ratio = None

                # Determine signal strength
                if confidence >= 8:
                    strength = SignalStrength.STRONG
                elif confidence >= 6:
                    strength = SignalStrength.MODERATE
                else:
                    strength = SignalStrength.WEAK

                # Extract reasoning
                reasoning_match = re.search(
                    r"REASONING:\s*(.*?)(?:\n\n|\Z)",
                    ai_response,
                    re.DOTALL | re.IGNORECASE,
                )
                reasoning = (
                    reasoning_match.group(1).strip()
                    if reasoning_match
                    else "AI analysis provided"
                )

                # Create trading signal
                signal = TradingSignal(
                    symbol=market_data.symbol,
                    strategy=strategy,
                    action=signal_action,
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=reasoning,
                    risk_reward_ratio=risk_reward_ratio,
                    position_size_pct=position_size_pct,
                )

                signals.append(signal)

                logger.debug(
                    "Trading signal parsed",
                    action=signal_action.value,
                    confidence=confidence,
                    entry_price=entry_price,
                    risk_reward=risk_reward_ratio,
                )

            # If no valid signal found, create a default WAIT signal
            if not signals:
                default_signal = TradingSignal(
                    symbol=market_data.symbol,
                    strategy=strategy,
                    action=SignalAction.WAIT,
                    strength=SignalStrength.WEAK,
                    confidence=1,
                    entry_price=market_data.current_price,
                    reasoning="Unable to parse clear signal from AI response",
                )
                signals.append(default_signal)

                logger.warning("No valid signal parsed, created default WAIT signal")

        except Exception as e:
            logger.error("Error parsing trading signals", error=str(e))

            # Create fallback signal
            fallback_signal = TradingSignal(
                symbol=market_data.symbol,
                strategy=strategy,
                action=SignalAction.WAIT,
                strength=SignalStrength.WEAK,
                confidence=1,
                entry_price=market_data.current_price,
                reasoning=f"Signal parsing error: {str(e)}",
            )
            signals.append(fallback_signal)

        return signals

    async def analyze_market(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy: StrategyType,
        session_config: SessionConfig = None,
    ) -> AnalysisResult:
        """
        Analyze market data using AI and generate trading signals.

        Args:
            market_data: Market data to analyze
            technical_analysis: Technical analysis results
            strategy: Trading strategy to apply

        Returns:
            AnalysisResult with AI analysis and signals

        Raises:
            DataValidationError: Invalid input data
            APIConnectionError: API communication error
            APITimeoutError: Request timeout
        """
        # Validate inputs
        self._validate_market_data(market_data)

        logger.info(
            "Starting AI market analysis",
            symbol=market_data.symbol,
            strategy=strategy.value,
            timeframe=market_data.timeframe,
            current_price=market_data.current_price,
        )

        start_time = time.time()

        try:
            # Generate strategy-specific prompt
            prompt = self._generate_prompt(market_data, technical_analysis, strategy)

            # Prepare API request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent analysis
                    "top_k": 10,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            }

            # Make API request
            response = await self._make_request("/api/generate", payload)
            ai_response = response.get("response", "")

            if not ai_response:
                raise OllamaAPIError(
                    "Empty response from Ollama API",
                    error_code=ErrorCodes.API_INVALID_RESPONSE,
                )

            # Parse trading signals
            signals = self._parse_trading_signals(
                ai_response, market_data, strategy, session_config
            )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create analysis result
            analysis_result = AnalysisResult(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                strategy=strategy,
                market_data=market_data,
                signals=signals,
                ai_analysis=ai_response,
                execution_time=execution_time,
            )

            logger.info(
                "AI market analysis completed",
                symbol=market_data.symbol,
                strategy=strategy.value,
                signals_count=len(signals),
                execution_time=execution_time,
                primary_signal=signals[0].action.value if signals else None,
                confidence=signals[0].confidence if signals else None,
            )

            return analysis_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "AI market analysis failed",
                symbol=market_data.symbol,
                strategy=strategy.value,
                execution_time=execution_time,
                error=str(e),
            )
            raise

    async def check_model_health(self) -> bool:
        """
        Check if the specified model is available and healthy.

        Returns:
            True if model is available and healthy
        """
        try:
            logger.debug("Checking Ollama model health", model=self.model)

            response = await self._make_request("/api/tags", {})
            models = response.get("models", [])

            # Check if our model is in the list
            model_found = any(model.get("name") == self.model for model in models)

            logger.debug(
                "Model health check completed",
                model=self.model,
                found=model_found,
                total_models=len(models),
            )

            return model_found

        except Exception as e:
            logger.error("Model health check failed", model=self.model, error=str(e))
            return False

    async def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current model.

        Returns:
            Model information dictionary or None if not found
        """
        try:
            response = await self._make_request("/api/tags", {})
            models = response.get("models", [])

            for model in models:
                if model.get("name") == self.model:
                    return model

            return None

        except Exception as e:
            logger.error("Failed to get model info", model=self.model, error=str(e))
            return None
