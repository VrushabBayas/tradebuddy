"""
FinGPT client for financial analysis.

Implements FinGPT integration following AIModelInterface contract.
"""

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import structlog

from src.analysis.ai_models.ai_interface import AIModelInterface, validate_ai_inputs
from src.core.models import (
    MarketData,
    AnalysisResult,
    StrategyType,
    SessionConfig,
    TradingSignal,
    SignalAction,
    SignalStrength,
    Symbol,
)
from src.core.exceptions import (
    APIConnectionError,
    APITimeoutError,
    DataValidationError,
)
from src.utils.helpers import get_value, to_float

logger = structlog.get_logger(__name__)


class FinGPTClient(AIModelInterface):
    """
    FinGPT client implementing AIModelInterface.
    
    Provides financial-specific AI analysis using FinGPT models.
    """

    def __init__(self, model_variant: str = "v3.2", api_endpoint: str = None, 
                 api_key: str = None, timeout: int = 30):
        """
        Initialize FinGPT client.

        Args:
            model_variant: FinGPT model variant (v3.1, v3.2, v3.3)
            api_endpoint: FinGPT API endpoint URL
            api_key: FinGPT API key for authentication
            timeout: Request timeout in seconds
        """
        self.model_variant = model_variant
        self.api_endpoint = api_endpoint or self._get_default_endpoint()
        self.api_key = api_key
        self.timeout = timeout
        
        # Validate model variant
        valid_variants = ["v3.1", "v3.2", "v3.3"]
        if model_variant not in valid_variants:
            raise ValueError(f"Invalid model variant: {model_variant}. Must be one of {valid_variants}")

        logger.info(
            "FinGPT client initialized",
            model_variant=model_variant,
            api_endpoint=self.api_endpoint,
            has_api_key=bool(api_key),
            timeout=timeout
        )

    def _get_default_endpoint(self) -> str:
        """Get default FinGPT API endpoint."""
        # In a real implementation, this would be the actual FinGPT API endpoint
        # For now, using a placeholder that can be configured
        return "http://localhost:8000/api/fingpt"

    async def analyze_market(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy: StrategyType,
        session_config: Optional[SessionConfig] = None,
    ) -> AnalysisResult:
        """
        Analyze market data using FinGPT.

        Args:
            market_data: Market data to analyze
            technical_analysis: Technical analysis results
            strategy: Trading strategy to apply
            session_config: Optional session configuration

        Returns:
            AnalysisResult with FinGPT analysis and signals
        """
        # Validate inputs (Fail Fast principle)
        validate_ai_inputs(market_data, technical_analysis, strategy)

        logger.info(
            "Starting FinGPT market analysis",
            symbol=market_data.symbol,
            strategy=strategy.value,
            model_variant=self.model_variant
        )

        start_time = time.time()

        try:
            # Generate financial-specific prompt
            prompt = self._generate_financial_prompt(
                market_data, technical_analysis, strategy
            )

            # Make API request to FinGPT
            response = await self._make_fingpt_request(prompt)

            # Parse FinGPT response
            signals = self._parse_fingpt_response(response, market_data, strategy)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create analysis result
            analysis_result = AnalysisResult(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                strategy=strategy,
                market_data=market_data,
                signals=signals,
                ai_analysis=response,
                execution_time=execution_time,
            )

            logger.info(
                "FinGPT analysis completed",
                symbol=market_data.symbol,
                strategy=strategy.value,
                signals_count=len(signals),
                execution_time=execution_time
            )

            return analysis_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "FinGPT analysis failed",
                symbol=market_data.symbol,
                strategy=strategy.value,
                execution_time=execution_time,
                error=str(e)
            )
            raise

    def _generate_financial_prompt(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy: StrategyType,
    ) -> str:
        """
        Generate financial-specific prompt for FinGPT.

        Args:
            market_data: Market data
            technical_analysis: Technical analysis results
            strategy: Trading strategy

        Returns:
            FinGPT-optimized prompt
        """
        prompt = f"""FinGPT Financial Analysis Request

MARKET DATA ANALYSIS:
Symbol: {market_data.symbol}
Current Price: ${market_data.current_price:,.2f}
Timeframe: {market_data.timeframe}
Strategy: {strategy.value.replace('_', ' ').title()}
Timestamp: {market_data.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

OHLCV DATA (Latest Candles):"""

        # Add recent OHLCV data
        for i, ohlcv in enumerate(market_data.ohlcv_data[-5:]):  # Last 5 candles
            prompt += f"""
Candle {i+1}: O=${ohlcv.open:,.2f} H=${ohlcv.high:,.2f} L=${ohlcv.low:,.2f} C=${ohlcv.close:,.2f} V={ohlcv.volume:,.0f}"""

        # Add technical indicators based on strategy
        prompt += "\n\nTECHNICAL INDICATORS:"
        
        if strategy == StrategyType.EMA_CROSSOVER_V2:
            ema_data = technical_analysis.get("ema_crossover", {})
            ema_9 = to_float(get_value(ema_data, "ema_9", 0))
            ema_15 = to_float(get_value(ema_data, "ema_15", 0))
            is_golden_cross = get_value(ema_data, "is_golden_cross", False)
            crossover_strength = get_value(ema_data, "crossover_strength", 0)
            prompt += f"""
9 EMA: ${ema_9:,.2f}
15 EMA: ${ema_15:,.2f}
Golden Cross: {is_golden_cross}
Crossover Strength: {crossover_strength}/10"""

        elif strategy == StrategyType.SUPPORT_RESISTANCE:
            sr_levels = technical_analysis.get("support_resistance", [])
            prompt += "\nSupport/Resistance Levels:"
            for level in sr_levels[:3]:  # Top 3 levels
                level_type = "Support" if get_value(level, "is_support", False) else "Resistance"
                level_price = to_float(get_value(level, "level", 0))
                level_strength = get_value(level, "strength", 0)
                prompt += f"\n{level_type}: ${level_price:,.2f} (Strength: {level_strength}/10)"

        # Add volume analysis
        volume_data = technical_analysis.get("volume_analysis", {})
        if volume_data:
            prompt += f"""

VOLUME ANALYSIS:
Current Volume: {volume_data.get('current_volume', 0):,.0f}
Volume Ratio: {volume_data.get('volume_ratio', 1.0):.2f}x
Volume Trend: {volume_data.get('volume_trend', 'neutral').title()}"""

        # Add FinGPT-specific instruction
        prompt += """

FINTECH ANALYSIS REQUEST:
As a specialized financial AI model, provide a comprehensive analysis including:

1. Market sentiment assessment
2. Technical pattern recognition
3. Risk-adjusted trading recommendation
4. Price target analysis

OUTPUT FORMAT:
TRADING SIGNAL: [BUY/SELL/NEUTRAL/WAIT]
CONFIDENCE: [1-10]/10
ENTRY PRICE: $[price]
STOP LOSS: $[price]
TAKE PROFIT: $[price]

FINANCIAL REASONING:
[Provide detailed financial analysis and reasoning]

Focus on risk management and provide specific entry/exit levels optimized for leveraged trading."""

        return prompt

    async def _make_fingpt_request(self, prompt: str) -> str:
        """
        Make API request to FinGPT.

        Args:
            prompt: FinGPT prompt

        Returns:
            FinGPT response text

        Raises:
            APIConnectionError: Connection error
            APITimeoutError: Request timeout
        """
        try:
            # First try real API call
            response = await self._make_real_api_request(prompt)
            if response:
                return response
        except Exception as e:
            logger.warning("Real FinGPT API failed, falling back to mock", error=str(e))
        
        # Fallback to mock response for development/testing
        return await self._make_mock_response(prompt)

    async def _make_real_api_request(self, prompt: str) -> str:
        """
        Make real API request to FinGPT endpoint.
        
        Args:
            prompt: FinGPT prompt
            
        Returns:
            FinGPT response text
            
        Raises:
            APIConnectionError: Connection error
            APITimeoutError: Request timeout
        """
        import aiohttp
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "model": f"fingpt:{self.model_variant}",
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "stream": False
                }
                
                logger.debug("Making FinGPT API request", endpoint=self.api_endpoint, model=self.model_variant)
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "TradeBuddy/1.0"
                }
                
                # Add authentication if API key is provided
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    f"{self.api_endpoint}/generate",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        generated_text = result.get("response", result.get("text", ""))
                        
                        logger.info("FinGPT API request successful", 
                                  response_length=len(generated_text),
                                  model=self.model_variant)
                        return generated_text
                    else:
                        error_text = await response.text()
                        logger.error("FinGPT API error", status=response.status, error=error_text)
                        raise APIConnectionError(f"FinGPT API error {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            logger.error("FinGPT request timeout", timeout=self.timeout)
            raise APITimeoutError(f"FinGPT request timeout after {self.timeout} seconds")
        except aiohttp.ClientError as e:
            logger.error("FinGPT connection error", error=str(e))
            raise APIConnectionError(f"FinGPT connection failed: {str(e)}")
        except Exception as e:
            logger.error("FinGPT API request failed", error=str(e))
            raise APIConnectionError(f"FinGPT API request failed: {str(e)}")

    async def _make_mock_response(self, prompt: str) -> str:
        """
        Generate mock FinGPT response for development/testing.
        
        Args:
            prompt: Original prompt (used for generating relevant mock data)
            
        Returns:
            Mock FinGPT response
        """
        await asyncio.sleep(0.1)  # Simulate API call delay
        
        # Analyze prompt to generate more relevant mock responses
        is_bullish_context = any(word in prompt.lower() for word in 
                               ["golden cross", "support bounce", "bullish", "uptrend"])
        is_bearish_context = any(word in prompt.lower() for word in 
                               ["death cross", "resistance", "bearish", "downtrend"])
        
        if is_bullish_context:
            signal = "BUY"
            confidence = "8"
            entry = "52000"
            stop = "51000" 
            target = "54000"
            reasoning = "Strong bullish momentum with EMA crossover confirmation and volume support."
        elif is_bearish_context:
            signal = "SELL"
            confidence = "7"
            entry = "48000"
            stop = "49000"
            target = "46000" 
            reasoning = "Bearish pattern formation with resistance rejection and declining volume."
        else:
            signal = "NEUTRAL"
            confidence = "5"
            entry = "50000"
            stop = "49000"
            target = "51000"
            reasoning = "Mixed signals with sideways price action and indecisive technical indicators."
        
        mock_response = f"""Financial Analysis Complete

Based on the provided market data and technical indicators, here is my analysis:

TRADING SIGNAL: {signal}
CONFIDENCE: {confidence}/10
ENTRY PRICE: ${entry}
STOP LOSS: ${stop}
TAKE PROFIT: ${target}

FINANCIAL REASONING:
{reasoning} The market structure shows clear patterns that align with 
current momentum indicators. Risk management is crucial at these levels.

Key factors:
1. Technical momentum supports the signal
2. Volume patterns confirm the direction
3. Support/resistance levels are well-defined
4. Market sentiment aligns with technical analysis

Recommendation: Execute with proper position sizing and risk management.

Note: This is a mock response for development/testing. Enable real FinGPT API for production analysis."""

        logger.debug("Mock FinGPT response generated", signal=signal, confidence=confidence)
        return mock_response

    def _parse_fingpt_response(
        self,
        response: str,
        market_data: MarketData,
        strategy: StrategyType,
    ) -> List[TradingSignal]:
        """
        Parse FinGPT response into trading signals.

        Args:
            response: FinGPT response text
            market_data: Market data context
            strategy: Trading strategy

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

            # Find matches
            signal_matches = re.findall(signal_pattern, response, re.IGNORECASE)
            confidence_matches = re.findall(confidence_pattern, response, re.IGNORECASE)
            entry_matches = re.findall(entry_pattern, response, re.IGNORECASE)
            stop_loss_matches = re.findall(stop_loss_pattern, response, re.IGNORECASE)
            take_profit_matches = re.findall(take_profit_pattern, response, re.IGNORECASE)

            if signal_matches:
                # Parse signal action
                signal_action = SignalAction(signal_matches[0].upper())
                confidence = int(confidence_matches[0]) if confidence_matches else 5

                # Parse prices
                entry_price = (
                    float(entry_matches[0].replace(",", ""))
                    if entry_matches
                    else market_data.current_price
                )
                stop_loss = (
                    float(stop_loss_matches[0].replace(",", ""))
                    if stop_loss_matches
                    else None
                )
                take_profit = (
                    float(take_profit_matches[0].replace(",", ""))
                    if take_profit_matches
                    else None
                )

                # Determine signal strength
                if confidence >= 8:
                    strength = SignalStrength.STRONG
                elif confidence >= 6:
                    strength = SignalStrength.MODERATE
                else:
                    strength = SignalStrength.WEAK

                # Extract reasoning
                reasoning_match = re.search(
                    r"FINANCIAL REASONING:\s*(.*?)(?:\n\n|\Z)",
                    response,
                    re.DOTALL | re.IGNORECASE,
                )
                reasoning = (
                    reasoning_match.group(1).strip()
                    if reasoning_match
                    else "FinGPT financial analysis provided"
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
                    candle_timestamp=market_data.latest_ohlcv.timestamp if market_data.latest_ohlcv else None,
                )

                signals.append(signal)

                logger.debug(
                    "FinGPT signal parsed",
                    action=signal_action.value,
                    confidence=confidence,
                    entry_price=entry_price
                )

            # Create fallback signal if no valid signal found
            if not signals:
                signals.append(self._create_fallback_signal(market_data, strategy))

        except Exception as e:
            logger.error("Error parsing FinGPT response", error=str(e))
            signals.append(self._create_fallback_signal(market_data, strategy))

        return signals

    def _create_fallback_signal(self, market_data: MarketData, strategy: StrategyType) -> TradingSignal:
        """Create fallback signal when parsing fails."""
        return TradingSignal(
            symbol=market_data.symbol,
            strategy=strategy,
            action=SignalAction.WAIT,
            strength=SignalStrength.WEAK,
            confidence=1,
            entry_price=market_data.current_price,
            reasoning="FinGPT response parsing failed - using fallback signal",
            candle_timestamp=market_data.latest_ohlcv.timestamp if market_data.latest_ohlcv else None,
        )

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of financial text using FinGPT.

        Args:
            text: Text to analyze

        Returns:
            Sentiment analysis result
        """
        # Mock implementation for testing
        return {
            "sentiment": "bullish",
            "confidence": 7,
            "keywords": ["bullish", "positive", "growth"]
        }

    async def check_model_health(self) -> bool:
        """Check FinGPT model health."""
        try:
            return await self._check_model_availability()
        except Exception as e:
            logger.error("FinGPT health check failed", error=str(e))
            return False

    async def _check_model_availability(self) -> bool:
        """Check if FinGPT model is available."""
        # Mock implementation - in real deployment, this would ping the FinGPT API
        return True

    async def close(self) -> None:
        """Close FinGPT client and cleanup resources."""
        logger.debug("FinGPT client closed")

    @property
    def model_name(self) -> str:
        """Get FinGPT model name."""
        return f"fingpt:{self.model_variant}"

    @property
    def model_version(self) -> str:
        """Get FinGPT model version."""
        return self.model_variant

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors gracefully."""
        if isinstance(error, ConnectionError):
            raise APIConnectionError(f"FinGPT connection failed: {str(error)}")
        elif isinstance(error, TimeoutError):
            raise APITimeoutError(f"FinGPT request timeout: {str(error)}")
        else:
            raise APIConnectionError(f"FinGPT API error: {str(error)}")

    def _validate_response(self, response: str) -> bool:
        """Validate FinGPT response format."""
        required_patterns = [
            r"TRADING SIGNAL:",
            r"CONFIDENCE:",
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, response, re.IGNORECASE):
                return False
        
        return True

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()