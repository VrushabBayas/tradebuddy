"""
Technical indicators for TradeBuddy.

Provides comprehensive technical analysis indicators including:
- Exponential Moving Averages (EMA)
- Support and Resistance levels
- Volume analysis
- Price action analysis
- RSI, Bollinger Bands, Fibonacci retracements
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from src.core.constants import TradingConstants
from src.core.exceptions import DataValidationError
from src.core.models import OHLCV, EMACrossover, SupportResistanceLevel

logger = structlog.get_logger(__name__)


class TechnicalIndicators:
    """
    Technical analysis indicators for trading strategies.

    Provides methods for calculating various technical indicators and
    performing comprehensive market analysis using OHLCV data.
    """

    def __init__(self):
        """Initialize the TechnicalIndicators class."""
        logger.debug("TechnicalIndicators initialized")

    def _validate_data(self, data: List[OHLCV], min_periods: int = 1) -> None:
        """
        Validate OHLCV data for analysis.

        Args:
            data: List of OHLCV data points
            min_periods: Minimum number of periods required

        Raises:
            DataValidationError: If data is invalid or insufficient
        """
        if not data:
            raise DataValidationError("No data provided for analysis")

        if len(data) < min_periods:
            raise DataValidationError(
                f"Insufficient data: {len(data)} periods provided, "
                f"minimum {min_periods} required"
            )

        # Validate data integrity
        for i, ohlcv in enumerate(data):
            if ohlcv.high < max(ohlcv.open, ohlcv.close, ohlcv.low):
                logger.warning(f"Invalid OHLCV data at index {i}: high < other values")
            if ohlcv.low > min(ohlcv.open, ohlcv.close, ohlcv.high):
                logger.warning(f"Invalid OHLCV data at index {i}: low > other values")

    def _to_dataframe(self, data: List[OHLCV]) -> pd.DataFrame:
        """Convert OHLCV data to pandas DataFrame."""
        df_data = []
        for ohlcv in data:
            df_data.append(
                {
                    "timestamp": ohlcv.timestamp,
                    "open": ohlcv.open,
                    "high": ohlcv.high,
                    "low": ohlcv.low,
                    "close": ohlcv.close,
                    "volume": ohlcv.volume,
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)
        return df

    def calculate_ema(self, data: List[OHLCV], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average.

        Args:
            data: List of OHLCV data points
            period: EMA period

        Returns:
            List of EMA values

        Raises:
            DataValidationError: If data is invalid or period is invalid
        """
        if period <= 0:
            raise DataValidationError(f"Invalid period: {period}. Must be positive.")

        self._validate_data(data, min_periods=period)

        logger.debug(f"Calculating {period}-period EMA", data_points=len(data))

        df = self._to_dataframe(data)
        ema_values = df["close"].ewm(span=period, adjust=False).mean().tolist()

        logger.debug(
            f"EMA calculation completed", period=period, values_count=len(ema_values)
        )
        return ema_values

    def calculate_sma(self, data: List[OHLCV], period: int) -> List[float]:
        """
        Calculate Simple Moving Average.

        Args:
            data: List of OHLCV data points
            period: SMA period

        Returns:
            List of SMA values
        """
        if period <= 0:
            raise DataValidationError(f"Invalid period: {period}. Must be positive.")

        self._validate_data(data, min_periods=period)

        df = self._to_dataframe(data)
        sma_values = df["close"].rolling(window=period).mean().tolist()

        return sma_values

    def detect_ema_crossover(self, data: List[OHLCV]) -> EMACrossover:
        """
        Detect EMA crossover signals.

        Args:
            data: List of OHLCV data points

        Returns:
            EMACrossover object with crossover information
        """
        self._validate_data(data, min_periods=TradingConstants.EMA_LONG_PERIOD)

        logger.debug("Detecting EMA crossover", data_points=len(data))

        # Calculate EMAs
        ema_9 = self.calculate_ema(data, TradingConstants.EMA_SHORT_PERIOD)
        ema_15 = self.calculate_ema(data, TradingConstants.EMA_LONG_PERIOD)

        # Get latest values
        latest_ema_9 = ema_9[-1]
        latest_ema_15 = ema_15[-1]

        # Determine if it's a golden cross (9 EMA > 15 EMA)
        is_golden_cross = latest_ema_9 > latest_ema_15

        # Calculate crossover strength based on the separation
        ema_separation = abs(latest_ema_9 - latest_ema_15)
        price = data[-1].close
        separation_pct = (ema_separation / price) * 100

        # Map separation percentage to strength (1-10)
        if separation_pct >= 2.0:
            strength = 10
        elif separation_pct >= 1.5:
            strength = 8
        elif separation_pct >= 1.0:
            strength = 6
        elif separation_pct >= 0.5:
            strength = 4
        else:
            strength = 2

        crossover = EMACrossover(
            ema_9=latest_ema_9,
            ema_15=latest_ema_15,
            is_golden_cross=is_golden_cross,
            crossover_strength=strength,
        )

        logger.debug(
            "EMA crossover detected",
            ema_9=latest_ema_9,
            ema_15=latest_ema_15,
            is_golden_cross=is_golden_cross,
            strength=strength,
        )

        return crossover

    def detect_support_resistance(
        self, data: List[OHLCV], min_touches: int = 2, tolerance_pct: float = 0.5
    ) -> List[SupportResistanceLevel]:
        """
        Detect support and resistance levels.

        Args:
            data: List of OHLCV data points
            min_touches: Minimum number of touches for a valid level
            tolerance_pct: Tolerance percentage for level detection

        Returns:
            List of SupportResistanceLevel objects
        """
        self._validate_data(data, min_periods=10)

        logger.debug(
            "Detecting support/resistance levels",
            data_points=len(data),
            min_touches=min_touches,
            tolerance_pct=tolerance_pct,
        )

        df = self._to_dataframe(data)
        levels = []

        # Find local highs and lows
        highs = df["high"].values
        lows = df["low"].values

        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_peaks(-lows)  # Invert to find troughs

        # Analyze resistance levels (peaks)
        resistance_levels = self._cluster_levels(
            [highs[i] for i in peaks], tolerance_pct, min_touches
        )

        for level_price, touches in resistance_levels:
            level = SupportResistanceLevel(
                level=level_price,
                strength=min(10, max(1, touches * 2)),  # Strength based on touches
                is_support=False,
                touches=touches,
                last_touch=data[
                    -1
                ].timestamp,  # Simplified - would need actual last touch
            )
            levels.append(level)

        # Analyze support levels (troughs)
        support_levels = self._cluster_levels(
            [lows[i] for i in troughs], tolerance_pct, min_touches
        )

        for level_price, touches in support_levels:
            level = SupportResistanceLevel(
                level=level_price,
                strength=min(10, max(1, touches * 2)),
                is_support=True,
                touches=touches,
                last_touch=data[-1].timestamp,
            )
            levels.append(level)

        logger.debug(f"Found {len(levels)} support/resistance levels")
        return levels

    def _find_peaks(self, values: np.ndarray, distance: int = 3) -> List[int]:
        """Find peaks in a series of values."""
        peaks = []

        for i in range(distance, len(values) - distance):
            is_peak = True

            # Check if current value is higher than surrounding values
            for j in range(i - distance, i + distance + 1):
                if j != i and values[j] >= values[i]:
                    is_peak = False
                    break

            if is_peak:
                peaks.append(i)

        return peaks

    def _cluster_levels(
        self, prices: List[float], tolerance_pct: float, min_touches: int
    ) -> List[Tuple[float, int]]:
        """Cluster similar price levels and count touches."""
        if not prices:
            return []

        clusters = []
        prices_sorted = sorted(prices)

        current_cluster = [prices_sorted[0]]

        for price in prices_sorted[1:]:
            # Check if price is within tolerance of current cluster
            cluster_avg = sum(current_cluster) / len(current_cluster)
            tolerance = cluster_avg * (tolerance_pct / 100)

            if abs(price - cluster_avg) <= tolerance:
                current_cluster.append(price)
            else:
                # Finalize current cluster if it has enough touches
                if len(current_cluster) >= min_touches:
                    cluster_level = sum(current_cluster) / len(current_cluster)
                    clusters.append((cluster_level, len(current_cluster)))

                # Start new cluster
                current_cluster = [price]

        # Don't forget the last cluster
        if len(current_cluster) >= min_touches:
            cluster_level = sum(current_cluster) / len(current_cluster)
            clusters.append((cluster_level, len(current_cluster)))

        return clusters

    def analyze_volume(self, data: List[OHLCV]) -> Dict[str, Any]:
        """
        Analyze volume patterns with enhanced confirmation thresholds.

        Args:
            data: List of OHLCV data points

        Returns:
            Dictionary with volume analysis including 110% threshold confirmation
        """
        self._validate_data(data, min_periods=5)

        logger.debug("Analyzing volume patterns", data_points=len(data))

        volumes = [ohlcv.volume for ohlcv in data]
        current_volume = volumes[-1]

        # Calculate average volume (last 20 periods or available data)
        lookback = min(TradingConstants.VOLUME_LOOKBACK_PERIOD, len(volumes))
        recent_volumes = volumes[-lookback:]
        average_volume = sum(recent_volumes) / len(recent_volumes)

        # Volume ratio
        volume_ratio = current_volume / average_volume if average_volume > 0 else 1.0

        # Calculate 20-period SMA for enhanced strategy
        volume_sma_20_periods = min(20, len(volumes))
        volume_sma_20_data = volumes[-volume_sma_20_periods:]
        volume_sma_20 = (
            sum(volume_sma_20_data) / len(volume_sma_20_data)
            if volume_sma_20_data
            else 0
        )

        # Enhanced volume confirmation (110% of 20-period SMA)
        volume_vs_sma_20 = current_volume / volume_sma_20 if volume_sma_20 > 0 else 1.0
        volume_confirmation_110pct = current_volume >= (volume_sma_20 * 1.1)

        # Volume trend
        if len(volumes) >= 3:
            recent_avg = sum(volumes[-3:]) / 3
            older_avg = sum(volumes[-6:-3]) / 3 if len(volumes) >= 6 else recent_avg

            if recent_avg > older_avg * 1.1:
                volume_trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_trend = "stable"

        analysis = {
            "current_volume": current_volume,
            "average_volume": average_volume,
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "volume_sma_20": volume_sma_20,
            "volume_vs_sma_20": volume_vs_sma_20,
            "volume_confirmation_110pct": volume_confirmation_110pct,
        }

        logger.debug("Volume analysis completed", **analysis)
        return analysis

    def analyze_price_action(self, data: List[OHLCV]) -> Dict[str, Any]:
        """
        Analyze price action patterns.

        Args:
            data: List of OHLCV data points

        Returns:
            Dictionary with price action analysis
        """
        self._validate_data(data, min_periods=10)

        logger.debug("Analyzing price action", data_points=len(data))

        df = self._to_dataframe(data)
        closes = df["close"].values

        # Trend direction based on price movement
        price_change = closes[-1] - closes[0]
        price_change_pct = (price_change / closes[0]) * 100

        if price_change_pct > 2:
            trend_direction = "bullish"
        elif price_change_pct < -2:
            trend_direction = "bearish"
        else:
            trend_direction = "sideways"

        # Trend strength based on consistency
        up_periods = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
        trend_consistency = up_periods / (len(closes) - 1)

        if trend_direction == "bearish":
            trend_consistency = 1 - trend_consistency

        trend_strength = max(1, min(10, int(trend_consistency * 10)))

        # Momentum (rate of change)
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
        else:
            momentum = 0.0

        # Volatility (standard deviation of returns)
        if len(closes) >= 2:
            returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
            ]
            volatility = np.std(returns) * 100
        else:
            volatility = 0.0

        analysis = {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "volatility": volatility,
        }

        logger.debug("Price action analysis completed", **analysis)
        return analysis

    def calculate_rsi(
        self, data: List[OHLCV], period: int = 14
    ) -> List[Optional[float]]:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: List of OHLCV data points
            period: RSI period

        Returns:
            List of RSI values
        """
        self._validate_data(data, min_periods=period + 1)

        df = self._to_dataframe(data)
        closes = df["close"]

        # Calculate price changes
        delta = closes.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi.tolist()

    def calculate_bollinger_bands(
        self, data: List[OHLCV], period: int = 20, std_dev: float = 2
    ) -> Dict[str, List[Optional[float]]]:
        """
        Calculate Bollinger Bands.

        Args:
            data: List of OHLCV data points
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Dictionary with upper, middle, and lower band values
        """
        self._validate_data(data, min_periods=period)

        df = self._to_dataframe(data)
        closes = df["close"]

        # Calculate middle band (SMA)
        middle_band = closes.rolling(window=period).mean()

        # Calculate standard deviation
        std = closes.rolling(window=period).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return {
            "upper_band": upper_band.tolist(),
            "middle_band": middle_band.tolist(),
            "lower_band": lower_band.tolist(),
        }

    def calculate_atr(
        self, data: List[OHLCV], period: int = 14
    ) -> List[Optional[float]]:
        """
        Calculate Average True Range (ATR) for volatility measurement.

        Args:
            data: List of OHLCV data points
            period: ATR period (default 14)

        Returns:
            List of ATR values
        """
        if period <= 0:
            raise DataValidationError(f"Invalid period: {period}. Must be positive.")

        self._validate_data(data, min_periods=period + 1)

        logger.debug(f"Calculating {period}-period ATR", data_points=len(data))

        df = self._to_dataframe(data)

        # Calculate True Range components
        tr1 = df["high"] - df["low"]  # Current high - current low
        tr2 = abs(df["high"] - df["close"].shift(1))  # Current high - previous close
        tr3 = abs(df["low"] - df["close"].shift(1))  # Current low - previous close

        # True Range is the maximum of the three components
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the moving average of True Range
        atr = true_range.rolling(window=period).mean()

        logger.debug(
            f"ATR calculation completed", period=period, values_count=len(atr.dropna())
        )
        return atr.tolist()

    def calculate_fibonacci_retracements(self, data: List[OHLCV]) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            data: List of OHLCV data points

        Returns:
            Dictionary with Fibonacci levels
        """
        self._validate_data(data, min_periods=2)

        df = self._to_dataframe(data)

        # Find the highest high and lowest low in the period
        highest_high = df["high"].max()
        lowest_low = df["low"].min()

        # Calculate the range
        price_range = highest_high - lowest_low

        # Calculate Fibonacci levels
        fib_levels = {
            "0.0%": highest_high,
            "23.6%": highest_high - (0.236 * price_range),
            "38.2%": highest_high - (0.382 * price_range),
            "50.0%": highest_high - (0.500 * price_range),
            "61.8%": highest_high - (0.618 * price_range),
            "78.6%": highest_high - (0.786 * price_range),
            "100.0%": lowest_low,
        }

        return fib_levels

    def detect_advanced_candlestick_patterns(self, data: List[OHLCV]) -> Dict[str, Any]:
        """
        Detect advanced candlestick patterns with precise criteria.

        Args:
            data: List of OHLCV data points

        Returns:
            Dictionary with advanced pattern analysis
        """
        self._validate_data(data, min_periods=3)

        logger.debug("Detecting advanced candlestick patterns", data_points=len(data))

        current = data[-1]
        previous = data[-2] if len(data) >= 2 else current

        # Basic candle properties
        is_bullish_candle = current.close > current.open
        is_bearish_candle = current.close < current.open
        
        # Calculate precise measurements
        body_size = abs(current.close - current.open)
        candle_range = current.high - current.low
        upper_shadow = current.high - max(current.open, current.close)
        lower_shadow = min(current.open, current.close) - current.low
        
        # Avoid division by zero
        body_ratio = body_size / candle_range if candle_range > 0 else 0
        upper_shadow_ratio = upper_shadow / body_size if body_size > 0 else 0
        lower_shadow_ratio = lower_shadow / body_size if body_size > 0 else 0
        upper_shadow_pct = upper_shadow / candle_range if candle_range > 0 else 0
        lower_shadow_pct = lower_shadow / candle_range if candle_range > 0 else 0

        # Advanced pattern detection
        patterns_detected = []
        pattern_strength = 0  # 1-10 scale
        pattern_type = "none"
        signal_direction = "neutral"

        # BEARISH MARABOZU - Most bearish pattern
        if (is_bearish_candle and 
            body_ratio >= 0.95 and  # Body is 95%+ of candle
            upper_shadow_pct <= 0.02 and  # Upper shadow ≤2% of range
            lower_shadow_pct <= 0.02):  # Lower shadow ≤2% of range
            patterns_detected.append("bearish_marabozu")
            pattern_strength = 10
            pattern_type = "bearish_marabozu"
            signal_direction = "strong_bearish"

        # SHOOTING STAR - Second most bearish pattern  
        elif (upper_shadow_ratio >= 2.0 and  # Upper shadow ≥2x body
              lower_shadow_ratio <= 0.5 and  # Lower shadow ≤50% of body
              body_ratio <= 0.3):  # Small body (≤30% of range)
            patterns_detected.append("shooting_star")
            pattern_strength = 9
            pattern_type = "shooting_star"
            signal_direction = "strong_bearish"

        # BULLISH MARABOZU - Most bullish pattern
        elif (is_bullish_candle and
              body_ratio >= 0.95 and  # Body is 95%+ of candle  
              upper_shadow_pct <= 0.02 and  # Upper shadow ≤2% of range
              lower_shadow_pct <= 0.02):  # Lower shadow ≤2% of range
            patterns_detected.append("bullish_marabozu")
            pattern_strength = 10
            pattern_type = "bullish_marabozu"
            signal_direction = "strong_bullish"

        # ENHANCED HAMMER - Bullish reversal at support
        elif (lower_shadow_ratio >= 2.0 and  # Lower shadow ≥2x body
              upper_shadow_ratio <= 0.5 and  # Upper shadow ≤50% of body
              body_ratio <= 0.3):  # Small body
            patterns_detected.append("hammer")
            pattern_strength = 7
            pattern_type = "hammer"
            signal_direction = "bullish"

        # ENHANCED DOJI - Indecision
        elif body_ratio <= 0.05:  # Body ≤5% of range
            patterns_detected.append("doji")
            pattern_strength = 3
            pattern_type = "doji"
            signal_direction = "neutral"

        # STRONG BULLISH CANDLE - Enhanced detection
        elif (is_bullish_candle and body_ratio >= 0.7):
            patterns_detected.append("strong_bullish")
            pattern_strength = 8
            pattern_type = "strong_bullish"
            signal_direction = "bullish"

        # STRONG BEARISH CANDLE - Enhanced detection  
        elif (is_bearish_candle and body_ratio >= 0.7):
            patterns_detected.append("strong_bearish")
            pattern_strength = 8
            pattern_type = "strong_bearish"
            signal_direction = "bearish"

        # MODERATE PATTERNS
        elif is_bullish_candle and body_ratio >= 0.4:
            patterns_detected.append("moderate_bullish")
            pattern_strength = 5
            pattern_type = "moderate_bullish"
            signal_direction = "bullish"
        elif is_bearish_candle and body_ratio >= 0.4:
            patterns_detected.append("moderate_bearish")
            pattern_strength = 5
            pattern_type = "moderate_bearish"
            signal_direction = "bearish"
        else:
            pattern_strength = 2
            pattern_type = "indecisive"
            signal_direction = "neutral"

        # Calculate trend context for pattern validation
        trend_context = self._analyze_trend_context(data)
        
        # Adjust pattern strength based on trend alignment
        if signal_direction == "strong_bearish" and trend_context == "downtrend":
            pattern_strength = min(10, pattern_strength + 1)  # Boost bearish patterns in downtrend
        elif signal_direction == "strong_bullish" and trend_context == "uptrend":
            pattern_strength = min(10, pattern_strength + 1)  # Boost bullish patterns in uptrend
        elif signal_direction != "neutral" and trend_context == "sideways":
            pattern_strength = max(1, pattern_strength - 1)  # Reduce strength in sideways markets

        analysis = {
            "patterns_detected": patterns_detected,
            "primary_pattern": pattern_type,
            "pattern_strength": pattern_strength,
            "signal_direction": signal_direction,
            "body_ratio": body_ratio,
            "upper_shadow_ratio": upper_shadow_ratio,
            "lower_shadow_ratio": lower_shadow_ratio,
            "trend_context": trend_context,
            "is_bullish": is_bullish_candle,
            "is_bearish": is_bearish_candle,
            "body_size": body_size,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "candle_range": candle_range,
        }

        logger.debug(
            "Advanced pattern detection completed",
            primary_pattern=pattern_type,
            strength=pattern_strength,
            signal=signal_direction,
            trend_context=trend_context,
        )
        return analysis

    def _analyze_trend_context(self, data: List[OHLCV]) -> str:
        """
        Analyze recent trend context for pattern validation.

        Args:
            data: List of OHLCV data points

        Returns:
            Trend context: 'uptrend', 'downtrend', or 'sideways'
        """
        if len(data) < 5:
            return "sideways"

        # Use last 5 closes to determine short-term trend
        recent_closes = [candle.close for candle in data[-5:]]
        
        # Calculate trend slope using linear regression
        x = np.arange(len(recent_closes))
        slope = np.polyfit(x, recent_closes, 1)[0]
        
        # Calculate percentage change over period
        price_change_pct = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0]) * 100
        
        # Determine trend based on slope and price change
        if slope > 0 and price_change_pct > 1.0:
            return "uptrend"
        elif slope < 0 and price_change_pct < -1.0:
            return "downtrend"
        else:
            return "sideways"

    def analyze_candlestick_confirmation(self, data: List[OHLCV]) -> Dict[str, Any]:
        """
        Analyze candlestick patterns for trend confirmation (backward compatibility).

        Args:
            data: List of OHLCV data points

        Returns:
            Dictionary with candlestick analysis
        """
        self._validate_data(data, min_periods=2)

        logger.debug("Analyzing candlestick patterns", data_points=len(data))

        # Use advanced pattern detection
        advanced_analysis = self.detect_advanced_candlestick_patterns(data)
        
        # Map to legacy format for backward compatibility
        signal_direction = advanced_analysis["signal_direction"]
        pattern_strength = advanced_analysis["pattern_strength"]
        
        # Map signal direction to confirmation
        if signal_direction in ["strong_bullish", "bullish"]:
            confirmation = "bullish"
        elif signal_direction in ["strong_bearish", "bearish"]:
            confirmation = "bearish"
        else:
            confirmation = "neutral"
            
        # Map pattern strength to legacy strength categories
        if pattern_strength >= 8:
            strength = "strong"
        elif pattern_strength >= 5:
            strength = "moderate"
        else:
            strength = "weak"

        # Enhanced pattern detection for specific patterns
        current = data[-1]
        body_size = abs(current.close - current.open)
        upper_shadow = current.high - max(current.open, current.close)
        lower_shadow = min(current.open, current.close) - current.low

        # Check for specific patterns using advanced detection
        is_hammer = "hammer" in advanced_analysis["patterns_detected"]
        is_shooting_star = "shooting_star" in advanced_analysis["patterns_detected"]
        is_doji = "doji" in advanced_analysis["patterns_detected"]
        
        # Enhanced engulfing pattern detection
        is_engulfing = False
        if len(data) >= 2:
            previous = data[-2]
            is_bullish_candle = current.close > current.open
            is_bearish_candle = current.close < current.open
            prev_body_size = abs(previous.close - previous.open)
            
            is_engulfing = (
                body_size > prev_body_size * 1.1
                and (
                    (
                        is_bullish_candle
                        and previous.close < previous.open
                        and current.close > previous.open
                        and current.open < previous.close
                    )
                    or (
                        is_bearish_candle
                        and previous.close > previous.open
                        and current.close < previous.open
                        and current.open > previous.close
                    )
                )
            )

        # Legacy format with enhanced data
        analysis = {
            "confirmation": confirmation,
            "pattern": advanced_analysis["primary_pattern"],
            "strength": strength,
            "body_ratio": advanced_analysis["body_ratio"],
            "is_bullish": advanced_analysis["is_bullish"],
            "is_bearish": advanced_analysis["is_bearish"],
            "is_doji": is_doji,
            "is_hammer": is_hammer,
            "is_shooting_star": is_shooting_star,
            "is_engulfing": is_engulfing,
            "body_size": body_size,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            # Enhanced fields
            "pattern_strength_score": pattern_strength,
            "signal_direction": signal_direction,
            "patterns_detected": advanced_analysis["patterns_detected"],
            "trend_context": advanced_analysis["trend_context"],
        }

        logger.debug(
            "Candlestick analysis completed",
            confirmation=confirmation,
            pattern=advanced_analysis["primary_pattern"],
            strength=strength,
            pattern_strength_score=pattern_strength,
        )
        return analysis

    def comprehensive_analysis(self, data: List[OHLCV]) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.

        Args:
            data: List of OHLCV data points

        Returns:
            Dictionary with complete analysis results
        """
        self._validate_data(data, min_periods=TradingConstants.EMA_LONG_PERIOD)

        logger.info(
            "Performing comprehensive technical analysis", data_points=len(data)
        )

        try:
            # Perform all analyses
            ema_crossover = self.detect_ema_crossover(data)
            support_resistance = self.detect_support_resistance(data)
            volume_analysis = self.analyze_volume(data)
            price_action = self.analyze_price_action(data)

            # Determine overall sentiment
            sentiment_signals = []

            # EMA signal
            if ema_crossover.is_golden_cross:
                sentiment_signals.append("bullish")
            else:
                sentiment_signals.append("bearish")

            # Price action signal
            sentiment_signals.append(price_action["trend_direction"])

            # Volume confirmation
            if volume_analysis["volume_ratio"] > 1.2:  # Above average volume
                if price_action["trend_direction"] == "bullish":
                    sentiment_signals.append("bullish")
                elif price_action["trend_direction"] == "bearish":
                    sentiment_signals.append("bearish")

            # Count sentiment signals
            bullish_count = sentiment_signals.count("bullish")
            bearish_count = sentiment_signals.count("bearish")
            neutral_count = sentiment_signals.count("sideways")

            # Determine overall sentiment
            if bullish_count > bearish_count + neutral_count:
                overall_sentiment = "bullish"
            elif bearish_count > bullish_count + neutral_count:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"

            # Calculate confidence score
            total_signals = len(sentiment_signals)
            dominant_count = max(bullish_count, bearish_count, neutral_count)
            confidence_ratio = (
                dominant_count / total_signals if total_signals > 0 else 0
            )

            # Factor in trend strength and crossover strength
            trend_factor = price_action["trend_strength"] / 10
            crossover_factor = ema_crossover.crossover_strength / 10

            confidence_score = min(
                10,
                max(
                    1,
                    int(
                        (
                            confidence_ratio * 0.5
                            + trend_factor * 0.3
                            + crossover_factor * 0.2
                        )
                        * 10
                    ),
                ),
            )

            analysis = {
                "ema_crossover": ema_crossover,
                "support_resistance": support_resistance,
                "volume_analysis": volume_analysis,
                "price_action": price_action,
                "overall_sentiment": overall_sentiment,
                "confidence_score": confidence_score,
            }

            logger.info(
                "Comprehensive analysis completed",
                sentiment=overall_sentiment,
                confidence=confidence_score,
                indicators_count=4,
            )

            return analysis

        except Exception as e:
            logger.error("Error in comprehensive analysis", error=str(e))
            raise DataValidationError(f"Analysis failed: {str(e)}")
    
    def create_candlestick_formation(self, data: List[OHLCV]) -> Optional:
        """
        Create a CandlestickFormation object from the latest candle analysis.
        
        Args:
            data: List of OHLCV data points
            
        Returns:
            CandlestickFormation object or None if no clear pattern
        """
        try:
            from src.core.models import CandlestickFormation
            
            # Get advanced pattern analysis
            pattern_analysis = self.detect_advanced_candlestick_patterns(data)
            
            if not pattern_analysis or pattern_analysis.get("primary_pattern") == "none":
                return None
            
            current = data[-1]
            
            # Generate visual description
            visual_desc = self._generate_visual_description(
                current, 
                pattern_analysis["primary_pattern"],
                pattern_analysis["body_ratio"],
                pattern_analysis["upper_shadow_ratio"],
                pattern_analysis["lower_shadow_ratio"]
            )
            
            # Determine pattern type
            signal_direction = pattern_analysis["signal_direction"]
            if signal_direction in ["strong_bullish", "bullish"]:
                pattern_type = "bullish"
            elif signal_direction in ["strong_bearish", "bearish"]:
                pattern_type = "bearish"
            else:
                pattern_type = "neutral"
            
            return CandlestickFormation(
                pattern_name=pattern_analysis["primary_pattern"],
                pattern_type=pattern_type,
                strength=pattern_analysis["pattern_strength"],
                signal_direction=signal_direction,
                body_ratio=pattern_analysis["body_ratio"],
                upper_shadow_ratio=pattern_analysis["upper_shadow_ratio"],
                lower_shadow_ratio=pattern_analysis["lower_shadow_ratio"],
                visual_description=visual_desc,
                trend_context=pattern_analysis["trend_context"],
                volume_confirmation=False  # Will be set by strategy
            )
            
        except Exception as e:
            logger.warning("Failed to create candlestick formation", error=str(e))
            return None
    
    def _generate_visual_description(
        self, 
        candle: OHLCV, 
        pattern_name: str, 
        body_ratio: float,
        upper_shadow_ratio: float,
        lower_shadow_ratio: float
    ) -> str:
        """Generate human-readable visual description of candle."""
        
        # Determine candle color and size
        is_bullish = candle.close > candle.open
        color = "green" if is_bullish else "red"
        
        # Body size description
        if body_ratio >= 0.8:
            body_desc = "large"
        elif body_ratio >= 0.5:
            body_desc = "medium"
        elif body_ratio >= 0.2:
            body_desc = "small"
        else:
            body_desc = "tiny"
        
        # Shadow description
        shadow_desc = ""
        if upper_shadow_ratio >= 2.0:
            shadow_desc += "long upper shadow"
        elif upper_shadow_ratio >= 1.0:
            shadow_desc += "medium upper shadow"
        
        if lower_shadow_ratio >= 2.0:
            if shadow_desc:
                shadow_desc += " and long lower shadow"
            else:
                shadow_desc += "long lower shadow"
        elif lower_shadow_ratio >= 1.0:
            if shadow_desc:
                shadow_desc += " and medium lower shadow"
            else:
                shadow_desc += "medium lower shadow"
        
        if not shadow_desc:
            shadow_desc = "minimal shadows"
        
        # Special pattern descriptions
        pattern_descriptions = {
            "bearish_marabozu": f"Strong {color} candle with virtually no shadows - maximum bearish pressure",
            "bullish_marabozu": f"Strong {color} candle with virtually no shadows - maximum bullish pressure",
            "shooting_star": f"Small body with long upper shadow - potential reversal signal",
            "hammer": f"Small body with long lower shadow - potential reversal signal",
            "doji": f"Very small body indicating market indecision",
        }
        
        if pattern_name in pattern_descriptions:
            return pattern_descriptions[pattern_name]
        
        # Default description
        return f"{body_desc.title()} {color} body with {shadow_desc}"
    
    def generate_pattern_reasoning(
        self, 
        formation: "CandlestickFormation", 
        current_price: float,
        trend_context: str = "neutral"
    ) -> str:
        """
        Generate detailed educational reasoning about the candlestick pattern.
        
        Args:
            formation: CandlestickFormation object
            current_price: Current market price
            trend_context: Current trend context (uptrend/downtrend/sideways)
            
        Returns:
            Detailed reasoning about the pattern's significance
        """
        pattern_reasoning = {
            "bearish_marabozu": {
                "description": "A Bearish Marabozu indicates maximum selling pressure with no buying support.",
                "psychology": "Sellers controlled the entire session, opening at the high and closing at the low.",
                "reliability": "High reliability pattern, especially effective in downtrends.",
                "trading_context": "Strong continuation signal in downtrends, reversal signal after uptrends."
            },
            "bullish_marabozu": {
                "description": "A Bullish Marabozu shows maximum buying pressure with no selling resistance.",
                "psychology": "Buyers dominated completely, opening at the low and closing at the high.",
                "reliability": "High reliability pattern, particularly strong in uptrends.",
                "trading_context": "Strong continuation signal in uptrends, reversal signal after downtrends."
            },
            "shooting_star": {
                "description": "A Shooting Star indicates potential trend reversal with failed upward attempt.",
                "psychology": "Bulls pushed price higher but bears rejected the advance, closing near the open.",
                "reliability": "Moderate reliability, requires confirmation from next candle.",
                "trading_context": "Most effective at resistance levels or after strong uptrends."
            },
            "hammer": {
                "description": "A Hammer suggests potential bullish reversal with strong support found.",
                "psychology": "Bears drove price down but bulls regained control, closing near the high.",
                "reliability": "Moderate to high reliability, especially at support levels.",
                "trading_context": "Most effective after downtrends or at key support levels."
            },
            "doji": {
                "description": "A Doji represents market indecision with equal buying and selling pressure.",
                "psychology": "Neither bulls nor bears could gain control, suggesting potential reversal.",
                "reliability": "Low to moderate reliability, heavily dependent on context.",
                "trading_context": "Most significant after strong trends or at key support/resistance levels."
            }
        }
        
        pattern_name = formation.pattern_name
        if pattern_name not in pattern_reasoning:
            return f"Standard candle formation with {formation.pattern_type} bias."
        
        info = pattern_reasoning[pattern_name]
        
        # Build comprehensive reasoning
        reasoning = f"{info['description']} "
        reasoning += f"{info['psychology']} "
        
        # Add trend context
        if trend_context == "uptrend":
            if "reversal" in info['trading_context'] and "uptrend" in info['trading_context']:
                reasoning += f"In the current uptrend, this pattern suggests potential bearish reversal. "
            elif "continuation" in info['trading_context'] and "uptrend" in info['trading_context']:
                reasoning += f"In the current uptrend, this pattern supports continued bullish momentum. "
        elif trend_context == "downtrend":
            if "reversal" in info['trading_context'] and "downtrend" in info['trading_context']:
                reasoning += f"In the current downtrend, this pattern suggests potential bullish reversal. "
            elif "continuation" in info['trading_context'] and "downtrend" in info['trading_context']:
                reasoning += f"In the current downtrend, this pattern supports continued bearish momentum. "
        
        # Add reliability assessment
        reasoning += f"Pattern reliability: {info['reliability']} "
        
        # Add strength-based context
        if formation.strength >= 8:
            reasoning += "This is a very strong pattern formation with clear defined characteristics."
        elif formation.strength >= 6:
            reasoning += "This is a moderate strength pattern with decent reliability."
        else:
            reasoning += "This is a weak pattern that should be confirmed with other indicators."
        
        # Add volume context if available
        if formation.volume_confirmation:
            reasoning += " Volume confirms the pattern, increasing its reliability."
        else:
            reasoning += " Volume does not confirm the pattern, reducing its reliability."
        
        return reasoning
