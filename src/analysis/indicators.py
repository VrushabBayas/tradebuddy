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
        # Filter out zero volumes for more meaningful average
        lookback = min(TradingConstants.VOLUME_LOOKBACK_PERIOD, len(volumes))
        recent_volumes = volumes[-lookback:]
        non_zero_volumes = [v for v in recent_volumes if v > 0]
        
        if non_zero_volumes:
            average_volume = sum(non_zero_volumes) / len(non_zero_volumes)
        else:
            # Fallback: use all volumes if all are zero
            average_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1.0

        # Volume ratio (handle zero current volume)
        if current_volume > 0 and average_volume > 0:
            volume_ratio = current_volume / average_volume
        elif current_volume == 0 and average_volume > 0:
            volume_ratio = 0.1  # Treat zero volume as very low activity
        else:
            volume_ratio = 1.0  # Neutral when both are zero or average is zero

        # Calculate 20-period SMA for enhanced strategy (filter out zeros)
        volume_sma_20_periods = min(20, len(volumes))
        volume_sma_20_data = volumes[-volume_sma_20_periods:]
        non_zero_sma_data = [v for v in volume_sma_20_data if v > 0]
        
        if non_zero_sma_data:
            volume_sma_20 = sum(non_zero_sma_data) / len(non_zero_sma_data)
        else:
            # Fallback: use all data if all are zero
            volume_sma_20 = sum(volume_sma_20_data) / len(volume_sma_20_data) if volume_sma_20_data else 1.0

        # Enhanced volume confirmation (110% of 20-period SMA)
        if volume_sma_20 > 0:
            volume_vs_sma_20 = current_volume / volume_sma_20
            volume_confirmation_110pct = current_volume >= (volume_sma_20 * 1.1)
        else:
            # When SMA is zero, can't confirm volume, so assume neutral
            volume_vs_sma_20 = 1.0
            volume_confirmation_110pct = False

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

    def calculate_body_significance(
        self, 
        current_candle: OHLCV, 
        atr_value: float, 
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Calculate candlestick body significance using multi-factor analysis.
        
        Combines timeframe-specific point requirements, percentage analysis,
        and ATR-based volatility assessment to determine if a candlestick
        has sufficient body size for directional bias classification.
        
        Args:
            current_candle: Current OHLCV candle data
            atr_value: Current ATR value for volatility context
            timeframe: Trading timeframe (e.g., "1h", "1d")
            
        Returns:
            Dictionary with body significance analysis
        """
        from src.core.constants import CandlestickConstants
        
        # Basic measurements
        body_size = abs(current_candle.close - current_candle.open)
        candle_range = current_candle.high - current_candle.low
        
        # Avoid division by zero
        body_ratio = body_size / candle_range if candle_range > 0 else 0
        atr_ratio = body_size / atr_value if atr_value > 0 else 0
        
        # Initialize result
        result = {
            "body_size": body_size,
            "body_ratio": body_ratio,
            "atr_ratio": atr_ratio,
            "candle_range": candle_range,
            "classification": "unknown",
            "allow_directional_bias": False,
            "confidence_boost": 1.0,
            "is_doji": False,
            "is_spinning_top": False,
            "is_significant_body": False,
            "reasoning": ""
        }
        
        # 1. Always check for doji first (overrides everything)
        if body_ratio <= (CandlestickConstants.DOJI_BODY_THRESHOLD_PCT / 100):
            result.update({
                "classification": "doji",
                "allow_directional_bias": False,
                "is_doji": True,
                "reasoning": f"Doji pattern: body only {body_ratio:.1%} of candle range"
            })
            return result
        
        # 2. Check for spinning top
        if body_ratio <= (CandlestickConstants.SPINNING_TOP_THRESHOLD_PCT / 100):
            result.update({
                "classification": "spinning_top",
                "allow_directional_bias": False,
                "is_spinning_top": True,
                "reasoning": f"Spinning top: body {body_ratio:.1%} of range with long shadows"
            })
            return result
        
        # 3. Apply timeframe-specific point requirements
        timeframe_config = CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS.get(timeframe)
        
        if timeframe_config:
            min_points = timeframe_config["min"]
            max_points = timeframe_config["max"]
            
            if body_size < min_points:
                result.update({
                    "classification": "insufficient_body",
                    "allow_directional_bias": False,
                    "reasoning": f"Body too small ({body_size:.0f} points < {min_points} min for {timeframe})"
                })
                return result
            elif body_size > max_points:
                result.update({
                    "classification": "exceptional_body",
                    "allow_directional_bias": True,
                    "confidence_boost": CandlestickConstants.EXCEPTIONAL_BODY_CONFIDENCE_BOOST,
                    "is_significant_body": True,
                    "reasoning": f"Exceptional body ({body_size:.0f} points > {max_points} max for {timeframe})"
                })
                return result
            else:
                result.update({
                    "classification": "significant_body",
                    "allow_directional_bias": True,
                    "is_significant_body": True,
                    "reasoning": f"Significant body ({body_size:.0f} points within {min_points}-{max_points} range for {timeframe})"
                })
                return result
        else:
            # 4. Fallback for unknown timeframes using ATR
            return self._atr_based_body_classification(result, body_size, atr_value, atr_ratio, timeframe)
    
    def _atr_based_body_classification(
        self, 
        result: Dict[str, Any], 
        body_size: float, 
        atr_value: float, 
        atr_ratio: float, 
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Fallback body classification using ATR when timeframe is unknown.
        
        Args:
            result: Existing result dictionary to update
            body_size: Calculated body size
            atr_value: Current ATR value
            atr_ratio: Body size / ATR ratio
            timeframe: Trading timeframe
            
        Returns:
            Updated result dictionary
        """
        from src.core.constants import CandlestickConstants
        
        # Try to interpolate from known timeframes first
        interpolated_config = self._interpolate_timeframe_config(timeframe)
        if interpolated_config:
            min_points = interpolated_config["min"]
            max_points = interpolated_config["max"]
            
            if body_size < min_points:
                result.update({
                    "classification": "insufficient_body",
                    "allow_directional_bias": False,
                    "reasoning": f"Body too small ({body_size:.0f} < {min_points} interpolated min for {timeframe})"
                })
            elif body_size > max_points:
                result.update({
                    "classification": "exceptional_body",
                    "allow_directional_bias": True,
                    "confidence_boost": CandlestickConstants.EXCEPTIONAL_BODY_CONFIDENCE_BOOST,
                    "is_significant_body": True,
                    "reasoning": f"Exceptional body ({body_size:.0f} > {max_points} interpolated max for {timeframe})"
                })
            else:
                result.update({
                    "classification": "significant_body",
                    "allow_directional_bias": True,
                    "is_significant_body": True,
                    "reasoning": f"Significant body ({body_size:.0f} within interpolated {min_points}-{max_points} for {timeframe})"
                })
            return result
        
        # Pure ATR-based fallback
        if atr_ratio < CandlestickConstants.ATR_TINY_BODY_MULTIPLIER:
            result.update({
                "classification": "tiny_body",
                "allow_directional_bias": False,
                "reasoning": f"Tiny body ({atr_ratio:.1f}x ATR < {CandlestickConstants.ATR_TINY_BODY_MULTIPLIER}x threshold)"
            })
        elif atr_ratio < CandlestickConstants.ATR_SMALL_BODY_MULTIPLIER:
            result.update({
                "classification": "small_body",
                "allow_directional_bias": False,
                "reasoning": f"Small body ({atr_ratio:.1f}x ATR < {CandlestickConstants.ATR_SMALL_BODY_MULTIPLIER}x threshold)"
            })
        elif atr_ratio >= CandlestickConstants.ATR_SIGNIFICANT_BODY_MULTIPLIER:
            result.update({
                "classification": "significant_body",
                "allow_directional_bias": True,
                "is_significant_body": True,
                "reasoning": f"Significant body ({atr_ratio:.1f}x ATR ≥ {CandlestickConstants.ATR_SIGNIFICANT_BODY_MULTIPLIER}x threshold)"
            })
        else:
            result.update({
                "classification": "neutral_body",
                "allow_directional_bias": False,
                "reasoning": f"Neutral body ({atr_ratio:.1f}x ATR between thresholds)"
            })
        
        return result
    
    def _interpolate_timeframe_config(self, timeframe: str) -> Optional[Dict[str, int]]:
        """
        Interpolate timeframe configuration for unknown timeframes.
        
        Args:
            timeframe: Trading timeframe (e.g., "2h", "8h")
            
        Returns:
            Interpolated config or None if interpolation not possible
        """
        from src.core.constants import CandlestickConstants
        
        # Extract numeric value and unit
        if timeframe.endswith('m'):
            try:
                minutes = int(timeframe[:-1])
                # Interpolate between minute timeframes
                if 1 <= minutes <= 5:
                    return self._linear_interpolate(
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["1m"],
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["5m"],
                        (minutes - 1) / 4
                    )
                elif 5 < minutes <= 15:
                    return self._linear_interpolate(
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["5m"],
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["15m"],
                        (minutes - 5) / 10
                    )
                elif 15 < minutes <= 30:
                    return self._linear_interpolate(
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["15m"],
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["30m"],
                        (minutes - 15) / 15
                    )
            except ValueError:
                pass
        elif timeframe.endswith('h'):
            try:
                hours = int(timeframe[:-1])
                # Interpolate between hour timeframes
                if 1 <= hours <= 4:
                    return self._linear_interpolate(
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["1h"],
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["4h"],
                        (hours - 1) / 3
                    )
                elif 4 < hours <= 24:
                    return self._linear_interpolate(
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["4h"],
                        CandlestickConstants.TIMEFRAME_MIN_BODY_POINTS["1d"],
                        (hours - 4) / 20
                    )
            except ValueError:
                pass
        
        return None
    
    def _linear_interpolate(self, config1: Dict[str, int], config2: Dict[str, int], ratio: float) -> Dict[str, int]:
        """
        Linear interpolation between two timeframe configs.
        
        Args:
            config1: First timeframe config
            config2: Second timeframe config  
            ratio: Interpolation ratio (0.0 to 1.0)
            
        Returns:
            Interpolated configuration
        """
        return {
            "min": int(config1["min"] + (config2["min"] - config1["min"]) * ratio),
            "max": int(config1["max"] + (config2["max"] - config1["max"]) * ratio)
        }

    def detect_advanced_candlestick_patterns(
        self, 
        data: List[OHLCV], 
        timeframe: str = "1h", 
        atr_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect advanced candlestick patterns with timeframe-adaptive body significance analysis.

        Args:
            data: List of OHLCV data points
            timeframe: Trading timeframe for body significance analysis (default: "1h")
            atr_value: Current ATR value for volatility context (optional)

        Returns:
            Dictionary with advanced pattern analysis including body significance
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

        # Calculate ATR for body significance analysis if not provided
        if atr_value is None:
            try:
                atr_values = self.calculate_atr(data, period=14)
                atr_value = atr_values[-1] if atr_values else body_size  # Fallback to body_size
            except Exception:
                atr_value = body_size  # Safe fallback

        # Perform body significance analysis
        body_analysis = self.calculate_body_significance(current, atr_value, timeframe)
        
        logger.debug(
            "Body significance analysis", 
            body_size=body_analysis["body_size"],
            classification=body_analysis["classification"],
            allow_directional_bias=body_analysis["allow_directional_bias"],
            reasoning=body_analysis["reasoning"]
        )

        # Advanced pattern detection
        patterns_detected = []
        pattern_strength = 0  # 1-10 scale
        pattern_type = "none"
        signal_direction = "neutral"

        # Gate directional patterns behind body significance analysis
        if not body_analysis["allow_directional_bias"]:
            # Force neutral classification for insufficient bodies
            if body_analysis["is_doji"]:
                pattern_type = "doji"
                signal_direction = "doji"
                pattern_strength = 3
                patterns_detected.append("doji")
            elif body_analysis["is_spinning_top"]:
                pattern_type = "spinning_top"
                signal_direction = "spinning_top"
                pattern_strength = 4
                patterns_detected.append("spinning_top")
            else:
                pattern_type = "insufficient_body"
                signal_direction = "neutral"
                pattern_strength = 1
                patterns_detected.append("insufficient_body")
        else:
            # Apply exceptional body confidence boost if applicable
            confidence_multiplier = body_analysis.get("confidence_boost", 1.0)

            # BEARISH MARABOZU - Most bearish pattern
            if (is_bearish_candle and 
                body_ratio >= 0.95 and  # Body is 95%+ of candle
                upper_shadow_pct <= 0.02 and  # Upper shadow ≤2% of range
                lower_shadow_pct <= 0.02):  # Lower shadow ≤2% of range
                patterns_detected.append("bearish_marabozu")
                pattern_strength = int(10 * confidence_multiplier)
                pattern_type = "bearish_marabozu"
                signal_direction = "strong_bearish"

            # SHOOTING STAR - Second most bearish pattern  
            elif (upper_shadow_ratio >= 2.0 and  # Upper shadow ≥2x body
                  lower_shadow_ratio <= 0.5 and  # Lower shadow ≤50% of body
                  body_ratio <= 0.3):  # Small body (≤30% of range)
                patterns_detected.append("shooting_star")
                pattern_strength = int(9 * confidence_multiplier)
                pattern_type = "shooting_star"
                signal_direction = "strong_bearish"

            # BULLISH MARABOZU - Most bullish pattern
            elif (is_bullish_candle and
                  body_ratio >= 0.95 and  # Body is 95%+ of candle  
                  upper_shadow_pct <= 0.02 and  # Upper shadow ≤2% of range
                  lower_shadow_pct <= 0.02):  # Lower shadow ≤2% of range
                patterns_detected.append("bullish_marabozu")
                pattern_strength = int(10 * confidence_multiplier)
                pattern_type = "bullish_marabozu"
                signal_direction = "strong_bullish"

            # ENHANCED HAMMER - Bullish reversal at support
            elif (lower_shadow_ratio >= 2.0 and  # Lower shadow ≥2x body
                  upper_shadow_ratio <= 0.5 and  # Upper shadow ≤50% of body
                  body_ratio <= 0.3):  # Small body
                patterns_detected.append("hammer")
                pattern_strength = int(7 * confidence_multiplier)
                pattern_type = "hammer"
                signal_direction = "bullish"

            # STRONG BULLISH CANDLE - Enhanced detection (ONLY for bullish candles)
            elif (is_bullish_candle and body_ratio >= 0.7):
                patterns_detected.append("strong_bullish")
                pattern_strength = int(8 * confidence_multiplier)
                pattern_type = "strong_bullish"
                signal_direction = "strong_bullish"

            # STRONG BEARISH CANDLE - Enhanced detection (ONLY for bearish candles)
            elif (is_bearish_candle and body_ratio >= 0.7):
                patterns_detected.append("strong_bearish")
                pattern_strength = int(8 * confidence_multiplier)
                pattern_type = "strong_bearish"
                signal_direction = "strong_bearish"

            # MODERATE PATTERNS (with strict candle type validation)
            elif is_bullish_candle and body_ratio >= 0.4:
                patterns_detected.append("moderate_bullish")
                pattern_strength = int(5 * confidence_multiplier)
                pattern_type = "moderate_bullish"
                signal_direction = "bullish"
            elif is_bearish_candle and body_ratio >= 0.4:
                patterns_detected.append("moderate_bearish")
                pattern_strength = int(5 * confidence_multiplier)
                pattern_type = "moderate_bearish"
                signal_direction = "bearish"
            # WEAK PATTERNS (for small bodies)
            elif is_bullish_candle and body_ratio >= 0.1:
                patterns_detected.append("weak_bullish")
                pattern_strength = int(3 * confidence_multiplier)
                pattern_type = "weak_bullish"
                signal_direction = "bullish"
            elif is_bearish_candle and body_ratio >= 0.1:
                patterns_detected.append("weak_bearish")
                pattern_strength = int(3 * confidence_multiplier)
                pattern_type = "weak_bearish"
                signal_direction = "bearish"
            else:
                # For very small bodies or equal open/close (but significant enough to allow directional bias)
                if is_bullish_candle:
                    pattern_strength = int(2 * confidence_multiplier)
                    pattern_type = "weak_bullish"
                    signal_direction = "weak_bullish"
                elif is_bearish_candle:
                    pattern_strength = int(2 * confidence_multiplier)
                    pattern_type = "weak_bearish"
                    signal_direction = "weak_bearish"
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
            # Body significance analysis data
            "body_significance": body_analysis,
            "timeframe": timeframe,
            "atr_value": atr_value,
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

        # Use advanced pattern detection (default to 1h timeframe for backward compatibility)
        advanced_analysis = self.detect_advanced_candlestick_patterns(data, timeframe="1h")
        
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
            
            # Get advanced pattern analysis (default to 1h timeframe for backward compatibility)
            pattern_analysis = self.detect_advanced_candlestick_patterns(data, timeframe="1h")
            
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
        
        # Determine candle color and size (validate actual candle direction)
        is_bullish = candle.close > candle.open
        is_bearish = candle.close < candle.open
        
        # Be explicit about color determination
        if is_bullish:
            color = "green"
            candle_direction = "bullish"
        elif is_bearish:
            color = "red"
            candle_direction = "bearish"
        else:
            color = "neutral"  # Doji case where close == open
            candle_direction = "neutral"
        
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
        
        # Special pattern descriptions (validate pattern matches candle direction)
        pattern_descriptions = {
            "bearish_marabozu": f"Strong {color} candle with virtually no shadows - maximum bearish pressure",
            "bullish_marabozu": f"Strong {color} candle with virtually no shadows - maximum bullish pressure", 
            "shooting_star": f"Small {color} body with long upper shadow - potential reversal signal",
            "hammer": f"Small {color} body with long lower shadow - potential reversal signal",
            "doji": f"Very small {color} body indicating market indecision",
            "strong_bullish": f"Large {color} body with {shadow_desc}",
            "strong_bearish": f"Large {color} body with {shadow_desc}",
            "moderate_bullish": f"Medium {color} body with {shadow_desc}",
            "moderate_bearish": f"Medium {color} body with {shadow_desc}",
            "weak_bullish": f"Small {color} body with {shadow_desc}",
            "weak_bearish": f"Small {color} body with {shadow_desc}",
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

    # Enhanced V2 Technical Analysis Methods
    def calculate_trend_strength(self, data: List[OHLCV]) -> float:
        """
        Calculate multi-factor trend strength score (0-100).
        
        Combines price momentum, volume confirmation, and trend consistency
        to provide a comprehensive trend strength assessment.
        
        Args:
            data: List of OHLCV data points
            
        Returns:
            Trend strength score from 0 (no trend) to 100 (very strong trend)
        """
        if len(data) < 10:
            return 0.0
            
        logger.debug("Calculating trend strength", data_points=len(data))
        
        df = self._to_dataframe(data)
        closes = df["close"].values
        volumes = df["volume"].values
        
        # Factor 1: Price momentum (30% weight)
        momentum_periods = min(10, len(closes))
        price_change = (closes[-1] - closes[-momentum_periods]) / closes[-momentum_periods]
        momentum_score = min(100, abs(price_change) * 100 * 5)  # Scale to 0-100
        
        # Factor 2: Trend consistency (40% weight)
        # Count periods moving in same direction
        direction_changes = 0
        for i in range(1, len(closes)):
            if i > 1:
                prev_direction = closes[i-1] > closes[i-2]
                curr_direction = closes[i] > closes[i-1]
                if prev_direction != curr_direction:
                    direction_changes += 1
        
        consistency_ratio = 1 - (direction_changes / max(1, len(closes) - 2))
        consistency_score = consistency_ratio * 100
        
        # Factor 3: Volume confirmation (30% weight)
        if len(volumes) >= 5:
            recent_vol_avg = np.mean(volumes[-5:])
            older_vol_avg = np.mean(volumes[-10:-5]) if len(volumes) >= 10 else recent_vol_avg
            volume_trend = recent_vol_avg / max(older_vol_avg, 1)
            volume_score = min(100, volume_trend * 50)  # Scale to 0-100
        else:
            volume_score = 50  # Neutral when insufficient data
        
        # Weighted combination
        trend_strength = (
            momentum_score * 0.3 +
            consistency_score * 0.4 +
            volume_score * 0.3
        )
        
        return max(0, min(100, trend_strength))
    
    def is_trending_market(self, data: List[OHLCV], min_trend_strength: int = 40) -> bool:
        """
        Determine if market is in a trending state vs sideways/choppy.
        
        Args:
            data: List of OHLCV data points
            min_trend_strength: Minimum trend strength threshold (0-100)
            
        Returns:
            True if market is trending, False if sideways/choppy
        """
        trend_strength = self.calculate_trend_strength(data)
        return trend_strength >= min_trend_strength
    
    def calculate_trend_quality(self, data: List[OHLCV]) -> float:
        """
        Calculate comprehensive trend quality score (0-100).
        
        Combines trend strength, duration, EMA alignment, and market structure
        to assess overall trend quality.
        
        Args:
            data: List of OHLCV data points
            
        Returns:
            Trend quality score from 0 (poor) to 100 (excellent)
        """
        if len(data) < 15:
            return 0.0
            
        logger.debug("Calculating trend quality", data_points=len(data))
        
        # Component 1: Base trend strength (40% weight)
        trend_strength = self.calculate_trend_strength(data)
        
        # Component 2: Trend duration (20% weight)
        trend_duration = self.calculate_trend_duration(data)
        duration_score = min(100, trend_duration * 10)  # Scale duration periods
        
        # Component 3: EMA alignment (20% weight)
        ema_alignment = self.detect_ema_alignment(data)
        if ema_alignment in ["strong_bullish", "strong_bearish"]:
            alignment_score = 100
        elif ema_alignment in ["bullish", "bearish"]:
            alignment_score = 70
        else:
            alignment_score = 30
        
        # Component 4: Market structure (20% weight)
        structure = self.calculate_market_structure(data)
        structure_score = structure.get("structure_strength", 50)
        
        # Weighted combination using V2 constants
        from src.core.constants import TradingConstants
        trend_quality = (
            trend_strength * (TradingConstants.V2_TREND_STRENGTH_WEIGHT or 0.4) +
            duration_score * (TradingConstants.V2_TREND_DURATION_WEIGHT or 0.2) +
            alignment_score * (TradingConstants.V2_EMA_ALIGNMENT_WEIGHT or 0.2) +
            structure_score * (TradingConstants.V2_MARKET_STRUCTURE_WEIGHT or 0.2)
        )
        
        return max(0, min(100, trend_quality))
    
    def detect_ema_alignment(self, data: List[OHLCV]) -> str:
        """
        Detect EMA alignment patterns (9>15>50 for bullish, 9<15<50 for bearish).
        
        Args:
            data: List of OHLCV data points or dict with pre-calculated EMAs
            
        Returns:
            Alignment pattern: 'strong_bullish', 'bullish', 'strong_bearish', 'bearish', 'mixed', 'neutral'
        """
        # Handle both OHLCV data and pre-calculated EMA dict
        if isinstance(data, dict):
            ema_9 = data.get("ema_9")
            ema_15 = data.get("ema_15")
            ema_50 = data.get("ema_50")
            current_price = data.get("current_price")
        else:
            # Calculate EMAs from OHLCV data
            if len(data) < 50:
                return "neutral"
                
            ema_9_values = self.calculate_ema(data, 9)
            ema_15_values = self.calculate_ema(data, 15)
            ema_50_values = self.calculate_ema(data, 50)
            
            ema_9 = ema_9_values[-1]
            ema_15 = ema_15_values[-1]
            ema_50 = ema_50_values[-1]
            current_price = data[-1].close
        
        if not all([ema_9, ema_15, ema_50, current_price]):
            return "neutral"
        
        # Perfect bullish alignment: Price > 9 EMA > 15 EMA > 50 EMA
        if current_price > ema_9 > ema_15 > ema_50:
            # Check separation strength
            sep_9_15 = (ema_9 - ema_15) / ema_15 * 100
            sep_15_50 = (ema_15 - ema_50) / ema_50 * 100
            if sep_9_15 > 1.0 and sep_15_50 > 1.0:
                return "strong_bullish"
            else:
                return "bullish"
        
        # Perfect bearish alignment: Price < 9 EMA < 15 EMA < 50 EMA
        elif current_price < ema_9 < ema_15 < ema_50:
            # Check separation strength
            sep_9_15 = (ema_15 - ema_9) / ema_9 * 100
            sep_15_50 = (ema_50 - ema_15) / ema_15 * 100
            if sep_9_15 > 1.0 and sep_15_50 > 1.0:
                return "strong_bearish"
            else:
                return "bearish"
        
        # Partial alignments
        elif ema_9 > ema_15 > ema_50:
            return "bullish"  # EMAs aligned bullish but price may be below
        elif ema_9 < ema_15 < ema_50:
            return "bearish"  # EMAs aligned bearish but price may be above
        
        # Mixed or unclear alignment
        else:
            return "mixed"
    
    def calculate_volatility_percentile(self, data: List[OHLCV], lookback_periods: int = 50) -> float:
        """
        Calculate ATR-based volatility percentile (0-1 range).
        
        Compares current ATR to historical ATR distribution to determine
        if current volatility is high or low relative to recent history.
        
        Args:
            data: List of OHLCV data points
            lookback_periods: Periods to use for percentile calculation
            
        Returns:
            Volatility percentile from 0 (low volatility) to 1 (high volatility)
        """
        if len(data) < max(14, lookback_periods):
            return 0.5  # Neutral when insufficient data
            
        logger.debug("Calculating volatility percentile", data_points=len(data), lookback=lookback_periods)
        
        # Calculate ATR values
        atr_values = self.calculate_atr(data, period=14)
        
        # Filter out None values and get recent ATR data
        valid_atr = [atr for atr in atr_values if atr is not None]
        if not valid_atr:
            return 0.5
            
        # Use lookback periods for percentile calculation
        recent_atr = valid_atr[-lookback_periods:] if len(valid_atr) >= lookback_periods else valid_atr
        current_atr = valid_atr[-1]
        
        # Calculate percentile rank
        values_below = sum(1 for atr in recent_atr if atr < current_atr)
        percentile = values_below / len(recent_atr)
        
        return percentile
    
    def calculate_market_structure(self, data: List[OHLCV]) -> Dict[str, Any]:
        """
        Analyze market structure including swing highs/lows and trend patterns.
        
        Args:
            data: List of OHLCV data points
            
        Returns:
            Dictionary with market structure analysis
        """
        if len(data) < 10:
            return {
                "swing_highs": [],
                "swing_lows": [],
                "higher_highs_count": 0,
                "lower_lows_count": 0,
                "trend_structure": "unclear",
                "structure_strength": 0
            }
            
        logger.debug("Analyzing market structure", data_points=len(data))
        
        df = self._to_dataframe(data)
        highs = df["high"].values
        lows = df["low"].values
        
        # Detect swing points using configurable period
        from src.core.constants import TradingConstants
        swing_period = getattr(TradingConstants, 'V2_SWING_DETECTION_PERIOD', 5)
        
        swing_highs = self._find_peaks(highs, distance=swing_period)
        swing_lows = self._find_peaks(-lows, distance=swing_period)  # Invert for troughs
        
        # Analyze higher highs and lower lows
        higher_highs_count = 0
        lower_lows_count = 0
        
        # Count higher highs
        if len(swing_highs) >= 2:
            for i in range(1, len(swing_highs)):
                if highs[swing_highs[i]] > highs[swing_highs[i-1]]:
                    higher_highs_count += 1
        
        # Count lower lows
        if len(swing_lows) >= 2:
            for i in range(1, len(swing_lows)):
                if lows[swing_lows[i]] < lows[swing_lows[i-1]]:
                    lower_lows_count += 1
        
        # Determine trend structure
        hh_threshold = getattr(TradingConstants, 'V2_HIGHER_HIGHS_THRESHOLD', 2)
        ll_threshold = getattr(TradingConstants, 'V2_LOWER_LOWS_THRESHOLD', 2)
        
        if higher_highs_count >= hh_threshold:
            trend_structure = "bullish"
        elif lower_lows_count >= ll_threshold:
            trend_structure = "bearish"
        else:
            trend_structure = "sideways"
        
        # Calculate structure strength (0-100)
        max_swings = max(len(swing_highs), len(swing_lows))
        if max_swings > 0:
            if trend_structure == "bullish":
                structure_strength = min(100, (higher_highs_count / max_swings) * 100)
            elif trend_structure == "bearish":
                structure_strength = min(100, (lower_lows_count / max_swings) * 100)
            else:
                structure_strength = 50  # Neutral for sideways
        else:
            structure_strength = 0
        
        return {
            "swing_highs": [int(idx) for idx in swing_highs],
            "swing_lows": [int(idx) for idx in swing_lows],
            "higher_highs_count": higher_highs_count,
            "lower_lows_count": lower_lows_count,
            "trend_structure": trend_structure,
            "structure_strength": structure_strength
        }
    
    def calculate_trend_duration(self, data: List[OHLCV]) -> int:
        """
        Calculate trend duration in number of periods.
        
        Tracks how long the current trend has been in place by analyzing
        consecutive periods moving in the same direction.
        
        Args:
            data: List of OHLCV data points
            
        Returns:
            Number of periods the current trend has lasted
        """
        if len(data) < 3:
            return 0
            
        closes = [candle.close for candle in data]
        
        # Determine current trend direction
        current_direction = closes[-1] > closes[-2]
        
        # Count consecutive periods in same direction
        duration = 1  # Current period
        
        for i in range(len(closes) - 2, 0, -1):
            period_direction = closes[i] > closes[i-1]
            if period_direction == current_direction:
                duration += 1
            else:
                break  # Trend change detected
        
        return duration
