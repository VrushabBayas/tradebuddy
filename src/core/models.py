"""
Data models for TradeBuddy application.

Defines Pydantic models for market data, signals, and analysis results.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TimeFrame(str, Enum):
    """Supported timeframes for market data."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


class Symbol(str, Enum):
    """Supported trading symbols."""

    BTCUSD = "BTCUSD"
    ETHUSD = "ETHUSD"
    SOLUSDT = "SOLUSDT"  # Keep as USDT for now
    ADAUSDT = "ADAUSDT"  # Keep as USDT for now
    DOGEUSDT = "DOGEUSDT"  # Keep as USDT for now


class StrategyType(str, Enum):
    """Available trading strategies."""

    SUPPORT_RESISTANCE = "support_resistance"
    EMA_CROSSOVER = "ema_crossover"
    COMBINED = "combined"


class SignalAction(str, Enum):
    """Trading signal actions."""

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"


class SignalStrength(str, Enum):
    """Signal strength levels."""

    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class BaseModelWithTimestamp(BaseModel):
    """Base model with timestamp field."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        }
    )


class OHLCV(BaseModelWithTimestamp):
    """OHLCV (Open, High, Low, Close, Volume) market data."""

    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")

    @field_validator("open", "high", "low", "close")
    @classmethod
    def validate_positive_prices(cls, v):
        """Validate that price values are positive."""
        if v <= 0:
            raise ValueError("OHLCV price values must be positive")
        return v
    
    @field_validator("volume")
    @classmethod
    def validate_volume(cls, v):
        """Validate volume (allow zero for low-activity periods)."""
        if v < 0:
            raise ValueError("Volume cannot be negative")
        return v


class MarketData(BaseModelWithTimestamp):
    """Market data container."""

    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    current_price: float = Field(..., description="Current market price")
    ohlcv_data: List[OHLCV] = Field(
        default_factory=list, description="Historical OHLCV data"
    )

    @property
    def latest_ohlcv(self) -> Optional[OHLCV]:
        """Get the most recent OHLCV data by timestamp."""
        if not self.ohlcv_data:
            return None
        
        # Find the candle with the maximum timestamp to ensure we get the latest
        # This is more robust than assuming index 0 is always the newest
        return max(self.ohlcv_data, key=lambda candle: candle.timestamp)


class TechnicalIndicator(BaseModel):
    """Technical indicator data."""

    name: str = Field(..., description="Indicator name")
    value: Union[Decimal, Dict[str, Decimal]] = Field(
        ..., description="Indicator value"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SupportResistanceLevel(BaseModel):
    """Support or resistance level."""

    level: Decimal = Field(..., description="Price level")
    strength: int = Field(..., ge=1, le=10, description="Level strength (1-10)")
    is_support: bool = Field(..., description="True if support, False if resistance")
    touches: int = Field(
        default=0, description="Number of times price touched this level"
    )
    last_touch: Optional[datetime] = Field(
        None, description="Last time price touched this level"
    )


class EMACrossover(BaseModel):
    """EMA crossover data."""

    ema_9: Decimal = Field(..., description="9-period EMA value")
    ema_15: Decimal = Field(..., description="15-period EMA value")
    is_golden_cross: bool = Field(..., description="True if 9 EMA > 15 EMA")
    crossover_strength: int = Field(
        ..., ge=1, le=10, description="Crossover strength (1-10)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CandlestickFormation(BaseModel):
    """Candlestick pattern formation details."""
    
    pattern_name: str = Field(..., description="Name of the detected pattern")
    pattern_type: str = Field(..., description="Type category (bullish/bearish/neutral)")
    strength: int = Field(..., ge=1, le=10, description="Pattern strength score (1-10)")
    signal_direction: str = Field(..., description="Signal direction (strong_bullish, bullish, neutral, bearish, strong_bearish)")
    
    # Visual characteristics
    body_ratio: float = Field(..., description="Body size ratio to total candle range")
    upper_shadow_ratio: float = Field(..., description="Upper shadow ratio to body size")
    lower_shadow_ratio: float = Field(..., description="Lower shadow ratio to body size")
    
    # Descriptive information
    visual_description: str = Field(..., description="Human-readable candle description")
    trend_context: str = Field(..., description="Trend context (uptrend/downtrend/sideways)")
    volume_confirmation: bool = Field(default=False, description="Whether volume confirms the pattern")
    
    @property
    def is_strong_pattern(self) -> bool:
        """Check if this is a strong pattern (strength >= 8)."""
        return self.strength >= 8
    
    @property
    def pattern_display_name(self) -> str:
        """Get formatted pattern name for display."""
        return self.pattern_name.replace('_', ' ').title()


class TradingSignal(BaseModelWithTimestamp):
    """Trading signal with entry/exit levels."""

    symbol: Symbol = Field(..., description="Trading symbol")
    strategy: StrategyType = Field(
        ..., description="Strategy that generated the signal"
    )
    action: SignalAction = Field(..., description="Signal action")
    strength: SignalStrength = Field(..., description="Signal strength")
    confidence: int = Field(..., ge=1, le=10, description="Confidence level (1-10)")

    # Price levels
    entry_price: Decimal = Field(..., description="Entry price")
    stop_loss: Optional[Decimal] = Field(None, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(None, description="Take profit price")

    # Analysis
    reasoning: str = Field(..., description="Signal reasoning")
    supporting_indicators: List[TechnicalIndicator] = Field(
        default_factory=list, description="Supporting technical indicators"
    )

    # Risk management
    risk_reward_ratio: Optional[Decimal] = Field(None, description="Risk/reward ratio")
    position_size_pct: Optional[Decimal] = Field(
        None, description="Recommended position size %"
    )
    
    # Enhanced context information
    candle_timestamp: Optional[datetime] = Field(None, description="Timestamp of the candle that triggered this signal")
    candle_formation: Optional[CandlestickFormation] = Field(None, description="Candlestick formation details")
    pattern_context: Optional[str] = Field(None, description="Additional pattern context and confluence")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence is between 1 and 10."""
        if not 1 <= v <= 10:
            raise ValueError("Confidence must be between 1 and 10")
        return v

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (confidence >= 6)."""
        return self.confidence >= 6
    
    @property
    def candle_time_display(self) -> str:
        """Get formatted candle timestamp for display in IST."""
        if self.candle_timestamp:
            from src.utils.helpers import format_ist_time
            return format_ist_time(self.candle_timestamp, include_seconds=True)
        return "N/A"
    
    @property
    def formation_display(self) -> str:
        """Get formatted candlestick formation for display."""
        if self.candle_formation:
            return f"{self.candle_formation.pattern_display_name} ({self.candle_formation.strength}/10)"
        return "Standard Candle"
    
    @property
    def enhanced_reasoning(self) -> str:
        """Get enhanced reasoning with detailed pattern context."""
        base_reasoning = self.reasoning
        
        # Add candlestick formation context
        if self.candle_formation:
            if self.candle_formation.is_strong_pattern:
                pattern_info = f" [Strong Pattern: {self.candle_formation.visual_description}]"
            else:
                pattern_info = f" [Pattern: {self.candle_formation.visual_description}]"
            base_reasoning += pattern_info
        
        # Add detailed pattern context if available
        if self.pattern_context:
            # If pattern context contains detailed reasoning, prioritize it
            if len(self.pattern_context) > 50:  # Likely contains detailed reasoning
                base_reasoning += f" [Educational Context: {self.pattern_context}]"
            else:
                base_reasoning += f" [Context: {self.pattern_context}]"
        
        return base_reasoning


class AnalysisResult(BaseModelWithTimestamp):
    """Complete analysis result from AI model."""

    symbol: Symbol = Field(..., description="Analyzed symbol")
    timeframe: TimeFrame = Field(..., description="Analysis timeframe")
    strategy: StrategyType = Field(..., description="Strategy used")

    # Market data context
    market_data: MarketData = Field(..., description="Market data analyzed")

    # Technical analysis
    support_resistance_levels: List[SupportResistanceLevel] = Field(
        default_factory=list, description="Identified support/resistance levels"
    )
    ema_crossover: Optional[EMACrossover] = Field(
        None, description="EMA crossover data"
    )

    # Generated signals
    signals: List[TradingSignal] = Field(
        default_factory=list, description="Generated trading signals"
    )

    # AI analysis
    ai_analysis: str = Field(..., description="AI-generated analysis text")
    execution_time: Optional[float] = Field(
        None, description="Analysis execution time in seconds"
    )

    @property
    def primary_signal(self) -> Optional[TradingSignal]:
        """Get the primary (highest confidence) signal."""
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.confidence)


class EMAStrategyConfig(BaseModel):
    """Configuration for Enhanced EMA Crossover Strategy."""

    # Enhanced filter options (as requested - 50 EMA filter is optional)
    enable_rsi_filter: bool = Field(
        default=True, description="Enable RSI > 50 for longs, RSI < 50 for shorts"
    )
    enable_ema50_filter: bool = Field(
        default=False, description="Enable 50 EMA trend filter (optional)"
    )
    enable_volume_filter: bool = Field(
        default=True, description="Enable 110% volume confirmation"
    )
    enable_candlestick_filter: bool = Field(
        default=True, description="Enable candlestick confirmation"
    )

    # RSI parameters
    rsi_period: int = Field(
        default=14, ge=5, le=50, description="RSI calculation period"
    )
    rsi_bullish_threshold: float = Field(
        default=50.0, ge=30.0, le=70.0, description="RSI bullish threshold"
    )

    # EMA parameters
    ema_short_period: int = Field(
        default=9, ge=5, le=20, description="Short EMA period"
    )
    ema_long_period: int = Field(
        default=15, ge=10, le=30, description="Long EMA period"
    )
    ema_trend_period: int = Field(
        default=50, ge=20, le=100, description="Trend filter EMA period"
    )

    # Volume parameters
    volume_period: int = Field(
        default=20, ge=10, le=50, description="Volume SMA period"
    )
    volume_threshold_pct: float = Field(
        default=110.0,
        ge=100.0,
        le=200.0,
        description="Volume confirmation threshold (%)",
    )

    # ATR parameters
    atr_period: int = Field(
        default=14, ge=5, le=30, description="ATR calculation period"
    )
    atr_stop_multiplier: float = Field(
        default=1.5, ge=1.0, le=3.0, description="ATR stop loss multiplier"
    )

    model_config = ConfigDict(use_enum_values=True)


class SessionConfig(BaseModel):
    """Configuration for an analysis session."""

    strategy: StrategyType = Field(..., description="Selected strategy")
    symbol: Symbol = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Analysis timeframe")

    # Capital Management (risk-based approach)
    total_capital_inr: Decimal = Field(
        default=Decimal("100000"),
        description="Total available capital in INR (e.g., ₹1,00,000)",
    )
    trading_capital_pct: Decimal = Field(
        default=Decimal("50.0"),
        description="Percentage of total capital used for trading (e.g., 50%)",
    )
    risk_per_trade_pct: Decimal = Field(
        default=Decimal("2.0"),
        description="Risk percentage per trade of trading capital (e.g., 2%)",
    )
    
    # Risk management (optimized for 10x leverage crypto trading)
    stop_loss_pct: Decimal = Field(
        default=Decimal("1.5"),
        description="Stop loss percentage (tighter for leverage)",
    )
    take_profit_pct: Decimal = Field(
        default=Decimal("3.0"),
        description="Take profit percentage (realistic for crypto)",
    )
    position_size_pct: Decimal = Field(
        default=Decimal("5.0"),
        description="Position size percentage (legacy - use risk-based calculation)",
    )

    # Leverage and position sizing
    leverage: int = Field(
        default=10, ge=1, le=100, description="Leverage multiplier (1-100x)"
    )
    min_lot_size: Decimal = Field(
        default=Decimal("0.001"), description="Minimum lot size in BTC"
    )
    max_position_risk: Decimal = Field(
        default=Decimal("10.0"), description="Maximum position risk percentage"
    )

    # Analysis parameters
    confidence_threshold: int = Field(
        default=6, ge=1, le=10, description="Minimum confidence for signals"
    )
    max_signals_per_session: int = Field(
        default=10, description="Maximum signals per session"
    )

    # Strategy-specific configurations
    ema_config: "Optional[EMAStrategyConfig]" = Field(
        default=None, description="EMA strategy configuration"
    )

    model_config = ConfigDict(use_enum_values=True)
    
    @property
    def trading_capital_inr(self) -> Decimal:
        """Calculate trading capital in INR."""
        return self.total_capital_inr * (self.trading_capital_pct / Decimal("100"))
    
    @property
    def risk_amount_per_trade_inr(self) -> Decimal:
        """Calculate risk amount per trade in INR."""
        return self.trading_capital_inr * (self.risk_per_trade_pct / Decimal("100"))
    
    @property
    def backup_capital_inr(self) -> Decimal:
        """Calculate backup capital (not used for trading) in INR."""
        return self.total_capital_inr - self.trading_capital_inr


class RealTimeConfig(BaseModel):
    """Configuration for real-time analysis sessions."""

    strategy: StrategyType = Field(
        ..., description="Strategy to use for real-time analysis"
    )
    symbol: Symbol = Field(default=Symbol.BTCUSD, description="Trading symbol")
    timeframe: TimeFrame = Field(
        default=TimeFrame.ONE_MINUTE, description="Analysis timeframe"
    )
    duration_minutes: int = Field(
        default=5, ge=1, le=60, description="Session duration in minutes"
    )
    buffer_size: int = Field(default=50, ge=20, le=100, description="OHLCV buffer size")
    confidence_threshold: int = Field(
        default=5, ge=1, le=10, description="Minimum confidence for signals"
    )

    # Real-time specific settings
    historical_candles: int = Field(
        default=45, ge=20, le=100, description="Historical candles to preload"
    )
    max_analysis_count: int = Field(
        default=50, ge=1, le=200, description="Maximum number of analyses per session"
    )

    model_config = ConfigDict(use_enum_values=True)


class MonitoringConfig(BaseModel):
    """Configuration for continuous market monitoring mode."""

    strategy: StrategyType = Field(..., description="Strategy to use for monitoring")
    symbols: List[Symbol] = Field(
        default=[Symbol.BTCUSD], description="Symbols to monitor"
    )
    timeframe: TimeFrame = Field(
        default=TimeFrame.ONE_MINUTE, description="Monitoring timeframe"
    )

    # Monitoring behavior
    signal_threshold: int = Field(
        default=7, ge=5, le=10, description="Minimum confidence for alerts"
    )
    refresh_interval: int = Field(
        default=60, ge=30, le=300, description="Refresh interval in seconds"
    )
    max_signals_per_hour: int = Field(
        default=10, ge=1, le=50, description="Maximum signals per hour per symbol"
    )

    # Display settings
    show_neutral_signals: bool = Field(
        default=False, description="Show neutral/wait signals"
    )
    compact_display: bool = Field(
        default=True, description="Use compact display format"
    )
    auto_scroll: bool = Field(default=True, description="Auto-scroll to latest signals")

    # Buffer and history
    buffer_size: int = Field(
        default=100, ge=50, le=200, description="OHLCV buffer size for analysis"
    )
    historical_candles: int = Field(
        default=50, ge=30, le=100, description="Historical candles to preload"
    )

    model_config = ConfigDict(use_enum_values=True)


class SessionResults(BaseModelWithTimestamp):
    """Results from a trading session."""

    config: SessionConfig = Field(..., description="Session configuration")
    duration: float = Field(..., description="Session duration in seconds")

    # Signal statistics
    total_signals: int = Field(default=0, description="Total signals generated")
    buy_signals: int = Field(default=0, description="Number of buy signals")
    sell_signals: int = Field(default=0, description="Number of sell signals")
    neutral_signals: int = Field(default=0, description="Number of neutral signals")

    # Performance metrics
    avg_confidence: float = Field(default=0.0, description="Average signal confidence")
    max_confidence: int = Field(default=0, description="Maximum signal confidence")
    min_confidence: int = Field(default=0, description="Minimum signal confidence")

    # Analysis results
    analysis_results: List[AnalysisResult] = Field(
        default_factory=list, description="All analysis results from the session"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate based on actionable signals."""
        if self.total_signals == 0:
            return 0.0

        actionable_signals = sum(
            1
            for result in self.analysis_results
            for signal in result.signals
            if signal.is_actionable
        )

        return (actionable_signals / self.total_signals) * 100
