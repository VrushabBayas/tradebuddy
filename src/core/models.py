"""
Data models for TradeBuddy application.

Defines Pydantic models for market data, signals, and analysis results.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


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
    BTCUSDT = "BTCUSDT"
    ETHUSDT = "ETHUSDT"
    SOLUSDT = "SOLUSDT"
    ADAUSDT = "ADAUSDT"
    DOGEUSDT = "DOGEUSDT"


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
    
    @field_validator("open", "high", "low", "close", "volume")
    @classmethod
    def validate_positive_values(cls, v):
        """Validate that all values are positive."""
        if v <= 0:
            raise ValueError("OHLCV values must be positive")
        return v


class MarketData(BaseModelWithTimestamp):
    """Market data container."""
    
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Data timeframe")
    current_price: float = Field(..., description="Current market price")
    ohlcv_data: List[OHLCV] = Field(default_factory=list, description="Historical OHLCV data")
    
    @property
    def latest_ohlcv(self) -> Optional[OHLCV]:
        """Get the most recent OHLCV data."""
        return self.ohlcv_data[-1] if self.ohlcv_data else None


class TechnicalIndicator(BaseModel):
    """Technical indicator data."""
    
    name: str = Field(..., description="Indicator name")
    value: Union[Decimal, Dict[str, Decimal]] = Field(..., description="Indicator value")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SupportResistanceLevel(BaseModel):
    """Support or resistance level."""
    
    level: Decimal = Field(..., description="Price level")
    strength: int = Field(..., ge=1, le=10, description="Level strength (1-10)")
    is_support: bool = Field(..., description="True if support, False if resistance")
    touches: int = Field(default=0, description="Number of times price touched this level")
    last_touch: Optional[datetime] = Field(None, description="Last time price touched this level")


class EMACrossover(BaseModel):
    """EMA crossover data."""
    
    ema_9: Decimal = Field(..., description="9-period EMA value")
    ema_15: Decimal = Field(..., description="15-period EMA value")
    is_golden_cross: bool = Field(..., description="True if 9 EMA > 15 EMA")
    crossover_strength: int = Field(..., ge=1, le=10, description="Crossover strength (1-10)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradingSignal(BaseModelWithTimestamp):
    """Trading signal with entry/exit levels."""
    
    symbol: Symbol = Field(..., description="Trading symbol")
    strategy: StrategyType = Field(..., description="Strategy that generated the signal")
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
        default_factory=list,
        description="Supporting technical indicators"
    )
    
    # Risk management
    risk_reward_ratio: Optional[Decimal] = Field(None, description="Risk/reward ratio")
    position_size_pct: Optional[Decimal] = Field(None, description="Recommended position size %")
    
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


class AnalysisResult(BaseModelWithTimestamp):
    """Complete analysis result from AI model."""
    
    symbol: Symbol = Field(..., description="Analyzed symbol")
    timeframe: TimeFrame = Field(..., description="Analysis timeframe")
    strategy: StrategyType = Field(..., description="Strategy used")
    
    # Market data context
    market_data: MarketData = Field(..., description="Market data analyzed")
    
    # Technical analysis
    support_resistance_levels: List[SupportResistanceLevel] = Field(
        default_factory=list,
        description="Identified support/resistance levels"
    )
    ema_crossover: Optional[EMACrossover] = Field(None, description="EMA crossover data")
    
    # Generated signals
    signals: List[TradingSignal] = Field(
        default_factory=list,
        description="Generated trading signals"
    )
    
    # AI analysis
    ai_analysis: str = Field(..., description="AI-generated analysis text")
    execution_time: Optional[float] = Field(None, description="Analysis execution time in seconds")
    
    @property
    def primary_signal(self) -> Optional[TradingSignal]:
        """Get the primary (highest confidence) signal."""
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.confidence)


class SessionConfig(BaseModel):
    """Configuration for an analysis session."""
    
    strategy: StrategyType = Field(..., description="Selected strategy")
    symbol: Symbol = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Analysis timeframe")
    
    # Risk management
    stop_loss_pct: Decimal = Field(default=Decimal("2.5"), description="Stop loss percentage")
    take_profit_pct: Decimal = Field(default=Decimal("5.0"), description="Take profit percentage")
    position_size_pct: Decimal = Field(default=Decimal("2.0"), description="Position size percentage")
    
    # Analysis parameters
    confidence_threshold: int = Field(default=6, ge=1, le=10, description="Minimum confidence for signals")
    max_signals_per_session: int = Field(default=10, description="Maximum signals per session")
    
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
        default_factory=list,
        description="All analysis results from the session"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate based on actionable signals."""
        if self.total_signals == 0:
            return 0.0
        
        actionable_signals = sum(
            1 for result in self.analysis_results
            for signal in result.signals
            if signal.is_actionable
        )
        
        return (actionable_signals / self.total_signals) * 100