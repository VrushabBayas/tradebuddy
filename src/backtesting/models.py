"""
Data models for TradeBuddy backtesting module.

Defines comprehensive data structures for backtesting configuration,
trades, results, and performance metrics.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.models import (
    BaseModelWithTimestamp,
    SignalAction,
    StrategyType,
    Symbol,
    TimeFrame,
)


class BacktestConfig(BaseModel):
    """Complete backtesting configuration."""

    strategy_type: StrategyType = Field(..., description="Trading strategy to test")
    symbol: Symbol = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    days_back: int = Field(
        default=30, ge=1, le=365, description="Number of days to backtest"
    )
    initial_capital: float = Field(
        default=850000.0, ge=85000.0, le=85000000.0, description="Initial capital in INR (default ₹10000 USD equivalent)"
    )
    leverage: int = Field(
        default=10, ge=1, le=10, description="Trading leverage multiplier"
    )
    confidence_threshold: int = Field(
        default=6, ge=1, le=10, description="Minimum signal confidence to trade"
    )
    commission_pct: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Commission percentage per trade (Delta Exchange futures taker fee)"
    )
    slippage_pct: float = Field(
        default=0.05, ge=0.0, le=0.5, description="Slippage percentage per trade"
    )
    max_position_size_pct: float = Field(
        default=100.0,
        ge=10.0,
        le=100.0,
        description="Maximum position size as percentage of capital",
    )
    stop_loss_pct: Optional[float] = Field(
        default=None, ge=0.1, le=20.0, description="Stop loss percentage (optional)"
    )
    take_profit_pct: Optional[float] = Field(
        default=None, ge=0.1, le=50.0, description="Take profit percentage (optional)"
    )

    # Strategy-specific configuration
    enable_rsi_filter: bool = Field(
        default=True, description="Enable RSI filter for EMA strategy"
    )
    enable_ema50_filter: bool = Field(
        default=False, description="Enable 50 EMA filter for EMA strategy"
    )
    enable_volume_filter: bool = Field(
        default=True, description="Enable volume confirmation filter"
    )
    enable_candlestick_filter: bool = Field(
        default=True, description="Enable candlestick pattern filter"
    )
    
    # Currency configuration
    use_maker_fees: bool = Field(
        default=False, description="Use maker fees instead of taker fees"
    )
    currency: str = Field(
        default="INR", description="Currency for reporting (INR/USD)"
    )
    exchange_rate: float = Field(
        default=85.0, description="USD to INR exchange rate"
    )

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("days_back")
    @classmethod
    def validate_days_back(cls, v: int) -> int:
        """Validate days_back is reasonable for backtesting."""
        if v < 7:
            raise ValueError("Minimum 7 days required for meaningful backtesting")
        return v

    @property
    def start_date(self) -> datetime:
        """Calculate backtest start date."""
        return datetime.utcnow() - timedelta(days=self.days_back)

    @property
    def end_date(self) -> datetime:
        """Calculate backtest end date."""
        return datetime.utcnow()


class TradeStatus(str, Enum):
    """Trade execution status."""

    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"
    CANCELLED = "cancelled"


class BacktestTrade(BaseModelWithTimestamp):
    """Individual trade record with comprehensive details."""

    trade_id: int = Field(..., description="Unique trade identifier")
    signal_type: SignalAction = Field(..., description="Original signal action")
    trade_action: str = Field(..., description="Actual trade action (LONG/SHORT)")
    status: TradeStatus = Field(default=TradeStatus.OPEN, description="Trade status")

    # Entry details
    entry_time: datetime = Field(..., description="Trade entry timestamp")
    entry_price: float = Field(..., description="Entry price")
    entry_reason: str = Field(..., description="Reason for entry")

    # Exit details
    exit_time: Optional[datetime] = Field(
        default=None, description="Trade exit timestamp"
    )
    exit_price: Optional[float] = Field(default=None, description="Exit price")
    exit_reason: Optional[str] = Field(default=None, description="Reason for exit")

    # Position details
    quantity: float = Field(..., description="Position quantity")
    position_size_usd: float = Field(..., description="Position size in USD")
    leverage_used: int = Field(..., description="Leverage used for this trade")
    margin_used: float = Field(..., description="Margin required for position")

    # Performance
    confidence_score: int = Field(..., description="Signal confidence (1-10)")
    pnl_usd: Optional[float] = Field(default=None, description="P&L in USD")
    pnl_pct: Optional[float] = Field(default=None, description="P&L percentage")
    pnl_inr: Optional[float] = Field(default=None, description="P&L in INR")
    commission_paid: float = Field(default=0.0, description="Commission costs")
    slippage_cost: float = Field(default=0.0, description="Slippage costs")
    gst_paid: float = Field(default=0.0, description="GST paid on fees (18%)")
    duration_minutes: Optional[int] = Field(
        default=None, description="Trade duration in minutes"
    )

    # Strategy context
    strategy_context: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific data"
    )

    # Risk management
    stop_loss_price: Optional[float] = Field(
        default=None, description="Stop loss price"
    )
    take_profit_price: Optional[float] = Field(
        default=None, description="Take profit price"
    )

    @property
    def is_winner(self) -> Optional[bool]:
        """Check if trade is profitable."""
        if self.pnl_usd is None:
            return None
        return self.pnl_usd > 0

    @property
    def total_costs(self) -> float:
        """Calculate total trading costs including GST."""
        return self.commission_paid + self.slippage_cost + self.gst_paid

    @property
    def net_pnl_usd(self) -> Optional[float]:
        """Calculate net P&L after costs."""
        if self.pnl_usd is None:
            return None
        return self.pnl_usd - self.total_costs

    def close_trade(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        commission_pct: float,
        slippage_pct: float,
    ) -> None:
        """Close the trade and calculate final P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = TradeStatus.CLOSED

        # Calculate duration
        self.duration_minutes = int(
            (exit_time - self.entry_time).total_seconds() / 60
        )

        # Calculate gross P&L with proper leverage accounting
        if self.trade_action == "LONG":
            price_change = exit_price - self.entry_price
        else:  # SHORT
            price_change = self.entry_price - exit_price

        # P&L calculation: (price change / entry price) * position size * leverage
        # This gives the actual dollar P&L on the leveraged position
        raw_return_pct = (price_change / self.entry_price) * 100
        self.pnl_pct = raw_return_pct * self.leverage_used
        self.pnl_usd = (raw_return_pct / 100) * self.position_size_usd * self.leverage_used

        # Calculate exit commission and slippage on leveraged position value
        leveraged_position_value = self.position_size_usd * self.leverage_used
        exit_commission = leveraged_position_value * (commission_pct / 100)
        exit_slippage = leveraged_position_value * (slippage_pct / 100)

        self.commission_paid += exit_commission
        self.slippage_cost += exit_slippage
        
        # Calculate GST on total commission (18% on trading fees)
        from src.core.constants import TradingConstants
        self.gst_paid = self.commission_paid * (TradingConstants.GST_RATE / 100)
        
        # Calculate P&L in INR
        self.pnl_inr = self.pnl_usd * TradingConstants.USD_TO_INR if self.pnl_usd else None


class EquityPoint(BaseModel):
    """Single point in equity curve."""

    timestamp: datetime = Field(..., description="Timestamp")
    equity: float = Field(..., description="Portfolio equity")
    cash: float = Field(..., description="Available cash")
    position_value: float = Field(..., description="Open position value")
    drawdown_pct: float = Field(default=0.0, description="Drawdown percentage")
    benchmark_value: Optional[float] = Field(
        default=None, description="Benchmark portfolio value"
    )


class DrawdownPeriod(BaseModel):
    """Drawdown period analysis."""

    start_date: datetime = Field(..., description="Drawdown start")
    end_date: Optional[datetime] = Field(default=None, description="Drawdown end")
    peak_value: float = Field(..., description="Peak equity before drawdown")
    trough_value: float = Field(..., description="Lowest equity during drawdown")
    drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    duration_days: Optional[int] = Field(
        default=None, description="Drawdown duration in days"
    )
    recovery_days: Optional[int] = Field(
        default=None, description="Days to recover to new high"
    )


class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics."""

    # Basic metrics
    total_return_pct: float = Field(..., description="Total return percentage")
    annualized_return_pct: float = Field(..., description="Annualized return")
    cagr_pct: float = Field(..., description="Compound Annual Growth Rate")

    # Risk metrics  
    max_drawdown_pct: float = Field(..., description="Maximum drawdown")
    volatility_pct: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    ulcer_index: float = Field(..., description="Ulcer index")

    # Trading metrics
    total_trades: int = Field(..., description="Total number of trades")
    win_rate_pct: float = Field(..., description="Win rate percentage")
    loss_rate_pct: float = Field(..., description="Loss rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    expectancy: float = Field(..., description="Expected value per trade")

    # Trade analysis
    avg_win_pct: float = Field(..., description="Average winning trade %")
    avg_loss_pct: float = Field(..., description="Average losing trade %")
    largest_win_pct: float = Field(..., description="Largest winning trade %")
    largest_loss_pct: float = Field(..., description="Largest losing trade %")
    win_loss_ratio: float = Field(..., description="Average win/loss ratio")

    # Streak analysis
    max_win_streak: int = Field(..., description="Maximum consecutive wins")
    max_loss_streak: int = Field(..., description="Maximum consecutive losses")
    avg_win_streak: float = Field(..., description="Average win streak length")
    avg_loss_streak: float = Field(..., description="Average loss streak length")

    # Exposure metrics
    market_exposure_pct: float = Field(..., description="Time in market percentage")
    avg_trade_duration_hours: float = Field(
        ..., description="Average trade duration in hours"
    )

    # Drawdown analysis
    avg_drawdown_pct: float = Field(..., description="Average drawdown")
    drawdown_periods: List[DrawdownPeriod] = Field(
        default_factory=list, description="Detailed drawdown periods"
    )
    recovery_factor: float = Field(..., description="Recovery factor")

    # Benchmark comparison
    benchmark_return_pct: Optional[float] = Field(
        default=None, description="Benchmark return percentage"
    )
    alpha: Optional[float] = Field(default=None, description="Alpha vs benchmark")
    beta: Optional[float] = Field(default=None, description="Beta vs benchmark")
    tracking_error: Optional[float] = Field(
        default=None, description="Tracking error vs benchmark"
    )


class StrategyAnalysis(BaseModel):
    """Strategy-specific analysis results."""

    strategy_type: StrategyType = Field(..., description="Strategy type analyzed")

    # Signal analysis
    signal_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Distribution of signal types"
    )
    confidence_distribution: Dict[int, int] = Field(
        default_factory=dict, description="Distribution of confidence scores"
    )
    filter_effectiveness: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Filter effectiveness analysis"
    )

    # Strategy-specific metrics
    strategy_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific performance metrics"
    )

    # Best/worst performing conditions
    best_conditions: Dict[str, Any] = Field(
        default_factory=dict, description="Best performing market conditions"
    )
    worst_conditions: Dict[str, Any] = Field(
        default_factory=dict, description="Worst performing market conditions"
    )


class BenchmarkComparison(BaseModel):
    """Benchmark comparison analysis."""

    benchmark_name: str = Field(default="Buy & Hold", description="Benchmark name")
    benchmark_return_pct: float = Field(..., description="Benchmark total return")
    strategy_return_pct: float = Field(..., description="Strategy total return")
    outperformance_pct: float = Field(..., description="Strategy outperformance")
    
    # Risk-adjusted comparison
    benchmark_sharpe: float = Field(..., description="Benchmark Sharpe ratio")
    strategy_sharpe: float = Field(..., description="Strategy Sharpe ratio")
    sharpe_improvement: float = Field(..., description="Sharpe ratio improvement")

    benchmark_max_dd_pct: float = Field(..., description="Benchmark max drawdown")
    strategy_max_dd_pct: float = Field(..., description="Strategy max drawdown")
    drawdown_improvement_pct: float = Field(..., description="Drawdown improvement")


class BacktestResult(BaseModelWithTimestamp):
    """Complete backtesting results container."""

    config: BacktestConfig = Field(..., description="Backtest configuration")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    total_duration_days: int = Field(..., description="Total backtest duration")

    # Core results
    trades: List[BacktestTrade] = Field(
        default_factory=list, description="All executed trades"
    )
    equity_curve: List[EquityPoint] = Field(
        default_factory=list, description="Portfolio equity over time"
    )

    # Analysis results
    performance_metrics: PerformanceMetrics = Field(
        ..., description="Comprehensive performance metrics"
    )
    strategy_analysis: StrategyAnalysis = Field(
        ..., description="Strategy-specific analysis"
    )
    benchmark_comparison: BenchmarkComparison = Field(
        ..., description="Benchmark comparison results"
    )

    # Portfolio summary
    initial_capital: float = Field(..., description="Starting capital")
    final_capital: float = Field(..., description="Ending capital")
    peak_capital: float = Field(..., description="Highest capital reached")

    # Execution details
    total_signals_generated: int = Field(
        default=0, description="Total signals from strategy"
    )
    signals_traded: int = Field(default=0, description="Signals that resulted in trades")
    signal_conversion_rate_pct: float = Field(
        default=0.0, description="Percentage of signals that became trades"
    )
    
    # Signal quality analysis
    signal_quality_metrics: Optional[Dict[str, Any]] = Field(
        default=None, description="Signal quality analysis results"
    )
    optimization_results: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Parameter optimization results"
    )

    @property
    def winning_trades(self) -> List[BacktestTrade]:
        """Get all winning trades."""
        return [t for t in self.trades if t.is_winner is True]

    @property
    def losing_trades(self) -> List[BacktestTrade]:
        """Get all losing trades."""
        return [t for t in self.trades if t.is_winner is False]

    @property
    def open_trades(self) -> List[BacktestTrade]:
        """Get all currently open trades."""
        return [t for t in self.trades if t.status == TradeStatus.OPEN]

    @property
    def closed_trades(self) -> List[BacktestTrade]:
        """Get all closed trades."""
        return [t for t in self.trades if t.status == TradeStatus.CLOSED]