"""
Portfolio management for TradeBuddy backtesting.

Handles position tracking, risk management, and portfolio equity calculation
with support for leverage and realistic trading costs.
"""

import structlog
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.backtesting.models import BacktestTrade, EquityPoint, TradeStatus
from src.core.models import SignalAction
from src.utils.helpers import to_float, to_decimal


logger = structlog.get_logger(__name__)


class Portfolio:
    """
    Portfolio manager for backtesting with leverage support.
    
    Manages cash, positions, and portfolio equity with realistic
    trading costs and risk management.
    """

    def __init__(
        self,
        initial_capital: float,
        max_leverage: int = 10,
        commission_pct: float = 0.05,  # Delta Exchange India futures taker
        slippage_pct: float = 0.05,
    ):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital in USD
            max_leverage: Maximum allowed leverage
            commission_pct: Commission percentage per trade
            slippage_pct: Slippage percentage per trade
        """
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # Portfolio state
        self.cash = initial_capital
        self.open_positions: Dict[int, BacktestTrade] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.equity_history: List[EquityPoint] = []

        # Tracking
        self.trade_counter = 0
        self.peak_equity = initial_capital
        self.total_commission_paid = 0.0
        self.total_slippage_paid = 0.0

        logger.info(
            "Portfolio initialized",
            initial_capital=initial_capital,
            max_leverage=max_leverage,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )

    @property
    def current_equity(self) -> float:
        """Calculate current portfolio equity."""
        return self.cash + self.open_positions_value

    @property
    def open_positions_value(self) -> float:
        """Calculate total value of open positions."""
        return sum(pos.position_size_usd for pos in self.open_positions.values())

    @property
    def margin_used(self) -> float:
        """Calculate total margin currently used."""
        return sum(pos.margin_used for pos in self.open_positions.values())

    @property
    def available_margin(self) -> float:
        """Calculate available margin for new positions."""
        return max(0, self.current_equity - self.margin_used)

    @property
    def total_return_pct(self) -> float:
        """Calculate total return percentage."""
        return ((self.current_equity - self.initial_capital) / self.initial_capital) * 100

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L from open positions."""
        total_unrealized = 0.0
        for trade in self.open_positions.values():
            if trade.pnl_usd is not None:
                total_unrealized += trade.pnl_usd
        return total_unrealized

    def can_open_position(
        self, 
        position_size_usd: float, 
        leverage: int
    ) -> Tuple[bool, str]:
        """
        Check if portfolio can open a new position.

        Args:
            position_size_usd: Desired position size in USD
            leverage: Desired leverage

        Returns:
            Tuple of (can_open, reason)
        """
        if leverage > self.max_leverage:
            return False, f"Leverage {leverage}x exceeds maximum {self.max_leverage}x"

        required_margin = position_size_usd / leverage
        
        # Add commission and slippage to required capital (on leveraged value)
        leveraged_value = position_size_usd * leverage
        entry_commission = leveraged_value * (self.commission_pct / 100)
        entry_slippage = leveraged_value * (self.slippage_pct / 100)
        total_required = required_margin + entry_commission + entry_slippage

        if total_required > self.available_margin:
            return False, f"Insufficient margin: need ${total_required:.2f}, have ${self.available_margin:.2f}"

        return True, "Position can be opened"

    def open_position(
        self,
        signal: SignalAction,
        price: float,
        timestamp: datetime,
        confidence: int,
        position_size_usd: float,
        leverage: int,
        strategy_context: Dict,
        entry_reason: str = "Strategy signal",
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> Optional[BacktestTrade]:
        """
        Open a new position.

        Args:
            signal: Trading signal (BUY/SELL)
            price: Entry price
            timestamp: Entry timestamp
            confidence: Signal confidence (1-10)
            position_size_usd: Position size in USD
            leverage: Leverage to use
            strategy_context: Strategy-specific data
            entry_reason: Reason for opening position
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)

        Returns:
            BacktestTrade object if successful, None otherwise
        """
        # Check if position can be opened
        can_open, reason = self.can_open_position(position_size_usd, leverage)
        if not can_open:
            logger.warning(
                "Cannot open position",
                reason=reason,
                signal=signal,
                price=price,
                position_size=position_size_usd,
            )
            return None

        # Calculate costs
        required_margin = position_size_usd / leverage
        leveraged_position_value = position_size_usd * leverage
        entry_commission = leveraged_position_value * (self.commission_pct / 100)
        entry_slippage = leveraged_position_value * (self.slippage_pct / 100)
        
        # Calculate GST on commission (18%)
        from src.core.constants import TradingConstants
        entry_gst = entry_commission * (TradingConstants.GST_RATE / 100)

        # Apply slippage to entry price
        if signal == SignalAction.BUY:
            trade_action = "LONG"
            actual_entry_price = price * (1 + self.slippage_pct / 100)
        else:  # SignalAction.SELL
            trade_action = "SHORT" 
            actual_entry_price = price * (1 - self.slippage_pct / 100)

        # Calculate quantity
        quantity = position_size_usd / actual_entry_price

        # Calculate stop loss and take profit prices
        stop_loss_price = None
        take_profit_price = None

        if stop_loss_pct is not None:
            if trade_action == "LONG":
                stop_loss_price = actual_entry_price * (1 - stop_loss_pct / 100)
            else:  # SHORT
                stop_loss_price = actual_entry_price * (1 + stop_loss_pct / 100)

        if take_profit_pct is not None:
            if trade_action == "LONG":
                take_profit_price = actual_entry_price * (1 + take_profit_pct / 100)
            else:  # SHORT
                take_profit_price = actual_entry_price * (1 - take_profit_pct / 100)

        # Create trade
        self.trade_counter += 1
        trade = BacktestTrade(
            trade_id=self.trade_counter,
            signal_type=signal,
            trade_action=trade_action,
            status=TradeStatus.OPEN,
            entry_time=timestamp,
            entry_price=actual_entry_price,
            entry_reason=entry_reason,
            quantity=quantity,
            position_size_usd=position_size_usd,
            leverage_used=leverage,
            margin_used=required_margin,
            confidence_score=confidence,
            commission_paid=entry_commission,
            slippage_cost=entry_slippage,
            strategy_context=strategy_context,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        # Update portfolio
        self.cash -= (required_margin + entry_commission + entry_slippage + entry_gst)
        self.open_positions[trade.trade_id] = trade

        # Track costs
        self.total_commission_paid += entry_commission
        self.total_slippage_paid += entry_slippage
        
        # Store GST in trade (half of total GST, other half paid on exit)
        trade.gst_paid = entry_gst

        logger.info(
            "Position opened",
            trade_id=trade.trade_id,
            action=trade_action,
            entry_price=actual_entry_price,
            quantity=quantity,
            position_size=position_size_usd,
            leverage=leverage,
            confidence=confidence,
        )

        return trade

    def close_position(
        self,
        trade_id: int,
        price: float,
        timestamp: datetime,
        exit_reason: str = "Strategy signal",
    ) -> Optional[BacktestTrade]:
        """
        Close an open position.

        Args:
            trade_id: Trade ID to close
            price: Exit price
            timestamp: Exit timestamp
            exit_reason: Reason for closing

        Returns:
            Closed trade if successful, None otherwise
        """
        if trade_id not in self.open_positions:
            logger.warning("Cannot close position: trade not found", trade_id=trade_id)
            return None

        trade = self.open_positions[trade_id]

        # Apply slippage to exit price
        if trade.trade_action == "LONG":
            actual_exit_price = price * (1 - self.slippage_pct / 100)
        else:  # SHORT
            actual_exit_price = price * (1 + self.slippage_pct / 100)

        # Close the trade (this calculates P&L)
        trade.close_trade(
            exit_time=timestamp,
            exit_price=actual_exit_price,
            exit_reason=exit_reason,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
        )

        # Update portfolio cash
        proceeds = trade.margin_used + (trade.net_pnl_usd or 0)
        self.cash += proceeds

        # Track costs
        leveraged_value = trade.position_size_usd * trade.leverage_used
        exit_commission = leveraged_value * (self.commission_pct / 100)
        exit_slippage = leveraged_value * (self.slippage_pct / 100)
        
        # Calculate GST on exit commission
        from src.core.constants import TradingConstants
        exit_gst = exit_commission * (TradingConstants.GST_RATE / 100)
        
        self.total_commission_paid += exit_commission
        self.total_slippage_paid += exit_slippage
        
        # Add exit GST to trade's total GST
        trade.gst_paid += exit_gst

        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_positions[trade_id]

        logger.info(
            "Position closed",
            trade_id=trade_id,
            exit_price=actual_exit_price,
            pnl_usd=trade.pnl_usd,
            pnl_pct=trade.pnl_pct,
            net_pnl=trade.net_pnl_usd,
            duration_minutes=trade.duration_minutes,
            exit_reason=exit_reason,
        )

        return trade

    def close_all_positions(
        self, price: float, timestamp: datetime, reason: str = "End of backtest"
    ) -> List[BacktestTrade]:
        """
        Close all open positions.

        Args:
            price: Current market price
            timestamp: Current timestamp
            reason: Reason for closing all positions

        Returns:
            List of closed trades
        """
        closed_trades = []
        open_trade_ids = list(self.open_positions.keys())

        for trade_id in open_trade_ids:
            trade = self.close_position(trade_id, price, timestamp, reason)
            if trade:
                closed_trades.append(trade)

        logger.info(
            "All positions closed",
            count=len(closed_trades),
            reason=reason,
            final_cash=self.cash,
        )

        return closed_trades

    def update_equity_history(
        self, timestamp: datetime, current_price: float, benchmark_value: Optional[float] = None
    ) -> EquityPoint:
        """
        Record current portfolio state in equity history.

        Args:
            timestamp: Current timestamp
            current_price: Current market price for unrealized P&L
            benchmark_value: Benchmark portfolio value (optional)

        Returns:
            EquityPoint with current portfolio state
        """
        # Update unrealized P&L for open positions
        for trade in self.open_positions.values():
            if trade.trade_action == "LONG":
                price_change = current_price - trade.entry_price
            else:  # SHORT
                price_change = trade.entry_price - current_price

            trade.pnl_usd = (price_change / trade.entry_price) * trade.position_size_usd
            trade.pnl_pct = (price_change / trade.entry_price) * 100

        # Calculate current equity
        equity = self.current_equity
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate drawdown
        drawdown_pct = ((self.peak_equity - equity) / self.peak_equity) * 100

        # Create equity point
        equity_point = EquityPoint(
            timestamp=timestamp,
            equity=equity,
            cash=self.cash,
            position_value=self.open_positions_value,
            drawdown_pct=drawdown_pct,
            benchmark_value=benchmark_value,
        )

        self.equity_history.append(equity_point)
        return equity_point

    def check_stop_loss_take_profit(
        self, current_price: float, timestamp: datetime
    ) -> List[BacktestTrade]:
        """
        Check if any open positions hit stop loss or take profit.

        Args:
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            List of trades that were closed
        """
        closed_trades = []
        trades_to_close = []

        for trade in self.open_positions.values():
            close_reason = None

            if trade.trade_action == "LONG":
                if (
                    trade.stop_loss_price is not None
                    and current_price <= trade.stop_loss_price
                ):
                    close_reason = "Stop Loss"
                elif (
                    trade.take_profit_price is not None
                    and current_price >= trade.take_profit_price
                ):
                    close_reason = "Take Profit"
            else:  # SHORT
                if (
                    trade.stop_loss_price is not None
                    and current_price >= trade.stop_loss_price
                ):
                    close_reason = "Stop Loss"
                elif (
                    trade.take_profit_price is not None
                    and current_price <= trade.take_profit_price
                ):
                    close_reason = "Take Profit"

            if close_reason:
                trades_to_close.append((trade.trade_id, close_reason))

        # Close triggered trades
        for trade_id, reason in trades_to_close:
            closed_trade = self.close_position(trade_id, current_price, timestamp, reason)
            if closed_trade:
                closed_trades.append(closed_trade)

        return closed_trades

    def get_portfolio_summary(self) -> Dict:
        """
        Get current portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        total_trades = len(self.closed_trades) + len(self.open_positions)
        winning_trades = len([t for t in self.closed_trades if t.is_winner is True])
        
        return {
            "current_equity": self.current_equity,
            "cash": self.cash,
            "open_positions_value": self.open_positions_value,
            "margin_used": self.margin_used,
            "available_margin": self.available_margin,
            "total_return_pct": self.total_return_pct,
            "unrealized_pnl": self.unrealized_pnl,
            "peak_equity": self.peak_equity,
            "current_drawdown_pct": ((self.peak_equity - self.current_equity) / self.peak_equity) * 100,
            "total_trades": total_trades,
            "open_trades": len(self.open_positions),
            "closed_trades": len(self.closed_trades),
            "winning_trades": winning_trades,
            "win_rate_pct": (winning_trades / len(self.closed_trades) * 100) if self.closed_trades else 0,
            "total_commission_paid": self.total_commission_paid,
            "total_slippage_paid": self.total_slippage_paid,
            "total_costs": self.total_commission_paid + self.total_slippage_paid,
        }