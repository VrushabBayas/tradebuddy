"""
Professional trading metrics calculation for TradeBuddy backtesting.

Implements comprehensive performance metrics used in professional
trading analysis including risk-adjusted returns and drawdown analysis.
"""

import math
import structlog
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from src.backtesting.models import (
    BacktestTrade,
    DrawdownPeriod,
    EquityPoint,
    PerformanceMetrics,
)
from src.utils.helpers import to_float


logger = structlog.get_logger(__name__)


class MetricsCalculator:
    """
    Professional trading metrics calculator.
    
    Calculates comprehensive performance metrics following industry
    standards for trading strategy evaluation.
    """

    def __init__(
        self,
        trades: List[BacktestTrade],
        equity_history: List[EquityPoint],
        initial_capital: float,
        benchmark_returns: Optional[List[float]] = None,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    ):
        """
        Initialize metrics calculator.

        Args:
            trades: List of completed trades
            equity_history: Portfolio equity over time
            initial_capital: Starting capital
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.trades = trades
        self.equity_history = equity_history
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns or []
        self.risk_free_rate = risk_free_rate

        # Calculate derived data
        self.winning_trades = [t for t in trades if t.is_winner is True]
        self.losing_trades = [t for t in trades if t.is_winner is False]
        self.returns = self._calculate_returns()
        self.drawdown_periods = self._analyze_drawdown_periods()

        logger.debug(
            "Metrics calculator initialized",
            total_trades=len(trades),
            winning_trades=len(self.winning_trades),
            losing_trades=len(self.losing_trades),
            equity_points=len(equity_history),
        )

    def calculate_all_metrics(self) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Returns:
            Complete performance metrics
        """
        logger.info("Calculating comprehensive performance metrics")

        # Basic return metrics
        final_equity = self.equity_history[-1].equity if self.equity_history else self.initial_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Calculate time period for annualization
        if len(self.equity_history) >= 2:
            start_date = self.equity_history[0].timestamp
            end_date = self.equity_history[-1].timestamp
            total_days = (end_date - start_date).days
            years = max(total_days / 365.25, 1/365.25)  # Minimum 1 day
        else:
            years = 1.0

        annualized_return_pct = self._calculate_annualized_return(total_return_pct, years)
        cagr_pct = self._calculate_cagr(final_equity, self.initial_capital, years)

        # Risk metrics
        max_drawdown_pct = max((point.drawdown_pct for point in self.equity_history), default=0.0)
        volatility_pct = self._calculate_volatility()
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        calmar_ratio = self._calculate_calmar_ratio(annualized_return_pct, max_drawdown_pct)
        ulcer_index = self._calculate_ulcer_index()

        # Trading metrics
        total_trades = len(self.trades)
        win_rate_pct = (len(self.winning_trades) / total_trades * 100) if total_trades > 0 else 0
        loss_rate_pct = 100 - win_rate_pct
        profit_factor = self._calculate_profit_factor()
        expectancy = self._calculate_expectancy()

        # Trade analysis
        avg_win_pct = (
            sum(t.pnl_pct or 0 for t in self.winning_trades) / len(self.winning_trades)
            if self.winning_trades
            else 0
        )
        avg_loss_pct = (
            sum(t.pnl_pct or 0 for t in self.losing_trades) / len(self.losing_trades)
            if self.losing_trades
            else 0
        )
        largest_win_pct = max((t.pnl_pct or 0 for t in self.winning_trades), default=0)
        largest_loss_pct = min((t.pnl_pct or 0 for t in self.losing_trades), default=0)
        win_loss_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0

        # Streak analysis
        streaks = self._analyze_streaks()
        max_win_streak = streaks["max_win_streak"]
        max_loss_streak = streaks["max_loss_streak"]
        avg_win_streak = streaks["avg_win_streak"]
        avg_loss_streak = streaks["avg_loss_streak"]

        # Exposure metrics
        market_exposure_pct, avg_trade_duration_hours = self._calculate_exposure_metrics()

        # Drawdown analysis
        avg_drawdown_pct = (
            sum(point.drawdown_pct for point in self.equity_history) / len(self.equity_history)
            if self.equity_history
            else 0
        )
        recovery_factor = final_equity / self.initial_capital / (max_drawdown_pct / 100) if max_drawdown_pct > 0 else 0

        # Benchmark comparison
        benchmark_return_pct = None
        alpha = None
        beta = None
        tracking_error = None

        if self.benchmark_returns:
            benchmark_return_pct = self.benchmark_returns[-1] if self.benchmark_returns else 0
            alpha = total_return_pct - benchmark_return_pct
            beta, tracking_error = self._calculate_beta_and_tracking_error()

        return PerformanceMetrics(
            # Basic metrics
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            cagr_pct=cagr_pct,
            # Risk metrics
            max_drawdown_pct=max_drawdown_pct,
            volatility_pct=volatility_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            ulcer_index=ulcer_index,
            # Trading metrics
            total_trades=total_trades,
            win_rate_pct=win_rate_pct,
            loss_rate_pct=loss_rate_pct,
            profit_factor=profit_factor,
            expectancy=expectancy,
            # Trade analysis
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win_pct=largest_win_pct,
            largest_loss_pct=largest_loss_pct,
            win_loss_ratio=win_loss_ratio,
            # Streak analysis
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            avg_win_streak=avg_win_streak,
            avg_loss_streak=avg_loss_streak,
            # Exposure metrics
            market_exposure_pct=market_exposure_pct,
            avg_trade_duration_hours=avg_trade_duration_hours,
            # Drawdown analysis
            avg_drawdown_pct=avg_drawdown_pct,
            drawdown_periods=self.drawdown_periods,
            recovery_factor=recovery_factor,
            # Benchmark comparison
            benchmark_return_pct=benchmark_return_pct,
            alpha=alpha,
            beta=beta,
            tracking_error=tracking_error,
        )

    def _calculate_returns(self) -> List[float]:
        """Calculate period returns from equity history."""
        if len(self.equity_history) < 2:
            return []

        returns = []
        for i in range(1, len(self.equity_history)):
            prev_equity = self.equity_history[i - 1].equity
            curr_equity = self.equity_history[i].equity
            if prev_equity > 0:
                period_return = ((curr_equity - prev_equity) / prev_equity) * 100
                returns.append(period_return)

        return returns

    def _calculate_annualized_return(self, total_return_pct: float, years: float) -> float:
        """Calculate annualized return."""
        if years <= 0:
            return 0.0
        return ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100

    def _calculate_cagr(self, final_value: float, initial_value: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate."""
        if years <= 0 or initial_value <= 0:
            return 0.0
        return ((final_value / initial_value) ** (1 / years) - 1) * 100

    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self.returns) < 2:
            return 0.0

        mean_return = sum(self.returns) / len(self.returns)
        variance = sum((r - mean_return) ** 2 for r in self.returns) / (len(self.returns) - 1)
        std_dev = math.sqrt(variance)

        # Annualize volatility (assuming daily returns)
        # For other frequencies, this would need adjustment
        return std_dev * math.sqrt(252)  # 252 trading days per year

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0

        mean_return = sum(self.returns) / len(self.returns)
        volatility = self._calculate_volatility()

        if volatility == 0:
            return 0.0

        # Convert risk-free rate to period rate
        period_risk_free_rate = self.risk_free_rate / 252  # Daily rate

        excess_return = mean_return - period_risk_free_rate
        return (excess_return * 252) / volatility  # Annualized Sharpe

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(self.returns) < 2:
            return 0.0

        mean_return = sum(self.returns) / len(self.returns)
        negative_returns = [r for r in self.returns if r < 0]

        if not negative_returns:
            return float('inf')  # No downside risk

        downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_deviation = math.sqrt(downside_variance)

        if downside_deviation == 0:
            return 0.0

        # Annualized Sortino ratio
        annualized_downside_dev = downside_deviation * math.sqrt(252)
        period_risk_free_rate = self.risk_free_rate / 252
        
        excess_return = mean_return - period_risk_free_rate
        return (excess_return * 252) / annualized_downside_dev

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        return annualized_return / max_drawdown

    def _calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index (measure of downside volatility)."""
        if not self.equity_history:
            return 0.0

        squared_drawdowns = []
        for point in self.equity_history:
            squared_drawdowns.append(point.drawdown_pct ** 2)

        if not squared_drawdowns:
            return 0.0

        mean_squared_dd = sum(squared_drawdowns) / len(squared_drawdowns)
        return math.sqrt(mean_squared_dd)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl_usd or 0 for t in self.winning_trades)
        gross_loss = abs(sum(t.pnl_usd or 0 for t in self.losing_trades))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_expectancy(self) -> float:
        """Calculate expected value per trade."""
        if not self.trades:
            return 0.0

        total_pnl = sum(t.pnl_usd or 0 for t in self.trades)
        return total_pnl / len(self.trades)

    def _analyze_streaks(self) -> dict:
        """Analyze winning and losing streaks."""
        if not self.trades:
            return {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "avg_win_streak": 0.0,
                "avg_loss_streak": 0.0,
            }

        # Calculate streaks
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        win_streaks = []
        loss_streaks = []

        for trade in self.trades:
            if trade.is_winner is True:
                current_win_streak += 1
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
            elif trade.is_winner is False:
                current_loss_streak += 1
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0

            max_win_streak = max(max_win_streak, current_win_streak)
            max_loss_streak = max(max_loss_streak, current_loss_streak)

        # Add final streak
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)

        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "avg_win_streak": sum(win_streaks) / len(win_streaks) if win_streaks else 0.0,
            "avg_loss_streak": sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0.0,
        }

    def _calculate_exposure_metrics(self) -> Tuple[float, float]:
        """Calculate market exposure and average trade duration."""
        if not self.trades or not self.equity_history:
            return 0.0, 0.0

        # Calculate time in market
        total_duration_minutes = 0
        for trade in self.trades:
            if trade.duration_minutes:
                total_duration_minutes += trade.duration_minutes

        # Total backtest period
        if len(self.equity_history) >= 2:
            total_backtest_minutes = (
                self.equity_history[-1].timestamp - self.equity_history[0].timestamp
            ).total_seconds() / 60
        else:
            total_backtest_minutes = 1

        market_exposure_pct = (total_duration_minutes / total_backtest_minutes) * 100
        avg_trade_duration_hours = (
            total_duration_minutes / len(self.trades) / 60 if self.trades else 0
        )

        return market_exposure_pct, avg_trade_duration_hours

    def _analyze_drawdown_periods(self) -> List[DrawdownPeriod]:
        """Analyze detailed drawdown periods."""
        if not self.equity_history:
            return []

        drawdown_periods = []
        current_drawdown = None
        peak_equity = self.equity_history[0].equity

        for point in self.equity_history:
            # Update peak
            if point.equity > peak_equity:
                # End current drawdown if exists
                if current_drawdown:
                    current_drawdown.end_date = point.timestamp
                    current_drawdown.duration_days = (
                        current_drawdown.end_date - current_drawdown.start_date
                    ).days
                    
                    # Calculate recovery days (simplified)
                    current_drawdown.recovery_days = current_drawdown.duration_days
                    
                    drawdown_periods.append(current_drawdown)
                    current_drawdown = None

                peak_equity = point.equity

            # Check for new drawdown
            elif point.drawdown_pct > 0:
                if not current_drawdown:
                    current_drawdown = DrawdownPeriod(
                        start_date=point.timestamp,
                        peak_value=peak_equity,
                        trough_value=point.equity,
                        drawdown_pct=point.drawdown_pct,
                    )
                else:
                    # Update existing drawdown
                    if point.equity < current_drawdown.trough_value:
                        current_drawdown.trough_value = point.equity
                        current_drawdown.drawdown_pct = point.drawdown_pct

        # Close final drawdown if still open
        if current_drawdown:
            current_drawdown.end_date = self.equity_history[-1].timestamp
            current_drawdown.duration_days = (
                current_drawdown.end_date - current_drawdown.start_date
            ).days
            drawdown_periods.append(current_drawdown)

        logger.debug(
            "Drawdown analysis completed",
            total_drawdown_periods=len(drawdown_periods),
            max_drawdown_duration=max((dd.duration_days or 0 for dd in drawdown_periods), default=0),
        )

        return drawdown_periods

    def _calculate_beta_and_tracking_error(self) -> Tuple[Optional[float], Optional[float]]:
        """Calculate beta and tracking error vs benchmark."""
        if not self.benchmark_returns or len(self.returns) != len(self.benchmark_returns):
            return None, None

        # Calculate covariance and variance for beta
        n = len(self.returns)
        if n < 2:
            return None, None

        mean_strategy = sum(self.returns) / n
        mean_benchmark = sum(self.benchmark_returns) / n

        covariance = sum(
            (self.returns[i] - mean_strategy) * (self.benchmark_returns[i] - mean_benchmark)
            for i in range(n)
        ) / (n - 1)

        benchmark_variance = sum(
            (self.benchmark_returns[i] - mean_benchmark) ** 2 for i in range(n)
        ) / (n - 1)

        # Calculate beta
        beta = covariance / benchmark_variance if benchmark_variance != 0 else None

        # Calculate tracking error
        excess_returns = [
            self.returns[i] - self.benchmark_returns[i] for i in range(n)
        ]
        tracking_error = math.sqrt(
            sum(er ** 2 for er in excess_returns) / (n - 1)
        ) * math.sqrt(252)  # Annualized

        return beta, tracking_error