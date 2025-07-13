"""
Core backtesting engine for TradeBuddy.

Orchestrates the complete backtesting process including historical data loading,
strategy execution, portfolio management, and results analysis.
"""

import asyncio
import structlog
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Type

from src.analysis.strategies.base_strategy import BaseStrategy
from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
from src.analysis.strategies.support_resistance import SupportResistanceStrategy
from src.analysis.strategies.combined import CombinedStrategy
from src.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    BenchmarkComparison,
    PerformanceMetrics,
    StrategyAnalysis,
    TradeStatus,
)
from src.backtesting.portfolio import Portfolio
from src.backtesting.signal_quality import SignalQualityAnalyzer
from src.core.exceptions import StrategyError, DataValidationError
from src.core.models import (
    MarketData,
    OHLCV,
    SessionConfig,
    SignalAction,
    StrategyType,
    Symbol,
    TimeFrame,
)
from src.data.delta_client import DeltaExchangeClient
from src.utils.helpers import to_float


logger = structlog.get_logger(__name__)


class BacktestEngine:
    """
    Main backtesting engine.
    
    Orchestrates the complete backtesting workflow from data loading
    through strategy execution to performance analysis and reporting.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.strategy = self._create_strategy(config.strategy_type)
        self.delta_client = DeltaExchangeClient()
        self.portfolio = Portfolio(
            initial_capital=config.initial_capital,
            max_leverage=config.leverage,
            commission_pct=config.commission_pct,
            slippage_pct=config.slippage_pct,
        )

        # Results tracking
        self.historical_data: List[OHLCV] = []
        self.all_signals_generated = 0
        self.signals_traded = 0
        self.benchmark_returns: List[float] = []
        
        # Signal quality analysis
        self.signal_quality_analyzer = SignalQualityAnalyzer()
        self.all_generated_signals: List = []

        logger.info(
            "Backtesting engine initialized",
            strategy=config.strategy_type,
            symbol=config.symbol,
            timeframe=config.timeframe,
            days_back=config.days_back,
            initial_capital=config.initial_capital,
        )

    def _create_strategy(self, strategy_type: StrategyType) -> BaseStrategy:
        """Create strategy instance based on type."""
        strategy_map: Dict[StrategyType, Type[BaseStrategy]] = {
            StrategyType.EMA_CROSSOVER: EMACrossoverStrategy,
            StrategyType.EMA_CROSSOVER_V2: EMACrossoverV2Strategy,
            StrategyType.SUPPORT_RESISTANCE: SupportResistanceStrategy,
            StrategyType.COMBINED: CombinedStrategy,
        }

        if strategy_type not in strategy_map:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

        return strategy_map[strategy_type]()

    async def run_backtest(self) -> BacktestResult:
        """
        Execute complete backtesting workflow.

        Returns:
            Complete backtesting results
        """
        logger.info("Starting backtest execution")
        start_time = datetime.now(timezone.utc)

        try:
            # Phase 1: Load historical data
            logger.info("Loading historical data")
            await self._load_historical_data()

            # Phase 2: Execute strategy backtesting
            logger.info("Executing strategy backtesting")
            await self._execute_backtest()

            # Phase 3: Calculate performance metrics
            logger.info("Calculating performance metrics")
            performance_metrics = self._calculate_performance_metrics()

            # Phase 4: Generate strategy analysis
            logger.info("Generating strategy analysis")
            strategy_analysis = self._generate_strategy_analysis()

            # Phase 5: Create benchmark comparison
            logger.info("Creating benchmark comparison")
            benchmark_comparison = self._create_benchmark_comparison()

            # Phase 6: Generate signal quality analysis
            logger.info("Analyzing signal quality")
            signal_quality_metrics = await self._analyze_signal_quality()

            # Phase 7: Compile final results
            end_time = datetime.now(timezone.utc)
            execution_duration = (end_time - start_time).total_seconds()

            result = BacktestResult(
                config=self.config,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                total_duration_days=self.config.days_back,
                trades=self.portfolio.closed_trades,
                equity_curve=self.portfolio.equity_history,
                performance_metrics=performance_metrics,
                strategy_analysis=strategy_analysis,
                benchmark_comparison=benchmark_comparison,
                initial_capital=self.config.initial_capital,
                final_capital=self.portfolio.current_equity,
                peak_capital=self.portfolio.peak_equity,
                signal_quality_metrics=signal_quality_metrics.to_dict() if signal_quality_metrics else None,
                total_signals_generated=self.all_signals_generated,
                signals_traded=self.signals_traded,
                signal_conversion_rate_pct=(
                    (self.signals_traded / self.all_signals_generated * 100)
                    if self.all_signals_generated > 0
                    else 0.0
                ),
            )

            logger.info(
                "Backtest completed successfully",
                execution_time_seconds=execution_duration,
                total_trades=len(result.trades),
                final_equity=result.final_capital,
                total_return_pct=performance_metrics.total_return_pct,
                max_drawdown_pct=performance_metrics.max_drawdown_pct,
            )

            return result

        except Exception as e:
            logger.error("Backtest execution failed", error=str(e))
            raise StrategyError(f"Backtest execution failed: {str(e)}")

    async def _load_historical_data(self) -> None:
        """Load historical OHLCV data from Delta Exchange."""
        try:
            # Calculate required data points based on timeframe
            timeframe_minutes = self._get_timeframe_minutes(self.config.timeframe)
            total_minutes = self.config.days_back * 24 * 60
            required_candles = int(total_minutes / timeframe_minutes)

            # Add buffer for technical indicators
            required_candles += 100  # Buffer for indicators like EMA, RSI

            logger.info(
                "Fetching historical data",
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                required_candles=required_candles,
            )

            # Calculate start and end dates for historical data
            end_date = self.config.end_date
            start_date = self.config.start_date
            
            # Fetch historical data using get_candles method
            self.historical_data = await self.delta_client.get_candles(
                symbol=self.config.symbol.value,
                resolution=self.config.timeframe.value,
                start=start_date,
                end=end_date,
            )

            if not self.historical_data:
                raise DataValidationError("No historical data retrieved")

            if len(self.historical_data) < 50:
                raise DataValidationError(
                    f"Insufficient historical data: {len(self.historical_data)} candles"
                )

            # Reverse data to chronological order (Delta returns newest first)
            self.historical_data.reverse()

            # Calculate benchmark returns (buy and hold)
            first_price = self.historical_data[0].close
            for candle in self.historical_data:
                benchmark_return = ((candle.close - first_price) / first_price) * 100
                self.benchmark_returns.append(benchmark_return)

            logger.info(
                "Historical data loaded successfully",
                total_candles=len(self.historical_data),
                date_range=f"{self.historical_data[0].timestamp} to {self.historical_data[-1].timestamp}",
                first_price=first_price,
                last_price=self.historical_data[-1].close,
            )

        except Exception as e:
            logger.error("Failed to load historical data", error=str(e))
            raise DataValidationError(f"Historical data loading failed: {str(e)}")

    def _get_timeframe_minutes(self, timeframe: TimeFrame) -> int:
        """Get timeframe in minutes."""
        timeframe_map = {
            TimeFrame.ONE_MINUTE: 1,
            TimeFrame.FIVE_MINUTES: 5,
            TimeFrame.FIFTEEN_MINUTES: 15,
            TimeFrame.ONE_HOUR: 60,
            TimeFrame.FOUR_HOURS: 240,
            TimeFrame.ONE_DAY: 1440,
        }
        return timeframe_map.get(timeframe, 60)

    async def _execute_backtest(self) -> None:
        """Execute strategy backtesting on historical data."""
        logger.info("Starting strategy execution on historical data")

        # Get minimum data requirement from strategy
        min_periods = getattr(self.strategy, '_get_minimum_periods', lambda: 50)()
        
        # Start backtesting from minimum required data point
        for i in range(min_periods, len(self.historical_data)):
            current_candle = self.historical_data[i]
            current_price = current_candle.close
            timestamp = current_candle.timestamp

            # Prepare data slice for strategy analysis
            data_slice = self.historical_data[:i+1]  # Include current candle
            
            # Create MarketData object for strategy
            market_data = MarketData(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                current_price=current_price,
                ohlcv_data=data_slice,
                timestamp=timestamp,
            )

            # Create session config with backtest settings
            session_config = SessionConfig(
                strategy=self.config.strategy_type,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                limit=len(data_slice),
                confidence_threshold=self.config.confidence_threshold,
            )

            try:
                # Use strategy's backtesting analysis method (no AI calls)
                analysis_result = await self.strategy.analyze_for_backtesting(
                    market_data, session_config
                )
                
                # Process signals
                await self._process_signals(analysis_result, current_price, timestamp)

                # Check stop loss / take profit
                self.portfolio.check_stop_loss_take_profit(current_price, timestamp)

                # Update portfolio equity history
                benchmark_value = None
                if i < len(self.benchmark_returns):
                    benchmark_value = self.config.initial_capital * (
                        1 + self.benchmark_returns[i] / 100
                    )

                self.portfolio.update_equity_history(
                    timestamp, current_price, benchmark_value
                )

                # Log progress periodically
                if i % 100 == 0:
                    summary = self.portfolio.get_portfolio_summary()
                    logger.info(
                        "Backtest progress",
                        progress_pct=(i / len(self.historical_data)) * 100,
                        current_date=timestamp,
                        equity=summary["current_equity"],
                        return_pct=summary["total_return_pct"],
                        open_trades=summary["open_trades"],
                        total_trades=summary["total_trades"],
                    )

            except Exception as e:
                logger.warning(
                    "Strategy analysis failed for candle",
                    timestamp=timestamp,
                    price=current_price,
                    error=str(e),
                )
                continue

        # Close all remaining open positions at the end
        if self.historical_data:
            final_price = self.historical_data[-1].close
            final_timestamp = self.historical_data[-1].timestamp
            self.portfolio.close_all_positions(
                final_price, final_timestamp, "End of backtest"
            )

        logger.info(
            "Strategy execution completed",
            total_candles_processed=len(self.historical_data) - min_periods,
            total_signals=self.all_signals_generated,
            signals_traded=self.signals_traded,
            final_trades=len(self.portfolio.closed_trades),
        )

    async def _process_signals(
        self, analysis_result, current_price: float, timestamp: datetime
    ) -> None:
        """Process strategy signals and execute trades."""
        if not analysis_result.signals:
            return

        # Track all signals generated
        self.all_signals_generated += len(analysis_result.signals)
        
        # Store signals for quality analysis
        self.all_generated_signals.extend(analysis_result.signals)

        # Filter signals by confidence threshold
        valid_signals = [
            signal
            for signal in analysis_result.signals
            if signal.confidence >= self.config.confidence_threshold
        ]

        if not valid_signals:
            return

        # Process each valid signal
        for signal in valid_signals:
            # Skip neutral and wait signals for actual trading
            if signal.action in [SignalAction.NEUTRAL, SignalAction.WAIT]:
                continue

            # Check if we already have a position in the same direction
            same_direction_positions = [
                trade
                for trade in self.portfolio.open_positions.values()
                if (
                    signal.action == SignalAction.BUY
                    and trade.trade_action == "LONG"
                )
                or (
                    signal.action == SignalAction.SELL
                    and trade.trade_action == "SHORT"
                )
            ]

            if same_direction_positions:
                continue  # Skip if already have position in same direction

            # Calculate position size (respect max position size limit)
            max_position_usd = self.config.initial_capital * (
                self.config.max_position_size_pct / 100
            )
            available_margin = self.portfolio.available_margin
            position_size_usd = min(max_position_usd, available_margin * 0.95)  # Leave 5% buffer

            if position_size_usd < self.config.initial_capital * 0.01:  # Minimum 1% of capital
                continue

            # Prepare strategy context
            strategy_context = {
                "signal_reason": signal.reasoning,
                "market_conditions": getattr(analysis_result, "market_conditions", ""),
                "technical_context": getattr(analysis_result, "technical_analysis", {}),
            }

            # Add strategy-specific context
            if hasattr(analysis_result, "ema_crossover"):
                strategy_context["ema_crossover"] = analysis_result.ema_crossover
            if hasattr(analysis_result, "support_resistance_levels"):
                strategy_context["sr_levels"] = analysis_result.support_resistance_levels

            # Open position
            trade = self.portfolio.open_position(
                signal=signal.action,
                price=current_price,
                timestamp=timestamp,
                confidence=signal.confidence,
                position_size_usd=position_size_usd,
                leverage=self.config.leverage,
                strategy_context=strategy_context,
                entry_reason=f"Strategy signal: {signal.reasoning}",
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
            )

            if trade:
                self.signals_traded += 1
                logger.info(
                    "Signal executed",
                    signal_action=signal.action,
                    confidence=signal.confidence,
                    trade_id=trade.trade_id,
                    position_size=position_size_usd,
                    entry_price=current_price,
                )

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating performance metrics")

        # Import metrics calculator here to avoid circular imports
        from src.backtesting.metrics import MetricsCalculator

        calculator = MetricsCalculator(
            trades=self.portfolio.closed_trades,
            equity_history=self.portfolio.equity_history,
            initial_capital=self.config.initial_capital,
            benchmark_returns=self.benchmark_returns,
        )

        return calculator.calculate_all_metrics()

    def _generate_strategy_analysis(self) -> StrategyAnalysis:
        """Generate strategy-specific analysis."""
        logger.info("Generating strategy analysis")

        # Analyze signal distribution
        signal_distribution = {}
        confidence_distribution = {}
        
        for trade in self.portfolio.closed_trades:
            # Signal type distribution - handle both enum and string cases
            signal_type = trade.signal_type.value if hasattr(trade.signal_type, 'value') else str(trade.signal_type)
            signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1

            # Confidence distribution
            confidence = trade.confidence_score
            confidence_distribution[confidence] = confidence_distribution.get(confidence, 0) + 1

        # Strategy-specific metrics
        strategy_metrics = {}
        
        if self.config.strategy_type == StrategyType.EMA_CROSSOVER:
            strategy_metrics = self._analyze_ema_strategy()
        elif self.config.strategy_type == StrategyType.SUPPORT_RESISTANCE:
            strategy_metrics = self._analyze_sr_strategy()
        elif self.config.strategy_type == StrategyType.COMBINED:
            strategy_metrics = self._analyze_combined_strategy()

        # Analyze filter effectiveness (placeholder data for now)
        filter_effectiveness = {
            "rsi_filter": {"enabled": self.config.enable_rsi_filter, "impact": 0.1},
            "volume_filter": {"enabled": self.config.enable_volume_filter, "impact": 0.2},
            "candlestick_filter": {"enabled": self.config.enable_candlestick_filter, "impact": 0.0},
        }

        return StrategyAnalysis(
            strategy_type=self.config.strategy_type,
            signal_distribution=signal_distribution,
            confidence_distribution=confidence_distribution,
            filter_effectiveness=filter_effectiveness,
            strategy_metrics=strategy_metrics,
            best_conditions=self._find_best_conditions(),
            worst_conditions=self._find_worst_conditions(),
        )

    def _analyze_ema_strategy(self) -> Dict:
        """Analyze EMA crossover strategy specific metrics."""
        golden_cross_trades = []
        death_cross_trades = []

        for trade in self.portfolio.closed_trades:
            if "ema_crossover" in trade.strategy_context:
                ema_data = trade.strategy_context["ema_crossover"]
                if isinstance(ema_data, dict) and ema_data.get("is_golden_cross"):
                    golden_cross_trades.append(trade)
                else:
                    death_cross_trades.append(trade)

        golden_cross_avg_return = (
            sum(t.pnl_pct or 0 for t in golden_cross_trades) / len(golden_cross_trades)
            if golden_cross_trades
            else 0
        )
        
        death_cross_avg_return = (
            sum(t.pnl_pct or 0 for t in death_cross_trades) / len(death_cross_trades)
            if death_cross_trades
            else 0
        )

        return {
            "golden_cross_trades": len(golden_cross_trades),
            "death_cross_trades": len(death_cross_trades),
            "golden_cross_avg_return": golden_cross_avg_return,
            "death_cross_avg_return": death_cross_avg_return,
            "golden_cross_win_rate": (
                len([t for t in golden_cross_trades if t.is_winner]) / len(golden_cross_trades) * 100
                if golden_cross_trades
                else 0
            ),
            "death_cross_win_rate": (
                len([t for t in death_cross_trades if t.is_winner]) / len(death_cross_trades) * 100
                if death_cross_trades
                else 0
            ),
        }

    def _analyze_sr_strategy(self) -> Dict:
        """Analyze support/resistance strategy specific metrics."""
        support_trades = []
        resistance_trades = []

        for trade in self.portfolio.closed_trades:
            if trade.trade_action == "LONG":
                support_trades.append(trade)
            else:
                resistance_trades.append(trade)

        return {
            "support_bounce_trades": len(support_trades),
            "resistance_rejection_trades": len(resistance_trades),
            "support_avg_return": (
                sum(t.pnl_pct or 0 for t in support_trades) / len(support_trades)
                if support_trades
                else 0
            ),
            "resistance_avg_return": (
                sum(t.pnl_pct or 0 for t in resistance_trades) / len(resistance_trades)
                if resistance_trades
                else 0
            ),
        }

    def _analyze_combined_strategy(self) -> Dict:
        """Analyze combined strategy specific metrics."""
        return {
            "total_combined_signals": len(self.portfolio.closed_trades),
            "avg_confidence": (
                sum(t.confidence_score for t in self.portfolio.closed_trades) / len(self.portfolio.closed_trades)
                if self.portfolio.closed_trades
                else 0
            ),
        }


    def _find_best_conditions(self) -> Dict:
        """Find best performing market conditions."""
        if not self.portfolio.closed_trades:
            return {}

        best_trade = max(self.portfolio.closed_trades, key=lambda t: t.pnl_pct or 0)
        return {
            "best_trade_return": best_trade.pnl_pct,
            "best_trade_confidence": best_trade.confidence_score,
            "best_trade_duration": best_trade.duration_minutes,
        }

    def _find_worst_conditions(self) -> Dict:
        """Find worst performing market conditions."""
        if not self.portfolio.closed_trades:
            return {}

        worst_trade = min(self.portfolio.closed_trades, key=lambda t: t.pnl_pct or 0)
        return {
            "worst_trade_return": worst_trade.pnl_pct,
            "worst_trade_confidence": worst_trade.confidence_score,
            "worst_trade_duration": worst_trade.duration_minutes,
        }

    def _create_benchmark_comparison(self) -> BenchmarkComparison:
        """Create benchmark comparison analysis."""
        if not self.benchmark_returns:
            # Fallback if no benchmark data
            return BenchmarkComparison(
                benchmark_return_pct=0.0,
                strategy_return_pct=self.portfolio.total_return_pct,
                outperformance_pct=self.portfolio.total_return_pct,
                benchmark_sharpe=0.0,
                strategy_sharpe=0.0,
                sharpe_improvement=0.0,
                benchmark_max_dd_pct=0.0,
                strategy_max_dd_pct=0.0,
                drawdown_improvement_pct=0.0,
            )

        # Calculate benchmark metrics
        benchmark_final_return = self.benchmark_returns[-1] if self.benchmark_returns else 0.0
        strategy_return = self.portfolio.total_return_pct

        # Simple Sharpe calculation (would need risk-free rate for accurate calculation)
        benchmark_volatility = self._calculate_volatility(self.benchmark_returns)
        benchmark_sharpe = benchmark_final_return / benchmark_volatility if benchmark_volatility > 0 else 0

        # Calculate strategy Sharpe from equity curve
        returns = []
        for i in range(1, len(self.portfolio.equity_history)):
            prev_equity = self.portfolio.equity_history[i-1].equity
            curr_equity = self.portfolio.equity_history[i].equity
            ret = ((curr_equity - prev_equity) / prev_equity) * 100
            returns.append(ret)

        strategy_volatility = self._calculate_volatility(returns)
        strategy_sharpe = strategy_return / strategy_volatility if strategy_volatility > 0 else 0

        # Calculate benchmark max drawdown
        benchmark_max_dd = self._calculate_max_drawdown(self.benchmark_returns)
        strategy_max_dd = max(point.drawdown_pct for point in self.portfolio.equity_history) if self.portfolio.equity_history else 0.0

        return BenchmarkComparison(
            benchmark_return_pct=benchmark_final_return,
            strategy_return_pct=strategy_return,
            outperformance_pct=strategy_return - benchmark_final_return,
            benchmark_sharpe=benchmark_sharpe,
            strategy_sharpe=strategy_sharpe,
            sharpe_improvement=strategy_sharpe - benchmark_sharpe,
            benchmark_max_dd_pct=benchmark_max_dd,
            strategy_max_dd_pct=strategy_max_dd,
            drawdown_improvement_pct=benchmark_max_dd - strategy_max_dd,
        )

    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility from returns."""
        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        volatility = variance ** 0.5

        # Annualize based on timeframe
        periods_per_year = self._get_periods_per_year()
        return volatility * (periods_per_year ** 0.5)

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        equity = 100.0  # Start with 100%
        peak = 100.0
        max_dd = 0.0

        for ret in returns:
            equity *= (1 + ret / 100)
            if equity > peak:
                peak = equity
            drawdown = ((peak - equity) / peak) * 100
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on timeframe."""
        timeframe_map = {
            TimeFrame.ONE_MINUTE: 525600,  # 365 * 24 * 60
            TimeFrame.FIVE_MINUTES: 105120,  # 365 * 24 * 12
            TimeFrame.FIFTEEN_MINUTES: 35040,  # 365 * 24 * 4
            TimeFrame.ONE_HOUR: 8760,  # 365 * 24
            TimeFrame.FOUR_HOURS: 2190,  # 365 * 6
            TimeFrame.ONE_DAY: 365,
        }
        return timeframe_map.get(self.config.timeframe, 8760)

    async def _analyze_signal_quality(self):
        """Analyze signal quality and generate optimization recommendations."""
        try:
            if not self.all_generated_signals or not self.portfolio.closed_trades:
                logger.warning("Insufficient data for signal quality analysis")
                return None

            # Create a dummy BacktestResult for signal quality analysis
            temp_result = BacktestResult(
                config=self.config,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                total_duration_days=self.config.days_back,
                trades=self.portfolio.closed_trades,
                equity_curve=self.portfolio.equity_history,
                performance_metrics=PerformanceMetrics(
                    total_return_pct=self.portfolio.total_return_pct,
                    annual_return_pct=0.0,  # Will be calculated
                    total_trades=len(self.portfolio.closed_trades),
                    winning_trades=0,  # Will be calculated
                    win_rate_pct=0.0,  # Will be calculated
                    avg_win_pct=0.0,
                    avg_loss_pct=0.0,
                    profit_factor=0.0,
                    max_drawdown_pct=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    calmar_ratio=0.0,
                    volatility_pct=0.0,
                    skewness=0.0,
                    kurtosis=0.0,
                    var_95_pct=0.0,
                    cvar_95_pct=0.0,
                    recovery_time_days=0
                ),
                strategy_analysis=StrategyAnalysis(
                    avg_trade_duration_minutes=0,
                    best_trade_pnl_pct=0.0,
                    worst_trade_pnl_pct=0.0,
                    consecutive_wins=0,
                    consecutive_losses=0,
                    avg_bars_held=0,
                    best_trade_duration=0,
                    worst_trade_duration=0
                ),
                benchmark_comparison=BenchmarkComparison(
                    benchmark_return_pct=0.0,
                    strategy_return_pct=0.0,
                    outperformance_pct=0.0,
                    benchmark_sharpe=0.0,
                    strategy_sharpe=0.0,
                    sharpe_improvement=0.0,
                    benchmark_max_dd_pct=0.0,
                    strategy_max_dd_pct=0.0,
                    drawdown_improvement_pct=0.0
                ),
                initial_capital=self.config.initial_capital,
                final_capital=self.portfolio.current_equity,
                peak_capital=self.portfolio.peak_equity
            )

            # Perform signal quality analysis
            signal_quality_metrics = self.signal_quality_analyzer.analyze_signal_quality(
                temp_result, self.portfolio.closed_trades, self.all_generated_signals
            )

            # Perform parameter optimization
            current_params = {
                "confidence_threshold": self.config.confidence_threshold,
                "leverage": self.config.leverage,
                "max_position_size_pct": self.config.max_position_size_pct
            }

            optimization_results = self.signal_quality_analyzer.optimize_parameters(
                temp_result, self.portfolio.closed_trades, current_params
            )

            # Store optimization results in the signal quality metrics
            signal_quality_metrics.optimization_results = [opt.to_dict() for opt in optimization_results]

            logger.info(
                "Signal quality analysis completed",
                total_signals=len(self.all_generated_signals),
                total_trades=len(self.portfolio.closed_trades),
                win_rate=signal_quality_metrics.win_rate,
                confidence_accuracy=signal_quality_metrics.confidence_accuracy,
                optimization_recommendations=len(optimization_results)
            )

            return signal_quality_metrics

        except Exception as e:
            logger.error("Signal quality analysis failed", error=str(e))
            return None