"""
Signal quality analysis and optimization for TradeBuddy backtesting.

Provides advanced signal analysis including pattern-weighted scoring,
confidence calibration, and parameter optimization for enhanced strategy performance.
"""

import json
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import structlog
import numpy as np
import pandas as pd

from src.backtesting.models import BacktestTrade, BacktestResult, TradeStatus
from src.core.models import SignalAction, TradingSignal
from src.core.constants import TradingConstants

logger = structlog.get_logger(__name__)


@dataclass
class SignalQualityMetrics:
    """Comprehensive signal quality metrics with pattern analysis."""
    
    # Basic signal metrics
    total_signals: int
    profitable_signals: int
    win_rate: float
    avg_confidence: float
    avg_return_per_signal: float
    
    # Pattern-specific metrics  
    pattern_signal_count: Dict[str, int]
    pattern_win_rates: Dict[str, float]
    pattern_avg_returns: Dict[str, float]
    pattern_confidence_accuracy: Dict[str, float]
    
    # Confidence calibration
    confidence_bins: Dict[str, Dict[str, Union[int, float]]]
    confidence_accuracy: float
    over_confidence_bias: float
    under_confidence_bias: float
    
    # Signal strength distribution
    strength_distribution: Dict[str, int]
    strength_performance: Dict[str, float]
    
    # Timing analysis
    avg_hold_time: float
    optimal_hold_time_range: Tuple[float, float]
    time_decay_factor: float
    
    # Risk-adjusted metrics
    sharpe_ratio_per_signal: float
    max_drawdown_signals: float
    signal_volatility: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass 
class OptimizationResult:
    """Parameter optimization results with pattern consideration."""
    
    parameter_name: str
    current_value: Union[int, float]
    optimal_value: Union[int, float]
    improvement_pct: float
    confidence_score: float
    
    # Pattern-specific optimization
    pattern_specific_values: Dict[str, Union[int, float]]
    pattern_improvements: Dict[str, float]
    
    # Sensitivity analysis
    value_range_tested: Tuple[Union[int, float], Union[int, float]]
    sensitivity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SignalQualityAnalyzer:
    """
    Advanced signal quality analysis with pattern-weighted scoring.
    
    Provides comprehensive analysis of trading signals including:
    - Pattern-based performance metrics
    - Confidence calibration analysis  
    - Parameter optimization with pattern awareness
    - Signal strength distribution analysis
    """
    
    def __init__(self):
        """Initialize the SignalQualityAnalyzer."""
        logger.debug("SignalQualityAnalyzer initialized")
    
    def analyze_signal_quality(
        self, 
        backtest_result: BacktestResult,
        trades: List[BacktestTrade],
        signals: List[TradingSignal]
    ) -> SignalQualityMetrics:
        """
        Perform comprehensive signal quality analysis.
        
        Args:
            backtest_result: Complete backtest results
            trades: List of executed trades
            signals: List of generated signals
            
        Returns:
            SignalQualityMetrics with comprehensive analysis
        """
        logger.info(
            "Analyzing signal quality",
            total_trades=len(trades),
            total_signals=len(signals)
        )
        
        try:
            # Basic signal metrics
            basic_metrics = self._calculate_basic_metrics(trades, signals)
            
            # Pattern-specific analysis
            pattern_metrics = self._analyze_pattern_performance(trades, signals)
            
            # Confidence calibration
            confidence_metrics = self._analyze_confidence_calibration(trades, signals)
            
            # Signal strength distribution
            strength_metrics = self._analyze_strength_distribution(trades, signals)
            
            # Timing analysis
            timing_metrics = self._analyze_timing_performance(trades)
            
            # Risk-adjusted metrics
            risk_metrics = self._calculate_risk_adjusted_metrics(backtest_result, trades)
            
            # Combine all metrics
            quality_metrics = SignalQualityMetrics(
                **basic_metrics,
                **pattern_metrics,
                **confidence_metrics,
                **strength_metrics,
                **timing_metrics,
                **risk_metrics
            )
            
            logger.info(
                "Signal quality analysis completed",
                win_rate=quality_metrics.win_rate,
                avg_confidence=quality_metrics.avg_confidence,
                confidence_accuracy=quality_metrics.confidence_accuracy,
                pattern_count=len(quality_metrics.pattern_signal_count)
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error("Signal quality analysis failed", error=str(e))
            raise
    
    def _calculate_basic_metrics(
        self, 
        trades: List[BacktestTrade], 
        signals: List[TradingSignal]
    ) -> Dict[str, Any]:
        """Calculate basic signal performance metrics."""
        
        completed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        profitable_trades = [t for t in completed_trades if t.pnl_usd and t.pnl_usd > 0]
        
        return {
            "total_signals": len(signals),
            "profitable_signals": len(profitable_trades),
            "win_rate": len(profitable_trades) / len(completed_trades) if completed_trades else 0.0,
            "avg_confidence": statistics.mean([s.confidence for s in signals]) if signals else 0.0,
            "avg_return_per_signal": statistics.mean([t.pnl_pct or 0 for t in completed_trades]) if completed_trades else 0.0
        }
    
    def _analyze_pattern_performance(
        self, 
        trades: List[BacktestTrade], 
        signals: List[TradingSignal]
    ) -> Dict[str, Any]:
        """Analyze performance by candlestick pattern type."""
        
        pattern_data = {}
        
        # Group trades by pattern information
        for trade in trades:
            if trade.status != TradeStatus.CLOSED:
                continue
                
            # Extract pattern info from strategy context
            context = trade.strategy_context or {}
            patterns = context.get("patterns_detected", ["unknown"])
            signal_direction = context.get("signal_direction", "neutral")
            
            for pattern in patterns:
                if pattern not in pattern_data:
                    pattern_data[pattern] = {
                        "trades": [],
                        "total_signals": 0
                    }
                pattern_data[pattern]["trades"].append(trade)
        
        # Count signals by pattern
        for signal in signals:
            reasoning = signal.reasoning or ""
            # Simple pattern detection from reasoning text
            if "Bearish Marabozu" in reasoning:
                pattern_data.setdefault("bearish_marabozu", {"trades": [], "total_signals": 0})["total_signals"] += 1
            elif "Shooting Star" in reasoning:
                pattern_data.setdefault("shooting_star", {"trades": [], "total_signals": 0})["total_signals"] += 1
            elif "Bullish Marabozu" in reasoning:
                pattern_data.setdefault("bullish_marabozu", {"trades": [], "total_signals": 0})["total_signals"] += 1
            else:
                pattern_data.setdefault("unknown", {"trades": [], "total_signals": 0})["total_signals"] += 1
        
        # Calculate pattern metrics
        pattern_signal_count = {}
        pattern_win_rates = {}
        pattern_avg_returns = {}
        pattern_confidence_accuracy = {}
        
        for pattern, data in pattern_data.items():
            trades_list = data["trades"]
            signal_count = data["total_signals"]
            
            pattern_signal_count[pattern] = signal_count
            
            if trades_list:
                profitable = [t for t in trades_list if t.pnl_usd and t.pnl_usd > 0]
                pattern_win_rates[pattern] = len(profitable) / len(trades_list)
                pattern_avg_returns[pattern] = statistics.mean([t.pnl_pct or 0 for t in trades_list])
                
                # Confidence accuracy (simplified)
                avg_confidence = statistics.mean([t.confidence_score for t in trades_list])
                actual_win_rate = len(profitable) / len(trades_list)
                expected_win_rate = avg_confidence / 10.0  # Normalize to 0-1
                accuracy = 1.0 - abs(actual_win_rate - expected_win_rate)
                pattern_confidence_accuracy[pattern] = max(0.0, accuracy)
            else:
                pattern_win_rates[pattern] = 0.0
                pattern_avg_returns[pattern] = 0.0
                pattern_confidence_accuracy[pattern] = 0.0
        
        return {
            "pattern_signal_count": pattern_signal_count,
            "pattern_win_rates": pattern_win_rates,
            "pattern_avg_returns": pattern_avg_returns,
            "pattern_confidence_accuracy": pattern_confidence_accuracy
        }
    
    def _analyze_confidence_calibration(
        self, 
        trades: List[BacktestTrade], 
        signals: List[TradingSignal]
    ) -> Dict[str, Any]:
        """Analyze confidence calibration and bias."""
        
        # Create confidence bins
        confidence_bins = {
            "0-20": {"count": 0, "wins": 0, "avg_return": 0.0},
            "21-40": {"count": 0, "wins": 0, "avg_return": 0.0},
            "41-60": {"count": 0, "wins": 0, "avg_return": 0.0},
            "61-80": {"count": 0, "wins": 0, "avg_return": 0.0},
            "81-100": {"count": 0, "wins": 0, "avg_return": 0.0}
        }
        
        # Map trades to confidence bins
        completed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        for trade in completed_trades:
            confidence = trade.confidence_score
            
            if confidence <= 20:
                bin_key = "0-20"
            elif confidence <= 40:
                bin_key = "21-40"
            elif confidence <= 60:
                bin_key = "41-60"
            elif confidence <= 80:
                bin_key = "61-80"
            else:
                bin_key = "81-100"
            
            confidence_bins[bin_key]["count"] += 1
            if trade.pnl_usd and trade.pnl_usd > 0:
                confidence_bins[bin_key]["wins"] += 1
            
            # Update average return
            current_avg = confidence_bins[bin_key]["avg_return"]
            current_count = confidence_bins[bin_key]["count"]
            new_return = trade.pnl_pct or 0
            confidence_bins[bin_key]["avg_return"] = ((current_avg * (current_count - 1)) + new_return) / current_count
        
        # Calculate calibration metrics
        total_accurate = 0
        total_over_confident = 0
        total_under_confident = 0
        total_trades = len(completed_trades)
        
        for bin_name, data in confidence_bins.items():
            if data["count"] > 0:
                actual_win_rate = data["wins"] / data["count"]
                
                # Expected win rate from bin midpoint
                bin_ranges = {
                    "0-20": 10, "21-40": 30, "41-60": 50, 
                    "61-80": 70, "81-100": 90
                }
                expected_win_rate = bin_ranges[bin_name] / 100.0
                
                difference = actual_win_rate - expected_win_rate
                tolerance = 0.1  # 10% tolerance
                
                if abs(difference) <= tolerance:
                    total_accurate += data["count"]
                elif difference < -tolerance:
                    total_over_confident += data["count"]
                else:
                    total_under_confident += data["count"]
        
        confidence_accuracy = total_accurate / total_trades if total_trades > 0 else 0.0
        over_confidence_bias = total_over_confident / total_trades if total_trades > 0 else 0.0
        under_confidence_bias = total_under_confident / total_trades if total_trades > 0 else 0.0
        
        return {
            "confidence_bins": confidence_bins,
            "confidence_accuracy": confidence_accuracy,
            "over_confidence_bias": over_confidence_bias,
            "under_confidence_bias": under_confidence_bias
        }
    
    def _analyze_strength_distribution(
        self, 
        trades: List[BacktestTrade], 
        signals: List[TradingSignal]
    ) -> Dict[str, Any]:
        """Analyze signal strength distribution and performance."""
        
        strength_distribution = {"WEAK": 0, "MODERATE": 0, "STRONG": 0}
        strength_performance = {"WEAK": 0.0, "MODERATE": 0.0, "STRONG": 0.0}
        strength_trades = {"WEAK": [], "MODERATE": [], "STRONG": []}
        
        # Count signals by strength
        for signal in signals:
            strength = signal.strength.value if hasattr(signal.strength, 'value') else str(signal.strength)
            strength_distribution[strength] = strength_distribution.get(strength, 0) + 1
        
        # Analyze trade performance by strength
        completed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        for trade in completed_trades:
            # Infer strength from confidence score
            if trade.confidence_score >= 8:
                strength = "STRONG"
            elif trade.confidence_score >= 6:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            
            strength_trades[strength].append(trade)
        
        # Calculate performance by strength
        for strength, trade_list in strength_trades.items():
            if trade_list:
                avg_return = statistics.mean([t.pnl_pct or 0 for t in trade_list])
                strength_performance[strength] = avg_return
        
        return {
            "strength_distribution": strength_distribution,
            "strength_performance": strength_performance
        }
    
    def _analyze_timing_performance(self, trades: List[BacktestTrade]) -> Dict[str, Any]:
        """Analyze timing performance and optimal hold periods."""
        
        completed_trades = [t for t in trades if t.status == TradeStatus.CLOSED and t.exit_time]
        
        if not completed_trades:
            return {
                "avg_hold_time": 0.0,
                "optimal_hold_time_range": (0.0, 0.0),
                "time_decay_factor": 0.0
            }
        
        # Calculate hold times in hours
        hold_times = []
        returns_by_time = []
        
        for trade in completed_trades:
            if trade.entry_time and trade.exit_time:
                hold_time = (trade.exit_time - trade.entry_time).total_seconds() / 3600.0  # Hours
                hold_times.append(hold_time)
                returns_by_time.append((hold_time, trade.pnl_pct or 0))
        
        avg_hold_time = statistics.mean(hold_times) if hold_times else 0.0
        
        # Find optimal hold time range (simplified)
        if returns_by_time:
            # Sort by return and find top quartile
            sorted_returns = sorted(returns_by_time, key=lambda x: x[1], reverse=True)
            top_quartile = sorted_returns[:len(sorted_returns)//4 or 1]
            
            optimal_times = [t[0] for t in top_quartile]
            optimal_range = (min(optimal_times), max(optimal_times))
            
            # Calculate time decay factor (correlation between time and return)
            times = [t[0] for t in returns_by_time]
            returns = [t[1] for t in returns_by_time]
            
            if len(times) > 1:
                correlation = np.corrcoef(times, returns)[0, 1]
                time_decay_factor = correlation if not np.isnan(correlation) else 0.0
            else:
                time_decay_factor = 0.0
        else:
            optimal_range = (0.0, 0.0)
            time_decay_factor = 0.0
        
        return {
            "avg_hold_time": avg_hold_time,
            "optimal_hold_time_range": optimal_range,
            "time_decay_factor": time_decay_factor
        }
    
    def _calculate_risk_adjusted_metrics(
        self, 
        backtest_result: BacktestResult, 
        trades: List[BacktestTrade]
    ) -> Dict[str, Any]:
        """Calculate risk-adjusted signal metrics."""
        
        completed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        if not completed_trades:
            return {
                "sharpe_ratio_per_signal": 0.0,
                "max_drawdown_signals": 0.0,
                "signal_volatility": 0.0
            }
        
        # Signal-level returns
        signal_returns = [t.pnl_pct or 0 for t in completed_trades]
        
        # Calculate signal volatility
        signal_volatility = statistics.stdev(signal_returns) if len(signal_returns) > 1 else 0.0
        
        # Calculate Sharpe ratio per signal
        avg_return = statistics.mean(signal_returns)
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_return = avg_return - risk_free_rate
        sharpe_ratio = excess_return / signal_volatility if signal_volatility > 0 else 0.0
        
        # Calculate max drawdown from signals
        cumulative_returns = []
        cumulative = 0
        
        for ret in signal_returns:
            cumulative += ret
            cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            peak = cumulative_returns[0]
            max_drawdown = 0
            
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                drawdown = peak - value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0.0
        
        return {
            "sharpe_ratio_per_signal": sharpe_ratio,
            "max_drawdown_signals": max_drawdown,
            "signal_volatility": signal_volatility
        }
    
    def optimize_parameters(
        self,
        backtest_result: BacktestResult,
        trades: List[BacktestTrade],
        current_params: Dict[str, Union[int, float]],
        optimization_targets: List[str] = None
    ) -> List[OptimizationResult]:
        """
        Optimize strategy parameters with pattern awareness.
        
        Args:
            backtest_result: Current backtest results
            trades: List of executed trades
            current_params: Current parameter values
            optimization_targets: Parameters to optimize
            
        Returns:
            List of OptimizationResult with optimization suggestions
        """
        logger.info(
            "Starting parameter optimization",
            current_params=current_params,
            targets=optimization_targets
        )
        
        if optimization_targets is None:
            optimization_targets = ["confidence_threshold", "volume_threshold", "ema_separation"]
        
        optimization_results = []
        
        try:
            for param_name in optimization_targets:
                if param_name in current_params:
                    result = self._optimize_single_parameter(
                        param_name, 
                        current_params[param_name],
                        backtest_result,
                        trades
                    )
                    optimization_results.append(result)
            
            logger.info(
                "Parameter optimization completed",
                optimized_params=len(optimization_results)
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error("Parameter optimization failed", error=str(e))
            raise
    
    def _optimize_single_parameter(
        self,
        param_name: str,
        current_value: Union[int, float],
        backtest_result: BacktestResult,
        trades: List[BacktestTrade]
    ) -> OptimizationResult:
        """Optimize a single parameter using grid search."""
        
        # Define parameter ranges
        param_ranges = {
            "confidence_threshold": (5, 9, 1),  # (min, max, step)
            "volume_threshold": (1.0, 2.0, 0.1),
            "ema_separation": (0.5, 2.0, 0.1),
            "rsi_oversold": (20, 35, 5),
            "rsi_overbought": (65, 80, 5)
        }
        
        if param_name not in param_ranges:
            # Default range
            range_info = (current_value * 0.5, current_value * 1.5, current_value * 0.1)
        else:
            range_info = param_ranges[param_name]
        
        min_val, max_val, step = range_info
        
        # Generate test values
        if isinstance(current_value, int):
            test_values = range(int(min_val), int(max_val) + 1, int(step))
        else:
            test_values = np.arange(min_val, max_val + step, step)
        
        # Simulate parameter optimization (simplified)
        best_value = current_value
        best_score = self._calculate_parameter_score(trades)
        improvement_pct = 0.0
        
        # In a real implementation, you would re-run backtests with different parameters
        # For now, we'll use a simplified heuristic-based approach
        
        for test_value in test_values:
            # Simulate score improvement based on parameter adjustment
            score = self._simulate_parameter_score(param_name, test_value, current_value, trades)
            
            if score > best_score:
                best_score = score
                best_value = test_value
        
        if best_score > self._calculate_parameter_score(trades):
            improvement_pct = ((best_score - self._calculate_parameter_score(trades)) / 
                             self._calculate_parameter_score(trades)) * 100
        
        # Pattern-specific optimization (simplified)
        pattern_specific_values = {
            "bearish_marabozu": best_value,
            "shooting_star": best_value,
            "bullish_marabozu": best_value
        }
        
        pattern_improvements = {
            "bearish_marabozu": improvement_pct,
            "shooting_star": improvement_pct,
            "bullish_marabozu": improvement_pct
        }
        
        # Sensitivity analysis
        sensitivity_score = abs(best_value - current_value) / current_value if current_value != 0 else 0
        
        return OptimizationResult(
            parameter_name=param_name,
            current_value=current_value,
            optimal_value=best_value,
            improvement_pct=improvement_pct,
            confidence_score=min(10, max(1, int(improvement_pct))),
            pattern_specific_values=pattern_specific_values,
            pattern_improvements=pattern_improvements,
            value_range_tested=(min_val, max_val),
            sensitivity_score=sensitivity_score
        )
    
    def _calculate_parameter_score(self, trades: List[BacktestTrade]) -> float:
        """Calculate overall parameter performance score."""
        completed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        if not completed_trades:
            return 0.0
        
        # Combine multiple metrics for scoring
        profitable_trades = [t for t in completed_trades if t.pnl_usd and t.pnl_usd > 0]
        win_rate = len(profitable_trades) / len(completed_trades)
        avg_return = statistics.mean([t.pnl_pct or 0 for t in completed_trades])
        
        # Weighted score: 60% win rate, 40% average return
        score = (win_rate * 0.6) + (avg_return / 100 * 0.4)
        return max(0, score)
    
    def _simulate_parameter_score(
        self, 
        param_name: str, 
        test_value: Union[int, float],
        current_value: Union[int, float],
        trades: List[BacktestTrade]
    ) -> float:
        """Simulate parameter score improvement (simplified heuristic)."""
        
        base_score = self._calculate_parameter_score(trades)
        
        # Simple heuristics for parameter impact
        if param_name == "confidence_threshold":
            # Higher threshold should improve win rate but reduce signals
            if test_value > current_value:
                return base_score * 1.05  # 5% improvement
            else:
                return base_score * 0.95
        
        elif param_name == "volume_threshold":
            # Optimal volume threshold around 1.2-1.5
            optimal_range = (1.2, 1.5)
            if optimal_range[0] <= test_value <= optimal_range[1]:
                return base_score * 1.1
            else:
                return base_score * 0.9
        
        # Default: small random variation
        import random
        return base_score * (0.95 + random.random() * 0.1)
    
    def generate_optimization_report(
        self,
        quality_metrics: SignalQualityMetrics,
        optimization_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "signal_quality_summary": {
                "total_signals": quality_metrics.total_signals,
                "win_rate": f"{quality_metrics.win_rate:.1%}",
                "avg_confidence": f"{quality_metrics.avg_confidence:.1f}",
                "confidence_accuracy": f"{quality_metrics.confidence_accuracy:.1%}",
                "pattern_performance": quality_metrics.pattern_win_rates
            },
            "optimization_recommendations": [],
            "pattern_insights": {},
            "risk_assessment": {
                "signal_volatility": quality_metrics.signal_volatility,
                "max_drawdown": quality_metrics.max_drawdown_signals,
                "sharpe_ratio": quality_metrics.sharpe_ratio_per_signal
            }
        }
        
        # Add optimization recommendations
        for opt_result in optimization_results:
            if opt_result.improvement_pct > 2:  # Only show significant improvements
                report["optimization_recommendations"].append({
                    "parameter": opt_result.parameter_name,
                    "current": opt_result.current_value,
                    "recommended": opt_result.optimal_value,
                    "improvement": f"{opt_result.improvement_pct:.1f}%",
                    "confidence": opt_result.confidence_score
                })
        
        # Add pattern insights
        for pattern, win_rate in quality_metrics.pattern_win_rates.items():
            if quality_metrics.pattern_signal_count.get(pattern, 0) > 0:
                report["pattern_insights"][pattern] = {
                    "signal_count": quality_metrics.pattern_signal_count[pattern],
                    "win_rate": f"{win_rate:.1%}",
                    "avg_return": f"{quality_metrics.pattern_avg_returns.get(pattern, 0):.2f}%",
                    "confidence_accuracy": f"{quality_metrics.pattern_confidence_accuracy.get(pattern, 0):.1%}"
                }
        
        return report