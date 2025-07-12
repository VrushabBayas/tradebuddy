#!/usr/bin/env python3
"""
Test script for TradeBuddy backtesting module.

Quick test to verify the backtesting engine works with existing strategies.
"""

import asyncio
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/Users/vrushabhbayas/personal/tradebuddy')

from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.core.models import StrategyType, Symbol, TimeFrame


async def test_backtesting():
    """Test basic backtesting functionality."""
    print("🧪 Testing TradeBuddy Backtesting Module")
    
    # Create test configuration
    config = BacktestConfig(
        strategy_type=StrategyType.EMA_CROSSOVER,
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.ONE_HOUR,
        days_back=7,  # Minimum test period
        initial_capital=850000.0,  # INR equivalent
        leverage=5,  # Conservative leverage for testing
        confidence_threshold=6,
        commission_pct=0.05,  # Delta Exchange India futures taker
        slippage_pct=0.05,
    )
    
    print(f"📊 Configuration:")
    print(f"  Strategy: {config.strategy_type}")
    print(f"  Symbol: {config.symbol}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Period: {config.days_back} days")
    print(f"  Capital: ₹{config.initial_capital:,.0f}")
    print(f"  Leverage: {config.leverage}x")
    
    try:
        # Initialize backtesting engine
        print(f"\n🔧 Initializing backtesting engine...")
        engine = BacktestEngine(config)
        
        # Run backtest
        print(f"🚀 Running backtest...")
        start_time = datetime.now()
        
        result = await engine.run_backtest()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Display results
        print(f"\n✅ Backtest completed in {execution_time:.2f} seconds")
        print(f"\n📈 Results Summary:")
        print(f"  Total Return: {result.performance_metrics.total_return_pct:.2f}%")
        print(f"  Max Drawdown: {result.performance_metrics.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio: {result.performance_metrics.sharpe_ratio:.2f}")
        print(f"  Total Trades: {result.performance_metrics.total_trades}")
        print(f"  Win Rate: {result.performance_metrics.win_rate_pct:.1f}%")
        print(f"  Profit Factor: {result.performance_metrics.profit_factor:.2f}")
        
        print(f"\n💰 Portfolio Summary:")
        print(f"  Initial Capital: ${result.initial_capital:,.2f}")
        print(f"  Final Capital: ${result.final_capital:,.2f}")
        print(f"  Peak Capital: ${result.peak_capital:,.2f}")
        
        print(f"\n📊 Trading Summary:")
        print(f"  Signals Generated: {result.total_signals_generated}")
        print(f"  Signals Traded: {result.signals_traded}")
        print(f"  Conversion Rate: {result.signal_conversion_rate_pct:.1f}%")
        
        if result.trades:
            winning_trades = len([t for t in result.trades if t.is_winner])
            print(f"  Winning Trades: {winning_trades}")
            print(f"  Losing Trades: {len(result.trades) - winning_trades}")
            
            if result.trades:
                avg_duration = sum(t.duration_minutes or 0 for t in result.trades) / len(result.trades)
                print(f"  Avg Trade Duration: {avg_duration:.1f} minutes")
        
        print(f"\n🎯 Strategy Analysis:")
        print(f"  Strategy: {result.strategy_analysis.strategy_type}")
        print(f"  Signal Distribution: {result.strategy_analysis.signal_distribution}")
        print(f"  Confidence Distribution: {result.strategy_analysis.confidence_distribution}")
        
        print(f"\n📊 Benchmark Comparison:")
        print(f"  Benchmark Return: {result.benchmark_comparison.benchmark_return_pct:.2f}%")
        print(f"  Outperformance: {result.benchmark_comparison.outperformance_pct:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting TradeBuddy Backtesting Test")
    
    # Run test
    success = asyncio.run(test_backtesting())
    
    if success:
        print(f"\n🎉 Backtesting test completed successfully!")
    else:
        print(f"\n💥 Backtesting test failed!")
        sys.exit(1)