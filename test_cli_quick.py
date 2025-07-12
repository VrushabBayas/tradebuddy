#!/usr/bin/env python3
"""
Quick test for CLI backtesting integration.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.backtesting.reports import BacktestReportGenerator
from src.core.models import StrategyType, Symbol, TimeFrame


async def test_quick_backtest():
    """Quick test of backtesting functionality."""
    print("🧪 Quick CLI Backtesting Test")
    print("=" * 40)
    
    # Quick test with 7 days only
    config = BacktestConfig(
        strategy_type=StrategyType.EMA_CROSSOVER,
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.ONE_HOUR,
        days_back=7,  # Reduced for quick test
        initial_capital=850000.0,  # INR
        leverage=5,
        confidence_threshold=6,
        commission_pct=0.05,  # Delta Exchange India
        slippage_pct=0.05,
    )
    
    print(f"📊 Testing {config.strategy_type} strategy")
    print(f"⏱️ Period: {config.days_back} days")
    
    try:
        # Create and run backtesting engine
        engine = BacktestEngine(config)
        print(f"✅ Engine created successfully")
        
        # Run backtest
        result = await engine.run_backtest()
        print(f"✅ Backtest completed successfully")
        
        # Display key metrics
        metrics = result.performance_metrics
        print(f"📈 Total Return: {metrics.total_return_pct:.2f}%")
        print(f"📉 Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        print(f"🎯 Win Rate: {metrics.win_rate_pct:.1f}%")
        print(f"🔄 Total Trades: {metrics.total_trades}")
        print(f"📊 Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        
        # Generate HTML report
        report_generator = BacktestReportGenerator()
        report_path = report_generator.generate_report(result)
        print(f"📄 HTML report generated: {report_path}")
        
        print(f"\n✅ Quick Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_quick_backtest())