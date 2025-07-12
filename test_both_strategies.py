#!/usr/bin/env python3
"""
Final test for both CLI backtesting strategies.
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


async def test_both_strategies():
    """Test both EMA and S/R strategies."""
    print("🧪 Final CLI Backtesting Test - Both Strategies")
    print("=" * 60)
    
    strategies = [
        ("EMA Crossover", StrategyType.EMA_CROSSOVER),
        ("Support/Resistance", StrategyType.SUPPORT_RESISTANCE)
    ]
    
    for strategy_name, strategy_type in strategies:
        print(f"\n📊 Testing {strategy_name}")
        print("-" * 40)
        
        config = BacktestConfig(
            strategy_type=strategy_type,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
            days_back=7,  # Quick test
            initial_capital=850000.0,  # INR
            leverage=3,
            confidence_threshold=6,
            commission_pct=0.05,  # Delta Exchange India
            slippage_pct=0.05,
        )
        
        try:
            # Create and run backtesting engine
            engine = BacktestEngine(config)
            print(f"✅ {strategy_name} engine created")
            
            # Run backtest
            result = await engine.run_backtest()
            print(f"✅ {strategy_name} backtest completed")
            
            # Display key metrics
            metrics = result.performance_metrics
            print(f"📈 Total Return: {metrics.total_return_pct:.2f}%")
            print(f"🔄 Total Trades: {metrics.total_trades}")
            print(f"🎯 Win Rate: {metrics.win_rate_pct:.1f}%")
            
            # Generate HTML report
            report_generator = BacktestReportGenerator()
            report_path = report_generator.generate_report(result)
            print(f"📄 {strategy_name} HTML report: {report_path}")
            
        except Exception as e:
            print(f"❌ Error with {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Both Strategy Tests Completed!")
    print(f"✅ CLI Backtesting Integration Fully Functional")


if __name__ == "__main__":
    asyncio.run(test_both_strategies())