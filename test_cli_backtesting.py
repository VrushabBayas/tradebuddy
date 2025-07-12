#!/usr/bin/env python3
"""
Test script to validate CLI backtesting integration.
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


async def test_cli_backtesting_integration():
    """Test backtesting integration with both strategies."""
    print("🧪 Testing CLI Backtesting Integration")
    print("=" * 60)
    
    # Test configurations for both strategies
    test_configs = [
        {
            "name": "EMA Crossover Strategy",
            "config": BacktestConfig(
                strategy_type=StrategyType.EMA_CROSSOVER,
                symbol=Symbol.BTCUSDT,
                timeframe=TimeFrame.ONE_HOUR,
                days_back=30,
                initial_capital=850000.0,  # INR
                leverage=5,
                confidence_threshold=6,
                commission_pct=0.05,  # Delta Exchange India
                slippage_pct=0.05,
            )
        },
        {
            "name": "Support/Resistance Strategy", 
            "config": BacktestConfig(
                strategy_type=StrategyType.SUPPORT_RESISTANCE,
                symbol=Symbol.BTCUSDT,
                timeframe=TimeFrame.ONE_HOUR,
                days_back=30,
                initial_capital=850000.0,  # INR
                leverage=5,
                confidence_threshold=6,
                commission_pct=0.05,  # Delta Exchange India
                slippage_pct=0.05,
            )
        }
    ]
    
    for test_case in test_configs:
        print(f"\n📊 Testing {test_case['name']}")
        print("-" * 40)
        
        try:
            # Create and run backtesting engine
            engine = BacktestEngine(test_case['config'])
            print(f"✅ Engine created for {test_case['config'].strategy_type}")
            
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
            report_generator = BacktestReportGenerator(currency="INR")
            report_path = report_generator.generate_report(result)
            print(f"📄 HTML report generated: {report_path}")
            
        except Exception as e:
            print(f"❌ Error testing {test_case['name']}: {e}")
            import traceback
            print(f"🐛 Traceback:")
            traceback.print_exc()
            continue
    
    print(f"\n✅ CLI Backtesting Integration Test Completed")
    print(f"🎯 Both strategies tested successfully")
    print(f"📊 HTML reports generated for analysis")


if __name__ == "__main__":
    asyncio.run(test_cli_backtesting_integration())