#!/usr/bin/env python3
"""
Test complete system with IST time formatting.
"""

import asyncio
from src.data.delta_client import DeltaExchangeClient
from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
from src.core.models import SessionConfig, StrategyType, Symbol, TimeFrame
from src.cli.displays import CLIDisplays
from rich.console import Console
from decimal import Decimal

async def test_complete_ist_system():
    console = Console()
    displays = CLIDisplays(console)
    
    console.print("🎯 Testing Complete System with IST Time Display")
    console.print("=" * 60)
    
    # Create session config
    config = SessionConfig(
        strategy=StrategyType.EMA_CROSSOVER,
        symbol=Symbol.BTCUSD,
        timeframe=TimeFrame.ONE_HOUR,
        stop_loss_pct=Decimal("2.0"),
        take_profit_pct=Decimal("4.0"),
        position_size_pct=Decimal("5.0"),
    )
    
    console.print("\n📋 Session Configuration:")
    displays.display_configuration_summary(config)
    
    async with DeltaExchangeClient() as client:
        console.print("\n📊 Fetching market data...")
        
        try:
            # Get market data
            market_data = await client.get_market_data(
                symbol=str(config.symbol),
                timeframe=str(config.timeframe),
                limit=50,
            )
            
            console.print(f"✅ Market data fetched: {len(market_data.ohlcv_data)} candles")
            
            # Display market data summary (includes IST candlestick analysis)
            displays.display_market_data_summary(market_data)
            
            # Run strategy analysis
            console.print("\n🧠 Running EMA Crossover Analysis...")
            strategy = EMACrossoverStrategy()
            
            analysis_result = await strategy.analyze(market_data, config)
            
            console.print(f"✅ Analysis completed")
            console.print(f"   Signals generated: {len(analysis_result.signals)}")
            
            # Display results with IST timestamps
            if analysis_result.signals:
                displays.display_trading_signals(analysis_result, config)
                
                # Show detailed signal timestamp info
                console.print("\n🕐 Signal Timestamp Details:")
                for i, signal in enumerate(analysis_result.signals[:3], 1):
                    console.print(f"  Signal #{i}:")
                    console.print(f"    IST Time: {signal.candle_time_display}")
                    console.print(f"    Formation: {signal.formation_display}")
                    if signal.candle_formation:
                        console.print(f"    Pattern: {signal.candle_formation.pattern_display_name}")
                        console.print(f"    Strength: {signal.candle_formation.strength}/10")
            
            console.print("\n✅ Complete system test with IST formatting successful!")
            
        except Exception as e:
            console.print(f"❌ Error: {e}")
            import traceback
            console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_complete_ist_system())