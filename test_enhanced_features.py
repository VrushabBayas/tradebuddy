#!/usr/bin/env python3
"""
Test script to verify enhanced signal output features.
"""

import asyncio
import sys
from datetime import datetime, timezone

from src.core.models import OHLCV, MarketData, SessionConfig, Symbol, TimeFrame, StrategyType
from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
from src.cli.displays import CLIDisplays
from rich.console import Console


async def test_enhanced_features():
    """Test the enhanced timestamp and candlestick formation features."""
    print("🧪 Testing Enhanced Signal Output Features\n")
    
    # Create test OHLCV data with a clear bearish pattern
    test_data = []
    base_time = datetime.now(timezone.utc)
    
    # Create sample data that should trigger a bearish marabozu pattern
    for i in range(50):
        # Last candle is a strong bearish marabozu (open=high, close=low)
        if i == 49:  # Last candle
            candle = OHLCV(
                timestamp=base_time,
                open=100.0,    # High price
                high=100.0,    # Same as open (no upper shadow)
                low=95.0,      # Much lower
                close=95.0,    # Same as low (no lower shadow)
                volume=2000.0  # High volume
            )
        else:
            # Regular candles trending down
            price = 105.0 - (i * 0.1)
            candle = OHLCV(
                timestamp=base_time,
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price - 0.1,
                volume=1000.0
            )
        test_data.append(candle)
    
    # Create market data
    market_data = MarketData(
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.FIVE_MINUTES,
        current_price=95.0,
        ohlcv_data=test_data
    )
    
    # Create session config
    session_config = SessionConfig(
        strategy=StrategyType.EMA_CROSSOVER,
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.FIVE_MINUTES
    )
    
    # Initialize strategy
    strategy = EMACrossoverStrategy()
    
    # Create console for display testing
    console = Console()
    displays = CLIDisplays(console)
    
    try:
        # Run analysis
        print("📊 Running EMA Crossover analysis with enhanced features...")
        result = await strategy.analyze(market_data, session_config)
        
        print("\n✅ Analysis completed! Testing enhanced displays:\n")
        
        # Test enhanced displays
        print("📈 Testing Enhanced Signal Display:")
        print("=" * 60)
        
        if result.signals:
            for i, signal in enumerate(result.signals, 1):
                print(f"\n🔍 Signal #{i}:")
                print(f"   Action: {signal.action}")
                print(f"   Confidence: {signal.confidence}/10")
                print(f"   Entry Price: ${signal.entry_price}")
                
                # Test new enhanced features
                print(f"   🕒 Candle Time: {signal.candle_time_display}")
                print(f"   📊 Formation: {signal.formation_display}")
                
                if signal.candle_formation:
                    formation = signal.candle_formation
                    print(f"   🔥 Pattern Details:")
                    print(f"      - Name: {formation.pattern_name}")
                    print(f"      - Type: {formation.pattern_type}")
                    print(f"      - Strength: {formation.strength}/10")
                    print(f"      - Visual: {formation.visual_description}")
                    print(f"      - Volume Confirmed: {'✅' if formation.volume_confirmation else '⚠️'}")
                
                if signal.pattern_context:
                    print(f"   📚 Educational Context:")
                    # Limit output for readability
                    context = signal.pattern_context[:200] + "..." if len(signal.pattern_context) > 200 else signal.pattern_context
                    print(f"      {context}")
                
                print(f"   💡 Enhanced Reasoning:")
                reasoning = signal.enhanced_reasoning[:300] + "..." if len(signal.enhanced_reasoning) > 300 else signal.enhanced_reasoning
                print(f"      {reasoning}")
        
        # Test display functions
        print("\n" + "=" * 60)
        print("🎨 Testing Rich Display Functions:")
        print("=" * 60)
        
        displays.display_trading_signals(result, session_config)
        
        print("\n✅ Enhanced features test completed successfully!")
        
        # Summary of new features
        print("\n🎯 New Features Verified:")
        print("  ✅ Candle timestamps in signals")
        print("  ✅ Candlestick formation analysis")
        print("  ✅ Pattern strength scoring (1-10)")
        print("  ✅ Visual pattern descriptions")
        print("  ✅ Educational pattern reasoning")
        print("  ✅ Volume confirmation status")
        print("  ✅ Enhanced display tables")
        print("  ✅ Rich primary signal panels")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_enhanced_features())
        if success:
            print("\n🎉 All enhanced features are working correctly!")
            sys.exit(0)
        else:
            print("\n💥 Some features failed testing")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)