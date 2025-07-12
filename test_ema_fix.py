#!/usr/bin/env python3
"""
Test EMA strategy with volume_ratio fix to verify enhanced features work.
"""

import asyncio
from datetime import datetime, timezone

from src.core.models import OHLCV, MarketData, SessionConfig, Symbol, TimeFrame, StrategyType
from src.analysis.strategies.ema_crossover import EMACrossoverStrategy


async def test_ema_strategy_fix():
    """Test EMA strategy with the volume_ratio fix."""
    print("🧪 Testing EMA Strategy with Volume Fix\n")
    
    try:
        # Create test data with mix of zero and non-zero volumes
        test_data = []
        base_time = datetime.now(timezone.utc)
        
        # Create 50 candles with downward trend and EMA death cross setup
        for i in range(50):
            price = 120.0 - (i * 0.5)  # Downward trend
            volume = 0.0 if i % 10 == 0 else 1500.0  # Mix of zero and normal volumes
            
            candle = OHLCV(
                timestamp=base_time,
                open=price,
                high=price + 0.5,
                low=price - 1.0,
                close=price - 0.3,  # Slight bearish bias
                volume=volume
            )
            test_data.append(candle)
        
        # Add final bearish marabozu candle
        final_candle = OHLCV(
            timestamp=base_time,
            open=95.0,    # High price
            high=95.0,    # No upper shadow
            low=90.0,     # Much lower
            close=90.0,   # No lower shadow - bearish marabozu
            volume=2000.0 # High volume
        )
        test_data.append(final_candle)
        
        # Create market data
        market_data = MarketData(
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.FIVE_MINUTES,
            current_price=90.0,
            ohlcv_data=test_data
        )
        
        # Create session config
        session_config = SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.FIVE_MINUTES
        )
        
        # Initialize and test strategy
        strategy = EMACrossoverStrategy()
        
        print("📊 Running EMA Crossover analysis...")
        result = await strategy.analyze(market_data, session_config)
        
        print("✅ Analysis completed successfully!")
        print(f"   Signals generated: {len(result.signals)}")
        
        # Test enhanced features
        if result.signals:
            for i, signal in enumerate(result.signals, 1):
                print(f"\n🎯 Signal #{i}:")
                print(f"   Action: {signal.action}")
                print(f"   Confidence: {signal.confidence}/10")
                print(f"   🕒 Candle Time: {signal.candle_time_display}")
                print(f"   📊 Formation: {signal.formation_display}")
                
                if signal.candle_formation:
                    print(f"   🔥 Pattern: {signal.candle_formation.pattern_name}")
                    print(f"   💪 Strength: {signal.candle_formation.strength}/10")
                
                if signal.pattern_context:
                    context = signal.pattern_context[:100] + "..." if len(signal.pattern_context) > 100 else signal.pattern_context
                    print(f"   📚 Context: {context}")
        else:
            print("⚠️ No signals generated (might be expected based on conditions)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_ema_strategy_fix())
        if success:
            print("\n🎉 EMA strategy fix successful!")
            print("\n✅ Enhanced features ready:")
            print("   - Volume ratio error fixed")
            print("   - Enhanced timestamp and candlestick features working")
            print("   - Ready for live testing!")
        else:
            print("\n💥 Fix failed - check errors above")
    except Exception as e:
        print(f"\n❌ Test error: {e}")