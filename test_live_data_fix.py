#!/usr/bin/env python3
"""
Test script to verify live data fix.
This tests that we get current/recent candle data instead of stale historical data.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from src.data.delta_client import DeltaExchangeClient
from src.core.models import Symbol, TimeFrame
from src.cli.displays import CLIDisplays
from rich.console import Console

async def test_live_data_freshness():
    """Test that we get fresh/current candle data."""
    console = Console()
    displays = CLIDisplays(console)
    client = DeltaExchangeClient()
    
    try:
        print("🧪 Testing Live Data Freshness Fix")
        print("=" * 60)
        
        # Get current time for comparison
        current_time = datetime.now(timezone.utc)
        print(f"⏰ Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Fetch market data
        print("\n🔄 Fetching BTCUSDT 5m data from Delta Exchange...")
        market_data = await client.get_market_data(Symbol.BTCUSDT, TimeFrame.FIVE_MINUTES)
        
        if not market_data.ohlcv_data:
            print("❌ No candle data received!")
            return
            
        # Check data freshness
        latest_candle = market_data.latest_ohlcv
        if not latest_candle:
            print("❌ No latest candle found!")
            return
            
        print(f"\n📊 Latest Candle Data:")
        print(f"   Timestamp: {latest_candle.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   OHLC: O={latest_candle.open} H={latest_candle.high} L={latest_candle.low} C={latest_candle.close}")
        print(f"   Volume: {latest_candle.volume}")
        
        # Calculate time difference
        time_diff = current_time - latest_candle.timestamp
        print(f"\n⏱️  Time Difference: {time_diff}")
        
        # Verify freshness (should be within last hour for 5m timeframe)
        if time_diff > timedelta(hours=1):
            print(f"❌ STALE DATA DETECTED!")
            print(f"   Latest candle is {time_diff} old")
            print(f"   This indicates the fix didn't work properly")
        else:
            print(f"✅ FRESH DATA CONFIRMED!")
            print(f"   Latest candle is only {time_diff} old")
            print(f"   This is expected for live market data")
        
        # Test the immediate candlestick display
        print(f"\n🕯️  Testing Immediate Candlestick Display:")
        print("=" * 60)
        displays.display_market_data_summary(market_data)
        
        # Test comparison with oldest candle
        oldest_candle = market_data.ohlcv_data[-1]
        print(f"\n🔍 Data Order Verification:")
        print(f"   Latest candle (index 0): {latest_candle.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   Oldest candle (index -1): {oldest_candle.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        if latest_candle.timestamp > oldest_candle.timestamp:
            print("✅ Data ordering is correct (newest first)")
        else:
            print("❌ Data ordering is incorrect!")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_live_data_freshness())