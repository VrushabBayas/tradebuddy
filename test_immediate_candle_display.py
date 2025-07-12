#!/usr/bin/env python3
"""
Test immediate candlestick display in Market Data Summary.
"""

from datetime import datetime, timezone
from rich.console import Console

from src.core.models import OHLCV, MarketData, Symbol, TimeFrame
from src.cli.displays import CLIDisplays


def test_immediate_candlestick_display():
    """Test the immediate candlestick display feature."""
    print("🧪 Testing Immediate Candlestick Display in Market Data Summary\n")
    
    # Create test OHLCV data with a bearish marabozu pattern
    test_data = []
    base_time = datetime.now(timezone.utc)
    
    # Add some regular candles first
    for i in range(15):
        candle = OHLCV(
            timestamp=base_time,
            open=100.0 + i,
            high=102.0 + i,
            low=98.0 + i,
            close=101.0 + i,
            volume=1500.0
        )
        test_data.append(candle)
    
    # Add a final bearish marabozu candle
    bearish_marabozu = OHLCV(
        timestamp=base_time,
        open=120.0,    # High price (open = high)
        high=120.0,    # Same as open
        low=115.0,     # Much lower
        close=115.0,   # Same as low (close = low)
        volume=2500.0  # High volume
    )
    test_data.append(bearish_marabozu)
    
    # Create market data
    market_data = MarketData(
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.FIVE_MINUTES,
        current_price=115.0,
        ohlcv_data=test_data
    )
    
    # Test the display
    console = Console()
    displays = CLIDisplays(console)
    
    print("📊 Testing Market Data Summary with Immediate Candlestick Analysis:\n")
    print("=" * 70)
    
    # This should now show the enhanced candlestick analysis
    displays.display_market_data_summary(market_data)
    
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("\nExpected to see:")
    print("  ✅ Standard market data table")
    print("  ✅ 🕯️ Current Candle Analysis section")
    print("  ✅ Candlestick Pattern Details table")
    print("  ✅ Bearish Marabozu pattern detection")
    print("  ✅ Pattern strength and description")
    print("  ✅ Educational context panel")
    
    return True


if __name__ == "__main__":
    try:
        test_immediate_candlestick_display()
        print("\n🎉 Immediate candlestick display test successful!")
        print("\n🔧 This feature will now show immediately after Delta Exchange data fetch!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()