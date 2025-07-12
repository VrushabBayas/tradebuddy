#!/usr/bin/env python3
"""
Test IST time display with candlestick pattern details.
"""

import asyncio
from src.data.delta_client import DeltaExchangeClient
from src.cli.displays import CLIDisplays
from rich.console import Console

async def test_ist_candlestick_display():
    console = Console()
    displays = CLIDisplays(console)
    
    console.print("🕯️ Testing IST Time Display with Candlestick Pattern Details")
    console.print("=" * 70)
    
    async with DeltaExchangeClient() as client:
        # Test with BTCUSD and more candles to trigger pattern analysis
        console.print("\n📊 Fetching BTCUSD market data (15 candles)...")
        
        try:
            market_data = await client.get_market_data('BTCUSD', '1h', limit=15)
            
            console.print(f"\n✅ Market data fetched successfully")
            console.print(f"   Current Price: ${market_data.current_price:,.2f}")
            console.print(f"   Candles: {len(market_data.ohlcv_data)}")
            console.print(f"   Latest Candle UTC: {market_data.latest_ohlcv.timestamp}")
            
            # Show raw UTC vs IST conversion for comparison
            from src.utils.helpers import format_ist_time
            utc_time = market_data.latest_ohlcv.timestamp
            ist_time = format_ist_time(utc_time, include_seconds=True)
            
            console.print(f"\n🕐 Time Conversion:")
            console.print(f"   UTC: {utc_time.strftime('%Y-%m-%d %I:%M:%S %p UTC')}")
            console.print(f"   IST: {ist_time}")
            
            # Test the displays with IST formatting (should show candlestick details now)
            console.print("\n🎯 Testing Market Data Summary Display (with IST candlestick details):")
            displays.display_market_data_summary(market_data)
            
        except Exception as e:
            console.print(f"❌ Error: {e}")
            import traceback
            console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_ist_candlestick_display())