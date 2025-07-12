#!/usr/bin/env python3
"""
Test IST time display with real market data.
"""

import asyncio
from src.data.delta_client import DeltaExchangeClient
from src.cli.displays import CLIDisplays
from rich.console import Console

async def test_ist_market_display():
    console = Console()
    displays = CLIDisplays(console)
    
    console.print("🕐 Testing IST Time Display with Real Market Data")
    console.print("=" * 60)
    
    async with DeltaExchangeClient() as client:
        # Test with BTCUSD
        console.print("\n📊 Fetching BTCUSD market data...")
        
        try:
            market_data = await client.get_market_data('BTCUSD', '1h', limit=5)
            
            console.print(f"\n✅ Market data fetched successfully")
            console.print(f"   Current Price: ${market_data.current_price:,.2f}")
            console.print(f"   Latest Candle UTC: {market_data.latest_ohlcv.timestamp}")
            
            # Test the displays with IST formatting
            console.print("\n🎯 Testing Market Data Summary Display (with IST):")
            displays.display_market_data_summary(market_data)
            
        except Exception as e:
            console.print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ist_market_display())