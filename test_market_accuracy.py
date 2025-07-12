#!/usr/bin/env python3
"""
Test market data accuracy for USD symbols.
"""

import asyncio
from datetime import datetime, timezone
from src.data.delta_client import DeltaExchangeClient

async def test_market_accuracy():
    async with DeltaExchangeClient() as client:
        print(f"🕐 Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 60)
        
        symbols = ['BTCUSD', 'ETHUSD', 'BTCUSDT']
        
        for symbol in symbols:
            print(f"\n📊 Testing {symbol} (1h timeframe)...")
            try:
                market_data = await client.get_market_data(symbol, '1h', limit=5)
                
                print(f"   Current Price: ${market_data.current_price:,.2f}")
                print(f"   Latest Candle Time: {market_data.latest_ohlcv.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"   Latest OHLC: O=${market_data.latest_ohlcv.open:,.2f} H=${market_data.latest_ohlcv.high:,.2f} L=${market_data.latest_ohlcv.low:,.2f} C=${market_data.latest_ohlcv.close:,.2f}")
                print(f"   Volume: {market_data.latest_ohlcv.volume:,.0f}")
                
                # Show last 3 candles for context
                print(f"   Recent candles:")
                for i, candle in enumerate(market_data.ohlcv_data[:3]):
                    print(f"     {i+1}. {candle.timestamp.strftime('%H:%M')} UTC: ${candle.close:,.2f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Market data test completed")
        print("\nNote: Compare these prices with your Delta Exchange futures chart")
        print("      BTCUSD and ETHUSD should match your USD futures contracts")

if __name__ == "__main__":
    asyncio.run(test_market_accuracy())