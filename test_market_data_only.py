#!/usr/bin/env python3
"""
Test script to verify market data retrieval with live prices.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data.delta_client import DeltaExchangeClient

async def test_market_data():
    """Test market data retrieval with live prices."""
    
    client = DeltaExchangeClient()
    
    try:
        print("🔍 Testing market data retrieval...")
        
        # Test BTC data
        btc_data = await client.get_market_data("BTCUSDT", "1h", 50)
        print(f"✅ BTC Market Data:")
        print(f"   - Symbol: {btc_data.symbol}")
        print(f"   - Current Price: ${btc_data.current_price:,.2f}")
        print(f"   - Candles: {len(btc_data.ohlcv_data)}")
        print(f"   - Timeframe: {btc_data.timeframe}")
        print(f"   - Last Candle Close: ${btc_data.ohlcv_data[-1].close:,.2f}")
        print()
        
        # Test ETH data
        eth_data = await client.get_market_data("ETHUSDT", "1h", 20)
        print(f"✅ ETH Market Data:")
        print(f"   - Symbol: {eth_data.symbol}")
        print(f"   - Current Price: ${eth_data.current_price:,.2f}")
        print(f"   - Candles: {len(eth_data.ohlcv_data)}")
        print()
        
        # Verify prices are reasonable
        if btc_data.current_price > 100000:  # BTC should be > $100k
            print("✅ BTC price looks correct (> $100k)")
        else:
            print("❌ BTC price seems wrong")
            
        if eth_data.current_price > 3000:  # ETH should be > $3k
            print("✅ ETH price looks correct (> $3k)")
        else:
            print("❌ ETH price seems wrong")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.close()

if __name__ == "__main__":
    success = asyncio.run(test_market_data())
    sys.exit(0 if success else 1)