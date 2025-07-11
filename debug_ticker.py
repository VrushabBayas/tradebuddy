#!/usr/bin/env python3
"""
Debug script to check ticker data from Delta Exchange API.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data.delta_client import DeltaExchangeClient

async def debug_ticker():
    """Debug ticker data to understand the response format."""
    
    client = DeltaExchangeClient()
    
    try:
        print("🔍 Fetching ticker data...")
        ticker = await client.get_ticker("BTCUSDT")
        
        print(f"📊 Ticker type: {type(ticker)}")
        print(f"📊 Ticker length: {len(ticker) if isinstance(ticker, list) else 'N/A'}")
        
        if isinstance(ticker, list):
            print(f"\n🔍 Searching for BTC-related symbols...")
            btc_symbols = []
            for item in ticker:
                if isinstance(item, dict):
                    symbol = item.get('symbol', '')
                    if 'BTC' in symbol.upper():
                        btc_symbols.append({
                            'symbol': symbol,
                            'mark_price': item.get('mark_price', 'N/A'),
                            'close': item.get('close', 'N/A'),
                            'spot_price': item.get('spot_price', 'N/A'),
                            'price': item.get('price', 'N/A')
                        })
            
            print(f"📊 Found {len(btc_symbols)} BTC-related symbols:")
            for btc_symbol in btc_symbols[:5]:  # Show first 5
                print(f"  - {btc_symbol}")
        
        print(f"\n📊 First 3 ticker items:")
        if isinstance(ticker, list) and len(ticker) > 0:
            for i, item in enumerate(ticker[:3]):
                if isinstance(item, dict):
                    print(f"  Item {i+1}: {item.get('symbol', 'Unknown')} - Mark Price: {item.get('mark_price', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.close()

if __name__ == "__main__":
    success = asyncio.run(debug_ticker())
    sys.exit(0 if success else 1)