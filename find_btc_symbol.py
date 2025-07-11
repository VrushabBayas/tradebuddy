#!/usr/bin/env python3
"""
Find the correct BTC perpetual symbol on Delta Exchange.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data.delta_client import DeltaExchangeClient

async def find_btc_symbol():
    """Find the correct BTC perpetual symbol."""
    
    client = DeltaExchangeClient()
    
    try:
        print("🔍 Fetching all ticker data...")
        ticker = await client.get_ticker("BTCUSDT")  # This returns all symbols
        
        print(f"🔍 Searching for BTC perpetual contracts...")
        
        # Look for perpetual contracts with BTC
        btc_perpetuals = []
        for item in ticker:
            if isinstance(item, dict):
                symbol = item.get('symbol', '')
                contract_type = item.get('contract_type', '')
                
                # Look for BTC perpetual contracts (not options)
                if ('BTC' in symbol.upper() and 
                    contract_type == 'perpetual_futures' and 
                    'USDT' in symbol.upper() and
                    '-' not in symbol):  # Exclude options like P-BTC-118000-110725
                    
                    btc_perpetuals.append({
                        'symbol': symbol,
                        'mark_price': item.get('mark_price', 'N/A'),
                        'close': item.get('close', 'N/A'),
                        'spot_price': item.get('spot_price', 'N/A'),
                        'contract_type': contract_type,
                        'description': item.get('description', 'N/A')
                    })
        
        print(f"📊 Found {len(btc_perpetuals)} BTC perpetual contracts:")
        for btc in btc_perpetuals:
            print(f"  - Symbol: {btc['symbol']}")
            print(f"    Mark Price: {btc['mark_price']}")
            print(f"    Close: {btc['close']}")
            print(f"    Description: {btc['description']}")
            print(f"    Contract Type: {btc['contract_type']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.close()

if __name__ == "__main__":
    success = asyncio.run(find_btc_symbol())
    sys.exit(0 if success else 1)