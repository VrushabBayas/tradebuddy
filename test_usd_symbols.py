#!/usr/bin/env python3
"""
Test USD symbols compatibility with Delta Exchange API.
"""

import asyncio
from src.data.delta_client import DeltaExchangeClient

async def test_usd_symbols():
    async with DeltaExchangeClient() as client:
        # Test BTCUSD
        print('Testing BTCUSD symbol...')
        try:
            market_data = await client.get_market_data('BTCUSD', '1h', limit=10)
            print(f'✅ BTCUSD works - Current price: ${market_data.current_price:,.2f}')
            print(f'   Latest candle: {market_data.latest_ohlcv.timestamp} - Close: ${market_data.latest_ohlcv.close:,.2f}')
            print(f'   Candles received: {len(market_data.ohlcv_data)}')
        except Exception as e:
            print(f'❌ BTCUSD failed: {e}')
        
        # Test ETHUSD  
        print('\nTesting ETHUSD symbol...')
        try:
            market_data = await client.get_market_data('ETHUSD', '1h', limit=10)
            print(f'✅ ETHUSD works - Current price: ${market_data.current_price:,.2f}')
            print(f'   Latest candle: {market_data.latest_ohlcv.timestamp} - Close: ${market_data.latest_ohlcv.close:,.2f}')
            print(f'   Candles received: {len(market_data.ohlcv_data)}')
        except Exception as e:
            print(f'❌ ETHUSD failed: {e}')

        # Test old symbols for comparison
        print('\nTesting BTCUSDT (old) for comparison...')
        try:
            market_data = await client.get_market_data('BTCUSDT', '1h', limit=10)
            print(f'✅ BTCUSDT works - Current price: ${market_data.current_price:,.2f}')
            print(f'   Latest candle: {market_data.latest_ohlcv.timestamp} - Close: ${market_data.latest_ohlcv.close:,.2f}')
            print(f'   Candles received: {len(market_data.ohlcv_data)}')
        except Exception as e:
            print(f'❌ BTCUSDT failed: {e}')

        print('\n🔍 Checking if USD symbols are futures contracts...')
        try:
            products = await client.get_products()
            usd_products = [p for p in products if 'USD' in p.get('symbol', '') and 'BTC' in p.get('symbol', '')]
            for product in usd_products[:5]:  # First 5 USD products
                print(f"   {product.get('symbol')}: {product.get('product_type', 'unknown type')}")
        except Exception as e:
            print(f'❌ Products check failed: {e}')

if __name__ == "__main__":
    asyncio.run(test_usd_symbols())