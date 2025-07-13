#!/usr/bin/env python3
"""
Analyze Delta Exchange position sizing and leverage mechanics.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data.delta_client import DeltaExchangeClient

async def analyze_delta_position():
    """Analyze Delta Exchange position sizing and contract details."""
    
    client = DeltaExchangeClient()
    
    try:
        print("🔍 Analyzing Delta Exchange BTC contract details...")
        
        # Get ticker data to analyze contract specifications
        ticker = await client.get_ticker("BTCUSD")
        
        # Find BTC perpetual contract
        btc_contract = None
        for item in ticker:
            if (isinstance(item, dict) and 
                item.get('symbol') == 'BTCUSDT' and 
                item.get('contract_type') == 'perpetual_futures'):
                btc_contract = item
                break
        
        if btc_contract:
            print(f"📊 BTC Perpetual Contract Details:")
            print(f"   - Symbol: {btc_contract.get('symbol')}")
            print(f"   - Contract Type: {btc_contract.get('contract_type')}")
            print(f"   - Mark Price: ${float(btc_contract.get('mark_price', 0)):,.2f}")
            print(f"   - Tick Size: {btc_contract.get('tick_size')}")
            print(f"   - Initial Margin: {btc_contract.get('initial_margin', 'N/A')}")
            print(f"   - Underlying Asset: {btc_contract.get('underlying_asset_symbol')}")
            print(f"   - Description: {btc_contract.get('description')}")
            print()
            
            # Calculate position sizing examples
            btc_price = float(btc_contract.get('mark_price', 118000))
            
            print("💰 Position Sizing Examples:")
            print(f"   BTC Price: ${btc_price:,.2f}")
            print()
            
            # Different account sizes
            account_sizes = [1000, 5000, 10000, 50000]
            position_percentages = [0.5, 1.0, 2.0, 5.0]
            
            print("📊 Position Size Analysis:")
            print(f"{'Account Size':<12} {'Pos %':<6} {'USD Value':<12} {'BTC Amount':<12} {'10x Leverage':<12}")
            print("-" * 70)
            
            for account_size in account_sizes:
                for pos_pct in position_percentages:
                    usd_value = account_size * (pos_pct / 100)
                    btc_amount = usd_value / btc_price
                    leverage_amount = btc_amount * 10  # 10x leverage
                    
                    print(f"${account_size:<11} {pos_pct:<5}% ${usd_value:<11.2f} {btc_amount:<11.6f} {leverage_amount:<11.4f}")
            
            print()
            print("🔧 Delta Exchange Mechanics:")
            print(f"   - With 10x leverage, your ${account_sizes[2]} account can control:")
            print(f"     • 1% position = ${account_sizes[2] * 0.01 * 10:,.2f} worth of BTC")
            print(f"     • 2% position = ${account_sizes[2] * 0.02 * 10:,.2f} worth of BTC")
            print(f"     • 5% position = ${account_sizes[2] * 0.05 * 10:,.2f} worth of BTC")
            print()
            
            # Lot size analysis
            print("📦 Lot Size Analysis:")
            print(f"   - You mentioned: 1 lot = 0.001 BTC")
            print(f"   - At current price: 1 lot = ${btc_price * 0.001:,.2f}")
            print(f"   - For $10,000 account with 1% position:")
            print(f"     • Position value: $100")
            print(f"     • BTC amount: {100 / btc_price:.6f} BTC")
            print(f"     • Equivalent lots: {(100 / btc_price) / 0.001:.2f} lots")
            
        else:
            print("❌ Could not find BTC perpetual contract")
            
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.close()

if __name__ == "__main__":
    success = asyncio.run(analyze_delta_position())
    sys.exit(0 if success else 1)
