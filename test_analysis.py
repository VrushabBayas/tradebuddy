#!/usr/bin/env python3
"""
Test script to verify the analysis fixes without interactive CLI.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data.delta_client import DeltaExchangeClient
from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
from src.core.models import SessionConfig, StrategyType, Symbol, TimeFrame

async def test_analysis():
    """Test the analysis to verify fixes."""
    
    # Create client and strategy
    client = DeltaExchangeClient()
    strategy = EMACrossoverStrategy()
    
    # Create session config
    session_config = SessionConfig(
        strategy=StrategyType.EMA_CROSSOVER,
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.ONE_HOUR,
        confidence_threshold=6
    )
    
    try:
        # Test market data retrieval
        print("🔍 Testing market data retrieval...")
        market_data = await client.get_market_data("BTCUSDT", "1h", 50)
        print(f"✅ Market data retrieved: {len(market_data.ohlcv_data)} candles, current price: ${market_data.current_price}")
        
        # Test analysis
        print("🧠 Testing EMA Crossover analysis...")
        result = await strategy.analyze(market_data, session_config)
        print(f"✅ Analysis completed: {len(result.signals)} signals generated")
        
        if result.primary_signal:
            print(f"📊 Primary signal: {result.primary_signal.action} (confidence: {result.primary_signal.confidence}/10)")
        else:
            print("📊 No primary signal generated")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.close()
        await strategy.close()

if __name__ == "__main__":
    success = asyncio.run(test_analysis())
    sys.exit(0 if success else 1)