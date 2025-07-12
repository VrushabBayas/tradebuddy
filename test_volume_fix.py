#!/usr/bin/env python3
"""
Quick test to verify volume validation fix works with Delta Exchange data.
"""

import asyncio
from datetime import datetime, timezone

from src.core.models import OHLCV, MarketData, Symbol, TimeFrame
from src.analysis.indicators import TechnicalIndicators


def test_zero_volume_validation():
    """Test that zero volume is now accepted."""
    print("🧪 Testing Zero Volume Validation Fix\n")
    
    try:
        # Test creating OHLCV with zero volume
        test_candle = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=0.0  # This should now be allowed
        )
        print("✅ OHLCV with volume=0 accepted")
        
        # Test volume analysis with zero volume data
        test_data = []
        for i in range(20):
            candle = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=100.0 + i,
                high=102.0 + i,
                low=98.0 + i,
                close=101.0 + i,
                volume=0.0 if i % 5 == 0 else 1000.0  # Mix of zero and non-zero volumes
            )
            test_data.append(candle)
        
        # Test technical indicators with mixed volume data
        tech_indicators = TechnicalIndicators()
        volume_analysis = tech_indicators.analyze_volume(test_data)
        
        print("✅ Volume analysis completed with zero volumes")
        print(f"   Current volume: {volume_analysis['current_volume']}")
        print(f"   Volume ratio: {volume_analysis['volume_ratio']:.2f}")
        print(f"   Volume confirmation: {volume_analysis['volume_confirmation_110pct']}")
        print(f"   Volume SMA 20: {volume_analysis['volume_sma_20']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_pattern_analysis_with_zero_volume():
    """Test candlestick pattern analysis with zero volume."""
    print("\n🔍 Testing Pattern Analysis with Zero Volume\n")
    
    try:
        # Create test data with zero volume
        test_data = []
        current_time = datetime.now(timezone.utc)
        
        # Add some regular candles
        for i in range(45):
            candle = OHLCV(
                timestamp=current_time,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0
            )
            test_data.append(candle)
        
        # Add a bearish marabozu with zero volume
        marabozu_candle = OHLCV(
            timestamp=current_time,
            open=100.0,   # High
            high=100.0,   # Same as open
            low=95.0,     # Much lower
            close=95.0,   # Same as low
            volume=0.0    # Zero volume
        )
        test_data.append(marabozu_candle)
        
        # Test candlestick formation creation
        tech_indicators = TechnicalIndicators()
        formation = tech_indicators.create_candlestick_formation(test_data)
        
        if formation:
            print("✅ Candlestick formation created with zero volume")
            print(f"   Pattern: {formation.pattern_name}")
            print(f"   Type: {formation.pattern_type}")
            print(f"   Strength: {formation.strength}/10")
            print(f"   Description: {formation.visual_description}")
            print(f"   Volume Confirmation: {formation.volume_confirmation}")
        else:
            print("⚠️ No formation detected (might be expected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern analysis failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing Volume Validation Fix for Enhanced Features\n")
    
    success1 = test_zero_volume_validation()
    success2 = test_pattern_analysis_with_zero_volume()
    
    if success1 and success2:
        print("\n🎉 Volume validation fix successful!")
        print("\n✅ Ready to test enhanced features:")
        print("   - Zero volume validation now works")
        print("   - Enhanced timestamp and candlestick features should display")
        print("   - Pattern analysis works with zero volume data")
        print("\n🔧 Next: Run 'make run' and test the enhanced features!")
    else:
        print("\n💥 Some tests failed - check the errors above")