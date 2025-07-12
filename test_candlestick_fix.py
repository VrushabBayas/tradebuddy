#!/usr/bin/env python3
"""
Test candlestick pattern consistency fixes.
"""

from datetime import datetime, timezone
from src.core.models import OHLCV
from src.analysis.indicators import TechnicalIndicators
from src.cli.displays import CLIDisplays
from rich.console import Console

def test_candlestick_consistency():
    console = Console()
    displays = CLIDisplays(console)
    tech_indicators = TechnicalIndicators()
    
    console.print("🕯️ Testing Candlestick Pattern Consistency Fixes")
    console.print("=" * 60)
    
    # Test Case 1: Large Bearish Candle (should be classified as Strong Bearish, not Strong Bullish)
    console.print("\n📊 Test Case 1: Large Bearish Candle")
    bearish_candle = OHLCV(
        timestamp=datetime.now(timezone.utc),
        open=100.0,      # Higher open
        high=102.0,      # Small upper shadow
        low=90.0,        # Large range
        close=91.0,      # Much lower close (bearish)
        volume=1000.0
    )
    
    # Create test data with enough history
    test_data = []
    for i in range(15):
        test_data.append(OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=95.0 + i * 0.5,
            high=96.0 + i * 0.5,
            low=94.0 + i * 0.5,
            close=95.5 + i * 0.5,
            volume=1000.0
        ))
    
    # Add our test bearish candle at the end
    test_data.append(bearish_candle)
    
    console.print(f"   Candle: Open=${bearish_candle.open}, High=${bearish_candle.high}, Low=${bearish_candle.low}, Close=${bearish_candle.close}")
    console.print(f"   Expected: Bearish candle type, Strong Bearish pattern, Red body description")
    
    # Test advanced pattern detection
    pattern_analysis = tech_indicators.detect_advanced_candlestick_patterns(test_data)
    console.print(f"   Pattern Analysis:")
    console.print(f"     - Primary Pattern: {pattern_analysis['primary_pattern']}")
    console.print(f"     - Signal Direction: {pattern_analysis['signal_direction']}")
    console.print(f"     - Is Bullish: {pattern_analysis['is_bullish']}")
    console.print(f"     - Is Bearish: {pattern_analysis['is_bearish']}")
    
    # Test candlestick formation creation
    formation = tech_indicators.create_candlestick_formation(test_data)
    if formation:
        console.print(f"   Formation:")
        console.print(f"     - Pattern Name: {formation.pattern_name}")
        console.print(f"     - Pattern Type: {formation.pattern_type}")
        console.print(f"     - Signal Direction: {formation.signal_direction}")
        console.print(f"     - Visual Description: {formation.visual_description}")
        console.print(f"     - Strength: {formation.strength}/10")
        
        # Check for consistency
        is_candle_bearish = bearish_candle.close < bearish_candle.open
        pattern_is_bearish = formation.signal_direction in ["strong_bearish", "bearish"]
        
        if is_candle_bearish and pattern_is_bearish:
            console.print("     ✅ CONSISTENT: Bearish candle correctly classified as bearish pattern", style="green")
        elif is_candle_bearish and not pattern_is_bearish:
            console.print("     ❌ INCONSISTENT: Bearish candle incorrectly classified as non-bearish pattern", style="red")
        
        # Check description matches color
        if is_candle_bearish and "red" in formation.visual_description.lower():
            console.print("     ✅ DESCRIPTION CORRECT: Red body mentioned for bearish candle", style="green")
        elif is_candle_bearish and "green" in formation.visual_description.lower():
            console.print("     ❌ DESCRIPTION WRONG: Green body mentioned for bearish candle", style="red")
    
    # Test Case 2: Large Bullish Candle (should be classified as Strong Bullish)
    console.print("\n📊 Test Case 2: Large Bullish Candle")
    bullish_candle = OHLCV(
        timestamp=datetime.now(timezone.utc),
        open=90.0,       # Lower open
        high=102.0,      # Good upper reach
        low=89.0,        # Small lower shadow
        close=101.0,     # Much higher close (bullish)
        volume=1000.0
    )
    
    # Replace last candle with bullish one
    test_data[-1] = bullish_candle
    
    console.print(f"   Candle: Open=${bullish_candle.open}, High=${bullish_candle.high}, Low=${bullish_candle.low}, Close=${bullish_candle.close}")
    console.print(f"   Expected: Bullish candle type, Strong Bullish pattern, Green body description")
    
    pattern_analysis = tech_indicators.detect_advanced_candlestick_patterns(test_data)
    formation = tech_indicators.create_candlestick_formation(test_data)
    
    if formation:
        console.print(f"   Formation:")
        console.print(f"     - Pattern Name: {formation.pattern_name}")
        console.print(f"     - Signal Direction: {formation.signal_direction}")
        console.print(f"     - Visual Description: {formation.visual_description}")
        
        # Check for consistency
        is_candle_bullish = bullish_candle.close > bullish_candle.open
        pattern_is_bullish = formation.signal_direction in ["strong_bullish", "bullish"]
        
        if is_candle_bullish and pattern_is_bullish:
            console.print("     ✅ CONSISTENT: Bullish candle correctly classified as bullish pattern", style="green")
        elif is_candle_bullish and not pattern_is_bullish:
            console.print("     ❌ INCONSISTENT: Bullish candle incorrectly classified as non-bullish pattern", style="red")
        
        # Check description matches color
        if is_candle_bullish and "green" in formation.visual_description.lower():
            console.print("     ✅ DESCRIPTION CORRECT: Green body mentioned for bullish candle", style="green")
        elif is_candle_bullish and "red" in formation.visual_description.lower():
            console.print("     ❌ DESCRIPTION WRONG: Red body mentioned for bullish candle", style="red")
    
    # Test Case 3: Edge case - Doji (open == close)
    console.print("\n📊 Test Case 3: Doji Pattern (open == close)")
    doji_candle = OHLCV(
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,     # Same as open
        volume=1000.0
    )
    
    test_data[-1] = doji_candle
    formation = tech_indicators.create_candlestick_formation(test_data)
    
    if formation:
        console.print(f"   Formation:")
        console.print(f"     - Pattern Name: {formation.pattern_name}")
        console.print(f"     - Signal Direction: {formation.signal_direction}")
        console.print(f"     - Visual Description: {formation.visual_description}")
        
        if formation.signal_direction == "neutral":
            console.print("     ✅ CORRECT: Doji correctly classified as neutral", style="green")
        else:
            console.print("     ❌ INCORRECT: Doji should be neutral", style="red")
    
    console.print("\n✅ Candlestick consistency test completed!")

if __name__ == "__main__":
    test_candlestick_consistency()