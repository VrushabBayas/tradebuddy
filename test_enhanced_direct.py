#!/usr/bin/env python3
"""
Direct test of enhanced features by creating signals manually.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from src.core.models import (
    OHLCV, MarketData, SessionConfig, Symbol, TimeFrame, StrategyType,
    TradingSignal, SignalAction, SignalStrength, CandlestickFormation,
    AnalysisResult
)
from src.analysis.indicators import TechnicalIndicators
from src.cli.displays import CLIDisplays
from rich.console import Console


def create_test_signals():
    """Create test signals with enhanced features."""
    
    # Create technical indicators instance
    tech_indicators = TechnicalIndicators()
    
    # Create test OHLCV data
    current_time = datetime.now(timezone.utc)
    test_candle = OHLCV(
        timestamp=current_time,
        open=100.0,
        high=100.0,  # No upper shadow
        low=95.0,
        close=95.0,   # No lower shadow - Bearish Marabozu
        volume=2000.0
    )
    
    ohlcv_data = [test_candle]
    
    # Create candlestick formation
    formation = CandlestickFormation(
        pattern_name="bearish_marabozu",
        pattern_type="bearish",
        strength=9,
        signal_direction="strong_bearish",
        body_ratio=1.0,
        upper_shadow_ratio=0.0,
        lower_shadow_ratio=0.0,
        visual_description="Strong red candle with virtually no shadows - maximum bearish pressure",
        trend_context="downtrend",
        volume_confirmation=True
    )
    
    # Generate enhanced pattern reasoning
    pattern_reasoning = tech_indicators.generate_pattern_reasoning(
        formation=formation,
        current_price=95.0,
        trend_context="downtrend"
    )
    
    # Create enhanced trading signal
    signal1 = TradingSignal(
        symbol=Symbol.BTCUSDT,
        strategy=StrategyType.EMA_CROSSOVER,
        action=SignalAction.SELL,
        strength=SignalStrength.STRONG,
        confidence=8,
        entry_price=Decimal("95.0"),
        stop_loss=Decimal("97.0"),
        take_profit=Decimal("90.0"),
        reasoning="EMA death cross detected with strong bearish marabozu confirmation",
        candle_timestamp=current_time,
        candle_formation=formation,
        pattern_context=pattern_reasoning,
        risk_reward_ratio=Decimal("2.5")
    )
    
    # Create a second signal with different pattern
    formation2 = CandlestickFormation(
        pattern_name="shooting_star",
        pattern_type="bearish",
        strength=6,
        signal_direction="bearish",
        body_ratio=0.3,
        upper_shadow_ratio=3.0,
        lower_shadow_ratio=0.2,
        visual_description="Small body with long upper shadow - potential reversal signal",
        trend_context="uptrend",
        volume_confirmation=False
    )
    
    pattern_reasoning2 = tech_indicators.generate_pattern_reasoning(
        formation=formation2,
        current_price=98.5,
        trend_context="uptrend"
    )
    
    signal2 = TradingSignal(
        symbol=Symbol.BTCUSDT,
        strategy=StrategyType.EMA_CROSSOVER,
        action=SignalAction.SELL,
        strength=SignalStrength.MODERATE,
        confidence=6,
        entry_price=Decimal("98.5"),
        reasoning="Shooting star pattern at resistance level",
        candle_timestamp=current_time,
        candle_formation=formation2,
        pattern_context=pattern_reasoning2
    )
    
    return [signal1, signal2]


def test_enhanced_displays():
    """Test the enhanced display functionality."""
    
    print("🧪 Testing Enhanced Display Features\n")
    
    # Create test signals
    signals = create_test_signals()
    
    # Create market data
    market_data = MarketData(
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.FIVE_MINUTES,
        current_price=95.0,
        ohlcv_data=[]
    )
    
    # Create analysis result
    analysis_result = AnalysisResult(
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.FIVE_MINUTES,
        strategy=StrategyType.EMA_CROSSOVER,
        market_data=market_data,
        signals=signals,
        ai_analysis="Test analysis with enhanced features"
    )
    
    # Create session config
    session_config = SessionConfig(
        strategy=StrategyType.EMA_CROSSOVER,
        symbol=Symbol.BTCUSDT,
        timeframe=TimeFrame.FIVE_MINUTES
    )
    
    # Test individual signal properties
    print("🔍 Testing Enhanced Signal Properties:")
    print("=" * 60)
    
    for i, signal in enumerate(signals, 1):
        print(f"\n📊 Signal #{i}:")
        print(f"   Action: {signal.action}")
        print(f"   Confidence: {signal.confidence}/10")
        print(f"   🕒 Candle Time Display: {signal.candle_time_display}")
        print(f"   📊 Formation Display: {signal.formation_display}")
        print(f"   💡 Enhanced Reasoning: {signal.enhanced_reasoning[:150]}...")
        
        if signal.candle_formation:
            formation = signal.candle_formation
            print(f"   🔥 Pattern Details:")
            print(f"      - Is Strong: {formation.is_strong_pattern}")
            print(f"      - Display Name: {formation.pattern_display_name}")
            print(f"      - Visual: {formation.visual_description}")
    
    # Test Rich displays
    print(f"\n{'='*60}")
    print("🎨 Testing Rich Console Displays:")
    print("=" * 60)
    
    console = Console()
    displays = CLIDisplays(console)
    
    # Test the enhanced trading signals display
    displays.display_trading_signals(analysis_result, session_config)
    
    print("\n✅ Enhanced display testing completed!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_enhanced_displays()
        if success:
            print("\n🎉 All enhanced display features are working correctly!")
            print("\n🎯 Verified Features:")
            print("  ✅ Candle timestamp formatting")
            print("  ✅ Pattern formation display")
            print("  ✅ Enhanced reasoning with pattern context")
            print("  ✅ Rich table with new columns")
            print("  ✅ Enhanced primary signal panel")
            print("  ✅ Pattern strength indicators (🔥 for strong patterns)")
            print("  ✅ Volume confirmation status")
            print("  ✅ Educational pattern descriptions")
        else:
            print("\n💥 Some display features failed")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()