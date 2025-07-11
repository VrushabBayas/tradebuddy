#!/usr/bin/env python3
"""
Test script to verify the new risk management settings for 10x leverage.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.core.models import SessionConfig, StrategyType, Symbol, TimeFrame
from src.utils.risk_management import (
    calculate_leveraged_position_size,
    calculate_stop_loss_take_profit,
    optimize_position_for_delta_exchange,
    validate_position_safety
)
from src.core.models import SignalAction

async def test_risk_management():
    """Test the new risk management settings."""
    
    try:
        print("🔧 Testing Risk Management for 10x Leverage Trading")
        print("="*60)
        
        # Create updated session config
        session_config = SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.FIFTEEN_MINUTES,  # 15m timeframe
            stop_loss_pct=1.5,  # Tighter stop loss
            take_profit_pct=3.0,  # Realistic take profit
            position_size_pct=5.0,  # Higher position size
            leverage=10,  # 10x leverage
            min_lot_size=0.001,  # Delta Exchange lot size
            confidence_threshold=6
        )
        
        # Test parameters
        account_balance = 10000.0
        btc_price = 118000.0
        
        print(f"📊 Session Configuration:")
        print(f"   - Strategy: {session_config.strategy}")
        print(f"   - Symbol: {session_config.symbol}")
        print(f"   - Timeframe: {session_config.timeframe}")
        print(f"   - Position Size: {session_config.position_size_pct}%")
        print(f"   - Stop Loss: {session_config.stop_loss_pct}%")
        print(f"   - Take Profit: {session_config.take_profit_pct}%")
        print(f"   - Leverage: {session_config.leverage}x")
        print(f"   - Min Lot Size: {session_config.min_lot_size} BTC")
        print(f"   - Account Balance: ${account_balance:,.2f}")
        print(f"   - BTC Price: ${btc_price:,.2f}")
        print()
        
        # Test position sizing
        print("💰 Position Sizing Analysis:")
        position_value, position_btc, margin_required = calculate_leveraged_position_size(
            account_balance=account_balance,
            position_size_pct=float(session_config.position_size_pct),
            leverage=session_config.leverage,
            current_price=btc_price,
            min_lot_size=float(session_config.min_lot_size)
        )
        
        lots = position_btc / float(session_config.min_lot_size)
        print(f"   - Margin Required: ${margin_required:,.2f}")
        print(f"   - Position Value: ${position_value:,.2f}")
        print(f"   - BTC Amount: {position_btc:.6f} BTC")
        print(f"   - Lots: {lots:.3f} lots")
        print()
        
        # Test stop loss and take profit for BUY signal
        print("📈 BUY Signal Risk Management:")
        stop_loss_buy, take_profit_buy, risk_reward_buy = calculate_stop_loss_take_profit(
            entry_price=btc_price,
            signal_action=SignalAction.BUY,
            stop_loss_pct=float(session_config.stop_loss_pct),
            take_profit_pct=float(session_config.take_profit_pct),
            leverage=session_config.leverage
        )
        
        print(f"   - Entry Price: ${btc_price:,.2f}")
        print(f"   - Stop Loss: ${stop_loss_buy:,.2f}")
        print(f"   - Take Profit: ${take_profit_buy:,.2f}")
        print(f"   - Risk/Reward: {risk_reward_buy:.2f}")
        print()
        
        # Test stop loss and take profit for SELL signal
        print("📉 SELL Signal Risk Management:")
        stop_loss_sell, take_profit_sell, risk_reward_sell = calculate_stop_loss_take_profit(
            entry_price=btc_price,
            signal_action=SignalAction.SELL,
            stop_loss_pct=float(session_config.stop_loss_pct),
            take_profit_pct=float(session_config.take_profit_pct),
            leverage=session_config.leverage
        )
        
        print(f"   - Entry Price: ${btc_price:,.2f}")
        print(f"   - Stop Loss: ${stop_loss_sell:,.2f}")
        print(f"   - Take Profit: ${take_profit_sell:,.2f}")
        print(f"   - Risk/Reward: {risk_reward_sell:.2f}")
        print()
        
        # Test full optimization
        print("🔧 Delta Exchange Optimization:")
        position_params = optimize_position_for_delta_exchange(
            session_config=session_config,
            current_price=btc_price,
            account_balance=account_balance
        )
        
        print(f"   - Position Value: ${position_params['position_value_usd']:,.2f}")
        print(f"   - Position Amount: {position_params['position_amount_btc']:.6f} BTC")
        print(f"   - Margin Required: ${position_params['margin_required']:,.2f}")
        print(f"   - Lots: {position_params['lots']:.3f}")
        print(f"   - Max Loss: ${position_params['max_loss_usd']:,.2f}")
        print(f"   - Max Loss %: {position_params['max_loss_pct']:.1f}%")
        print(f"   - Effective Leverage: {position_params['effective_leverage']:.1f}x")
        print(f"   - Risk/Reward: {position_params['risk_reward_ratio']:.2f}")
        print()
        
        # Test safety validation
        print("🛡️ Position Safety Validation:")
        is_safe, warnings = validate_position_safety(
            session_config=session_config,
            position_params=position_params,
            account_balance=account_balance
        )
        
        print(f"   - Position Safe: {'✅ Yes' if is_safe else '❌ No'}")
        if warnings:
            print(f"   - Warnings:")
            for warning in warnings:
                print(f"     • {warning}")
        else:
            print(f"   - No warnings")
        print()
        
        # Compare with old vs new settings
        print("📊 Old vs New Settings Comparison:")
        print(f"{'Parameter':<20} {'Old':<15} {'New':<15} {'Delta Exchange'}")
        print("-" * 70)
        print(f"{'Position Size':<20} {'2.0%':<15} {f'{session_config.position_size_pct}%':<15} {'Higher for leverage'}")
        print(f"{'Stop Loss':<20} {'2.5%':<15} {f'{session_config.stop_loss_pct}%':<15} {'Tighter for leverage'}")
        print(f"{'Take Profit':<20} {'5.0%':<15} {f'{session_config.take_profit_pct}%':<15} {'Realistic for crypto'}")
        print(f"{'Leverage':<20} {'Not specified':<15} {f'{session_config.leverage}x':<15} {'10x is typical'}")
        print(f"{'Min Lot Size':<20} {'Not specified':<15} {f'{session_config.min_lot_size}':<15} {'0.001 BTC standard'}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_risk_management())
    sys.exit(0 if success else 1)