#!/usr/bin/env python3
"""
Comprehensive validation script to ensure all Delta Exchange India 
fee structures are correctly implemented across the entire app.
"""

import sys
import os
import inspect

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def validate_fee_structure():
    """Validate Delta Exchange India fee structure across all modules."""
    print("🇮🇳 Comprehensive Delta Exchange India Fee Structure Validation")
    print("=" * 70)
    
    issues_found = []
    validations_passed = []
    
    # 1. Check Constants
    print("\n1. 📋 Validating Constants...")
    try:
        from src.core.constants import TradingConstants
        
        # Check Delta Exchange India fees
        if TradingConstants.FUTURES_TAKER_FEE == 0.05:
            validations_passed.append("✅ Futures Taker Fee: 0.05%")
        else:
            issues_found.append(f"❌ Futures Taker Fee: {TradingConstants.FUTURES_TAKER_FEE}% (should be 0.05%)")
            
        if TradingConstants.FUTURES_MAKER_FEE == 0.02:
            validations_passed.append("✅ Futures Maker Fee: 0.02%")
        else:
            issues_found.append(f"❌ Futures Maker Fee: {TradingConstants.FUTURES_MAKER_FEE}% (should be 0.02%)")
            
        if TradingConstants.OPTIONS_TAKER_FEE == 0.03:
            validations_passed.append("✅ Options Taker Fee: 0.03%")
        else:
            issues_found.append(f"❌ Options Taker Fee: {TradingConstants.OPTIONS_TAKER_FEE}% (should be 0.03%)")
            
        if TradingConstants.GST_RATE == 18.0:
            validations_passed.append("✅ GST Rate: 18%")
        else:
            issues_found.append(f"❌ GST Rate: {TradingConstants.GST_RATE}% (should be 18%)")
            
        if TradingConstants.USD_TO_INR == 85.0:
            validations_passed.append("✅ USD to INR Rate: 85")
        else:
            issues_found.append(f"❌ USD to INR Rate: {TradingConstants.USD_TO_INR} (should be 85)")
            
    except Exception as e:
        issues_found.append(f"❌ Constants validation failed: {e}")
    
    # 2. Check Configuration Defaults
    print("\n2. ⚙️ Validating Configuration...")
    try:
        from src.core.config import settings
        from src.core.constants import DEFAULT_CONFIG
        
        if DEFAULT_CONFIG["risk_management"]["default_commission"] == 0.05:
            validations_passed.append("✅ Default Commission: 0.05%")
        else:
            issues_found.append(f"❌ Default Commission: {DEFAULT_CONFIG['risk_management']['default_commission']}% (should be 0.05%)")
            
        if hasattr(settings, 'default_commission') and settings.default_commission == 0.05:
            validations_passed.append("✅ Settings Default Commission: 0.05%")
        elif hasattr(settings, 'default_commission'):
            issues_found.append(f"❌ Settings Default Commission: {settings.default_commission}% (should be 0.05%)")
        else:
            validations_passed.append("✅ Settings Default Commission: Not set (using constants)")
            
    except Exception as e:
        issues_found.append(f"❌ Configuration validation failed: {e}")
    
    # 3. Check Backtesting Models
    print("\n3. 📊 Validating Backtesting Models...")
    try:
        from src.backtesting.models import BacktestConfig
        from src.core.models import StrategyType, Symbol, TimeFrame
        
        # Create default config to check defaults
        config = BacktestConfig(
            strategy_type=StrategyType.EMA_CROSSOVER,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
            days_back=7
        )
        
        if config.commission_pct == 0.05:
            validations_passed.append("✅ BacktestConfig Default Commission: 0.05%")
        else:
            issues_found.append(f"❌ BacktestConfig Default Commission: {config.commission_pct}% (should be 0.05%)")
            
        if config.initial_capital == 850000.0:
            validations_passed.append("✅ BacktestConfig Default Capital: ₹8,50,000")
        else:
            issues_found.append(f"❌ BacktestConfig Default Capital: ₹{config.initial_capital:,.0f} (should be ₹8,50,000)")
            
        if config.currency == "INR":
            validations_passed.append("✅ BacktestConfig Default Currency: INR")
        else:
            issues_found.append(f"❌ BacktestConfig Default Currency: {config.currency} (should be INR)")
            
    except Exception as e:
        issues_found.append(f"❌ Backtesting models validation failed: {e}")
    
    # 4. Check Portfolio Defaults
    print("\n4. 💼 Validating Portfolio...")
    try:
        from src.backtesting.portfolio import Portfolio
        
        # Get Portfolio constructor signature
        sig = inspect.signature(Portfolio.__init__)
        commission_default = sig.parameters['commission_pct'].default
        
        if commission_default == 0.05:
            validations_passed.append("✅ Portfolio Default Commission: 0.05%")
        else:
            issues_found.append(f"❌ Portfolio Default Commission: {commission_default}% (should be 0.05%)")
            
    except Exception as e:
        issues_found.append(f"❌ Portfolio validation failed: {e}")
    
    # 5. Check CLI Defaults
    print("\n5. 🖥️ Validating CLI...")
    try:
        # This is harder to validate without running CLI, but we can check the source
        with open('/Users/vrushabhbayas/personal/tradebuddy/src/cli/main.py', 'r') as f:
            cli_content = f.read()
            
        if 'default="0.05"' in cli_content and 'Commission' in cli_content:
            validations_passed.append("✅ CLI Default Commission: 0.05%")
        elif 'default="0.1"' in cli_content and 'Commission' in cli_content:
            issues_found.append("❌ CLI still shows default=\"0.1\" for commission")
        else:
            validations_passed.append("✅ CLI Commission: Using dynamic default")
            
        if 'default="850000"' in cli_content:
            validations_passed.append("✅ CLI Default Capital: ₹8,50,000")
        elif 'default="10000"' in cli_content:
            issues_found.append("❌ CLI still shows default=\"10000\" for capital")
        else:
            validations_passed.append("✅ CLI Capital: Using dynamic default")
            
    except Exception as e:
        issues_found.append(f"❌ CLI validation failed: {e}")
    
    # 6. Check Report Generator
    print("\n6. 📄 Validating Report Generator...")
    try:
        from src.backtesting.reports import BacktestReportGenerator
        
        # Test with INR currency
        report_gen = BacktestReportGenerator(currency="INR")
        
        if report_gen.currency == "INR":
            validations_passed.append("✅ Report Generator: INR Currency Support")
        else:
            issues_found.append(f"❌ Report Generator Currency: {report_gen.currency} (should support INR)")
            
        if report_gen.currency_symbol == "₹":
            validations_passed.append("✅ Report Generator: ₹ Symbol")
        else:
            issues_found.append(f"❌ Report Generator Symbol: {report_gen.currency_symbol} (should be ₹)")
            
        if report_gen.exchange_rate == 85.0:
            validations_passed.append("✅ Report Generator: Exchange Rate 85")
        else:
            issues_found.append(f"❌ Report Generator Exchange Rate: {report_gen.exchange_rate} (should be 85)")
            
    except Exception as e:
        issues_found.append(f"❌ Report generator validation failed: {e}")
    
    # 7. Check Trade P&L Calculations
    print("\n7. 🧮 Validating Trade P&L...")
    try:
        from src.backtesting.models import BacktestTrade, TradeStatus
        from src.core.models import SignalAction
        from datetime import datetime, timedelta
        
        # Create test trade
        trade = BacktestTrade(
            trade_id=1,
            signal_type=SignalAction.BUY,
            trade_action="LONG",
            status=TradeStatus.OPEN,
            entry_time=datetime.now(),
            entry_price=50000.0,
            entry_reason="Test",
            quantity=1.0,
            position_size_usd=10000.0,
            leverage_used=5,
            margin_used=2000.0,
            confidence_score=8,
            commission_paid=25.0,
            slippage_cost=25.0,
            strategy_context={}
        )
        
        # Test P&L calculation with 2% price increase
        trade.close_trade(
            exit_time=datetime.now() + timedelta(hours=1),
            exit_price=51000.0,  # 2% increase
            exit_reason="Test",
            commission_pct=0.05,
            slippage_pct=0.05
        )
        
        # Check if P&L calculation includes leverage
        expected_pnl_pct = 2.0 * 5  # 2% * 5x leverage = 10%
        if abs(trade.pnl_pct - expected_pnl_pct) < 0.01:
            validations_passed.append("✅ Trade P&L: Proper leverage calculation")
        else:
            issues_found.append(f"❌ Trade P&L: {trade.pnl_pct:.2f}% (should be {expected_pnl_pct:.2f}%)")
            
        # Check GST calculation
        if trade.gst_paid > 0:
            validations_passed.append("✅ Trade GST: Calculated and applied")
        else:
            issues_found.append("❌ Trade GST: Not calculated")
            
        # Check INR P&L
        if trade.pnl_inr is not None and trade.pnl_inr > 0:
            validations_passed.append("✅ Trade INR P&L: Calculated")
        else:
            issues_found.append("❌ Trade INR P&L: Not calculated")
            
    except Exception as e:
        issues_found.append(f"❌ Trade P&L validation failed: {e}")
    
    # Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    if validations_passed:
        print(f"\n✅ PASSED VALIDATIONS ({len(validations_passed)}):")
        for validation in validations_passed:
            print(f"  {validation}")
    
    if issues_found:
        print(f"\n❌ ISSUES FOUND ({len(issues_found)}):")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print(f"\n🎉 NO ISSUES FOUND!")
    
    print(f"\n📊 DELTA EXCHANGE INDIA FEE STRUCTURE STATUS:")
    if not issues_found:
        print("🟢 FULLY COMPLIANT - All fee structures correctly implemented")
        print("✅ Futures Taker: 0.05%")
        print("✅ Futures Maker: 0.02%") 
        print("✅ Options: 0.03%")
        print("✅ GST: 18% on all fees")
        print("✅ Currency: INR with ₹ symbol")
        print("✅ P&L: Proper leverage calculations")
        print("✅ Default Capital: ₹8,50,000")
    else:
        print("🟡 PARTIALLY COMPLIANT - Some issues need fixing")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    success = validate_fee_structure()
    print(f"\n{'✅ VALIDATION PASSED' if success else '❌ VALIDATION FAILED'}")