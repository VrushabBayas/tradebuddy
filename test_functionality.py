#!/usr/bin/env python3
"""
Comprehensive functionality test script for TradeBuddy.

Tests core functionality to ensure both branches produce identical results.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import traceback

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def log_test(message: str, status: str = "INFO"):
    """Log test message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {status}: {message}")

async def test_core_imports():
    """Test that all core modules can be imported."""
    log_test("Testing core module imports...")
    results = {}
    
    try:
        # Test core imports
        from src.core.models import Symbol, TimeFrame, MarketData, TradingSignal, AnalysisResult
        from src.core.config import settings, get_settings
        from src.core.constants import TradingConstants, APIEndpoints
        results["core_imports"] = "SUCCESS"
        log_test("Core imports successful", "SUCCESS")
    except Exception as e:
        results["core_imports"] = f"FAILED: {str(e)}"
        log_test(f"Core imports failed: {e}", "ERROR")
    
    return results

async def test_utility_functions():
    """Test utility functions to ensure they work correctly."""
    log_test("Testing utility functions...")
    results = {}
    
    try:
        # Try importing from the appropriate helper module based on what's available
        try:
            # New consolidated helpers (simplified branch)
            from src.utils.helpers import to_float, to_decimal, get_value, normalize_price
            log_test("Using consolidated helpers module", "INFO")
        except ImportError:
            # Old separate modules (main branch)
            from src.utils.type_conversion import to_float, to_decimal, normalize_price
            from src.utils.data_helpers import get_value
            log_test("Using separate utility modules", "INFO")
        
        # Test type conversions
        test_cases = [
            ("123.45", 123.45),
            (456, 456.0),
            ("invalid", 0.0),  # Should return default
            (None, 0.0)
        ]
        
        conversion_results = []
        for input_val, expected in test_cases:
            result = to_float(input_val)
            conversion_results.append({
                "input": input_val,
                "expected": expected,
                "result": result,
                "correct": abs(result - expected) < 0.001
            })
        
        # Test decimal conversions
        from decimal import Decimal
        decimal_result = to_decimal("123.456", precision=2)
        decimal_correct = decimal_result == Decimal("123.46")
        
        # Test data access
        test_obj = {"price": 100.5, "volume": 1000}
        price_result = get_value(test_obj, "price", 0)
        volume_result = get_value(test_obj, "volume", 0)
        missing_result = get_value(test_obj, "missing", "default")
        
        # Test price normalization
        normalized = normalize_price("99.999", precision=2)
        
        results["utility_functions"] = {
            "float_conversions": conversion_results,
            "decimal_conversion": {"result": float(decimal_result), "correct": decimal_correct},
            "data_access": {
                "price": price_result,
                "volume": volume_result,
                "missing": missing_result
            },
            "price_normalization": float(normalized)
        }
        
        log_test("Utility functions test completed", "SUCCESS")
        
    except Exception as e:
        results["utility_functions"] = f"FAILED: {str(e)}"
        log_test(f"Utility functions test failed: {e}", "ERROR")
        traceback.print_exc()
    
    return results

async def test_strategy_initialization():
    """Test that all trading strategies can be initialized."""
    log_test("Testing strategy initialization...")
    results = {}
    
    try:
        from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
        from src.analysis.strategies.support_resistance import SupportResistanceStrategy
        from src.analysis.strategies.combined import CombinedStrategy
        
        # Initialize strategies
        ema_strategy = EMACrossoverStrategy()
        sr_strategy = SupportResistanceStrategy()
        combined_strategy = CombinedStrategy()
        
        results["strategy_initialization"] = {
            "ema_crossover": "SUCCESS",
            "support_resistance": "SUCCESS", 
            "combined": "SUCCESS"
        }
        
        log_test("All strategies initialized successfully", "SUCCESS")
        
    except Exception as e:
        results["strategy_initialization"] = f"FAILED: {str(e)}"
        log_test(f"Strategy initialization failed: {e}", "ERROR")
        traceback.print_exc()
    
    return results

async def test_data_models():
    """Test core data models functionality."""
    log_test("Testing data models...")
    results = {}
    
    try:
        from src.core.models import Symbol, TimeFrame, OHLCV, MarketData, TradingSignal, AnalysisResult
        from decimal import Decimal
        from datetime import datetime
        
        # Test OHLCV creation
        ohlcv = OHLCV(
            open=Decimal("100.0"),
            high=Decimal("105.0"),
            low=Decimal("99.0"),
            close=Decimal("102.0"),
            volume=Decimal("1000.0")
        )
        
        # Test MarketData creation
        market_data = MarketData(
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_MINUTE,
            ohlcv_data=[ohlcv],
            current_price=102.0
        )
        
        # Test TradingSignal creation
        from src.core.models import SignalAction
        signal = TradingSignal(
            action=SignalAction.BUY,
            confidence=8,
            price=102.0,
            symbol=Symbol.BTCUSDT,
            reasoning="Test signal"
        )
        
        # Test model serialization
        market_data_dict = market_data.model_dump()
        signal_dict = signal.model_dump()
        
        results["data_models"] = {
            "ohlcv_creation": "SUCCESS",
            "market_data_creation": "SUCCESS",
            "signal_creation": "SUCCESS",
            "serialization": "SUCCESS",
            "market_data_price": market_data.current_price,
            "signal_confidence": signal.confidence,
            "signal_action": str(signal.action)
        }
        
        log_test("Data models test completed", "SUCCESS")
        
    except Exception as e:
        results["data_models"] = f"FAILED: {str(e)}"
        log_test(f"Data models test failed: {e}", "ERROR")
        traceback.print_exc()
    
    return results

async def test_technical_indicators():
    """Test technical indicators calculations."""
    log_test("Testing technical indicators...")
    results = {}
    
    try:
        from src.analysis.indicators import TechnicalIndicators
        from src.core.models import OHLCV
        from decimal import Decimal
        
        # Create sample OHLCV data
        sample_data = []
        prices = [100, 101, 102, 101, 103, 104, 103, 105, 106, 105]
        
        for i, price in enumerate(prices):
            ohlcv = OHLCV(
                open=Decimal(str(price - 0.5)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=Decimal("1000")
            )
            sample_data.append(ohlcv)
        
        indicators = TechnicalIndicators()
        
        # Test EMA calculation
        ema_9 = indicators.calculate_ema(sample_data, 9)
        ema_15 = indicators.calculate_ema(sample_data, 15)
        
        # Test RSI calculation
        rsi_values = indicators.calculate_rsi(sample_data, 14)
        
        # Test volume analysis
        volume_analysis = indicators.analyze_volume(sample_data)
        
        results["technical_indicators"] = {
            "ema_9_count": len(ema_9),
            "ema_15_count": len(ema_15),
            "ema_9_last": float(ema_9[-1]) if ema_9 else None,
            "rsi_count": len(rsi_values),
            "rsi_last": rsi_values[-1] if rsi_values else None,
            "volume_analysis": volume_analysis
        }
        
        log_test("Technical indicators test completed", "SUCCESS")
        
    except Exception as e:
        results["technical_indicators"] = f"FAILED: {str(e)}"
        log_test(f"Technical indicators test failed: {e}", "ERROR")
        traceback.print_exc()
    
    return results

async def test_risk_management():
    """Test risk management calculations."""
    log_test("Testing risk management...")
    results = {}
    
    try:
        from src.utils.risk_management import (
            calculate_leveraged_position_size,
            calculate_stop_loss_take_profit,
            calculate_position_risk
        )
        from src.core.models import SignalAction
        
        # Test position size calculation
        position_value, position_amount, margin = calculate_leveraged_position_size(
            account_balance=10000.0,
            position_size_pct=5.0,
            leverage=10,
            current_price=50000.0,
            min_lot_size=0.001
        )
        
        # Test stop loss/take profit calculation
        stop_loss, take_profit, risk_reward = calculate_stop_loss_take_profit(
            entry_price=50000.0,
            signal_action=SignalAction.BUY,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            leverage=10
        )
        
        # Test position risk calculation
        max_loss, max_loss_pct, effective_leverage = calculate_position_risk(
            position_value_usd=position_value,
            account_balance=10000.0,
            leverage=10,
            stop_loss_pct=2.0
        )
        
        results["risk_management"] = {
            "position_value": position_value,
            "position_amount": position_amount,
            "margin_required": margin,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward,
            "max_loss_usd": max_loss,
            "max_loss_pct": max_loss_pct,
            "effective_leverage": effective_leverage
        }
        
        log_test("Risk management test completed", "SUCCESS")
        
    except Exception as e:
        results["risk_management"] = f"FAILED: {str(e)}"
        log_test(f"Risk management test failed: {e}", "ERROR")
        traceback.print_exc()
    
    return results

async def run_all_tests():
    """Run all functionality tests."""
    log_test("Starting comprehensive functionality tests...")
    
    all_results = {}
    
    # Run all test suites
    test_suites = [
        ("core_imports", test_core_imports),
        ("utility_functions", test_utility_functions),
        ("strategy_initialization", test_strategy_initialization),
        ("data_models", test_data_models),
        ("technical_indicators", test_technical_indicators),
        ("risk_management", test_risk_management)
    ]
    
    for suite_name, test_func in test_suites:
        log_test(f"Running {suite_name} tests...")
        try:
            results = await test_func()
            all_results[suite_name] = results
        except Exception as e:
            log_test(f"Test suite {suite_name} failed: {e}", "ERROR")
            all_results[suite_name] = f"SUITE_FAILED: {str(e)}"
    
    return all_results

def main():
    """Main test function."""
    # Get git branch info
    import subprocess
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], 
                                       text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                       text=True).strip()[:8]
    except:
        branch = "unknown"
        commit = "unknown"
    
    log_test(f"Testing branch: {branch} (commit: {commit})")
    
    # Run tests
    results = asyncio.run(run_all_tests())
    
    # Add metadata
    results["_metadata"] = {
        "branch": branch,
        "commit": commit,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version
    }
    
    # Save results to file
    output_file = f"test_results_{branch}_{commit}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log_test(f"Test results saved to {output_file}")
    
    # Print summary
    log_test("Test Summary:")
    for suite_name, suite_results in results.items():
        if suite_name.startswith("_"):
            continue
        if isinstance(suite_results, dict) and "FAILED" not in str(suite_results):
            log_test(f"  {suite_name}: PASSED", "SUCCESS")
        else:
            log_test(f"  {suite_name}: FAILED", "ERROR")
    
    return results

if __name__ == "__main__":
    main()