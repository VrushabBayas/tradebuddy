#!/usr/bin/env python3
"""
Test script to compare functionality between main and simplification branches.
Tests core imports, utility functions, and strategy initialization.
"""

import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List

def test_imports() -> Dict[str, Any]:
    """Test core module imports."""
    results = {"imports": {}}
    
    test_modules = [
        "src.core.models",
        "src.core.config", 
        "src.core.constants",
        "src.analysis.strategies.ema_crossover",
        "src.analysis.strategies.support_resistance",
        "src.analysis.strategies.combined",
        "src.data.delta_client",
        "src.data.websocket_client",
        "src.utils.risk_management",
    ]
    
    # Test utility imports based on branch
    try:
        import src.utils.helpers
        test_modules.append("src.utils.helpers")
        results["branch_type"] = "simplification"
    except ImportError:
        test_modules.extend(["src.utils.type_conversion", "src.utils.data_helpers"])
        results["branch_type"] = "main"
    
    for module in test_modules:
        try:
            __import__(module)
            results["imports"][module] = "SUCCESS"
        except Exception as e:
            results["imports"][module] = f"FAILED: {str(e)}"
    
    return results

def test_utility_functions() -> Dict[str, Any]:
    """Test utility function functionality."""
    results = {"utility_functions": {}}
    
    try:
        # Try importing from helpers (simplification branch)
        from src.utils.helpers import to_float, to_decimal, get_value
        results["import_source"] = "helpers"
    except ImportError:
        # Fall back to original modules (main branch)
        from src.utils.type_conversion import to_float, to_decimal
        from src.utils.data_helpers import get_value
        results["import_source"] = "separate_modules"
    
    # Test conversion functions
    test_cases = [
        ("to_float", to_float, ["123.45", 123, None, "invalid"]),
        ("to_decimal", to_decimal, ["123.45", 123, None]),
        ("get_value", get_value, [
            ({"key": "value"}, "key"),
            ({"nested": {"inner": 42}}, "nested"),
            ({}, "missing"),
        ]),
    ]
    
    for func_name, func, test_inputs in test_cases:
        results["utility_functions"][func_name] = {}
        
        for i, test_input in enumerate(test_inputs):
            try:
                if func_name == "get_value":
                    result = func(*test_input)
                else:
                    result = func(test_input)
                results["utility_functions"][func_name][f"test_{i}"] = str(result)
            except Exception as e:
                results["utility_functions"][func_name][f"test_{i}"] = f"ERROR: {str(e)}"
    
    return results

def test_strategy_initialization() -> Dict[str, Any]:
    """Test strategy class initialization."""
    results = {"strategies": {}}
    
    strategy_classes = [
        ("EMA Crossover", "src.analysis.strategies.ema_crossover", "EMACrossoverStrategy"),
        ("Support Resistance", "src.analysis.strategies.support_resistance", "SupportResistanceStrategy"),
        ("Combined", "src.analysis.strategies.combined", "CombinedStrategy"),
    ]
    
    for strategy_name, module_name, class_name in strategy_classes:
        try:
            module = __import__(module_name, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            strategy_instance = strategy_class()
            
            results["strategies"][strategy_name] = {
                "initialization": "SUCCESS",
                "strategy_type": str(strategy_instance.strategy_type),
                "has_technical_indicators": hasattr(strategy_instance, "technical_indicators"),
            }
        except Exception as e:
            results["strategies"][strategy_name] = {
                "initialization": f"FAILED: {str(e)}",
                "error_traceback": traceback.format_exc()
            }
    
    return results

def test_models_and_enums() -> Dict[str, Any]:
    """Test model creation and enum handling."""
    results = {"models": {}}
    
    try:
        from src.core.models import Symbol, TimeFrame, StrategyType, SessionConfig
        
        # Test enum creation
        results["models"]["enums"] = {
            "Symbol.BTCUSDT": str(Symbol.BTCUSDT),
            "TimeFrame.ONE_MINUTE": str(TimeFrame.ONE_MINUTE),
            "StrategyType.EMA_CROSSOVER": str(StrategyType.EMA_CROSSOVER),
        }
        
        # Test model creation
        config = SessionConfig(
            strategy=StrategyType.EMA_CROSSOVER,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_MINUTE,
            limit=50,
            confidence_threshold=7
        )
        
        results["models"]["SessionConfig"] = {
            "creation": "SUCCESS",
            "strategy": str(config.strategy),
            "symbol": str(config.symbol),
            "timeframe": str(config.timeframe),
        }
        
    except Exception as e:
        results["models"]["error"] = f"FAILED: {str(e)}"
        results["models"]["traceback"] = traceback.format_exc()
    
    return results

def run_all_tests() -> Dict[str, Any]:
    """Run all tests and return comprehensive results."""
    test_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "test_results": {}
    }
    
    test_functions = [
        ("imports", test_imports),
        ("utility_functions", test_utility_functions),
        ("strategy_initialization", test_strategy_initialization),
        ("models_and_enums", test_models_and_enums),
    ]
    
    for test_name, test_func in test_functions:
        try:
            test_results["test_results"][test_name] = test_func()
        except Exception as e:
            test_results["test_results"][test_name] = {
                "error": f"Test failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    return test_results

if __name__ == "__main__":
    print("🧪 Running TradeBuddy branch comparison tests...")
    
    results = run_all_tests()
    
    # Determine branch type for filename
    branch_type = results["test_results"].get("imports", {}).get("branch_type", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{branch_type}_{timestamp}.json"
    
    # Save results to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Tests completed! Results saved to: {filename}")
    
    # Print summary
    print(f"\n📊 Test Summary ({branch_type} branch):")
    for test_name, test_result in results["test_results"].items():
        if isinstance(test_result, dict):
            if "error" in test_result:
                print(f"  ❌ {test_name}: FAILED")
            else:
                print(f"  ✅ {test_name}: PASSED")
        else:
            print(f"  ⚠️  {test_name}: UNKNOWN")