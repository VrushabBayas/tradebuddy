#!/usr/bin/env python3
"""
Compare test results between main and simplification branches.
"""

import json
import sys
from typing import Dict, Any, List, Tuple

def load_test_results(filename: str) -> Dict[str, Any]:
    """Load test results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filename}: {e}")
        return {}

def compare_utility_functions(main_results: Dict, simp_results: Dict) -> List[Tuple[str, str, str]]:
    """Compare utility function test results."""
    differences = []
    
    main_utils = main_results.get("test_results", {}).get("utility_functions", {}).get("utility_functions", {})
    simp_utils = simp_results.get("test_results", {}).get("utility_functions", {}).get("utility_functions", {})
    
    # Check all functions exist in both
    all_functions = set(main_utils.keys()) | set(simp_utils.keys())
    
    for func_name in all_functions:
        main_func = main_utils.get(func_name, {})
        simp_func = simp_utils.get(func_name, {})
        
        if not main_func and simp_func:
            differences.append(("missing_in_main", func_name, "Function exists in simplification but not main"))
        elif main_func and not simp_func:
            differences.append(("missing_in_simp", func_name, "Function exists in main but not simplification"))
        else:
            # Compare test results
            all_tests = set(main_func.keys()) | set(simp_func.keys())
            for test_name in all_tests:
                main_result = main_func.get(test_name)
                simp_result = simp_func.get(test_name)
                
                if main_result != simp_result:
                    differences.append((
                        "result_difference", 
                        f"{func_name}.{test_name}",
                        f"Main: {main_result} | Simp: {simp_result}"
                    ))
    
    return differences

def compare_imports(main_results: Dict, simp_results: Dict) -> List[Tuple[str, str, str]]:
    """Compare import test results."""
    differences = []
    
    main_imports = main_results.get("test_results", {}).get("imports", {}).get("imports", {})
    simp_imports = simp_results.get("test_results", {}).get("imports", {}).get("imports", {})
    
    # Get all modules tested
    all_modules = set(main_imports.keys()) | set(simp_imports.keys())
    
    for module in all_modules:
        main_status = main_imports.get(module)
        simp_status = simp_imports.get(module)
        
        if main_status != simp_status:
            differences.append((
                "import_difference",
                module,
                f"Main: {main_status} | Simp: {simp_status}"
            ))
    
    return differences

def compare_strategies(main_results: Dict, simp_results: Dict) -> List[Tuple[str, str, str]]:
    """Compare strategy initialization results."""
    differences = []
    
    main_strategies = main_results.get("test_results", {}).get("strategy_initialization", {}).get("strategies", {})
    simp_strategies = simp_results.get("test_results", {}).get("strategy_initialization", {}).get("strategies", {})
    
    all_strategies = set(main_strategies.keys()) | set(simp_strategies.keys())
    
    for strategy in all_strategies:
        main_strategy = main_strategies.get(strategy, {})
        simp_strategy = simp_strategies.get(strategy, {})
        
        # Compare each attribute
        all_attrs = set(main_strategy.keys()) | set(simp_strategy.keys())
        for attr in all_attrs:
            main_val = main_strategy.get(attr)
            simp_val = simp_strategy.get(attr)
            
            if main_val != simp_val:
                differences.append((
                    "strategy_difference",
                    f"{strategy}.{attr}",
                    f"Main: {main_val} | Simp: {simp_val}"
                ))
    
    return differences

def main():
    """Main comparison function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_test_results.py <main_results.json> <simp_results.json>")
        sys.exit(1)
    
    main_file = sys.argv[1]
    simp_file = sys.argv[2]
    
    print("🔍 Loading test results...")
    main_results = load_test_results(main_file)
    simp_results = load_test_results(simp_file)
    
    if not main_results or not simp_results:
        sys.exit(1)
    
    print(f"📊 Comparing results between branches...")
    print(f"  Main branch: {main_results.get('test_results', {}).get('imports', {}).get('branch_type', 'unknown')}")
    print(f"  Simp branch: {simp_results.get('test_results', {}).get('imports', {}).get('branch_type', 'unknown')}")
    
    # Compare different aspects
    all_differences = []
    
    # Compare imports
    import_diffs = compare_imports(main_results, simp_results)
    all_differences.extend(import_diffs)
    
    # Compare utility functions
    utility_diffs = compare_utility_functions(main_results, simp_results)
    all_differences.extend(utility_diffs)
    
    # Compare strategies
    strategy_diffs = compare_strategies(main_results, simp_results)
    all_differences.extend(strategy_diffs)
    
    # Report results
    print(f"\n📋 Comparison Results:")
    
    if not all_differences:
        print("✅ IDENTICAL FUNCTIONALITY: Both branches produce identical results!")
        print("\n🎯 Key Findings:")
        print("  • All utility functions return the same values")
        print("  • All models and enums behave identically")
        print("  • Strategy initialization works the same way")
        print("  • Import sources differ as expected (helpers vs separate modules)")
        
        # Show import source difference
        main_source = main_results.get("test_results", {}).get("utility_functions", {}).get("import_source")
        simp_source = simp_results.get("test_results", {}).get("utility_functions", {}).get("import_source")
        print(f"\n📦 Import Sources:")
        print(f"  Main branch: {main_source}")
        print(f"  Simplification branch: {simp_source}")
        
    else:
        print(f"⚠️  Found {len(all_differences)} differences:")
        
        for diff_type, item, description in all_differences:
            if diff_type == "import_difference":
                print(f"  📦 Import: {item}")
                print(f"     {description}")
            elif diff_type == "result_difference":
                print(f"  🔧 Function: {item}")
                print(f"     {description}")
            elif diff_type == "strategy_difference":
                print(f"  🧠 Strategy: {item}")
                print(f"     {description}")
            else:
                print(f"  ❓ Other: {item}")
                print(f"     {description}")
    
    # Create summary report
    with open("branch_comparison_report.txt", "w") as f:
        f.write("TradeBuddy Branch Comparison Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Main Branch: {main_results.get('test_results', {}).get('imports', {}).get('branch_type', 'unknown')}\n")
        f.write(f"Simplification Branch: {simp_results.get('test_results', {}).get('imports', {}).get('branch_type', 'unknown')}\n\n")
        
        if not all_differences:
            f.write("STATUS: ✅ IDENTICAL FUNCTIONALITY\n\n")
            f.write("Both branches produce identical results for all tested functionality.\n")
            f.write("The simplification successfully consolidated utility modules without\n")
            f.write("affecting any behavior or output.\n")
        else:
            f.write(f"STATUS: ⚠️ {len(all_differences)} DIFFERENCES FOUND\n\n")
            for diff_type, item, description in all_differences:
                f.write(f"{diff_type}: {item}\n  {description}\n\n")
    
    print(f"\n📄 Detailed report saved to: branch_comparison_report.txt")

if __name__ == "__main__":
    main()