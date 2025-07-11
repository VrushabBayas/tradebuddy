# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBuddy is a **production-ready** AI-powered trading signal analysis system that combines Delta Exchange live market data with Ollama's Qwen2.5:14b local LLM to generate BUY/SELL signals. The system implements three distinct trading strategies optimized for 10x leverage cryptocurrency trading.

## Current System Status

✅ **Production Ready**: Complete implementation with comprehensive refactoring  
✅ **Live Data Integration**: Real-time market data from Delta Exchange API  
✅ **AI Analysis**: Local Ollama integration with Qwen2.5:14b model  
✅ **Interactive CLI**: Rich terminal interface with strategy selection  
✅ **Risk Management**: Optimized for 10x leverage with proper position sizing  
✅ **Code Quality**: Recently refactored with shared utilities and type safety  

## Essential Development Commands

```bash
# Setup (run once)
make setup-env                 # Complete environment setup
source venv/bin/activate       # Activate virtual environment

# Development workflow
make test                      # Run full test suite
make test-cov                  # Run tests with coverage report
make lint                      # Run linting (flake8, mypy, bandit)
make format                    # Format code (black, isort)
make run                       # Run the application

# Quick development cycle
make quick-test                # format + lint + test-unit
make pre-commit                # Run pre-commit checks
make ci                        # Full CI pipeline simulation

# Testing variants
make test-unit                 # Unit tests only
make test-integration          # Integration tests only

# Run application variants
make run-dev                   # Development mode with debug logging
make run-demo                  # System demonstration
```

## Architecture Overview

### Core System Design
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │  Analysis Layer │    │   Data Layer    │
│                 │    │                 │    │                 │
│ • Interactive   │───▶│ • 3 Strategies  │───▶│ • Delta Exchange│
│   Terminal      │    │ • Technical     │    │   API Client    │
│ • Strategy      │    │   Indicators    │    │ • Live Market   │
│   Selection     │    │ • AI Analysis   │    │   Data          │
│ • Risk Config   │    │ • Signal Gen    │    │ • Rate Limiting │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Utilities     │
                    │                 │
                    │ • Data Helpers  │
                    │ • Type Convert  │
                    │ • Risk Mgmt     │
                    │ • Constants     │
                    └─────────────────┘
```

### Three-Strategy System
1. **Support/Resistance Strategy** - Identifies key price levels with volume confirmation
2. **EMA Crossover Strategy** - 9 EMA and 15 EMA crossover signals  
3. **Combined Strategy** - Requires confirmation from both strategies

### Data Flow
```
Delta Exchange API → Market Data → Technical Analysis → AI Analysis → Signal Generation → Risk Management → Trading Decision
```

## Key Refactoring Improvements (Recently Completed)

### 1. Shared Utilities System
- **`src/utils/data_helpers.py`**: Centralized data manipulation utilities
- **`src/utils/type_conversion.py`**: Safe type conversion functions
- **`get_value()` function**: Safely accesses Pydantic models and dictionaries
- **Eliminated code duplication**: Removed duplicate helper functions across strategies

### 2. Type Safety Enhancements
- **Fixed Decimal/float mixing**: Safe conversions before arithmetic operations
- **Type-safe utilities**: `to_float()`, `to_decimal()`, `financial_*()` functions
- **Consistent type handling**: Standardized across all financial calculations

### 3. Constants Centralization
- **`src/core/constants.py`**: Added 23+ trading constants
- **Eliminated magic numbers**: Replaced hardcoded values with descriptive constants
- **TradingConstants class**: Centralized thresholds and trading parameters

### 4. Architecture Improvements
- **Cleaner imports**: Removed unused imports and fixed shadowing issues
- **Consistent patterns**: Standardized utility usage across strategy files
- **Better error handling**: Improved type conversion error handling

## Critical Implementation Details

### Type Conversion Pattern (IMPORTANT)
```python
# Always use safe type conversion utilities
from src.utils.data_helpers import get_value
from src.utils.type_conversion import to_float

# Correct approach - use shared utilities
ema_9 = to_float(get_value(ema_crossover, 'ema_9', 0))
price_diff = to_float(current_price) - to_float(support_price)

# Avoid direct attribute access without conversion
# price_diff = current_price - level.level  # May cause type errors
```

### Safe Data Access Pattern
```python
# Use get_value() for both Pydantic models and dictionaries
from src.utils.data_helpers import get_value

# Works with both model.field and dict['field']
level_price = get_value(level, 'level', 0)
is_support = get_value(level, 'is_support', False)
```

### Constants Usage
```python
# Use centralized constants instead of magic numbers
from src.core.constants import TradingConstants

# Correct - use constants
if volume_ratio > TradingConstants.VOLUME_CONFIRMATION_THRESHOLD:
    confidence += TradingConstants.VOLUME_CONFIDENCE_BONUS

# Avoid hardcoded values
# if volume_ratio > 1.2:  # Magic number
```

## File Organization and Key Modules

### Core Strategy Files
- **`src/analysis/strategies/base_strategy.py`**: Abstract base class for all strategies
- **`src/analysis/strategies/support_resistance.py`**: S/R strategy with shared utilities
- **`src/analysis/strategies/ema_crossover.py`**: EMA crossover strategy
- **`src/analysis/strategies/combined.py`**: Combined strategy with confirmation logic

### Utility Modules (Recently Added)
- **`src/utils/data_helpers.py`**: Data manipulation and safe access utilities
- **`src/utils/type_conversion.py`**: Type conversion and financial arithmetic
- **`src/utils/risk_management.py`**: Position sizing and risk calculations
- **`src/utils/logging.py`**: Structured logging configuration
- **`src/utils/environment.py`**: Environment validation

### Core Infrastructure
- **`src/core/models.py`**: Pydantic models with proper type validation
- **`src/core/constants.py`**: Centralized constants and trading parameters
- **`src/core/config.py`**: Environment-based configuration management
- **`src/core/exceptions.py`**: Custom exception hierarchy

### Data Integration
- **`src/data/delta_client.py`**: Delta Exchange API client with proper error handling
- **`src/analysis/indicators.py`**: Technical indicator calculations
- **`src/analysis/ollama_client.py`**: AI analysis integration

## Development Patterns and Best Practices

### Adding New Strategy Logic
1. **Extend BaseStrategy**: Inherit from `src/analysis/strategies/base_strategy.py`
2. **Use shared utilities**: Import from `src/utils/data_helpers.py` and `src/utils/type_conversion.py`
3. **Use constants**: Reference `TradingConstants` instead of hardcoded values
4. **Safe type conversion**: Always use `to_float()` before arithmetic operations

### Working with Market Data
```python
# Pattern for processing market data
from src.utils.data_helpers import get_value
from src.utils.type_conversion import to_float

# Safe data extraction
for level in sr_levels:
    level_price = to_float(get_value(level, 'level', 0))
    is_support = get_value(level, 'is_support', False)
    
    # Safe arithmetic operations
    distance = abs(to_float(current_price) - level_price)
    distance_pct = (distance / to_float(current_price)) * 100
```

### Error Handling Pattern
```python
# Use structured logging with context
logger.info(
    "Analysis completed",
    strategy=self.strategy_type.value,
    symbol=market_data.symbol,
    signals_count=len(signals)
)

# Graceful error handling
try:
    analysis_result = await self._generate_analysis()
except Exception as e:
    logger.error("Analysis failed", error=str(e))
    raise StrategyError(f"Analysis failed: {str(e)}")
```

## Risk Management System

### Current Settings (Optimized for 10x Leverage)
- **Position Size**: 5.0% of account (configurable)
- **Stop Loss**: 1.5% (tighter for leverage control)
- **Take Profit**: 3.0% (realistic for crypto markets)
- **Leverage**: 10x (configurable 1-100x)
- **Min Lot Size**: 0.001 BTC (Delta Exchange standard)

### Risk Calculation Functions
```python
# Located in src/utils/risk_management.py
calculate_leveraged_position_size()     # Position sizing with leverage
calculate_stop_loss_take_profit()       # Risk level calculations
optimize_position_for_delta_exchange()  # Exchange-specific optimization
validate_position_safety()             # Safety checks and warnings
```

## Testing Framework

### Test Structure
```
tests/
├── unit/                 # Unit tests for individual components
├── integration/         # End-to-end integration tests
├── fixtures/           # Test data and fixtures
└── conftest.py         # Pytest configuration with comprehensive fixtures
```

### Running Tests
```bash
# Basic test commands
make test                 # Full test suite
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-cov            # Tests with coverage report

# Development testing
make test-watch          # Watch mode for development
make quick-test          # format + lint + test-unit
```

## Common Development Tasks

### Adding New Constants
1. Add to `src/core/constants.py` in the `TradingConstants` class
2. Use descriptive names and comments
3. Update existing files to use the new constants

### Modifying Strategy Logic
1. Update the appropriate strategy file in `src/analysis/strategies/`
2. Use shared utilities from `src/utils/`
3. Maintain type safety with proper conversions
4. Add corresponding tests

### Working with Configuration
1. Environment variables defined in `.env.example`
2. Configuration models in `src/core/config.py`
3. Use `Settings` class for type-safe configuration access

## Troubleshooting Common Issues

### Import Errors
```bash
# Check Python path and virtual environment
make check-env
source venv/bin/activate

# Ensure all dependencies are installed
make install
```

### Type Conversion Errors
```python
# Always use safe conversion utilities
from src.utils.type_conversion import to_float, to_decimal

# Convert before arithmetic operations
result = to_float(value1) + to_float(value2)
```

### Build Dependencies (macOS)
```bash
# Fix build dependencies automatically
make fix-build-deps

# Or manual setup
brew install pkg-config
xcode-select --install
```

### Ollama Connection Issues
```bash
# Start Ollama service
ollama serve

# Verify model availability
ollama pull qwen2.5:14b
ollama run qwen2.5:14b
```

## API Integration Notes

### Delta Exchange API
- **Base URL**: `https://api.delta.exchange`
- **Rate Limit**: 10 requests/second (automatically handled)
- **Authentication**: Public endpoints (no API key required for market data)
- **Key Fields**: Uses `mark_price` for accurate pricing

### Ollama Integration
- **Local LLM**: Qwen2.5:14b model (14B parameters)
- **Timeout**: 30 seconds for analysis requests
- **Privacy**: Runs entirely on local machine
- **Requirements**: 16GB+ RAM recommended

## Performance Characteristics

### Response Times
- **Market Data Retrieval**: <2 seconds
- **Technical Analysis**: <1 second
- **AI Analysis**: <10 seconds (with Ollama)
- **Signal Generation**: <2 seconds total

### Resource Usage
- **RAM**: 8-16GB (for Qwen2.5:14b model)
- **CPU**: Minimal during analysis
- **Network**: <1MB per analysis cycle

## Entry Points

### Main Application
- **`tradebuddy.py`**: Primary entry point with path handling
- **`src/cli/main.py`**: CLI application logic
- **`src/__main__.py`**: Module execution support

### Development Tools
- **`test_market_data_only.py`**: Test without AI
- **`test_risk_management.py`**: Test risk calculations
- **`demo_complete_system.py`**: Full system demonstration

## Code Quality Standards

### Automated Checks
- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting and organization
- **flake8**: Linting and style checking
- **mypy**: Type checking and validation
- **bandit**: Security analysis
- **pytest**: Test execution with coverage >80%

### Development Workflow
1. **Make changes** to source code
2. **Run quick-test**: `make quick-test` (format + lint + unit tests)
3. **Run full tests**: `make test-cov` (with coverage)
4. **Test application**: `make run` (manual testing)
5. **Pre-commit checks**: `make pre-commit` (before committing)

## Future Considerations

### Known Limitations
- **WebSocket**: Real-time data streaming not fully implemented
- **Backtesting**: Limited historical testing capabilities
- **Multi-timeframe**: Single timeframe analysis only

### Potential Enhancements
- **Real-time WebSocket**: For tick-by-tick data
- **Portfolio Management**: Multiple position tracking
- **Advanced Strategies**: Machine learning integration
- **Performance Analytics**: Detailed performance tracking

The codebase is well-structured, thoroughly tested, and ready for production use. The recent refactoring has significantly improved code quality, maintainability, and type safety while maintaining all existing functionality.