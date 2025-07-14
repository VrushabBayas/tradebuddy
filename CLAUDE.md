# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBuddy is a **production-ready** AI-powered trading signal analysis system that combines Delta Exchange live market data with Ollama's Qwen2.5:14b local LLM to generate BUY/SELL signals. The system implements three distinct trading strategies optimized for 10x leverage cryptocurrency trading.

## Current System Status

✅ **Production Ready**: Complete implementation with modular architecture  
✅ **Live Data Integration**: Real-time market data from Delta Exchange API  
  
✅ **AI Analysis**: Local Ollama integration with Qwen2.5:14b model  
✅ **Interactive CLI**: Rich terminal interface with 5 operation modes  
✅ **Risk Management**: Optimized for 10x leverage with proper position sizing  
✅ **Modular Architecture**: Clean separation of concerns and reusable components  
✅ **Type Safety**: Comprehensive Pydantic models and configuration management  
✅ **High Test Coverage**: Comprehensive functionality-focused test suite with 90%+ coverage

## Development Principles

- **Always test the functionality and not the implementation**
- **Avoid over-engineering**
- **Test-Driven Quality**: Focus on testing what the system does, not how it does it

## Available Operation Modes

1. **Support & Resistance Analysis** - Key price level identification
2. **EMA Crossover Analysis** - Trend change detection with 9/15 EMA
3. **EMA Crossover V2 Analysis** - Enhanced EMA strategy with trend filtering and market structure analysis
4. **Combined Strategy** - High-confidence signals from multiple strategies
4. **Backtesting Mode** - Historical strategy performance analysis
5. **Exit** - Clean application termination

## Architecture Overview

### Modular Structure
```
src/
├── cli/
│   ├── main.py           # Main CLI application with 5 operation modes
│   └── displays.py       # Centralized display utilities (271 lines)
├── core/
│   ├── models.py         # Pydantic models with enhanced strategy configs
│   ├── config.py         # Environment-based configuration
│   └── constants.py      # Trading constants and parameters
├── data/
│   ├── delta_client.py   # Delta Exchange API client (200 candle default)
├── analysis/
│   ├── strategies/       # Four trading strategy implementations
│   │   ├── ema_crossover.py     # Original EMA crossover (9/15)
│   │   ├── ema_crossover_v2.py  # Enhanced with 50 EMA trend filter
│   │   ├── support_resistance.py # Price level analysis
│   │   └── combined.py          # Multi-strategy confluence
│   ├── indicators.py     # Technical indicator calculations
│   ├── ema_analysis.py   # Advanced EMA trend analysis
│   ├── market_context.py # Market structure analysis
│   └── scoring/          # Signal scoring framework
│       └── signal_scorer.py # Configurable scoring system
├── backtesting/
│   ├── engine.py         # Backtesting engine
│   └── models.py         # Backtesting configuration
└── utils/
    ├── data_helpers.py   # Safe data access utilities
    ├── type_conversion.py # Type safety conversions
    └── risk_management.py # Position sizing calculations
```

### Key Components

**CLI Layer** (`src/cli/`):
- `main.py`: Application orchestration and traditional analysis flows
- `realtime.py`: Real-time streaming and continuous monitoring modes  
- `displays.py`: Reusable UI components and formatting utilities

**Configuration Models**:
- `SessionConfig`: Analysis session parameters

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


## Recent Major Updates (Latest Changes)

### EMA Crossover V2 Strategy Implementation
- **Enhanced EMA Strategy**: New `ema_crossover_v2.py` with advanced features
  - 50 EMA trend filter for higher quality signals
  - Multi-factor trend strength scoring (0-100 scale)
  - Market structure analysis with swing detection
  - Volatility-aware signal generation
  - Enhanced confidence scoring system

- **Signal Scoring Framework**: New modular scoring system (`src/analysis/scoring/`)
  - Configurable signal assessment components
  - Weighted multi-factor analysis
  - Reusable across all strategies
  - Performance and risk metrics integration

- **Market Analysis Modules**: 
  - `ema_analysis.py`: Advanced EMA trend analysis
  - `market_context.py`: Market structure and trend quality assessment
  - Enhanced base strategy class with improved market analysis

### Architecture Enhancements
- **Simplified Strategy Options**: Now 3 distinct strategies
- **Enhanced Configuration**: Updated models with V2 strategy parameters
- **Improved Buffer Management**: Default 200 historical candles for better context
- **Type Safety**: Comprehensive Pydantic V2 validation
- **Test Coverage**: 90%+ coverage with functionality-focused testing

### Modular Architecture (Previous Implementation)
- **Display System**: `src/cli/displays.py` (271 lines)  
- **Streamlined CLI**: `src/cli/main.py` with 5 operation modes
- **Shared Utilities**: `src/utils/` with data helpers and type conversions

## Usage Patterns

### Quick Start
```bash
make setup-env    # One-time setup
make run          # Start application
# Select option 3 for combined strategy analysis
```

### Development Workflow
```bash
make quick-test   # format + lint + unit tests
make test-cov     # full test suite with coverage
make run-dev      # development mode with debug logging
```

### Strategy Development
- Add new strategies to `src/analysis/strategies/`
- Use `enhanced_base_strategy.py` as foundation for advanced features
- Implement signal scoring in `src/analysis/scoring/signal_scorer.py`
- Update `StrategyType` enum in `src/core/models.py`
- Register strategy in CLI application (`src/cli/main.py`)

### Analysis Module Development
- Add indicators to `src/analysis/indicators.py`
- Enhance market analysis in `src/analysis/ema_analysis.py` or `market_context.py`
- Use shared utilities from `src/utils/` for data handling and type safety

The system is now **production-ready** with modular architecture and robust analysis features for professional cryptocurrency trading.

## Testing Infrastructure

### Test Philosophy
The test suite follows a **functionality-focused approach** rather than implementation testing:
- Tests verify **what the system does** (behavior and outcomes)
- Avoids testing **how the system does it** (internal implementation details)
- Emphasizes real-world scenarios and edge case handling
- Maintains 90%+ code coverage with meaningful tests

### Test Suite Structure
```
tests/
├── unit/
│   ├── analysis/
│   │   ├── test_ollama_functionality.py      # AI analysis functionality
│   │   └── strategies/                       # Strategy behavior tests
│   ├── cli/
│   │   └── test_cli_workflow_functionality.py # End-to-end CLI workflows
│   ├── data/
│   │   └── test_data_functionality.py        # Data processing
│   ├── backtesting/
│   │   └── test_backtesting_functionality.py # Risk management & backtesting
│   └── test_edge_cases_functionality.py      # Edge cases & error handling
├── integration/
│   └── test_integration.py                   # System integration tests
└── fixtures/
    └── conftest.py                           # Shared test fixtures
```

### Key Test Categories

**1. AI Analysis Functionality** (`test_ollama_functionality.py`):
- Market data analysis workflows
- Prompt generation and processing
- AI response parsing and validation
- Error handling for AI service issues

**2. CLI Workflow Testing** (`test_cli_workflow_functionality.py`):
- Complete user journeys through the application
- Strategy selection and execution flows
- Interactive CLI component behavior

**3. Data Processing Tests** (`test_data_functionality.py`):
- Delta Exchange API integration
- Data validation and error handling
- Market data retrieval and processing

**4. Risk Management Tests** (`test_backtesting_functionality.py`):
- Position sizing calculations
- Stop loss and take profit logic
- Leverage impact on risk calculations
- Portfolio management functionality
- Backtesting engine workflows

**5. Edge Case Handling** (`test_edge_cases_functionality.py`):
- Invalid data scenarios (malformed JSON, network errors)
- Extreme values (very large/small numbers, zero volume)
- Concurrent operations and performance stress tests
- Error recovery and graceful degradation

### Test Execution Commands
```bash
# Run complete test suite
make test                    # All tests with standard output
make test-cov               # Tests with coverage report (target: 90%+)

# Focused testing
make test-unit              # Unit tests only
make test-integration       # Integration tests only
make test-watch            # Watch mode for development

# Quality assurance
make lint                   # Code quality checks
make format                 # Code formatting
make pre-commit            # Pre-commit validation
make ci                    # Full CI pipeline
```

### Coverage Targets
- **Overall Coverage**: 90%+ (improved from initial 6%)
- **Critical Components**: 95%+ coverage
  - Risk management functions
  - Trading strategy logic
  - Data processing pipelines
  - API integration layers

### Testing Best Practices
1. **Functionality Focus**: Test business logic and user-facing behavior
2. **Realistic Scenarios**: Use real market data patterns in tests
3. **Error Resilience**: Comprehensive error handling validation
4. **Performance Awareness**: Include performance and concurrency tests
5. **Documentation**: Clear test descriptions explaining expected behavior

### Recent Test Improvements
- **Symbol Enum Migration**: Updated from `BTCUSDT` to `BTCUSD` throughout test suite
- **SessionConfig Enhancement**: Support for new risk management model with INR-based calculations
- **Type Safety**: Comprehensive Pydantic V2 validation in test fixtures
- **Edge Case Coverage**: Extensive testing of unusual conditions and error scenarios
- **Functional Testing**: Transformed implementation tests to behavior-focused tests

The test suite provides confidence in system reliability and serves as living documentation of expected system behavior.