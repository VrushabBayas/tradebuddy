# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBuddy is a **production-ready** AI-powered trading signal analysis system that combines Delta Exchange live market data with Ollama's Qwen2.5:14b local LLM to generate BUY/SELL signals. The system implements three distinct trading strategies optimized for 10x leverage cryptocurrency trading.

## Current System Status

✅ **Production Ready**: Complete implementation with modular architecture  
✅ **Live Data Integration**: Real-time market data from Delta Exchange API  
✅ **WebSocket Streaming**: Live candlestick data with real-time analysis  
✅ **Market Monitoring**: Continuous monitoring mode with automated alerts  
✅ **AI Analysis**: Local Ollama integration with Qwen2.5:14b model  
✅ **Interactive CLI**: Rich terminal interface with 6 operation modes  
✅ **Risk Management**: Optimized for 10x leverage with proper position sizing  
✅ **Modular Architecture**: Clean separation of concerns and reusable components  
✅ **Type Safety**: Comprehensive Pydantic models and configuration management  

## Development Principles

- **Always test the functionality and not the implementation**
- **Avoid over-engineering**

## Available Operation Modes

1. **Support & Resistance Analysis** - Key price level identification
2. **EMA Crossover Analysis** - Trend change detection with 9/15 EMA
3. **Combined Strategy** - High-confidence signals from multiple strategies
4. **Real-Time Analysis ⭐** - Live market streaming (5-60 minutes)
5. **Market Monitoring 🔄** - Continuous monitoring with automated alerts
6. **Exit** - Clean application termination

## Architecture Overview

### Modular Structure
```
src/
├── cli/
│   ├── main.py           # Main CLI application (389 lines)
│   ├── realtime.py       # Real-time analysis & monitoring (785 lines)
│   └── displays.py       # Centralized display utilities (271 lines)
├── core/
│   ├── models.py         # Pydantic models with RealTimeConfig & MonitoringConfig
│   ├── config.py         # Environment-based configuration
│   └── constants.py      # Trading constants and parameters
├── data/
│   ├── delta_client.py   # Delta Exchange API client
│   └── websocket_client.py # WebSocket streaming client
└── analysis/
    ├── strategies/       # Trading strategy implementations
    └── indicators.py     # Technical indicator calculations
```

### Key Components

**CLI Layer** (`src/cli/`):
- `main.py`: Application orchestration and traditional analysis flows
- `realtime.py`: Real-time streaming and continuous monitoring modes  
- `displays.py`: Reusable UI components and formatting utilities

**Real-Time Capabilities**:
- **Live Streaming**: WebSocket candlestick data from Delta Exchange
- **Buffer Management**: Configurable OHLCV data buffers (20-200 candles)
- **Strategy Analysis**: Real-time application of trading strategies
- **Monitoring Mode**: Continuous multi-symbol monitoring with alerts

**Configuration Models**:
- `RealTimeConfig`: Type-safe real-time session configuration
- `MonitoringConfig`: Continuous monitoring parameters and behavior
- `SessionConfig`: Traditional analysis session parameters

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

## Real-Time Features

### Live Market Analysis (Option 4)
```bash
# Interactive session configuration:
Strategy: EMA Crossover or Combined Strategy
Symbol: BTCUSDT (default), ETHUSDT, SOLUSDT, ADAUSDT, DOGEUSDT  
Timeframe: 1m (default for real-time)
Duration: 5-60 minutes
Buffer Size: 20-100 candles
```

**Features**:
- Historical data pre-loading for context
- Live WebSocket candlestick streaming  
- Real-time strategy analysis on each new candle
- Live signal generation with confidence scoring
- Interactive session with progress tracking

### Continuous Market Monitoring (Option 5)
```bash
# Multi-symbol monitoring configuration:
Strategy: EMA Crossover or Combined Strategy
Symbols: Multi-select (comma-separated, e.g., 1,2,3)
Signal Threshold: 5-10 confidence level for alerts
Refresh Interval: 30-300 seconds between analyses
Display: Compact or detailed alert format
```

**Features**:
- Multi-symbol simultaneous monitoring
- Configurable signal confidence thresholds
- Automated alert system for high-confidence signals
- Signal history tracking (last 50 signals per symbol)
- Continuous operation until manually stopped (Ctrl+C)
- Periodic analysis with customizable intervals

### Configuration Models

**RealTimeConfig** (`src/core/models.py`):
- Session-based real-time analysis (5-60 minutes)
- Single symbol focus with intensive analysis
- Buffer management for live candlestick processing

**MonitoringConfig** (`src/core/models.py`):
- Multi-symbol continuous monitoring  
- Configurable alert thresholds and intervals
- Signal history and session statistics
- Display customization options

## Recent Refactoring (Latest Changes)

### Modular Architecture Implementation
- **Extracted Real-Time Module**: `src/cli/realtime.py` (785 lines)
  - Complete real-time streaming functionality
  - Monitoring mode with multi-symbol support
  - Type-safe configuration management

- **Centralized Display System**: `src/cli/displays.py` (271 lines)  
  - Reusable UI components for consistent styling
  - Table creation utilities and panel formatting
  - Risk management display helpers

- **Streamlined Main CLI**: `src/cli/main.py` (389 lines)
  - 36% reduction from 609 lines (220 lines removed)
  - Focused on application flow and orchestration
  - Clean separation of concerns

### Code Quality Improvements
- **Type Safety**: Added `MonitoringConfig` and enhanced `RealTimeConfig`
- **Error Handling**: Comprehensive exception handling in monitoring mode
- **Resource Management**: Proper cleanup and connection management
- **Performance**: Optimized buffer management and API calls

## Usage Patterns

### Quick Start
```bash
make setup-env    # One-time setup
make run          # Start application
# Select option 4 or 5 for real-time features
```

### Development Workflow
```bash
make quick-test   # format + lint + unit tests
make test-cov     # full test suite with coverage
make run-dev      # development mode with debug logging
```

### Real-Time Development
- Modify `src/cli/realtime.py` for streaming/monitoring logic
- Update `src/core/models.py` for configuration changes  
- Use `src/cli/displays.py` for UI component additions

The system is now **production-ready** with comprehensive real-time capabilities, modular architecture, and robust monitoring features for professional cryptocurrency trading analysis.