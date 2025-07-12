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

## Development Principles

- **alwasy test the functionality and not the implimenttion**

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

[... rest of the file remains unchanged ...]