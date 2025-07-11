# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBuddy is an AI-powered trading signal analysis system that combines Delta Exchange live market data with Ollama's Qwen2.5:14b local LLM to generate BUY/SELL signals. The system implements three distinct trading strategies that can be used standalone or in combination.

## Architecture

### Three-Strategy Approach
1. **Support and Resistance Strategy** - Analyzes key price levels where price historically bounces or gets rejected
2. **9 EMA and 15 EMA Crossover Strategy** - Uses exponential moving average crossovers to identify trend changes
3. **Combined Strategy** - Merges both approaches for maximum confidence signals

### Data Flow
```
Delta Exchange API → Data Processing → Ollama Qwen2.5:14b → Signal Generation → Trading Decision
```

## Technology Stack

- **Data Source**: Delta Exchange REST API and WebSocket feeds
- **AI Model**: Ollama Qwen2.5:14b (local LLM, 14 billion parameters)
- **Implementation Language**: Python (planned)
- **Key Dependencies**: requests, websocket-client, pandas, numpy (for future implementation)

## Development Setup

### System Requirements
- **RAM**: 16GB minimum, 32GB recommended for Qwen2.5:14b
- **Storage**: 10GB for model + additional space for data
- **CPU**: Modern multi-core processor
- **Internet**: Stable connection for Delta Exchange API

### Ollama Setup
```bash
# Install Ollama from https://ollama.ai/
# Download the model (8.7GB)
ollama pull qwen2.5:14b

# Test the model
ollama run qwen2.5:14b
```

### Delta Exchange Setup
- **Account**: Free account for market data access
- **API Access**: Public market data available without API keys
- **Documentation**: https://docs.delta.exchange/
- **Rate Limits**: 10 requests/second for REST API
- **WebSocket**: Unlimited real-time data streams

## Strategy Implementation Guide

### Mode Selection Logic
- **Mode 1 (S/R Only)**: Strong trending markets with clear support/resistance levels
- **Mode 2 (EMA Only)**: Trending markets with clear momentum, longer timeframes
- **Mode 3 (Combined)**: Uncertain conditions requiring confirmation, major positions

### Signal Confidence Thresholds
- **Entry Signals**: Only enter when confidence >6
- **Strong Signals**: Confidence >8 (both strategies agree in Mode 3)
- **Position Sizing**: 2-4% portfolio based on confidence level

### Risk Management
- **Stop Loss**: Always 2-3% from entry
- **Take Profit**: Minimum 2:1 reward-to-risk ratio
- **Position Limits**: Max 5% portfolio for confluence signals

## Data Requirements

### From Delta Exchange
- **Symbols**: BTC, ETH, SOL, 100+ crypto futures/perpetuals
- **Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **Data Points**: OHLCV (Open, High, Low, Close, Volume)
- **Historical**: Up to 1000 candles per request
- **Real-time**: WebSocket feeds for live updates

### Calculated Indicators
- **EMAs**: 9-period and 15-period exponential moving averages
- **S/R Levels**: 3-5 key historical support/resistance prices
- **Volume Analysis**: Current vs 20-period average volume

## Implementation Phases

### Phase 1: Data Integration
1. Implement Delta Exchange REST API client
2. Set up WebSocket connection for real-time data
3. Create data processing pipeline for EMA calculations
4. Build support/resistance level identification

### Phase 2: AI Integration
1. Format market data for Qwen2.5:14b consumption
2. Create prompt templates for all three modes
3. Implement signal parsing and validation
4. Test with historical data

### Phase 3: Strategy Testing
1. **Week 1**: Test Mode 1 (S/R) with paper trading
2. **Week 2**: Test Mode 2 (EMA) with paper trading
3. **Week 3**: Test Mode 3 (Combined) with paper trading
4. **Week 4**: Performance comparison and optimization

### Phase 4: Live Implementation
1. Start with minimal position sizes
2. Monitor performance for first month
3. Gradually scale based on results
4. Continuous refinement based on market feedback

## Testing Strategy

### Validation Methods
- **Historical Backtest**: 3 months of Delta Exchange data
- **Paper Trading**: 2 weeks live signals
- **Performance Tracking**: Daily signal accuracy by mode
- **Success Metrics**: >60% profitable signals, <5s data latency

### Performance Targets
- **Signal Accuracy**: >60% across all modes
- **Data Latency**: <5 seconds from Delta Exchange to analysis
- **AI Response Time**: <10 seconds for Qwen2.5:14b
- **Risk-Reward**: Average 1.5:1 or better

## Key Files Reference

- **trading_strategy_framework.md**: Complete project specifications (18KB detailed documentation)
- **README.md**: Project overview
- **LICENSE**: MIT License

## Development Guidelines

### Code Structure (Future Implementation)
```
tradebuddy/
├── src/
│   ├── data/          # Delta Exchange API integration
│   ├── analysis/      # Ollama AI integration
│   ├── strategies/    # Three strategy implementations
│   └── utils/         # Common utilities
├── tests/            # Unit and integration tests
├── config/           # Configuration files
└── docs/             # Additional documentation
```

### API Integration Notes
- Use WebSocket for real-time data (unlimited)
- REST API for historical data (10 req/sec limit)
- Implement retry logic with exponential backoff
- Validate data integrity before AI analysis
- Have backup data sources ready

### AI Model Optimization
- Monitor RAM usage (8-16GB required)
- Optimize prompts for <10 second responses
- Keep Ollama and model updated
- Implement output validation and parsing
- Leverage 24/7 local availability

## Troubleshooting

### Common Issues
- **Ollama slow responses**: Check RAM availability, close unnecessary apps
- **Delta Exchange API timeouts**: Implement retry logic, use WebSocket for real-time
- **Signal conflicts**: Clear resolution rules for conflicting signals
- **Model crashes**: Monitor system resources, restart if needed

### Mode Switching Guidelines
- **Trending markets**: Favor Mode 2 (EMA)
- **Range-bound markets**: Favor Mode 1 (S/R)
- **Uncertain markets**: Use Mode 3 (Combined)
- **High volatility**: Reduce position sizes across all modes

## Running the Project

### Project Execution Commands
- Use below commands to run the project
  - `python src/cli/main.py`
  - `make run`