# Docker Deployment Guide

## Quick Start

### 1. Setup Environment
```bash
# Check if .env file exists (should already exist)
ls -la .env

# For production deployment, edit .env with your API keys:
nano .env

# Required for trading operations:
DELTA_API_KEY=your_actual_api_key_here
DELTA_API_SECRET=your_actual_secret_here

# Optional for FinGPT integration:
FINGPT_API_KEY=your_fingpt_key_here

# For development/testing, you can leave API keys empty
```

### 2. Basic Deployment
```bash
# Start TradeBuddy with Ollama
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

## Deployment Options

### Production Mode
```bash
make docker-up
```
- Runs optimized production build
- Includes Ollama AI service
- Persistent data volumes

### Development Mode
```bash
make docker-dev
```
- Live code reloading
- Full development tools
- Debug logging enabled

### FinGPT Integration
```bash
make docker-fingpt
```
- Includes your existing FinGPT server
- Full AI model comparison capabilities

## Service Management

```bash
# Build images
make docker-build

# View logs
make docker-logs

# Clean up resources
make docker-clean
```

## Configuration

### Environment Variables

**Required for trading operations:**
- `DELTA_API_KEY` - Your Delta Exchange API key (can be empty for testing)
- `DELTA_API_SECRET` - Your Delta Exchange secret (can be empty for testing)

**AI Model Configuration:**
- `OLLAMA_HOST` - Ollama service URL (default: http://ollama:11434)
- `FINGPT_API_ENDPOINT` - FinGPT API URL (default: http://fingpt-api:8000)
- `FINGPT_API_KEY` - FinGPT API key (optional, can be empty)

**Application Settings:**
- `PYTHON_ENV` - Environment mode (development/production)
- `DEBUG` - Enable debug logging (true/false)

### API Key Setup

**For Development/Testing:**
- Leave API keys empty in `.env`
- TradeBuddy will work in analysis-only mode
- All strategies and AI models will function for backtesting

**For Production Trading:**
- Add real Delta Exchange API credentials
- Required for live market data and trading operations

## Troubleshooting

**Environment Variable Warnings**: Normal if API keys are empty - application will work in analysis mode
**TradeBuddy connection issues**: Check `.env` file exists and is properly formatted
**Ollama connection failed**: Wait for Ollama service: `docker-compose logs ollama`
**Port conflicts**: Modify ports in `docker-compose.yml`

**Common Issues:**
- Missing `.env` file: Copy from `.env.docker` template
- Invalid API keys: Use empty values for development
- Permission errors: Check file ownership and Docker permissions
- **Port 11434 already in use**: Docker uses port 11435 to avoid conflicts with local Ollama
- Service startup fails: Check `docker-compose logs <service-name>` for detailed errors