# Docker Deployment Guide

## Quick Start

### 1. Setup Environment
```bash
# Copy Docker environment template
cp .env.docker .env

# Edit .env with your API keys
nano .env
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

Essential environment variables in `.env`:
- `DELTA_API_KEY` - Your Delta Exchange API key
- `DELTA_API_SECRET` - Your Delta Exchange secret
- `OLLAMA_HOST` - Ollama service URL (default: http://ollama:11434)
- `FINGPT_API_ENDPOINT` - FinGPT API URL (default: http://fingpt-api:8000)

## Troubleshooting

**TradeBuddy won't start**: Check API keys in `.env`
**Ollama connection failed**: Wait for Ollama service: `docker-compose logs ollama`
**Port conflicts**: Modify ports in `docker-compose.yml`