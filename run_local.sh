#!/bin/bash

# TradeBuddy Local Startup Script
# This script activates the virtual environment and runs TradeBuddy locally

# Get script directory to find TradeBuddy root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADEBUDDY_ROOT="$SCRIPT_DIR"

echo "🚀 Starting TradeBuddy locally (de-dockerized)"
echo "TradeBuddy root: $TRADEBUDDY_ROOT"
echo "=====================================:"

# Change to TradeBuddy directory
cd "$TRADEBUDDY_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env"
    exit 1
fi

# Export environment variables
echo "📝 Loading environment variables..."
export $(grep -v '^#' .env | xargs)

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import src.core.config; print('✅ Core modules imported successfully')" || {
    echo "❌ Failed to import core modules"
    exit 1
}

# Check external services
echo "🌐 Checking external services..."

# Check FinGPT service (our FinBERT server)
curl -s --max-time 3 "http://localhost:8001/health" > /dev/null && \
    echo "✅ FinGPT service available at ${FINGPT_API_ENDPOINT}" || \
    echo "⚠️  FinGPT service not available (optional)"

# Check Ollama service
curl -s --max-time 3 "${OLLAMA_API_URL}/api/tags" > /dev/null && \
    echo "✅ Ollama service available at ${OLLAMA_API_URL}" || \
    echo "⚠️  Ollama service not available - install Ollama or use FinGPT only"

echo ""
echo "🎯 Starting TradeBuddy..."
echo "📝 Logs will appear below:"
echo "================================="

# Run TradeBuddy
python -m src "$@"