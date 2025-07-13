# Multi-stage build for TradeBuddy
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd --gid 1000 tradebuddy && \
    useradd --uid 1000 --gid tradebuddy --shell /bin/bash --create-home tradebuddy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements/ requirements/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements/prod.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -r requirements/dev.txt

# Copy source code
COPY --chown=tradebuddy:tradebuddy . .

# Switch to app user
USER tradebuddy

# Expose port for development server
EXPOSE 8000

# Command for development
CMD ["python", "tradebuddy.py"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=tradebuddy:tradebuddy src/ src/
COPY --chown=tradebuddy:tradebuddy tradebuddy.py .
COPY --chown=tradebuddy:tradebuddy pyproject.toml .
COPY --chown=tradebuddy:tradebuddy .env.example .env

# Create necessary directories
RUN mkdir -p backtest_reports logs && \
    chown -R tradebuddy:tradebuddy /app

# Switch to app user
USER tradebuddy

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '.'); from src.core.config import get_config; get_config()" || exit 1

# Default command
CMD ["python", "tradebuddy.py"]