# TradeBuddy Development Makefile
# ================================

.PHONY: help install test test-cov lint format run clean setup-env setup-venv check-env

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

# Python executable detection
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null || echo "python-not-found")
PIP := pip
PYTEST := pytest
VENV_DIR := venv

# Handle different platforms for venv paths
ifeq ($(OS),Windows_NT)
    VENV_BIN := $(VENV_DIR)/Scripts
    VENV_PYTHON := $(VENV_BIN)/python.exe
    VENV_PIP := $(VENV_BIN)/pip.exe
    VENV_PYTEST := $(VENV_BIN)/pytest.exe
else
    VENV_BIN := $(VENV_DIR)/bin
    VENV_PYTHON := $(VENV_BIN)/python
    VENV_PIP := $(VENV_BIN)/pip
    VENV_PYTEST := $(VENV_BIN)/pytest
endif

# Check Python version and availability
define check_python
	@if [ "$(PYTHON)" = "python-not-found" ]; then \
		echo "$(RED)❌ Python not found!$(NC)"; \
		echo "$(YELLOW)💡 Please install Python 3.9+ from https://python.org$(NC)"; \
		echo "$(YELLOW)   On macOS: brew install python$(NC)"; \
		echo "$(YELLOW)   On Ubuntu: sudo apt install python3$(NC)"; \
		exit 1; \
	fi
	@PYTHON_VERSION=$$($(PYTHON) --version 2>&1 | sed 's/Python //'); \
	PYTHON_MAJOR=$$(echo $$PYTHON_VERSION | cut -d. -f1); \
	PYTHON_MINOR=$$(echo $$PYTHON_VERSION | cut -d. -f2); \
	if [ $$PYTHON_MAJOR -lt 3 ] || [ $$PYTHON_MAJOR -eq 3 -a $$PYTHON_MINOR -lt 9 ]; then \
		echo "$(RED)❌ Python 3.9+ required, found $$PYTHON_VERSION$(NC)"; \
		echo "$(YELLOW)💡 Please upgrade Python to 3.9 or higher$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Found Python $$($(PYTHON) --version 2>&1 | sed 's/Python //')$(NC)"
endef

# Check build dependencies (especially for macOS)
define check_build_deps
	@echo "$(BLUE)🔍 Checking build dependencies...$(NC)"
	@if command -v pkg-config >/dev/null 2>&1; then \
		echo "$(GREEN)✅ pkg-config found$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  pkg-config not found$(NC)"; \
		if [[ "$$(uname)" == "Darwin" ]]; then \
			echo "$(YELLOW)💡 Installing pkg-config via Homebrew...$(NC)"; \
			if command -v brew >/dev/null 2>&1; then \
				brew install pkg-config || echo "$(RED)❌ Failed to install pkg-config$(NC)"; \
			else \
				echo "$(RED)❌ Homebrew not found. Please install:$(NC)"; \
				echo "$(YELLOW)   /bin/bash -c \"\$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"$(NC)"; \
				echo "$(YELLOW)   Then run: brew install pkg-config$(NC)"; \
			fi; \
		else \
			echo "$(YELLOW)💡 Please install build dependencies:$(NC)"; \
			echo "$(YELLOW)   Ubuntu/Debian: sudo apt install build-essential pkg-config$(NC)"; \
			echo "$(YELLOW)   CentOS/RHEL: sudo yum install gcc pkg-config$(NC)"; \
		fi; \
	fi
	@if [[ "$$(uname)" == "Darwin" ]] && ! xcode-select -p >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠️  Xcode Command Line Tools not found$(NC)"; \
		echo "$(YELLOW)💡 Installing Xcode Command Line Tools...$(NC)"; \
		xcode-select --install || echo "$(RED)❌ Please install Xcode Command Line Tools manually$(NC)"; \
	fi
endef

# Check if virtual environment is activated
define check_venv
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(RED)❌ Virtual environment not activated!$(NC)"; \
		echo "$(YELLOW)💡 Run 'source venv/bin/activate' or 'make setup-venv'$(NC)"; \
		exit 1; \
	fi
endef

help: ## Show this help message
	@echo "$(CYAN)TradeBuddy Development Commands$(NC)"
	@echo "$(CYAN)==============================$(NC)"
	@echo ""
	@echo "$(WHITE)Setup Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(setup|install|clean)"
	@echo ""
	@echo "$(WHITE)Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST) | grep -E "(test|lint|format|run|check)"
	@echo ""
	@echo "$(WHITE)Usage:$(NC)"
	@echo "  1. $(YELLOW)make check-python$(NC)  # Check Python installation"
	@echo "  2. $(YELLOW)make setup-env$(NC)     # First time setup"
	@echo "  3. $(YELLOW)source $(VENV_BIN)/activate$(NC)  # Activate virtual environment"
	@echo "  4. $(YELLOW)make test$(NC)          # Run tests"
	@echo "  5. $(YELLOW)make run$(NC)           # Run the application"
	@echo ""
	@echo "$(WHITE)Troubleshooting:$(NC)"
	@echo "  • $(YELLOW)make check-env$(NC)      # Diagnose environment issues"
	@echo "  • $(YELLOW)make fix-build-deps$(NC) # Fix macOS build dependencies"
	@echo "  • $(YELLOW)make clean-all$(NC)      # Clean and restart setup"

setup-env: ## Set up complete development environment
	@echo "$(BLUE)🔧 Setting up TradeBuddy development environment...$(NC)"
	$(call check_python)
	$(call check_build_deps)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)📦 Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
		if [ ! -d "$(VENV_BIN)" ]; then \
			echo "$(RED)❌ Virtual environment creation failed$(NC)"; \
			echo "$(YELLOW)💡 Please ensure Python has venv module installed$(NC)"; \
			exit 1; \
		fi; \
		echo "$(GREEN)✅ Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)📦 Virtual environment already exists$(NC)"; \
	fi
	@echo "$(YELLOW)⬆️  Upgrading pip...$(NC)"
	@if [ -f "$(VENV_PIP)" ]; then \
		$(VENV_PIP) install --upgrade pip setuptools wheel; \
	else \
		echo "$(RED)❌ Virtual environment pip not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)📥 Installing development dependencies...$(NC)"
	@echo "$(BLUE)  Note: Using compatible versions with pre-built wheels$(NC)"
	@$(VENV_PIP) install -r requirements/dev.txt --prefer-binary
	@if [ ! -f ".env" ]; then \
		echo "$(YELLOW)🔧 Creating .env file from template...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✅ .env file created$(NC)"; \
	else \
		echo "$(YELLOW)🔧 .env file already exists$(NC)"; \
	fi
	@echo "$(GREEN)🎉 Environment setup complete!$(NC)"
	@echo "$(CYAN)💡 Next steps:$(NC)"
	@echo "  1. $(YELLOW)source $(VENV_BIN)/activate$(NC)"
	@echo "  2. $(YELLOW)make test$(NC)"
	@echo "  3. $(YELLOW)make run$(NC)"

setup-venv: ## Create and activate virtual environment
	$(call check_python)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)📦 Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
		if [ ! -d "$(VENV_BIN)" ]; then \
			echo "$(RED)❌ Virtual environment creation failed$(NC)"; \
			exit 1; \
		fi; \
		echo "$(GREEN)✅ Virtual environment created$(NC)"; \
	fi
	@echo "$(CYAN)💡 To activate: source $(VENV_BIN)/activate$(NC)"

install: ## Install all dependencies
	$(call check_venv)
	@echo "$(YELLOW)📥 Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements/dev.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

install-prod: ## Install production dependencies only
	$(call check_venv)
	@echo "$(YELLOW)📥 Installing production dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements/prod.txt
	@echo "$(GREEN)✅ Production dependencies installed$(NC)"

test: ## Run test suite
	$(call check_venv)
	@echo "$(YELLOW)🧪 Running tests...$(NC)"
	@$(PYTEST) tests/ -v --tb=short
	@echo "$(GREEN)✅ Tests completed$(NC)"

test-cov: ## Run tests with coverage report
	$(call check_venv)
	@echo "$(YELLOW)🧪 Running tests with coverage...$(NC)"
	@$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=80
	@echo "$(GREEN)✅ Coverage report generated$(NC)"
	@echo "$(CYAN)💡 Open htmlcov/index.html to view detailed coverage$(NC)"

test-unit: ## Run unit tests only
	$(call check_venv)
	@echo "$(YELLOW)🧪 Running unit tests...$(NC)"
	@$(PYTEST) tests/unit/ -v
	@echo "$(GREEN)✅ Unit tests completed$(NC)"

test-integration: ## Run integration tests only
	$(call check_venv)
	@echo "$(YELLOW)🧪 Running integration tests...$(NC)"
	@$(PYTEST) tests/integration/ -v
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

test-watch: ## Run tests in watch mode
	$(call check_venv)
	@echo "$(YELLOW)👀 Running tests in watch mode...$(NC)"
	@$(PYTEST) tests/ -v --looponfail

lint: ## Run linting checks
	$(call check_venv)
	@echo "$(YELLOW)🔍 Running linting checks...$(NC)"
	@echo "$(BLUE)  Running flake8...$(NC)"
	@flake8 src/ tests/ || echo "$(RED)❌ flake8 found issues$(NC)"
	@echo "$(BLUE)  Running mypy...$(NC)"
	@mypy src/ || echo "$(RED)❌ mypy found issues$(NC)"
	@echo "$(BLUE)  Running bandit...$(NC)"
	@bandit -r src/ || echo "$(RED)❌ bandit found issues$(NC)"
	@echo "$(GREEN)✅ Linting completed$(NC)"

format: ## Format code with black and isort
	$(call check_venv)
	@echo "$(YELLOW)🎨 Formatting code...$(NC)"
	@echo "$(BLUE)  Running black...$(NC)"
	@black src/ tests/
	@echo "$(BLUE)  Running isort...$(NC)"
	@isort src/ tests/
	@echo "$(GREEN)✅ Code formatted$(NC)"

format-check: ## Check code formatting without making changes
	$(call check_venv)
	@echo "$(YELLOW)🔍 Checking code formatting...$(NC)"
	@black --check src/ tests/
	@isort --check-only src/ tests/
	@echo "$(GREEN)✅ Code formatting is correct$(NC)"

run: ## Run the TradeBuddy application
	$(call check_venv)
	@echo "$(YELLOW)🚀 Starting TradeBuddy...$(NC)"
	@$(PYTHON) tradebuddy.py

run-dev: ## Run the application in development mode
	$(call check_venv)
	@echo "$(YELLOW)🚀 Starting TradeBuddy in development mode...$(NC)"
	@PYTHON_ENV=development DEBUG=true $(PYTHON) tradebuddy.py

run-demo: ## Run the TradeBuddy system demonstration
	$(call check_venv)
	@echo "$(YELLOW)🎬 Starting TradeBuddy demo...$(NC)"
	@$(PYTHON) demo_complete_system.py

check-python: ## Check Python installation and version
	@echo "$(YELLOW)🔍 Checking Python installation...$(NC)"
	$(call check_python)

fix-build-deps: ## Install build dependencies for macOS
	@echo "$(BLUE)🔧 Installing build dependencies...$(NC)"
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		echo "$(YELLOW)Installing macOS build dependencies...$(NC)"; \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "$(YELLOW)Installing Homebrew...$(NC)"; \
			/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		fi; \
		echo "$(YELLOW)Installing pkg-config and build tools...$(NC)"; \
		brew install pkg-config; \
		if ! xcode-select -p >/dev/null 2>&1; then \
			echo "$(YELLOW)Installing Xcode Command Line Tools...$(NC)"; \
			xcode-select --install; \
		fi; \
	else \
		echo "$(YELLOW)Please install build dependencies for your system:$(NC)"; \
		echo "$(YELLOW)  Ubuntu/Debian: sudo apt install build-essential pkg-config$(NC)"; \
		echo "$(YELLOW)  CentOS/RHEL: sudo yum install gcc pkg-config$(NC)"; \
	fi

check-env: ## Check environment and dependencies
	@echo "$(YELLOW)🔍 Checking environment...$(NC)"
	@if [ "$(PYTHON)" != "python-not-found" ]; then \
		echo "$(BLUE)Python executable:$(NC) $(PYTHON)"; \
		echo "$(BLUE)Python version:$(NC) $$($(PYTHON) --version 2>&1)"; \
	else \
		echo "$(BLUE)Python executable:$(NC) $(RED)❌ Not found$(NC)"; \
	fi
	@echo "$(BLUE)Virtual environment:$(NC) $${VIRTUAL_ENV:-Not activated}"
	@echo "$(BLUE)Current directory:$(NC) $$(pwd)"
	@if [ -f ".env" ]; then \
		echo "$(BLUE).env file:$(NC) ✅ Found"; \
	else \
		echo "$(BLUE).env file:$(NC) ❌ Not found"; \
	fi
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "$(BLUE)Virtual environment:$(NC) ✅ Found at $(VENV_DIR)"; \
		if [ -f "$(VENV_PYTHON)" ]; then \
			echo "$(BLUE)Venv Python:$(NC) ✅ $$($(VENV_PYTHON) --version 2>&1)"; \
		else \
			echo "$(BLUE)Venv Python:$(NC) ❌ Not found"; \
		fi; \
	else \
		echo "$(BLUE)Virtual environment:$(NC) ❌ Not found"; \
	fi

clean: ## Clean up temporary files and caches
	@echo "$(YELLOW)🧹 Cleaning up...$(NC)"
	@echo "$(BLUE)  Removing Python cache files...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(BLUE)  Removing test artifacts...$(NC)"
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf .mypy_cache/
	@echo "$(BLUE)  Removing build artifacts...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(YELLOW)🧹 Deep cleaning...$(NC)"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "$(BLUE)  Removing virtual environment...$(NC)"; \
		rm -rf $(VENV_DIR); \
		echo "$(GREEN)✅ Virtual environment removed$(NC)"; \
	fi

deps-check: ## Check for dependency updates
	$(call check_venv)
	@echo "$(YELLOW)🔍 Checking for dependency updates...$(NC)"
	@$(PIP) list --outdated

deps-update: ## Update all dependencies
	$(call check_venv)
	@echo "$(YELLOW)⬆️  Updating dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements/dev.txt --upgrade
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

pre-commit: ## Run pre-commit checks
	$(call check_venv)
	@echo "$(YELLOW)🔍 Running pre-commit checks...$(NC)"
	@make format-check
	@make lint
	@make test-unit
	@echo "$(GREEN)✅ Pre-commit checks passed$(NC)"

build: ## Build the package
	$(call check_venv)
	@echo "$(YELLOW)📦 Building package...$(NC)"
	@$(PYTHON) -m build
	@echo "$(GREEN)✅ Package built$(NC)"

# Docker commands
docker-build: ## Build TradeBuddy Docker image
	@echo "$(YELLOW)🐳 Building TradeBuddy Docker image...$(NC)"
	@docker build -t tradebuddy:latest .
	@echo "$(GREEN)✅ Docker image built$(NC)"

docker-up: ## Start TradeBuddy with Ollama (production)
	@echo "$(YELLOW)🐳 Starting TradeBuddy services...$(NC)"
	@docker-compose up -d ollama
	@echo "$(BLUE)⏳ Waiting for Ollama to start...$(NC)"
	@sleep 10
	@docker-compose up tradebuddy
	@echo "$(GREEN)✅ TradeBuddy started$(NC)"

docker-dev: ## Start TradeBuddy in development mode
	@echo "$(YELLOW)🐳 Starting TradeBuddy development environment...$(NC)"
	@docker-compose --profile dev up

docker-fingpt: ## Start TradeBuddy with FinGPT integration
	@echo "$(YELLOW)🐳 Starting TradeBuddy with FinGPT...$(NC)"
	@docker-compose --profile fingpt up

docker-down: ## Stop all Docker services
	@echo "$(YELLOW)🐳 Stopping TradeBuddy services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped$(NC)"

docker-logs: ## View TradeBuddy logs
	@echo "$(YELLOW)📋 TradeBuddy logs:$(NC)"
	@docker-compose logs -f tradebuddy

docker-clean: ## Clean Docker images and volumes
	@echo "$(YELLOW)🧹 Cleaning Docker resources...$(NC)"
	@docker-compose down -v
	@docker system prune -f
	@echo "$(GREEN)✅ Docker cleanup completed$(NC)"

# Development shortcuts
dev: setup-env ## Complete development setup
	@echo "$(GREEN)🎉 Development environment ready!$(NC)"

quick-test: format lint test-unit ## Quick development test cycle

full-test: format lint test-cov ## Full test cycle with coverage

ci: format-check lint test-cov ## CI/CD pipeline simulation