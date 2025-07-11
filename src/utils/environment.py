"""
Environment validation utilities for TradeBuddy.

Validates system requirements, dependencies, and external services.
"""

import sys
import importlib
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import aiohttp
from src.core.config import settings
from src.core.exceptions import EnvironmentError


@dataclass
class ValidationResult:
    """Result of environment validation."""
    is_valid: bool
    error: Optional[str] = None
    warnings: List[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}


class EnvironmentValidator:
    """Validates the application environment."""
    
    def __init__(self):
        self.warnings: List[str] = []
        self.details: Dict[str, Any] = {}
    
    async def validate_environment(self) -> ValidationResult:
        """Perform comprehensive environment validation."""
        try:
            # Check Python version
            self._check_python_version()
            
            # Check required packages
            self._check_required_packages()
            
            # Check environment variables
            self._check_environment_variables()
            
            # Check external services
            await self._check_external_services()
            
            # Check system resources
            self._check_system_resources()
            
            return ValidationResult(
                is_valid=True,
                warnings=self.warnings,
                details=self.details
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error=str(e),
                warnings=self.warnings,
                details=self.details
            )
    
    def _check_python_version(self):
        """Check Python version requirements."""
        min_version = (3, 9)
        current_version = sys.version_info[:2]
        
        self.details["python_version"] = {
            "current": f"{current_version[0]}.{current_version[1]}",
            "required": f"{min_version[0]}.{min_version[1]}+",
            "valid": current_version >= min_version
        }
        
        if current_version < min_version:
            raise EnvironmentError(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"but {current_version[0]}.{current_version[1]} found"
            )
    
    def _check_required_packages(self):
        """Check if all required packages are installed."""
        required_packages = [
            "aiohttp",
            "pydantic", 
            "pandas",
            "numpy",
            "websockets",
            "structlog",
            "yaml",
            "click",
            "rich",
            "prompt_toolkit",
            "colorama",
            "tabulate",
            "dotenv"
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                # Try to import the package
                if package == "yaml":
                    import yaml
                    installed_packages[package] = getattr(yaml, "__version__", "unknown")
                elif package == "dotenv":
                    import dotenv
                    installed_packages[package] = getattr(dotenv, "__version__", "unknown")
                else:
                    module = importlib.import_module(package)
                    installed_packages[package] = getattr(module, "__version__", "unknown")
            except ImportError:
                missing_packages.append(package)
        
        self.details["packages"] = {
            "installed": installed_packages,
            "missing": missing_packages
        }
        
        if missing_packages:
            raise EnvironmentError(
                f"Missing required packages: {', '.join(missing_packages)}. "
                f"Run 'pip install -r requirements/dev.txt' to install."
            )
    
    def _check_environment_variables(self):
        """Check critical environment variables."""
        critical_vars = [
            "DELTA_EXCHANGE_API_URL",
            "OLLAMA_API_URL",
            "OLLAMA_MODEL"
        ]
        
        optional_vars = [
            "REDIS_URL",
            "DATABASE_URL",
            "LOG_LEVEL",
            "PYTHON_ENV"
        ]
        
        env_status = {}
        missing_critical = []
        
        # Check critical variables
        for var in critical_vars:
            value = getattr(settings, var.lower(), None)
            if value:
                env_status[var] = "✅ Set"
            else:
                env_status[var] = "❌ Missing"
                missing_critical.append(var)
        
        # Check optional variables
        for var in optional_vars:
            value = getattr(settings, var.lower(), None)
            if value:
                env_status[var] = "✅ Set"
            else:
                env_status[var] = "⚠️ Default"
        
        self.details["environment_variables"] = env_status
        
        if missing_critical:
            raise EnvironmentError(
                f"Missing critical environment variables: {', '.join(missing_critical)}. "
                f"Check your .env file."
            )
    
    async def _check_external_services(self):
        """Check connectivity to external services."""
        services_status = {}
        
        # Check Ollama API
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{settings.ollama_api_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        services_status["ollama"] = {
                            "status": "✅ Connected",
                            "models": models,
                            "has_required_model": settings.ollama_model in models
                        }
                        
                        if settings.ollama_model not in models:
                            self.warnings.append(
                                f"Required Ollama model '{settings.ollama_model}' not found. "
                                f"Available models: {', '.join(models)}"
                            )
                    else:
                        services_status["ollama"] = {
                            "status": f"❌ HTTP {response.status}",
                            "models": [],
                            "has_required_model": False
                        }
                        self.warnings.append(
                            f"Ollama API returned status {response.status}"
                        )
        except asyncio.TimeoutError:
            services_status["ollama"] = {
                "status": "⚠️ Timeout",
                "models": [],
                "has_required_model": False
            }
            self.warnings.append(
                "Ollama API connection timeout. Make sure Ollama is running."
            )
        except Exception as e:
            services_status["ollama"] = {
                "status": f"❌ Error: {str(e)}",
                "models": [],
                "has_required_model": False
            }
            self.warnings.append(
                f"Cannot connect to Ollama API: {str(e)}"
            )
        
        # Check Delta Exchange API (basic connectivity)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{settings.delta_exchange_api_url}/v2/products") as response:
                    if response.status == 200:
                        services_status["delta_exchange"] = {
                            "status": "✅ Connected",
                            "api_url": settings.delta_exchange_api_url
                        }
                    else:
                        services_status["delta_exchange"] = {
                            "status": f"⚠️ HTTP {response.status}",
                            "api_url": settings.delta_exchange_api_url
                        }
                        self.warnings.append(
                            f"Delta Exchange API returned status {response.status}"
                        )
        except Exception as e:
            services_status["delta_exchange"] = {
                "status": f"⚠️ Error: {str(e)}",
                "api_url": settings.delta_exchange_api_url
            }
            self.warnings.append(
                f"Cannot connect to Delta Exchange API: {str(e)}"
            )
        
        self.details["external_services"] = services_status
    
    def _check_system_resources(self):
        """Check system resources."""
        try:
            import psutil
            
            # Get system information
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Memory requirements (16GB recommended for Ollama)
            memory_gb = memory.total / (1024**3)
            recommended_memory_gb = 16
            
            resources = {
                "memory": {
                    "total_gb": round(memory_gb, 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "usage_percent": memory.percent,
                    "sufficient": memory_gb >= recommended_memory_gb
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "usage_percent": round((disk.used / disk.total) * 100, 2)
                }
            }
            
            if memory_gb < recommended_memory_gb:
                self.warnings.append(
                    f"System has {memory_gb:.1f}GB RAM. "
                    f"Recommended: {recommended_memory_gb}GB+ for optimal Ollama performance."
                )
            
            if disk.free < 10 * (1024**3):  # Less than 10GB free
                self.warnings.append(
                    f"Low disk space: {disk.free / (1024**3):.1f}GB free. "
                    f"Recommend at least 10GB free space."
                )
                
        except ImportError:
            resources = {
                "memory": {"status": "⚠️ psutil not available for system monitoring"},
                "disk": {"status": "⚠️ psutil not available for system monitoring"}
            }
            self.warnings.append(
                "psutil not installed. System resource monitoring unavailable."
            )
        except Exception as e:
            resources = {
                "memory": {"status": f"❌ Error: {str(e)}"},
                "disk": {"status": f"❌ Error: {str(e)}"}
            }
        
        self.details["system_resources"] = resources


# Global validator instance
validator = EnvironmentValidator()


async def validate_environment() -> ValidationResult:
    """Validate the current environment setup."""
    return await validator.validate_environment()


def print_validation_result(result: ValidationResult):
    """Print validation result in a formatted way."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
    
    console = Console()
    
    if result.is_valid:
        console.print(Panel(
            "✅ Environment validation passed!",
            title="Environment Status",
            style="green"
        ))
    else:
        console.print(Panel(
            f"❌ Environment validation failed: {result.error}",
            title="Environment Status",
            style="red"
        ))
    
    # Show warnings if any
    if result.warnings:
        console.print("\n⚠️ Warnings:")
        for warning in result.warnings:
            console.print(f"  • {warning}", style="yellow")
    
    # Show details if available
    if result.details:
        console.print("\n📋 Environment Details:")
        tree = Tree("Environment")
        
        for category, data in result.details.items():
            category_tree = tree.add(f"[bold]{category.replace('_', ' ').title()}[/bold]")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        sub_tree = category_tree.add(f"[cyan]{key}[/cyan]")
                        for sub_key, sub_value in value.items():
                            sub_tree.add(f"{sub_key}: {sub_value}")
                    else:
                        category_tree.add(f"{key}: {value}")
            else:
                category_tree.add(str(data))
        
        console.print(tree)