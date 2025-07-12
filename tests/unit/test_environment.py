"""
Unit tests for environment validation.

Tests environment validation utilities and system checks.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.core.exceptions import EnvironmentError
from src.utils.environment import (
    EnvironmentValidator,
    ValidationResult,
    print_validation_result,
    validate_environment,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result_creation(self):
        """Test creating valid ValidationResult."""
        result = ValidationResult(
            is_valid=True, warnings=["Warning 1", "Warning 2"], details={"key": "value"}
        )

        assert result.is_valid is True
        assert len(result.warnings) == 2
        assert result.details["key"] == "value"
        assert result.error is None

    def test_invalid_result_creation(self):
        """Test creating invalid ValidationResult."""
        result = ValidationResult(
            is_valid=False, error="Validation failed", warnings=["Warning 1"]
        )

        assert result.is_valid is False
        assert result.error == "Validation failed"
        assert len(result.warnings) == 1

    def test_result_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult(is_valid=True)

        assert result.warnings == []
        assert result.details == {}
        assert result.error is None


class TestEnvironmentValidator:
    """Test EnvironmentValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = EnvironmentValidator()

        assert validator.warnings == []
        assert validator.details == {}

    @pytest.mark.asyncio
    async def test_successful_validation(self):
        """Test successful environment validation."""
        validator = EnvironmentValidator()

        with patch.object(validator, "_check_python_version"), patch.object(
            validator, "_check_required_packages"
        ), patch.object(validator, "_check_environment_variables"), patch.object(
            validator, "_check_external_services"
        ), patch.object(
            validator, "_check_system_resources"
        ):
            result = await validator.validate_environment()

            assert result.is_valid is True
            assert result.error is None

    @pytest.mark.asyncio
    async def test_validation_with_error(self):
        """Test validation with error."""
        validator = EnvironmentValidator()

        with patch.object(
            validator,
            "_check_python_version",
            side_effect=EnvironmentError("Python version error"),
        ):
            result = await validator.validate_environment()

            assert result.is_valid is False
            assert "Python version error" in result.error

    def test_check_python_version_valid(self):
        """Test Python version check with valid version."""
        validator = EnvironmentValidator()

        # Should not raise exception for current Python version
        validator._check_python_version()

        assert "python_version" in validator.details
        assert validator.details["python_version"]["valid"] is True

    def test_check_python_version_invalid(self):
        """Test Python version check with invalid version."""
        validator = EnvironmentValidator()

        with patch("sys.version_info", (3, 8)):  # Below minimum requirement
            with pytest.raises(EnvironmentError) as exc_info:
                validator._check_python_version()

            assert "Python 3.9+ required" in str(exc_info.value)

    def test_check_required_packages_success(self):
        """Test required packages check with all packages available."""
        validator = EnvironmentValidator()

        # Mock all required packages as available
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module

            validator._check_required_packages()

            assert "packages" in validator.details
            assert len(validator.details["packages"]["missing"]) == 0

    def test_check_required_packages_missing(self):
        """Test required packages check with missing packages."""
        validator = EnvironmentValidator()

        # Mock some packages as missing
        def mock_import(package):
            if package in ["aiohttp", "pydantic"]:
                return MagicMock()
            raise ImportError(f"No module named '{package}'")

        with patch("importlib.import_module", side_effect=mock_import):
            with pytest.raises(EnvironmentError) as exc_info:
                validator._check_required_packages()

            assert "Missing required packages" in str(exc_info.value)

    def test_check_environment_variables_success(self, test_settings):
        """Test environment variables check with all variables set."""
        validator = EnvironmentValidator()

        with patch("src.utils.environment.settings", test_settings):
            validator._check_environment_variables()

            assert "environment_variables" in validator.details
            # Should not raise exception

    def test_check_environment_variables_missing_critical(self):
        """Test environment variables check with missing critical variables."""
        validator = EnvironmentValidator()

        # Mock settings with missing critical variables
        mock_settings = MagicMock()
        mock_settings.delta_exchange_api_url = None
        mock_settings.ollama_api_url = None
        mock_settings.ollama_model = None

        with patch("src.utils.environment.settings", mock_settings):
            with pytest.raises(EnvironmentError) as exc_info:
                validator._check_environment_variables()

            assert "Missing critical environment variables" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_external_services_ollama_success(self, test_settings):
        """Test external services check with Ollama available."""
        validator = EnvironmentValidator()

        # Mock successful Ollama response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "models": [{"name": "test-model"}, {"name": "qwen2.5:14b"}]
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            with patch("src.utils.environment.settings", test_settings):
                await validator._check_external_services()

            assert "external_services" in validator.details
            assert (
                validator.details["external_services"]["ollama"]["status"]
                == "✅ Connected"
            )

    @pytest.mark.asyncio
    async def test_check_external_services_ollama_timeout(self, test_settings):
        """Test external services check with Ollama timeout."""
        validator = EnvironmentValidator()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = (
                asyncio.TimeoutError()
            )

            with patch("src.utils.environment.settings", test_settings):
                await validator._check_external_services()

            assert "external_services" in validator.details
            assert (
                "Timeout" in validator.details["external_services"]["ollama"]["status"]
            )
            assert any("timeout" in warning.lower() for warning in validator.warnings)

    @pytest.mark.asyncio
    async def test_check_external_services_delta_exchange_success(self, test_settings):
        """Test external services check with Delta Exchange available."""
        validator = EnvironmentValidator()

        # Mock successful Delta Exchange response
        mock_response = AsyncMock()
        mock_response.status = 200

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            with patch("src.utils.environment.settings", test_settings):
                await validator._check_external_services()

            assert "external_services" in validator.details
            assert (
                validator.details["external_services"]["delta_exchange"]["status"]
                == "✅ Connected"
            )

    def test_check_system_resources_with_psutil(self):
        """Test system resources check with psutil available."""
        validator = EnvironmentValidator()

        # Mock psutil
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(
            total=32 * (1024**3),  # 32 GB
            available=16 * (1024**3),  # 16 GB available
            percent=50.0,
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            total=1000 * (1024**3),  # 1 TB
            free=500 * (1024**3),  # 500 GB free
            used=500 * (1024**3),  # 500 GB used
        )

        with patch("src.utils.environment.psutil", mock_psutil):
            validator._check_system_resources()

            assert "system_resources" in validator.details
            assert validator.details["system_resources"]["memory"]["sufficient"] is True

    def test_check_system_resources_insufficient_memory(self):
        """Test system resources check with insufficient memory."""
        validator = EnvironmentValidator()

        # Mock psutil with low memory
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = MagicMock(
            total=8 * (1024**3),  # 8 GB (below recommended)
            available=4 * (1024**3),  # 4 GB available
            percent=50.0,
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            total=1000 * (1024**3), free=500 * (1024**3), used=500 * (1024**3)
        )

        with patch("src.utils.environment.psutil", mock_psutil):
            validator._check_system_resources()

            assert "system_resources" in validator.details
            assert (
                validator.details["system_resources"]["memory"]["sufficient"] is False
            )
            assert any("RAM" in warning for warning in validator.warnings)

    def test_check_system_resources_without_psutil(self):
        """Test system resources check without psutil."""
        validator = EnvironmentValidator()

        with patch("src.utils.environment.psutil", None):
            with patch("importlib.import_module", side_effect=ImportError):
                validator._check_system_resources()

                assert "system_resources" in validator.details
                assert (
                    "psutil not available"
                    in validator.details["system_resources"]["memory"]["status"]
                )
                assert any(
                    "psutil not installed" in warning for warning in validator.warnings
                )


class TestValidateEnvironmentFunction:
    """Test validate_environment function."""

    @pytest.mark.asyncio
    async def test_validate_environment_success(self):
        """Test successful environment validation."""
        with patch.object(
            EnvironmentValidator, "validate_environment"
        ) as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)

            result = await validate_environment()

            assert result.is_valid is True
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_environment_failure(self):
        """Test failed environment validation."""
        with patch.object(
            EnvironmentValidator, "validate_environment"
        ) as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=False, error="Validation failed"
            )

            result = await validate_environment()

            assert result.is_valid is False
            assert result.error == "Validation failed"


class TestPrintValidationResult:
    """Test print_validation_result function."""

    def test_print_successful_result(self, capsys):
        """Test printing successful validation result."""
        result = ValidationResult(
            is_valid=True, warnings=["Warning 1"], details={"test": "data"}
        )

        with patch("rich.console.Console") as mock_console:
            print_validation_result(result)

            # Should call console methods
            mock_console.return_value.print.assert_called()

    def test_print_failed_result(self, capsys):
        """Test printing failed validation result."""
        result = ValidationResult(
            is_valid=False,
            error="Validation failed",
            warnings=["Warning 1"],
            details={"test": "data"},
        )

        with patch("rich.console.Console") as mock_console:
            print_validation_result(result)

            # Should call console methods
            mock_console.return_value.print.assert_called()

    def test_print_result_with_complex_details(self):
        """Test printing result with complex details structure."""
        result = ValidationResult(
            is_valid=True,
            details={
                "python_version": {"current": "3.11.0", "valid": True},
                "packages": {"installed": {"aiohttp": "3.9.1"}, "missing": []},
                "external_services": {
                    "ollama": {"status": "Connected", "models": ["test-model"]}
                },
            },
        )

        with patch("rich.console.Console") as mock_console:
            print_validation_result(result)

            # Should handle nested dictionaries
            mock_console.return_value.print.assert_called()


class TestEnvironmentValidatorIntegration:
    """Integration tests for environment validator."""

    @pytest.mark.asyncio
    async def test_full_validation_with_mocked_dependencies(self, test_settings):
        """Test full validation with all dependencies mocked."""
        validator = EnvironmentValidator()

        # Mock all external dependencies
        with patch("sys.version_info", (3, 11, 0)), patch(
            "importlib.import_module"
        ) as mock_import, patch("aiohttp.ClientSession") as mock_session, patch(
            "src.utils.environment.settings", test_settings
        ):
            # Mock successful package imports
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module

            # Mock successful API responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"models": [{"name": "test-model"}]}
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            result = await validator.validate_environment()

            assert result.is_valid is True
            assert "python_version" in result.details
            assert "packages" in result.details
            assert "environment_variables" in result.details
            assert "external_services" in result.details

    @pytest.mark.asyncio
    async def test_validation_with_warnings_but_valid(self, test_settings):
        """Test validation that passes but has warnings."""
        validator = EnvironmentValidator()

        with patch("sys.version_info", (3, 11, 0)), patch(
            "importlib.import_module"
        ) as mock_import, patch("aiohttp.ClientSession") as mock_session, patch(
            "src.utils.environment.settings", test_settings
        ):
            # Mock successful package imports
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module

            # Mock API timeout for warnings
            mock_session.return_value.__aenter__.return_value.get.side_effect = (
                asyncio.TimeoutError()
            )

            result = await validator.validate_environment()

            assert result.is_valid is True
            assert len(result.warnings) > 0
            assert any("timeout" in warning.lower() for warning in result.warnings)


class TestEnvironmentValidatorEdgeCases:
    """Test edge cases for environment validator."""

    def test_validator_with_special_package_names(self):
        """Test validator with special package name mappings."""
        validator = EnvironmentValidator()

        def mock_import(package):
            if package == "yaml":
                import yaml

                return yaml
            elif package == "dotenv":
                import dotenv

                return dotenv
            else:
                return MagicMock(__version__="1.0.0")

        with patch("importlib.import_module", side_effect=mock_import):
            validator._check_required_packages()

            assert "packages" in validator.details
            assert "yaml" in validator.details["packages"]["installed"]
            assert "dotenv" in validator.details["packages"]["installed"]

    @pytest.mark.asyncio
    async def test_validator_with_http_error_responses(self, test_settings):
        """Test validator with HTTP error responses."""
        validator = EnvironmentValidator()

        # Mock HTTP 500 error
        mock_response = AsyncMock()
        mock_response.status = 500

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            with patch("src.utils.environment.settings", test_settings):
                await validator._check_external_services()

            assert "external_services" in validator.details
            assert (
                "HTTP 500" in validator.details["external_services"]["ollama"]["status"]
            )

    def test_validator_handles_missing_model_in_response(self, test_settings):
        """Test validator handles missing required model in Ollama response."""
        validator = EnvironmentValidator()

        # Mock response with models but not the required one
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "models": [{"name": "other-model"}, {"name": "another-model"}]
        }

        async def run_test():
            with patch("aiohttp.ClientSession") as mock_session:
                mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                    mock_response
                )

                with patch("src.utils.environment.settings", test_settings):
                    await validator._check_external_services()

                assert "external_services" in validator.details
                assert not validator.details["external_services"]["ollama"][
                    "has_required_model"
                ]
                assert any(
                    "Required Ollama model" in warning for warning in validator.warnings
                )

        asyncio.run(run_test())
