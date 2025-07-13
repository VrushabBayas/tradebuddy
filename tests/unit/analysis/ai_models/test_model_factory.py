"""
Test model factory pattern implementation.

Tests factory pattern for creating AI model instances.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.analysis.ai_models.model_factory import ModelFactory
from src.analysis.ai_models.ai_interface import AIModelInterface
from src.core.models import AIModelType, AIModelConfig


class TestModelFactory:
    """Test model factory functionality."""

    def test_factory_creates_ollama_model(self):
        """Test factory creates Ollama model correctly."""
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockOllama.return_value = mock_instance
            
            model = ModelFactory.create_model(AIModelType.OLLAMA)
            
            assert model is not None
            assert model == mock_instance
            MockOllama.assert_called_once()

    def test_factory_creates_fingpt_model(self):
        """Test factory creates FinGPT model correctly."""
        with patch('src.analysis.ai_models.model_factory.FinGPTClient') as MockFinGPT:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockFinGPT.return_value = mock_instance
            
            model = ModelFactory.create_model(AIModelType.FINGPT)
            
            assert model is not None
            assert model == mock_instance
            MockFinGPT.assert_called_once()

    def test_factory_with_config(self):
        """Test factory uses configuration properly."""
        config = AIModelConfig(
            model_type=AIModelType.FINGPT,
            fingpt_model_variant="v3.3"
        )
        
        with patch('src.analysis.ai_models.model_factory.FinGPTClient') as MockFinGPT:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockFinGPT.return_value = mock_instance
            
            model = ModelFactory.create_model(AIModelType.FINGPT, config=config)
            
            assert model is not None
            MockFinGPT.assert_called_once_with(model_variant="v3.3")

    def test_factory_with_kwargs(self):
        """Test factory passes kwargs to model constructors."""
        custom_timeout = 60
        custom_url = "http://custom-ollama:11434"
        
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockOllama.return_value = mock_instance
            
            model = ModelFactory.create_model(
                AIModelType.OLLAMA,
                timeout=custom_timeout,
                base_url=custom_url
            )
            
            assert model is not None
            MockOllama.assert_called_once_with(timeout=custom_timeout, base_url=custom_url)

    def test_factory_invalid_model_type_fails_fast(self):
        """Test factory fails fast with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported AI model type"):
            ModelFactory.create_model("invalid_type")

    def test_factory_handles_import_errors(self):
        """Test factory handles missing dependencies gracefully."""
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Ollama dependencies not available"):
                ModelFactory.create_model(AIModelType.OLLAMA)

    def test_get_available_models(self):
        """Test getting list of available model types."""
        available_models = ModelFactory.get_available_models()
        
        assert isinstance(available_models, list)
        assert "ollama" in available_models
        assert "fingpt" in available_models
        assert len(available_models) == len(AIModelType)

    def test_is_model_available_ollama(self):
        """Test checking if Ollama model is available."""
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockOllama.return_value = mock_instance
            
            is_available = ModelFactory.is_model_available(AIModelType.OLLAMA)
            assert is_available is True

        # Test with import error
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper', side_effect=ImportError()):
            is_available = ModelFactory.is_model_available(AIModelType.OLLAMA)
            assert is_available is False

    def test_is_model_available_fingpt(self):
        """Test checking if FinGPT model is available."""
        with patch('src.analysis.ai_models.model_factory.FinGPTClient') as MockFinGPT:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockFinGPT.return_value = mock_instance
            
            is_available = ModelFactory.is_model_available(AIModelType.FINGPT)
            assert is_available is True

        # Test with import error
        with patch('src.analysis.ai_models.model_factory.FinGPTClient', side_effect=ImportError()):
            is_available = ModelFactory.is_model_available(AIModelType.FINGPT)
            assert is_available is False

    def test_factory_supports_all_enum_values(self):
        """Test factory supports all defined AI model types."""
        for model_type in AIModelType:
            with patch(f'src.analysis.ai_models.model_factory.{self._get_mock_class(model_type)}') as MockClass:
                mock_instance = MagicMock(spec=AIModelInterface)
                MockClass.return_value = mock_instance
                
                model = ModelFactory.create_model(model_type)
                assert model is not None
                assert model == mock_instance

    def _get_mock_class(self, model_type: AIModelType) -> str:
        """Get the mock class name for a model type."""
        if model_type == AIModelType.OLLAMA:
            return "OllamaWrapper"
        elif model_type == AIModelType.FINGPT:
            return "FinGPTClient"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def test_factory_error_handling(self):
        """Test factory error handling scenarios."""
        # Test with None model type
        with pytest.raises((ValueError, TypeError)):
            ModelFactory.create_model(None)

        # Test with invalid enum value
        with pytest.raises(ValueError):
            ModelFactory.create_model("not_a_real_model")

    def test_factory_config_validation(self):
        """Test factory validates configuration properly."""
        # Test with mismatched config
        config = AIModelConfig(model_type=AIModelType.OLLAMA)
        
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockOllama.return_value = mock_instance
            
            # Should work with Ollama type
            model = ModelFactory.create_model(AIModelType.OLLAMA, config=config)
            assert model is not None

    def test_factory_singleton_behavior(self):
        """Test that factory creates new instances each time."""
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance1 = MagicMock(spec=AIModelInterface)
            mock_instance2 = MagicMock(spec=AIModelInterface)
            MockOllama.side_effect = [mock_instance1, mock_instance2]
            
            model1 = ModelFactory.create_model(AIModelType.OLLAMA)
            model2 = ModelFactory.create_model(AIModelType.OLLAMA)
            
            # Should create separate instances
            assert model1 != model2
            assert MockOllama.call_count == 2


class TestFactoryIntegration:
    """Test factory integration scenarios."""

    def test_factory_with_real_config_objects(self):
        """Test factory with actual configuration objects."""
        # Test Ollama config
        ollama_config = AIModelConfig(
            model_type=AIModelType.OLLAMA,
            fallback_enabled=True
        )
        
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockOllama.return_value = mock_instance
            
            model = ModelFactory.create_model(AIModelType.OLLAMA, config=ollama_config)
            assert model is not None

        # Test FinGPT config
        fingpt_config = AIModelConfig(
            model_type=AIModelType.FINGPT,
            fingpt_model_variant="v3.1",
            comparative_mode=True
        )
        
        with patch('src.analysis.ai_models.model_factory.FinGPTClient') as MockFinGPT:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockFinGPT.return_value = mock_instance
            
            model = ModelFactory.create_model(AIModelType.FINGPT, config=fingpt_config)
            assert model is not None
            MockFinGPT.assert_called_once_with(model_variant="v3.1")

    def test_factory_extensibility(self):
        """Test that factory can be extended with new model types."""
        # This test ensures the factory pattern is extensible
        # When new AI models are added, only the factory needs to be updated
        
        current_types = len(AIModelType)
        available_models = ModelFactory.get_available_models()
        
        assert len(available_models) == current_types
        
        # Verify all enum values are handled
        for model_type in AIModelType:
            assert model_type.value in available_models

    def test_factory_performance(self):
        """Test factory performance characteristics."""
        # Factory should be fast and not do heavy initialization
        import time
        
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockOllama.return_value = mock_instance
            
            start_time = time.time()
            model = ModelFactory.create_model(AIModelType.OLLAMA)
            end_time = time.time()
            
            # Factory should be very fast (< 1ms typically)
            assert end_time - start_time < 0.1  # 100ms threshold for CI
            assert model is not None