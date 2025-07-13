"""
Factory for creating AI model instances.

Implements Factory pattern following SOLID principles.
"""

import structlog
from typing import Optional

from src.core.models import AIModelType, AIModelConfig
from src.analysis.ai_models.ai_interface import AIModelInterface

logger = structlog.get_logger(__name__)


class ModelFactory:
    """
    Factory for creating AI model instances.
    
    Following SOLID principles:
    - Single Responsibility: Create AI model instances
    - Open/Closed: Easy to add new models without modifying existing code
    - Dependency Inversion: Returns abstract interface
    """

    @staticmethod
    def create_model(
        model_type: AIModelType,
        config: Optional[AIModelConfig] = None,
        **kwargs
    ) -> AIModelInterface:
        """
        Create AI model instance based on type.

        Args:
            model_type: Type of AI model to create
            config: Optional AI model configuration
            **kwargs: Additional model-specific parameters

        Returns:
            AI model instance implementing AIModelInterface

        Raises:
            ValueError: If model type is not supported
            ImportError: If required dependencies are not installed
        """
        # Validate model_type is correct enum type (Fail Fast)
        if not isinstance(model_type, AIModelType):
            raise ValueError(f"Invalid model type: {model_type}. Must be an AIModelType enum.")

        logger.debug("Creating AI model", model_type=model_type.value)

        if model_type == AIModelType.OLLAMA:
            return ModelFactory._create_ollama_model(config, **kwargs)
        elif model_type == AIModelType.FINGPT:
            return ModelFactory._create_fingpt_model(config, **kwargs)
        else:
            raise ValueError(f"Unsupported AI model type: {model_type}")

    @staticmethod
    def _create_ollama_model(
        config: Optional[AIModelConfig] = None, **kwargs
    ) -> AIModelInterface:
        """Create Ollama model instance."""
        try:
            from src.analysis.ai_models.ollama_wrapper import OllamaWrapper
            
            logger.debug("Creating Ollama model wrapper")
            return OllamaWrapper(**kwargs)
        except ImportError as e:
            logger.error("Failed to import Ollama dependencies", error=str(e))
            raise ImportError(
                "Ollama dependencies not available. "
                "Ensure Ollama client is properly installed."
            ) from e

    @staticmethod
    def _create_fingpt_model(
        config: Optional[AIModelConfig] = None, **kwargs
    ) -> AIModelInterface:
        """Create FinGPT model instance."""
        try:
            from src.analysis.ai_models.fingpt_client import FinGPTClient
            
            # Use config for FinGPT-specific parameters
            model_variant = "v3.2"  # Default
            if config and config.fingpt_model_variant:
                model_variant = config.fingpt_model_variant
            
            logger.debug("Creating FinGPT model", variant=model_variant)
            return FinGPTClient(model_variant=model_variant, **kwargs)
        except ImportError as e:
            logger.error("Failed to import FinGPT dependencies", error=str(e))
            raise ImportError(
                "FinGPT dependencies not available. "
                "Ensure required packages are installed."
            ) from e

    @staticmethod
    def get_available_models() -> list[str]:
        """
        Get list of available AI model types.

        Returns:
            List of available model type names
        """
        return [model_type.value for model_type in AIModelType]

    @staticmethod
    def is_model_available(model_type: AIModelType) -> bool:
        """
        Check if a specific model type is available.

        Args:
            model_type: AI model type to check

        Returns:
            True if model can be created
        """
        try:
            # Try to create model to check availability
            model = ModelFactory.create_model(model_type)
            return model is not None
        except (ImportError, ValueError):
            return False