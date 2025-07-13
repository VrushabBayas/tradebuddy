"""
Integration tests for dual AI system.

Tests the interaction between different AI models and the overall system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.analysis.ai_models.ai_interface import AIModelInterface
from src.analysis.ai_models.model_factory import ModelFactory
from src.core.models import (
    AIModelType,
    AIModelConfig,
    MarketData,
    StrategyType,
    Symbol,
    OHLCV,
    TradingSignal,
    SignalAction,
)


class TestDualAISystem:
    """Test dual AI system functionality."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return MarketData(
            symbol=Symbol.BTCUSD,
            timeframe="1m",
            current_price=45000.0,
            ohlcv_data=[
                OHLCV(open=44900, high=45100, low=44800, close=45000, volume=1000)
            ]
        )

    @pytest.fixture
    def sample_technical_analysis(self):
        """Create sample technical analysis data."""
        return {
            "ema_crossover": {
                "ema_9": 45050.0,
                "ema_15": 44950.0,
                "is_golden_cross": True,
                "crossover_strength": 7
            }
        }

    def test_ai_model_interface_contract(self):
        """Test that all AI models follow the same interface contract."""
        # This test ensures LSP (Liskov Substitution Principle)
        
        # Verify interface methods exist
        interface_methods = ['analyze_market', 'check_model_health', 'close', 'model_name', 'model_version']
        
        for method in interface_methods:
            assert hasattr(AIModelInterface, method), f"Interface missing method: {method}"

    @pytest.mark.asyncio
    async def test_ollama_model_via_factory(self, sample_market_data, sample_technical_analysis):
        """Test Ollama model creation and basic functionality via factory."""
        
        with patch('src.analysis.ai_models.ollama_wrapper.OllamaWrapper') as MockOllamaWrapper:
            # Create mock instance
            mock_instance = AsyncMock(spec=AIModelInterface)
            mock_instance.model_name = "ollama:qwen2.5:14b"
            mock_instance.model_version = "ollama-local"
            mock_instance.analyze_market.return_value = MagicMock()
            mock_instance.check_model_health.return_value = True
            
            MockOllamaWrapper.return_value = mock_instance
            
            # Test factory creation
            with patch('src.analysis.ai_models.model_factory.OllamaWrapper', MockOllamaWrapper):
                model = ModelFactory.create_model(AIModelType.OLLAMA)
                
                assert model is not None
                assert model == mock_instance
                assert model.model_name == "ollama:qwen2.5:14b"
                
                # Test interface compliance
                health = await model.check_model_health()
                assert health is True

    @pytest.mark.asyncio
    async def test_fingpt_model_via_factory(self, sample_market_data, sample_technical_analysis):
        """Test FinGPT model creation and basic functionality via factory."""
        
        with patch('src.analysis.ai_models.fingpt_client.FinGPTClient') as MockFinGPTClient:
            # Create mock instance
            mock_instance = AsyncMock(spec=AIModelInterface)
            mock_instance.model_name = "fingpt:v3.2"
            mock_instance.model_version = "v3.2"
            mock_instance.analyze_market.return_value = MagicMock()
            mock_instance.check_model_health.return_value = True
            
            MockFinGPTClient.return_value = mock_instance
            
            # Test factory creation
            with patch('src.analysis.ai_models.model_factory.FinGPTClient', MockFinGPTClient):
                model = ModelFactory.create_model(AIModelType.FINGPT)
                
                assert model is not None
                assert model == mock_instance
                assert model.model_name == "fingpt:v3.2"
                
                # Test interface compliance
                health = await model.check_model_health()
                assert health is True

    @pytest.mark.asyncio
    async def test_model_switching_functionality(self, sample_market_data, sample_technical_analysis):
        """Test switching between different AI models."""
        
        # Mock both models
        ollama_mock = AsyncMock(spec=AIModelInterface)
        ollama_mock.model_name = "ollama:qwen2.5:14b"
        ollama_mock.analyze_market.return_value = MagicMock()
        
        fingpt_mock = AsyncMock(spec=AIModelInterface)
        fingpt_mock.model_name = "fingpt:v3.2"
        fingpt_mock.analyze_market.return_value = MagicMock()
        
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper', return_value=ollama_mock), \
             patch('src.analysis.ai_models.model_factory.FinGPTClient', return_value=fingpt_mock):
            
            # Test switching from Ollama to FinGPT
            ollama_model = ModelFactory.create_model(AIModelType.OLLAMA)
            fingpt_model = ModelFactory.create_model(AIModelType.FINGPT)
            
            assert ollama_model.model_name == "ollama:qwen2.5:14b"
            assert fingpt_model.model_name == "fingpt:v3.2"
            
            # Both should support the same interface
            assert hasattr(ollama_model, 'analyze_market')
            assert hasattr(fingpt_model, 'analyze_market')

    def test_ai_config_integration(self):
        """Test AI model configuration integration."""
        # Test default configuration
        default_config = AIModelConfig()
        assert default_config.model_type == AIModelType.OLLAMA
        assert default_config.fallback_enabled is True
        assert default_config.comparative_mode is False
        
        # Test FinGPT configuration
        fingpt_config = AIModelConfig(
            model_type=AIModelType.FINGPT,
            fingpt_model_variant="v3.3",
            comparative_mode=True
        )
        assert fingpt_config.model_type == AIModelType.FINGPT
        assert fingpt_config.fingpt_model_variant == "v3.3"
        assert fingpt_config.comparative_mode is True

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test that both models handle errors consistently."""
        from src.core.exceptions import APIConnectionError, DataValidationError
        
        # Test that both models raise same exceptions for same scenarios
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama, \
             patch('src.analysis.ai_models.model_factory.FinGPTClient') as MockFinGPT:
            
            # Mock connection errors
            ollama_mock = AsyncMock(spec=AIModelInterface)
            ollama_mock.analyze_market.side_effect = APIConnectionError("Connection failed")
            MockOllama.return_value = ollama_mock
            
            fingpt_mock = AsyncMock(spec=AIModelInterface)
            fingpt_mock.analyze_market.side_effect = APIConnectionError("Connection failed")
            MockFinGPT.return_value = fingpt_mock
            
            ollama_model = ModelFactory.create_model(AIModelType.OLLAMA)
            fingpt_model = ModelFactory.create_model(AIModelType.FINGPT)
            
            # Both should raise same exception type
            with pytest.raises(APIConnectionError):
                await ollama_model.analyze_market(None, {}, StrategyType.EMA_CROSSOVER)
                
            with pytest.raises(APIConnectionError):
                await fingpt_model.analyze_market(None, {}, StrategyType.EMA_CROSSOVER)

    def test_model_factory_extensibility(self):
        """Test that the factory pattern supports easy extension."""
        # This test ensures Open/Closed Principle
        
        # Get current available models
        available_models = ModelFactory.get_available_models()
        
        # Should include both current models
        assert "ollama" in available_models
        assert "fingpt" in available_models
        
        # Should be extensible (factory returns interface)
        for model_type in AIModelType:
            # Each enum value should be handled
            assert model_type.value in available_models

    @pytest.mark.asyncio
    async def test_concurrent_model_usage(self, sample_market_data, sample_technical_analysis):
        """Test using multiple models concurrently."""
        
        # Mock both models
        ollama_result = MagicMock()
        ollama_result.signals = [MagicMock(action=SignalAction.BUY, confidence=8)]
        
        fingpt_result = MagicMock()
        fingpt_result.signals = [MagicMock(action=SignalAction.BUY, confidence=7)]
        
        ollama_mock = AsyncMock(spec=AIModelInterface)
        ollama_mock.analyze_market.return_value = ollama_result
        
        fingpt_mock = AsyncMock(spec=AIModelInterface)
        fingpt_mock.analyze_market.return_value = fingpt_result
        
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper', return_value=ollama_mock), \
             patch('src.analysis.ai_models.model_factory.FinGPTClient', return_value=fingpt_mock):
            
            # Create both models
            ollama_model = ModelFactory.create_model(AIModelType.OLLAMA)
            fingpt_model = ModelFactory.create_model(AIModelType.FINGPT)
            
            # Should be able to use both concurrently
            import asyncio
            results = await asyncio.gather(
                ollama_model.analyze_market(sample_market_data, sample_technical_analysis, StrategyType.EMA_CROSSOVER),
                fingpt_model.analyze_market(sample_market_data, sample_technical_analysis, StrategyType.EMA_CROSSOVER),
                return_exceptions=True
            )
            
            assert len(results) == 2
            assert results[0] == ollama_result
            assert results[1] == fingpt_result

    def test_ai_model_types_enum_completeness(self):
        """Test that all AI model types are properly defined."""
        # Test enum values
        assert hasattr(AIModelType, 'OLLAMA')
        assert hasattr(AIModelType, 'FINGPT')
        
        # Test enum values are strings
        assert AIModelType.OLLAMA == "ollama"
        assert AIModelType.FINGPT == "fingpt"
        
        # Test all enum values are supported by factory
        for model_type in AIModelType:
            available_models = ModelFactory.get_available_models()
            assert model_type.value in available_models


class TestSystemIntegration:
    """Test system-level integration scenarios."""

    @pytest.mark.asyncio
    async def test_fallback_mechanism_concept(self):
        """Test fallback mechanism concept (mock implementation)."""
        # This tests the concept of fallback between models
        
        # Mock primary model failure
        primary_mock = AsyncMock(spec=AIModelInterface)
        primary_mock.check_model_health.return_value = False
        
        # Mock fallback model success
        fallback_mock = AsyncMock(spec=AIModelInterface)
        fallback_mock.check_model_health.return_value = True
        
        # Test health check concept
        primary_healthy = await primary_mock.check_model_health()
        fallback_healthy = await fallback_mock.check_model_health()
        
        assert primary_healthy is False
        assert fallback_healthy is True
        
        # In a real implementation, this would trigger fallback logic

    def test_comparative_mode_concept(self):
        """Test comparative mode concept."""
        # Test configuration for comparative mode
        config = AIModelConfig(comparative_mode=True)
        assert config.comparative_mode is True
        
        # In comparative mode, both models would be run and results compared
        # This is a design validation test

    @pytest.mark.asyncio
    async def test_model_lifecycle_management(self):
        """Test proper model lifecycle management."""
        
        mock_model = AsyncMock(spec=AIModelInterface)
        mock_model.close = AsyncMock()
        
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper', return_value=mock_model):
            model = ModelFactory.create_model(AIModelType.OLLAMA)
            
            # Test context manager support
            async with model:
                pass  # Model should be properly managed
            
            # Test manual cleanup
            await model.close()
            mock_model.close.assert_called()

    def test_dependency_injection_pattern(self):
        """Test that the system follows dependency injection patterns."""
        # Factory creates objects that implement the interface
        # This allows for easy testing and swapping of implementations
        
        # Test that factory returns interface implementations
        with patch('src.analysis.ai_models.model_factory.OllamaWrapper') as MockOllama:
            mock_instance = MagicMock(spec=AIModelInterface)
            MockOllama.return_value = mock_instance
            
            model = ModelFactory.create_model(AIModelType.OLLAMA)
            
            # Should be injectable wherever AIModelInterface is expected
            assert isinstance(model, AIModelInterface)
            
            # This enables SOLID principles:
            # - Dependency Inversion: High-level modules depend on abstractions
            # - Open/Closed: Open for extension, closed for modification