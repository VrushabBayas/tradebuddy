"""
Test AI model interface abstraction.

Tests the contract that all AI models must follow.
"""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

from src.analysis.ai_models.ai_interface import AIModelInterface
from src.core.models import (
    MarketData, 
    AnalysisResult, 
    StrategyType, 
    SessionConfig,
    Symbol,
    OHLCV
)
from src.core.exceptions import APIConnectionError, DataValidationError


class TestAIModelInterface:
    """Test AI model interface contract."""
    
    def test_ai_interface_is_abstract(self):
        """Test that AIModelInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AIModelInterface()
    
    @pytest.mark.asyncio
    async def test_ai_interface_contract_methods(self):
        """Test that interface defines required methods."""
        # Verify interface has required methods
        assert hasattr(AIModelInterface, 'analyze_market')
        assert hasattr(AIModelInterface, 'check_model_health')
        assert hasattr(AIModelInterface, 'close')
        
        # Check method signatures
        import inspect
        analyze_sig = inspect.signature(AIModelInterface.analyze_market)
        assert 'market_data' in analyze_sig.parameters
        assert 'technical_analysis' in analyze_sig.parameters
        assert 'strategy' in analyze_sig.parameters
        assert 'session_config' in analyze_sig.parameters
    
    @pytest.mark.asyncio
    async def test_analyze_market_input_validation(self):
        """Test that analyze_market validates inputs properly."""
        from src.analysis.ai_models.ai_interface import validate_ai_inputs
        
        # Valid inputs should pass
        market_data = MarketData(
            symbol=Symbol.BTCUSD,
            timeframe="1m",
            current_price=45000.0,
            ohlcv_data=[
                OHLCV(open=44900, high=45100, low=44800, close=45000, volume=1000)
            ]
        )
        technical_analysis = {"test": "data"}
        strategy = StrategyType.EMA_CROSSOVER
        
        # Should not raise exception
        validate_ai_inputs(market_data, technical_analysis, strategy)
        
        # Invalid market data should fail
        with pytest.raises(DataValidationError):
            validate_ai_inputs(None, technical_analysis, strategy)
        
        # Empty OHLCV data should fail
        invalid_market_data = MarketData(
            symbol=Symbol.BTCUSD,
            timeframe="1m",
            current_price=45000.0,
            ohlcv_data=[]
        )
        with pytest.raises(DataValidationError):
            validate_ai_inputs(invalid_market_data, technical_analysis, strategy)
    
    @pytest.mark.asyncio
    async def test_analysis_result_structure(self):
        """Test that analysis result follows expected structure."""
        # This test ensures all AI models return consistent AnalysisResult
        expected_fields = {
            'symbol', 'timeframe', 'strategy', 'market_data', 
            'signals', 'ai_analysis', 'execution_time', 'timestamp'
        }
        
        result_fields = set(AnalysisResult.model_fields.keys())
        assert expected_fields.issubset(result_fields)


class TestAIModelErrorHandling:
    """Test error handling for AI models."""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test that connection errors are properly handled."""
        from src.analysis.ai_models.ai_interface import handle_ai_errors
        
        async def mock_failing_function():
            raise APIConnectionError("Connection failed")
        
        with pytest.raises(APIConnectionError):
            await handle_ai_errors(mock_failing_function)
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test that timeout errors are properly handled."""
        from src.analysis.ai_models.ai_interface import handle_ai_errors
        from src.core.exceptions import APITimeoutError
        
        async def mock_timeout_function():
            raise APITimeoutError("Request timeout")
        
        with pytest.raises(APITimeoutError):
            await handle_ai_errors(mock_timeout_function)


class TestAIModelFactory:
    """Test AI model factory pattern."""
    
    def test_factory_creates_correct_model_type(self):
        """Test that factory creates correct AI model based on type."""
        from src.analysis.ai_models.model_factory import ModelFactory
        from src.core.models import AIModelType
        
        # Test Ollama model creation
        ollama_model = ModelFactory.create_model(AIModelType.OLLAMA)
        assert ollama_model is not None
        assert isinstance(ollama_model, AIModelInterface)
        
        # Test FinGPT model creation
        fingpt_model = ModelFactory.create_model(AIModelType.FINGPT)
        assert fingpt_model is not None
        assert isinstance(fingpt_model, AIModelInterface)
    
    def test_factory_invalid_model_type_fails_fast(self):
        """Test that invalid model type fails fast."""
        from src.analysis.ai_models.model_factory import ModelFactory
        
        with pytest.raises(ValueError):
            ModelFactory.create_model("invalid_model_type")
    
    def test_factory_supports_all_ai_model_types(self):
        """Test that factory supports all defined AI model types."""
        from src.analysis.ai_models.model_factory import ModelFactory
        from src.core.models import AIModelType
        
        # Should not raise exception for any valid type
        for model_type in AIModelType:
            model = ModelFactory.create_model(model_type)
            assert model is not None
            assert isinstance(model, AIModelInterface)


class TestAIModelConfiguration:
    """Test AI model configuration."""
    
    def test_ai_model_config_validation(self):
        """Test AI model configuration validation."""
        from src.core.models import AIModelConfig, AIModelType
        
        # Valid config should pass
        config = AIModelConfig(
            model_type=AIModelType.OLLAMA,
            fallback_enabled=True
        )
        assert config.model_type == AIModelType.OLLAMA
        assert config.fallback_enabled is True
        
        # Default values should be set
        default_config = AIModelConfig()
        assert default_config.model_type == AIModelType.OLLAMA
        assert default_config.fallback_enabled is True
        assert default_config.comparative_mode is False
    
    def test_fingpt_specific_config(self):
        """Test FinGPT-specific configuration options."""
        from src.core.models import AIModelConfig, AIModelType
        
        config = AIModelConfig(
            model_type=AIModelType.FINGPT,
            fingpt_model_variant="v3.2",
            comparative_mode=True
        )
        assert config.model_type == AIModelType.FINGPT
        assert config.fingpt_model_variant == "v3.2"
        assert config.comparative_mode is True