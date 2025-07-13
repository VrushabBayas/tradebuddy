"""
Ollama wrapper implementing AI model interface.

Wraps existing OllamaClient to implement AIModelInterface.
"""

from typing import Any, Dict, Optional
import structlog

from src.analysis.ai_models.ai_interface import AIModelInterface, validate_ai_inputs
from src.analysis.ollama_client import OllamaClient
from src.core.models import (
    MarketData,
    AnalysisResult,
    StrategyType,
    SessionConfig,
)

logger = structlog.get_logger(__name__)


class OllamaWrapper(AIModelInterface):
    """
    Wrapper for OllamaClient implementing AIModelInterface.
    
    Following SOLID principles:
    - Single Responsibility: Adapt OllamaClient to interface
    - Liskov Substitution: Can be used wherever AIModelInterface expected
    - Dependency Inversion: Depends on OllamaClient abstraction
    """

    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        """
        Initialize Ollama wrapper.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self._client = OllamaClient(
            base_url=base_url,
            model=model,
            timeout=timeout
        )
        
        logger.info(
            "Ollama wrapper initialized",
            model=self._client.model,
            base_url=self._client.base_url
        )

    async def analyze_market(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy: StrategyType,
        session_config: Optional[SessionConfig] = None,
    ) -> AnalysisResult:
        """Analyze market data using Ollama."""
        # Validate inputs (Fail Fast)
        validate_ai_inputs(market_data, technical_analysis, strategy)
        
        logger.debug(
            "Starting Ollama market analysis",
            symbol=market_data.symbol,
            strategy=strategy.value
        )
        
        # Delegate to existing OllamaClient
        return await self._client.analyze_market(
            market_data=market_data,
            technical_analysis=technical_analysis,
            strategy=strategy,
            session_config=session_config
        )

    async def check_model_health(self) -> bool:
        """Check Ollama model health."""
        try:
            return await self._client.check_model_health()
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close Ollama client connection."""
        try:
            await self._client.close()
            logger.debug("Ollama wrapper closed")
        except Exception as e:
            logger.error("Error closing Ollama wrapper", error=str(e))

    @property
    def model_name(self) -> str:
        """Get Ollama model name."""
        return f"ollama:{self._client.model}"

    @property
    def model_version(self) -> str:
        """Get Ollama model version."""
        return "ollama-local"

    async def __aenter__(self):
        """Async context manager entry."""
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._client.__aexit__(exc_type, exc_val, exc_tb)