"""
Comparative AI analysis module.

Runs multiple AI models simultaneously and compares their results
for enhanced trading decision making.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
import structlog

from src.analysis.ai_models.ai_interface import AIModelInterface
from src.analysis.ai_models.model_factory import ModelFactory
from src.core.models import (
    MarketData, AnalysisResult, SessionConfig, AIModelType, 
    TradingSignal, SignalAction
)
from src.core.exceptions import StrategyError

logger = structlog.get_logger(__name__)


class ComparativeAnalyzer:
    """
    Comparative analysis system for running multiple AI models simultaneously.
    
    Features:
    - Parallel execution of multiple AI models
    - Result comparison and consensus analysis
    - Performance tracking and accuracy metrics
    - Ensemble signal generation
    - Detailed comparison reports
    """

    def __init__(self):
        """Initialize comparative analyzer."""
        self.models: Dict[str, AIModelInterface] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        
        logger.info("Comparative analyzer initialized")

    async def analyze_with_comparison(
        self,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy_type: str,
        session_config: SessionConfig,
        models_to_compare: List[AIModelType] = None
    ) -> Dict[str, Any]:
        """
        Run comparative analysis with multiple AI models.

        Args:
            market_data: Market data to analyze
            technical_analysis: Technical analysis results
            strategy_type: Trading strategy type
            session_config: Session configuration
            models_to_compare: List of AI models to compare (defaults to Ollama + FinGPT)

        Returns:
            Comprehensive comparison results with consensus analysis
        """
        if models_to_compare is None:
            models_to_compare = [AIModelType.OLLAMA, AIModelType.FINGPT]

        logger.info(
            "Starting comparative analysis",
            symbol=market_data.symbol,
            models=len(models_to_compare),
            model_types=[model.value for model in models_to_compare]
        )

        start_time = time.time()
        
        # Initialize models if needed
        await self._ensure_models_initialized(models_to_compare, session_config)

        # Run analyses in parallel
        analysis_tasks = []
        for model_type in models_to_compare:
            task = self._run_single_analysis(
                model_type, market_data, technical_analysis, strategy_type, session_config
            )
            analysis_tasks.append(task)

        # Execute all analyses concurrently
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Process results
        comparison_results = self._process_comparative_results(
            models_to_compare, results, market_data, start_time
        )

        # Generate consensus analysis
        consensus = self._generate_consensus_analysis(comparison_results)

        # Store in history for performance tracking
        self._update_analysis_history(comparison_results, consensus)

        logger.info(
            "Comparative analysis completed",
            symbol=market_data.symbol,
            execution_time=time.time() - start_time,
            successful_models=len([r for r in results if not isinstance(r, Exception)]),
            consensus_signal=consensus.get("consensus_signal", {}).get("action", "NONE")
        )

        return {
            "individual_results": comparison_results,
            "consensus": consensus,
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }

    async def _ensure_models_initialized(
        self, model_types: List[AIModelType], session_config: SessionConfig
    ):
        """Ensure all required AI models are initialized."""
        for model_type in model_types:
            model_key = model_type.value
            if model_key not in self.models:
                try:
                    self.models[model_key] = ModelFactory.create_model(
                        model_type, session_config.ai_model_config
                    )
                    logger.debug("Model initialized", model=model_key)
                except Exception as e:
                    logger.error("Failed to initialize model", model=model_key, error=str(e))
                    raise StrategyError(f"Failed to initialize {model_key} model: {str(e)}")

    async def _run_single_analysis(
        self,
        model_type: AIModelType,
        market_data: MarketData,
        technical_analysis: Dict[str, Any],
        strategy_type: str,
        session_config: SessionConfig
    ) -> Tuple[AIModelType, AnalysisResult]:
        """Run analysis with a single AI model."""
        model_key = model_type.value
        
        try:
            logger.debug("Starting analysis", model=model_key)
            start_time = time.time()
            
            model = self.models[model_key]
            result = await model.analyze_market(
                market_data, technical_analysis, strategy_type, session_config
            )
            
            execution_time = time.time() - start_time
            logger.debug(
                "Analysis completed", 
                model=model_key, 
                execution_time=execution_time,
                signals_count=len(result.signals)
            )
            
            return (model_type, result)
            
        except Exception as e:
            logger.error("Analysis failed", model=model_key, error=str(e))
            return (model_type, e)

    def _process_comparative_results(
        self,
        model_types: List[AIModelType],
        results: List[Any],
        market_data: MarketData,
        start_time: float
    ) -> Dict[str, Dict[str, Any]]:
        """Process and structure comparative analysis results."""
        comparison_results = {}
        
        for model_type, result in zip(model_types, results):
            model_key = model_type.value
            
            if isinstance(result, Exception):
                comparison_results[model_key] = {
                    "status": "failed",
                    "error": str(result),
                    "signals": [],
                    "analysis": f"Analysis failed: {str(result)}",
                    "execution_time": None
                }
            else:
                try:
                    # Ensure result is a tuple with (model_type, AnalysisResult)
                    if isinstance(result, tuple) and len(result) == 2:
                        analysis_result = result[1]
                    else:
                        # If not a tuple, assume it's the AnalysisResult directly
                        analysis_result = result
                    
                    # Verify the analysis_result has required attributes
                    if not hasattr(analysis_result, 'signals'):
                        raise AttributeError(f"Analysis result missing 'signals' attribute. Type: {type(analysis_result)}")
                    
                    comparison_results[model_key] = {
                        "status": "success",
                        "error": None,
                        "signals": analysis_result.signals,
                        "analysis": getattr(analysis_result, 'ai_analysis', 'No analysis available'),
                        "primary_signal": getattr(analysis_result, 'primary_signal', None),
                        "execution_time": getattr(analysis_result, 'execution_time', None),
                        "signal_count": len(analysis_result.signals),
                        "avg_confidence": self._calculate_average_confidence(analysis_result.signals)
                    }
                except Exception as e:
                    logger.error(f"Error processing result for {model_key}: {str(e)}")
                    comparison_results[model_key] = {
                        "status": "failed",
                        "error": f"Result processing error: {str(e)}",
                        "signals": [],
                        "analysis": f"Failed to process analysis result: {str(e)}",
                        "execution_time": None
                    }

        return comparison_results

    def _calculate_average_confidence(self, signals: List[TradingSignal]) -> float:
        """Calculate average confidence across signals."""
        if not signals:
            return 0.0
        return sum(signal.confidence for signal in signals) / len(signals)

    def _generate_consensus_analysis(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate consensus analysis from multiple model results."""
        successful_results = {
            model: result for model, result in comparison_results.items() 
            if result["status"] == "success"
        }

        if not successful_results:
            return {
                "consensus_signal": {"action": "WAIT", "confidence": 1, "reasoning": "All models failed"},
                "agreement_level": 0.0,
                "model_agreement": {},
                "recommendation": "No models available for analysis"
            }

        # Analyze signal agreement
        signal_actions = []
        signal_confidences = []
        
        for model, result in successful_results.items():
            if result["primary_signal"]:
                signal_actions.append(result["primary_signal"].action)
                signal_confidences.append(result["primary_signal"].confidence)

        # Calculate consensus
        consensus = self._calculate_signal_consensus(signal_actions, signal_confidences)
        
        # Calculate agreement metrics
        agreement_metrics = self._calculate_agreement_metrics(successful_results)
        
        # Generate recommendation
        recommendation = self._generate_consensus_recommendation(
            consensus, agreement_metrics, successful_results
        )

        return {
            "consensus_signal": consensus,
            "agreement_level": agreement_metrics["overall_agreement"],
            "model_agreement": agreement_metrics["model_agreement"],
            "recommendation": recommendation,
            "participating_models": list(successful_results.keys()),
            "failed_models": [
                model for model, result in comparison_results.items() 
                if result["status"] == "failed"
            ]
        }

    def _calculate_signal_consensus(
        self, actions: List[SignalAction], confidences: List[float]
    ) -> Dict[str, Any]:
        """Calculate consensus signal from multiple model outputs."""
        if not actions:
            return {"action": "WAIT", "confidence": 1, "reasoning": "No signals available"}

        # Count action occurrences
        action_counts = {}
        action_confidences = {}
        
        for action, confidence in zip(actions, confidences):
            action_str = str(action)
            if action_str not in action_counts:
                action_counts[action_str] = 0
                action_confidences[action_str] = []
            
            action_counts[action_str] += 1
            action_confidences[action_str].append(confidence)

        # Find majority action
        majority_action = max(action_counts.keys(), key=action_counts.get)
        majority_count = action_counts[majority_action]
        total_models = len(actions)

        # Calculate consensus confidence
        avg_confidence = sum(action_confidences[majority_action]) / len(action_confidences[majority_action])
        agreement_ratio = majority_count / total_models
        
        # Adjust confidence based on agreement
        consensus_confidence = avg_confidence * agreement_ratio

        # Generate reasoning
        if agreement_ratio == 1.0:
            reasoning = f"All {total_models} models agree on {majority_action} signal"
        elif agreement_ratio >= 0.5:
            reasoning = f"{majority_count}/{total_models} models agree on {majority_action} signal"
        else:
            reasoning = f"No clear consensus among {total_models} models"

        return {
            "action": majority_action,
            "confidence": min(10, max(1, int(consensus_confidence))),
            "reasoning": reasoning,
            "agreement_ratio": agreement_ratio,
            "participating_models": total_models
        }

    def _calculate_agreement_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed agreement metrics between models."""
        if len(results) < 2:
            return {"overall_agreement": 1.0, "model_agreement": {}}

        # Extract primary signals
        model_signals = {}
        for model, result in results.items():
            if result["primary_signal"]:
                model_signals[model] = {
                    "action": str(result["primary_signal"].action),
                    "confidence": result["primary_signal"].confidence
                }

        # Calculate pairwise agreement
        agreements = []
        model_pairs = []
        
        models = list(model_signals.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                signal1 = model_signals[model1]
                signal2 = model_signals[model2]
                
                # Action agreement (binary)
                action_agreement = 1.0 if signal1["action"] == signal2["action"] else 0.0
                
                # Confidence similarity (scaled 0-1)
                conf_diff = abs(signal1["confidence"] - signal2["confidence"])
                confidence_agreement = max(0, 1 - (conf_diff / 10))  # Normalize by max confidence diff
                
                # Combined agreement
                combined_agreement = (action_agreement * 0.7) + (confidence_agreement * 0.3)
                
                agreements.append(combined_agreement)
                model_pairs.append(f"{model1}_vs_{model2}")

        overall_agreement = sum(agreements) / len(agreements) if agreements else 1.0

        return {
            "overall_agreement": overall_agreement,
            "model_agreement": dict(zip(model_pairs, agreements)),
            "action_agreements": [a >= 0.7 for a in agreements],  # Strong action agreement
            "confidence_agreements": [a >= 0.5 for a in agreements]  # Reasonable confidence agreement
        }

    def _generate_consensus_recommendation(
        self, 
        consensus: Dict[str, Any], 
        agreement_metrics: Dict[str, Any],
        results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate human-readable consensus recommendation."""
        action = consensus["action"]
        confidence = consensus["confidence"]
        agreement = agreement_metrics["overall_agreement"]
        
        # Base recommendation
        if action in ["BUY", "SELL"]:
            if agreement >= 0.8 and confidence >= 7:
                strength = "Strong"
            elif agreement >= 0.6 and confidence >= 6:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            recommendation = f"{strength} consensus for {action} signal"
        else:
            recommendation = f"Models suggest {action} - mixed or unclear signals"

        # Add agreement context
        if agreement >= 0.8:
            recommendation += " with high model agreement"
        elif agreement >= 0.6:
            recommendation += " with moderate model agreement"
        else:
            recommendation += " with low model agreement"

        # Add participating models context
        participating = len(results)
        if participating == 1:
            recommendation += f" (single model analysis)"
        else:
            recommendation += f" ({participating} models analyzed)"

        return recommendation

    def _update_analysis_history(
        self, comparison_results: Dict[str, Dict[str, Any]], consensus: Dict[str, Any]
    ):
        """Update analysis history for performance tracking."""
        history_entry = {
            "timestamp": time.time(),
            "models_used": list(comparison_results.keys()),
            "successful_models": [
                model for model, result in comparison_results.items() 
                if result["status"] == "success"
            ],
            "consensus_action": consensus.get("consensus_signal", {}).get("action"),
            "agreement_level": consensus.get("agreement_level", 0),
            "execution_time": sum(
                result.get("execution_time", 0) or 0 
                for result in comparison_results.values()
            )
        }
        
        self.analysis_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from analysis history."""
        if not self.analysis_history:
            return {"message": "No analysis history available"}

        recent_analyses = self.analysis_history[-20:]  # Last 20 analyses
        
        return {
            "total_analyses": len(self.analysis_history),
            "recent_analyses": len(recent_analyses),
            "avg_agreement_level": sum(
                entry.get("agreement_level", 0) for entry in recent_analyses
            ) / len(recent_analyses),
            "avg_execution_time": sum(
                entry.get("execution_time", 0) for entry in recent_analyses
            ) / len(recent_analyses),
            "model_success_rates": self._calculate_model_success_rates(recent_analyses),
            "consensus_distribution": self._calculate_consensus_distribution(recent_analyses)
        }

    def _calculate_model_success_rates(self, analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate success rates for each model."""
        model_attempts = {}
        model_successes = {}
        
        for analysis in analyses:
            for model in analysis.get("models_used", []):
                model_attempts[model] = model_attempts.get(model, 0) + 1
                
            for model in analysis.get("successful_models", []):
                model_successes[model] = model_successes.get(model, 0) + 1

        return {
            model: (model_successes.get(model, 0) / attempts) * 100
            for model, attempts in model_attempts.items()
        }

    def _calculate_consensus_distribution(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of consensus signals."""
        distribution = {}
        for analysis in analyses:
            action = analysis.get("consensus_action", "UNKNOWN")
            distribution[action] = distribution.get(action, 0) + 1
        return distribution

    async def close(self):
        """Close all AI model connections."""
        for model_name, model in self.models.items():
            try:
                await model.close()
                logger.debug("Model closed", model=model_name)
            except Exception as e:
                logger.warning("Error closing model", model=model_name, error=str(e))
        
        self.models.clear()
        logger.info("Comparative analyzer closed")