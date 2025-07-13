"""
Shared EMA Analysis Components.

Provides reusable EMA calculation and analysis functionality for trading strategies.
Eliminates code duplication between EMA Crossover and EMA Crossover V2 strategies.
"""

import structlog
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from decimal import Decimal

from src.core.models import OHLCV
from src.core.constants import TradingConstants

logger = structlog.get_logger(__name__)


@dataclass
class EMAValues:
    """Container for EMA calculation results."""
    ema_9: float
    ema_15: float
    ema_50: Optional[float] = None
    current_price: float = 0.0
    
    @property
    def has_50_ema(self) -> bool:
        """Check if 50 EMA is available."""
        return self.ema_50 is not None


@dataclass
class EMACrossoverData:
    """Container for EMA crossover analysis results."""
    ema_values: EMAValues
    is_golden_cross: bool
    crossover_strength: int  # 1-10 scale
    separation_pct: float
    alignment: str  # bullish, bearish, strong_bullish, etc.
    
    @property
    def crossover_type(self) -> str:
        """Get crossover type as string."""
        return "golden_cross" if self.is_golden_cross else "death_cross"


class EMACalculator:
    """
    Specialized calculator for EMA-related analysis.
    
    Provides optimized, reusable EMA calculations with caching support.
    """
    
    def __init__(self):
        """Initialize EMA calculator."""
        self._cache = {}  # Simple cache for expensive calculations
        logger.debug("EMA calculator initialized")
    
    def calculate_basic_emas(
        self, 
        data: List[OHLCV], 
        periods: Tuple[int, int, int] = (9, 15, 50)
    ) -> EMAValues:
        """
        Calculate basic EMA values for given periods.
        
        Args:
            data: OHLCV data points
            periods: Tuple of (short, medium, long) EMA periods
            
        Returns:
            EMAValues with calculated EMAs
            
        Raises:
            ValueError: If insufficient data for calculations
        """
        short_period, medium_period, long_period = periods
        
        # Validate data sufficiency
        min_required = max(periods)
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data: {len(data)} periods provided, "
                f"minimum {min_required} required for EMA({long_period})"
            )
        
        # Import here to avoid circular imports
        from src.analysis.indicators import TechnicalIndicators
        indicators = TechnicalIndicators()
        
        logger.debug(
            "Calculating basic EMAs",
            data_points=len(data),
            periods=periods
        )
        
        # Calculate EMAs
        ema_9_values = indicators.calculate_ema(data, short_period)
        ema_15_values = indicators.calculate_ema(data, medium_period)
        
        # Calculate 50 EMA only if we have enough data
        ema_50_values = None
        if len(data) >= long_period:
            ema_50_values = indicators.calculate_ema(data, long_period)
        
        # Get current values
        current_ema_9 = ema_9_values[-1] if ema_9_values else 0.0
        current_ema_15 = ema_15_values[-1] if ema_15_values else 0.0
        current_ema_50 = ema_50_values[-1] if ema_50_values else None
        current_price = data[-1].close
        
        return EMAValues(
            ema_9=current_ema_9,
            ema_15=current_ema_15,
            ema_50=current_ema_50,
            current_price=current_price
        )
    
    def detect_crossover(self, ema_values: EMAValues) -> Tuple[bool, int]:
        """
        Detect EMA crossover and calculate strength.
        
        Args:
            ema_values: EMA values to analyze
            
        Returns:
            Tuple of (is_golden_cross, crossover_strength)
        """
        is_golden_cross = ema_values.ema_9 > ema_values.ema_15
        
        # Calculate crossover strength based on separation
        separation_pct = self.calculate_ema_separation(ema_values)
        
        # Map separation percentage to strength (1-10)
        if separation_pct >= TradingConstants.EMA_SEPARATION_STRONG:
            strength = 10
        elif separation_pct >= TradingConstants.EMA_SEPARATION_MODERATE:
            strength = 8
        elif separation_pct >= TradingConstants.EMA_SEPARATION_WEAK:
            strength = 6
        elif separation_pct >= 0.5:
            strength = 4
        else:
            strength = 2
        
        logger.debug(
            "EMA crossover detected",
            is_golden_cross=is_golden_cross,
            separation_pct=separation_pct,
            strength=strength
        )
        
        return is_golden_cross, strength
    
    def calculate_ema_separation(self, ema_values: EMAValues) -> float:
        """
        Calculate EMA separation as percentage of current price.
        
        Args:
            ema_values: EMA values to analyze
            
        Returns:
            Separation percentage
        """
        if ema_values.current_price == 0:
            return 0.0
            
        separation = abs(ema_values.ema_9 - ema_values.ema_15)
        separation_pct = (separation / ema_values.current_price) * 100
        
        return separation_pct
    
    def assess_ema_alignment(self, ema_values: EMAValues) -> str:
        """
        Assess EMA alignment patterns.
        
        Args:
            ema_values: EMA values to analyze
            
        Returns:
            Alignment pattern: 'strong_bullish', 'bullish', 'strong_bearish', 
                            'bearish', 'mixed', 'neutral'
        """
        price = ema_values.current_price
        ema_9 = ema_values.ema_9
        ema_15 = ema_values.ema_15
        ema_50 = ema_values.ema_50
        
        # Perfect bullish alignment: Price > 9 EMA > 15 EMA > 50 EMA
        if ema_50 and price > ema_9 > ema_15 > ema_50:
            # Check separation strength for strong vs regular bullish
            sep_9_15 = (ema_9 - ema_15) / ema_15 * 100
            sep_15_50 = (ema_15 - ema_50) / ema_50 * 100
            if sep_9_15 > 1.0 and sep_15_50 > 1.0:
                return "strong_bullish"
            else:
                return "bullish"
        
        # Perfect bearish alignment: Price < 9 EMA < 15 EMA < 50 EMA
        elif ema_50 and price < ema_9 < ema_15 < ema_50:
            # Check separation strength for strong vs regular bearish
            sep_9_15 = (ema_15 - ema_9) / ema_9 * 100
            sep_15_50 = (ema_50 - ema_15) / ema_15 * 100
            if sep_9_15 > 1.0 and sep_15_50 > 1.0:
                return "strong_bearish"
            else:
                return "bearish"
        
        # Partial alignments (without 50 EMA or price consideration)
        elif ema_9 > ema_15:
            if not ema_50:
                return "bullish"  # Only 9/15 alignment available
            elif price > ema_9:
                return "bullish"
            else:
                return "mixed"
        elif ema_9 < ema_15:
            if not ema_50:
                return "bearish"  # Only 9/15 alignment available
            elif price < ema_9:
                return "bearish"
            else:
                return "mixed"
        else:
            return "neutral"
    
    def create_crossover_analysis(
        self, 
        data: List[OHLCV], 
        include_50_ema: bool = True
    ) -> EMACrossoverData:
        """
        Create comprehensive EMA crossover analysis.
        
        Args:
            data: OHLCV data points
            include_50_ema: Whether to include 50 EMA in analysis
            
        Returns:
            Complete EMACrossoverData analysis
        """
        logger.debug(
            "Creating EMA crossover analysis",
            data_points=len(data),
            include_50_ema=include_50_ema
        )
        
        # Calculate EMAs
        periods = (9, 15, 50) if include_50_ema else (9, 15, 21)  # Use 21 as fallback
        ema_values = self.calculate_basic_emas(data, periods)
        
        # If we don't want 50 EMA, set it to None
        if not include_50_ema:
            ema_values.ema_50 = None
        
        # Detect crossover
        is_golden_cross, crossover_strength = self.detect_crossover(ema_values)
        
        # Calculate separation
        separation_pct = self.calculate_ema_separation(ema_values)
        
        # Assess alignment
        alignment = self.assess_ema_alignment(ema_values)
        
        return EMACrossoverData(
            ema_values=ema_values,
            is_golden_cross=is_golden_cross,
            crossover_strength=crossover_strength,
            separation_pct=separation_pct,
            alignment=alignment
        )
    
    def clear_cache(self):
        """Clear calculation cache."""
        self._cache.clear()
        logger.debug("EMA calculator cache cleared")


class EMAAnalyzer:
    """
    High-level EMA analysis orchestrator.
    
    Provides unified interface for all EMA-related analysis needs.
    """
    
    def __init__(self):
        """Initialize EMA analyzer."""
        self.calculator = EMACalculator()
        logger.debug("EMA analyzer initialized")
    
    def analyze_for_strategy(
        self, 
        data: List[OHLCV], 
        strategy_type: str = "basic"
    ) -> Dict[str, any]:
        """
        Perform EMA analysis optimized for specific strategy type.
        
        Args:
            data: OHLCV data points
            strategy_type: Type of strategy ("basic", "v2", "combined")
            
        Returns:
            Dictionary with strategy-specific EMA analysis
        """
        if strategy_type == "v2":
            return self._analyze_for_v2_strategy(data)
        elif strategy_type == "combined":
            return self._analyze_for_combined_strategy(data)
        else:
            return self._analyze_for_basic_strategy(data)
    
    def _analyze_for_basic_strategy(self, data: List[OHLCV]) -> Dict[str, any]:
        """Analyze for basic EMA crossover strategy."""
        crossover_data = self.calculator.create_crossover_analysis(data, include_50_ema=False)
        
        return {
            "ema_crossover": {
                "ema_9": crossover_data.ema_values.ema_9,
                "ema_15": crossover_data.ema_values.ema_15,
                "is_golden_cross": crossover_data.is_golden_cross,
                "crossover_strength": crossover_data.crossover_strength,
            },
            "ema_context": {
                "separation_pct": crossover_data.separation_pct,
                "alignment": crossover_data.alignment,
                "crossover_type": crossover_data.crossover_type
            }
        }
    
    def _analyze_for_v2_strategy(self, data: List[OHLCV]) -> Dict[str, any]:
        """Analyze for enhanced V2 EMA crossover strategy."""
        crossover_data = self.calculator.create_crossover_analysis(data, include_50_ema=True)
        
        return {
            "ema_crossover": {
                "ema_9": crossover_data.ema_values.ema_9,
                "ema_15": crossover_data.ema_values.ema_15,
                "ema_50": crossover_data.ema_values.ema_50,
                "is_golden_cross": crossover_data.is_golden_cross,
                "crossover_strength": crossover_data.crossover_strength,
            },
            "ema_alignment": crossover_data.alignment,
            "ema_context": {
                "separation_pct": crossover_data.separation_pct,
                "alignment": crossover_data.alignment,
                "crossover_type": crossover_data.crossover_type,
                "has_50_ema_filter": crossover_data.ema_values.has_50_ema
            }
        }
    
    def _analyze_for_combined_strategy(self, data: List[OHLCV]) -> Dict[str, any]:
        """Analyze for combined strategy (includes all EMA data)."""
        return self._analyze_for_v2_strategy(data)  # Combined uses most comprehensive analysis