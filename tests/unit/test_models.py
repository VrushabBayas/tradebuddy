"""
Unit tests for data models.

Tests Pydantic models for data validation and serialization.
"""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.core.models import (
    OHLCV,
    AnalysisResult,
    EMACrossover,
    MarketData,
    SessionConfig,
    SessionResults,
    SignalAction,
    SignalStrength,
    StrategyType,
    SupportResistanceLevel,
    Symbol,
    TimeFrame,
    TradingSignal,
)


class TestOHLCV:
    """Test OHLCV model."""

    def test_valid_ohlcv_creation(self):
        """Test creating valid OHLCV instance."""
        ohlcv = OHLCV(
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49500.00"),
            close=Decimal("50500.00"),
            volume=Decimal("1000.50"),
        )

        assert ohlcv.open == Decimal("50000.00")
        assert ohlcv.high == Decimal("51000.00")
        assert ohlcv.low == Decimal("49500.00")
        assert ohlcv.close == Decimal("50500.00")
        assert ohlcv.volume == Decimal("1000.50")
        assert isinstance(ohlcv.timestamp, datetime)

    def test_ohlcv_positive_values_validation(self):
        """Test that OHLCV values must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            OHLCV(
                open=Decimal("-50000.00"),
                high=Decimal("51000.00"),
                low=Decimal("49500.00"),
                close=Decimal("50500.00"),
                volume=Decimal("1000.50"),
            )

        assert "OHLCV values must be positive" in str(exc_info.value)

    def test_ohlcv_high_validation(self):
        """Test high value validation."""
        # High cannot be less than open
        with pytest.raises(ValidationError) as exc_info:
            OHLCV(
                open=Decimal("50000.00"),
                high=Decimal("49000.00"),  # Less than open
                low=Decimal("49500.00"),
                close=Decimal("50500.00"),
                volume=Decimal("1000.50"),
            )

        assert "High cannot be less than open" in str(exc_info.value)

    def test_ohlcv_low_validation(self):
        """Test low value validation."""
        # Low cannot be greater than open
        with pytest.raises(ValidationError) as exc_info:
            OHLCV(
                open=Decimal("50000.00"),
                high=Decimal("51000.00"),
                low=Decimal("50500.00"),  # Greater than open
                close=Decimal("50500.00"),
                volume=Decimal("1000.50"),
            )

        assert "Low cannot be greater than open" in str(exc_info.value)

    def test_ohlcv_serialization(self):
        """Test OHLCV serialization."""
        ohlcv = OHLCV(
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49500.00"),
            close=Decimal("50500.00"),
            volume=Decimal("1000.50"),
        )

        data = ohlcv.dict()

        assert data["open"] == Decimal("50000.00")
        assert data["high"] == Decimal("51000.00")
        assert data["low"] == Decimal("49500.00")
        assert data["close"] == Decimal("50500.00")
        assert data["volume"] == Decimal("1000.50")
        assert "timestamp" in data


class TestMarketData:
    """Test MarketData model."""

    def test_valid_market_data_creation(self, sample_ohlcv):
        """Test creating valid MarketData instance."""
        market_data = MarketData(
            symbol=Symbol.BTCUSDT, timeframe=TimeFrame.ONE_HOUR, ohlcv=sample_ohlcv
        )

        assert market_data.symbol == Symbol.BTCUSDT
        assert market_data.timeframe == TimeFrame.ONE_HOUR
        assert market_data.ohlcv == sample_ohlcv
        assert isinstance(market_data.timestamp, datetime)

    def test_market_data_price_property(self, sample_ohlcv):
        """Test price property returns close price."""
        market_data = MarketData(
            symbol=Symbol.BTCUSDT, timeframe=TimeFrame.ONE_HOUR, ohlcv=sample_ohlcv
        )

        assert market_data.price == sample_ohlcv.close

    def test_market_data_serialization(self, sample_ohlcv):
        """Test MarketData serialization."""
        market_data = MarketData(
            symbol=Symbol.BTCUSDT, timeframe=TimeFrame.ONE_HOUR, ohlcv=sample_ohlcv
        )

        data = market_data.dict()

        assert data["symbol"] == Symbol.BTCUSDT
        assert data["timeframe"] == TimeFrame.ONE_HOUR
        assert "ohlcv" in data
        assert "timestamp" in data


class TestTradingSignal:
    """Test TradingSignal model."""

    def test_valid_trading_signal_creation(self):
        """Test creating valid TradingSignal instance."""
        signal = TradingSignal(
            symbol=Symbol.BTCUSDT,
            strategy=StrategyType.COMBINED,
            action=SignalAction.BUY,
            strength=SignalStrength.STRONG,
            confidence=8,
            entry_price=Decimal("50500.00"),
            stop_loss=Decimal("49500.00"),
            take_profit=Decimal("52500.00"),
            reasoning="Strong support level with volume confirmation",
            risk_reward_ratio=Decimal("2.0"),
            position_size_pct=Decimal("3.0"),
        )

        assert signal.symbol == Symbol.BTCUSDT
        assert signal.strategy == StrategyType.COMBINED
        assert signal.action == SignalAction.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence == 8
        assert signal.entry_price == Decimal("50500.00")
        assert signal.stop_loss == Decimal("49500.00")
        assert signal.take_profit == Decimal("52500.00")
        assert signal.reasoning == "Strong support level with volume confirmation"
        assert signal.risk_reward_ratio == Decimal("2.0")
        assert signal.position_size_pct == Decimal("3.0")

    def test_trading_signal_confidence_validation(self):
        """Test confidence validation."""
        # Valid confidence
        signal = TradingSignal(
            symbol=Symbol.BTCUSDT,
            strategy=StrategyType.COMBINED,
            action=SignalAction.BUY,
            strength=SignalStrength.STRONG,
            confidence=8,
            entry_price=Decimal("50500.00"),
            reasoning="Test signal",
        )
        assert signal.confidence == 8

        # Invalid confidence (too low)
        with pytest.raises(ValidationError) as exc_info:
            TradingSignal(
                symbol=Symbol.BTCUSDT,
                strategy=StrategyType.COMBINED,
                action=SignalAction.BUY,
                strength=SignalStrength.STRONG,
                confidence=0,
                entry_price=Decimal("50500.00"),
                reasoning="Test signal",
            )

        assert "Confidence must be between 1 and 10" in str(exc_info.value)

        # Invalid confidence (too high)
        with pytest.raises(ValidationError) as exc_info:
            TradingSignal(
                symbol=Symbol.BTCUSDT,
                strategy=StrategyType.COMBINED,
                action=SignalAction.BUY,
                strength=SignalStrength.STRONG,
                confidence=15,
                entry_price=Decimal("50500.00"),
                reasoning="Test signal",
            )

        assert "Confidence must be between 1 and 10" in str(exc_info.value)

    def test_trading_signal_is_actionable_property(self):
        """Test is_actionable property."""
        # High confidence signal (actionable)
        signal = TradingSignal(
            symbol=Symbol.BTCUSDT,
            strategy=StrategyType.COMBINED,
            action=SignalAction.BUY,
            strength=SignalStrength.STRONG,
            confidence=8,
            entry_price=Decimal("50500.00"),
            reasoning="Test signal",
        )
        assert signal.is_actionable is True

        # Low confidence signal (not actionable)
        signal = TradingSignal(
            symbol=Symbol.BTCUSDT,
            strategy=StrategyType.COMBINED,
            action=SignalAction.BUY,
            strength=SignalStrength.WEAK,
            confidence=4,
            entry_price=Decimal("50500.00"),
            reasoning="Test signal",
        )
        assert signal.is_actionable is False

    def test_trading_signal_with_indicators(self):
        """Test trading signal with supporting indicators."""
        from src.core.models import TechnicalIndicator

        indicators = [
            TechnicalIndicator(name="RSI", value=Decimal("65.5")),
            TechnicalIndicator(
                name="MACD", value={"macd": Decimal("1.2"), "signal": Decimal("0.8")}
            ),
        ]

        signal = TradingSignal(
            symbol=Symbol.BTCUSDT,
            strategy=StrategyType.COMBINED,
            action=SignalAction.BUY,
            strength=SignalStrength.STRONG,
            confidence=8,
            entry_price=Decimal("50500.00"),
            reasoning="Test signal",
            supporting_indicators=indicators,
        )

        assert len(signal.supporting_indicators) == 2
        assert signal.supporting_indicators[0].name == "RSI"
        assert signal.supporting_indicators[1].name == "MACD"


class TestSessionConfig:
    """Test SessionConfig model."""

    def test_valid_session_config_creation(self):
        """Test creating valid SessionConfig instance."""
        config = SessionConfig(
            strategy=StrategyType.COMBINED,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
            stop_loss_pct=Decimal("2.5"),
            take_profit_pct=Decimal("5.0"),
            position_size_pct=Decimal("2.0"),
            confidence_threshold=6,
            max_signals_per_session=10,
        )

        assert config.strategy == StrategyType.COMBINED
        assert config.symbol == Symbol.BTCUSDT
        assert config.timeframe == TimeFrame.ONE_HOUR
        assert config.stop_loss_pct == Decimal("2.5")
        assert config.take_profit_pct == Decimal("5.0")
        assert config.position_size_pct == Decimal("2.0")
        assert config.confidence_threshold == 6
        assert config.max_signals_per_session == 10

    def test_session_config_defaults(self):
        """Test SessionConfig default values."""
        config = SessionConfig(
            strategy=StrategyType.COMBINED,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
        )

        assert config.stop_loss_pct == Decimal("2.5")
        assert config.take_profit_pct == Decimal("5.0")
        assert config.position_size_pct == Decimal("2.0")
        assert config.confidence_threshold == 6
        assert config.max_signals_per_session == 10

    def test_session_config_serialization(self):
        """Test SessionConfig serialization."""
        config = SessionConfig(
            strategy=StrategyType.COMBINED,
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
        )

        data = config.dict()

        # Should use enum values
        assert data["strategy"] == "combined"
        assert data["symbol"] == "BTCUSDT"
        assert data["timeframe"] == "1h"


class TestSupportResistanceLevel:
    """Test SupportResistanceLevel model."""

    def test_valid_support_resistance_level(self):
        """Test creating valid SupportResistanceLevel instance."""
        level = SupportResistanceLevel(
            level=Decimal("50000.00"),
            strength=8,
            is_support=True,
            touches=3,
            last_touch=datetime(2024, 1, 1, 12, 0, 0),
        )

        assert level.level == Decimal("50000.00")
        assert level.strength == 8
        assert level.is_support is True
        assert level.touches == 3
        assert level.last_touch == datetime(2024, 1, 1, 12, 0, 0)

    def test_support_resistance_level_strength_validation(self):
        """Test strength validation."""
        # Valid strength
        level = SupportResistanceLevel(
            level=Decimal("50000.00"), strength=5, is_support=True
        )
        assert level.strength == 5

        # Invalid strength (too low)
        with pytest.raises(ValidationError):
            SupportResistanceLevel(
                level=Decimal("50000.00"), strength=0, is_support=True
            )

        # Invalid strength (too high)
        with pytest.raises(ValidationError):
            SupportResistanceLevel(
                level=Decimal("50000.00"), strength=15, is_support=True
            )


class TestEMACrossover:
    """Test EMACrossover model."""

    def test_valid_ema_crossover(self):
        """Test creating valid EMACrossover instance."""
        crossover = EMACrossover(
            ema_9=Decimal("50450.00"),
            ema_15=Decimal("50400.00"),
            is_golden_cross=True,
            crossover_strength=7,
        )

        assert crossover.ema_9 == Decimal("50450.00")
        assert crossover.ema_15 == Decimal("50400.00")
        assert crossover.is_golden_cross is True
        assert crossover.crossover_strength == 7
        assert isinstance(crossover.timestamp, datetime)

    def test_ema_crossover_strength_validation(self):
        """Test crossover strength validation."""
        # Valid strength
        crossover = EMACrossover(
            ema_9=Decimal("50450.00"),
            ema_15=Decimal("50400.00"),
            is_golden_cross=True,
            crossover_strength=5,
        )
        assert crossover.crossover_strength == 5

        # Invalid strength
        with pytest.raises(ValidationError):
            EMACrossover(
                ema_9=Decimal("50450.00"),
                ema_15=Decimal("50400.00"),
                is_golden_cross=True,
                crossover_strength=0,
            )


class TestAnalysisResult:
    """Test AnalysisResult model."""

    def test_valid_analysis_result(self, sample_market_data, sample_trading_signal):
        """Test creating valid AnalysisResult instance."""
        result = AnalysisResult(
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
            strategy=StrategyType.COMBINED,
            market_data=sample_market_data,
            signals=[sample_trading_signal],
            ai_analysis="Strong bullish signal detected",
            execution_time=2.5,
        )

        assert result.symbol == Symbol.BTCUSDT
        assert result.timeframe == TimeFrame.ONE_HOUR
        assert result.strategy == StrategyType.COMBINED
        assert result.market_data == sample_market_data
        assert len(result.signals) == 1
        assert result.signals[0] == sample_trading_signal
        assert result.ai_analysis == "Strong bullish signal detected"
        assert result.execution_time == 2.5

    def test_analysis_result_primary_signal_property(self, sample_market_data):
        """Test primary_signal property."""
        # Multiple signals with different confidence
        signals = [
            TradingSignal(
                symbol=Symbol.BTCUSDT,
                strategy=StrategyType.COMBINED,
                action=SignalAction.BUY,
                strength=SignalStrength.WEAK,
                confidence=5,
                entry_price=Decimal("50500.00"),
                reasoning="Weak signal",
            ),
            TradingSignal(
                symbol=Symbol.BTCUSDT,
                strategy=StrategyType.COMBINED,
                action=SignalAction.BUY,
                strength=SignalStrength.STRONG,
                confidence=9,
                entry_price=Decimal("50500.00"),
                reasoning="Strong signal",
            ),
            TradingSignal(
                symbol=Symbol.BTCUSDT,
                strategy=StrategyType.COMBINED,
                action=SignalAction.SELL,
                strength=SignalStrength.MODERATE,
                confidence=7,
                entry_price=Decimal("50500.00"),
                reasoning="Moderate signal",
            ),
        ]

        result = AnalysisResult(
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
            strategy=StrategyType.COMBINED,
            market_data=sample_market_data,
            signals=signals,
            ai_analysis="Mixed signals",
        )

        # Should return the highest confidence signal
        primary_signal = result.primary_signal
        assert primary_signal is not None
        assert primary_signal.confidence == 9
        assert primary_signal.reasoning == "Strong signal"

    def test_analysis_result_no_signals(self, sample_market_data):
        """Test primary_signal property with no signals."""
        result = AnalysisResult(
            symbol=Symbol.BTCUSDT,
            timeframe=TimeFrame.ONE_HOUR,
            strategy=StrategyType.COMBINED,
            market_data=sample_market_data,
            signals=[],
            ai_analysis="No signals detected",
        )

        assert result.primary_signal is None


class TestSessionResults:
    """Test SessionResults model."""

    def test_valid_session_results(self, sample_session_config):
        """Test creating valid SessionResults instance."""
        results = SessionResults(
            config=sample_session_config,
            duration=3600.0,
            total_signals=10,
            buy_signals=6,
            sell_signals=3,
            neutral_signals=1,
            avg_confidence=7.5,
            max_confidence=9,
            min_confidence=5,
        )

        assert results.config == sample_session_config
        assert results.duration == 3600.0
        assert results.total_signals == 10
        assert results.buy_signals == 6
        assert results.sell_signals == 3
        assert results.neutral_signals == 1
        assert results.avg_confidence == 7.5
        assert results.max_confidence == 9
        assert results.min_confidence == 5

    def test_session_results_success_rate_property(
        self, sample_session_config, sample_market_data
    ):
        """Test success_rate property calculation."""
        # Create analysis results with actionable signals
        analysis_results = [
            AnalysisResult(
                symbol=Symbol.BTCUSDT,
                timeframe=TimeFrame.ONE_HOUR,
                strategy=StrategyType.COMBINED,
                market_data=sample_market_data,
                signals=[
                    TradingSignal(
                        symbol=Symbol.BTCUSDT,
                        strategy=StrategyType.COMBINED,
                        action=SignalAction.BUY,
                        strength=SignalStrength.STRONG,
                        confidence=8,  # Actionable
                        entry_price=Decimal("50500.00"),
                        reasoning="Strong signal",
                    ),
                    TradingSignal(
                        symbol=Symbol.BTCUSDT,
                        strategy=StrategyType.COMBINED,
                        action=SignalAction.SELL,
                        strength=SignalStrength.WEAK,
                        confidence=4,  # Not actionable
                        entry_price=Decimal("50500.00"),
                        reasoning="Weak signal",
                    ),
                ],
                ai_analysis="Mixed signals",
            )
        ]

        results = SessionResults(
            config=sample_session_config,
            duration=3600.0,
            total_signals=2,
            analysis_results=analysis_results,
        )

        # Should calculate success rate based on actionable signals
        # 1 actionable out of 2 total = 50%
        assert results.success_rate == 50.0

    def test_session_results_no_signals(self, sample_session_config):
        """Test success_rate property with no signals."""
        results = SessionResults(
            config=sample_session_config, duration=3600.0, total_signals=0
        )

        assert results.success_rate == 0.0
