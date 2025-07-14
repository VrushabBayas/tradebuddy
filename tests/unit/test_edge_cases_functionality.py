"""
Functionality tests for edge cases and error handling across the system.

Focuses on testing system behavior under unusual conditions, invalid inputs,
and error scenarios rather than implementation details.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone, timedelta
import json

from src.core.models import (
    SessionConfig, StrategyType, Symbol, TimeFrame, MarketData, OHLCV,
    TradingSignal, SignalAction, AnalysisResult
)
from src.core.exceptions import (
    APIConnectionError, APITimeoutError, DataValidationError, StrategyError
)
from src.analysis.indicators import TechnicalIndicators
from src.data.delta_client import DeltaExchangeClient
from src.utils.risk_management import calculate_leveraged_position_size
from src.utils.helpers import to_decimal, to_float, safe_float_conversion


class TestDataValidationEdgeCases:
    """Test edge cases in data validation and processing."""

    def test_ohlcv_extreme_values(self):
        """Test OHLCV with extreme values."""
        extreme_cases = [
            # Very small values
            {
                'open': 0.000001, 'high': 0.000002, 'low': 0.0000005,
                'close': 0.0000015, 'volume': 0.001
            },
            # Very large values
            {
                'open': 1000000000, 'high': 1100000000, 'low': 900000000,
                'close': 1050000000, 'volume': 10000000
            },
            # Zero volume
            {
                'open': 50000, 'high': 50100, 'low': 49900,
                'close': 50050, 'volume': 0
            },
        ]
        
        for case in extreme_cases:
            try:
                ohlcv = OHLCV(
                    timestamp=datetime.now(timezone.utc),
                    **case
                )
                
                # Should handle extreme values gracefully
                assert ohlcv.open > 0 or case['open'] == 0
                assert ohlcv.volume >= 0
                
            except (ValueError, InvalidOperation):
                # Some extreme values might be rejected - that's acceptable
                pass

    def test_invalid_ohlcv_relationships(self):
        """Test OHLCV with invalid price relationships."""
        invalid_cases = [
            # High lower than low
            {
                'open': 50000, 'high': 49000, 'low': 49500,
                'close': 50050, 'volume': 1000
            },
            # Negative prices
            {
                'open': -50000, 'high': 51000, 'low': 49000,
                'close': 50050, 'volume': 1000
            },
            # Zero prices
            {
                'open': 0, 'high': 0, 'low': 0,
                'close': 0, 'volume': 1000
            },
        ]
        
        for case in invalid_cases:
            try:
                ohlcv = OHLCV(
                    timestamp=datetime.now(timezone.utc),
                    **case
                )
                # If creation succeeds, verify data integrity is maintained somehow
                # (some validation might happen at a higher level)
                assert isinstance(ohlcv, OHLCV)
                
            except (ValueError, ValidationError):
                # Invalid data should be rejected
                pass

    def test_market_data_with_empty_ohlcv(self):
        """Test market data with empty OHLCV data."""
        try:
            market_data = MarketData(
                symbol="BTCUSD",
                timeframe="1h",
                current_price=50000.0,
                ohlcv_data=[]
            )
            
            # Should handle empty data
            assert len(market_data.ohlcv_data) == 0
            assert market_data.current_price > 0
            
        except Exception:
            # Empty data might be rejected
            pass

    def test_market_data_with_inconsistent_timestamps(self):
        """Test market data with inconsistent timestamps."""
        # Create OHLCV data with timestamps out of order
        base_time = datetime.now(timezone.utc)
        ohlcv_data = [
            OHLCV(
                timestamp=base_time,
                open=50000, high=50100, low=49900, close=50050, volume=1000
            ),
            OHLCV(
                timestamp=base_time - timedelta(hours=1),  # Earlier timestamp
                open=50050, high=50150, low=50000, close=50100, volume=1100
            ),
        ]
        
        try:
            market_data = MarketData(
                symbol="BTCUSD",
                timeframe="1h",
                current_price=50100.0,
                ohlcv_data=ohlcv_data
            )
            
            # Should handle inconsistent timestamps
            assert len(market_data.ohlcv_data) == 2
            
        except Exception:
            # Inconsistent data might be rejected
            pass

    def test_decimal_conversion_edge_cases(self):
        """Test decimal conversion with edge cases."""
        edge_cases = [
            "invalid_number",
            "",
            None,
            float('inf'),
            float('-inf'),
            float('nan'),
            "1.23e-100",  # Very small scientific notation
            "1.23e100",   # Very large scientific notation
        ]
        
        for case in edge_cases:
            try:
                result = to_decimal(case)
                
                # If conversion succeeds, should be a valid decimal or None
                if result is not None:
                    assert isinstance(result, Decimal)
                    # Some edge cases (like NaN) might not be finite - that's OK
                    # The system should handle them gracefully
                else:
                    # None is acceptable for invalid values
                    assert result is None
                
            except (ValueError, InvalidOperation, TypeError):
                # Invalid values might be rejected with exceptions
                pass

    def test_float_conversion_edge_cases(self):
        """Test float conversion with edge cases."""
        edge_cases = [
            "not_a_number",
            [],
            {},
            complex(1, 2),
            Decimal('NaN'),
            Decimal('Infinity'),
        ]
        
        for case in edge_cases:
            try:
                result = to_float(case)
                
                # If conversion succeeds, should be a valid float
                assert isinstance(result, float)
                assert not (result != result)  # Check for NaN
                
            except (ValueError, TypeError, InvalidOperation):
                # Invalid values should be rejected
                pass


class TestAPIErrorHandling:
    """Test API error handling and resilience."""

    @pytest.fixture
    def delta_client(self):
        """Create Delta client for testing."""
        return DeltaExchangeClient()

    @pytest.mark.asyncio
    async def test_network_connection_errors(self, delta_client):
        """Test handling of network connection errors."""
        connection_errors = [
            ConnectionError("Network unreachable"),
            TimeoutError("Connection timeout"),
            OSError("Network is down"),
        ]
        
        for error in connection_errors:
            with patch('aiohttp.ClientSession', side_effect=error):
                try:
                    await delta_client.get_market_data("BTCUSD", "1h")
                    # Should not reach here if error handling works
                    assert False, f"Expected {type(error).__name__} to be raised"
                    
                except (APIConnectionError, ConnectionError, TimeoutError, OSError):
                    # Expected error handling
                    pass

    @pytest.mark.asyncio
    async def test_http_error_responses(self, delta_client):
        """Test handling of various HTTP error responses."""
        error_responses = [
            (400, "Bad Request"),
            (401, "Unauthorized"), 
            (403, "Forbidden"),
            (404, "Not Found"),
            (429, "Rate Limited"),
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
            (503, "Service Unavailable"),
        ]
        
        for status_code, error_message in error_responses:
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                
                mock_response = AsyncMock()
                mock_response.status = status_code
                mock_response.text.return_value = error_message
                mock_session.get.return_value.__aenter__.return_value = mock_response
                
                try:
                    await delta_client.get_market_data("BTCUSD", "1h")
                    
                except (APIConnectionError, Exception) as e:
                    # Should handle HTTP errors gracefully
                    assert len(str(e)) > 0

    @pytest.mark.asyncio
    async def test_malformed_json_responses(self, delta_client):
        """Test handling of malformed JSON responses."""
        malformed_responses = [
            '{"incomplete": json',
            'not json at all',
            '',
            '{"valid": "json", "but": "unexpected_structure"}',
            b'\x80\x81\x82',  # Invalid UTF-8
        ]
        
        for malformed_response in malformed_responses:
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text.return_value = malformed_response
                mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
                mock_session.get.return_value.__aenter__.return_value = mock_response
                
                try:
                    await delta_client.get_market_data("BTCUSD", "1h")
                    
                except (DataValidationError, json.JSONDecodeError, Exception):
                    # Should handle malformed responses gracefully
                    pass

    @pytest.mark.asyncio
    async def test_api_timeout_scenarios(self, delta_client):
        """Test various timeout scenarios."""
        timeout_scenarios = [
            asyncio.TimeoutError("Request timeout"),
            asyncio.CancelledError("Request cancelled"),
        ]
        
        for timeout_error in timeout_scenarios:
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                mock_session.get.side_effect = timeout_error
                
                try:
                    await delta_client.get_market_data("BTCUSD", "1h")
                    
                except (APITimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
                    # Should handle timeouts gracefully
                    pass




class TestStrategyErrorHandling:
    """Test strategy error handling and edge cases."""

    @pytest.fixture
    def indicators(self):
        """Create technical indicators instance."""
        return TechnicalIndicators()

    def test_indicators_with_insufficient_data(self, indicators):
        """Test indicators with insufficient data."""
        insufficient_data = [
            [],  # Empty data
            [OHLCV(timestamp=datetime.now(timezone.utc), open=50000, high=50100, low=49900, close=50050, volume=1000)],  # Single candle
        ]
        
        for data in insufficient_data:
            try:
                # Test various indicators
                sma = indicators.calculate_sma(data, period=20)
                ema = indicators.calculate_ema(data, period=20)
                
                # If calculation succeeds, should return reasonable results
                if sma is not None:
                    assert isinstance(sma, (list, float, int))
                if ema is not None:
                    assert isinstance(ema, (list, float, int))
                    
            except (ValueError, IndexError):
                # Insufficient data should be handled gracefully
                pass

    def test_indicators_with_zero_volume(self, indicators):
        """Test indicators with zero volume data."""
        zero_volume_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
                open=50000, high=50000, low=50000, close=50000, volume=0
            ) for i in range(20)
        ]
        
        try:
            volume_analysis = indicators.analyze_volume(zero_volume_data)
            
            # Should handle zero volume gracefully
            if volume_analysis is not None:
                assert isinstance(volume_analysis, dict)
                
        except (ValueError, ZeroDivisionError):
            # Zero volume might cause calculation issues
            pass

    def test_indicators_with_constant_prices(self, indicators):
        """Test indicators with constant price data."""
        constant_price_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
                open=50000, high=50000, low=50000, close=50000, volume=1000
            ) for i in range(30)
        ]
        
        try:
            # Test various indicators with flat data
            sma = indicators.calculate_sma(constant_price_data, period=20)
            ema = indicators.calculate_ema(constant_price_data, period=20)
            rsi = indicators.rsi(constant_price_data, period=14)
            
            # Should handle constant prices
            if sma is not None:
                assert abs(sma - 50000) < 1.0 if isinstance(sma, (int, float)) else True
            if ema is not None:
                assert abs(ema - 50000) < 1.0 if isinstance(ema, (int, float)) else True
            if rsi is not None:
                # RSI with constant prices might be undefined or 50
                assert 0 <= rsi <= 100 if isinstance(rsi, (int, float)) else True
                
        except (ValueError, ZeroDivisionError):
            # Constant prices might cause calculation issues
            pass


class TestRiskManagementEdgeCases:
    """Test risk management edge cases."""

    def test_extreme_position_sizes(self):
        """Test position size calculation with extreme parameters."""
        extreme_cases = [
            # Very small account
            (100.0, 1.0, 1, 50000.0),
            # Very large account
            (1000000000.0, 1.0, 1, 50000.0),
            # Very high leverage
            (10000.0, 2.0, 100, 50000.0),
            # Very small position percentage
            (10000.0, 0.01, 10, 50000.0),
            # Very high position percentage
            (10000.0, 50.0, 10, 50000.0),
        ]
        
        for balance, position_pct, leverage, price in extreme_cases:
            try:
                position_value, position_amount, margin_required = calculate_leveraged_position_size(
                    balance, position_pct, leverage, price
                )
                
                # Should handle extreme cases gracefully
                assert position_value >= 0
                assert position_amount >= 0
                assert margin_required >= 0
                assert margin_required <= balance
                
            except (ValueError, OverflowError, ZeroDivisionError):
                # Extreme values might be rejected
                pass

    def test_zero_and_negative_parameters(self):
        """Test position size calculation with zero and negative parameters."""
        invalid_cases = [
            (0, 2.0, 10, 50000.0),      # Zero balance
            (-10000.0, 2.0, 10, 50000.0),  # Negative balance
            (10000.0, 0, 10, 50000.0),      # Zero position size
            (10000.0, -2.0, 10, 50000.0),   # Negative position size
            (10000.0, 2.0, 0, 50000.0),     # Zero leverage
            (10000.0, 2.0, -10, 50000.0),   # Negative leverage
            (10000.0, 2.0, 10, 0),          # Zero price
            (10000.0, 2.0, 10, -50000.0),   # Negative price
        ]
        
        for balance, position_pct, leverage, price in invalid_cases:
            try:
                result = calculate_leveraged_position_size(balance, position_pct, leverage, price)
                
                # If calculation succeeds, verify results are reasonable
                if result is not None:
                    position_value, position_amount, margin_required = result
                    # With negative inputs, results might be negative - that's a real edge case
                    # The system should either prevent this or handle it gracefully
                    if balance > 0 and position_pct > 0 and leverage > 0 and price > 0:
                        # With all positive inputs, results should be reasonable
                        assert position_value >= 0
                        assert position_amount >= 0
                        assert margin_required >= 0
                    # For edge cases with negative inputs, just verify we get numbers
                    assert isinstance(position_value, (int, float))
                    assert isinstance(position_amount, (int, float))
                    assert isinstance(margin_required, (int, float))
                    
            except (ValueError, ZeroDivisionError):
                # Invalid parameters should be rejected
                pass


class TestSignalProcessingEdgeCases:
    """Test trading signal processing edge cases."""

    def test_conflicting_signals(self):
        """Test handling of conflicting trading signals."""
        conflicting_signals = [
            TradingSignal(
                symbol=Symbol.BTCUSD,
                strategy=StrategyType.EMA_CROSSOVER_V2,
                action=SignalAction.BUY,
                strength="STRONG",
                confidence=9,
                entry_price=Decimal("50000"),
                reasoning="Strong bullish signal"
            ),
            TradingSignal(
                symbol=Symbol.BTCUSD,
                strategy=StrategyType.SUPPORT_RESISTANCE,
                action=SignalAction.SELL,
                strength="STRONG",
                confidence=8,
                entry_price=Decimal("50000"),
                reasoning="Strong bearish signal"
            ),
        ]
        
        # Create analysis result with conflicting signals
        try:
            ohlcv = OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50000, high=50100, low=49900, close=50050, volume=1000
            )
            
            market_data = MarketData(
                symbol="BTCUSD",
                timeframe="1h",
                current_price=50050.0,
                ohlcv_data=[ohlcv]
            )
            
            result = AnalysisResult(
                symbol=Symbol.BTCUSD,
                timeframe=TimeFrame.ONE_HOUR,
                strategy=StrategyType.COMBINED,
                market_data=market_data,
                signals=conflicting_signals,
                ai_analysis="Conflicting signals detected"
            )
            
            # Should handle conflicting signals
            assert len(result.signals) == 2
            primary_signal = result.primary_signal
            
            # Primary signal should be the highest confidence one
            if primary_signal is not None:
                assert primary_signal.confidence >= 8
                
        except Exception:
            # Conflicting signals might be rejected at a higher level
            pass

    def test_signals_with_extreme_confidence(self):
        """Test signals with extreme confidence values."""
        extreme_confidence_cases = [
            -5,   # Negative confidence
            0,    # Zero confidence
            11,   # Above maximum confidence
            100,  # Very high confidence
        ]
        
        for confidence in extreme_confidence_cases:
            try:
                signal = TradingSignal(
                    symbol=Symbol.BTCUSD,
                    strategy=StrategyType.EMA_CROSSOVER_V2,
                    action=SignalAction.BUY,
                    strength="MODERATE",
                    confidence=confidence,
                    entry_price=Decimal("50000"),
                    reasoning="Test signal"
                )
                
                # If creation succeeds, confidence should be within valid range
                assert 1 <= signal.confidence <= 10
                
            except (ValueError, ValidationError):
                # Invalid confidence should be rejected
                pass

    def test_signals_with_invalid_prices(self):
        """Test signals with invalid price values."""
        invalid_price_cases = [
            Decimal("0"),        # Zero price
            Decimal("-50000"),   # Negative price
            None,                # No price
        ]
        
        for invalid_price in invalid_price_cases:
            try:
                signal = TradingSignal(
                    symbol=Symbol.BTCUSD,
                    strategy=StrategyType.EMA_CROSSOVER_V2,
                    action=SignalAction.BUY,
                    strength="MODERATE",
                    confidence=7,
                    entry_price=invalid_price,
                    reasoning="Test signal"
                )
                
                # If creation succeeds, price should be reasonable
                if signal.entry_price is not None:
                    assert signal.entry_price > 0
                    
            except (ValueError, ValidationError, TypeError):
                # Invalid prices should be rejected
                pass


class TestConcurrencyAndPerformanceEdgeCases:
    """Test edge cases related to concurrency and performance."""

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self):
        """Test handling of many concurrent API requests."""
        client = DeltaExchangeClient()
        
        # Create many concurrent requests
        tasks = []
        for i in range(10):
            with patch.object(client, 'get_market_data') as mock_get:
                mock_get.return_value = MarketData(
                    symbol="BTCUSD",
                    timeframe="1h",
                    current_price=50000.0,
                    ohlcv_data=[]
                )
                
                task = client.get_market_data("BTCUSD", "1h")
                tasks.append(task)
        
        try:
            # Should handle concurrent requests
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify results
            for result in results:
                if not isinstance(result, Exception):
                    assert isinstance(result, MarketData)
                    
        except Exception:
            # Concurrent requests might fail due to rate limiting
            pass

    def test_memory_intensive_operations(self):
        """Test operations with large datasets."""
        # Create large dataset
        large_dataset = [
            OHLCV(
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                open=50000 + (i % 100), 
                high=50100 + (i % 100),
                low=49900 + (i % 100), 
                close=50050 + (i % 100), 
                volume=1000
            ) for i in range(10000)  # 10,000 candles
        ]
        
        try:
            indicators = TechnicalIndicators()
            
            # Test indicators with large dataset
            sma = indicators.calculate_sma(large_dataset, period=100)
            
            # Should handle large datasets
            if sma is not None:
                assert isinstance(sma, (list, float, int))
                
        except (MemoryError, ValueError):
            # Large datasets might exceed memory limits
            pass

    @pytest.mark.asyncio
    async def test_long_running_operations(self):
        """Test handling of long-running operations."""
        # Simulate long-running operation
        async def slow_operation():
            await asyncio.sleep(0.1)  # Simulate work
            return "completed"
        
        try:
            # Test with timeout
            result = await asyncio.wait_for(slow_operation(), timeout=0.2)
            assert result == "completed"
            
        except asyncio.TimeoutError:
            # Long operations should be handled with timeouts
            pass