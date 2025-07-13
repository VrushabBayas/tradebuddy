"""
Functionality tests for CLI workflow and user interaction flows.

Focuses on testing end-to-end CLI workflows and user experience rather than implementation details.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

from src.cli.main import CLIApplication
from src.cli.displays import CLIDisplays
from src.core.models import (
    SessionConfig, StrategyType, Symbol, TimeFrame, AnalysisResult, TradingSignal,
    SignalAction, MarketData, OHLCV
)
from rich.console import Console


class TestCLIWorkflowFunctionality:
    """Test CLI workflow functionality from user perspective."""

    @pytest.fixture
    def cli_app(self):
        """Create CLI application instance."""
        return CLIApplication()

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data for testing."""
        ohlcv_data = [
            OHLCV(
                timestamp=datetime.now(timezone.utc),
                open=50000, high=50100, low=49900, close=50050, volume=1000
            )
        ]
        
        return MarketData(
            symbol="BTCUSD",
            timeframe="1h",
            current_price=50050.0,
            ohlcv_data=ohlcv_data
        )

    @pytest.fixture
    def mock_analysis_result(self, mock_market_data):
        """Create mock analysis result."""
        return AnalysisResult(
            symbol=Symbol.BTCUSD,
            timeframe=TimeFrame.ONE_HOUR,
            strategy=StrategyType.EMA_CROSSOVER,
            market_data=mock_market_data,
            signals=[
                TradingSignal(
                    symbol=Symbol.BTCUSD,
                    strategy=StrategyType.EMA_CROSSOVER,
                    action=SignalAction.BUY,
                    strength="STRONG",
                    confidence=8,
                    entry_price=Decimal("50100"),
                    reasoning="Strong bullish signal detected"
                )
            ],
            ai_analysis="Strong bullish momentum with high confidence."
        )

    @pytest.mark.asyncio
    async def test_complete_ema_crossover_workflow(self, cli_app, mock_analysis_result):
        """Test complete EMA crossover analysis workflow from start to finish."""
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', side_effect=['2', '1', '4', '100000', '50', '2.0', '5.0', '10']), \
             patch('rich.prompt.Confirm.ask', return_value=False), \
             patch.object(cli_app.strategies[StrategyType.EMA_CROSSOVER], 'analyze', return_value=mock_analysis_result), \
             patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
             patch.object(cli_app, 'display_configuration_summary'), \
             patch.object(cli_app, 'display_analysis_results'), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Verify workflow completed successfully
            cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.assert_called_once()
            cli_app.display_analysis_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_support_resistance_workflow(self, cli_app, mock_analysis_result):
        """Test complete support/resistance analysis workflow."""
        # Modify mock result for support/resistance strategy
        mock_analysis_result.strategy = StrategyType.SUPPORT_RESISTANCE
        mock_analysis_result.signals[0].strategy = StrategyType.SUPPORT_RESISTANCE
        
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', side_effect=['1', '1', '4', '100000', '50', '2.0', '5.0', '10']), \
             patch('rich.prompt.Confirm.ask', return_value=False), \
             patch.object(cli_app.strategies[StrategyType.SUPPORT_RESISTANCE], 'analyze', return_value=mock_analysis_result), \
             patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
             patch.object(cli_app, 'display_configuration_summary'), \
             patch.object(cli_app, 'display_analysis_results'), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Verify workflow completed successfully
            cli_app.strategies[StrategyType.SUPPORT_RESISTANCE].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_combined_strategy_workflow(self, cli_app, mock_analysis_result):
        """Test complete combined strategy workflow."""
        # Modify mock result for combined strategy
        mock_analysis_result.strategy = StrategyType.COMBINED
        mock_analysis_result.signals[0].strategy = StrategyType.COMBINED
        mock_analysis_result.signals[0].confidence = 9  # Higher confidence for combined
        
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', side_effect=['3', '1', '4', '100000', '50', '2.0', '5.0', '10']), \
             patch('rich.prompt.Confirm.ask', return_value=False), \
             patch.object(cli_app.strategies[StrategyType.COMBINED], 'analyze', return_value=mock_analysis_result), \
             patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
             patch.object(cli_app, 'display_configuration_summary'), \
             patch.object(cli_app, 'display_analysis_results'), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Verify combined strategy was executed
            cli_app.strategies[StrategyType.COMBINED].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_realtime_mode_workflow_selection(self, cli_app):
        """Test real-time analysis mode selection and workflow."""
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', return_value='4'), \
             patch.object(cli_app.realtime_analyzer, 'run_session', return_value=None) as mock_realtime, \
             patch('rich.prompt.Confirm.ask', return_value=False), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Should invoke real-time analyzer
            mock_realtime.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_mode_workflow_selection(self, cli_app):
        """Test continuous monitoring mode selection and workflow."""
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', return_value='5'), \
             patch.object(cli_app.realtime_analyzer, 'run_monitoring_session', return_value=None) as mock_monitoring, \
             patch('rich.prompt.Confirm.ask', return_value=False), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Should invoke monitoring session
            mock_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_backtesting_mode_workflow_selection(self, cli_app):
        """Test backtesting mode selection and workflow."""
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', return_value='6'), \
             patch.object(cli_app, 'run_backtesting_session', return_value=None) as mock_backtesting, \
             patch('rich.prompt.Confirm.ask', return_value=False), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Should invoke backtesting session
            mock_backtesting.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_session_continuation_workflow(self, cli_app, mock_analysis_result):
        """Test user choosing to continue for multiple sessions."""
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', side_effect=['2', '1', '4', '100000', '50', '2.0', '5.0', '10'] * 2), \
             patch('rich.prompt.Confirm.ask', side_effect=[True, False]), \
             patch.object(cli_app.strategies[StrategyType.EMA_CROSSOVER], 'analyze', return_value=mock_analysis_result), \
             patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
             patch.object(cli_app, 'display_configuration_summary'), \
             patch.object(cli_app, 'display_analysis_results'), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Should run two analysis cycles
            assert cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.call_count == 2

    @pytest.mark.asyncio
    async def test_symbol_selection_workflow_variations(self, cli_app, mock_analysis_result):
        """Test different symbol selections in workflow."""
        symbols_to_test = ['1', '2', '3', '4', '5']  # BTCUSD, ETHUSD, SOLUSDT, ADAUSDT, DOGEUSDT
        
        for symbol_choice in symbols_to_test:
            with patch.object(cli_app, 'display_welcome'), \
                 patch('rich.prompt.Prompt.ask', side_effect=['2', symbol_choice, '4', '100000', '50', '2.0', '5.0', '10']), \
                 patch('rich.prompt.Confirm.ask', return_value=False), \
                 patch.object(cli_app.strategies[StrategyType.EMA_CROSSOVER], 'analyze', return_value=mock_analysis_result), \
                 patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
                 patch.object(cli_app, 'display_configuration_summary'), \
                 patch.object(cli_app, 'display_analysis_results'), \
                 patch.object(cli_app, 'display_goodbye'):
                
                await cli_app.run()
                
                # Each symbol should trigger analysis
                cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.assert_called()
                cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.reset_mock()

    @pytest.mark.asyncio
    async def test_timeframe_selection_workflow_variations(self, cli_app, mock_analysis_result):
        """Test different timeframe selections in workflow."""
        timeframes_to_test = ['1', '2', '3', '4', '5', '6']  # All available timeframes
        
        for timeframe_choice in timeframes_to_test:
            with patch.object(cli_app, 'display_welcome'), \
                 patch('rich.prompt.Prompt.ask', side_effect=['2', '1', timeframe_choice, '100000', '50', '2.0', '5.0', '10']), \
                 patch('rich.prompt.Confirm.ask', return_value=False), \
                 patch.object(cli_app.strategies[StrategyType.EMA_CROSSOVER], 'analyze', return_value=mock_analysis_result), \
                 patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
                 patch.object(cli_app, 'display_configuration_summary'), \
                 patch.object(cli_app, 'display_analysis_results'), \
                 patch.object(cli_app, 'display_goodbye'):
                
                await cli_app.run()
                
                # Each timeframe should trigger analysis
                cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.assert_called()
                cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.reset_mock()

    @pytest.mark.asyncio
    async def test_risk_parameter_configuration_workflow(self, cli_app, mock_analysis_result):
        """Test risk parameter configuration variations."""
        risk_configs = [
            ['100000', '50', '2.0', '5.0', '10'],  # Conservative
            ['500000', '80', '3.5', '7.5', '20'],  # Aggressive
            ['50000', '25', '1.0', '3.0', '5'],    # Very conservative
        ]
        
        for risk_config in risk_configs:
            with patch.object(cli_app, 'display_welcome'), \
                 patch('rich.prompt.Prompt.ask', side_effect=['2', '1', '4'] + risk_config), \
                 patch('rich.prompt.Confirm.ask', return_value=False), \
                 patch.object(cli_app.strategies[StrategyType.EMA_CROSSOVER], 'analyze', return_value=mock_analysis_result), \
                 patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
                 patch.object(cli_app, 'display_configuration_summary'), \
                 patch.object(cli_app, 'display_analysis_results'), \
                 patch.object(cli_app, 'display_goodbye'):
                
                await cli_app.run()
                
                # Should complete workflow with any valid risk config
                cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.assert_called()
                cli_app.strategies[StrategyType.EMA_CROSSOVER].analyze.reset_mock()

    @pytest.mark.asyncio
    async def test_error_recovery_workflow_functionality(self, cli_app):
        """Test workflow error recovery and continuation."""
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', side_effect=['2', '1', '4', '100000', '50', '2.0', '5.0', '10']), \
             patch('rich.prompt.Confirm.ask', side_effect=[True, False]), \
             patch.object(cli_app.strategies[StrategyType.EMA_CROSSOVER], 'analyze', side_effect=[Exception("Network error"), None]), \
             patch.object(cli_app.delta_client, 'get_market_data', side_effect=[Exception("API error"), None]), \
             patch.object(cli_app, 'ask_continue_after_error', return_value=True), \
             patch.object(cli_app, 'display_configuration_summary'), \
             patch.object(cli_app, 'display_analysis_results'), \
             patch.object(cli_app, 'display_goodbye'), \
             patch.object(cli_app.console, 'print'):
            
            await cli_app.run()
            
            # Should recover from error and continue
            cli_app.ask_continue_after_error.assert_called_once()

    def test_cli_application_component_initialization(self, cli_app):
        """Test that CLI application initializes all necessary components."""
        # Verify core components are initialized
        assert cli_app.console is not None
        assert cli_app.delta_client is not None
        assert cli_app.websocket_client is not None
        assert cli_app.realtime_analyzer is not None
        assert cli_app.displays is not None
        
        # Verify all strategies are available
        assert StrategyType.EMA_CROSSOVER in cli_app.strategies
        assert StrategyType.SUPPORT_RESISTANCE in cli_app.strategies
        assert StrategyType.COMBINED in cli_app.strategies
        
        # Verify strategies are properly initialized
        for strategy in cli_app.strategies.values():
            assert strategy is not None

    @pytest.mark.asyncio
    async def test_graceful_exit_workflow(self, cli_app):
        """Test graceful exit when user selects exit option."""
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', return_value='7'), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Should exit gracefully without running any analysis
            cli_app.display_goodbye.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_configuration_persistence(self, cli_app, mock_analysis_result):
        """Test that session configuration is properly maintained throughout workflow."""
        expected_config = {
            'strategy': StrategyType.EMA_CROSSOVER,
            'symbol': Symbol.BTCUSD,
            'timeframe': TimeFrame.ONE_HOUR,
            'total_capital_inr': Decimal('100000'),
            'trading_capital_pct': Decimal('50'),
            'risk_per_trade_pct': Decimal('2.0'),
            'take_profit_pct': Decimal('5.0'),
            'leverage': 10
        }
        
        captured_config = None
        
        def capture_config(strategy, config):
            nonlocal captured_config
            captured_config = config
            return mock_analysis_result
        
        with patch.object(cli_app, 'display_welcome'), \
             patch('rich.prompt.Prompt.ask', side_effect=['2', '1', '4', '100000', '50', '2.0', '5.0', '10']), \
             patch('rich.prompt.Confirm.ask', return_value=False), \
             patch.object(cli_app.strategies[StrategyType.EMA_CROSSOVER], 'analyze', side_effect=capture_config), \
             patch.object(cli_app.delta_client, 'get_market_data', return_value=mock_analysis_result.market_data), \
             patch.object(cli_app, 'display_configuration_summary'), \
             patch.object(cli_app, 'display_analysis_results'), \
             patch.object(cli_app, 'display_goodbye'):
            
            await cli_app.run()
            
            # Verify configuration was properly created and passed
            assert captured_config is not None
            assert captured_config.strategy == expected_config['strategy']
            assert captured_config.symbol == expected_config['symbol']
            assert captured_config.timeframe == expected_config['timeframe']
            assert captured_config.total_capital_inr == expected_config['total_capital_inr']


class TestCLIDisplaysFunctionality:
    """Test CLI displays functionality."""

    @pytest.fixture
    def console(self):
        """Create console instance."""
        return Console()

    @pytest.fixture
    def displays(self, console):
        """Create CLI displays instance."""
        return CLIDisplays(console)

    def test_displays_initialization(self, displays, console):
        """Test that displays initialize correctly."""
        assert displays.console == console
        assert displays is not None

    def test_welcome_message_display_functionality(self, displays):
        """Test welcome message display functionality."""
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_welcome()
            
            # Should display welcome content
            mock_print.assert_called()
            # Check that meaningful content is displayed
            call_args = mock_print.call_args_list
            assert len(call_args) > 0

    def test_strategy_menu_display_functionality(self, displays):
        """Test strategy selection menu display."""
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_strategy_menu()
            
            # Should display strategy options
            mock_print.assert_called()
            call_content = str(mock_print.call_args_list)
            
            # Should contain strategy options
            assert "support" in call_content.lower() or "resistance" in call_content.lower()
            assert "ema" in call_content.lower() or "crossover" in call_content.lower()
            assert "combined" in call_content.lower()

    def test_symbol_menu_display_functionality(self, displays):
        """Test symbol selection menu display."""
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_symbol_menu()
            
            # Should display symbol options
            mock_print.assert_called()
            call_content = str(mock_print.call_args_list)
            
            # Should contain crypto symbols
            assert "btc" in call_content.lower() or "bitcoin" in call_content.lower()
            assert "eth" in call_content.lower() or "ethereum" in call_content.lower()

    def test_timeframe_menu_display_functionality(self, displays):
        """Test timeframe selection menu display."""
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_timeframe_menu()
            
            # Should display timeframe options
            mock_print.assert_called()
            call_content = str(mock_print.call_args_list)
            
            # Should contain timeframe options
            assert "minute" in call_content.lower() or "hour" in call_content.lower() or "day" in call_content.lower()

    def test_risk_parameters_display_functionality(self, displays):
        """Test risk parameters explanation display."""
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_risk_parameters_help()
            
            # Should display risk parameter information
            mock_print.assert_called()
            call_content = str(mock_print.call_args_list)
            
            # Should mention key risk concepts
            assert "risk" in call_content.lower() or "capital" in call_content.lower()

    def test_analysis_results_display_functionality(self, displays):
        """Test analysis results display functionality."""
        from src.core.models import AnalysisResult, MarketData, OHLCV
        
        # Create mock analysis result
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
            strategy=StrategyType.EMA_CROSSOVER,
            market_data=market_data,
            signals=[],
            ai_analysis="Test analysis"
        )
        
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_analysis_results(result)
            
            # Should display analysis results
            mock_print.assert_called()
            call_content = str(mock_print.call_args_list)
            
            # Should contain key analysis information
            assert "analysis" in call_content.lower() or "result" in call_content.lower()

    def test_error_message_display_functionality(self, displays):
        """Test error message display functionality."""
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_error("Test error message")
            
            # Should display error message
            mock_print.assert_called()
            call_content = str(mock_print.call_args_list)
            
            # Should contain error indication
            assert "error" in call_content.lower() or "test error message" in call_content

    def test_loading_indicator_functionality(self, displays):
        """Test loading indicator functionality."""
        with patch.object(displays.console, 'print') as mock_print:
            displays.show_loading("Loading market data...")
            
            # Should display loading message
            mock_print.assert_called()
            call_content = str(mock_print.call_args_list)
            
            # Should indicate loading state
            assert "loading" in call_content.lower() or "market data" in call_content.lower()