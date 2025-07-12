"""
Unit tests for CLI components.

Tests the command-line interface functionality and user interactions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from rich.prompt import Confirm, Prompt

from src.cli.main import CLIApplication
from src.core.exceptions import CLIError
from src.core.models import SessionConfig, StrategyType, Symbol, TimeFrame


class TestCLIApplication:
    """Test CLIApplication class."""

    def test_cli_app_initialization(self):
        """Test CLI application initialization."""
        app = CLIApplication()

        assert app.console is not None
        assert app.running is True
        assert isinstance(app.console, Console)

    @pytest.mark.asyncio
    async def test_select_strategy_support_resistance(self):
        """Test strategy selection - Support & Resistance."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="1"):
            strategy = await app.select_strategy()

            assert strategy == StrategyType.SUPPORT_RESISTANCE

    @pytest.mark.asyncio
    async def test_select_strategy_ema_crossover(self):
        """Test strategy selection - EMA Crossover."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="2"):
            strategy = await app.select_strategy()

            assert strategy == StrategyType.EMA_CROSSOVER

    @pytest.mark.asyncio
    async def test_select_strategy_combined(self):
        """Test strategy selection - Combined."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="3"):
            strategy = await app.select_strategy()

            assert strategy == StrategyType.COMBINED

    @pytest.mark.asyncio
    async def test_select_strategy_exit(self):
        """Test strategy selection - Exit."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="4"):
            strategy = await app.select_strategy()

            assert strategy is None

    @pytest.mark.asyncio
    async def test_select_symbol_btc(self):
        """Test symbol selection - Bitcoin."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="1"):
            symbol = await app.select_symbol()

            assert symbol == Symbol.BTCUSDT

    @pytest.mark.asyncio
    async def test_select_symbol_eth(self):
        """Test symbol selection - Ethereum."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="2"):
            symbol = await app.select_symbol()

            assert symbol == Symbol.ETHUSDT

    @pytest.mark.asyncio
    async def test_select_timeframe_1h(self):
        """Test timeframe selection - 1 hour."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="4"):
            timeframe = await app.select_timeframe()

            assert timeframe == TimeFrame.ONE_HOUR

    @pytest.mark.asyncio
    async def test_select_timeframe_1d(self):
        """Test timeframe selection - 1 day."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", return_value="6"):
            timeframe = await app.select_timeframe()

            assert timeframe == TimeFrame.ONE_DAY

    @pytest.mark.asyncio
    async def test_configure_risk_parameters_valid(self):
        """Test risk parameters configuration with valid inputs."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", side_effect=["2.5", "5.0", "2.0"]):
            risk_params = await app.configure_risk_parameters()

            assert risk_params["stop_loss"] == 2.5
            assert risk_params["take_profit"] == 5.0
            assert risk_params["position_size"] == 2.0

    @pytest.mark.asyncio
    async def test_configure_risk_parameters_invalid_then_valid(self):
        """Test risk parameters configuration with invalid then valid inputs."""
        app = CLIApplication()

        # First call: invalid values, second call: valid values
        with patch.object(
            Prompt,
            "ask",
            side_effect=[
                "-1.0",
                "5.0",
                "2.0",  # Invalid stop loss
                "2.5",
                "5.0",
                "2.0",  # Valid values
            ],
        ):
            with patch.object(
                app,
                "configure_risk_parameters",
                side_effect=[
                    app.configure_risk_parameters(),  # First call (invalid)
                    {
                        "stop_loss": 2.5,
                        "take_profit": 5.0,
                        "position_size": 2.0,
                    },  # Second call (valid)
                ],
            ):
                risk_params = await app.configure_risk_parameters()

                assert risk_params["stop_loss"] == 2.5
                assert risk_params["take_profit"] == 5.0
                assert risk_params["position_size"] == 2.0

    @pytest.mark.asyncio
    async def test_configure_session_complete(self):
        """Test complete session configuration."""
        app = CLIApplication()

        with patch.object(
            app, "select_symbol", return_value=Symbol.BTCUSDT
        ), patch.object(
            app, "select_timeframe", return_value=TimeFrame.ONE_HOUR
        ), patch.object(
            app,
            "configure_risk_parameters",
            return_value={"stop_loss": 2.5, "take_profit": 5.0, "position_size": 2.0},
        ), patch.object(
            app, "display_configuration_summary"
        ):
            config = await app.configure_session(StrategyType.COMBINED)

            assert isinstance(config, SessionConfig)
            assert config.strategy == StrategyType.COMBINED
            assert config.symbol == Symbol.BTCUSDT
            assert config.timeframe == TimeFrame.ONE_HOUR
            assert config.stop_loss_pct == 2.5
            assert config.take_profit_pct == 5.0
            assert config.position_size_pct == 2.0

    @pytest.mark.asyncio
    async def test_ask_continue_yes(self):
        """Test ask continue - yes."""
        app = CLIApplication()

        with patch.object(Confirm, "ask", return_value=True):
            result = await app.ask_continue()

            assert result is True

    @pytest.mark.asyncio
    async def test_ask_continue_no(self):
        """Test ask continue - no."""
        app = CLIApplication()

        with patch.object(Confirm, "ask", return_value=False):
            result = await app.ask_continue()

            assert result is False

    @pytest.mark.asyncio
    async def test_ask_continue_after_error_yes(self):
        """Test ask continue after error - yes."""
        app = CLIApplication()

        with patch.object(Confirm, "ask", return_value=True):
            result = await app.ask_continue_after_error()

            assert result is True

    @pytest.mark.asyncio
    async def test_ask_continue_after_error_no(self):
        """Test ask continue after error - no."""
        app = CLIApplication()

        with patch.object(Confirm, "ask", return_value=False):
            result = await app.ask_continue_after_error()

            assert result is False

    def test_display_welcome(self):
        """Test welcome message display."""
        app = CLIApplication()

        with patch.object(app.console, "print") as mock_print:
            app.display_welcome()

            mock_print.assert_called()
            # Check that welcome content is displayed
            call_args = mock_print.call_args[0][0]
            assert hasattr(call_args, "title") or hasattr(call_args, "renderable")

    def test_display_goodbye(self):
        """Test goodbye message display."""
        app = CLIApplication()

        with patch.object(app.console, "print") as mock_print:
            app.display_goodbye()

            mock_print.assert_called()
            # Check that goodbye content is displayed
            call_args = mock_print.call_args[0][0]
            assert hasattr(call_args, "title") or hasattr(call_args, "renderable")

    def test_display_configuration_summary(self, sample_session_config):
        """Test configuration summary display."""
        app = CLIApplication()

        with patch.object(app.console, "print") as mock_print:
            app.display_configuration_summary(sample_session_config)

            mock_print.assert_called()

    @pytest.mark.asyncio
    async def test_run_analysis_session_placeholder(self, sample_session_config):
        """Test analysis session (placeholder implementation)."""
        app = CLIApplication()

        with patch.object(app.console, "print") as mock_print, patch.object(
            Prompt, "ask", return_value=""
        ):
            await app.run_analysis_session(StrategyType.COMBINED, sample_session_config)

            mock_print.assert_called()


class TestCLIApplicationFlow:
    """Test CLI application flow and interactions."""

    @pytest.mark.asyncio
    async def test_full_session_flow_exit_immediately(self):
        """Test full session flow with immediate exit."""
        app = CLIApplication()

        with patch.object(app, "display_welcome"), patch.object(
            app, "select_strategy", return_value=None
        ), patch.object(app, "display_goodbye"):
            await app.run()

            # Should exit after strategy selection returns None
            assert app.running is True  # App doesn't change this flag, just exits loop

    @pytest.mark.asyncio
    async def test_full_session_flow_single_analysis(self):
        """Test full session flow with single analysis."""
        app = CLIApplication()

        with patch.object(app, "display_welcome"), patch.object(
            app, "select_strategy", return_value=StrategyType.COMBINED
        ), patch.object(
            app, "configure_session", return_value=MagicMock()
        ), patch.object(
            app, "run_analysis_session"
        ), patch.object(
            app, "ask_continue", return_value=False
        ), patch.object(
            app, "display_goodbye"
        ):
            await app.run()

            # Should complete one full cycle
            app.select_strategy.assert_called_once()
            app.configure_session.assert_called_once()
            app.run_analysis_session.assert_called_once()
            app.ask_continue.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_session_flow_multiple_analyses(self):
        """Test full session flow with multiple analyses."""
        app = CLIApplication()

        with patch.object(app, "display_welcome"), patch.object(
            app, "select_strategy", return_value=StrategyType.COMBINED
        ), patch.object(
            app, "configure_session", return_value=MagicMock()
        ), patch.object(
            app, "run_analysis_session"
        ), patch.object(
            app, "ask_continue", side_effect=[True, False]
        ), patch.object(
            app, "display_goodbye"
        ):
            await app.run()

            # Should complete two cycles
            assert app.select_strategy.call_count == 2
            assert app.configure_session.call_count == 2
            assert app.run_analysis_session.call_count == 2
            assert app.ask_continue.call_count == 2

    @pytest.mark.asyncio
    async def test_session_flow_with_keyboard_interrupt(self):
        """Test session flow with keyboard interrupt."""
        app = CLIApplication()

        with patch.object(app, "display_welcome"), patch.object(
            app, "select_strategy", side_effect=KeyboardInterrupt
        ), patch.object(app.console, "print") as mock_print:
            await app.run()

            # Should handle keyboard interrupt gracefully
            mock_print.assert_called()
            # Check that interruption message is displayed
            call_args_list = [call[0][0] for call in mock_print.call_args_list]
            assert any("interrupted" in str(arg).lower() for arg in call_args_list)

    @pytest.mark.asyncio
    async def test_session_flow_with_exception_continue(self):
        """Test session flow with exception and user chooses to continue."""
        app = CLIApplication()

        with patch.object(app, "display_welcome"), patch.object(
            app,
            "select_strategy",
            side_effect=[Exception("Test error"), StrategyType.COMBINED],
        ), patch.object(
            app, "configure_session", return_value=MagicMock()
        ), patch.object(
            app, "run_analysis_session"
        ), patch.object(
            app, "ask_continue_after_error", return_value=True
        ), patch.object(
            app, "ask_continue", return_value=False
        ), patch.object(
            app, "display_goodbye"
        ), patch.object(
            app.console, "print"
        ):
            await app.run()

            # Should handle exception and continue
            assert app.select_strategy.call_count == 2
            app.ask_continue_after_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_flow_with_exception_exit(self):
        """Test session flow with exception and user chooses to exit."""
        app = CLIApplication()

        with patch.object(app, "display_welcome"), patch.object(
            app, "select_strategy", side_effect=Exception("Test error")
        ), patch.object(
            app, "ask_continue_after_error", return_value=False
        ), patch.object(
            app, "display_goodbye"
        ), patch.object(
            app.console, "print"
        ):
            await app.run()

            # Should handle exception and exit
            app.select_strategy.assert_called_once()
            app.ask_continue_after_error.assert_called_once()


class TestCLIInputValidation:
    """Test CLI input validation and error handling."""

    @pytest.mark.asyncio
    async def test_risk_parameters_validation_negative_stop_loss(self):
        """Test risk parameters validation with negative stop loss."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", side_effect=["-1.0", "5.0", "2.0"]):
            with pytest.raises(ValueError, match="Stop loss must be between"):
                await app.configure_risk_parameters()

    @pytest.mark.asyncio
    async def test_risk_parameters_validation_excessive_stop_loss(self):
        """Test risk parameters validation with excessive stop loss."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", side_effect=["25.0", "5.0", "2.0"]):
            with pytest.raises(ValueError, match="Stop loss must be between"):
                await app.configure_risk_parameters()

    @pytest.mark.asyncio
    async def test_risk_parameters_validation_negative_take_profit(self):
        """Test risk parameters validation with negative take profit."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", side_effect=["2.5", "-5.0", "2.0"]):
            with pytest.raises(ValueError, match="Take profit must be between"):
                await app.configure_risk_parameters()

    @pytest.mark.asyncio
    async def test_risk_parameters_validation_excessive_take_profit(self):
        """Test risk parameters validation with excessive take profit."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", side_effect=["2.5", "75.0", "2.0"]):
            with pytest.raises(ValueError, match="Take profit must be between"):
                await app.configure_risk_parameters()

    @pytest.mark.asyncio
    async def test_risk_parameters_validation_negative_position_size(self):
        """Test risk parameters validation with negative position size."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", side_effect=["2.5", "5.0", "-1.0"]):
            with pytest.raises(ValueError, match="Position size must be between"):
                await app.configure_risk_parameters()

    @pytest.mark.asyncio
    async def test_risk_parameters_validation_excessive_position_size(self):
        """Test risk parameters validation with excessive position size."""
        app = CLIApplication()

        with patch.object(Prompt, "ask", side_effect=["2.5", "5.0", "15.0"]):
            with pytest.raises(ValueError, match="Position size must be between"):
                await app.configure_risk_parameters()


class TestCLIMainFunction:
    """Test CLI main function and click command."""

    @pytest.mark.asyncio
    async def test_main_function(self):
        """Test main function execution."""
        from src.cli.main import main

        with patch.object(CLIApplication, "run") as mock_run:
            await main()

            mock_run.assert_called_once()

    def test_cli_command_default_args(self):
        """Test CLI command with default arguments."""
        from click.testing import CliRunner

        from src.cli.main import cli

        runner = CliRunner()

        with patch("asyncio.run") as mock_run:
            result = runner.invoke(cli, [])

            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_cli_command_with_env_arg(self):
        """Test CLI command with environment argument."""
        from click.testing import CliRunner

        from src.cli.main import cli

        runner = CliRunner()

        with patch("asyncio.run") as mock_run, patch("os.environ", {}) as mock_env:
            result = runner.invoke(cli, ["--env", "production"])

            assert result.exit_code == 0
            assert mock_env["PYTHON_ENV"] == "production"
            mock_run.assert_called_once()

    def test_cli_command_with_debug_flag(self):
        """Test CLI command with debug flag."""
        from click.testing import CliRunner

        from src.cli.main import cli

        runner = CliRunner()

        with patch("asyncio.run") as mock_run, patch("os.environ", {}) as mock_env:
            result = runner.invoke(cli, ["--debug"])

            assert result.exit_code == 0
            assert mock_env["DEBUG"] == "true"
            mock_run.assert_called_once()


class TestCLIDisplayMethods:
    """Test CLI display methods and formatting."""

    def test_display_methods_call_console_print(self):
        """Test that all display methods call console.print."""
        app = CLIApplication()

        methods_to_test = [
            ("display_welcome", []),
            ("display_goodbye", []),
        ]

        for method_name, args in methods_to_test:
            with patch.object(app.console, "print") as mock_print:
                method = getattr(app, method_name)
                method(*args)

                mock_print.assert_called()

    def test_display_configuration_summary_table_format(self, sample_session_config):
        """Test configuration summary displays as table."""
        app = CLIApplication()

        with patch.object(app.console, "print") as mock_print:
            app.display_configuration_summary(sample_session_config)

            mock_print.assert_called()
            # Should display table with configuration
            call_args = mock_print.call_args_list
            assert len(call_args) >= 2  # Title and table


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    @pytest.mark.asyncio
    async def test_run_handles_general_exception_development_mode(self, test_settings):
        """Test run method handles general exception in development mode."""
        app = CLIApplication()

        with patch.object(app, "display_welcome"), patch.object(
            app, "select_strategy", side_effect=Exception("Test error")
        ), patch.object(
            app, "ask_continue_after_error", return_value=False
        ), patch.object(
            app, "display_goodbye"
        ), patch.object(
            app.console, "print"
        ) as mock_print, patch(
            "src.cli.main.settings", test_settings
        ):
            await app.run()

            # Should print error and traceback in development mode
            mock_print.assert_called()
            error_messages = [str(call[0][0]) for call in mock_print.call_args_list]
            assert any("error" in msg.lower() for msg in error_messages)

    @pytest.mark.asyncio
    async def test_run_handles_general_exception_production_mode(self):
        """Test run method handles general exception in production mode."""
        app = CLIApplication()

        # Mock production settings
        mock_settings = MagicMock()
        mock_settings.is_development = False

        with patch.object(app, "display_welcome"), patch.object(
            app, "select_strategy", side_effect=Exception("Test error")
        ), patch.object(
            app, "ask_continue_after_error", return_value=False
        ), patch.object(
            app, "display_goodbye"
        ), patch.object(
            app.console, "print"
        ) as mock_print, patch(
            "src.cli.main.settings", mock_settings
        ):
            await app.run()

            # Should print error but no traceback in production mode
            mock_print.assert_called()
            error_messages = [str(call[0][0]) for call in mock_print.call_args_list]
            assert any("error" in msg.lower() for msg in error_messages)
            # Should not contain traceback information
            assert not any("traceback" in msg.lower() for msg in error_messages)
