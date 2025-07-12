"""
Main CLI application for TradeBuddy.

Provides interactive terminal interface for trading strategy selection and analysis.
"""

import asyncio
import os
import sys
from typing import Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.analysis.strategies.combined import CombinedStrategy
from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
from src.analysis.strategies.support_resistance import SupportResistanceStrategy
from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.backtesting.reports import BacktestReportGenerator
from src.cli.displays import CLIDisplays
from src.cli.realtime import RealTimeAnalyzer
from src.core.config import settings
from src.core.exceptions import CLIError
from src.core.models import SessionConfig, StrategyType, Symbol, TimeFrame
from src.data.delta_client import DeltaExchangeClient
from src.data.websocket_client import DeltaWebSocketClient

console = Console()
logger = structlog.get_logger(__name__)


class CLIApplication:
    """Main CLI application class."""

    def __init__(self):
        self.console = console
        self.running = True
        self.delta_client = DeltaExchangeClient()
        self.websocket_client = DeltaWebSocketClient()
        self.strategies = {
            StrategyType.SUPPORT_RESISTANCE: SupportResistanceStrategy(),
            StrategyType.EMA_CROSSOVER: EMACrossoverStrategy(),
            StrategyType.COMBINED: CombinedStrategy(),
        }

        # Display utilities
        self.displays = CLIDisplays(self.console)

        # Real-time analyzer
        self.realtime_analyzer = RealTimeAnalyzer(
            console=self.console,
            delta_client=self.delta_client,
            websocket_client=self.websocket_client,
            strategies=self.strategies,
        )

    async def run(self):
        """Run the main CLI application loop."""
        self.display_welcome()

        while self.running:
            try:
                # Strategy selection
                strategy = await self.select_strategy()
                if strategy is None:
                    break

                if strategy == "REALTIME":
                    # Real-time analysis flow
                    await self.realtime_analyzer.run_session()
                elif strategy == "MONITORING":
                    # Continuous monitoring flow
                    await self.realtime_analyzer.run_monitoring_session()
                elif strategy == "BACKTESTING":
                    # Backtesting flow
                    await self.run_backtesting_session()
                else:
                    # Traditional historical analysis flow
                    config = await self.configure_session(strategy)
                    await self.run_analysis_session(strategy, config)

                # Ask to continue
                if not await self.ask_continue():
                    break

            except KeyboardInterrupt:
                self.console.print("\n👋 Session interrupted. Goodbye!", style="cyan")
                break
            except Exception as e:
                self.console.print(f"\n❌ Error: {e}", style="red")

                if settings.is_development:
                    import traceback

                    self.console.print("\n🐛 Traceback:", style="dim")
                    self.console.print(traceback.format_exc(), style="dim")

                if not await self.ask_continue_after_error():
                    break

        self.display_goodbye()

    def display_welcome(self):
        """Display welcome screen."""
        self.displays.display_welcome()

    async def select_strategy(self) -> Optional[StrategyType]:
        """Interactive strategy selection."""
        self.console.print("\n🎯 Trading Strategy Selection", style="bold cyan")

        # Create and display strategy table
        table = self.displays.create_strategy_table()
        self.console.print(table)

        # Get user choice
        choice = Prompt.ask(
            "\nSelect strategy", choices=["1", "2", "3", "4", "5", "6", "7"], default="3"
        )

        strategy_map = {
            "1": StrategyType.SUPPORT_RESISTANCE,
            "2": StrategyType.EMA_CROSSOVER,
            "3": StrategyType.COMBINED,
            "4": "REALTIME",  # Special marker for real-time analysis
            "5": "MONITORING",  # Special marker for monitoring mode
            "6": "BACKTESTING",  # Special marker for backtesting
            "7": None,
        }

        selected_strategy = strategy_map[choice]

        if selected_strategy == "REALTIME":
            self.console.print(f"\n✅ Selected: Real-Time Analysis", style="green")
        elif selected_strategy == "MONITORING":
            self.console.print(f"\n✅ Selected: Market Monitoring", style="green")
        elif selected_strategy == "BACKTESTING":
            self.console.print(f"\n✅ Selected: Strategy Backtesting", style="green")
        elif selected_strategy:
            self.console.print(
                f"\n✅ Selected: {selected_strategy.value.replace('_', ' ').title()}",
                style="green",
            )

        return selected_strategy

    async def configure_session(self, strategy: StrategyType) -> SessionConfig:
        """Configure trading session parameters."""
        self.console.print(f"\n⚙️ Session Configuration", style="bold cyan")

        # Symbol selection
        symbol = await self.select_symbol()

        # Timeframe selection
        timeframe = await self.select_timeframe()

        # Risk parameters
        risk_params = await self.configure_risk_parameters()

        config = SessionConfig(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            stop_loss_pct=risk_params["stop_loss"],
            take_profit_pct=risk_params["take_profit"],
            position_size_pct=risk_params["position_size"],
        )

        # Display configuration summary
        self.display_configuration_summary(config)

        return config

    async def select_symbol(self) -> Symbol:
        """Select trading symbol."""
        self.console.print("\n📊 Symbol Selection", style="bold")

        # Create and display symbol table
        table = self.displays.create_symbol_table()
        self.console.print(table)

        choice = Prompt.ask(
            "\nSelect symbol", choices=["1", "2", "3", "4", "5"], default="1"
        )

        symbol_map = {
            "1": Symbol.BTCUSDT,
            "2": Symbol.ETHUSDT,
            "3": Symbol.SOLUSDT,
            "4": Symbol.ADAUSDT,
            "5": Symbol.DOGEUSDT,
        }

        selected_symbol = symbol_map[choice]
        self.console.print(f"✅ Selected: {selected_symbol.value}", style="green")

        return selected_symbol

    async def select_timeframe(self) -> TimeFrame:
        """Select analysis timeframe."""
        self.console.print("\n⏱️ Timeframe Selection", style="bold")

        # Create and display timeframe table
        table = self.displays.create_timeframe_table()
        self.console.print(table)

        choice = Prompt.ask(
            "\nSelect timeframe", choices=["1", "2", "3", "4", "5", "6"], default="4"
        )

        timeframe_map = {
            "1": TimeFrame.ONE_MINUTE,
            "2": TimeFrame.FIVE_MINUTES,
            "3": TimeFrame.FIFTEEN_MINUTES,
            "4": TimeFrame.ONE_HOUR,
            "5": TimeFrame.FOUR_HOURS,
            "6": TimeFrame.ONE_DAY,
        }

        selected_timeframe = timeframe_map[choice]
        self.console.print(f"✅ Selected: {selected_timeframe.value}", style="green")

        return selected_timeframe

    async def configure_risk_parameters(self) -> dict:
        """Configure risk management parameters."""
        self.console.print("\n⚠️ Risk Management Configuration", style="bold yellow")

        # Stop loss percentage
        stop_loss = Prompt.ask(
            "Stop Loss percentage",
            default=str(settings.default_stop_loss),
            show_default=True,
        )

        # Take profit percentage
        take_profit = Prompt.ask(
            "Take Profit percentage",
            default=str(settings.default_take_profit),
            show_default=True,
        )

        # Position size percentage
        position_size = Prompt.ask(
            "Position Size percentage", default="2.0", show_default=True
        )

        try:
            risk_params = {
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "position_size": float(position_size),
            }

            # Validate risk parameters
            if risk_params["stop_loss"] <= 0 or risk_params["stop_loss"] > 20:
                raise ValueError("Stop loss must be between 0 and 20%")
            if risk_params["take_profit"] <= 0 or risk_params["take_profit"] > 50:
                raise ValueError("Take profit must be between 0 and 50%")
            if risk_params["position_size"] <= 0 or risk_params["position_size"] > 10:
                raise ValueError("Position size must be between 0 and 10%")

            self.console.print("✅ Risk parameters configured", style="green")
            return risk_params

        except ValueError as e:
            self.console.print(f"❌ Invalid risk parameters: {e}", style="red")
            return await self.configure_risk_parameters()

    def display_configuration_summary(self, config: SessionConfig):
        """Display configuration summary."""
        self.displays.display_configuration_summary(config)

    async def run_analysis_session(self, strategy: StrategyType, config: SessionConfig):
        """Run the analysis session."""
        self.console.print(f"\n🚀 Starting Analysis Session", style="bold green")

        try:
            # Get strategy instance
            strategy_instance = self.strategies[strategy]

            # Step 1: Fetch market data
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Fetching market data from Delta Exchange...", total=None
                )

                market_data = await self.delta_client.get_market_data(
                    symbol=str(config.symbol),
                    timeframe=str(config.timeframe),
                    limit=100,
                )

                progress.update(task, description="✅ Market data fetched successfully")
                await asyncio.sleep(0.5)  # Brief pause for user experience

            # Display market data summary
            self.displays.display_market_data_summary(market_data)

            # Step 2: Run strategy analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Running technical analysis...", total=None)

                # Run the strategy analysis
                analysis_result = await strategy_instance.analyze(market_data, config)

                progress.update(
                    task, description="🧠 Generating AI analysis with Ollama..."
                )
                await asyncio.sleep(1)  # AI processing time

                progress.update(task, description="✅ Analysis completed successfully")
                await asyncio.sleep(0.5)

            # Display analysis results
            self.displays.display_analysis_results(analysis_result)

            # Display trading signals
            self.displays.display_trading_signals(analysis_result, config)

        except Exception as e:
            logger.error("Analysis session failed", error=str(e))
            self.console.print(f"\n❌ Analysis failed: {e}", style="red")

            if settings.is_development:
                import traceback

                self.console.print("\n🐛 Traceback:", style="dim")
                self.console.print(traceback.format_exc(), style="dim")

        # Wait for user input
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]", default="")

    async def ask_continue(self) -> bool:
        """Ask if user wants to continue."""
        return Confirm.ask("\n🔄 Would you like to run another analysis?", default=True)

    async def ask_continue_after_error(self) -> bool:
        """Ask if user wants to continue after an error."""
        return Confirm.ask("\n🔄 Would you like to try again?", default=False)

    def display_goodbye(self):
        """Display goodbye message."""
        self.displays.display_goodbye()

    async def run_backtesting_session(self):
        """Run interactive backtesting session."""
        self.console.print("\n📊 Strategy Backtesting", style="bold cyan")
        self.console.print("Test your strategies against historical market data\n", style="dim")

        try:
            # Strategy selection for backtesting
            strategy = await self.select_backtesting_strategy()
            if strategy is None:
                return

            # Configure backtesting parameters
            config = await self.configure_backtesting_parameters(strategy)
            
            # Run backtesting
            await self.execute_backtest(config)

        except Exception as e:
            self.console.print(f"\n❌ Backtesting failed: {e}", style="red")
            if settings.is_development:
                import traceback
                self.console.print("\n🐛 Traceback:", style="dim")
                self.console.print(traceback.format_exc(), style="dim")

    async def select_backtesting_strategy(self) -> Optional[StrategyType]:
        """Select strategy for backtesting."""
        self.console.print("🎯 Select Strategy to Backtest", style="bold cyan")
        
        table = Table(title="Available Strategies for Backtesting")
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Strategy", style="white", width=25)
        table.add_column("Description", style="dim", width=50)

        table.add_row("1", "Support & Resistance", "Test S/R bounce and rejection signals")
        table.add_row("2", "EMA Crossover", "Test 9/15 EMA crossover signals with filters")
        table.add_row("3", "Combined Strategy", "Test combined EMA + S/R signals")
        table.add_row("4", "Cancel", "Return to main menu")

        self.console.print(table)

        choice = Prompt.ask("\nSelect strategy to backtest", choices=["1", "2", "3", "4"], default="2")
        
        strategy_map = {
            "1": StrategyType.SUPPORT_RESISTANCE,
            "2": StrategyType.EMA_CROSSOVER,
            "3": StrategyType.COMBINED,
            "4": None,
        }

        selected = strategy_map[choice]
        if selected:
            self.console.print(f"\n✅ Selected: {selected.value.replace('_', ' ').title()}", style="green")
        
        return selected

    async def configure_backtesting_parameters(self, strategy: StrategyType) -> BacktestConfig:
        """Configure backtesting parameters."""
        self.console.print(f"\n⚙️ Backtesting Configuration", style="bold cyan")

        # Symbol selection
        symbol = await self.select_symbol()
        
        # Timeframe selection
        timeframe = await self.select_timeframe()
        
        # Backtesting period
        self.console.print("\n📅 Backtesting Period")
        days_back = int(Prompt.ask("Days to backtest", choices=["7", "14", "30", "60", "90"], default="30"))
        
        # Capital and leverage
        self.console.print("\n💰 Trading Parameters")
        initial_capital = float(Prompt.ask("Initial capital (INR)", default="850000"))
        leverage = int(Prompt.ask("Leverage (1-10x)", choices=[str(i) for i in range(1, 11)], default="5"))
        
        # Risk parameters
        self.console.print("\n🛡️ Risk Management")
        self.console.print("Delta Exchange India Fees: Futures Taker 0.05%, Maker 0.02%, Options 0.03% + 18% GST")
        confidence_threshold = int(Prompt.ask("Minimum signal confidence (1-10)", choices=[str(i) for i in range(1, 11)], default="6"))
        commission_pct = float(Prompt.ask("Commission (%) [Taker: 0.05, Maker: 0.02]", default="0.05"))
        
        return BacktestConfig(
            strategy_type=strategy,
            symbol=symbol,
            timeframe=timeframe,
            days_back=days_back,
            initial_capital=initial_capital,
            leverage=leverage,
            confidence_threshold=confidence_threshold,
            commission_pct=commission_pct,
            slippage_pct=0.05,  # Default slippage
        )

    async def execute_backtest(self, config: BacktestConfig):
        """Execute backtesting and display results."""
        self.console.print(f"\n🚀 Starting Backtest", style="bold green")
        
        # Display configuration
        config_table = Table(title="Backtesting Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Strategy", str(config.strategy_type).replace('_', ' ').title())
        config_table.add_row("Symbol", str(config.symbol))
        config_table.add_row("Timeframe", str(config.timeframe))
        config_table.add_row("Period", f"{config.days_back} days")
        config_table.add_row("Initial Capital", f"₹{config.initial_capital:,.0f}")
        config_table.add_row("Leverage", f"{config.leverage}x")
        config_table.add_row("Min Confidence", f"{config.confidence_threshold}/10")
        config_table.add_row("Commission", f"{config.commission_pct}%")
        
        self.console.print(config_table)

        # Progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running backtest...", total=None)
            
            # Create and run backtesting engine
            engine = BacktestEngine(config)
            result = await engine.run_backtest()
            
            progress.update(task, description="Generating HTML report...")
            
            # Generate report (INR currency)
            report_generator = BacktestReportGenerator(currency="INR")
            report_path = report_generator.generate_report(result)
            
            progress.stop()

        # Display results summary
        self.console.print(f"\n📈 Backtest Results", style="bold green")
        
        results_table = Table(title="Performance Summary")
        results_table.add_column("Metric", style="cyan", width=20)
        results_table.add_column("Value", style="white", width=15)
        results_table.add_column("Status", style="white", width=10)

        # Color coding for performance
        total_return = result.performance_metrics.total_return_pct
        return_color = "green" if total_return > 0 else "red"
        
        max_dd = result.performance_metrics.max_drawdown_pct
        dd_color = "green" if max_dd < 5 else "yellow" if max_dd < 15 else "red"
        
        win_rate = result.performance_metrics.win_rate_pct
        wr_color = "green" if win_rate > 60 else "yellow" if win_rate > 40 else "red"

        results_table.add_row("Total Return", f"{total_return:.2f}%", f"[{return_color}]{'📈' if total_return > 0 else '📉'}[/]")
        results_table.add_row("Max Drawdown", f"{max_dd:.2f}%", f"[{dd_color}]{'🛡️' if max_dd < 10 else '⚠️'}[/]")
        results_table.add_row("Sharpe Ratio", f"{result.performance_metrics.sharpe_ratio:.2f}", "📊")
        results_table.add_row("Win Rate", f"{win_rate:.1f}%", f"[{wr_color}]{'🎯' if win_rate > 50 else '💔'}[/]")
        results_table.add_row("Total Trades", str(result.performance_metrics.total_trades), "🔄")
        results_table.add_row("Profit Factor", f"{result.performance_metrics.profit_factor:.2f}", "💰")

        self.console.print(results_table)

        # Report information
        self.console.print(f"\n📊 Detailed HTML report generated:", style="bold cyan")
        self.console.print(f"   📄 {report_path}", style="dim")
        self.console.print(f"\n💡 Open the HTML file in your browser to view interactive charts and detailed analysis.", style="dim")

        # Ask to open report
        if Confirm.ask("\nOpen HTML report in browser?", default=True):
            import webbrowser
            import os
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            self.console.print("🌐 Report opened in browser", style="green")

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.delta_client.close()
            await self.websocket_client.disconnect()
            for strategy in self.strategies.values():
                await strategy.close()
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))


# Create CLI application instance
cli_app = CLIApplication()


async def main():
    """Main CLI function."""
    try:
        await cli_app.run()
    finally:
        await cli_app.cleanup()


# Click command for script execution
import click


@click.command()
@click.option("--env", default="development", help="Environment to run in")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(env, debug):
    """TradeBuddy CLI application."""
    # Set environment
    import os

    os.environ["PYTHON_ENV"] = env
    if debug:
        os.environ["DEBUG"] = "true"

    # Run the main application
    asyncio.run(main())


if __name__ == "__main__":
    cli()
