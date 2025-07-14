"""
Main CLI application for TradeBuddy.

Provides interactive terminal interface for trading strategy selection and analysis.
"""

import asyncio
import os
import sys
from decimal import Decimal
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
from src.analysis.strategies.ema_crossover_v2 import EMACrossoverV2Strategy
from src.analysis.strategies.support_resistance import SupportResistanceStrategy
from src.analysis.ai_models.model_factory import ModelFactory
from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.backtesting.reports import BacktestReportGenerator
from src.cli.displays import CLIDisplays
from src.core.config import settings
from src.core.exceptions import CLIError
from src.core.models import SessionConfig, StrategyType, Symbol, TimeFrame, AIModelType, AIModelConfig
from src.data.delta_client import DeltaExchangeClient
from src.utils.logging_utils import analysis_progress, setup_cli_logging

console = Console()
logger = structlog.get_logger(__name__)


class CLIApplication:
    """Main CLI application class."""

    def __init__(self):
        self.console = console
        self.running = True
        self.delta_client = DeltaExchangeClient()
        # Strategies will be created dynamically with AI model configuration
        self.strategies = {}

        # Display utilities
        self.displays = CLIDisplays(self.console)

        # Setup optimized logging for CLI
        setup_cli_logging()

    async def run(self):
        """Run the main CLI application loop."""
        self.display_welcome()

        while self.running:
            try:
                # Strategy selection
                strategy = await self.select_strategy()
                if strategy is None:
                    break

                if strategy == "BACKTESTING":
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
            "\nSelect strategy", choices=["1", "2", "3", "4", "5"], default="3"
        )

        strategy_map = {
            "1": StrategyType.SUPPORT_RESISTANCE,
            "2": StrategyType.EMA_CROSSOVER_V2,
            "3": StrategyType.COMBINED,
            "4": "BACKTESTING",  # Special marker for backtesting
            "5": None,
        }

        selected_strategy = strategy_map[choice]

        if selected_strategy == "BACKTESTING":
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
        
        # AI model selection
        ai_model_config = await self.select_ai_model()

        config = SessionConfig(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            total_capital_inr=Decimal(str(risk_params["total_capital_inr"])),
            trading_capital_pct=Decimal(str(risk_params["trading_capital_pct"])),
            risk_per_trade_pct=Decimal(str(risk_params["risk_per_trade_pct"])),
            stop_loss_pct=Decimal("1.5"),  # Placeholder - will be calculated dynamically based on risk
            take_profit_pct=Decimal(str(risk_params["take_profit"])),
            leverage=risk_params["leverage"],
            ai_model_config=ai_model_config,
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
            "1": Symbol.BTCUSD,
            "2": Symbol.ETHUSD,
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

    async def select_ai_model(self) -> AIModelConfig:
        """Select AI model for analysis."""
        self.console.print("\n🤖 AI Model Selection", style="bold")
        
        # Create AI model selection table
        table = Table(title="Available AI Models", show_header=True, header_style="bold cyan")
        table.add_column("Option", style="cyan", justify="center")
        table.add_column("Model", style="white")
        table.add_column("Description", style="dim")
        table.add_column("Features", style="green")
        
        table.add_row(
            "1", 
            "Ollama (Qwen2.5:14b)", 
            "Local LLM - Current default", 
            "General AI, Fast, Private"
        )
        table.add_row(
            "2", 
            "FinGPT v3.2", 
            "Financial-specialized AI", 
            "Financial expertise, Sentiment analysis"
        )
        table.add_row(
            "3", 
            "FinGPT v3.3", 
            "Latest FinGPT model", 
            "Enhanced financial analysis, Latest features"
        )
        table.add_row(
            "4", 
            "Comparative Analysis ⚖️", 
            "Run both Ollama + FinGPT", 
            "Best of both worlds, Consensus signals"
        )
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "\nSelect AI model", choices=["1", "2", "3", "4"], default="1"
        )
        
        if choice == "1":
            ai_config = AIModelConfig(
                model_type=AIModelType.OLLAMA,
                fallback_enabled=True
            )
            self.console.print("✅ Selected: Ollama (Qwen2.5:14b)", style="green")
        elif choice == "2":
            ai_config = AIModelConfig(
                model_type=AIModelType.FINGPT,
                fingpt_model_variant="v3.2",
                fallback_enabled=True
            )
            self.console.print("✅ Selected: FinGPT v3.2", style="green")
        elif choice == "3":
            ai_config = AIModelConfig(
                model_type=AIModelType.FINGPT,
                fingpt_model_variant="v3.3",
                fallback_enabled=True
            )
            self.console.print("✅ Selected: FinGPT v3.3", style="green")
        else:  # choice == "4"
            ai_config = AIModelConfig(
                model_type=AIModelType.FINGPT,
                fingpt_model_variant="v3.2",
                comparative_mode=True,
                fallback_enabled=True
            )
            self.console.print("✅ Selected: Comparative Analysis (Ollama + FinGPT)", style="green")
        
        # Ask about comparative mode for individual FinGPT selections
        if ai_config.model_type == AIModelType.FINGPT and not ai_config.comparative_mode:
            comparative = Confirm.ask(
                "\n🔍 Enable comparative mode? (Run both Ollama and FinGPT for comparison)",
                default=False
            )
            ai_config.comparative_mode = comparative
            if comparative:
                self.console.print("✅ Comparative mode enabled", style="green")
        
        return ai_config

    def create_strategy_with_ai_model(self, strategy_type: StrategyType, ai_config: AIModelConfig):
        """Create strategy instance with specified AI model configuration."""
        # Create AI model based on configuration
        ai_model = ModelFactory.create_model(ai_config.model_type, config=ai_config)
        
        # Create strategy with AI model
        if strategy_type == StrategyType.SUPPORT_RESISTANCE:
            return SupportResistanceStrategy(ai_model=ai_model)
        elif strategy_type == StrategyType.EMA_CROSSOVER_V2:
            return EMACrossoverV2Strategy(ai_model=ai_model)
        elif strategy_type == StrategyType.COMBINED:
            return CombinedStrategy(ai_model=ai_model)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    async def configure_risk_parameters(self) -> dict:
        """Configure risk-based capital management parameters."""
        self.console.print("\n💰 Risk Management Configuration", style="bold yellow")
        
        # Capital Management Configuration
        self.console.print("\n📊 [bold]Capital Allocation[/bold]")
        
        # Total capital in INR
        total_capital = Prompt.ask(
            "Total Capital (₹)",
            default="100000",
            show_default=True,
        )
        
        # Trading capital percentage
        trading_capital_pct = Prompt.ask(
            "Trading Capital percentage (% of total)",
            default="50.0",
            show_default=True,
        )
        
        # Risk per trade percentage
        risk_per_trade_pct = Prompt.ask(
            "Risk per Trade (% of trading capital)",
            default="2.0",
            show_default=True,
        )
        
        self.console.print("\n📈 [bold]Position Parameters[/bold]")
        
        # Take profit percentage
        take_profit = Prompt.ask(
            "Take Profit percentage",
            default=str(settings.default_take_profit),
            show_default=True,
        )

        # Leverage setting
        leverage = Prompt.ask(
            "Leverage (1-100x)",
            default="10",
            show_default=True,
        )
        
        # Note: Stop loss is now automatically calculated based on risk per trade
        self.console.print("💡 [dim]Stop loss will be calculated automatically based on your risk per trade amount[/dim]")

        try:
            risk_params = {
                "total_capital_inr": float(total_capital),
                "trading_capital_pct": float(trading_capital_pct),
                "risk_per_trade_pct": float(risk_per_trade_pct),
                "take_profit": float(take_profit),
                "leverage": int(leverage),
            }

            # Validate capital management parameters
            if risk_params["total_capital_inr"] <= 0:
                raise ValueError("Total capital must be positive")
            if risk_params["trading_capital_pct"] <= 0 or risk_params["trading_capital_pct"] > 100:
                raise ValueError("Trading capital percentage must be between 0 and 100%")
            if risk_params["risk_per_trade_pct"] <= 0 or risk_params["risk_per_trade_pct"] > 10:
                raise ValueError("Risk per trade must be between 0 and 10%")
            
            # Validate position parameters
            if risk_params["take_profit"] <= 0 or risk_params["take_profit"] > 50:
                raise ValueError("Take profit must be between 0 and 50%")
            if risk_params["leverage"] < 1 or risk_params["leverage"] > 100:
                raise ValueError("Leverage must be between 1 and 100x")

            # Calculate and display summary
            trading_capital_inr = risk_params["total_capital_inr"] * (risk_params["trading_capital_pct"] / 100)
            risk_amount_inr = trading_capital_inr * (risk_params["risk_per_trade_pct"] / 100)
            backup_capital_inr = risk_params["total_capital_inr"] - trading_capital_inr
            
            self.console.print(f"\n📋 [bold]Capital Summary[/bold]")
            self.console.print(f"  • Total Capital: ₹{risk_params['total_capital_inr']:,.0f}")
            self.console.print(f"  • Trading Capital: ₹{trading_capital_inr:,.0f} ({risk_params['trading_capital_pct']}%)")
            self.console.print(f"  • Backup Capital: ₹{backup_capital_inr:,.0f}")
            self.console.print(f"  • Risk per Trade: ₹{risk_amount_inr:,.0f} ({risk_params['risk_per_trade_pct']}%)")
            self.console.print(f"  • Leverage: {risk_params['leverage']}x")
            
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
            # Step 1: Fetch market data
            with analysis_progress(self.console, "📊 Fetching market data from Delta Exchange...") as (progress, task):
                market_data = await self.delta_client.get_market_data(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    limit=200,
                )
                progress.update(task, description="✅ Market data fetched successfully")
                await asyncio.sleep(0.3)  # Brief pause for user experience

            # Display market data summary
            self.displays.display_market_data_summary(market_data)

            # Check if comparative mode is enabled
            if config.ai_model_config and config.ai_model_config.comparative_mode:
                await self._run_comparative_analysis(strategy, config, market_data)
            else:
                await self._run_single_model_analysis(strategy, config, market_data)

        except Exception as e:
            logger.error("Analysis session failed", error=str(e))
            self.console.print(f"\n❌ Analysis failed: {e}", style="red")

            if settings.is_development:
                import traceback

                self.console.print("\n🐛 Traceback:", style="dim")
                self.console.print(traceback.format_exc(), style="dim")

        # Wait for user input
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]", default="")

    async def _run_single_model_analysis(self, strategy: StrategyType, config: SessionConfig, market_data):
        """Run analysis with a single AI model."""
        # Create strategy instance with AI model configuration
        strategy_instance = self.create_strategy_with_ai_model(strategy, config.ai_model_config)

        # Step 2: Run strategy analysis
        # Get AI model name for display
        if config.ai_model_config:
            model_type = config.ai_model_config.model_type
            # Handle both enum and string values
            if hasattr(model_type, 'value'):
                ai_model_name = model_type.value.title()
            else:
                ai_model_name = str(model_type).title()
        else:
            ai_model_name = "Ollama"
        
        with analysis_progress(self.console, f"🧠 Running {ai_model_name} analysis...") as (progress, task):
            # Run the strategy analysis
            analysis_result = await strategy_instance.analyze(market_data, config)
            progress.update(task, description="✅ Analysis completed successfully")
            await asyncio.sleep(0.3)

        # Display analysis results
        self.displays.display_analysis_results(analysis_result, ai_model_name)

        # Display trading signals
        self.displays.display_trading_signals(analysis_result, config)
        
        # Cleanup strategy instance
        await strategy_instance.close()

    async def _run_comparative_analysis(self, strategy: StrategyType, config: SessionConfig, market_data):
        """Run comparative analysis with multiple AI models."""
        from src.analysis.ai_models.comparative_analyzer import ComparativeAnalyzer
        
        # Initialize comparative analyzer
        comparative_analyzer = ComparativeAnalyzer()
        
        # Create strategy instance for technical analysis
        strategy_instance = self.create_strategy_with_ai_model(strategy, config.ai_model_config)

        with analysis_progress(self.console, "🧠 Running comparative AI analysis...") as (progress, task):
            # Get technical analysis from strategy (without AI analysis)
            # We'll extract technical indicators from the strategy
            technical_analysis = await strategy_instance._calculate_technical_analysis(market_data)
            
            progress.update(task, description="🤖 Analyzing with multiple AI models...")
            
            # Run comparative analysis
            comparison_results = await comparative_analyzer.analyze_with_comparison(
                market_data=market_data,
                technical_analysis=technical_analysis,
                strategy_type=strategy.value,
                session_config=config
            )
            
            progress.update(task, description="✅ Comparative analysis completed")
            await asyncio.sleep(0.3)

        # Display comparative results
        await self._display_comparative_results(comparison_results, config)
        
        # Cleanup
        await comparative_analyzer.close()
        await strategy_instance.close()

    async def _display_comparative_results(self, comparison_results: dict, config: SessionConfig):
        """Display comparative analysis results."""
        individual_results = comparison_results["individual_results"]
        consensus = comparison_results["consensus"]
        execution_time = comparison_results["execution_time"]
        
        # Display header
        self.console.print(f"\n⚖️ Comparative AI Analysis Results", style="bold cyan")
        self.console.print(f"📊 Execution Time: {execution_time:.2f}s", style="dim")
        
        # Display individual model results
        for model_name, result in individual_results.items():
            self.console.print(f"\n🤖 {model_name.upper()} Analysis", style="bold")
            
            if result["status"] == "success":
                # Display individual analysis
                if result.get("analysis"):
                    panel = Panel(
                        result["analysis"],
                        title=f"{model_name.upper()} AI Analysis",
                        border_style="green"
                    )
                    self.console.print(panel)
                
                # Display individual signals
                if result.get("signals"):
                    self.console.print(f"📈 {model_name.upper()} Signals:")
                    for signal in result["signals"]:
                        signal_color = "green" if signal.action.value in ["BUY"] else "red" if signal.action.value == "SELL" else "yellow"
                        self.console.print(
                            f"  • {signal.action.value} | Confidence: {signal.confidence}/10 | Entry: ${signal.entry_price:,.2f}",
                            style=signal_color
                        )
            else:
                self.console.print(f"  ❌ Analysis failed: {result['error']}", style="red")
        
        # Display consensus analysis
        self.console.print(f"\n🎯 Consensus Analysis", style="bold cyan")
        
        consensus_signal = consensus["consensus_signal"]
        agreement_level = consensus["agreement_level"]
        recommendation = consensus["recommendation"]
        
        # Consensus signal panel
        consensus_color = "green" if consensus_signal["action"] in ["BUY"] else "red" if consensus_signal["action"] == "SELL" else "yellow"
        
        consensus_panel = Panel(
            f"""**Signal:** {consensus_signal['action']}
**Confidence:** {consensus_signal['confidence']}/10
**Agreement:** {agreement_level:.1%}
**Reasoning:** {consensus_signal['reasoning']}

**Recommendation:** {recommendation}""",
            title="🎯 Consensus Trading Signal",
            border_style=consensus_color
        )
        self.console.print(consensus_panel)
        
        # Model agreement details
        participating_models = consensus.get("participating_models", [])
        failed_models = consensus.get("failed_models", [])
        
        if len(participating_models) > 1:
            self.console.print(f"\n📋 Model Agreement Details", style="bold")
            self.console.print(f"  • Participating Models: {', '.join(participating_models)}")
            if failed_models:
                self.console.print(f"  • Failed Models: {', '.join(failed_models)}", style="red")
            self.console.print(f"  • Overall Agreement: {agreement_level:.1%}")
        elif len(participating_models) == 1:
            self.console.print(f"\n📋 Single Model Analysis", style="bold")
            self.console.print(f"  • Model Used: {participating_models[0]}")
            if failed_models:
                self.console.print(f"  • Failed Models: {', '.join(failed_models)}", style="red")
        
        # Performance metrics if available
        performance = comparison_results.get("performance_metrics")
        if performance:
            self.console.print(f"\n📊 Performance Metrics", style="bold")
            for metric, value in performance.items():
                self.console.print(f"  • {metric}: {value}")
                
        # Risk management note
        self.console.print(f"\n💡 Risk Management Note", style="bold yellow")
        self.console.print("Always validate signals with additional analysis and use proper position sizing.", style="dim")

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
        with analysis_progress(self.console, "📊 Running backtest analysis...") as (progress, task):
            # Create and run backtesting engine
            engine = BacktestEngine(config)
            result = await engine.run_backtest()
            
            progress.update(task, description="📋 Generating HTML report...")
            
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
        logger.debug("Starting CLI cleanup")
        
        # Close delta client
        try:
            await self.delta_client.close()
            logger.debug("Delta client closed")
        except Exception as e:
            logger.error("Error closing delta client", error=str(e))
        
        # Close any cached strategies
        try:
            for strategy_name, strategy in self.strategies.items():
                await strategy.close()
                logger.debug("Strategy closed", strategy=strategy_name)
        except Exception as e:
            logger.error("Error closing strategies", error=str(e))
        
        # Give a moment for connections to fully close
        await asyncio.sleep(0.1)
        
        logger.debug("CLI cleanup completed")


# Create CLI application instance
cli_app = CLIApplication()


async def main():
    """Main CLI function."""
    try:
        await cli_app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application error", error=str(e))
        if settings.is_development:
            import traceback
            traceback.print_exc()
    finally:
        # Ensure cleanup always runs
        try:
            await cli_app.cleanup()
        except Exception as e:
            logger.error("Cleanup error", error=str(e))


# Click command for script execution
import click


@click.command()
@click.option("--env", default="development", help="Environment to run in")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--verbose", is_flag=True, help="Enable verbose logging (shows all debug logs)")
def cli(env, debug, verbose):
    """TradeBuddy CLI application."""
    # Set environment
    import os

    os.environ["PYTHON_ENV"] = env
    if debug:
        os.environ["DEBUG"] = "true"
    if verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        print("🔧 Verbose logging enabled - you'll see detailed debug information")
    else:
        # Ensure clean UI by default
        os.environ["LOG_LEVEL"] = "INFO"

    # Run the main application
    asyncio.run(main())


if __name__ == "__main__":
    cli()
