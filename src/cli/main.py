"""
Main CLI application for TradeBuddy.

Provides interactive terminal interface for trading strategy selection and analysis.
"""

import asyncio
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
import structlog

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.config import settings
from src.core.models import StrategyType, Symbol, TimeFrame, SessionConfig
from src.core.exceptions import CLIError
from src.data.delta_client import DeltaExchangeClient
from src.analysis.strategies.support_resistance import SupportResistanceStrategy
from src.analysis.strategies.ema_crossover import EMACrossoverStrategy
from src.analysis.strategies.combined import CombinedStrategy


console = Console()
logger = structlog.get_logger(__name__)


class CLIApplication:
    """Main CLI application class."""
    
    def __init__(self):
        self.console = console
        self.running = True
        self.delta_client = DeltaExchangeClient()
        self.strategies = {
            StrategyType.SUPPORT_RESISTANCE: SupportResistanceStrategy(),
            StrategyType.EMA_CROSSOVER: EMACrossoverStrategy(),
            StrategyType.COMBINED: CombinedStrategy()
        }
    
    async def run(self):
        """Run the main CLI application loop."""
        self.display_welcome()
        
        while self.running:
            try:
                # Strategy selection
                strategy = await self.select_strategy()
                if strategy is None:
                    break
                
                # Configuration
                config = await self.configure_session(strategy)
                
                # Run analysis session
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
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="white")
        welcome_text.append("TradeBuddy", style="bold cyan")
        welcome_text.append("!\n\n", style="white")
        welcome_text.append("🤖 AI-Powered Trading Signal Analysis\n", style="white")
        welcome_text.append("📊 Delta Exchange Market Data\n", style="white")
        welcome_text.append("🧠 Local Ollama AI Analysis\n\n", style="white")
        welcome_text.append("Choose your trading strategy and get real-time signals.", style="dim")
        
        self.console.print(Panel(
            welcome_text,
            title="🎯 TradeBuddy Trading Terminal",
            style="blue",
            padding=(1, 2)
        ))
    
    async def select_strategy(self) -> Optional[StrategyType]:
        """Interactive strategy selection."""
        self.console.print("\n🎯 Trading Strategy Selection", style="bold cyan")
        
        # Create strategy table
        table = Table(title="Available Strategies")
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Strategy", style="white", width=25)
        table.add_column("Description", style="dim", width=50)
        
        table.add_row(
            "1", 
            "Support & Resistance", 
            "Identifies key price levels for bounce/rejection signals"
        )
        table.add_row(
            "2", 
            "EMA Crossover", 
            "Uses 9/15 EMA crossovers for trend change signals"
        )
        table.add_row(
            "3", 
            "Combined Strategy", 
            "Combines both strategies for high-confidence signals"
        )
        table.add_row(
            "4", 
            "Exit", 
            "Exit the application"
        )
        
        self.console.print(table)
        
        # Get user choice
        choice = Prompt.ask(
            "\nSelect strategy",
            choices=["1", "2", "3", "4"],
            default="3"
        )
        
        strategy_map = {
            "1": StrategyType.SUPPORT_RESISTANCE,
            "2": StrategyType.EMA_CROSSOVER,
            "3": StrategyType.COMBINED,
            "4": None
        }
        
        selected_strategy = strategy_map[choice]
        
        if selected_strategy:
            self.console.print(f"\n✅ Selected: {selected_strategy.value.replace('_', ' ').title()}", style="green")
        
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
            position_size_pct=risk_params["position_size"]
        )
        
        # Display configuration summary
        self.display_configuration_summary(config)
        
        return config
    
    async def select_symbol(self) -> Symbol:
        """Select trading symbol."""
        self.console.print("\n📊 Symbol Selection", style="bold")
        
        # Create symbol table
        table = Table(title="Available Symbols")
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Symbol", style="white", width=15)
        table.add_column("Name", style="dim", width=20)
        
        symbol_info = {
            "BTCUSDT": "Bitcoin",
            "ETHUSDT": "Ethereum",
            "SOLUSDT": "Solana",
            "ADAUSDT": "Cardano",
            "DOGEUSDT": "Dogecoin"
        }
        
        for i, (symbol, name) in enumerate(symbol_info.items(), 1):
            table.add_row(str(i), symbol, name)
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "\nSelect symbol",
            choices=["1", "2", "3", "4", "5"],
            default="1"
        )
        
        symbol_map = {
            "1": Symbol.BTCUSDT,
            "2": Symbol.ETHUSDT,
            "3": Symbol.SOLUSDT,
            "4": Symbol.ADAUSDT,
            "5": Symbol.DOGEUSDT
        }
        
        selected_symbol = symbol_map[choice]
        self.console.print(f"✅ Selected: {selected_symbol.value}", style="green")
        
        return selected_symbol
    
    async def select_timeframe(self) -> TimeFrame:
        """Select analysis timeframe."""
        self.console.print("\n⏱️ Timeframe Selection", style="bold")
        
        # Create timeframe table
        table = Table(title="Available Timeframes")
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Timeframe", style="white", width=12)
        table.add_column("Description", style="dim", width=25)
        
        timeframe_info = {
            "1m": "1 Minute (Scalping)",
            "5m": "5 Minutes (Short-term)",
            "15m": "15 Minutes (Short-term)",
            "1h": "1 Hour (Medium-term)",
            "4h": "4 Hours (Swing)",
            "1d": "1 Day (Long-term)"
        }
        
        for i, (tf, desc) in enumerate(timeframe_info.items(), 1):
            table.add_row(str(i), tf, desc)
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "\nSelect timeframe",
            choices=["1", "2", "3", "4", "5", "6"],
            default="4"
        )
        
        timeframe_map = {
            "1": TimeFrame.ONE_MINUTE,
            "2": TimeFrame.FIVE_MINUTES,
            "3": TimeFrame.FIFTEEN_MINUTES,
            "4": TimeFrame.ONE_HOUR,
            "5": TimeFrame.FOUR_HOURS,
            "6": TimeFrame.ONE_DAY
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
            show_default=True
        )
        
        # Take profit percentage
        take_profit = Prompt.ask(
            "Take Profit percentage",
            default=str(settings.default_take_profit),
            show_default=True
        )
        
        # Position size percentage
        position_size = Prompt.ask(
            "Position Size percentage",
            default="2.0",
            show_default=True
        )
        
        try:
            risk_params = {
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "position_size": float(position_size)
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
        self.console.print("\n📋 Configuration Summary", style="bold cyan")
        
        summary_table = Table(title="Session Configuration")
        summary_table.add_column("Parameter", style="cyan", width=20)
        summary_table.add_column("Value", style="white", width=20)
        
        # Handle strategy display (Pydantic converts enums to strings)
        strategy_name = str(config.strategy)
        summary_table.add_row("Strategy", strategy_name.replace('_', ' ').title())
        
        # Handle symbol display (Pydantic converts enums to strings)
        symbol_name = str(config.symbol)
        summary_table.add_row("Symbol", symbol_name)
        
        # Handle timeframe display (Pydantic converts enums to strings)
        timeframe_name = str(config.timeframe)
        summary_table.add_row("Timeframe", timeframe_name)
        summary_table.add_row("Stop Loss", f"{config.stop_loss_pct}%")
        summary_table.add_row("Take Profit", f"{config.take_profit_pct}%")
        summary_table.add_row("Position Size", f"{config.position_size_pct}%")
        
        self.console.print(summary_table)
    
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
                console=self.console
            ) as progress:
                task = progress.add_task("Fetching market data from Delta Exchange...", total=None)
                
                market_data = await self.delta_client.get_market_data(
                    symbol=str(config.symbol),
                    timeframe=str(config.timeframe),
                    limit=100
                )
                
                progress.update(task, description="✅ Market data fetched successfully")
                await asyncio.sleep(0.5)  # Brief pause for user experience
            
            # Display market data summary
            self.display_market_data_summary(market_data)
            
            # Step 2: Run strategy analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Running technical analysis...", total=None)
                
                # Run the strategy analysis
                analysis_result = await strategy_instance.analyze(market_data, config)
                
                progress.update(task, description="🧠 Generating AI analysis with Ollama...")
                await asyncio.sleep(1)  # AI processing time
                
                progress.update(task, description="✅ Analysis completed successfully")
                await asyncio.sleep(0.5)
            
            # Display analysis results
            self.display_analysis_results(analysis_result, config)
            
            # Display trading signals
            self.display_trading_signals(analysis_result, config)
            
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
    
    def display_market_data_summary(self, market_data):
        """Display market data summary."""
        self.console.print("\n📊 Market Data Summary", style="bold cyan")
        
        # Create market data table
        data_table = Table(title=f"{market_data.symbol} - {market_data.timeframe}")
        data_table.add_column("Metric", style="cyan", width=20)
        data_table.add_column("Value", style="white", width=20)
        
        # Get latest candle
        latest_candle = market_data.ohlcv_data[-1] if market_data.ohlcv_data else None
        
        if latest_candle:
            data_table.add_row("Current Price", f"${market_data.current_price:,.2f}")
            data_table.add_row("Open", f"${latest_candle.open:,.2f}")
            data_table.add_row("High", f"${latest_candle.high:,.2f}")
            data_table.add_row("Low", f"${latest_candle.low:,.2f}")
            data_table.add_row("Close", f"${latest_candle.close:,.2f}")
            data_table.add_row("Volume", f"{latest_candle.volume:,.0f}")
        
        data_table.add_row("Data Points", str(len(market_data.ohlcv_data)))
        
        self.console.print(data_table)
    
    def display_analysis_results(self, analysis_result, config: SessionConfig):
        """Display analysis results."""
        self.console.print("\n🧠 AI Analysis Results", style="bold magenta")
        
        # Display AI analysis text
        self.console.print(Panel(
            analysis_result.ai_analysis,
            title="🤖 Ollama AI Analysis",
            style="blue"
        ))
    
    def display_trading_signals(self, analysis_result, config: SessionConfig):
        """Display trading signals."""
        self.console.print("\n📈 Trading Signals", style="bold green")
        
        if not analysis_result.signals:
            self.console.print("⚠️ No trading signals generated", style="yellow")
            return
        
        # Create signals table
        signals_table = Table(title="Generated Signals")
        signals_table.add_column("Signal", style="cyan", width=8)
        signals_table.add_column("Action", style="white", width=8)
        signals_table.add_column("Confidence", style="white", width=10)
        signals_table.add_column("Entry Price", style="white", width=12)
        signals_table.add_column("Stop Loss", style="red", width=12)
        signals_table.add_column("Take Profit", style="green", width=12)
        signals_table.add_column("Risk/Reward", style="white", width=10)
        
        for i, signal in enumerate(analysis_result.signals, 1):
            # Calculate stop loss and take profit
            entry_price = signal.entry_price
            
            # Handle action display (Pydantic converts enums to strings)
            action_name = str(signal.action)
            action_upper = action_name.upper()
            
            if action_upper == "BUY":
                stop_loss = entry_price * (1 - float(config.stop_loss_pct) / 100)
                take_profit = entry_price * (1 + float(config.take_profit_pct) / 100)
            else:
                stop_loss = entry_price * (1 + float(config.stop_loss_pct) / 100)
                take_profit = entry_price * (1 - float(config.take_profit_pct) / 100)
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Style based on action
            action_style = "green" if action_upper == "BUY" else "red"
            
            signals_table.add_row(
                f"#{i}",
                f"[{action_style}]{action_upper}[/{action_style}]",
                f"{signal.confidence}/10",
                f"${entry_price:,.2f}",
                f"${stop_loss:,.2f}",
                f"${take_profit:,.2f}",
                f"{risk_reward:.2f}:1"
            )
        
        self.console.print(signals_table)
        
        # Display primary signal if available
        if analysis_result.primary_signal:
            primary = analysis_result.primary_signal
            # Handle primary signal action display (Pydantic converts enums to strings)
            primary_action = str(primary.action)
            primary_action_upper = primary_action.upper()
            primary_style = "green" if primary_action_upper == "BUY" else "red"
            
            self.console.print(Panel(
                f"[bold]Primary Signal: [{primary_style}]{primary_action_upper}[/{primary_style}][/bold]\n"
                f"Confidence: {primary.confidence}/10\n"
                f"Entry Price: ${float(primary.entry_price):,.2f}\n"
                f"Reasoning: {primary.reasoning}",
                title="🎯 Primary Signal",
                style="yellow"
            ))
        
        # Risk management reminder
        self.console.print(Panel(
            f"⚠️ [bold]Risk Management Reminder[/bold]\n\n"
            f"• Position Size: {config.position_size_pct}% of portfolio\n"
            f"• Stop Loss: {config.stop_loss_pct}% from entry\n"
            f"• Take Profit: {config.take_profit_pct}% from entry\n"
            f"• These are educational signals only\n"
            f"• Always do your own research\n"
            f"• Never risk more than you can afford to lose",
            title="⚠️ Risk Management",
            style="red"
        ))
    
    def display_goodbye(self):
        """Display goodbye message."""
        self.console.print(Panel(
            "[bold cyan]Thank you for using TradeBuddy![/bold cyan]\n\n"
            "💡 Remember:\n"
            "• Trading involves risk\n"
            "• Use proper risk management\n"
            "• Signals are for educational purposes\n"
            "• Always do your own research\n\n"
            "[dim]Happy trading! 📈[/dim]",
            title="👋 Goodbye",
            style="green"
        ))
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.delta_client.close()
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
@click.option('--env', default='development', help='Environment to run in')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(env, debug):
    """TradeBuddy CLI application."""
    # Set environment
    import os
    os.environ['PYTHON_ENV'] = env
    if debug:
        os.environ['DEBUG'] = 'true'
    
    # Run the main application
    asyncio.run(main())


if __name__ == "__main__":
    cli()