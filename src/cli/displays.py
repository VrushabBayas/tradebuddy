"""
Display helpers for TradeBuddy CLI interface.

Provides reusable display components and formatting utilities.
"""

from typing import List, Optional
from decimal import Decimal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.models import (
    SessionConfig, AnalysisResult, TradingSignal, OHLCV, MarketData,
    StrategyType, Symbol, TimeFrame, SignalAction
)


class CLIDisplays:
    """Centralized display utilities for CLI application."""
    
    def __init__(self, console: Console):
        """Initialize display utilities."""
        self.console = console
    
    def display_welcome(self) -> None:
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
    
    def display_goodbye(self) -> None:
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
    
    def create_strategy_table(self) -> Table:
        """Create strategy selection table."""
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
            "Real-Time Analysis ⭐", 
            "Live market data streaming with real-time signals"
        )
        table.add_row(
            "5", 
            "Exit", 
            "Exit the application"
        )
        
        return table
    
    def create_symbol_table(self) -> Table:
        """Create symbol selection table."""
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
        
        return table
    
    def create_timeframe_table(self) -> Table:
        """Create timeframe selection table."""
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
        
        return table
    
    def display_configuration_summary(self, config: SessionConfig) -> None:
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
    
    def display_market_data_summary(self, market_data: MarketData) -> None:
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
    
    def display_analysis_results(self, analysis_result: AnalysisResult) -> None:
        """Display analysis results."""
        self.console.print("\n🧠 AI Analysis Results", style="bold magenta")
        
        # Display AI analysis text
        self.console.print(Panel(
            analysis_result.ai_analysis,
            title="🤖 Ollama AI Analysis",
            style="blue"
        ))
    
    def display_trading_signals(self, analysis_result: AnalysisResult, config: SessionConfig) -> None:
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
        self.display_risk_management_reminder(config)
    
    def display_risk_management_reminder(self, config: SessionConfig) -> None:
        """Display risk management reminder."""
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