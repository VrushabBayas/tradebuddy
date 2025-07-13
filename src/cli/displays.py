"""
Display helpers for TradeBuddy CLI interface.

Provides reusable display components and formatting utilities.
"""

from decimal import Decimal
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.models import (
    OHLCV,
    AnalysisResult,
    MarketData,
    SessionConfig,
    SignalAction,
    StrategyType,
    Symbol,
    TimeFrame,
    TradingSignal,
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
        welcome_text.append(
            "Choose your trading strategy and get real-time signals.", style="dim"
        )

        self.console.print(
            Panel(
                welcome_text,
                title="🎯 TradeBuddy Trading Terminal",
                style="blue",
                padding=(1, 2),
            )
        )

    def display_goodbye(self) -> None:
        """Display goodbye message."""
        self.console.print(
            Panel(
                "[bold cyan]Thank you for using TradeBuddy![/bold cyan]\n\n"
                "💡 Remember:\n"
                "• Trading involves risk\n"
                "• Use proper risk management\n"
                "• Signals are for educational purposes\n"
                "• Always do your own research\n\n"
                "[dim]Happy trading! 📈[/dim]",
                title="👋 Goodbye",
                style="green",
            )
        )

    def create_strategy_table(self) -> Table:
        """Create strategy selection table."""
        table = Table(title="Available Strategies")
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Strategy", style="white", width=25)
        table.add_column("Description", style="dim", width=50)

        table.add_row(
            "1",
            "Support & Resistance",
            "Identifies key price levels for bounce/rejection signals",
        )
        table.add_row(
            "2", "EMA Crossover", "Uses 9/15 EMA crossovers for trend change signals"
        )
        table.add_row(
            "3",
            "Combined Strategy",
            "Combines both strategies for high-confidence signals",
        )
        table.add_row(
            "4",
            "Real-Time Analysis ⭐",
            "Live market data streaming with real-time signals",
        )
        table.add_row(
            "5", "Market Monitoring 🔄", "Continuous market monitoring with alerts"
        )
        table.add_row(
            "6", "Strategy Backtesting 📊", "Test strategies against historical data"
        )
        table.add_row("7", "Exit", "Exit the application")

        return table

    def create_symbol_table(self) -> Table:
        """Create symbol selection table."""
        table = Table(title="Available Symbols")
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Symbol", style="white", width=15)
        table.add_column("Name", style="dim", width=20)

        symbol_info = {
            "BTCUSD": "Bitcoin",
            "ETHUSD": "Ethereum", 
            "SOLUSDT": "Solana",
            "ADAUSDT": "Cardano",
            "DOGEUSDT": "Dogecoin",
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
            "1d": "1 Day (Long-term)",
        }

        for i, (tf, desc) in enumerate(timeframe_info.items(), 1):
            table.add_row(str(i), tf, desc)

        return table

    def display_configuration_summary(self, config: SessionConfig) -> None:
        """Display enhanced configuration summary with risk-based capital management."""
        self.console.print("\n📋 Configuration Summary", style="bold cyan")

        summary_table = Table(title="Session Configuration")
        summary_table.add_column("Parameter", style="cyan", width=25)
        summary_table.add_column("Value", style="white", width=25)

        # Trading Configuration
        strategy_name = str(config.strategy)
        summary_table.add_row("Strategy", strategy_name.replace("_", " ").title())

        symbol_name = str(config.symbol)
        summary_table.add_row("Symbol", symbol_name)

        timeframe_name = str(config.timeframe)
        summary_table.add_row("Timeframe", timeframe_name)
        
        # Capital Management
        summary_table.add_row("", "")  # Separator
        summary_table.add_row("[bold]Capital Management[/bold]", "")
        summary_table.add_row("Total Capital", f"₹{config.total_capital_inr:,.0f}")
        summary_table.add_row("Trading Capital", f"₹{config.trading_capital_inr:,.0f} ({config.trading_capital_pct}%)")
        summary_table.add_row("Backup Capital", f"₹{config.backup_capital_inr:,.0f}")
        summary_table.add_row("Risk per Trade", f"₹{config.risk_amount_per_trade_inr:,.0f} ({config.risk_per_trade_pct}%)")
        
        # Position Parameters
        summary_table.add_row("", "")  # Separator
        summary_table.add_row("[bold]Position Parameters[/bold]", "")
        summary_table.add_row("Stop Loss", "🧮 Calculated by risk amount")
        summary_table.add_row("Take Profit", f"{config.take_profit_pct}%")
        summary_table.add_row("Leverage", f"{config.leverage}x")

        self.console.print(summary_table)

    def display_market_data_summary(self, market_data: MarketData) -> None:
        """Display market data summary with immediate candlestick pattern analysis."""
        self.console.print("\n📊 Market Data Summary", style="bold cyan")

        # Create market data table
        data_table = Table(title=f"{market_data.symbol} - {market_data.timeframe}")
        data_table.add_column("Metric", style="cyan", width=20)
        data_table.add_column("Value", style="white", width=20)

        # Get latest candle
        latest_candle = market_data.latest_ohlcv if market_data.ohlcv_data else None

        if latest_candle:
            data_table.add_row("Current Price", f"${market_data.current_price:,.2f}")
            data_table.add_row("Open", f"${latest_candle.open:,.2f}")
            data_table.add_row("High", f"${latest_candle.high:,.2f}")
            data_table.add_row("Low", f"${latest_candle.low:,.2f}")
            data_table.add_row("Close", f"${latest_candle.close:,.2f}")
            data_table.add_row("Volume", f"{latest_candle.volume:,.0f}")

        data_table.add_row("Data Points", str(len(market_data.ohlcv_data)))

        self.console.print(data_table)
        
        # Add immediate candlestick pattern analysis
        if latest_candle and len(market_data.ohlcv_data) >= 10:
            self._display_immediate_candlestick_analysis(market_data)

    def display_analysis_results(self, analysis_result: AnalysisResult) -> None:
        """Display analysis results."""
        self.console.print("\n🧠 AI Analysis Results", style="bold magenta")

        # Display AI analysis text
        self.console.print(
            Panel(
                analysis_result.ai_analysis, title="🤖 Ollama AI Analysis", style="blue"
            )
        )

    def display_trading_signals(
        self, analysis_result: AnalysisResult, config: SessionConfig
    ) -> None:
        """Display trading signals."""
        self.console.print("\n📈 Trading Signals", style="bold green")

        if not analysis_result.signals:
            self.console.print("⚠️ No trading signals generated", style="yellow")
            return

        # Create signals table with enhanced columns
        signals_table = Table(title="Generated Signals")
        signals_table.add_column("Signal", style="cyan", width=8)
        signals_table.add_column("Action", style="white", width=8)
        signals_table.add_column("Confidence", style="white", width=10)
        signals_table.add_column("Candle Time", style="white", width=17)
        signals_table.add_column("Formation", style="yellow", width=25)
        signals_table.add_column("Entry Price", style="white", width=12)
        signals_table.add_column("Risk/Reward", style="white", width=10)

        for i, signal in enumerate(analysis_result.signals, 1):
            # Calculate stop loss and take profit
            entry_price = float(signal.entry_price)

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

            # Get formation display with styling
            formation_display = signal.formation_display
            if signal.candle_formation and signal.candle_formation.is_strong_pattern:
                formation_display = f"[bold]{formation_display}[/bold]"
            
            signals_table.add_row(
                f"#{i}",
                f"[{action_style}]{action_upper}[/{action_style}]",
                f"{signal.confidence}/10",
                signal.candle_time_display,
                formation_display,
                f"${entry_price:,.2f}",
                f"{risk_reward:.2f}:1",
            )

        self.console.print(signals_table)

        # Display primary signal if available
        if analysis_result.primary_signal:
            primary = analysis_result.primary_signal
            # Handle primary signal action display (Pydantic converts enums to strings)
            primary_action = str(primary.action)
            primary_action_upper = primary_action.upper()
            primary_style = "green" if primary_action_upper == "BUY" else "red"

            # Build enhanced signal display content
            signal_content = (
                f"[bold]Primary Signal: [{primary_style}]{primary_action_upper}[/{primary_style}][/bold]\n"
                f"Confidence: {primary.confidence}/10\n"
                f"Entry Price: ${float(primary.entry_price):,.2f}\n"
            )
            
            # Add candle timing information
            if primary.candle_timestamp:
                signal_content += f"Candle Time: {primary.candle_time_display}\n"
            
            # Add candlestick formation details
            if primary.candle_formation:
                formation = primary.candle_formation
                strength_indicator = "🔥" if formation.is_strong_pattern else "📈"
                signal_content += (
                    f"Formation: {strength_indicator} {formation.pattern_display_name} "
                    f"({formation.strength}/10)\n"
                    f"Pattern Type: {formation.pattern_type.title()}\n"
                    f"Description: {formation.visual_description}\n"
                )
                
                # Add volume confirmation if available
                if formation.volume_confirmation:
                    signal_content += "Volume Confirmation: ✅ Confirmed\n"
                else:
                    signal_content += "Volume Confirmation: ⚠️ Not Confirmed\n"
            else:
                signal_content += "Formation: Standard Candle\n"
            
            # Add pattern context if available
            if primary.pattern_context:
                signal_content += f"Context: {primary.pattern_context}\n"
            
            # Add reasoning (use enhanced reasoning if available)
            reasoning_text = primary.enhanced_reasoning if hasattr(primary, 'enhanced_reasoning') else primary.reasoning
            signal_content += f"Reasoning: {reasoning_text}"

            self.console.print(
                Panel(
                    signal_content,
                    title="🎯 Primary Signal",
                    style="yellow",
                )
            )

        # Risk management reminder
        self.display_risk_management_reminder(config)

    def display_risk_management_reminder(self, config: SessionConfig) -> None:
        """Display enhanced risk management with capital breakdown."""
        self.console.print(
            Panel(
                f"💰 [bold]Capital Management[/bold]\n"
                f"• Total Capital: ₹{config.total_capital_inr:,.0f}\n"
                f"• Trading Capital: ₹{config.trading_capital_inr:,.0f} ({config.trading_capital_pct}%)\n"
                f"• Backup Capital: ₹{config.backup_capital_inr:,.0f}\n"
                f"• Risk per Trade: ₹{config.risk_amount_per_trade_inr:,.0f} ({config.risk_per_trade_pct}%)\n\n"
                f"📊 [bold]Position Parameters[/bold]\n"
                f"• Stop Loss: 🧮 Calculated based on risk amount per trade\n"
                f"• Take Profit: {config.take_profit_pct}% from entry\n"
                f"• Leverage: {config.leverage}x\n\n"
                f"⚠️ [bold]Risk Disclaimer[/bold]\n"
                f"• These are educational signals only\n"
                f"• Always do your own research\n"
                f"• Never risk more than you can afford to lose",
                title="🛡️ Risk Management",
                style="red",
            )
        )

    def display_position_sizing_details(self, quantity_lots: float, position_value_inr: float, risk_breakdown: dict) -> None:
        """Display detailed position sizing calculation results."""
        self.console.print("\n🎯 Position Sizing Calculation", style="bold cyan")
        
        # Create position sizing table
        sizing_table = Table(title="Risk-Based Position Calculation")
        sizing_table.add_column("Parameter", style="cyan", width=25)
        sizing_table.add_column("Value", style="white", width=20)
        sizing_table.add_column("Details", style="dim", width=30)
        
        # Capital breakdown
        sizing_table.add_row(
            "Total Capital",
            f"₹{risk_breakdown['total_capital_inr']:,.0f}",
            "Available capital"
        )
        sizing_table.add_row(
            "Trading Capital",
            f"₹{risk_breakdown['trading_capital_inr']:,.0f}",
            f"{risk_breakdown['total_capital_inr'] * risk_breakdown['risk_per_trade_pct'] / 100:.0f}% of total"
        )
        sizing_table.add_row(
            "Risk Amount",
            f"₹{risk_breakdown['risk_amount_inr']:,.0f}",
            f"{risk_breakdown['risk_per_trade_pct']}% of trading capital"
        )
        sizing_table.add_row(
            "Stop Loss Points",
            f"${risk_breakdown['stop_loss_points']:,.2f}",
            "Entry to stop loss distance"
        )
        
        # Position calculation
        sizing_table.add_row(
            "Calculated Quantity",
            f"{risk_breakdown['quantity_base']:.6f}",
            "Risk Amount ÷ Stop Loss Points"
        )
        sizing_table.add_row(
            "Adjusted Quantity",
            f"{quantity_lots:.6f}",
            "Rounded to minimum lot size"
        )
        sizing_table.add_row(
            "Position Value",
            f"₹{position_value_inr:,.0f}",
            "Quantity × Current Price"
        )
        sizing_table.add_row(
            "Margin Required",
            f"₹{risk_breakdown['margin_required_inr']:,.0f}",
            f"Position Value ÷ {risk_breakdown.get('leverage', 10)}x leverage"
        )
        sizing_table.add_row(
            "Effective Leverage",
            f"{risk_breakdown['effective_leverage']:.2f}x",
            "Position Value ÷ Trading Capital"
        )
        sizing_table.add_row(
            "Risk/Reward Ratio",
            f"{risk_breakdown['risk_reward_ratio']:.2f}:1",
            "Take Profit % ÷ Stop Loss %"
        )
        
        self.console.print(sizing_table)
        
        # Risk validation summary
        effective_leverage = risk_breakdown['effective_leverage']
        if effective_leverage > 2.0:
            leverage_status = "[red]⚠️ High[/red]"
        elif effective_leverage > 1.0:
            leverage_status = "[yellow]📊 Moderate[/yellow]"
        else:
            leverage_status = "[green]✅ Conservative[/green]"
            
        self.console.print(
            Panel(
                f"📊 [bold]Position Summary[/bold]\n"
                f"Formula Used: Quantity = Risk Amount ÷ Stop Loss Points\n"
                f"Risk Amount: ₹{risk_breakdown['risk_amount_inr']:,.0f}\n"
                f"Stop Loss Distance: ${risk_breakdown['stop_loss_points']:,.2f}\n"
                f"Final Quantity: {quantity_lots:.6f} lots\n"
                f"Leverage Status: {leverage_status}",
                title="💡 Calculation Summary",
                style="blue",
            )
        )

    def _display_immediate_candlestick_analysis(self, market_data: MarketData) -> None:
        """Display immediate candlestick pattern analysis from OHLCV data."""
        from src.analysis.indicators import TechnicalIndicators
        
        self.console.print("\n🕯️ Current Candle Analysis", style="bold yellow")
        
        try:
            # Create technical indicators instance for pattern analysis
            tech_indicators = TechnicalIndicators()
            
            # Get candlestick formation from OHLCV data
            formation = tech_indicators.create_candlestick_formation(market_data.ohlcv_data)
            latest_candle = market_data.latest_ohlcv
            
            # Create candle analysis table
            candle_table = Table(title="Candlestick Pattern Details")
            candle_table.add_column("Property", style="cyan", width=18)
            candle_table.add_column("Value", style="white", width=40)
            
            # Add timestamp
            from src.utils.helpers import format_ist_time
            candle_table.add_row(
                "Candle Time", 
                format_ist_time(latest_candle.timestamp, include_seconds=True)
            )
            
            # Determine basic candle type
            is_bullish = latest_candle.close > latest_candle.open
            candle_type = "🟢 Bullish" if is_bullish else "🔴 Bearish"
            candle_table.add_row("Candle Type", candle_type)
            
            # Add formation details if detected
            if formation:
                # Consistency validation: ensure pattern direction matches candle type
                pattern_is_bullish = formation.signal_direction in ["strong_bullish", "bullish"]
                pattern_is_bearish = formation.signal_direction in ["strong_bearish", "bearish"]
                
                if (is_bullish and pattern_is_bearish) or (not is_bullish and pattern_is_bullish):
                    # Log inconsistency for debugging
                    self.console.print("⚠️  Pattern inconsistency detected - pattern may need recalibration", style="yellow dim")
                    import structlog
                    logger = structlog.get_logger(__name__)
                    logger.warning(
                        "Candlestick pattern inconsistency detected",
                        candle_bullish=is_bullish,
                        pattern_direction=formation.signal_direction,
                        pattern_name=formation.pattern_name
                    )
                # Add fire emoji for strong patterns
                pattern_display = f"🔥 {formation.pattern_display_name}" if formation.is_strong_pattern else f"📈 {formation.pattern_display_name}"
                candle_table.add_row("Pattern", pattern_display)
                candle_table.add_row("Strength", f"{formation.strength}/10")
                candle_table.add_row("Signal Direction", formation.signal_direction.replace('_', ' ').title())
                candle_table.add_row("Description", formation.visual_description)
                candle_table.add_row("Trend Context", formation.trend_context.title())
            else:
                candle_table.add_row("Pattern", "Standard Candle")
                candle_table.add_row("Strength", "N/A")
                candle_table.add_row("Description", "Regular price movement")
            
            # Add body/shadow analysis
            body_size = abs(latest_candle.close - latest_candle.open)
            total_range = latest_candle.high - latest_candle.low
            body_ratio = (body_size / total_range * 100) if total_range > 0 else 0
            
            candle_table.add_row("Body Size", f"{body_ratio:.1f}% of total range")
            
            # Add volume analysis
            if latest_candle.volume > 0:
                # Compare with average volume if we have enough data
                if len(market_data.ohlcv_data) >= 10:
                    recent_volumes = [c.volume for c in market_data.ohlcv_data[-10:] if c.volume > 0]
                    if recent_volumes:
                        avg_volume = sum(recent_volumes) / len(recent_volumes)
                        volume_ratio = latest_candle.volume / avg_volume
                        volume_status = "🔥 High" if volume_ratio > 1.5 else "📊 Normal" if volume_ratio > 0.8 else "📉 Low"
                        candle_table.add_row("Volume Status", f"{volume_status} ({volume_ratio:.1f}x avg)")
                    else:
                        candle_table.add_row("Volume Status", "📊 Normal")
                else:
                    candle_table.add_row("Volume Status", f"{latest_candle.volume:,.0f}")
            else:
                candle_table.add_row("Volume Status", "⚠️ No Volume Data")
            
            self.console.print(candle_table)
                
        except Exception as e:
            self.console.print(f"⚠️ Could not analyze candlestick pattern: {e}", style="yellow")
