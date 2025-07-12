"""
Real-time analysis module for TradeBuddy CLI.

Handles live market data streaming and real-time strategy analysis.
"""

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import structlog

from src.core.config import settings
from src.core.models import (
    StrategyType, Symbol, TimeFrame, SessionConfig, OHLCV, MarketData,
    RealTimeConfig
)
from src.data.delta_client import DeltaExchangeClient
from src.data.websocket_client import DeltaWebSocketClient, process_candlestick_data

logger = structlog.get_logger(__name__)


class RealTimeAnalyzer:
    """
    Real-time market analysis with live WebSocket streaming.
    
    Features:
    - Live candlestick data streaming
    - Historical data pre-loading
    - Real-time strategy analysis
    - Live signal generation
    - Interactive configuration
    """
    
    def __init__(self, console: Console, delta_client: DeltaExchangeClient, 
                 websocket_client: DeltaWebSocketClient, strategies: dict):
        """
        Initialize real-time analyzer.
        
        Args:
            console: Rich console for display
            delta_client: Delta Exchange API client
            websocket_client: WebSocket client for live data
            strategies: Available strategy instances
        """
        self.console = console
        self.delta_client = delta_client
        self.websocket_client = websocket_client
        self.strategies = strategies
        
        # Analysis state
        self.ohlcv_buffer = deque(maxlen=50)
        self.analysis_count = 0
        self.config: Optional[RealTimeConfig] = None
        
        logger.info("Real-time analyzer initialized")
    
    async def run_session(self) -> None:
        """Run a complete real-time analysis session."""
        self.console.print(f"\n🚀 Real-Time Analysis Mode", style="bold green")
        
        try:
            # Step 1: Configure session
            self.config = await self.configure_session()
            if not self.config:
                return
            
            # Step 2: Reset state
            self._reset_session_state()
            
            # Step 3: Load historical data
            if not await self.load_historical_data():
                return
            
            # Step 4: Start real-time streaming
            await self.start_streaming()
            
        except Exception as e:
            logger.error("Real-time session failed", error=str(e))
            self.console.print(f"\n❌ Real-time analysis failed: {e}", style="red")
            
            if settings.is_development:
                import traceback
                self.console.print("\n🐛 Traceback:", style="dim")
                self.console.print(traceback.format_exc(), style="dim")
        
        # Wait for user input
        Prompt.ask("\n[dim]Press Enter to continue...[/dim]", default="")
    
    async def configure_session(self) -> Optional[RealTimeConfig]:
        """Configure real-time analysis session with user input."""
        self.console.print(f"\n⚙️ Real-Time Configuration", style="bold cyan")
        
        # Strategy selection
        strategy = await self._select_strategy()
        if not strategy:
            return None
        
        # Duration selection
        duration = await self._select_duration()
        if not duration:
            return None
        
        # Create configuration
        config = RealTimeConfig(
            strategy=strategy,
            duration_minutes=duration
        )
        
        # Display configuration summary
        self._display_config_summary(config)
        
        return config
    
    async def _select_strategy(self) -> Optional[StrategyType]:
        """Interactive strategy selection for real-time analysis."""
        self.console.print("\n🎯 Select Strategy for Real-Time Analysis", style="bold")
        
        strategy_table = Table(title="Real-Time Strategies")
        strategy_table.add_column("Option", style="cyan", width=6)
        strategy_table.add_column("Strategy", style="white", width=20)
        
        strategy_table.add_row("1", "EMA Crossover")
        strategy_table.add_row("2", "Combined Strategy")
        
        self.console.print(strategy_table)
        
        choice = Prompt.ask(
            "\nSelect strategy for real-time analysis",
            choices=["1", "2"],
            default="1"
        )
        
        strategy_map = {
            "1": StrategyType.EMA_CROSSOVER,
            "2": StrategyType.COMBINED
        }
        
        return strategy_map[choice]
    
    async def _select_duration(self) -> Optional[int]:
        """Interactive duration selection."""
        duration = Prompt.ask(
            "\nAnalysis duration (minutes)",
            default="5",
            show_default=True
        )
        
        try:
            duration_minutes = int(duration)
            if duration_minutes <= 0 or duration_minutes > 60:
                self.console.print("❌ Duration must be between 1-60 minutes", style="red")
                return None
            return duration_minutes
        except ValueError:
            self.console.print("❌ Invalid duration format", style="red")
            return None
    
    def _display_config_summary(self, config: RealTimeConfig) -> None:
        """Display configuration summary."""
        self.console.print("\n📋 Real-Time Configuration Summary", style="bold cyan")
        
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan", width=20)
        config_table.add_column("Value", style="white", width=20)
        
        config_table.add_row("Strategy", str(config.strategy).replace('_', ' ').title())
        config_table.add_row("Symbol", str(config.symbol))
        config_table.add_row("Timeframe", str(config.timeframe))
        config_table.add_row("Duration", f"{config.duration_minutes} minutes")
        config_table.add_row("Buffer Size", f"{config.buffer_size} candles")
        
        self.console.print(config_table)
    
    def _reset_session_state(self) -> None:
        """Reset session state for new analysis."""
        self.ohlcv_buffer.clear()
        self.analysis_count = 0
        
        # Update buffer size based on config
        if self.config:
            self.ohlcv_buffer = deque(maxlen=self.config.buffer_size)
    
    async def load_historical_data(self) -> bool:
        """Load historical data to prime the analysis buffer."""
        if not self.config:
            return False
            
        self.console.print("\n📚 Loading historical data to prime analysis buffer...", style="bold yellow")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Fetching historical candles...", total=None)
                
                # Get recent historical data
                historical_data = await self.delta_client.get_market_data(
                    symbol=self.config.symbol.value,
                    timeframe=self.config.timeframe.value,
                    limit=self.config.historical_candles
                )
                
                # Add to buffer
                for ohlcv in historical_data.ohlcv_data:
                    self.ohlcv_buffer.append(ohlcv)
                
                progress.update(task, description=f"✅ Loaded {len(historical_data.ohlcv_data)} historical candles")
                await asyncio.sleep(0.5)
            
            self.console.print(f"💾 Buffer ready with {len(self.ohlcv_buffer)} candles", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Failed to load historical data: {e}", style="red")
            return False
    
    async def start_streaming(self) -> None:
        """Start real-time WebSocket streaming and analysis."""
        if not self.config:
            return
            
        self.console.print(f"\n📡 Starting Real-Time Market Streaming", style="bold green")
        
        # Get strategy instance
        strategy_instance = self.strategies[self.config.strategy]
        
        try:
            # Connect to WebSocket
            await self.websocket_client.connect()
            self.console.print("✅ Connected to Delta Exchange WebSocket", style="green")
            
            # Subscribe to candlestick data
            await self.websocket_client.subscribe_candlestick(
                symbol=self.config.symbol.value,
                timeframe=self.config.timeframe.value,
                callback=lambda data: self.handle_candlestick(data, strategy_instance)
            )
            
            self.console.print(f"📊 Subscribed to {self.config.symbol} live candlesticks", style="green")
            self.console.print(f"⏳ Real-time analysis for {self.config.duration_minutes} minutes...\n", style="cyan")
            
            # Start timer
            start_time = asyncio.get_event_loop().time()
            timeout_seconds = self.config.duration_minutes * 60
            
            # Listen for messages
            async for message in self.websocket_client.listen():
                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout_seconds:
                    self.console.print(f"\n⏰ Real-time session completed ({self.config.duration_minutes} minutes)", style="cyan")
                    break
                
                # Check analysis count limit
                if self.analysis_count >= self.config.max_analysis_count:
                    self.console.print(f"\n📊 Analysis limit reached ({self.config.max_analysis_count} analyses)", style="cyan")
                    break
                    
        except KeyboardInterrupt:
            self.console.print("\n🛑 Real-time session stopped by user", style="cyan")
        except Exception as e:
            self.console.print(f"❌ Real-time streaming error: {e}", style="red")
        finally:
            await self.websocket_client.disconnect()
            self.console.print("✅ Disconnected from WebSocket", style="green")
    
    async def handle_candlestick(self, data, strategy_instance):
        """Handle incoming real-time candlestick data."""
        self.analysis_count += 1
        
        self.console.print(f"\n📊 LIVE CANDLESTICK #{self.analysis_count}", style="bold yellow")
        self.console.print(f"   Close: ${data.get('close', 0):,.2f}")
        self.console.print(f"   Volume: {data.get('volume', 0)}")
        
        # Process candlestick
        ohlcv = process_candlestick_data(data)
        if not ohlcv:
            self.console.print("❌ Failed to process candlestick", style="red")
            return
        
        # Add to buffer
        self.ohlcv_buffer.append(ohlcv)
        self.console.print(f"💾 Buffer: {len(self.ohlcv_buffer)} candles")
        
        # Run analysis if we have enough data
        if len(self.ohlcv_buffer) >= 20:
            await self.run_strategy_analysis(strategy_instance)
        else:
            needed = 20 - len(self.ohlcv_buffer)
            self.console.print(f"⏳ Need {needed} more candles for analysis", style="yellow")
    
    async def run_strategy_analysis(self, strategy_instance):
        """Run strategy analysis on live data."""
        if not self.config:
            return
            
        try:
            self.console.print("🧠 Running LIVE strategy analysis...", style="bold magenta")
            
            # Create market data from buffer
            ohlcv_list = list(self.ohlcv_buffer)
            current_price = float(ohlcv_list[-1].close)
            
            # Create session config for strategy
            session_config = SessionConfig(
                strategy=self.config.strategy,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                confidence_threshold=self.config.confidence_threshold
            )
            
            market_data = MarketData(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                current_price=current_price,
                ohlcv_data=ohlcv_list,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Run strategy analysis
            result = await strategy_instance.analyze(market_data, session_config)
            
            # Display live results
            self.display_analysis_results(result, current_price)
            
        except Exception as e:
            self.console.print(f"❌ Live analysis failed: {e}", style="red")
    
    def display_analysis_results(self, result, current_price: float):
        """Display live analysis results in a compact format."""
        self.console.print("\n" + "="*60, style="cyan")
        self.console.print("🎯 LIVE ANALYSIS RESULTS", style="bold cyan")
        self.console.print("="*60, style="cyan")
        
        self.console.print(f"💰 Current Price: ${current_price:,.2f}", style="bold white")
        
        # Primary signal
        if result.primary_signal:
            signal = result.primary_signal
            action_style = "green" if str(signal.action).upper() == "BUY" else "red"
            self.console.print(f"🚨 PRIMARY SIGNAL: [{action_style}]{signal.action}[/{action_style}]", style="bold")
            self.console.print(f"   Confidence: {signal.confidence}/10")
            self.console.print(f"   Entry: ${signal.entry_price:,.2f}")
        else:
            self.console.print("🔍 No primary signal", style="yellow")
        
        # EMA info if available
        if hasattr(result, 'ema_crossover') and result.ema_crossover:
            ema = result.ema_crossover
            if hasattr(ema, 'ema_9'):
                crossover_style = "green" if ema.is_golden_cross else "red"
                crossover_type = "Golden Cross" if ema.is_golden_cross else "Death Cross"
                self.console.print(f"📈 EMA: 9=${ema.ema_9:,.2f} | 15=${ema.ema_15:,.2f}")
                self.console.print(f"   [{crossover_style}]{crossover_type}[/{crossover_style}] (Strength: {ema.crossover_strength}/10)")
        
        self.console.print("="*60, style="cyan")