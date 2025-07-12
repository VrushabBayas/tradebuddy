"""
Real-time analysis module for TradeBuddy CLI.

Handles live market data streaming and real-time strategy analysis.
"""

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from src.core.config import settings
from src.core.models import (
    OHLCV,
    MarketData,
    MonitoringConfig,
    RealTimeConfig,
    SessionConfig,
    StrategyType,
    Symbol,
    TimeFrame,
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
    - Continuous market monitoring mode
    """

    def __init__(
        self,
        console: Console,
        delta_client: DeltaExchangeClient,
        websocket_client: DeltaWebSocketClient,
        strategies: dict,
    ):
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

        # Buffer management for monitoring mode
        self.monitoring_buffers = {}
        self.signal_history = {}
        self.last_buffer_refresh = {}

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
        config = RealTimeConfig(strategy=strategy, duration_minutes=duration)

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
            "\nSelect strategy for real-time analysis", choices=["1", "2"], default="1"
        )

        strategy_map = {"1": StrategyType.EMA_CROSSOVER, "2": StrategyType.COMBINED}

        return strategy_map[choice]

    async def _select_duration(self) -> Optional[int]:
        """Interactive duration selection."""
        duration = Prompt.ask(
            "\nAnalysis duration (minutes)", default="5", show_default=True
        )

        try:
            duration_minutes = int(duration)
            if duration_minutes <= 0 or duration_minutes > 60:
                self.console.print(
                    "❌ Duration must be between 1-60 minutes", style="red"
                )
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

        config_table.add_row("Strategy", str(config.strategy).replace("_", " ").title())
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

        self.console.print(
            "\n📚 Loading historical data to prime analysis buffer...",
            style="bold yellow",
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Fetching historical candles...", total=None)

                # Get recent historical data
                historical_data = await self.delta_client.get_market_data(
                    symbol=str(self.config.symbol),
                    timeframe=str(self.config.timeframe),
                    limit=self.config.historical_candles,
                )

                # Add to buffer
                for ohlcv in historical_data.ohlcv_data:
                    self.ohlcv_buffer.append(ohlcv)

                progress.update(
                    task,
                    description=f"✅ Loaded {len(historical_data.ohlcv_data)} historical candles",
                )
                await asyncio.sleep(0.5)

            self.console.print(
                f"💾 Buffer ready with {len(self.ohlcv_buffer)} candles", style="green"
            )
            return True

        except Exception as e:
            self.console.print(f"❌ Failed to load historical data: {e}", style="red")
            return False

    async def start_streaming(self) -> None:
        """Start real-time WebSocket streaming and analysis."""
        if not self.config:
            return

        self.console.print(
            f"\n📡 Starting Real-Time Market Streaming", style="bold green"
        )

        # Get strategy instance
        strategy_instance = self.strategies[self.config.strategy]

        try:
            # Connect to WebSocket
            await self.websocket_client.connect()
            self.console.print("✅ Connected to Delta Exchange WebSocket", style="green")

            # Subscribe to candlestick data
            await self.websocket_client.subscribe_candlestick(
                symbol=str(self.config.symbol),
                timeframe=str(self.config.timeframe),
                callback=lambda data: self.handle_candlestick(data, strategy_instance),
            )

            self.console.print(
                f"📊 Subscribed to {self.config.symbol} live candlesticks", style="green"
            )
            self.console.print(
                f"⏳ Real-time analysis for {self.config.duration_minutes} minutes...\n",
                style="cyan",
            )

            # Start timer
            start_time = asyncio.get_event_loop().time()
            timeout_seconds = self.config.duration_minutes * 60

            # Listen for messages
            async for message in self.websocket_client.listen():
                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout_seconds:
                    self.console.print(
                        f"\n⏰ Real-time session completed ({self.config.duration_minutes} minutes)",
                        style="cyan",
                    )
                    break

                # Check analysis count limit
                if self.analysis_count >= self.config.max_analysis_count:
                    self.console.print(
                        f"\n📊 Analysis limit reached ({self.config.max_analysis_count} analyses)",
                        style="cyan",
                    )
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

        self.console.print(
            f"\n📊 LIVE CANDLESTICK #{self.analysis_count}", style="bold yellow"
        )
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
            self.console.print(
                f"⏳ Need {needed} more candles for analysis", style="yellow"
            )

    async def run_strategy_analysis(self, strategy_instance):
        """Run strategy analysis on live data."""
        if not self.config:
            return

        try:
            self.console.print(
                "🧠 Running LIVE strategy analysis...", style="bold magenta"
            )

            # Create market data from buffer
            ohlcv_list = list(self.ohlcv_buffer)
            current_price = float(ohlcv_list[-1].close)

            # Create session config for strategy
            session_config = SessionConfig(
                strategy=self.config.strategy,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                confidence_threshold=self.config.confidence_threshold,
            )

            market_data = MarketData(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                current_price=current_price,
                ohlcv_data=ohlcv_list,
                timestamp=datetime.now(timezone.utc),
            )

            # Run strategy analysis
            result = await strategy_instance.analyze(market_data, session_config)

            # Display live results
            self.display_analysis_results(result, current_price)

        except Exception as e:
            self.console.print(f"❌ Live analysis failed: {e}", style="red")

    def display_analysis_results(self, result, current_price: float):
        """Display live analysis results in a compact format."""
        self.console.print("\n" + "=" * 60, style="cyan")
        self.console.print("🎯 LIVE ANALYSIS RESULTS", style="bold cyan")
        self.console.print("=" * 60, style="cyan")

        self.console.print(
            f"💰 Current Price: ${current_price:,.2f}", style="bold white"
        )

        # Primary signal
        if result.primary_signal:
            signal = result.primary_signal
            action_style = "green" if str(signal.action).upper() == "BUY" else "red"
            self.console.print(
                f"🚨 PRIMARY SIGNAL: [{action_style}]{signal.action}[/{action_style}]",
                style="bold",
            )
            self.console.print(f"   Confidence: {signal.confidence}/10")
            self.console.print(f"   Entry: ${signal.entry_price:,.2f}")
            
            # Add candle timing information
            if signal.candle_timestamp:
                self.console.print(f"   Candle Time: {signal.candle_time_display}")
            
            # Add candlestick formation details in compact format
            if signal.candle_formation:
                formation = signal.candle_formation
                strength_indicator = "🔥" if formation.is_strong_pattern else "📈"
                self.console.print(f"   Formation: {strength_indicator} {formation.pattern_display_name} ({formation.strength}/10)")
                volume_status = "✅" if formation.volume_confirmation else "⚠️"
                self.console.print(f"   Volume: {volume_status} {'Confirmed' if formation.volume_confirmation else 'Not Confirmed'}")
            else:
                self.console.print("   Formation: Standard Candle")
        else:
            self.console.print("🔍 No primary signal", style="yellow")

        # EMA info if available
        if hasattr(result, "ema_crossover") and result.ema_crossover:
            ema = result.ema_crossover
            if hasattr(ema, "ema_9"):
                crossover_style = "green" if ema.is_golden_cross else "red"
                crossover_type = (
                    "Golden Cross" if ema.is_golden_cross else "Death Cross"
                )
                self.console.print(
                    f"📈 EMA: 9=${ema.ema_9:,.2f} | 15=${ema.ema_15:,.2f}"
                )
                self.console.print(
                    f"   [{crossover_style}]{crossover_type}[/{crossover_style}] (Strength: {ema.crossover_strength}/10)"
                )

        self.console.print("=" * 60, style="cyan")

    async def run_monitoring_session(self) -> None:
        """Run continuous market monitoring mode."""
        self.console.print(f"\n🚀 Continuous Market Monitoring Mode", style="bold green")

        try:
            # Step 1: Configure monitoring session
            config = await self.configure_monitoring_session()
            if not config:
                return

            # Step 2: Initialize monitoring
            if not await self.initialize_monitoring(config):
                return

            # Step 3: Start continuous monitoring
            await self.start_continuous_monitoring(config)

        except Exception as e:
            logger.error("Monitoring session failed", error=str(e))
            self.console.print(f"\n❌ Monitoring failed: {e}", style="red")

            if settings.is_development:
                import traceback

                self.console.print("\n🐛 Traceback:", style="dim")
                self.console.print(traceback.format_exc(), style="dim")

        # Wait for user input
        Prompt.ask("\n[dim]Press Enter to return to main menu...[/dim]", default="")

    async def configure_monitoring_session(self) -> Optional[MonitoringConfig]:
        """Configure continuous monitoring session with user input."""
        self.console.print(f"\n⚙️ Monitoring Configuration", style="bold cyan")

        # Strategy selection
        strategy = await self._select_monitoring_strategy()
        if not strategy:
            return None

        # Symbol selection (multi-select)
        symbols = await self._select_monitoring_symbols()
        if not symbols:
            return None

        # Monitoring parameters
        params = await self._configure_monitoring_parameters()
        if not params:
            return None

        # Create configuration
        config = MonitoringConfig(
            strategy=strategy,
            symbols=symbols,
            signal_threshold=params["signal_threshold"],
            refresh_interval=params["refresh_interval"],
            show_neutral_signals=params["show_neutral"],
            compact_display=params["compact_display"],
        )

        # Display configuration summary
        self._display_monitoring_config_summary(config)

        return config

    async def _select_monitoring_strategy(self) -> Optional[StrategyType]:
        """Interactive strategy selection for monitoring."""
        self.console.print(
            "\n🎯 Select Strategy for Continuous Monitoring", style="bold"
        )

        strategy_table = Table(title="Monitoring Strategies")
        strategy_table.add_column("Option", style="cyan", width=6)
        strategy_table.add_column("Strategy", style="white", width=20)
        strategy_table.add_column("Best For", style="dim", width=30)

        strategy_table.add_row(
            "1", "EMA Crossover", "Trend detection across timeframes"
        )
        strategy_table.add_row("2", "Combined Strategy", "High-confidence signals only")

        self.console.print(strategy_table)

        choice = Prompt.ask(
            "\nSelect strategy for monitoring", choices=["1", "2"], default="2"
        )

        strategy_map = {"1": StrategyType.EMA_CROSSOVER, "2": StrategyType.COMBINED}

        selected = strategy_map[choice]
        self.console.print(
            f"✅ Selected: {selected.value.replace('_', ' ').title()}", style="green"
        )
        return selected

    async def _select_monitoring_symbols(self) -> Optional[List[Symbol]]:
        """Interactive symbol selection for monitoring."""
        self.console.print("\n📊 Select Symbols to Monitor", style="bold")

        symbol_table = Table(title="Available Symbols")
        symbol_table.add_column("Option", style="cyan", width=6)
        symbol_table.add_column("Symbol", style="white", width=15)
        symbol_table.add_column("Name", style="dim", width=20)

        symbol_info = {
            "1": (Symbol.BTCUSDT, "Bitcoin"),
            "2": (Symbol.ETHUSDT, "Ethereum"),
            "3": (Symbol.SOLUSDT, "Solana"),
            "4": (Symbol.ADAUSDT, "Cardano"),
            "5": (Symbol.DOGEUSDT, "Dogecoin"),
        }

        for option, (symbol, name) in symbol_info.items():
            symbol_table.add_row(option, symbol.value, name)

        self.console.print(symbol_table)

        choices = Prompt.ask(
            "\nSelect symbols (comma-separated, e.g., 1,2,3)", default="1"
        )

        try:
            selected_options = [opt.strip() for opt in choices.split(",")]
            symbols = []

            for opt in selected_options:
                if opt in symbol_info:
                    symbols.append(symbol_info[opt][0])

            if not symbols:
                self.console.print("❌ No valid symbols selected", style="red")
                return None

            symbol_names = [s.value for s in symbols]
            self.console.print(f"✅ Selected: {', '.join(symbol_names)}", style="green")
            return symbols

        except Exception:
            self.console.print("❌ Invalid symbol selection", style="red")
            return None

    async def _configure_monitoring_parameters(self) -> Optional[dict]:
        """Configure monitoring parameters."""
        self.console.print("\n⚙️ Monitoring Parameters", style="bold yellow")

        # Signal threshold
        signal_threshold = Prompt.ask(
            "Signal confidence threshold (5-10)", default="7", show_default=True
        )

        # Refresh interval
        refresh_interval = Prompt.ask(
            "Refresh interval in seconds (30-300)", default="60", show_default=True
        )

        # Display options
        show_neutral = (
            Prompt.ask(
                "Show neutral/wait signals? (y/n)", default="n", show_default=True
            ).lower()
            == "y"
        )

        compact_display = (
            Prompt.ask(
                "Use compact display? (y/n)", default="y", show_default=True
            ).lower()
            == "y"
        )

        try:
            params = {
                "signal_threshold": int(signal_threshold),
                "refresh_interval": int(refresh_interval),
                "show_neutral": show_neutral,
                "compact_display": compact_display,
            }

            # Validate parameters
            if not 5 <= params["signal_threshold"] <= 10:
                raise ValueError("Signal threshold must be between 5 and 10")
            if not 30 <= params["refresh_interval"] <= 300:
                raise ValueError("Refresh interval must be between 30 and 300 seconds")

            self.console.print("✅ Monitoring parameters configured", style="green")
            return params

        except ValueError as e:
            self.console.print(f"❌ Invalid parameters: {e}", style="red")
            return None

    def _display_monitoring_config_summary(self, config: MonitoringConfig) -> None:
        """Display monitoring configuration summary."""
        self.console.print("\n📋 Monitoring Configuration Summary", style="bold cyan")

        config_table = Table(title="Monitoring Setup")
        config_table.add_column("Parameter", style="cyan", width=20)
        config_table.add_column("Value", style="white", width=30)

        config_table.add_row("Strategy", str(config.strategy).replace("_", " ").title())
        config_table.add_row("Symbols", ", ".join([str(s) for s in config.symbols]))
        config_table.add_row("Timeframe", str(config.timeframe))
        config_table.add_row("Signal Threshold", f"{config.signal_threshold}/10")
        config_table.add_row("Refresh Interval", f"{config.refresh_interval} seconds")
        config_table.add_row(
            "Show Neutral", "Yes" if config.show_neutral_signals else "No"
        )
        config_table.add_row(
            "Compact Display", "Yes" if config.compact_display else "No"
        )

        self.console.print(config_table)

    async def initialize_monitoring(self, config: MonitoringConfig) -> bool:
        """Initialize monitoring buffers and historical data."""
        self.console.print(
            "\n📚 Initializing monitoring buffers...", style="bold yellow"
        )

        # Create symbol buffers
        self.monitoring_buffers = {}
        self.signal_history = {}

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                for symbol in config.symbols:
                    task = progress.add_task(
                        f"Loading {symbol} data...", total=None
                    )

                    # Initialize buffer for this symbol
                    self.monitoring_buffers[symbol] = deque(maxlen=config.buffer_size)
                    self.signal_history[symbol] = deque(
                        maxlen=50
                    )  # Keep last 50 signals

                    # Load historical data
                    historical_data = await self.delta_client.get_market_data(
                        symbol=str(symbol),
                        timeframe=str(config.timeframe),
                        limit=config.historical_candles,
                    )

                    # Add to buffer
                    for ohlcv in historical_data.ohlcv_data:
                        self.monitoring_buffers[symbol].append(ohlcv)

                    progress.update(
                        task,
                        description=f"✅ {symbol} ready ({len(historical_data.ohlcv_data)} candles)",
                    )
                    await asyncio.sleep(0.2)

            self.console.print(
                f"💾 Monitoring initialized for {len(config.symbols)} symbols",
                style="green",
            )
            return True

        except Exception as e:
            self.console.print(f"❌ Failed to initialize monitoring: {e}", style="red")
            return False

    async def start_continuous_monitoring(self, config: MonitoringConfig) -> None:
        """Start continuous market monitoring with periodic analysis."""
        self.console.print(
            f"\n📡 Starting Continuous Market Monitoring", style="bold green"
        )
        self.console.print(
            f"🔄 Analyzing every {config.refresh_interval} seconds", style="cyan"
        )
        self.console.print(f"⚠️ Press Ctrl+C to stop monitoring\n", style="yellow")

        # Get strategy instance
        strategy_instance = self.strategies[config.strategy]
        analysis_count = 0

        try:
            while True:
                analysis_count += 1

                self.console.print(
                    f"🔍 Analysis #{analysis_count} - {datetime.now().strftime('%H:%M:%S')}",
                    style="bold cyan",
                )

                # Analyze each symbol
                for symbol in config.symbols:
                    await self.analyze_symbol_for_monitoring(
                        symbol, config, strategy_instance
                    )

                # Display monitoring summary if not compact
                if not config.compact_display:
                    self.display_monitoring_summary(config)

                # Wait for next analysis
                self.console.print(
                    f"⏳ Next analysis in {config.refresh_interval} seconds...\n",
                    style="dim",
                )
                await asyncio.sleep(config.refresh_interval)

        except KeyboardInterrupt:
            self.console.print(f"\n🛑 Monitoring stopped by user", style="cyan")
            self.console.print(
                f"📊 Total analyses performed: {analysis_count}", style="green"
            )
        except Exception as e:
            self.console.print(f"❌ Monitoring error: {e}", style="red")

    async def analyze_symbol_for_monitoring(
        self, symbol: Symbol, config: MonitoringConfig, strategy_instance
    ):
        """Analyze a single symbol during monitoring."""
        try:
            # Check if buffer needs refresh (every 10 minutes)
            await self._refresh_buffer_if_needed(symbol, config)

            # Get latest market data for this symbol
            latest_data = await self.delta_client.get_market_data(
                symbol=str(symbol),
                timeframe=str(config.timeframe),
                limit=5,  # Just get latest candles
            )

            # Update buffer with latest data
            buffer = self.monitoring_buffers[symbol]
            for ohlcv in latest_data.ohlcv_data[-2:]:  # Last 2 candles
                buffer.append(ohlcv)

            # Validate buffer data freshness
            if buffer and self._is_buffer_stale(buffer, symbol, config):
                self.console.print(
                    f"🔄 {symbol}: Refreshing stale buffer data", style="yellow"
                )
                await self._force_buffer_refresh(symbol, config)

            # Run analysis if we have enough data
            if len(buffer) >= 20:
                await self.run_monitoring_analysis(symbol, config, strategy_instance)
            else:
                needed = 20 - len(buffer)
                self.console.print(
                    f"📊 {symbol}: Need {needed} more candles", style="yellow"
                )

        except Exception as e:
            self.console.print(f"❌ {symbol} analysis failed: {e}", style="red")

    async def run_monitoring_analysis(
        self, symbol: Symbol, config: MonitoringConfig, strategy_instance
    ):
        """Run strategy analysis for monitoring mode."""
        try:
            # Create market data from buffer
            buffer = self.monitoring_buffers[symbol]
            ohlcv_list = list(buffer)
            current_price = float(ohlcv_list[-1].close)

            # Create session config for strategy
            session_config = SessionConfig(
                strategy=config.strategy,
                symbol=symbol,
                timeframe=config.timeframe,
                confidence_threshold=config.signal_threshold,
            )

            market_data = MarketData(
                symbol=symbol,
                timeframe=config.timeframe,
                current_price=current_price,
                ohlcv_data=ohlcv_list,
                timestamp=datetime.now(timezone.utc),
            )

            # Run strategy analysis
            result = await strategy_instance.analyze(market_data, session_config)

            # Process and display results
            self.process_monitoring_results(symbol, result, config, current_price)

        except Exception as e:
            self.console.print(
                f"❌ {symbol} strategy analysis failed: {e}", style="red"
            )

    def process_monitoring_results(
        self, symbol: Symbol, result, config: MonitoringConfig, current_price: float
    ):
        """Process and display monitoring analysis results."""
        # Check for significant signals
        if (
            result.primary_signal
            and result.primary_signal.confidence >= config.signal_threshold
        ):
            signal = result.primary_signal

            # Add to signal history
            self.signal_history[symbol].append(
                {"timestamp": datetime.now(), "signal": signal, "price": current_price}
            )

            # Display alert
            action_style = "green" if str(signal.action).upper() == "BUY" else "red"

            if config.compact_display:
                self.console.print(
                    f"🚨 {symbol}: [{action_style}]{signal.action}[/{action_style}] "
                    f"({signal.confidence}/10) @ ${current_price:,.2f}",
                    style="bold",
                )
            else:
                self.display_monitoring_signal_details(symbol, signal, current_price)

        elif config.show_neutral_signals:
            self.console.print(
                f"📊 {symbol}: No signals @ ${current_price:,.2f}", style="dim"
            )

    def display_monitoring_signal_details(
        self, symbol: Symbol, signal, current_price: float
    ):
        """Display detailed signal information in monitoring mode."""
        action_style = "green" if str(signal.action).upper() == "BUY" else "red"

        # Build signal content with enhanced details
        signal_content = (
            f"🚨 [bold]{symbol.value} SIGNAL ALERT[/bold]\n\n"
            f"Action: [{action_style}]{signal.action}[/{action_style}]\n"
            f"Confidence: {signal.confidence}/10\n"
            f"Current Price: ${current_price:,.2f}\n"
            f"Entry Price: ${signal.entry_price:,.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
        )
        
        # Add candle timing information
        if signal.candle_timestamp:
            signal_content += f"Candle Time: {signal.candle_time_display}\n"
        
        # Add candlestick formation details
        if signal.candle_formation:
            formation = signal.candle_formation
            strength_indicator = "🔥" if formation.is_strong_pattern else "📈"
            signal_content += (
                f"Formation: {strength_indicator} {formation.pattern_display_name} "
                f"({formation.strength}/10)\n"
                f"Pattern: {formation.pattern_type.title()}\n"
            )
            # Add volume confirmation status
            volume_status = "✅ Confirmed" if formation.volume_confirmation else "⚠️ Not Confirmed"
            signal_content += f"Volume: {volume_status}\n"
        else:
            signal_content += "Formation: Standard Candle\n"
        
        signal_content += f"\nReasoning: {signal.reasoning[:100]}..."

        self.console.print(
            Panel(
                signal_content,
                title=f"🎯 {symbol.value} Alert",
                style="yellow",
            )
        )

    def display_monitoring_summary(self, config: MonitoringConfig):
        """Display monitoring session summary."""
        self.console.print("\n📊 Monitoring Session Summary", style="bold cyan")

        summary_table = Table(title="Signal Summary")
        summary_table.add_column("Symbol", style="cyan", width=10)
        summary_table.add_column("Signals", style="white", width=8)
        summary_table.add_column("Last Signal", style="white", width=15)
        summary_table.add_column("Last Price", style="white", width=12)

        for symbol in config.symbols:
            signal_count = len(self.signal_history.get(symbol, []))
            last_signal = "None"
            last_price = "N/A"

            if signal_count > 0:
                latest = self.signal_history[symbol][-1]
                last_signal = (
                    f"{latest['signal'].action} ({latest['signal'].confidence}/10)"
                )
                last_price = f"${latest['price']:,.2f}"

            summary_table.add_row(
                symbol.value, str(signal_count), last_signal, last_price
            )

        self.console.print(summary_table)
        self.console.print("=" * 60, style="cyan")

    async def _refresh_buffer_if_needed(
        self, symbol: Symbol, config: MonitoringConfig
    ) -> None:
        """Check if buffer needs periodic refresh and refresh if needed."""
        current_time = datetime.now()
        last_refresh = self.last_buffer_refresh.get(symbol, datetime.min)

        # Refresh buffer every 10 minutes to ensure data freshness
        refresh_interval_minutes = 10

        if (
            current_time - last_refresh
        ).total_seconds() / 60 >= refresh_interval_minutes:
            self.console.print(
                f"🔄 {symbol.value}: Periodic buffer refresh", style="cyan"
            )
            await self._force_buffer_refresh(symbol, config)
            self.last_buffer_refresh[symbol] = current_time

    def _is_buffer_stale(
        self, buffer: deque, symbol: Symbol, config: MonitoringConfig
    ) -> bool:
        """Check if buffer data is stale based on timestamps."""
        if not buffer:
            return True

        current_time = datetime.now(timezone.utc)
        latest_candle = buffer[-1]

        # Calculate maximum allowed age based on timeframe
        timeframe_str = str(config.timeframe)
        if timeframe_str.endswith("m"):
            max_age_minutes = int(timeframe_str[:-1]) * 3  # 3x timeframe
        elif timeframe_str.endswith("h"):
            max_age_minutes = (
                int(timeframe_str[:-1]) * 60 * 3
            )  # 3x timeframe in minutes
        else:
            max_age_minutes = 60  # Default 1 hour

        age_minutes = (current_time - latest_candle.timestamp).total_seconds() / 60

        is_stale = age_minutes > max_age_minutes

        if is_stale:
            logger.warning(
                "Buffer data is stale",
                symbol=symbol.value,
                age_minutes=age_minutes,
                max_age_minutes=max_age_minutes,
            )

        return is_stale

    async def _force_buffer_refresh(
        self, symbol: Symbol, config: MonitoringConfig
    ) -> None:
        """Force complete refresh of buffer with fresh market data."""
        try:
            # Clear existing buffer
            if symbol in self.monitoring_buffers:
                self.monitoring_buffers[symbol].clear()

            # Get fresh historical data
            fresh_data = await self.delta_client.get_market_data(
                symbol=str(symbol),
                timeframe=str(config.timeframe),
                limit=config.historical_candles,
            )

            # Rebuild buffer with fresh data
            buffer = self.monitoring_buffers[symbol]
            for ohlcv in fresh_data.ohlcv_data:
                buffer.append(ohlcv)

            logger.info(
                "Buffer refreshed with fresh data",
                symbol=str(symbol),
                candles_loaded=len(fresh_data.ohlcv_data),
                latest_timestamp=fresh_data.ohlcv_data[-1].timestamp.isoformat()
                if fresh_data.ohlcv_data
                else None,
            )

        except Exception as e:
            logger.error(f"Failed to refresh buffer for {symbol}", error=str(e))
