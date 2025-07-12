"""
Main entry point for TradeBuddy application.

This module provides the entry point when running `python -m src`.
"""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel

from src.cli.main import main as cli_main
from src.core.config import settings
from src.utils.environment import print_validation_result, validate_environment

console = Console()


async def app_main():
    """Main application entry point with environment validation."""

    # Display startup banner
    console.print(
        Panel(
            f"[bold cyan]TradeBuddy v0.1.0[/bold cyan]\n"
            f"AI-Powered Trading Signal Analysis\n\n"
            f"Environment: [yellow]{settings.python_env}[/yellow]\n"
            f"Log Level: [yellow]{settings.log_level}[/yellow]",
            title="🚀 Starting TradeBuddy",
            style="blue",
        )
    )

    # Validate environment
    console.print("🔍 Validating environment...", style="yellow")
    validation_result = await validate_environment()

    if not validation_result.is_valid:
        console.print(f"\n❌ Environment validation failed:", style="red")
        console.print(f"   {validation_result.error}", style="red")

        if validation_result.warnings:
            console.print("\n⚠️ Additional warnings:", style="yellow")
            for warning in validation_result.warnings:
                console.print(f"   • {warning}", style="yellow")

        console.print(
            "\n💡 Please fix the environment issues and try again.", style="cyan"
        )
        return False

    # Show warnings if any
    if validation_result.warnings:
        console.print("\n⚠️ Environment warnings:", style="yellow")
        for warning in validation_result.warnings:
            console.print(f"   • {warning}", style="yellow")
        console.print("")

    # Environment validation passed
    console.print("✅ Environment validation passed!", style="green")

    # Show detailed validation results in development mode
    if settings.is_development:
        print_validation_result(validation_result)

    # Start the CLI application
    await cli_main()

    return True


def main():
    """Synchronous main function for script execution."""
    try:
        # Run the async main function
        success = asyncio.run(app_main())

        if not success:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="cyan")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ Fatal error: {e}", style="red")

        # Show traceback in development mode
        if settings.is_development:
            import traceback

            console.print("\n🐛 Traceback:", style="dim")
            console.print(traceback.format_exc(), style="dim")

        sys.exit(1)


if __name__ == "__main__":
    main()
