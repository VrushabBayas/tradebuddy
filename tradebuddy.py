#!/usr/bin/env python3
"""
TradeBuddy - Simple Entry Point

Easy way to run TradeBuddy with proper imports.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for TradeBuddy."""
    try:
        # Import here to avoid import issues
        from src.cli.main import main as cli_main
        
        print("🚀 Starting TradeBuddy...")
        asyncio.run(cli_main())
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted. Goodbye!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the TradeBuddy directory")
        print("2. Activate virtual environment: source venv/bin/activate")
        print("3. Install dependencies: pip install -r requirements/base.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Check if we're in development mode
        if os.getenv('DEBUG', '').lower() == 'true':
            import traceback
            print("\n🐛 Debug traceback:")
            traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":
    main()