#!/usr/bin/env python3
"""
Test IST timezone conversion functionality.
"""

from datetime import datetime, timezone
from src.utils.helpers import utc_to_ist, format_ist_time, format_ist_time_only

def test_ist_conversion():
    print("🕐 Testing IST Timezone Conversion")
    print("=" * 50)
    
    # Test case 1: Your screenshot timestamp (2025-07-12 05:30 PM UTC)
    test_utc = datetime(2025, 7, 12, 17, 30, 0, tzinfo=timezone.utc)  # 5:30 PM UTC
    
    print(f"Test Case 1:")
    print(f"  UTC Time: {test_utc.strftime('%Y-%m-%d %I:%M:%S %p UTC')}")
    
    ist_converted = utc_to_ist(test_utc)
    print(f"  IST Time: {ist_converted.strftime('%Y-%m-%d %I:%M:%S %p IST')}")
    
    formatted_full = format_ist_time(test_utc, include_seconds=True)
    print(f"  Formatted (with seconds): {formatted_full}")
    
    formatted_time_only = format_ist_time_only(test_utc, include_seconds=True)
    print(f"  Time Only (with seconds): {formatted_time_only}")
    
    # Expected: 2025-07-12 11:00:00 PM IST (5:30 PM + 5:30 hours = 11:00 PM)
    expected_hour = 23  # 11 PM in 24-hour format
    expected_minute = 0
    
    print(f"\n✅ Conversion Test:")
    print(f"  Expected: 11:00:00 PM IST")
    print(f"  Actual:   {ist_converted.strftime('%I:%M:%S %p')} IST")
    
    if ist_converted.hour == expected_hour and ist_converted.minute == expected_minute:
        print("  ✅ Time conversion is CORRECT!")
    else:
        print("  ❌ Time conversion is INCORRECT!")
    
    print("\n" + "=" * 50)
    
    # Test case 2: Current time
    print(f"Test Case 2 - Current Time:")
    current_utc = datetime.now(timezone.utc)
    current_ist = utc_to_ist(current_utc)
    
    print(f"  Current UTC: {current_utc.strftime('%Y-%m-%d %I:%M:%S %p UTC')}")
    print(f"  Current IST: {format_ist_time(current_utc, include_seconds=True)}")
    
    # Test case 3: Edge cases
    print(f"\nTest Case 3 - Edge Cases:")
    
    # Midnight UTC
    midnight_utc = datetime(2025, 7, 12, 0, 0, 0, tzinfo=timezone.utc)
    print(f"  Midnight UTC: {midnight_utc.strftime('%Y-%m-%d %I:%M:%S %p UTC')}")
    print(f"  In IST: {format_ist_time(midnight_utc, include_seconds=True)}")
    
    # Noon UTC
    noon_utc = datetime(2025, 7, 12, 12, 0, 0, tzinfo=timezone.utc)
    print(f"  Noon UTC: {noon_utc.strftime('%Y-%m-%d %I:%M:%S %p UTC')}")
    print(f"  In IST: {format_ist_time(noon_utc, include_seconds=True)}")
    
    print("\n🎯 Format Examples for UI:")
    print(f"  Full format: {format_ist_time(test_utc, include_seconds=True)}")
    print(f"  Time only: {format_ist_time_only(test_utc, include_seconds=True)}")
    print(f"  Without seconds: {format_ist_time(test_utc, include_seconds=False)}")

if __name__ == "__main__":
    test_ist_conversion()