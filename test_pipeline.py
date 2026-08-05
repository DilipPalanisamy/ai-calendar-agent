from parser import parse_schedule_message
from calendar_service import create_google_calendar_event

try:
    print("1. Testing Gemini Parsing...")
    parsed = parse_schedule_message("Strategy sync meeting tomorrow at 3pm")
    print("Parsed Result:", parsed)

    print("\n2. Testing Google Calendar Event Creation...")
    link = create_google_calendar_event(parsed)
    print("SUCCESS! Calendar Event Link:", link)

except Exception as e:
    print("\n❌ ERROR ENCOUNTERED:")
    import traceback
    traceback.print_exc()