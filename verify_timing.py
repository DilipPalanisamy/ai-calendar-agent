import os
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from parser import parse_schedule_message, _fallback_parse
from calendar_service import (
    _format_rfc3339_with_tz,
    _api_time,
    get_calendar_service,
    create_google_calendar_event,
    delete_google_calendar_event,
    get_primary_calendar_timezone,
    sync_calendar_timezone,
)
from bot import format_display_time

def test_timing():
    print("--- 1. Testing Fallback Parser with 'meeting with dilp at 2 pm' ---")
    res_fb = _fallback_parse("meeting with dilp at 2 pm", "Asia/Kolkata")
    evt_fb = res_fb.events[0]
    print(f"Fallback Event: {evt_fb.event_name}, Start: {evt_fb.start_time}, End: {evt_fb.end_time}")
    assert "14:00:00" in evt_fb.start_time, f"Expected 14:00:00, got {evt_fb.start_time}"
    assert "14:30:00" in evt_fb.end_time, f"Expected 14:30:00, got {evt_fb.end_time}"

    print("\n--- 2. Testing Fallback Parser with 'meeting with 2 people at 3 pm' ---")
    res_num = _fallback_parse("meeting with 2 people at 3 pm", "Asia/Kolkata")
    evt_num = res_num.events[0]
    print(f"Number disambiguation Event: {evt_num.event_name}, Start: {evt_num.start_time}")
    assert "15:00:00" in evt_num.start_time, f"Expected 15:00:00, got {evt_num.start_time}"

    print("\n--- 3. Testing LLM Parser with 'meeting with dilp at 2 pm' ---")
    res_llm = parse_schedule_message("meeting with dilp at 2 pm", "Asia/Kolkata")
    evt_llm = res_llm.events[0]
    print(f"LLM Event: {evt_llm.event_name}, Start: {evt_llm.start_time}, End: {evt_llm.end_time}")
    assert "14:00:00" in evt_llm.start_time, f"Expected 14:00:00, got {evt_llm.start_time}"
    assert "14:30:00" in evt_llm.end_time, f"Expected 14:30:00, got {evt_llm.end_time}"

    print("\n--- 4. Testing RFC3339 Formatting ---")
    rfc = _format_rfc3339_with_tz(evt_llm.start_time, "Asia/Kolkata")
    print(f"RFC3339 Output: {rfc}")
    assert "+05:30" in rfc, f"Expected +05:30 offset, got {rfc}"
    assert "14:00:00" in rfc, f"Expected 14:00:00, got {rfc}"

    print("\n--- 5. Testing Display Formatting ---")
    disp = format_display_time(evt_llm.start_time, evt_llm.end_time, "Asia/Kolkata")
    print(f"Telegram Display:\n{disp}")
    assert "2:00 PM" in disp or "02:00 PM" in disp, f"Expected 2:00 PM in display, got {disp}"

    print("\n--- 6. Testing Google Calendar Event Creation & Verification ---")
    svc = get_calendar_service()
    sync_calendar_timezone(svc, "Asia/Kolkata")
    link = create_google_calendar_event(evt_llm)
    print(f"Created Event Link: {link}")
    
    # Retrieve created event to verify Google Calendar stored time
    cal_list = svc.events().list(calendarId='primary', q=evt_llm.event_name, maxResults=5).execute()
    items = cal_list.get('items', [])
    assert items, "Event not found in Google Calendar"
    created_item = items[0]
    print(f"Google Calendar Item: summary='{created_item.get('summary')}', start={created_item.get('start')}")
    
    # Cleanup test event
    delete_google_calendar_event(created_item['id'])
    print("Test event cleaned up successfully.")

    print("\n✅ ALL TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_timing()
