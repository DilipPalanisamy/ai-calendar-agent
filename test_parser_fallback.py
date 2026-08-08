from parser import parse_schedule_message


def test_tomorrow_parsing_fallback():
    event = parse_schedule_message("Strategy sync meeting tomorrow at 3pm").events[0]
    assert event.event_name == "Strategy sync meeting"
    assert event.start_time.endswith("15:00:00")
    assert event.end_time.endswith("15:30:00")


if __name__ == "__main__":
    test_tomorrow_parsing_fallback()
    print("fallback parser test passed")
