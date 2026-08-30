import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

from calendar_service import find_linked_travel_buffer_ids, delete_google_calendar_event
from main import delete_calendar_event_tool


class MockGoogleResource:
    """Mock Google Calendar API Events Resource."""
    def __init__(self, events_db=None):
        self.events_db = events_db or {}
        self.deleted_ids = []

    def get(self, calendarId, eventId):
        mock_req = MagicMock()
        if eventId in self.events_db:
            mock_req.execute.return_value = self.events_db[eventId]
        else:
            mock_req.execute.side_effect = Exception("Event not found")
        return mock_req

    def list(self, **kwargs):
        mock_req = MagicMock()
        items = list(self.events_db.values())
        q = kwargs.get("q")
        if q:
            items = [ev for ev in items if q.lower() in ev.get("summary", "").lower()]
        mock_req.execute.return_value = {"items": items}
        return mock_req

    def delete(self, calendarId, eventId, **kwargs):
        mock_req = MagicMock()
        if eventId in self.events_db:
            self.deleted_ids.append(eventId)
            del self.events_db[eventId]
            mock_req.execute.return_value = {}
        else:
            # Idempotent or not found
            self.deleted_ids.append(eventId)
            mock_req.execute.return_value = {}
        return mock_req


class MockGoogleCalendarListResource:
    def get(self, calendarId):
        mock_req = MagicMock()
        mock_req.execute.return_value = {"timeZone": "Asia/Kolkata"}
        return mock_req

    def list(self):
        mock_req = MagicMock()
        mock_req.execute.return_value = {"items": [{"id": "primary", "primary": True, "timeZone": "Asia/Kolkata"}]}
        return mock_req


class MockGoogleService:
    def __init__(self, events_db=None):
        self._events = MockGoogleResource(events_db)
        self._cal_list = MockGoogleCalendarListResource()

    def events(self):
        return self._events

    def calendarList(self):
        return self._cal_list


class TestTravelCascadeDeletion(unittest.TestCase):

    def test_find_linked_by_extended_properties_on_main(self):
        main_event = {
            "id": "main_123",
            "summary": "Client Strategy Meeting",
            "location": "MG Road Office, Bangalore",
            "start": {"dateTime": "2026-09-01T15:00:00+05:30"},
            "end": {"dateTime": "2026-09-01T16:00:00+05:30"},
            "extendedProperties": {
                "private": {
                    "travel_buffer_event_id": "travel_buf_456",
                    "has_travel_buffer": "true",
                }
            }
        }
        mock_svc = MockGoogleService({"main_123": main_event})
        linked = find_linked_travel_buffer_ids(mock_svc, "primary", "main_123", event_data=main_event)
        self.assertIn("travel_buf_456", linked)
        self.assertEqual(len(linked), 1)

    def test_find_linked_by_parent_event_id_on_buffer(self):
        main_event = {
            "id": "main_999",
            "summary": "Doctor Appointment",
            "location": "Apollo Hospital",
            "start": {"dateTime": "2026-09-01T10:00:00+05:30"},
            "end": {"dateTime": "2026-09-01T11:00:00+05:30"},
        }
        travel_event = {
            "id": "buf_999",
            "summary": "🚗 Travel to Apollo Hospital",
            "description": "Automated 30-minute travel buffer before 'Doctor Appointment'.",
            "start": {"dateTime": "2026-09-01T09:30:00+05:30"},
            "end": {"dateTime": "2026-09-01T10:00:00+05:30"},
            "extendedProperties": {
                "private": {
                    "is_travel_buffer": "true",
                    "parent_event_id": "main_999",
                }
            }
        }
        mock_svc = MockGoogleService({"main_999": main_event, "buf_999": travel_event})
        linked = find_linked_travel_buffer_ids(mock_svc, "primary", "main_999", event_data=main_event)
        self.assertIn("buf_999", linked)

    def test_find_linked_by_heuristic_timing_and_summary(self):
        main_event = {
            "id": "main_legacy",
            "summary": "Dinner with Alex",
            "location": "Indiranagar 100ft Rd",
            "start": {"dateTime": "2026-09-01T20:00:00+05:30"},
            "end": {"dateTime": "2026-09-01T21:30:00+05:30"},
        }
        travel_event = {
            "id": "buf_legacy",
            "summary": "🚗 Travel to Indiranagar 100ft Rd",
            "description": "Automated 30-minute travel buffer before 'Dinner with Alex'.",
            "start": {"dateTime": "2026-09-01T19:30:00+05:30"},
            "end": {"dateTime": "2026-09-01T20:00:00+05:30"},
        }
        mock_svc = MockGoogleService({"main_legacy": main_event, "buf_legacy": travel_event})
        linked = find_linked_travel_buffer_ids(mock_svc, "primary", "main_legacy", event_data=main_event)
        self.assertIn("buf_legacy", linked)

    def test_sync_delete_google_calendar_event_cascades(self):
        main_event = {
            "id": "event_to_del",
            "summary": "Gym Workout",
            "location": "Cult Fit",
            "start": {"dateTime": "2026-09-02T07:00:00+05:30"},
            "end": {"dateTime": "2026-09-02T08:00:00+05:30"},
            "extendedProperties": {
                "private": {"travel_buffer_event_id": "buf_to_del"}
            }
        }
        travel_event = {
            "id": "buf_to_del",
            "summary": "🚗 Travel to Cult Fit",
            "start": {"dateTime": "2026-09-02T06:30:00+05:30"},
            "end": {"dateTime": "2026-09-02T07:00:00+05:30"},
        }
        mock_svc = MockGoogleService({"event_to_del": main_event, "buf_to_del": travel_event})

        with patch("calendar_service.get_calendar_service", return_value=mock_svc):
            success = delete_google_calendar_event("event_to_del")
            self.assertTrue(success)
            self.assertIn("event_to_del", mock_svc._events.deleted_ids)
            self.assertIn("buf_to_del", mock_svc._events.deleted_ids)
            self.assertEqual(len(mock_svc._events.events_db), 0)

    def test_async_delete_tool_cascades_travel_buffer(self):
        main_event = {
            "id": "meeting_abc",
            "summary": "Design Sprint Review",
            "location": "HSR Layout Studio",
            "start": {"dateTime": "2026-09-03T14:00:00+05:30"},
            "end": {"dateTime": "2026-09-03T15:00:00+05:30"},
            "extendedProperties": {
                "private": {"travel_buffer_event_id": "buf_abc"}
            }
        }
        travel_event = {
            "id": "buf_abc",
            "summary": "🚗 Travel to HSR Layout Studio",
            "start": {"dateTime": "2026-09-03T13:30:00+05:30"},
            "end": {"dateTime": "2026-09-03T14:00:00+05:30"},
        }
        mock_svc = MockGoogleService({"meeting_abc": main_event, "buf_abc": travel_event})
        mock_creds = MagicMock()

        async def run_test():
            with patch("main.build", return_value=mock_svc):
                res_str = await delete_calendar_event_tool(
                    creds=mock_creds,
                    event_id="meeting_abc"
                )
                res = json.loads(res_str)
                self.assertEqual(res["status"], "success")
                self.assertIn("meeting_abc", res["deleted_event_ids"])
                self.assertIn("buf_abc", res["deleted_travel_buffers"])
                self.assertIn("meeting_abc", mock_svc._events.deleted_ids)
                self.assertIn("buf_abc", mock_svc._events.deleted_ids)
                self.assertEqual(len(mock_svc._events.events_db), 0)

        asyncio.run(run_test())

    def test_event_without_travel_buffer_deletes_normally(self):
        main_event = {
            "id": "zoom_call_123",
            "summary": "Quick Virtual Sync",
            "location": "",
            "start": {"dateTime": "2026-09-04T16:00:00+05:30"},
            "end": {"dateTime": "2026-09-04T16:30:00+05:30"},
        }
        mock_svc = MockGoogleService({"zoom_call_123": main_event})
        mock_creds = MagicMock()

        async def run_test():
            with patch("main.build", return_value=mock_svc):
                res_str = await delete_calendar_event_tool(
                    creds=mock_creds,
                    event_id="zoom_call_123"
                )
                res = json.loads(res_str)
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["deleted_count"], 1)
                self.assertEqual(res["deleted_travel_buffers"], [])
                self.assertIn("zoom_call_123", mock_svc._events.deleted_ids)

        asyncio.run(run_test())

    def test_bulk_deletion_cascades_all_travel_buffers(self):
        e1 = {
            "id": "e1",
            "summary": "Meeting 1",
            "location": "Loc 1",
            "start": {"dateTime": "2026-09-05T10:00:00+05:30"},
            "end": {"dateTime": "2026-09-05T11:00:00+05:30"},
            "extendedProperties": {"private": {"travel_buffer_event_id": "tb1"}}
        }
        tb1 = {
            "id": "tb1",
            "summary": "🚗 Travel to Loc 1",
            "start": {"dateTime": "2026-09-05T09:30:00+05:30"},
            "end": {"dateTime": "2026-09-05T10:00:00+05:30"},
        }
        e2 = {
            "id": "e2",
            "summary": "Meeting 2",
            "location": "Loc 2",
            "start": {"dateTime": "2026-09-05T14:00:00+05:30"},
            "end": {"dateTime": "2026-09-05T15:00:00+05:30"},
            "extendedProperties": {"private": {"travel_buffer_event_id": "tb2"}}
        }
        tb2 = {
            "id": "tb2",
            "summary": "🚗 Travel to Loc 2",
            "start": {"dateTime": "2026-09-05T13:30:00+05:30"},
            "end": {"dateTime": "2026-09-05T14:00:00+05:30"},
        }
        mock_svc = MockGoogleService({"e1": e1, "tb1": tb1, "e2": e2, "tb2": tb2})
        mock_creds = MagicMock()

        async def run_test():
            with patch("main.build", return_value=mock_svc):
                res_str = await delete_calendar_event_tool(
                    creds=mock_creds,
                    event_ids=["e1", "e2"]
                )
                res = json.loads(res_str)
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["deleted_count"], 4)
                self.assertIn("e1", res["deleted_event_ids"])
                self.assertIn("e2", res["deleted_event_ids"])
                self.assertIn("tb1", res["deleted_travel_buffers"])
                self.assertIn("tb2", res["deleted_travel_buffers"])
                self.assertEqual(len(mock_svc._events.events_db), 0)

        asyncio.run(run_test())

    def test_api_calendar_delete_endpoint_cascades(self):
        from main import delete_calendar_event_endpoint, DeleteEventRequest

        main_event = {
            "id": "api_event_1",
            "summary": "Partner Lunch",
            "location": "Toit, Indiranagar",
            "start": {"dateTime": "2026-09-06T13:00:00+05:30"},
            "end": {"dateTime": "2026-09-06T14:00:00+05:30"},
            "extendedProperties": {"private": {"travel_buffer_event_id": "api_tb_1"}}
        }
        travel_event = {
            "id": "api_tb_1",
            "summary": "🚗 Travel to Toit, Indiranagar",
            "start": {"dateTime": "2026-09-06T12:30:00+05:30"},
            "end": {"dateTime": "2026-09-06T13:00:00+05:30"},
        }
        mock_svc = MockGoogleService({"api_event_1": main_event, "api_tb_1": travel_event})
        mock_request = MagicMock()

        async def run_test():
            with patch("main.get_user_credentials", return_value=MagicMock()), \
                 patch("main.build", return_value=mock_svc):
                response = await delete_calendar_event_endpoint(
                    request=mock_request,
                    body=DeleteEventRequest(event_id="api_event_1")
                )
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.body.decode("utf-8"))
                self.assertEqual(data["status"], "success")
                self.assertEqual(data["event_id"], "api_event_1")
                self.assertIn("api_tb_1", data["deleted_travel_buffers"])
                self.assertIn("api_event_1", mock_svc._events.deleted_ids)
                self.assertIn("api_tb_1", mock_svc._events.deleted_ids)
                self.assertEqual(len(mock_svc._events.events_db), 0)

        asyncio.run(run_test())

    def test_format_google_calendar_link(self):
        from calendar_service import format_google_calendar_link

        # Standard event link
        link1 = "https://www.google.com/calendar/event?eid=YWJjMTIz"
        formatted1 = format_google_calendar_link(link1, "user@gmail.com")
        self.assertEqual(formatted1, "https://www.google.com/calendar/event?eid=YWJjMTIz&authuser=user@gmail.com")

        # Link with existing authuser replaced
        link2 = "https://www.google.com/calendar/event?eid=YWJjMTIz&authuser=old@gmail.com"
        formatted2 = format_google_calendar_link(link2, "new@gmail.com")
        self.assertEqual(formatted2, "https://www.google.com/calendar/event?eid=YWJjMTIz&authuser=new@gmail.com")

        # Empty link fallback
        formatted3 = format_google_calendar_link(None, "user@gmail.com")
        self.assertEqual(formatted3, "https://calendar.google.com/calendar/r?authuser=user@gmail.com")

    def test_create_event_tool_returns_authuser_link(self):
        from main import create_event_tool

        class MockInsertResource(MockGoogleResource):
            def insert(self, calendarId, body, **kwargs):
                mock_req = MagicMock()
                mock_req.execute.return_value = {
                    "id": "new_created_event",
                    "htmlLink": "https://www.google.com/calendar/event?eid=bmV3X2NyZWF0ZWQ=",
                    "summary": body.get("summary"),
                }
                return mock_req

            def patch(self, calendarId, eventId, body, **kwargs):
                mock_req = MagicMock()
                mock_req.execute.return_value = {"id": eventId, **body}
                return mock_req

        mock_svc = MockGoogleService()
        mock_svc._events = MockInsertResource()
        mock_creds = MagicMock()

        async def run_test():
            with patch("main.build", return_value=mock_svc):
                res_str = await create_event_tool(
                    creds=mock_creds,
                    summary="Leadership Sync",
                    start_time="2026-09-10T11:00:00+05:30",
                    end_time="2026-09-10T12:00:00+05:30",
                    location="Bangalore HQ",
                    user_email="dilip@gmail.com"
                )
                data = json.loads(res_str)
                self.assertEqual(data["status"], "success")
                self.assertIn("authuser=dilip@gmail.com", data["htmlLink"])
                self.assertIn("authuser=dilip@gmail.com", data["google_calendar_link"])

        asyncio.run(run_test())

    def test_normalize_iso_datetime_converts_all_timezones_accurately(self):
        from main import normalize_iso_datetime

        # 1. Naive 3:00 PM -> 15:00:00+05:30
        res1 = normalize_iso_datetime("2026-08-30T15:00:00", "Asia/Kolkata")
        self.assertEqual(res1, "2026-08-30T15:00:00+05:30")

        # 2. Already IST +05:30
        res2 = normalize_iso_datetime("2026-08-30T15:00:00+05:30", "Asia/Kolkata")
        self.assertEqual(res2, "2026-08-30T15:00:00+05:30")

        # 3. UTC string with Z (09:30:00Z -> 15:00:00+05:30)
        res3 = normalize_iso_datetime("2026-08-30T09:30:00Z", "Asia/Kolkata")
        self.assertEqual(res3, "2026-08-30T15:00:00+05:30")

        # 4. UTC string with +00:00 (09:30:00+00:00 -> 15:00:00+05:30)
        res4 = normalize_iso_datetime("2026-08-30T09:30:00+00:00", "Asia/Kolkata")
        self.assertEqual(res4, "2026-08-30T15:00:00+05:30")

        # 5. Space separated datetime "2026-08-30 15:00:00"
        res5 = normalize_iso_datetime("2026-08-30 15:00:00", "Asia/Kolkata")
        self.assertEqual(res5, "2026-08-30T15:00:00+05:30")


if __name__ == "__main__":
    unittest.main()

