import os.path
import re
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']


def get_calendar_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds)


def check_calendar_conflict(start_time_iso: str, end_time_iso: str) -> list:
    """Checks if there are existing events overlapping with the requested window."""
    if not start_time_iso or not end_time_iso:
        return []

    service = get_calendar_service()
    start_dt = start_time_iso if ("Z" in start_time_iso or "+" in start_time_iso) else f"{start_time_iso}Z"
    end_dt = end_time_iso if ("Z" in end_time_iso or "+" in end_time_iso) else f"{end_time_iso}Z"

    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_dt,
        timeMax=end_dt,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    conflicts = events_result.get('items', [])
    return [event.get('summary', 'Untitled Event') for event in conflicts]


def find_event_by_title(title_query: str, time_min_iso: str = None) -> dict:
    """Searches primary calendar for an event matching a title/summary query."""
    service = get_calendar_service()

    clean_query = re.sub(r"\b(tomorrow|today|next\s+\w+)\b", "", title_query, flags=re.IGNORECASE).strip()

    if not time_min_iso:
        time_min_iso = datetime.datetime.utcnow().isoformat() + "Z"
    elif "Z" not in time_min_iso and "+" not in time_min_iso:
        time_min_iso = f"{time_min_iso}Z"

    events_result = service.events().list(
        calendarId='primary',
        q=clean_query,
        timeMin=time_min_iso,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    items = events_result.get('items', [])
    return items[0] if items else None


def delete_google_calendar_event(event_id: str) -> bool:
    """Deletes an event from Google Calendar by ID."""
    service = get_calendar_service()
    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return True
    except Exception:
        return False


def reschedule_google_calendar_event(event_id: str, new_start_iso: str, new_end_iso: str) -> str:
    """Updates start and end time of an existing Google Calendar event."""
    service = get_calendar_service()

    start_dt = new_start_iso if ("Z" in new_start_iso or "+" in new_start_iso) else f"{new_start_iso}Z"
    end_dt = new_end_iso if ("Z" in new_end_iso or "+" in new_end_iso) else f"{new_end_iso}Z"

    event = service.events().get(calendarId='primary', eventId=event_id).execute()
    event['start']['dateTime'] = start_dt
    event['end']['dateTime'] = end_dt

    updated_event = service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
    return updated_event.get('htmlLink')


def create_google_calendar_event(event):
    """Creates a new event on Google Calendar."""
    service = get_calendar_service()

    start_dt = event.start_time if ("Z" in event.start_time or "+" in event.start_time) else f"{event.start_time}Z"
    end_dt = event.end_time if ("Z" in event.end_time or "+" in event.end_time) else f"{event.end_time}Z"

    event_body = {
        'summary': event.event_name,
        'location': getattr(event, 'location', 'Not specified'),
        'description': f'Created by AI Calendar Agent. Priority: {getattr(event, "priority", "Medium")}',
        'start': {
            'dateTime': start_dt,
            'timeZone': 'UTC',
        },
        'end': {
            'dateTime': end_dt,
            'timeZone': 'UTC',
        },
    }

    created_event = service.events().insert(calendarId='primary', body=event_body).execute()
    return created_event.get('htmlLink')


def list_google_calendar_events(start_time_iso: str, end_time_iso: str) -> list:
    """Retrieves all events within a specific time window."""
    service = get_calendar_service()

    start_dt = start_time_iso if ("Z" in start_time_iso or "+" in start_time_iso) else f"{start_time_iso}Z"
    end_dt = end_time_iso if ("Z" in end_time_iso or "+" in end_time_iso) else f"{end_time_iso}Z"

    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_dt,
        timeMax=end_dt,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    items = events_result.get('items', [])
    events_list = []
    for item in items:
        events_list.append({
            "summary": item.get('summary', 'Untitled Event'),
            "start": item.get('start', {}).get('dateTime', item.get('start', {}).get('date')),
            "end": item.get('end', {}).get('dateTime', item.get('end', {}).get('date')),
            "link": item.get('htmlLink')
        })
    return events_list