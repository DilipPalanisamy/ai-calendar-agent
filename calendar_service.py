import re
import datetime
import base64
import json
import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.readonly'
]
BASE_DIR = Path(__file__).resolve().parent
TOKEN_PATH = BASE_DIR / 'token.json'
CREDENTIALS_PATH = BASE_DIR / 'credentials.json'


def _json_secret(*names):
    """Load a JSON secret from an environment variable, optionally base64 encoded."""
    for name in names:
        value = os.getenv(name)
        if not value:
            continue
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                decoded = base64.b64decode(value).decode('utf-8')
                return json.loads(decoded)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                continue
    return None


def _has_required_scopes(creds) -> bool:
    if not creds or not creds.valid:
        return False
    granted_scopes = set(getattr(creds, 'scopes', []) or [])
    return set(SCOPES).issubset(granted_scopes)


def get_google_credentials():
    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except Exception:
            creds = None
    else:
        token_info = _json_secret('GOOGLE_TOKEN_JSON', 'GOOGLE_TOKEN_BASE64', 'GOOGLE_TOKEN')
        if token_info:
            try:
                creds = Credentials.from_authorized_user_info(token_info, SCOPES)
            except Exception:
                creds = None

    if not creds or not creds.valid or not _has_required_scopes(creds):
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds or not creds.valid or not _has_required_scopes(creds):
            client_config = _json_secret(
                'GOOGLE_CREDENTIALS_JSON',
                'GOOGLE_CREDENTIALS_BASE64',
                'GOOGLE_CREDENTIALS',
            )
            if not CREDENTIALS_PATH.exists() and not client_config:
                raise FileNotFoundError(f'Missing credentials file: {CREDENTIALS_PATH}')

            if TOKEN_PATH.exists():
                TOKEN_PATH.unlink(missing_ok=True)

            if client_config:
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                creds = flow.run_console()

        TOKEN_PATH.write_text(creds.to_json(), encoding='utf-8')

    return creds


def get_calendar_service():
    creds = get_google_credentials()
    return build('calendar', 'v3', credentials=creds)


def get_primary_calendar_timezone(service) -> str:
    """Returns the primary Google Calendar timezone used for local event times."""
    configured_timezone = os.getenv('CALENDAR_TIMEZONE')
    if configured_timezone:
        return configured_timezone

    calendar = service.calendarList().get(calendarId='primary').execute()
    return calendar.get('timeZone', 'UTC')


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
    calendar_timezone = get_primary_calendar_timezone(service)

    start_dt = event.start_time
    end_dt = event.end_time

    event_body = {
        'summary': event.event_name,
        'location': getattr(event, 'location', 'Not specified'),
        'description': f'Created by AI Calendar Agent. Priority: {getattr(event, "priority", "Medium")}',
        'start': {
            'dateTime': start_dt,
            'timeZone': calendar_timezone,
        },
        'end': {
            'dateTime': end_dt,
            'timeZone': calendar_timezone,
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