import re
import datetime
import base64
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
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
                creds = flow.run_local_server(port=0, access_type='offline', prompt='consent')
            except Exception:
                creds = flow.run_console(access_type='offline', prompt='consent')

        TOKEN_PATH.write_text(creds.to_json(), encoding='utf-8')

    return creds


def get_calendar_service():
    creds = get_google_credentials()
    service = build('calendar', 'v3', credentials=creds)
    return service


def get_gmail_service():
    """Return an authenticated Gmail API client using the shared credentials."""
    creds = get_google_credentials()
    return build('gmail', 'v1', credentials=creds)


def find_gmail_drive_or_internship_messages(max_results: int = 5) -> list:
    """Find recent unread Gmail messages about Drive files or internships."""
    service = get_gmail_service()
    response = service.users().messages().list(
        userId='me',
        q='is:unread (drive OR internship)',
        maxResults=max_results,
    ).execute()

    messages = []
    for message in response.get('messages', []):
        email = service.users().messages().get(
            userId='me',
            id=message['id'],
            format='full',
        ).execute()
        headers = {
            header.get('name', '').lower(): header.get('value', '')
            for header in email.get('payload', {}).get('headers', [])
        }
        messages.append({
            'id': message['id'],
            'subject': headers.get('subject', 'No subject'),
            'from': headers.get('from', 'Unknown sender'),
            'date': headers.get('date', ''),
            'snippet': email.get('snippet', ''),
            'display_snippet': ' '.join(email.get('snippet', '').split())[:280],
        })
    return messages


def get_primary_calendar_timezone(service=None) -> str:
    """Returns the primary Google Calendar timezone used for local event times."""
    configured_timezone = os.getenv('CALENDAR_TIMEZONE')
    if configured_timezone:
        return configured_timezone

    if service is not None:
        try:
            calendar = service.calendarList().get(calendarId='primary').execute()
            tz = calendar.get('timeZone')
            if tz:
                return tz
        except Exception:
            pass

    return 'Asia/Kolkata'


def sync_calendar_timezone(service=None, target_timezone: str = None) -> str:
    """Ensures Google Calendar primary calendar timezone matches target timezone."""
    timezone_name = target_timezone or os.getenv('CALENDAR_TIMEZONE') or 'Asia/Kolkata'
    if service is None:
        service = get_calendar_service()

    try:
        # Patch primary calendar entry
        cal_list = service.calendarList().list().execute()
        for cal in cal_list.get('items', []):
            if cal.get('primary'):
                cal_id = cal.get('id')
                if cal.get('timeZone') != timezone_name:
                    service.calendarList().patch(calendarId=cal_id, body={'timeZone': timezone_name}).execute()
                    try:
                        service.calendars().patch(calendarId=cal_id, body={'timeZone': timezone_name}).execute()
                    except Exception:
                        pass
                return timezone_name
    except Exception:
        pass
    return timezone_name


def _format_rfc3339_with_tz(iso_value: str, timezone_name: str) -> str:
    """Ensure the datetime string is in RFC3339 format with the target timezone offset."""
    if not iso_value:
        return iso_value

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo('Asia/Kolkata')

    clean_str = str(iso_value).strip()
    if clean_str.endswith('Z'):
        dt = datetime.datetime.fromisoformat(clean_str.replace('Z', '+00:00'))
        dt = dt.astimezone(tz)
    elif len(clean_str) > 10 and ('+' in clean_str[10:] or '-' in clean_str[10:]):
        dt = datetime.datetime.fromisoformat(clean_str)
        dt = dt.astimezone(tz)
    else:
        dt = datetime.datetime.fromisoformat(clean_str)
        dt = dt.replace(tzinfo=tz)

    return dt.isoformat()


def _api_time(iso_value: str, timezone_name: str) -> str:
    """Convert any local or ISO time string to UTC RFC3339 format for Calendar API queries."""
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo('Asia/Kolkata')

    clean_str = str(iso_value).strip()
    if clean_str.endswith('Z'):
        dt = datetime.datetime.fromisoformat(clean_str.replace('Z', '+00:00'))
    elif len(clean_str) > 10 and ('+' in clean_str[10:] or '-' in clean_str[10:]):
        dt = datetime.datetime.fromisoformat(clean_str)
    else:
        dt = datetime.datetime.fromisoformat(clean_str).replace(tzinfo=tz)

    return dt.astimezone(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


def check_calendar_conflict(start_time_iso: str, end_time_iso: str) -> list:
    """Checks if there are existing events overlapping with the requested window."""
    if not start_time_iso or not end_time_iso:
        return []

    service = get_calendar_service()
    calendar_timezone = get_primary_calendar_timezone(service)
    start_dt = _api_time(start_time_iso, calendar_timezone)
    end_dt = _api_time(end_time_iso, calendar_timezone)

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
    calendar_timezone = get_primary_calendar_timezone(service)

    clean_query = re.sub(r"\b(tomorrow|today|next\s+\w+)\b", "", title_query, flags=re.IGNORECASE).strip()

    if not time_min_iso:
        time_min_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    else:
        time_min_iso = _api_time(time_min_iso, calendar_timezone)

    events_result = service.events().list(
        calendarId='primary',
        q=clean_query,
        timeMin=time_min_iso,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    items = events_result.get('items', [])
    return items[0] if items else None


def find_linked_travel_buffer_ids(service, calendar_id: str, event_id: str, event_data: dict = None) -> list:
    """
    Finds all associated travel buffer event IDs for a given Google Calendar event.
    Checks:
    1. extendedProperties (travel_buffer_event_id on main event or parent_event_id on travel buffer)
    2. Description metadata tags ([Event ID: ...] or [Travel Buffer ID: ...])
    3. Heuristic time-window search for events ending at main event start time with '🚗 Travel to ...' or target summary in description.
    """
    linked_ids = []
    if not event_id and not event_data:
        return linked_ids

    # 1. Fetch event data if not already passed
    if event_data is None:
        try:
            event_data = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        except Exception:
            event_data = None

    if not event_data:
        return linked_ids

    current_event_id = event_data.get("id") or event_id
    summary = event_data.get("summary", "").strip()
    description = event_data.get("description", "")
    location = event_data.get("location", "").strip()
    ext_props = event_data.get("extendedProperties", {})
    private_props = ext_props.get("private", {}) if isinstance(ext_props, dict) else {}

    # Check 1: Direct link in extendedProperties
    direct_travel_id = private_props.get("travel_buffer_event_id")
    if direct_travel_id and direct_travel_id != current_event_id and direct_travel_id not in linked_ids:
        linked_ids.append(direct_travel_id)

    # Check 2: Direct link in description
    match_buf_desc = re.search(r"\[Travel Buffer ID:\s*([^\]]+)\]", description)
    if match_buf_desc:
        buf_id_desc = match_buf_desc.group(1).strip()
        if buf_id_desc and buf_id_desc not in linked_ids and buf_id_desc != current_event_id:
            linked_ids.append(buf_id_desc)

    # Check 3: Search surrounding calendar events around start time
    start_raw = event_data.get("start", {}).get("dateTime") or event_data.get("start", {}).get("date")
    if start_raw:
        try:
            start_clean = str(start_raw).strip()
            if start_clean.endswith("Z"):
                start_dt = datetime.datetime.fromisoformat(start_clean.replace("Z", "+00:00"))
            elif len(start_clean) > 10 and ("+" in start_clean[10:] or "-" in start_clean[10:]):
                start_dt = datetime.datetime.fromisoformat(start_clean)
            else:
                tz_name = get_primary_calendar_timezone(service)
                start_dt = datetime.datetime.fromisoformat(start_clean).replace(tzinfo=ZoneInfo(tz_name))

            search_min = (start_dt - datetime.timedelta(hours=3)).astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            search_max = (start_dt + datetime.timedelta(minutes=5)).astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

            nearby_events = service.events().list(
                calendarId=calendar_id,
                timeMin=search_min,
                timeMax=search_max,
                singleEvents=True,
                maxResults=25
            ).execute().get("items", [])

            for item in nearby_events:
                cand_id = item.get("id")
                if not cand_id or cand_id == current_event_id or cand_id in linked_ids:
                    continue

                cand_ext = item.get("extendedProperties", {}).get("private", {}) if isinstance(item.get("extendedProperties"), dict) else {}
                cand_desc = item.get("description", "")
                cand_summary = item.get("summary", "")
                cand_end_raw = item.get("end", {}).get("dateTime") or item.get("end", {}).get("date")

                if cand_ext.get("parent_event_id") == current_event_id or cand_ext.get("travel_for_event_id") == current_event_id:
                    linked_ids.append(cand_id)
                    continue

                if f"[Event ID: {current_event_id}]" in cand_desc:
                    linked_ids.append(cand_id)
                    continue

                is_travel_named = cand_summary.startswith("🚗 Travel") or cand_summary.startswith("Travel to") or cand_ext.get("is_travel_buffer") == "true"
                if is_travel_named and cand_end_raw:
                    try:
                        cand_end_clean = str(cand_end_raw).strip()
                        if cand_end_clean.endswith("Z"):
                            cand_end_dt = datetime.datetime.fromisoformat(cand_end_clean.replace("Z", "+00:00"))
                        elif len(cand_end_clean) > 10 and ("+" in cand_end_clean[10:] or "-" in cand_end_clean[10:]):
                            cand_end_dt = datetime.datetime.fromisoformat(cand_end_clean)
                        else:
                            cand_end_dt = datetime.datetime.fromisoformat(cand_end_clean).replace(tzinfo=start_dt.tzinfo)

                        time_diff = abs((cand_end_dt - start_dt).total_seconds())
                        if time_diff <= 120:
                            if summary and f"before '{summary}'" in cand_desc:
                                linked_ids.append(cand_id)
                            elif summary and summary.lower() in cand_desc.lower():
                                linked_ids.append(cand_id)
                            elif location and location.lower() in cand_summary.lower():
                                linked_ids.append(cand_id)
                            elif cand_summary.startswith("🚗 Travel"):
                                linked_ids.append(cand_id)
                    except Exception:
                        pass
        except Exception:
            pass

    return linked_ids


def delete_google_calendar_event(event_id: str) -> bool:
    """Deletes an event and any associated travel buffer from Google Calendar by ID."""
    service = get_calendar_service()
    try:
        travel_buffer_ids = find_linked_travel_buffer_ids(service, "primary", event_id)
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        for tid in travel_buffer_ids:
            try:
                service.events().delete(calendarId='primary', eventId=tid).execute()
            except Exception:
                pass
        return True
    except Exception:
        return False


def format_google_calendar_link(raw_link: str = None, user_email: str = None) -> str:
    """
    Ensures Google Calendar URLs point accurately to the specific authenticated Google account
    by appending the authuser parameter.
    """
    clean_email = (user_email or "").strip().lower()

    if not raw_link:
        if clean_email:
            return f"https://calendar.google.com/calendar/r?authuser={clean_email}"
        return "https://calendar.google.com/calendar/r"

    link = raw_link.strip()

    if clean_email:
        if "authuser=" in link:
            link = re.sub(r"[?&]authuser=[^&#]+", "", link)
            sep = "&" if "?" in link else "?"
            link = f"{link}{sep}authuser={clean_email}"
        else:
            sep = "&" if "?" in link else "?"
            link = f"{link}{sep}authuser={clean_email}"

    return link


def reschedule_google_calendar_event(event_id: str, new_start_iso: str, new_end_iso: str, user_email: str = None) -> str:
    """Updates start and end time of an existing Google Calendar event."""
    service = get_calendar_service()
    calendar_timezone = get_primary_calendar_timezone(service)

    start_dt = _format_rfc3339_with_tz(new_start_iso, calendar_timezone)
    end_dt = _format_rfc3339_with_tz(new_end_iso, calendar_timezone)

    event = service.events().get(calendarId='primary', eventId=event_id).execute()
    event['start']['dateTime'] = start_dt
    event['start']['timeZone'] = calendar_timezone
    event['end']['dateTime'] = end_dt
    event['end']['timeZone'] = calendar_timezone

    updated_event = service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
    return format_google_calendar_link(updated_event.get('htmlLink'), user_email)


def create_google_calendar_event(event, user_email: str = None):
    """Creates a new event on Google Calendar with localized RFC3339 timestamps and automated travel buffer if location is specified."""
    # If a MultiCalendarEvents wrapper was passed, take the first event
    if hasattr(event, 'events') and event.events:
        event = event.events[0]

    service = get_calendar_service()
    calendar_timezone = get_primary_calendar_timezone(service)

    start_dt = _format_rfc3339_with_tz(event.start_time, calendar_timezone)
    end_dt = _format_rfc3339_with_tz(event.end_time, calendar_timezone)

    location = getattr(event, 'location', 'Not specified')
    has_valid_location = bool(location and location.strip() and location.strip().lower() not in ['not specified', 'none', 'n/a'])

    buffer_id = None
    if has_valid_location:
        try:
            req_start_dt = datetime.datetime.fromisoformat(start_dt)
            buf_start_dt = req_start_dt - datetime.timedelta(minutes=30)
            buffer_body = {
                'summary': f"🚗 Travel to {location.strip()}",
                'description': f"Automated 30-minute travel buffer before '{event.event_name}'.",
                'start': {'dateTime': buf_start_dt.isoformat(), 'timeZone': calendar_timezone},
                'end': {'dateTime': req_start_dt.isoformat(), 'timeZone': calendar_timezone},
                'colorId': '5',  # Yellow in Google Calendar
                'extendedProperties': {
                    'private': {
                        'is_travel_buffer': 'true',
                        'target_summary': event.event_name,
                    }
                }
            }
            created_buf = service.events().insert(calendarId='primary', body=buffer_body).execute()
            buffer_id = created_buf.get('id')
        except Exception:
            pass

    event_body = {
        'summary': event.event_name,
        'location': location,
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
    if buffer_id:
        event_body['extendedProperties'] = {
            'private': {
                'travel_buffer_event_id': buffer_id,
                'has_travel_buffer': 'true',
            }
        }

    created_event = service.events().insert(calendarId='primary', body=event_body).execute()

    if buffer_id and created_event.get('id'):
        try:
            service.events().patch(
                calendarId='primary',
                eventId=buffer_id,
                body={
                    'description': f"Automated 30-minute travel buffer before '{event.event_name}'. [Event ID: {created_event.get('id')}]",
                    'extendedProperties': {
                        'private': {
                            'is_travel_buffer': 'true',
                            'parent_event_id': created_event.get('id'),
                            'target_summary': event.event_name,
                        }
                    }
                }
            ).execute()
        except Exception:
            pass

    return format_google_calendar_link(created_event.get('htmlLink'), user_email)


def list_google_calendar_events(start_time_iso: str, end_time_iso: str, user_email: str = None) -> list:
    """Retrieves all events within a specific time window."""
    service = get_calendar_service()
    calendar_timezone = get_primary_calendar_timezone(service)
    start_dt = _api_time(start_time_iso, calendar_timezone)
    end_dt = _api_time(end_time_iso, calendar_timezone)

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
            "link": format_google_calendar_link(item.get('htmlLink'), user_email)
        })
    return events_list