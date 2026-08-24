import os
import re
from datetime import datetime, timedelta, time
from typing import List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

load_dotenv()


def get_gemini_model_candidates(preferred_model: str | None = None):
    """Return a safe list of Gemini model names, avoiding legacy unsupported ones."""
    configured_model = (preferred_model or os.getenv("GEMINI_MODEL") or os.getenv("GEMINI_MODEL_NAME") or "gemini-3.5-flash").strip()
    normalized = configured_model.removeprefix("models/") if configured_model.startswith("models/") else configured_model

    candidates = []
    if normalized and normalized not in {"gemini-1.5-flash-latest", "models/gemini-1.5-flash-latest"}:
        candidates.append(normalized)

    for fallback_model in ["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite", "gemini-2.0-flash"]:
        if fallback_model not in candidates:
            candidates.append(fallback_model)

    return candidates


class CalendarEvent(BaseModel):
    action: str = Field(default="CREATE", description="Action type: 'CREATE', 'DELETE', 'RESCHEDULE', or 'LIST'")
    event_name: str = Field(description="The title or query summary of the target event")
    start_time: str = Field(default="", description="Start time in ISO format YYYY-MM-DDTHH:MM:SS")
    end_time: str = Field(default="", description="End time in ISO format YYYY-MM-DDTHH:MM:SS")
    location: str = Field(default="Not specified", description="Location of the event")
    priority: str = Field(default="Medium", description="Priority level: High, Medium, or Low")


class MultiCalendarEvents(BaseModel):
    events: List[CalendarEvent] = Field(description="List of all extracted scheduling operations")


if ChatGoogleGenerativeAI is not None and os.getenv("GEMINI_API_KEY"):
    llm = ChatGoogleGenerativeAI(
        model=get_gemini_model_candidates()[0],
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )
    structured_llm = llm.with_structured_output(MultiCalendarEvents)
else:
    structured_llm = None


def _parse_time_value(text: str):
    """Accurately extract start hour and minute from natural text."""
    # 1. Look for explicit am/pm patterns: e.g. "at 2 pm", "2:30pm", "3 pm"
    match_ampm = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.IGNORECASE)
    if match_ampm:
        hour = int(match_ampm.group(1))
        minute = int(match_ampm.group(2) or "0")
        suffix = match_ampm.group(3).lower()
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        return hour, minute

    # 2. Look for 24-hour / colon time: e.g. "14:30", "09:00"
    match_colon = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if match_colon:
        hour = int(match_colon.group(1))
        minute = int(match_colon.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute

    # 3. Look for "at <number>": e.g. "at 2", "at 14"
    match_at = re.search(r"\bat\s+(\d{1,2})\b", text, re.IGNORECASE)
    if match_at:
        hour = int(match_at.group(1))
        minute = 0
        # If hour is between 1 and 7, assume PM for typical business meetings (e.g. 2 -> 14:00)
        if 1 <= hour <= 7:
            hour += 12
        return hour, minute

    # 4. Look for standalone time indicators
    match_any = re.search(r"\b(\d{1,2})\s*(?:o'?clock)\b", text, re.IGNORECASE)
    if match_any:
        hour = int(match_any.group(1))
        if 1 <= hour <= 7:
            hour += 12
        return hour, 0

    return None


def _get_timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except Exception:
        return ZoneInfo("Asia/Kolkata")


def _resolve_date(message_text: str, user_timezone: str):
    today = datetime.now(_get_timezone(user_timezone)).date()
    lower = message_text.lower()

    if "tomorrow" in lower:
        return today + timedelta(days=1)
    if "today" in lower:
        return today

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for idx, day_name in enumerate(weekdays):
        if re.search(rf"\b(?:next\s+|this\s+|on\s+)?{day_name}\b", lower):
            days_ahead = (idx - today.weekday() + 7) % 7
            if days_ahead == 0 and ("next" in lower or "tomorrow" in lower):
                days_ahead = 7
            return today + timedelta(days=days_ahead)

    return today


def _fallback_parse_single(part_text: str, user_timezone: str = "Asia/Kolkata") -> CalendarEvent:
    lower_text = part_text.lower()
    action = "CREATE"
    
    if any(word in lower_text for word in ["cancel", "delete", "remove"]):
        action = "DELETE"
    elif any(word in lower_text for word in ["reschedule", "move", "postpone", "shift"]):
        action = "RESCHEDULE"
    elif any(word in lower_text for word in ["list", "show", "what's on", "schedule for", "meetings"]):
        action = "LIST"

    date_value = _resolve_date(part_text, user_timezone)
    
    if action == "LIST":
        start_dt = datetime.combine(date_value, time.min)
        end_dt = datetime.combine(date_value, time.max.replace(microsecond=0))
        return CalendarEvent(
            action="LIST",
            event_name="Schedule Query",
            start_time=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end_time=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    time_value = _parse_time_value(part_text)
    start_hour, start_minute = time_value if time_value else (9, 0)
    
    start_dt = datetime.combine(date_value, time(hour=start_hour, minute=start_minute))
    end_dt = start_dt + timedelta(minutes=30)

    cleaned_name = re.sub(
        r"\b(tomorrow|today|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|this\s+\w+|on\s+\w+)\b",
        "",
        part_text,
        flags=re.IGNORECASE,
    )
    cleaned_name = re.sub(r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", "", cleaned_name, flags=re.IGNORECASE)
    cleaned_name = re.sub(r"\b(schedule|cancel|delete|remove|reschedule|move|to|at|on|for)\b", "", cleaned_name, flags=re.IGNORECASE)
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip(" -:")
    event_name = cleaned_name or "Scheduled Event"

    return CalendarEvent(
        action=action,
        event_name=event_name,
        start_time=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        end_time=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        location="Not specified",
        priority="Medium",
    )


def _fallback_parse(message_text: str, user_timezone: str | None = None) -> MultiCalendarEvents:
    user_timezone = user_timezone or os.getenv("CALENDAR_TIMEZONE") or "Asia/Kolkata"
    cleaned_text = re.sub(r"\s+", " ", message_text).strip()
    parts = re.split(r"\s+and\s+|,", cleaned_text, flags=re.IGNORECASE)
    events = [_fallback_parse_single(part, user_timezone) for part in parts if part.strip()]
    return MultiCalendarEvents(events=events if events else [_fallback_parse_single(cleaned_text, user_timezone)])


def parse_schedule_message(message_text: str, user_timezone: str | None = None) -> MultiCalendarEvents:
    user_timezone = user_timezone or os.getenv("CALENDAR_TIMEZONE") or "Asia/Kolkata"
    if structured_llm is None:
        return _fallback_parse(message_text, user_timezone)

    now_local = datetime.now(_get_timezone(user_timezone))
    prompt = f"""
    You are an expert scheduling assistant. Extract ALL calendar intent operations mentioned in the message.

    Current Reference Time: {now_local.strftime('%Y-%m-%d %H:%M:%S')} ({now_local.strftime('%A')})
    User Timezone: {user_timezone}

    Strict Guidelines:
    - Set 'action' to 'CREATE', 'DELETE', 'RESCHEDULE', or 'LIST'.
    - Output 'start_time' and 'end_time' strictly in naive ISO format YYYY-MM-DDTHH:MM:SS corresponding to the User Timezone ({user_timezone}). Do NOT add 'Z' and do NOT add UTC offsets (+00:00).
    - For CREATE actions:
        * Map 12-hour times (e.g. '2 pm' -> 14:00, '9:30 am' -> 09:30, '4 pm' -> 16:00).
        * If time of day is ambiguous like 'at 2', interpret as 2 PM (14:00) during daytime business hours.
        * Default duration to 30 minutes if end_time is not given.
        * If no date is given (e.g. 'meeting with dilp at 2 pm'):
          - If the requested time is still upcoming today, use today's date ({now_local.date().isoformat()}).
          - If the requested time has already passed today, use tomorrow's date.
        * For 'event_name', extract a clean, concise title string (e.g. 'Meeting with Dilip' or 'meeting with dilp').
    - For LIST actions (e.g. "What's on my schedule today?", "Show meetings for tomorrow"):
        * Set 'start_time' to the beginning of the requested day (e.g., YYYY-MM-DDT00:00:00).
        * Set 'end_time' to the end of the requested day (e.g., YYYY-MM-DDT23:59:59).
        * Set 'event_name' to "Schedule Query".
    - For DELETE actions:
        * Extract ONLY the clean event title into 'event_name'.
        * Set 'start_time' for the day referenced so search is localized.
    - For RESCHEDULE actions:
        * Extract clean target event_name and the new start_time / end_time in User Timezone.

    User Message: "{message_text}"
    """

    try:
        parsed = structured_llm.invoke(prompt)
        if isinstance(parsed, MultiCalendarEvents):
            return parsed
        if isinstance(parsed, CalendarEvent):
            return MultiCalendarEvents(events=[parsed])
        if hasattr(parsed, "events") and parsed.events:
            return MultiCalendarEvents(events=parsed.events)
    except Exception as exc:
        if any(err in str(exc).lower() for err in ["quota", "resource_exhausted", "429"]):
            return _fallback_parse(message_text, user_timezone)
        raise

    return _fallback_parse(message_text, user_timezone)