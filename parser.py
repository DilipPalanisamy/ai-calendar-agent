import os
import re
from datetime import datetime, timedelta
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
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, re.IGNORECASE)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    suffix = (match.group(3) or "").lower()

    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0

    return hour, minute


def _get_timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except Exception:
        return ZoneInfo("UTC")


def _resolve_date(message_text: str, user_timezone: str):
    today = datetime.now(_get_timezone(user_timezone)).date()
    lower = message_text.lower()

    if "tomorrow" in lower:
        return today + timedelta(days=1)
    if "today" in lower:
        return today

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for idx, day_name in enumerate(weekdays):
        if f"next {day_name}" in lower:
            days_ahead = (idx - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            return today + timedelta(days=days_ahead)

    return today


def _fallback_parse_single(part_text: str, user_timezone: str = "UTC") -> CalendarEvent:
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
        start_dt = datetime.combine(date_value, datetime.min.time())
        end_dt = datetime.combine(date_value, datetime.max.time().replace(microsecond=0))
        return CalendarEvent(
            action="LIST",
            event_name="Schedule Query",
            start_time=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end_time=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    time_value = _parse_time_value(part_text)
    start_hour, start_minute = time_value if time_value else (9, 0)
    end_hour, end_minute = start_hour, start_minute + 30
    if end_minute >= 60:
        end_hour += 1
        end_minute -= 60

    cleaned_name = re.sub(
        r"\b(tomorrow|today|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        "",
        part_text,
        flags=re.IGNORECASE,
    )
    cleaned_name = re.sub(r"\b(at|on|schedule|cancel|delete|remove|reschedule|move|to)\b", "", cleaned_name, flags=re.IGNORECASE)
    cleaned_name = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", "", cleaned_name, flags=re.IGNORECASE)
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip(" -:")
    event_name = cleaned_name or "Scheduled Event"

    start_dt = datetime.combine(date_value, datetime.min.time()).replace(hour=start_hour, minute=start_minute)
    end_dt = datetime.combine(date_value, datetime.min.time()).replace(hour=end_hour, minute=end_minute)

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

    prompt = f"""
    You are an expert scheduling assistant. Extract ALL intent operations mentioned in the message.

    Current Date: {datetime.now(_get_timezone(user_timezone)).date().isoformat()}
    User Timezone: {user_timezone}

    Guidelines:
    - Set 'action' to 'CREATE', 'DELETE', 'RESCHEDULE', or 'LIST'.
    - For LIST actions (e.g. "What's on my schedule for tomorrow?", "Show my meetings today"):
        * Set 'start_time' to the beginning of the requested day (e.g., 2026-08-06T00:00:00).
        * Set 'end_time' to the end of the requested day (e.g., 2026-08-06T23:59:59).
        * Set 'event_name' to "Schedule Query".
    - For DELETE actions:
        * Extract ONLY the clean title string into 'event_name' (e.g., for "Cancel client call tomorrow", event_name should be "client call").
        * Calculate 'start_time' for the day referenced so search is localized.
    - For RESCHEDULE actions, extract clean target event_name and the new start_time / end_time.
    - For CREATE actions, default duration to 30 minutes if end_time is not given.

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