from contextlib import asynccontextmanager
import os
import datetime
import asyncio
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import requests

from googleapiclient.discovery import build

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

import calendar_service as calendar_service_module
from parser import parse_schedule_message

# ---------------------------------------------------------
# 📦 LANGCHAIN AGENT IMPORT FALLBACKS
# ---------------------------------------------------------
USING_MODERN_CREATE_AGENT = False

try:
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
    MODERN_REACT_AVAILABLE = True
except ImportError:
    MODERN_REACT_AVAILABLE = False

if not MODERN_REACT_AVAILABLE:
    try:
        from langchain.agents import create_agent
        USING_MODERN_CREATE_AGENT = True
    except ImportError:
        create_agent = None

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALLOWED_CHAT_ID = os.getenv("ALLOWED_CHAT_ID")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Saved after first user message
CALENDAR_TIMEZONE = os.getenv("CALENDAR_TIMEZONE") or "Asia/Kolkata"
PENDING_GMAIL_EVENTS = {}


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


# ---------------------------------------------------------
# ⚙️ FASTAPI LIFESPAN & SCHEDULER SETUP
# ---------------------------------------------------------
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Schedule background tasks and sync calendar timezone
    try:
        if calendar_service is not None:
            calendar_service_module.sync_calendar_timezone(calendar_service, CALENDAR_TIMEZONE)
    except Exception as exc:
        print(f"Warning: Could not sync calendar timezone on startup: {exc}")

    scheduler.add_job(check_upcoming_reminders, 'interval', minutes=5)
    scheduler.add_job(auto_scan_tea_invites, 'interval', minutes=15)
    scheduler.add_job(send_daily_briefing, 'cron', hour=8, minute=0)
    scheduler.start()
    
    yield  # Application runs
    
    # Shutdown
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

# Initialize Google Calendar & Gmail Service
try:
    creds = calendar_service_module.get_google_credentials()
    calendar_service = build('calendar', 'v3', credentials=creds)
    CALENDAR_INIT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime fallback
    creds = None
    calendar_service = None
    CALENDAR_INIT_ERROR = (
        f"{exc}. On Render, set GOOGLE_TOKEN_BASE64 to the base64 contents of token.json."
    )


# ---------------------------------------------------------
# 📅 GOOGLE CALENDAR TOOLS & CONFLICT DETECTION
# ---------------------------------------------------------

def list_events(query: str = "") -> str:
    """Lists events for today or upcoming days in the configured timezone."""
    if calendar_service is None:
        return f"Calendar service is unavailable: {CALENDAR_INIT_ERROR or 'missing credentials'}"

    now = datetime.datetime.now(ZoneInfo(CALENDAR_TIMEZONE)).astimezone(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    events_result = calendar_service.events().list(
        calendarId='primary', timeMin=now,
        maxResults=10, singleEvents=True, orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])
    
    if not events:
        return "No upcoming events found."
    
    output = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        if start and 'T' in start:
            start_clean = start.replace('Z', '+00:00')
            start_dt = datetime.datetime.fromisoformat(start_clean)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=ZoneInfo(CALENDAR_TIMEZONE))
            else:
                start_dt = start_dt.astimezone(ZoneInfo(CALENDAR_TIMEZONE))
            start_formatted = start_dt.strftime('%Y-%m-%d %I:%M %p')
        else:
            start_formatted = start
        output.append(f"ID: {event['id']} | Summary: {event.get('summary')} | Start: {start_formatted}")
    return "\n".join(output)


def create_event(details: str) -> str:
    """
    Creates an event with conflict detection.
    Expects input format: 'Summary | StartISO | EndISO'
    Example: 'Team Sync | 2026-08-06T10:00:00 | 2026-08-06T11:00:00'
    """
    if calendar_service is None:
        return f"Calendar service is unavailable: {CALENDAR_INIT_ERROR or 'missing credentials'}"

    try:
        parts = [p.strip() for p in details.split('|')]
        summary, raw_start, raw_end = parts[0], parts[1], parts[2]
        
        start_time = calendar_service_module._format_rfc3339_with_tz(raw_start, CALENDAR_TIMEZONE)
        end_time = calendar_service_module._format_rfc3339_with_tz(raw_end, CALENDAR_TIMEZONE)

        # Conflict Detection Check
        conflicts = calendar_service_module.check_calendar_conflict(start_time, end_time)
        if conflicts:
            conflict_names = ", ".join(conflicts)
            return f"⚠️ CONFLICT DETECTED! You already have these event(s) at this time: {conflict_names}. Ask the user if they still want to proceed or reschedule."
        
        event_body = {
            'summary': summary,
            'start': {'dateTime': start_time, 'timeZone': CALENDAR_TIMEZONE},
            'end': {'dateTime': end_time, 'timeZone': CALENDAR_TIMEZONE},
        }
        created = calendar_service.events().insert(calendarId='primary', body=event_body).execute()
        return f"✅ Event created successfully: '{created.get('summary')}' at {start_time} ({CALENDAR_TIMEZONE})"
    except Exception as e:
        return f"Error creating event: {str(e)}"


def delete_event(event_id: str) -> str:
    """Deletes an event given its Event ID."""
    if calendar_service is None:
        return f"Calendar service is unavailable: {CALENDAR_INIT_ERROR or 'missing credentials'}"

    try:
        calendar_service.events().delete(calendarId='primary', eventId=event_id.strip()).execute()
        return f"🗑️ Event '{event_id}' deleted successfully."
    except Exception as e:
        return f"Error deleting event: {str(e)}"


def update_event(details: str) -> str:
    """
    Updates an event time or summary.
    Expects input format: 'EventID | NewSummary | NewStartISO | NewEndISO'
    """
    if calendar_service is None:
        return f"Calendar service is unavailable: {CALENDAR_INIT_ERROR or 'missing credentials'}"

    try:
        parts = [p.strip() for p in details.split('|')]
        event_id, summary, raw_start, raw_end = parts[0], parts[1], parts[2], parts[3]
        
        start_time = calendar_service_module._format_rfc3339_with_tz(raw_start, CALENDAR_TIMEZONE)
        end_time = calendar_service_module._format_rfc3339_with_tz(raw_end, CALENDAR_TIMEZONE)

        event = calendar_service.events().get(calendarId='primary', eventId=event_id).execute()
        event['summary'] = summary
        event['start'] = {'dateTime': start_time, 'timeZone': CALENDAR_TIMEZONE}
        event['end'] = {'dateTime': end_time, 'timeZone': CALENDAR_TIMEZONE}
        
        updated = calendar_service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
        return f"✏️ Event updated: '{updated.get('summary')}' is now at {start_time} ({CALENDAR_TIMEZONE})"
    except Exception as e:
        return f"Error updating event: {str(e)}"


# ---------------------------------------------------------
# 📧 GMAIL TOOLS
# ---------------------------------------------------------

def check_gmail_for_invites(query: str = "") -> str:
    """Checks unread Gmail messages for Drive or internship messages."""
    if creds is None:
        return "Google account is not authenticated yet. Please complete the sign-in flow to create token.json."

    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        results = gmail_service.users().messages().list(
            userId='me', q='is:unread (drive OR internship)'
        ).execute()
        
        messages = results.get('messages', [])
        if not messages:
            return "No unread Drive or internship messages found in Gmail."

        invites = []
        for msg in messages[:5]:
            email = gmail_service.users().messages().get(userId='me', id=msg['id']).execute()
            snippet = email.get('snippet', '')
            headers = {
                header.get('name', '').lower(): header.get('value', '')
                for header in email.get('payload', {}).get('headers', [])
            }
            short_snippet = ' '.join(snippet.split())[:280]
            invites.append(
                f"Email ID: {msg['id']} | Subject: {headers.get('subject', 'No subject')} | Content: {short_snippet}"
            )

        return "\n".join(invites)
    except Exception as e:
        return f"Error reading Gmail: {str(e)}"


# ---------------------------------------------------------
# 🛠️ TOOLS & AGENT SETUP
# ---------------------------------------------------------

tools = [
    Tool(name="ListEvents", func=list_events, description="Lists upcoming calendar events."),
    Tool(name="CreateEvent", func=create_event, description="Creates a new event. Format: 'Summary | StartISO | EndISO'"),
    Tool(name="DeleteEvent", func=delete_event, description="Deletes an event using its Event ID."),
    Tool(name="UpdateEvent", func=update_event, description="Updates an event. Format: 'EventID | Summary | StartISO | EndISO'"),
    Tool(name="CheckGmailInvites", func=check_gmail_for_invites, description="Checks unread Gmail messages for Drive or internship messages.")
]

DEFAULT_GEMINI_MODEL = get_gemini_model_candidates()[0]
llm = ChatGoogleGenerativeAI(
    model=DEFAULT_GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    max_retries=3
)
system_prompt = (
    "You are a helpful assistant for an AI calendar agent. "
    f"The user's timezone is {CALENDAR_TIMEZONE}. Always interpret and display event times in this timezone, "
    "never UTC unless the user explicitly requests UTC. Use the available tools to answer user requests "
    "about calendar events, Gmail messages, and scheduling."
)

if USING_MODERN_CREATE_AGENT:
    agent_instance = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
elif MODERN_REACT_AVAILABLE:
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        prompt = system_prompt
    agent_runner = create_react_agent(llm, tools, prompt)
    agent_instance = AgentExecutor(agent=agent_runner, tools=tools, verbose=True, handle_parsing_errors=True)
else:
    agent_instance = None


def is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(term in message for term in ["quota", "resource_exhausted", "429"])


def clean_agent_response(response) -> str:
    """Return only visible text from Gemini content blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()
    return str(content)


def run_calendar_without_llm(user_text: str) -> str:
    """Handle common calendar requests when Gemini quota is unavailable."""
    try:
        parsed_data = parse_schedule_message(user_text, CALENDAR_TIMEZONE)
        replies = []

        for event in parsed_data.events:
            action = event.action.upper()
            if action == "CREATE":
                conflicts = calendar_service_module.check_calendar_conflict(
                    event.start_time, event.end_time
                )
                if conflicts:
                    replies.append(f"Conflict detected with: {', '.join(conflicts)}")
                link = calendar_service_module.create_google_calendar_event(event)
                replies.append(
                    f"Event created: {event.event_name}\n"
                    f"Time ({CALENDAR_TIMEZONE}): {event.start_time} to {event.end_time}\n"
                    f"Calendar link: {link}"
                )
            elif action == "LIST":
                events = calendar_service_module.list_google_calendar_events(
                    event.start_time, event.end_time
                )
                if not events:
                    replies.append("No events found for that date.")
                else:
                    replies.append("\n".join(
                        f"{item['summary']} at {item['start']}" for item in events
                    ))
            elif action == "DELETE":
                target = calendar_service_module.find_event_by_title(event.event_name)
                if target and calendar_service_module.delete_google_calendar_event(target['id']):
                    replies.append(f"Deleted event: {target.get('summary', event.event_name)}")
                else:
                    replies.append(f"Could not find event: {event.event_name}")
            elif action == "RESCHEDULE":
                target = calendar_service_module.find_event_by_title(event.event_name)
                if not target:
                    replies.append(f"Could not find event: {event.event_name}")
                else:
                    link = calendar_service_module.reschedule_google_calendar_event(
                        target['id'], event.start_time, event.end_time
                    )
                    replies.append(f"Event rescheduled: {event.event_name}\nCalendar link: {link}")

        return "\n\n".join(replies)
    except Exception as exc:
        if "missing credentials" in str(exc).lower() or "credentials" in str(exc).lower():
            return "Calendar is not authenticated on Render. Set GOOGLE_TOKEN_BASE64 using your token.json contents, then redeploy."
        return f"Gemini quota is exhausted, and the Calendar fallback failed: {exc}"


def run_agent(user_text: str) -> str:
    cleaned_input = user_text.strip() if user_text else ""
    if not cleaned_input:
        return "Please enter a valid request or message."

    try:
        if agent_instance is None:
            response = llm.invoke(
                f"{system_prompt}\n\nUser request: {cleaned_input}\n"
                "Respond clearly. If the request requires a calendar or Gmail action, explain the requested action."
            )
            return clean_agent_response(response)
        if USING_MODERN_CREATE_AGENT:
            res = agent_instance.invoke({"messages": [HumanMessage(content=cleaned_input)]})
            if isinstance(res, dict) and "messages" in res and res["messages"]:
                return clean_agent_response(res["messages"][-1])
            return clean_agent_response(res)
        else:
            res = agent_instance.invoke({"input": cleaned_input})
            if isinstance(res, dict):
                return clean_agent_response(res.get("output", ""))
            return clean_agent_response(res)
    except Exception as exc:
        if is_quota_error(exc):
            return run_calendar_without_llm(cleaned_input)
        return f"Error processing request: {exc}"


# ---------------------------------------------------------
# ⏰ BACKGROUND SCHEDULER TASKS
# ---------------------------------------------------------

def send_telegram_message(chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})


async def check_upcoming_reminders():
    """Checks for events starting in the next 15 minutes and sends an alert."""
    if not TELEGRAM_CHAT_ID or calendar_service is None:
        return
    
    now = datetime.datetime.now(datetime.timezone.utc)
    in_15_mins = now + datetime.timedelta(minutes=15)
    
    events_result = calendar_service.events().list(
        calendarId='primary',
        timeMin=now.isoformat().replace('+00:00', 'Z'),
        timeMax=in_15_mins.isoformat().replace('+00:00', 'Z'),
        singleEvents=True
    ).execute()
    
    for event in events_result.get('items', []):
        summary = event.get('summary', 'Upcoming Event')
        start = event['start'].get('dateTime', 'soon')
        send_telegram_message(TELEGRAM_CHAT_ID, f"⏰ REMINDER: '{summary}' starts in 15 minutes ({start})!")


async def auto_scan_tea_invites():
    """Checks Gmail every 15 minutes for Drive or internship messages."""
    if not TELEGRAM_CHAT_ID:
        return
    try:
        messages = calendar_service_module.find_gmail_drive_or_internship_messages()
    except Exception as exc:
        send_telegram_message(TELEGRAM_CHAT_ID, f"❌ Could not read Gmail: {exc}")
        return

    if not messages:
        return

    PENDING_GMAIL_EVENTS[TELEGRAM_CHAT_ID] = []
    lines = ["📧 **New Drive/internship message detected in Gmail:**"]
    for message in messages:
        email_text = f"{message['subject']}\n{message['snippet']}"
        parsed_email = parse_schedule_message(email_text, CALENDAR_TIMEZONE)
        PENDING_GMAIL_EVENTS[TELEGRAM_CHAT_ID].extend(parsed_email.events)
        lines.append(f"\nSubject: {message['subject']}\nFrom: {message['from']}\nMessage: {message['display_snippet']}")
    lines.append("\nReply 'approve' after reviewing it to add it to the calendar.")
    send_telegram_message(TELEGRAM_CHAT_ID, "\n".join(lines))


async def send_daily_briefing():
    """Sends a morning schedule summary at 8:00 AM."""
    if not TELEGRAM_CHAT_ID:
        return
    
    events_summary = list_events()
    briefing_text = f"🌅 Good morning! Here is your agenda for today:\n\n{events_summary}"
    send_telegram_message(TELEGRAM_CHAT_ID, briefing_text)


# ---------------------------------------------------------
# 💬 TELEGRAM WEBHOOK ROUTE
# ---------------------------------------------------------

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    global TELEGRAM_CHAT_ID
    data = await request.json()
    
    if "message" in data and "text" in data["message"]:
        chat_id = str(data["message"]["chat"]["id"])
        user_text = data["message"]["text"].strip()

        # Security Check
        if ALLOWED_CHAT_ID and chat_id != str(ALLOWED_CHAT_ID):
            print(f"Unauthorized access blocked from Chat ID: {chat_id}")
            return {"status": "unauthorized"}
        
        TELEGRAM_CHAT_ID = chat_id

        normalized_text = user_text.lower()
        if normalized_text in {"approve", "approved", "yes", "add it", "add to calendar"} or normalized_text.startswith("approve "):
            pending_events = PENDING_GMAIL_EVENTS.pop(chat_id, None)
            if not pending_events:
                try:
                    messages = calendar_service_module.find_gmail_drive_or_internship_messages()
                    pending_events = []
                    for message in messages:
                        email_text = f"{message['subject']}\n{message['snippet']}"
                        pending_events.extend(parse_schedule_message(email_text, CALENDAR_TIMEZONE).events)
                except Exception as exc:
                    send_telegram_message(chat_id, f"❌ Could not recover the Gmail event: {exc}")
                    return {"status": "ok"}
            if not pending_events:
                send_telegram_message(chat_id, "There is no unread Gmail event waiting for approval.")
                return {"status": "ok"}
            links = [calendar_service_module.create_google_calendar_event(event) for event in pending_events]
            send_telegram_message(chat_id, "✅ Added the approved Gmail event(s) to Google Calendar.\n" + "\n".join(link for link in links if link))
            return {"status": "ok"}

        if any(word in normalized_text for word in ["gmail", "email", "drive", "internship"]):
            try:
                messages = calendar_service_module.find_gmail_drive_or_internship_messages()
            except Exception as exc:
                send_telegram_message(chat_id, f"❌ Could not read Gmail: {exc}")
                return {"status": "ok"}
            if not messages:
                send_telegram_message(chat_id, "No unread Drive or internship messages found in Gmail.")
                return {"status": "ok"}

            PENDING_GMAIL_EVENTS[chat_id] = []
            lines = ["📧 Gmail messages found (Drive/internship):"]
            for message in messages:
                email_text = f"{message['subject']}\n{message['snippet']}"
                parsed_email = parse_schedule_message(email_text, CALENDAR_TIMEZONE)
                PENDING_GMAIL_EVENTS[chat_id].extend(parsed_email.events)
                lines.append(f"\nSubject: {message['subject']}\nFrom: {message['from']}\nMessage: {message['display_snippet']}")
            lines.append("\nReply 'approve' to add these detected event(s) to your calendar.")
            send_telegram_message(chat_id, "\n".join(lines))
            return {"status": "ok"}

        # Welcome Response
        if user_text.lower() in ["/start", "/help", "hi", "hello"]:
            welcome_text = (
                "👋 **Welcome to your AI Calendar Assistant!**\n\n"
                "• 📅 View schedule: *'What is on my schedule today?'*\n"
                "• ➕ Add event: *'Schedule tea tomorrow at 4 PM'*\n"
                "• ☕ Scan Gmail: *'Check my emails for tea invites'*\n"
                "• 🗑️ Delete event: *'Delete event [Event ID]'"
            )
            send_telegram_message(chat_id, welcome_text)
            return {"status": "ok"}
        
        # Process request through Gemini Agent
        response = run_agent(user_text)
        send_telegram_message(chat_id, response)
        
    return {"status": "ok"}