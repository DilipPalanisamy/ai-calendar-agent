import os
import datetime
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import requests

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool

try:
    from langchain import hub
except ImportError:  # pragma: no cover - compatibility fallback
    hub = None

try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:  # pragma: no cover - compatibility fallback
    create_react_agent = None
    AgentExecutor = None

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Saved after first user message

app = FastAPI()
scheduler = AsyncIOScheduler()

# Initialize Google Calendar Service
SCOPES = ['https://www.googleapis.com/auth/calendar']
try:
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    calendar_service = build('calendar', 'v3', credentials=creds)
    CALENDAR_INIT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime fallback
    calendar_service = None
    CALENDAR_INIT_ERROR = str(exc)


# ---------------------------------------------------------
# 📅 GOOGLE CALENDAR TOOLS & CONFLICT DETECTION
# ---------------------------------------------------------

def list_events(query: str = "") -> str:
    """Lists events for today or upcoming days."""
    if calendar_service is None:
        return f"Calendar service is unavailable: {CALENDAR_INIT_ERROR or 'missing credentials'}"

    now = datetime.datetime.utcnow().isoformat() + 'Z'
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
        output.append(f"ID: {event['id']} | Summary: {event.get('summary')} | Start: {start}")
    return "\n".join(output)


def create_event(details: str) -> str:
    """
    Creates an event with conflict detection.
    Expects input format: 'Summary | StartISO | EndISO'
    Example: 'Team Sync | 2026-08-06T10:00:00Z | 2026-08-06T11:00:00Z'
    """
    if calendar_service is None:
        return f"Calendar service is unavailable: {CALENDAR_INIT_ERROR or 'missing credentials'}"

    try:
        parts = [p.strip() for p in details.split('|')]
        summary, start_time, end_time = parts[0], parts[1], parts[2]
        
        # Conflict Detection Check
        existing_events = calendar_service.events().list(
            calendarId='primary',
            timeMin=start_time,
            timeMax=end_time,
            singleEvents=True
        ).execute().get('items', [])
        
        if existing_events:
            conflicts = ", ".join([e.get('summary', 'Event') for e in existing_events])
            return f"⚠️ CONFLICT DETECTED! You already have these event(s) at this time: {conflicts}. Ask the user if they still want to proceed or reschedule."
        
        event_body = {
            'summary': summary,
            'start': {'dateTime': start_time},
            'end': {'dateTime': end_time},
        }
        created = calendar_service.events().insert(calendarId='primary', body=event_body).execute()
        return f"✅ Event created successfully: '{created.get('summary')}' at {start_time}"
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
        event_id, summary, start_time, end_time = parts[0], parts[1], parts[2], parts[3]
        
        event = calendar_service.events().get(calendarId='primary', eventId=event_id).execute()
        event['summary'] = summary
        event['start'] = {'dateTime': start_time}
        event['end'] = {'dateTime': end_time}
        
        updated = calendar_service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
        return f"✏️ Event updated: '{updated.get('summary')}' is now at {start_time}"
    except Exception as e:
        return f"Error updating event: {str(e)}"


# LangChain Agent Setup
tools = [
    Tool(name="ListEvents", func=list_events, description="Lists upcoming calendar events."),
    Tool(name="CreateEvent", func=create_event, description="Creates a new event. Format: 'Summary | StartISO | EndISO'"),
    Tool(name="DeleteEvent", func=delete_event, description="Deletes an event using its Event ID."),
    Tool(name="UpdateEvent", func=update_event, description="Updates an event. Format: 'EventID | Summary | StartISO | EndISO'")
]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
agent = None

try:
    if hub is not None and create_react_agent is not None and AgentExecutor is not None:
        prompt = hub.pull("hwchase17/react") if hasattr(hub, "pull") else None
        agent_runner = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent_runner,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        agent = agent_executor
except Exception:
    agent = None

if agent is None:
    from langchain.agents import initialize_agent, AgentType
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


def run_agent(user_text: str) -> str:
    try:
        if hasattr(agent, "invoke"):
            result = agent.invoke({"input": user_text})
            if isinstance(result, dict):
                return result.get("output", str(result))
            return str(result)
        return agent.run(user_text)
    except Exception as exc:
        return f"Error processing request: {exc}"


# ---------------------------------------------------------
# ⏰ BACKGROUND SCHEDULER (REMINDERS & DAILY BRIEFINGS)
# ---------------------------------------------------------

def send_telegram_message(chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})


async def check_upcoming_reminders():
    """Checks for events starting in the next 15 minutes and sends an alert."""
    if not TELEGRAM_CHAT_ID:
        return
    
    now = datetime.datetime.utcnow()
    in_15_mins = now + datetime.timedelta(minutes=15)
    
    events_result = calendar_service.events().list(
        calendarId='primary',
        timeMin=now.isoformat() + 'Z',
        timeMax=in_15_mins.isoformat() + 'Z',
        singleEvents=True
    ).execute()
    
    for event in events_result.get('items', []):
        # Send reminder alert
        summary = event.get('summary', 'Upcoming Event')
        start = event['start'].get('dateTime', 'soon')
        send_telegram_message(TELEGRAM_CHAT_ID, f"⏰ REMINDER: '{summary}' starts in 15 minutes ({start})!")


async def send_daily_briefing():
    """Sends a morning schedule summary at 8:00 AM."""
    if not TELEGRAM_CHAT_ID:
        return
    
    events_summary = list_events()
    briefing_text = f"🌅 Good morning! Here is your agenda for today:\n\n{events_summary}"
    send_telegram_message(TELEGRAM_CHAT_ID, briefing_text)


@app.on_event("startup")
def start_background_jobs():
    # Check for event reminders every 5 minutes
    scheduler.add_job(check_upcoming_reminders, 'interval', minutes=5)
    # Daily morning briefing at 08:00 AM
    scheduler.add_job(send_daily_briefing, 'cron', hour=8, minute=0)
    scheduler.start()


# ---------------------------------------------------------
# 💬 TELEGRAM WEBHOOK ROUTE
# ---------------------------------------------------------

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    global TELEGRAM_CHAT_ID
    data = await request.json()
    
    if "message" in data and "text" in data["message"]:
        chat_id = str(data["message"]["chat"]["id"])
        user_text = data["message"]["text"]
        TELEGRAM_CHAT_ID = chat_id  # Save chat ID for background notifications
        
        # Process message with Gemini Agent
        response = run_agent(user_text)
        
        # Reply back to Telegram
        send_telegram_message(chat_id, response)
        
    return {"status": "ok"}