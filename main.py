import os
import datetime
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import requests

from googleapiclient.discovery import build

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

import calendar_service as calendar_service_module

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

app = FastAPI()
scheduler = AsyncIOScheduler()

# Initialize Google Calendar & Gmail Service
try:
    creds = calendar_service_module.get_google_credentials()
    calendar_service = build('calendar', 'v3', credentials=creds)
    CALENDAR_INIT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime fallback
    creds = None
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


# ---------------------------------------------------------
# 📧 GMAIL TOOLS
# ---------------------------------------------------------

def check_gmail_for_invites(query: str = "") -> str:
    """Checks unread Gmail messages for tea/coffee invitations and returns details."""
    if creds is None:
        return "Google account is not authenticated yet. Please complete the sign-in flow to create token.json."

    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        results = gmail_service.users().messages().list(
            userId='me', q='is:unread (tea OR coffee OR meetup)'
        ).execute()
        
        messages = results.get('messages', [])
        if not messages:
            return "No new tea or coffee invites found in Gmail."

        invites = []
        for msg in messages[:5]:
            email = gmail_service.users().messages().get(userId='me', id=msg['id']).execute()
            snippet = email.get('snippet', '')
            invites.append(f"Email ID: {msg['id']} | Content: {snippet}")

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
    Tool(name="CheckGmailInvites", func=check_gmail_for_invites, description="Checks unread Gmail messages for tea, coffee, or meetup invitations.")
]

# UPDATED LINE:
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GEMINI_API_KEY)

system_prompt = "You are a helpful assistant for an AI calendar agent. Use the available tools to answer user requests about calendar events, Gmail invites, and scheduling."

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
    raise ImportError("Unable to import a valid LangChain agent runner.")


def run_agent(user_text: str) -> str:
    cleaned_input = user_text.strip() if user_text else ""
    if not cleaned_input:
        return "Please enter a valid request or message."

    try:
        if USING_MODERN_CREATE_AGENT:
            # Modern create_agent workflow using message objects
            res = agent_instance.invoke({"messages": [HumanMessage(content=cleaned_input)]})
            if isinstance(res, dict) and "messages" in res and res["messages"]:
                return res["messages"][-1].content
            return str(res)
        else:
            # ReAct AgentExecutor workflow
            res = agent_instance.invoke({"input": cleaned_input})
            if isinstance(res, dict):
                return res.get("output", str(res))
            return str(res)
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
    if not TELEGRAM_CHAT_ID or calendar_service is None:
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
        summary = event.get('summary', 'Upcoming Event')
        start = event['start'].get('dateTime', 'soon')
        send_telegram_message(TELEGRAM_CHAT_ID, f"⏰ REMINDER: '{summary}' starts in 15 minutes ({start})!")


async def auto_scan_tea_invites():
    """Checks Gmail every 15 minutes for new tea invites."""
    if not TELEGRAM_CHAT_ID:
        return
    invites = check_gmail_for_invites()
    if "No new" not in invites and "Error" not in invites:
        send_telegram_message(TELEGRAM_CHAT_ID, f"☕ **New Invitation Detected in Gmail:**\n\n{invites}")


async def send_daily_briefing():
    """Sends a morning schedule summary at 8:00 AM."""
    if not TELEGRAM_CHAT_ID:
        return
    
    events_summary = list_events()
    briefing_text = f"🌅 Good morning! Here is your agenda for today:\n\n{events_summary}"
    send_telegram_message(TELEGRAM_CHAT_ID, briefing_text)


@app.on_event("startup")
def start_background_jobs():
    scheduler.add_job(check_upcoming_reminders, 'interval', minutes=5)
    scheduler.add_job(auto_scan_tea_invites, 'interval', minutes=15)
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
        user_text = data["message"]["text"].strip()

        # 🔒 Security Lockdown Check
        if ALLOWED_CHAT_ID and chat_id != str(ALLOWED_CHAT_ID):
            print(f"Unauthorized access blocked from Chat ID: {chat_id}")
            return {"status": "unauthorized"}
        
        TELEGRAM_CHAT_ID = chat_id  # Save verified ID

        # Welcome Menu Response
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