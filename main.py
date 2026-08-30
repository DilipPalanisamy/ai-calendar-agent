import os
import re
import json
import uuid
import sqlite3
import asyncio
import logging
from datetime import datetime, timezone, timedelta, time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# Google Auth & API Client Libraries
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from parser import get_gemini_model_candidates
from gemini_resilience import acall_gemini_with_retry, call_gemini_with_retry

# ---------------------------------------------------------------------------
# 1. Environment & Path Setup
# ---------------------------------------------------------------------------
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_calendar_agent")

# Enable OAuth insecure transport during local development (HTTP)
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = os.getenv("OAUTHLIB_INSECURE_TRANSPORT", "1")

# Base directory & Templates configuration
# Base directory & Persistent Database configuration
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# Support persistent volume disk on cloud platforms (e.g., Render Persistent Disks at /var/data)
data_dir_env = os.getenv("DATA_DIR") or os.getenv("DATABASE_DIR")
if data_dir_env:
    data_dir = Path(data_dir_env)
    data_dir.mkdir(parents=True, exist_ok=True)
    DB_PATH = data_dir / "chat_history.db"
elif os.getenv("DATABASE_PATH"):
    DB_PATH = Path(os.getenv("DATABASE_PATH"))
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
else:
    DB_PATH = BASE_DIR / "chat_history.db"

# Jinja2 Templates setup
templates = Jinja2Templates(directory=str(TEMPLATES_DIR) if TEMPLATES_DIR.exists() else "templates")

# Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SESSION_SECRET_KEY") or os.getenv("SECRET_KEY", "ai-calendar-agent-secret-key-production-ready-2026")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
CALENDAR_TIMEZONE = os.getenv("CALENDAR_TIMEZONE", "Asia/Kolkata")


def normalize_iso_datetime(dt_str: str, default_tz_name: str = CALENDAR_TIMEZONE) -> str:
    """Ensures ISO datetime string includes the explicit Indian Standard Time (+05:30) offset."""
    if not dt_str:
        return dt_str
    dt_str = str(dt_str).strip()

    # If format is already with timezone offset (e.g. +05:30 or -04:00), return as is
    if re.search(r"[+-]\d{2}:\d{2}$", dt_str):
        return dt_str

    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1]

    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(default_tz_name))
        return dt.isoformat()
    except Exception:
        # Fallback appending offset if simple YYYY-MM-DDTHH:MM:SS
        if "T" in dt_str and len(dt_str) >= 16 and not dt_str.endswith("+05:30"):
            return f"{dt_str}+05:30"
        return dt_str

# Google OAuth 2.0 Scopes required for Calendar & Gmail
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]

# ---------------------------------------------------------------------------
# 2. SQLite Database Setup & Chat History Management
# ---------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with WAL mode."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def init_db():
    """Initializes SQLite database schema with strict user_email indexing for chat sessions and messages."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_email, updated_at DESC);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_email TEXT,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, timestamp ASC);")

        # Safe schema migration if existing table lacks user_email
        try:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN user_email TEXT;")
        except Exception:
            pass

        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON chat_messages(user_email, timestamp ASC);")
        conn.commit()
    logger.info("SQLite Chat History database initialized successfully with user_email isolation.")


# Initialize database at startup
init_db()


def create_chat_session(user_email: str, title: str) -> str:
    """Creates a new chat session strictly tied to the user's Gmail address."""
    session_id = str(uuid.uuid4())
    now_str = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, user_email, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_email, title, now_str, now_str)
        )
        conn.commit()
    return session_id


def get_user_chat_sessions(user_email: str) -> List[Dict[str, Any]]:
    """Retrieves all chat sessions strictly belonging to the specified user email."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE user_email = ? ORDER BY updated_at DESC",
            (user_email,)
        )
        rows = cursor.fetchall()
        return [{"id": r["id"], "title": r["title"], "created_at": r["created_at"], "updated_at": r["updated_at"]} for r in rows]


def get_chat_session_details(session_id: str, user_email: str) -> Optional[Dict[str, Any]]:
    """Retrieves a session and all its messages, verifying strict ownership by user_email."""
    with get_db_connection() as conn:
        s_cur = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ? AND user_email = ?",
            (session_id, user_email)
        )
        session_row = s_cur.fetchone()
        if not session_row:
            return None

        m_cur = conn.execute(
            "SELECT id, sender, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
            (session_id,)
        )
        msg_rows = m_cur.fetchall()
        messages = [{"id": m["id"], "sender": m["sender"], "content": m["content"], "timestamp": m["timestamp"]} for m in msg_rows]

        return {
            "session": {
                "id": session_row["id"],
                "title": session_row["title"],
                "created_at": session_row["created_at"],
                "updated_at": session_row["updated_at"]
            },
            "messages": messages
        }


def save_chat_message(session_id: str, sender: str, content: str, user_email: str = ""):
    """Appends a message to a session tied to the user email and updates timestamp."""
    now_str = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, user_email, sender, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_email, sender, content, now_str)
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now_str, session_id)
        )
        conn.commit()


def delete_chat_session(session_id: str, user_email: str) -> bool:
    """Deletes a specific chat session strictly owned by user_email."""
    with get_db_connection() as conn:
        cur = conn.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_email = ?",
            (session_id, user_email)
        )
        conn.commit()
        return cur.rowcount > 0


def delete_all_user_sessions(user_email: str) -> int:
    """Clears all chat history records strictly for the specified user_email."""
    with get_db_connection() as conn:
        cur = conn.execute("DELETE FROM chat_sessions WHERE user_email = ?", (user_email,))
        conn.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# 3. FastAPI Application Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Calendar Assistant",
    description="Multi-user & Multi-Account AI Calendar & Gmail Assistant powered by Gemini & FastAPI",
    version="2.3.0",
)

# Check if running in production / Render HTTPS
IS_PRODUCTION = bool(
    os.getenv("RENDER")
    or (RENDER_EXTERNAL_URL and RENDER_EXTERNAL_URL.startswith("https"))
    or os.getenv("ENVIRONMENT", "").lower() == "production"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Middleware for secure session token storage
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="ai_calendar_session",
    max_age=14 * 24 * 3600,  # 14 days
    same_site="lax",
    https_only=IS_PRODUCTION,
)


# ---------------------------------------------------------------------------
# 4. Multi-Account Google OAuth 2.0 Helper Functions
# ---------------------------------------------------------------------------
def get_client_config() -> Dict[str, Any]:
    """Retrieve Google OAuth Client configuration from env vars or credentials.json fallback."""
    client_id = GOOGLE_CLIENT_ID
    client_secret = GOOGLE_CLIENT_SECRET

    cred_file = BASE_DIR / "credentials.json"
    if (not client_id or not client_secret) and cred_file.exists():
        try:
            with open(cred_file, "r") as f:
                data = json.load(f)
                conf = data.get("web") or data.get("installed")
                if conf:
                    return {"web": conf}
        except Exception as e:
            logger.warning(f"Failed reading credentials.json: {e}")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Google OAuth credentials missing! Configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env or credentials.json"
        )

    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        }
    }


def get_redirect_uri(request: Request) -> str:
    """
    Dynamically determine the OAuth callback redirect URI.
    Enforces HTTPS when running on production (Render, onrender.com, or behind reverse proxies).
    """
    if RENDER_EXTERNAL_URL and "onrender.com" in RENDER_EXTERNAL_URL and RENDER_EXTERNAL_URL != "http://localhost:8000":
        base = RENDER_EXTERNAL_URL.rstrip("/")
        return f"{base}/auth/callback"

    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost:8000"
    scheme = (
        "https"
        if "onrender.com" in host
        or request.headers.get("x-forwarded-proto") == "https"
        or request.url.scheme == "https"
        else "http"
    )
    return f"{scheme}://{host}/auth/callback"


def get_accounts_dict(request: Request) -> Dict[str, Any]:
    """Retrieves the multi-account dictionary from session with legacy fallback."""
    raw_accounts = request.session.get("accounts")
    if isinstance(raw_accounts, dict) and raw_accounts:
        return dict(raw_accounts)

    legacy_creds = request.session.get("user_creds")
    legacy_email = request.session.get("user_email")
    if legacy_creds and legacy_email:
        acc_entry = dict(legacy_creds)
        acc_entry["name"] = request.session.get("user_name", legacy_email)
        acc_entry["picture"] = request.session.get("user_picture", "")
        accounts = {legacy_email: acc_entry}
        request.session["accounts"] = accounts
        request.session["active_account"] = legacy_email
        return accounts

    return {}


def get_active_account_email(request: Request) -> Optional[str]:
    """Retrieves the active account email, defaulting to the first connected account."""
    accounts = get_accounts_dict(request)
    if not accounts:
        return None

    active_email = request.session.get("active_account")
    if active_email and active_email in accounts:
        return active_email

    first_email = next(iter(accounts.keys()))
    request.session["active_account"] = first_email
    return first_email


def get_user_credentials(request: Request) -> Optional[Credentials]:
    """
    Dynamically deserializes and refreshes credentials for the ACTIVE account.
    Enforces persistent refresh_token usage:
    - If access token is expired, refreshes it using GoogleRequest().
    - If refresh fails or credentials lack refresh_token when invalid, clears invalid credentials entry and returns None to trigger re-consent.
    """
    accounts = get_accounts_dict(request)
    active_email = get_active_account_email(request)
    if not active_email or active_email not in accounts:
        return None

    creds_data = accounts[active_email]

    try:
        client_config = get_client_config()
        client_info = client_config.get("web") or client_config.get("installed") or {}
        client_id = creds_data.get("client_id") or client_info.get("client_id") or GOOGLE_CLIENT_ID
        client_secret = creds_data.get("client_secret") or client_info.get("client_secret") or GOOGLE_CLIENT_SECRET
        refresh_token = creds_data.get("refresh_token")

        creds = Credentials(
            token=creds_data.get("token"),
            refresh_token=refresh_token,
            token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=client_id,
            client_secret=client_secret,
            scopes=creds_data.get("scopes", SCOPES),
        )

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                logger.info(f"Access token expired for '{active_email}'. Refreshing using persistent refresh_token...")
                try:
                    creds.refresh(GoogleRequest())
                    creds_data["token"] = creds.token
                    creds_data["refresh_token"] = creds.refresh_token or refresh_token
                    creds_data["token_uri"] = creds.token_uri
                    accounts = get_accounts_dict(request)
                    accounts[active_email] = creds_data
                    request.session["accounts"] = accounts
                except Exception as refresh_err:
                    logger.error(f"Failed to refresh token for '{active_email}': {refresh_err}")
                    accounts.pop(active_email, None)
                    request.session["accounts"] = accounts
                    if request.session.get("active_account") == active_email:
                        request.session.pop("active_account", None)
                    return None
            elif not creds.token and not creds.refresh_token:
                logger.warning(f"Active account '{active_email}' is missing both token and refresh_token.")
                accounts.pop(active_email, None)
                request.session["accounts"] = accounts
                if request.session.get("active_account") == active_email:
                    request.session.pop("active_account", None)
                return None

        request.session["user_email"] = active_email
        request.session["user_name"] = creds_data.get("name", active_email)
        request.session["user_picture"] = creds_data.get("picture", "")

        return creds
    except Exception as e:
        logger.error(f"Error loading credentials for '{active_email}': {e}")
        return None


def handle_google_tool_error(err: Exception, action_name: str) -> str:
    """Helper to detect auth/token expiration issues and return a clean re-authentication link."""
    err_str = str(err).lower()
    if any(k in err_str for k in ["refresh_token", "invalid_grant", "unauthorized", "credentials", "401", "token expired"]):
        return (
            "⚠️ Google Calendar authentication is currently missing or expired. "
            "Please [Click here to Re-authenticate with Google](/auth/login) to enable scheduling."
        )
    return f"Failed to {action_name}: {str(err)}"


# ---------------------------------------------------------------------------
# 5. Async & High-Performance Google Tool Functions
# ---------------------------------------------------------------------------
async def list_events_tool(creds: Credentials, time_min: Optional[str] = None, max_results: int = 15) -> str:
    """Lists upcoming events from the active Google Calendar asynchronously."""
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)
        if not time_min:
            time_min = datetime.now(ZoneInfo(CALENDAR_TIMEZONE)).isoformat()
        else:
            time_min = normalize_iso_datetime(time_min)

        events_result = await asyncio.to_thread(
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute
        )
        events = events_result.get("items", [])

        if not events:
            return "No upcoming events found on your Google Calendar."

        formatted_events = []
        for ev in events:
            start = ev["start"].get("dateTime", ev["start"].get("date"))
            end = ev["end"].get("dateTime", ev["end"].get("date"))
            summary = ev.get("summary", "(No title)")
            location = ev.get("location", "No location specified")
            event_id = ev.get("id", "")
            link = ev.get("htmlLink", "")

            formatted_events.append({
                "id": event_id,
                "summary": summary,
                "start": start,
                "end": end,
                "location": location,
                "link": link,
            })

        return json.dumps(formatted_events, indent=2)

    except HttpError as err:
        logger.error(f"Google Calendar API Error: {err}")
        return handle_google_tool_error(err, "query calendar")
    except Exception as e:
        logger.error(f"Unexpected error in list_events: {e}")
        return handle_google_tool_error(e, "fetch calendar events")


# ---------------------------------------------------------------------------
# Google Calendar Color Mapping & Multi-Calendar Tools
# ---------------------------------------------------------------------------
GOOGLE_CALENDAR_COLORS: Dict[str, str] = {
    "lavender": "1",    # Light Purple / Personal
    "sage": "2",        # Light Green / Health & Wellness
    "grape": "3",       # Dark Purple
    "flamingo": "4",    # Light Pink
    "banana": "5",      # Yellow / Travel & Buffers
    "tangerine": "6",   # Orange / Tasks
    "peacock": "7",     # Cyan / Work Syncs
    "graphite": "8",    # Gray / Rest & Breaks
    "blueberry": "9",   # Blue / Standard Meetings
    "basil": "10",      # Green / Fitness & Habits
    "tomato": "11",     # Red / Urgent & Deadlines
}

COLOR_KEYWORD_MAP: Dict[str, str] = {
    "purple": "1",
    "lavender": "1",
    "personal": "1",
    "health": "2",
    "doctor": "2",
    "wellness": "2",
    "sage": "2",
    "dark purple": "3",
    "grape": "3",
    "pink": "4",
    "flamingo": "4",
    "yellow": "5",
    "travel": "5",
    "banana": "5",
    "orange": "6",
    "task": "6",
    "tangerine": "6",
    "cyan": "7",
    "work": "7",
    "sync": "7",
    "peacock": "7",
    "gray": "8",
    "grey": "8",
    "rest": "8",
    "break": "8",
    "graphite": "8",
    "blue": "9",
    "meeting": "9",
    "blueberry": "9",
    "green": "10",
    "fitness": "10",
    "workout": "10",
    "gym": "10",
    "sport": "10",
    "basil": "10",
    "red": "11",
    "urgent": "11",
    "critical": "11",
    "deadline": "11",
    "high priority": "11",
    "tomato": "11",
}


def map_to_google_color_id(color_or_keyword: Optional[str]) -> Optional[str]:
    """Translates natural language colors and category keywords into standard Google Calendar colorId values."""
    if not color_or_keyword:
        return None
    raw = str(color_or_keyword).strip().lower()
    if raw in COLOR_KEYWORD_MAP:
        return COLOR_KEYWORD_MAP[raw]
    for k, v in COLOR_KEYWORD_MAP.items():
        if k in raw:
            return v
    return None


async def list_user_calendars_tool(creds: Credentials) -> str:
    """
    Lists all Google Calendars accessible by the user (Primary, Work, Personal, etc.).
    """
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)
        cal_list = await asyncio.to_thread(service.calendarList().list().execute)
        calendars = []
        for item in cal_list.get("items", []):
            calendars.append({
                "id": item.get("id"),
                "summary": item.get("summary"),
                "description": item.get("description", ""),
                "primary": item.get("primary", False),
                "backgroundColor": item.get("backgroundColor"),
            })
        return json.dumps({"status": "success", "calendars": calendars}, indent=2)
    except Exception as e:
        logger.error(f"Error fetching user calendars: {e}")
        return f"Failed to list calendars: {str(e)}"


async def create_event_tool(
    creds: Credentials,
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    attendees: Optional[List[str]] = None,
    add_google_meet: bool = False,
    travel_buffer_minutes: Optional[int] = None,
    recurrence_rule: Optional[str] = None,
    color: Optional[str] = None,
    calendar_id: str = "primary",
    ignore_conflicts: bool = False,
) -> str:
    """
    Creates a new event on user's Google Calendar with smart conflict detection, automated travel time buffers, Google Meet video link generation, guest invitations, color coding, and RFC 5545 recurrence rules in IST.
    """
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)
        tz = ZoneInfo(CALENDAR_TIMEZONE)

        # Normalize start_time and end_time to ensure explicit Indian Standard Time (+05:30) offset
        norm_start = normalize_iso_datetime(start_time, CALENDAR_TIMEZONE)
        norm_end = normalize_iso_datetime(end_time, CALENDAR_TIMEZONE)

        try:
            req_start = datetime.fromisoformat(norm_start)
            req_end = datetime.fromisoformat(norm_end)
        except Exception:
            req_start = datetime.now(tz)
            req_end = req_start + timedelta(hours=1)
            norm_start = req_start.isoformat()
            norm_end = req_end.isoformat()

        if req_end <= req_start:
            req_end = req_start + timedelta(hours=1)
            norm_end = req_end.isoformat()

        duration = req_end - req_start

        # 1. Smart Conflict Detection & Alternative Slot Search (unless ignore_conflicts is requested or recurring)
        target_cal_id = calendar_id or "primary"
        if not ignore_conflicts and not recurrence_rule:
            search_window_start = req_start.replace(hour=0, minute=0, second=0, microsecond=0)
            search_window_end = (req_start + timedelta(days=2)).replace(hour=23, minute=59, second=59, microsecond=0)

            events_result = await asyncio.to_thread(
                service.events()
                .list(
                    calendarId=target_cal_id,
                    timeMin=search_window_start.isoformat(),
                    timeMax=search_window_end.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute
            )
            existing_events = events_result.get("items", [])

            # Check direct overlaps with requested [req_start, req_end]
            conflicts = []
            parsed_existing = []
            for ev in existing_events:
                s_raw = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
                e_raw = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date")
                if not s_raw or not e_raw:
                    continue

                try:
                    s_dt = datetime.fromisoformat(normalize_iso_datetime(s_raw, CALENDAR_TIMEZONE))
                    e_dt = datetime.fromisoformat(normalize_iso_datetime(e_raw, CALENDAR_TIMEZONE))
                    parsed_existing.append((s_dt, e_dt, ev))

                    # Direct overlap condition: (s_dt < req_end) and (e_dt > req_start)
                    if s_dt < req_end and e_dt > req_start:
                        conflicts.append((ev.get("summary", "Busy / Untitled Event"), s_dt, e_dt))
                except Exception as parse_err:
                    logger.debug(f"Error parsing existing event date: {parse_err}")

            if conflicts:
                # Conflicting event found! Build suggestions for next available free slots
                primary_conf_name, conf_start, conf_end = conflicts[0]
                conf_time_str = f"{conf_start.strftime('%I:%M %p')} - {conf_end.strftime('%I:%M %p')}"

                now_ist = datetime.now(tz)
                alternative_slots = []

                # Candidate days to check: same day + next 2 days
                candidate_days = [req_start.date(), (req_start + timedelta(days=1)).date(), (req_start + timedelta(days=2)).date()]

                for cand_date in candidate_days:
                    if len(alternative_slots) >= 3:
                        break

                    # Search between 9:00 AM and 7:00 PM IST
                    day_start = datetime(cand_date.year, cand_date.month, cand_date.day, 9, 0, 0, tzinfo=tz)
                    day_end = datetime(cand_date.year, cand_date.month, cand_date.day, 19, 0, 0, tzinfo=tz)

                    current_slot_start = day_start
                    while current_slot_start + duration <= day_end:
                        current_slot_end = current_slot_start + duration

                        # Must be in the future
                        if current_slot_start > now_ist:
                            # Check if this candidate slot overlaps with ANY existing event
                            has_overlap = False
                            for s_dt, e_dt, _ in parsed_existing:
                                if s_dt < current_slot_end and e_dt > current_slot_start:
                                    has_overlap = True
                                    break

                            # Also must not be identical to the requested slot that conflicted
                            if not has_overlap and (current_slot_start != req_start):
                                day_label = "Today" if cand_date == now_ist.date() else ("Tomorrow" if cand_date == (now_ist + timedelta(days=1)).date() else cand_date.strftime("%a, %b %d"))
                                slot_formatted = f"{current_slot_start.strftime('%I:%M %p')} - {current_slot_end.strftime('%I:%M %p')} IST ({day_label})"
                                alternative_slots.append({
                                    "start_time": current_slot_start.isoformat(),
                                    "end_time": current_slot_end.isoformat(),
                                    "formatted": slot_formatted,
                                })
                                if len(alternative_slots) >= 3:
                                    break

                        current_slot_start += timedelta(minutes=30)

                conflict_response = {
                    "status": "conflict_detected",
                    "message": f"Conflict detected! You already have '{primary_conf_name}' scheduled from {conf_time_str} on {req_start.strftime('%b %d, %Y')}.",
                    "conflicting_event": primary_conf_name,
                    "conflicting_time": conf_time_str,
                    "requested_slot": f"{req_start.strftime('%I:%M %p')} - {req_end.strftime('%I:%M %p')} on {req_start.strftime('%b %d, %Y')}",
                    "alternative_slots": alternative_slots,
                    "suggestion": "Inform user of the conflict and ask if they would like to schedule at one of these alternative slots or force schedule anyway.",
                }
                return json.dumps(conflict_response, indent=2)

        # 2. Automated Travel Time Buffer Creation (if location specified)
        buffer_info = None
        buffer_minutes = travel_buffer_minutes if travel_buffer_minutes is not None else (30 if location.strip() else 0)

        if location.strip() and buffer_minutes > 0:
            buf_start_dt = req_start - timedelta(minutes=buffer_minutes)
            buf_end_dt = req_start
            buf_start_iso = buf_start_dt.isoformat()
            buf_end_iso = buf_end_dt.isoformat()

            buffer_body = {
                "summary": f"🚗 Travel to {location.strip()}",
                "description": f"Automated {buffer_minutes}-minute travel buffer before '{summary}'.",
                "start": {
                    "dateTime": buf_start_iso,
                    "timeZone": CALENDAR_TIMEZONE,
                },
                "end": {
                    "dateTime": buf_end_iso,
                    "timeZone": CALENDAR_TIMEZONE,
                },
                "colorId": "5",  # Yellow / Banana in Google Calendar
            }
            try:
                created_buf = await asyncio.to_thread(
                    service.events().insert(calendarId=target_cal_id, body=buffer_body).execute
                )
                buffer_info = {
                    "minutes": buffer_minutes,
                    "start": buf_start_iso,
                    "end": buf_end_iso,
                    "event_id": created_buf.get("id"),
                    "summary": buffer_body["summary"],
                }
            except Exception as buf_err:
                logger.warning(f"Failed to create travel buffer event: {buf_err}")

        # 3. Insert Main Event (with optional RRULE recurrence & color)
        event_body: Dict[str, Any] = {
            "summary": summary,
            "description": description or "Scheduled via AI Calendar Agent",
            "start": {
                "dateTime": norm_start,
                "timeZone": CALENDAR_TIMEZONE,  # 👈 Forces IST interpretation ('Asia/Kolkata')
            },
            "end": {
                "dateTime": norm_end,
                "timeZone": CALENDAR_TIMEZONE,  # 👈 Forces IST interpretation ('Asia/Kolkata')
            },
        }

        if location:
            event_body["location"] = location

        if attendees:
            event_body["attendees"] = [{"email": email.strip()} for email in attendees if email.strip()]

        if recurrence_rule and recurrence_rule.strip():
            rrule_clean = recurrence_rule.strip()
            if not rrule_clean.upper().startswith("RRULE:"):
                rrule_clean = f"RRULE:{rrule_clean}"
            event_body["recurrence"] = [rrule_clean]

        color_id = map_to_google_color_id(color)
        if color_id:
            event_body["colorId"] = color_id

        if add_google_meet:
            req_id = f"meet-{uuid.uuid4().hex[:8]}-{int(datetime.now().timestamp())}"
            event_body["conferenceData"] = {
                "createRequest": {
                    "requestId": req_id,
                    "conferenceSolutionKey": {"type": "hangoutsMeet"}
                }
            }

        insert_kwargs: Dict[str, Any] = {
            "calendarId": target_cal_id,
            "body": event_body,
        }
        if add_google_meet:
            insert_kwargs["conferenceDataVersion"] = 1
        if attendees:
            insert_kwargs["sendUpdates"] = "all"

        created_event = await asyncio.to_thread(
            service.events().insert(**insert_kwargs).execute
        )

        meet_link = (
            created_event.get("hangoutLink")
            or created_event.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri")
            or None
        )

        msg_parts = [f"Successfully created event: '{summary}'"]
        if buffer_info:
            msg_parts.append(f"with an automated {buffer_minutes}-minute travel buffer (🚗 {buf_start_dt.strftime('%I:%M %p')} - {buf_end_dt.strftime('%I:%M %p')})")

        result = {
            "status": "success",
            "message": " ".join(msg_parts),
            "event_id": created_event.get("id"),
            "htmlLink": created_event.get("htmlLink"),
            "google_meet_link": meet_link,
            "attendees": attendees if attendees else [],
            "location": location,
            "color": color,
            "color_id": color_id,
            "calendar_id": target_cal_id,
            "travel_buffer": buffer_info,
            "start": norm_start,
            "end": norm_end,
            "timeZone": CALENDAR_TIMEZONE,
        }
        return json.dumps(result, indent=2)

    except HttpError as err:
        logger.error(f"Google Calendar Insert Error: {err}")
        return handle_google_tool_error(err, "create calendar event")
    except Exception as e:
        logger.error(f"Unexpected error in create_event: {e}")
        return handle_google_tool_error(e, "create event")


async def delete_calendar_event_tool(
    creds: Credentials,
    event_ids: Optional[List[str]] = None,
    event_id: Optional[str] = None,
    summary: Optional[str] = None,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    send_updates: str = "all",
) -> str:
    """
    Deletes single or multiple calendar events from user's primary Google Calendar with sendUpdates='all'.
    Accepts a list of event_ids, single event_id, or searches by title/summary and time range in IST.
    """
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)

        target_ids: List[str] = []
        if event_ids:
            target_ids.extend([eid.strip() for eid in event_ids if eid.strip()])
        if event_id and event_id.strip() and event_id.strip() not in target_ids:
            target_ids.append(event_id.strip())

        # If no explicit IDs provided, search by summary and time window
        if not target_ids:
            if not summary and not time_min and not time_max:
                return "Please specify event_ids, event_id, or a title/time range to delete."

            search_kwargs: Dict[str, Any] = {
                "calendarId": "primary",
                "singleEvents": True,
                "maxResults": 20,
            }
            if summary:
                search_kwargs["q"] = summary.strip()
            if time_min:
                search_kwargs["timeMin"] = normalize_iso_datetime(time_min, CALENDAR_TIMEZONE)
            if time_max:
                search_kwargs["timeMax"] = normalize_iso_datetime(time_max, CALENDAR_TIMEZONE)

            events_result = await asyncio.to_thread(
                service.events().list(**search_kwargs).execute
            )
            items = events_result.get("items", [])

            if not items and summary:
                # Fallback search without time constraints
                events_result = await asyncio.to_thread(
                    service.events().list(calendarId="primary", q=summary.strip(), maxResults=10, singleEvents=True).execute
                )
                items = events_result.get("items", [])

            if not items:
                search_desc = f"matching '{summary}'" if summary else "in the specified time range"
                return f"Could not find any calendar events {search_desc} to delete."

            target_ids = [item["id"] for item in items]

        deleted_count = 0
        deleted_details = []
        failed_errors = []

        for eid in target_ids:
            try:
                await asyncio.to_thread(
                    service.events().delete(
                        calendarId="primary",
                        eventId=eid,
                        sendUpdates=send_updates,
                    ).execute
                )
                deleted_count += 1
                deleted_details.append(eid)
            except Exception as del_err:
                logger.warning(f"Error deleting event {eid}: {del_err}")
                failed_errors.append(f"ID {eid}: {str(del_err)}")

        result = {
            "status": "success" if deleted_count > 0 else "error",
            "deleted_count": deleted_count,
            "deleted_event_ids": deleted_details,
            "message": f"Successfully deleted {deleted_count} calendar event(s)." if deleted_count > 0 else "Failed to delete specified events.",
            "errors": failed_errors if failed_errors else None,
        }
        return json.dumps(result, indent=2)

    except HttpError as err:
        logger.error(f"Google Calendar Delete Error: {err}")
        return handle_google_tool_error(err, "delete calendar event")
    except Exception as e:
        logger.error(f"Unexpected error in delete_calendar_event: {e}")
        return handle_google_tool_error(e, "delete event")


async def update_calendar_event_tool(
    creds: Credentials,
    event_id: Optional[str] = None,
    summary_search: Optional[str] = None,
    calendar_id: str = "primary",
    new_title: Optional[str] = None,
    new_start_datetime: Optional[str] = None,
    new_end_datetime: Optional[str] = None,
    new_duration_minutes: Optional[int] = None,
    new_color: Optional[str] = None,
    new_description: Optional[str] = None,
    new_location: Optional[str] = None,
) -> str:
    """
    Updates, reschedules, moves, extends, colors, or renames an existing Google Calendar event using patch() in IST.
    """
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)
        target_id = event_id.strip() if event_id else None

        # 1. Search for event_id if not provided directly
        if not target_id:
            if not summary_search:
                return "Error: Please specify either an `event_id` or `summary_search` (event title) to update."

            now_iso = datetime.now(ZoneInfo(CALENDAR_TIMEZONE)).isoformat()
            events_result = await asyncio.to_thread(
                service.events()
                .list(
                    calendarId=calendar_id,
                    q=summary_search,
                    timeMin=now_iso,
                    maxResults=5,
                    singleEvents=True,
                )
                .execute
            )
            items = events_result.get("items", [])
            if not items:
                # Fallback search without timeMin
                events_result = await asyncio.to_thread(
                    service.events()
                    .list(
                        calendarId=calendar_id,
                        q=summary_search,
                        maxResults=5,
                        singleEvents=True,
                    )
                    .execute
                )
                items = events_result.get("items", [])

            if not items:
                return f"Could not find any calendar event matching '{summary_search}' to update."

            matched_event = items[0]
            target_id = matched_event["id"]

        # 2. Retrieve existing event details for duration/time calculations
        existing_event = await asyncio.to_thread(
            service.events().get(calendarId=calendar_id, eventId=target_id).execute
        )

        patch_body: Dict[str, Any] = {}

        if new_title:
            patch_body["summary"] = new_title

        if new_color:
            c_id = map_to_google_color_id(new_color)
            if c_id:
                patch_body["colorId"] = c_id

        if new_description is not None:
            patch_body["description"] = new_description

        if new_location is not None:
            patch_body["location"] = new_location

        # Handle start and end times in IST
        if new_start_datetime:
            norm_new_start = normalize_iso_datetime(new_start_datetime, CALENDAR_TIMEZONE)
            patch_body["start"] = {
                "dateTime": norm_new_start,
                "timeZone": CALENDAR_TIMEZONE,
            }

            # If new end time is explicitly given
            if new_end_datetime:
                norm_new_end = normalize_iso_datetime(new_end_datetime, CALENDAR_TIMEZONE)
                patch_body["end"] = {
                    "dateTime": norm_new_end,
                    "timeZone": CALENDAR_TIMEZONE,
                }
            elif new_duration_minutes:
                start_dt = datetime.fromisoformat(norm_new_start)
                end_dt = start_dt + timedelta(minutes=new_duration_minutes)
                patch_body["end"] = {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": CALENDAR_TIMEZONE,
                }
            else:
                # Preserve original duration
                orig_start_str = existing_event.get("start", {}).get("dateTime") or existing_event.get("start", {}).get("date")
                orig_end_str = existing_event.get("end", {}).get("dateTime") or existing_event.get("end", {}).get("date")
                if orig_start_str and orig_end_str:
                    try:
                        orig_start = datetime.fromisoformat(normalize_iso_datetime(orig_start_str, CALENDAR_TIMEZONE))
                        orig_end = datetime.fromisoformat(normalize_iso_datetime(orig_end_str, CALENDAR_TIMEZONE))
                        orig_dur = orig_end - orig_start
                        start_dt = datetime.fromisoformat(norm_new_start)
                        patch_body["end"] = {
                            "dateTime": (start_dt + orig_dur).isoformat(),
                            "timeZone": CALENDAR_TIMEZONE,
                        }
                    except Exception:
                        pass
        elif new_end_datetime:
            norm_new_end = normalize_iso_datetime(new_end_datetime, CALENDAR_TIMEZONE)
            patch_body["end"] = {
                "dateTime": norm_new_end,
                "timeZone": CALENDAR_TIMEZONE,
            }
        elif new_duration_minutes:
            orig_start_str = existing_event.get("start", {}).get("dateTime") or existing_event.get("start", {}).get("date")
            if orig_start_str:
                orig_start = datetime.fromisoformat(normalize_iso_datetime(orig_start_str, CALENDAR_TIMEZONE))
                new_end = orig_start + timedelta(minutes=new_duration_minutes)
                patch_body["end"] = {
                    "dateTime": new_end.isoformat(),
                    "timeZone": CALENDAR_TIMEZONE,
                }

        if not patch_body:
            return "No modification parameters provided to update the event."

        updated_event = await asyncio.to_thread(
            service.events().patch(
                calendarId=calendar_id,
                eventId=target_id,
                body=patch_body,
            ).execute
        )

        start_val = updated_event.get("start", {}).get("dateTime") or updated_event.get("start", {}).get("date")
        end_val = updated_event.get("end", {}).get("dateTime") or updated_event.get("end", {}).get("date")

        result = {
            "status": "success",
            "message": f"Successfully updated event: '{updated_event.get('summary')}'",
            "event_id": updated_event.get("id"),
            "htmlLink": updated_event.get("htmlLink"),
            "start": start_val,
            "end": end_val,
            "timeZone": CALENDAR_TIMEZONE,
            "location": updated_event.get("location", ""),
        }
        return json.dumps(result, indent=2)

    except HttpError as err:
        logger.error(f"Google Calendar Patch Error: {err}")
        return handle_google_tool_error(err, "update calendar event")
    except Exception as e:
        logger.error(f"Unexpected error in update_calendar_event: {e}")
        return handle_google_tool_error(e, "update event")


async def check_gmail_invites_tool(creds: Credentials, max_results: int = 10) -> str:
    """
    Searches unread emails for meeting, tea, coffee, or sync invitations.
    Fetches message details in parallel with asyncio.gather to minimize latency.
    """
    try:
        service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds, static_discovery=False)
        query = "is:unread (tea OR coffee OR meetup OR meeting OR sync OR invite)"

        response = await asyncio.to_thread(
            service.users().messages().list(userId="me", q=query, maxResults=max_results).execute
        )
        messages = response.get("messages", [])

        if not messages:
            return "No unread invitation emails (tea, coffee, meetup, or meetings) found in your Gmail inbox."

        async def fetch_message_meta(msg_id: str):
            try:
                msg_data = await asyncio.to_thread(
                    service.users().messages().get(
                        userId="me", id=msg_id, format="metadata",
                        metadataHeaders=["Subject", "From", "Date"]
                    ).execute
                )
                headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
                return {
                    "message_id": msg_id,
                    "subject": headers.get("Subject", "(No Subject)"),
                    "sender": headers.get("From", "Unknown Sender"),
                    "date": headers.get("Date", ""),
                    "snippet": msg_data.get("snippet", ""),
                }
            except Exception as meta_err:
                logger.warning(f"Error fetching message {msg_id}: {meta_err}")
                return None

        # Execute parallel retrieval for sub-second performance
        invitations_raw = await asyncio.gather(*[fetch_message_meta(msg["id"]) for msg in messages])
        invitations = [inv for inv in invitations_raw if inv is not None]

        return json.dumps(invitations, indent=2)

    except HttpError as err:
        logger.error(f"Gmail API Error: {err}")
        return handle_google_tool_error(err, "query Gmail")
    except Exception as e:
        logger.error(f"Unexpected error in check_gmail_invites: {e}")
        return handle_google_tool_error(e, "search Gmail invites")


def create_oauth_flow(request: Request, state: Optional[str] = None) -> Flow:
    """Creates a configured Google OAuth 2.0 Flow instance."""
    client_config = get_client_config()
    redirect_uri = get_redirect_uri(request)

    return Flow.from_client_config(
        client_config=client_config,
        scopes=SCOPES,
        state=state,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False,
    )


# ---------------------------------------------------------------------------
# 6. Google OAuth 2.0 & Multi-Account Endpoints
# ---------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serves the dedicated Login/Sign Up page for unauthenticated users."""
    active_email = get_active_account_email(request)
    if active_email:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    try:
        return templates.TemplateResponse(request=request, name="login.html", context={"request": request})
    except Exception:
        return templates.TemplateResponse("login.html", {"request": request})


@app.get("/auth/login", response_class=RedirectResponse)
@app.get("/login/google", response_class=RedirectResponse)
async def auth_login(request: Request):
    """Initiates Google OAuth 2.0 consent flow."""
    try:
        flow = create_oauth_flow(request)
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        request.session["oauth_state"] = state
        return RedirectResponse(url=authorization_url, status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        logger.error(f"Login initiation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate Google OAuth: {str(e)}",
        )


@app.get("/auth/add-account", response_class=RedirectResponse)
async def auth_add_account(request: Request):
    """Initiates OAuth consent flow to connect an additional Google account with account selection and consent prompt."""
    try:
        flow = create_oauth_flow(request)
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            prompt="consent select_account",
            include_granted_scopes="true",
        )

        request.session["oauth_state"] = state
        return RedirectResponse(url=authorization_url, status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        logger.error(f"Add account initiation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate account connection: {str(e)}",
        )


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    """
    Handles OAuth callback:
    - Exchanges authorization code for tokens.
    - Captures and persists the refresh_token in session/database.
    - If Google omits the refresh_token, preserves the previously stored refresh_token if present.
    - Flags the session to force re-consent if refresh_token is missing.
    - Stores credentials inside the multi-account dictionary in session and sets it active.
    """
    state = request.session.get("oauth_state")
    code = request.query_params.get("code")

    if not code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing authorization code from Google.")

    try:
        flow = create_oauth_flow(request, state=state)

        auth_response_url = str(request.url)
        redirect_uri = get_redirect_uri(request)
        if redirect_uri.startswith("https://") and auth_response_url.startswith("http://"):
            auth_response_url = auth_response_url.replace("http://", "https://", 1)

        flow.fetch_token(authorization_response=auth_response_url)
        credentials = flow.credentials

        userinfo_service = build("oauth2", "v2", credentials=credentials, static_discovery=False)
        user_info = userinfo_service.userinfo().get().execute()
        email = user_info.get("email", "").strip()
        name = user_info.get("name", email)
        picture = user_info.get("picture", "")

        # Safely preserve existing accounts and merge new one
        existing_accounts = get_accounts_dict(request)
        accounts = dict(existing_accounts)
        already_connected = email in accounts
        existing_entry = accounts.get(email, {})

        # Extract refresh_token from token exchange payload, preserving existing one if not returned
        new_refresh_token = credentials.refresh_token or existing_entry.get("refresh_token")

        if not new_refresh_token:
            logger.warning(f"Google OAuth did not return a refresh_token for '{email}'. Flagging session to force re-consent.")
            request.session["account_warning"] = "Missing offline refresh token credentials. Please re-authenticate and grant full consent."

        # Immediately update and overwrite the existing user's credentials entry in session
        accounts[email] = {
            "token": credentials.token,
            "refresh_token": new_refresh_token,
            "token_uri": credentials.token_uri or "https://oauth2.googleapis.com/token",
            "name": name,
            "picture": picture,
            "scopes": credentials.scopes or SCOPES,
        }

        # Explicitly assign dictionary back to session
        request.session["accounts"] = accounts
        request.session["active_account"] = email

        if already_connected:
            request.session["account_notice"] = f"ℹ️ Account '{email}' credentials updated with persistent refresh token and set as active."
        else:
            request.session["account_notice"] = f"✅ Successfully connected Google account: '{email}'."

        # Top-level sync
        request.session["user_email"] = email
        request.session["user_name"] = name
        request.session["user_picture"] = picture
        request.session.pop("oauth_state", None)

        logger.info(f"Google Account '{email}' successfully merged into session (refresh_token present: {bool(new_refresh_token)}) & set as active.")
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        logger.error(f"OAuth token exchange failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to authenticate with Google: {str(e)}",
        )


@app.get("/logout")
async def logout(request: Request):
    """Clears all user sessions and accounts and redirects to /login."""
    active_email = get_active_account_email(request)
    request.session.clear()
    if active_email:
        logger.info(f"User '{active_email}' and all accounts logged out.")
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/api/me")
async def get_current_user(request: Request):
    """Returns current active user profile and connected accounts metadata."""
    active_email = get_active_account_email(request)
    if not active_email:
        return JSONResponse({"authenticated": False})

    accounts = get_accounts_dict(request)
    active_data = accounts.get(active_email, {})

    return JSONResponse({
        "authenticated": True,
        "email": active_email,
        "name": active_data.get("name", active_email),
        "picture": active_data.get("picture", ""),
        "total_accounts": len(accounts),
    })


# ---------------------------------------------------------------------------
# 7. Multi-Account Management API Endpoints
# ---------------------------------------------------------------------------
class SwitchAccountRequest(BaseModel):
    email: str


@app.get("/api/accounts")
async def list_accounts_endpoint(request: Request):
    """Returns all connected Google accounts, marks active one, and returns flash notice."""
    active_email = get_active_account_email(request)
    if not active_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized.")

    accounts = get_accounts_dict(request)
    account_list = []
    for email, acc in accounts.items():
        account_list.append({
            "email": email,
            "name": acc.get("name", email),
            "picture": acc.get("picture", ""),
            "is_active": (email == active_email),
        })

    notice = request.session.pop("account_notice", None)
    return JSONResponse({
        "active_account": active_email,
        "accounts": account_list,
        "notice": notice,
    })


@app.post("/api/accounts/switch")
async def switch_account_endpoint(request: Request, body: SwitchAccountRequest):
    """Switches the active Google account for Calendar and Gmail actions."""
    target_email = body.email.strip()
    accounts = dict(get_accounts_dict(request))

    if target_email not in accounts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{target_email}' is not connected.",
        )

    request.session["active_account"] = target_email
    acc_data = accounts[target_email]

    request.session["user_email"] = target_email
    request.session["user_name"] = acc_data.get("name", target_email)
    request.session["user_picture"] = acc_data.get("picture", "")
    request.session["accounts"] = accounts

    logger.info(f"Switched active account to: {target_email}")
    return JSONResponse({
        "success": True,
        "active_account": target_email,
        "name": acc_data.get("name", target_email),
    })


@app.post("/api/accounts/remove")
async def remove_account_endpoint(request: Request, body: SwitchAccountRequest):
    """Removes a specific connected Google account from session."""
    target_email = body.email.strip()
    accounts = dict(get_accounts_dict(request))

    if target_email not in accounts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found.")

    del accounts[target_email]
    request.session["accounts"] = accounts

    if not accounts:
        request.session.clear()
        return JSONResponse({"success": True, "redirect": "/login"})

    current_active = request.session.get("active_account")
    if current_active == target_email:
        new_active = next(iter(accounts.keys()))
        request.session["active_account"] = new_active
        acc_data = accounts[new_active]
        request.session["user_email"] = new_active
        request.session["user_name"] = acc_data.get("name", new_active)
        request.session["user_picture"] = acc_data.get("picture", "")
        request.session["user_creds"] = acc_data
    else:
        new_active = current_active

    return JSONResponse({
        "success": True,
        "active_account": new_active,
        "remaining_accounts": len(accounts),
    })


# ---------------------------------------------------------------------------
# 8. Calendar Event Direct API Endpoints
# ---------------------------------------------------------------------------
class DeleteEventRequest(BaseModel):
    event_id: str


@app.post("/api/calendar/delete")
async def delete_calendar_event_endpoint(request: Request, body: DeleteEventRequest):
    """
    Direct API endpoint to delete a Google Calendar event by ID
    using the active account's credentials.
    """
    user_creds = get_user_credentials(request)
    if not user_creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized. Please sign in.")

    event_id = body.event_id.strip()
    if not event_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="event_id is required.")

    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=user_creds, static_discovery=False)
        await asyncio.to_thread(service.events().delete(calendarId="primary", eventId=event_id).execute)
        logger.info(f"Direct API deleted calendar event: {event_id}")
        return JSONResponse({
            "status": "success",
            "message": "Event deleted successfully",
            "event_id": event_id
        })
    except HttpError as err:
        logger.error(f"Google Calendar Delete API error: {err}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": str(err)}
        )
    except Exception as e:
        logger.error(f"Unexpected error in delete_calendar_event_endpoint: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e)}
        )


# ---------------------------------------------------------------------------
# 9. Chat History API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history_endpoint(request: Request):
    """Returns all chat sessions for the active user."""
    active_email = get_active_account_email(request)
    if not active_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized.")

    sessions = await asyncio.to_thread(get_user_chat_sessions, active_email)
    return JSONResponse({"sessions": sessions})


@app.get("/api/history/{session_id}")
async def get_session_history_endpoint(session_id: str, request: Request):
    """Returns all messages belonging to a specific session."""
    active_email = get_active_account_email(request)
    if not active_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized.")

    data = await asyncio.to_thread(get_chat_session_details, session_id, active_email)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    return JSONResponse(data)


@app.delete("/api/history/{session_id}")
async def delete_session_endpoint(session_id: str, request: Request):
    """Deletes a specific chat session."""
    active_email = get_active_account_email(request)
    if not active_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized.")

    success = await asyncio.to_thread(delete_chat_session, session_id, active_email)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or already deleted.")

    return JSONResponse({"success": True, "message": "Session deleted."})


@app.delete("/api/history")
async def clear_all_history_endpoint(request: Request):
    """Deletes all chat history for the active user."""
    active_email = get_active_account_email(request)
    if not active_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized.")

    count = await asyncio.to_thread(delete_all_user_sessions, active_email)
    return JSONResponse({"success": True, "message": f"Cleared {count} chat sessions."})


# ---------------------------------------------------------------------------
# 10. AI Agent Runner & LangChain Integration
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@app.post("/api/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    High-Performance AI Chatbot endpoint:
    - Validates active user authentication.
    - Manages SQLite chat sessions and message persistence.
    - Binds asynchronous Google Calendar & Gmail tools with event creation & deletion.
    - Runs Gemini 3.6 Flash / 3.5 Flash with sub-second execution.
    """
    user_creds = get_user_credentials(request)
    active_email = get_active_account_email(request)
    if not user_creds or not active_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized. Please sign in with your Google Account first.",
        )

    clean_api_key = GEMINI_API_KEY.strip().strip("\"'") if GEMINI_API_KEY else None
    if not clean_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured on the server. Please add your Gemini API key in Render environment variables.",
        )

    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    # 1. Manage Active Chat Session in SQLite
    session_id = body.session_id
    session_title = ""

    if session_id:
        existing = await asyncio.to_thread(get_chat_session_details, session_id, active_email)
        if not existing:
            session_title = (user_message[:35] + "...") if len(user_message) > 35 else user_message
            session_id = await asyncio.to_thread(create_chat_session, active_email, session_title)
        else:
            session_title = existing["session"]["title"]
    else:
        session_title = (user_message[:35] + "...") if len(user_message) > 35 else user_message
        session_id = await asyncio.to_thread(create_chat_session, active_email, session_title)

    # Save incoming user message asynchronously
    await asyncio.to_thread(save_chat_message, session_id, "user", user_message, active_email)

    try:
        local_tz = ZoneInfo(CALENDAR_TIMEZONE)
        now_dt = datetime.now(local_tz)
        current_time_str = now_dt.strftime("%Y-%m-%d %I:%M %p %Z")
        current_iso_str = now_dt.isoformat()

        # 2. Async Wrappers for Tools
        async def list_events_wrapper(time_min: Optional[str] = None, max_results: int = 10) -> str:
            """Query upcoming Google Calendar events asynchronously."""
            return await list_events_tool(user_creds, time_min=time_min, max_results=max_results)

        async def create_event_wrapper(
            summary: str,
            start_time: str,
            end_time: str,
            description: str = "",
            location: str = "",
            attendees: Optional[List[str]] = None,
            add_google_meet: bool = False,
            travel_buffer_minutes: Optional[int] = None,
            recurrence_rule: Optional[str] = None,
            color: Optional[str] = None,
            calendar_id: str = "primary",
            ignore_conflicts: bool = False,
        ) -> str:
            """Create a new Google Calendar event in Indian Standard Time (IST) with smart conflict detection, automated travel time buffers, Google Meet video link generation, attendee invitations, color coding, and recurring schedule rules."""
            return await create_event_tool(
                user_creds,
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                description=description,
                location=location,
                attendees=attendees,
                add_google_meet=add_google_meet,
                travel_buffer_minutes=travel_buffer_minutes,
                recurrence_rule=recurrence_rule,
                color=color,
                calendar_id=calendar_id,
                ignore_conflicts=ignore_conflicts,
            )

        async def update_calendar_event_wrapper(
            event_id: Optional[str] = None,
            summary_search: Optional[str] = None,
            calendar_id: str = "primary",
            new_title: Optional[str] = None,
            new_start_datetime: Optional[str] = None,
            new_end_datetime: Optional[str] = None,
            new_duration_minutes: Optional[int] = None,
            new_color: Optional[str] = None,
            new_description: Optional[str] = None,
            new_location: Optional[str] = None,
        ) -> str:
            """Update, reschedule, move, extend, color, or rename an existing Google Calendar event."""
            return await update_calendar_event_tool(
                user_creds,
                event_id=event_id,
                summary_search=summary_search,
                calendar_id=calendar_id,
                new_title=new_title,
                new_start_datetime=new_start_datetime,
                new_end_datetime=new_end_datetime,
                new_duration_minutes=new_duration_minutes,
                new_color=new_color,
                new_description=new_description,
                new_location=new_location,
            )

        async def list_user_calendars_wrapper() -> str:
            """List all accessible Google Calendars for this user (Primary, Work, Personal, etc.)."""
            return await list_user_calendars_tool(user_creds)

        async def delete_calendar_events_wrapper(
            event_ids: Optional[List[str]] = None,
            event_id: Optional[str] = None,
            summary: Optional[str] = None,
            time_min: Optional[str] = None,
            time_max: Optional[str] = None,
            send_updates: str = "all",
        ) -> str:
            """Delete single or multiple Google Calendar events by list of event_ids, single event_id, or search query."""
            return await delete_calendar_event_tool(
                user_creds,
                event_ids=event_ids,
                event_id=event_id,
                summary=summary,
                time_min=time_min,
                time_max=time_max,
                send_updates=send_updates,
            )

        async def check_gmail_invites_wrapper(max_results: int = 10) -> str:
            """Scan unread Gmail messages for meeting, coffee, or tea invites in parallel."""
            return await check_gmail_invites_tool(user_creds, max_results=max_results)

        tools = [
            StructuredTool.from_function(
                coroutine=list_events_wrapper,
                name="list_events",
                description="List upcoming Google Calendar events. Optional time_min ISO string.",
            ),
            StructuredTool.from_function(
                coroutine=list_user_calendars_wrapper,
                name="list_user_calendars",
                description="List all accessible Google Calendars for this account (e.g. Primary, Work, Personal, etc.).",
            ),
            StructuredTool.from_function(
                coroutine=create_event_wrapper,
                name="create_event",
                description="Create Google Calendar event in IST (+05:30). Supports color ('blue', 'red', 'urgent', 'personal', 'green', 'yellow'), secondary calendars (calendar_id), recurrence_rule, travel buffer, and Google Meet.",
            ),
            StructuredTool.from_function(
                coroutine=update_calendar_event_wrapper,
                name="update_calendar_event",
                description="Update, reschedule, move, extend, color, or rename an existing Google Calendar event. Provide event_id or summary_search, and modified fields.",
            ),
            StructuredTool.from_function(
                coroutine=delete_calendar_events_wrapper,
                name="delete_calendar_events",
                description="Delete single or multiple Google Calendar events. Accepts a list of event_ids (for bulk cleanup), single event_id, or summary/time window search query.",
            ),
            StructuredTool.from_function(
                coroutine=check_gmail_invites_wrapper,
                name="check_gmail_invites",
                description="Scan unread Gmail for meeting, coffee, or sync invitations.",
            ),
        ]

        tool_map = {t.name: t for t in tools}

        # 3. Speed-Focused System Instructions with Explicit Indian Standard Time (IST), Multi-Calendar, Color Coding & Safe Bulk Deletions
        system_prompt = (
            "System Directive: You are a high-speed calendar AI assistant. Be concise, direct, and helpful. "
            f"The user's local timezone is Indian Standard Time (IST), timezone identifier '{CALENDAR_TIMEZONE}' (UTC+5:30).\n"
            f"All relative and absolute times mentioned by the user MUST be interpreted in IST ('{CALENDAR_TIMEZONE}'). "
            "Perform tool calls immediately without conversational filler.\n\n"
            f"Active Account: {active_email}\n"
            f"Current Local DateTime: {current_time_str} (ISO: {current_iso_str})\n"
            f"Default Timezone: {CALENDAR_TIMEZONE} (UTC+5:30)\n\n"
            "Instructions:\n"
            "1. When user asks about schedule or availability, call `list_events` with time_min relative to Current Local DateTime.\n"
            "2. When user asks to create/schedule an event:\n"
            "   - If a specific calendar is mentioned by name (e.g., 'Work calendar', 'Personal calendar'), call `list_user_calendars` to resolve its calendar_id.\n"
            "   - When a color (blue, red, green, yellow, purple, orange, cyan) or category/priority (urgent/critical -> red, fitness/gym -> green, personal/doctor -> purple/lavender, work/sync -> cyan) is requested, extract `color`.\n"
            "   - If recurring, parse frequency into valid RFC 5545 RRULE string for `recurrence_rule`:\n"
            "     * 'Every day' -> 'FREQ=DAILY'\n"
            "     * 'Every weekday' -> 'FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR'\n"
            "     * 'Every Monday' -> 'FREQ=WEEKLY;BYDAY=MO'\n"
            "     * 'Every 2 weeks on Tuesday' -> 'FREQ=WEEKLY;INTERVAL=2;BYDAY=TU'\n"
            "     * 'Monthly' -> 'FREQ=MONTHLY'\n"
            "   - If a venue is mentioned, extract it into `location` and specify `travel_buffer_minutes=30`.\n"
            "   - Extract guest emails into `attendees` (e.g. ['colleague@example.com']).\n"
            "   - Set `add_google_meet=True` if the user mentions Google Meet, video call, meeting link, online sync, or invites guests.\n"
            "   - Compute start_time and end_time in IST with '+05:30' offset and call `create_event`.\n"
            "   - If `create_event` returns `conflict_detected`:\n"
            "     * Inform the user clearly: \"You already have '[Conflicting Event]' scheduled at [Time].\"\n"
            "     * Present alternative slots cleanly as numbered options.\n"
            "     * If user picks an alternative slot, call `create_event` with that slot.\n"
            "     * If user says 'Schedule anyway' or 'Force schedule', call `create_event` with `ignore_conflicts=True`.\n"
            "   - If `create_event` succeeds, give a clear confirmation with event title, start/end time in IST, location/color (if any), recurrence/travel buffer (if any), Google Meet link (if any), attendees, and delete button: `<button class=\"btn-delete-event\" data-event-id=\"EVENT_ID\"><i class=\"fa-solid fa-trash-can\"></i> Delete</button>`.\n"
            "3. When user asks to edit, move, reschedule, extend, color, or rename an event:\n"
            "   - Call `update_calendar_event` with summary_search='...' and modified fields (new_title, new_color, new_start_datetime, new_end_datetime, etc.).\n"
            "   - Compute any new datetimes in IST ('Asia/Kolkata' +05:30).\n"
            "   - Confirm the modification in 1 short sentence with the updated event title, new time in IST, and delete button.\n"
            "4. When user asks to cancel, delete, or clean up events:\n"
            "   - For single or specific event deletion (e.g., 'Cancel my 3 PM meeting today', 'Delete Team Sync'):\n"
            "     * Call `delete_calendar_events` directly with summary or event_id and confirm in 1 sentence.\n"
            "   - For bulk deletion requests (e.g., 'Clear all my meetings for this Friday', 'Delete all events next week'):\n"
            "     * FIRST call `list_events` with the relevant time range (e.g., time_min and time_max for that day in IST) to retrieve matching events.\n"
            "     * If MORE THAN 3 events match: List the matching events (title and time) and ask for user confirmation before deleting.\n"
            "     * If user confirms or <= 3 events found: Call `delete_calendar_events` with `event_ids=[...]` and confirm the count of deleted events in IST.\n"
            "5. When checking emails, call `check_gmail_invites` and summarize briefly in 1-2 bullet points."
        )

        # 4. Initialize Gemini LLM with active Google AI models (excluding non-existent/deprecated ones)
        unique_candidates = get_gemini_model_candidates()

        # 5. Multi-turn Agent Execution Loop with Model Fallback
        max_iterations = 4
        final_text = ""
        last_error = None

        for model_name in unique_candidates:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=clean_api_key,
                    max_output_tokens=350,
                    max_retries=1,
                )
                llm_with_tools = llm.bind_tools(tools)

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]

                for _ in range(max_iterations):
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected for user '{active_email}'. Cancelling AI generation.")
                        return JSONResponse({"status": "cancelled", "message": "Request cancelled by user."}, status_code=200)

                    response = await acall_gemini_with_retry(
                        llm_with_tools.ainvoke,
                        messages,
                        max_retries=5,
                        initial_delay=2.0,
                    )
                    messages.append(response)

                    if not response.tool_calls:
                        content_raw = response.content
                        if isinstance(content_raw, str):
                            final_text = content_raw
                        elif isinstance(content_raw, list):
                            parts = []
                            for p in content_raw:
                                if isinstance(p, str):
                                    parts.append(p)
                                elif isinstance(p, dict):
                                    parts.append(p.get("text", json.dumps(p)))
                                else:
                                    parts.append(str(p))
                            final_text = "\n".join(parts)
                        elif isinstance(content_raw, dict):
                            final_text = content_raw.get("text", json.dumps(content_raw))
                        else:
                            final_text = str(content_raw) if content_raw else ""
                        break

                    # Check client disconnection before executing tools
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected before tool execution for user '{active_email}'. Cancelling.")
                        return JSONResponse({"status": "cancelled", "message": "Request cancelled by user."}, status_code=200)

                    # Execute tool calls asynchronously
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call.get("id", tool_name)

                        logger.info(f"Executing tool '{tool_name}' asynchronously with args: {tool_args}")

                        if tool_name in tool_map:
                            try:
                                tool_func = tool_map[tool_name]
                                tool_result = await tool_func.ainvoke(tool_args)
                            except Exception as tool_exec_err:
                                logger.error(f"Error running tool {tool_name}: {tool_exec_err}")
                                tool_result = f"Error executing {tool_name}: {str(tool_exec_err)}"
                        else:
                            tool_result = f"Tool '{tool_name}' not found."

                        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

                if final_text:
                    break

            except asyncio.CancelledError:
                logger.info(f"Task cancelled by client for user '{active_email}'.")
                return JSONResponse({"status": "cancelled", "message": "Request cancelled by user."}, status_code=200)
            except Exception as candidate_err:
                logger.warning(f"Model '{model_name}' execution attempt failed: {candidate_err}. Trying fallback model...")
                last_error = candidate_err
                continue

        if not final_text:
            if last_error:
                err_msg = str(last_error)
                if any(k in err_msg.lower() for k in ["resource_exhausted", "quota", "429"]):
                    logger.error(f"All Gemini models exhausted free tier quota: {last_error}")
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "error": "Google Gemini API free tier rate limit or quota exceeded. Please wait a few seconds before trying again, or check your Google AI Studio plan.",
                            "details": err_msg,
                        },
                    )
                raise last_error
            final_text = "I processed your request. Let me know if you need anything else with your schedule or emails!"

        # Save outgoing bot response to SQLite asynchronously
        await asyncio.to_thread(save_chat_message, session_id, "assistant", str(final_text), active_email)

        return JSONResponse({
            "response": str(final_text),
            "session_id": session_id,
            "title": session_title,
        })

    except Exception as e:
        logger.error(f"Chat execution failed: {e}", exc_info=True)
        err_str = str(e)
        if any(k in err_str.lower() for k in ["resource_exhausted", "quota", "429"]):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Google Gemini API free tier rate limit or quota exceeded. Please wait a few seconds before trying again, or check your Google AI Studio plan.",
                    "details": err_str,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"An error occurred while processing your request: {str(e)}"},
        )


# ---------------------------------------------------------------------------
# 10. Frontend UI & Legal Routes (/privacy, /terms, /)
# ---------------------------------------------------------------------------
@app.get("/privacy", response_class=HTMLResponse)
async def serve_privacy(request: Request):
    """
    Renders the official Privacy Policy for Google OAuth Verification & Google Search Console.
    Complies with Google API Services User Data Policy, including Limited Use requirements.
    """
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="google-site-verification" content="VEHl4mzgS2aSF1pyd69IZiLSW6EC2m2VrnC_A4tpTxo" />
    <title>Privacy Policy - AI Calendar Assistant</title>
    <meta name="description" content="Privacy Policy for AI Calendar Assistant. Learn how we access, use, and protect your Google Calendar and Gmail data in compliance with Google API Services User Data Policy.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        :root {
            --bg-primary: #0b0f17;
            --bg-card: rgba(17, 24, 39, 0.85);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-primary: #6366f1;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            --radius-lg: 16px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.7;
            padding: 2.5rem 1.25rem;
            min-height: 100vh;
        }
        .container {
            max-width: 840px;
            margin: 0 auto;
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 2.5rem 2.25rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .header {
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        .header h1 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .subtitle {
            font-size: 0.95rem;
            color: var(--text-secondary);
        }
        .header .last-updated {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }
        .btn-back {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            text-decoration: none;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 1.75rem;
            transition: all 0.2s ease;
        }
        .btn-back:hover {
            background: rgba(99, 102, 241, 0.25);
            color: #ffffff;
            transform: translateX(-3px);
        }
        h2 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 2rem;
            margin-bottom: 0.75rem;
        }
        p, ul {
            font-size: 0.92rem;
            color: var(--text-secondary);
            margin-bottom: 1.2rem;
        }
        ul { padding-left: 1.5rem; }
        li { margin-bottom: 0.5rem; }
        .highlight-box {
            background: rgba(99, 102, 241, 0.08);
            border-left: 4px solid var(--accent-primary);
            padding: 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1.5rem 0;
        }
        .highlight-box p {
            color: #e0e7ff;
            margin-bottom: 0;
            font-size: 0.9rem;
        }
        a { color: #818cf8; }
        .footer-note {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="btn-back"><i class="fa-solid fa-arrow-left"></i> Back to AI Calendar Assistant</a>
        
        <div class="header">
            <h1>Privacy Policy</h1>
            <div class="subtitle">AI Calendar Assistant</div>
            <div class="last-updated">Last Updated: August 29, 2026</div>
        </div>

        <p>This Privacy Policy describes how <strong>AI Calendar Assistant</strong> ("we", "our", or "the Service") collects, uses, and protects your information when you connect your Google Account to our application.</p>

        <h2>1. Google User Data Accessed and Scopes</h2>
        <p>AI Calendar Assistant requests explicit permission to access specific Google API scopes solely to deliver AI-driven scheduling assistance:</p>
        <ul>
            <li><strong>Google Calendar API (<code>https://www.googleapis.com/auth/calendar.events</code>, <code>https://www.googleapis.com/auth/calendar.readonly</code>)</strong>: Used solely to query upcoming schedule events, check availability, detect scheduling conflicts, insert new meetings with travel buffers and Google Meet links, reschedule or edit existing events, color-code, and delete events upon your conversational commands.</li>
            <li><strong>Gmail API (<code>https://www.googleapis.com/auth/gmail.readonly</code>)</strong>: Used solely to scan unread email threads for meeting, coffee, or sync invitations so you can schedule them in one click.</li>
            <li><strong>Google Profile & OpenID (<code>openid</code>, <code>https://www.googleapis.com/auth/userinfo.email</code>, <code>https://www.googleapis.com/auth/userinfo.profile</code>)</strong>: Used solely to authenticate your identity and display your email address and profile avatar.</li>
        </ul>

        <div class="highlight-box">
            <p><strong>Google API Services User Data Policy Compliance:</strong><br>
            AI Calendar Assistant's use and transfer to any other app of information received from Google APIs adheres to the <a href="https://developers.google.com/terms/api-services-user-data-policy" target="_blank">Google API Services User Data Policy</a>, including the <strong>Limited Use</strong> requirements.</p>
        </div>

        <h2>2. How We Use and Protect Your Data</h2>
        <ul>
            <li><strong>Zero Unauthorized Sharing:</strong> We do not sell, rent, trade, or transfer your Google Calendar or Gmail data to any third-party advertisers or data brokers.</li>
            <li><strong>No Model Training:</strong> Your Google Calendar events and email contents are never used to train or fine-tune generalized artificial intelligence (AI) or machine learning (ML) models.</li>
            <li><strong>Session-Based Encryption:</strong> Access and refresh tokens are encrypted and maintained in isolated, secure HTTP-only cookies on your browser during your active session.</li>
            <li><strong>Multi-User Isolation:</strong> Chat conversations are stored in a private SQLite database isolated strictly by authenticated user email. No other user can access your conversation logs.</li>
        </ul>

        <h2>3. Data Retention and Deletion Rights</h2>
        <p>You maintain 100% control over your data stored within AI Calendar Assistant:</p>
        <ul>
            <li><strong>Conversation Deletion:</strong> You can delete any individual chat conversation or wipe all history permanently using the "Clear all history" button in the app sidebar.</li>
            <li><strong>Account Sign-Out:</strong> Clicking "Sign Out" immediately destroys your active session and cached tokens.</li>
            <li><strong>OAuth Permissions Revocation:</strong> You can permanently revoke AI Calendar Assistant's access to your Google Account at any time through <a href="https://myaccount.google.com/permissions" target="_blank">Google Account Permissions Settings</a>.</li>
        </ul>

        <h2>4. Contact Information</h2>
        <p>If you have any questions regarding this Privacy Policy or our data practices, please contact us via our GitHub repository at <a href="https://github.com/DilipPalanisamy/ai-calendar-agent" target="_blank">github.com/DilipPalanisamy/ai-calendar-agent</a>.</p>

        <div class="footer-note">
            © 2026 AI Calendar Assistant. All rights reserved.
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/terms", response_class=HTMLResponse)
async def serve_terms(request: Request):
    """
    Renders the official Terms of Service for Google OAuth Verification & Google Search Console.
    """
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="google-site-verification" content="VEHl4mzgS2aSF1pyd69IZiLSW6EC2m2VrnC_A4tpTxo" />
    <title>Terms of Service - AI Calendar Assistant</title>
    <meta name="description" content="Terms of Service for AI Calendar Assistant. Learn the terms and conditions governing your use of our AI scheduling coordinator.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        :root {
            --bg-primary: #0b0f17;
            --bg-card: rgba(17, 24, 39, 0.85);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-primary: #6366f1;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            --radius-lg: 16px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.7;
            padding: 2.5rem 1.25rem;
            min-height: 100vh;
        }
        .container {
            max-width: 840px;
            margin: 0 auto;
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 2.5rem 2.25rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .header {
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        .header h1 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .subtitle {
            font-size: 0.95rem;
            color: var(--text-secondary);
        }
        .header .last-updated {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }
        .btn-back {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            text-decoration: none;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 1.75rem;
            transition: all 0.2s ease;
        }
        .btn-back:hover {
            background: rgba(99, 102, 241, 0.25);
            color: #ffffff;
            transform: translateX(-3px);
        }
        h2 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 2rem;
            margin-bottom: 0.75rem;
        }
        p, ul {
            font-size: 0.92rem;
            color: var(--text-secondary);
            margin-bottom: 1.2rem;
        }
        ul { padding-left: 1.5rem; }
        li { margin-bottom: 0.5rem; }
        a { color: #818cf8; }
        .footer-note {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="btn-back"><i class="fa-solid fa-arrow-left"></i> Back to AI Calendar Assistant</a>
        
        <div class="header">
            <h1>Terms of Service</h1>
            <div class="subtitle">AI Calendar Assistant</div>
            <div class="last-updated">Last Updated: August 29, 2026</div>
        </div>

        <h2>1. Acceptance of Terms</h2>
        <p>By accessing or using <strong>AI Calendar Assistant</strong>, you agree to be bound by these Terms of Service. If you disagree with any part of these terms, you may not access or use the Service.</p>

        <h2>2. Description of the Service</h2>
        <p>AI Calendar Assistant is an intelligent scheduling assistant powered by Google Gemini and Google APIs. It allows users to interactively manage their Google Calendar schedules, create and reschedule events, color-code meetings, and scan Gmail for invitations using natural language and voice commands.</p>

        <h2>3. User Responsibilities & Acceptable Use</h2>
        <ul>
            <li>You agree to use AI Calendar Assistant only for lawful scheduling and communication purposes.</li>
            <li>You are responsible for maintaining the confidentiality of your Google Account credentials and managing authorized devices.</li>
            <li>You agree not to attempt to reverse engineer, disrupt, or exploit the service infrastructure.</li>
        </ul>

        <h2>4. AI Disclaimers & Appointment Verification</h2>
        <p>AI Calendar Assistant utilizes advanced AI models to interpret dates, times, and participant lists. While designed for high accuracy and timezone awareness (IST / UTC+5:30), users are encouraged to verify important event details and invitations on their Google Calendar.</p>

        <h2>5. Service Availability & Modifications</h2>
        <p>We strive to provide uninterrupted service, but we do not guarantee 100% uptime. We reserve the right to modify, suspend, or discontinue any feature with or without notice.</p>

        <h2>6. Limitation of Liability</h2>
        <p>To the maximum extent permitted by law, AI Calendar Assistant and its maintainers shall not be liable for any indirect, incidental, special, or consequential damages resulting from your use of the service.</p>

        <h2>7. Contact</h2>
        <p>For questions or feedback regarding these Terms, please reach out via <a href="https://github.com/DilipPalanisamy/ai-calendar-agent" target="_blank">github.com/DilipPalanisamy/ai-calendar-agent</a>.</p>

        <div class="footer-note">
            © 2026 AI Calendar Assistant. All rights reserved.
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """
    Public Home Page compliant with Google OAuth Verification & Google Search Console.
    Serves a public HTML page with Google Sign-in for unauthenticated visitors (HTTP 200 OK, NO redirects).
    Serves the chat assistant interface if the user is authenticated.
    """
    creds = get_user_credentials(request)
    active_email = get_active_account_email(request)
    if not creds or not active_email:
        try:
            return templates.TemplateResponse(request=request, name="login.html", context={"request": request})
        except Exception:
            return templates.TemplateResponse("login.html", {"request": request})

    accounts = get_accounts_dict(request)
    active_data = accounts.get(active_email, {})

    context = {
        "request": request,
        "user_email": active_email,
        "user_name": active_data.get("name", active_email),
        "user_picture": active_data.get("picture", ""),
        "authenticated": True,
        "accounts_count": len(accounts),
    }

    try:
        return templates.TemplateResponse(request=request, name="index.html", context=context)
    except Exception:
        return templates.TemplateResponse("index.html", context)


# ---------------------------------------------------------------------------
# 11. Server Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting AI Calendar Assistant on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)