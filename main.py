import os
import json
import uuid
import sqlite3
import asyncio
import logging
from datetime import datetime, timezone
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
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DB_PATH = BASE_DIR / "chat_history.db"

# Jinja2 Templates setup
templates = Jinja2Templates(directory=str(TEMPLATES_DIR) if TEMPLATES_DIR.exists() else "templates")

# Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", "ai-calendar-agent-secret-key-production-ready-2026")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
CALENDAR_TIMEZONE = os.getenv("CALENDAR_TIMEZONE", "Asia/Kolkata")

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
    """Creates a connection to the SQLite database with row access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
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
    title="AI Calendar Agent",
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
    """Dynamically determine the OAuth callback redirect URI."""
    if RENDER_EXTERNAL_URL and RENDER_EXTERNAL_URL != "http://localhost:8000":
        base = RENDER_EXTERNAL_URL.rstrip("/")
        return f"{base}/auth/callback"

    # Handle proxy SSL on Render / Cloud Load Balancers
    if request.headers.get("x-forwarded-proto") == "https":
        url = str(request.url_for("auth_callback"))
        if url.startswith("http://"):
            return url.replace("http://", "https://", 1)

    return str(request.url_for("auth_callback"))


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
    Updates the session dictionary if tokens are refreshed.
    """
    accounts = get_accounts_dict(request)
    active_email = get_active_account_email(request)
    if not active_email or active_email not in accounts:
        return None

    creds_data = accounts[active_email]

    try:
        creds = Credentials(
            token=creds_data.get("token"),
            refresh_token=creds_data.get("refresh_token"),
            token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=creds_data.get("client_id"),
            client_secret=creds_data.get("client_secret"),
            scopes=creds_data.get("scopes", SCOPES),
        )

        if creds.expired and creds.refresh_token:
            logger.info(f"Access token expired for '{active_email}'. Refreshing...")
            creds.refresh(GoogleRequest())
            creds_data["token"] = creds.token
            creds_data["refresh_token"] = creds.refresh_token
            creds_data["token_uri"] = creds.token_uri
            accounts = get_accounts_dict(request)
            accounts[active_email] = creds_data
            request.session["accounts"] = accounts

        request.session["user_email"] = active_email
        request.session["user_name"] = creds_data.get("name", active_email)
        request.session["user_picture"] = creds_data.get("picture", "")
        request.session["user_creds"] = creds_data

        return creds
    except Exception as e:
        logger.error(f"Error loading credentials for '{active_email}': {e}")
        return None


# ---------------------------------------------------------------------------
# 5. Async & High-Performance Google Tool Functions
# ---------------------------------------------------------------------------
async def list_events_tool(creds: Credentials, time_min: Optional[str] = None, max_results: int = 15) -> str:
    """Lists upcoming events from the active Google Calendar asynchronously."""
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)
        if not time_min:
            time_min = datetime.now(timezone.utc).isoformat()

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
        return f"Error querying calendar: {err}"
    except Exception as e:
        logger.error(f"Unexpected error in list_events: {e}")
        return f"Failed to fetch calendar events: {str(e)}"


async def create_event_tool(
    creds: Credentials,
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    attendees: Optional[List[str]] = None,
) -> str:
    """Creates a new event on user's primary Google Calendar after asynchronous conflict checking."""
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)

        # Conflict checking in the target time window asynchronously
        conflict_msg = ""
        try:
            overlap_check = await asyncio.to_thread(
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_time if "T" in start_time else f"{start_time}T00:00:00Z",
                    timeMax=end_time if "T" in end_time else f"{end_time}T23:59:59Z",
                    singleEvents=True,
                )
                .execute
            )
            conflicts = overlap_check.get("items", [])
            if conflicts:
                conflict_names = [f"'{c.get('summary', 'Untitled')}'" for c in conflicts]
                conflict_msg = f"⚠️ Conflict Notice: You already have event(s) during this time: {', '.join(conflict_names)}."
        except Exception as check_err:
            logger.warning(f"Conflict check warning: {check_err}")

        # Construct event payload
        event_body: Dict[str, Any] = {
            "summary": summary,
            "description": description or "Scheduled via AI Calendar Agent",
            "start": {"dateTime": start_time, "timeZone": CALENDAR_TIMEZONE},
            "end": {"dateTime": end_time, "timeZone": CALENDAR_TIMEZONE},
        }

        if location:
            event_body["location"] = location

        if attendees:
            event_body["attendees"] = [{"email": email.strip()} for email in attendees if email.strip()]

        created_event = await asyncio.to_thread(
            service.events().insert(calendarId="primary", body=event_body).execute
        )

        result = {
            "status": "success",
            "message": f"Successfully created event: '{summary}'",
            "event_id": created_event.get("id"),
            "htmlLink": created_event.get("htmlLink"),
            "start": start_time,
            "end": end_time,
            "location": location,
            "conflict_warning": conflict_msg if conflict_msg else None,
        }
        return json.dumps(result, indent=2)

    except HttpError as err:
        logger.error(f"Google Calendar Insert Error: {err}")
        return f"Error creating calendar event: {err}"
    except Exception as e:
        logger.error(f"Unexpected error in create_event: {e}")
        return f"Failed to create event: {str(e)}"


async def delete_calendar_event_tool(
    creds: Credentials,
    event_id: Optional[str] = None,
    summary: Optional[str] = None,
    time_min: Optional[str] = None,
) -> str:
    """
    Deletes a calendar event from the user's primary Google Calendar.
    Accepts either an explicit event_id or searches by title/summary.
    """
    try:
        service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds, static_discovery=False)

        target_id = event_id.strip() if event_id else None
        target_summary = summary or ""

        if not target_id:
            if not summary:
                return "Please specify either an event_id or the title/summary of the event you would like to delete."

            search_time_min = time_min or datetime.now(timezone.utc).isoformat()
            events_result = await asyncio.to_thread(
                service.events()
                .list(
                    calendarId="primary",
                    q=summary,
                    timeMin=search_time_min,
                    maxResults=5,
                    singleEvents=True,
                )
                .execute
            )
            items = events_result.get("items", [])
            if not items:
                # Fallback search without time constraint
                events_result = await asyncio.to_thread(
                    service.events()
                    .list(
                        calendarId="primary",
                        q=summary,
                        maxResults=5,
                        singleEvents=True,
                    )
                    .execute
                )
                items = events_result.get("items", [])

            if not items:
                return f"Could not find any calendar event matching '{summary}' to delete."

            matched_event = items[0]
            target_id = matched_event["id"]
            target_summary = matched_event.get("summary", summary)

        # Delete event
        await asyncio.to_thread(
            service.events().delete(calendarId="primary", eventId=target_id).execute
        )

        result = {
            "status": "success",
            "message": f"Successfully deleted event '{target_summary}' (ID: {target_id}) from Google Calendar.",
            "event_id": target_id,
        }
        return json.dumps(result, indent=2)

    except HttpError as err:
        logger.error(f"Google Calendar Delete Error: {err}")
        return f"Error deleting calendar event: {err}"
    except Exception as e:
        logger.error(f"Unexpected error in delete_calendar_event: {e}")
        return f"Failed to delete event: {str(e)}"


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
        return f"Error querying Gmail: {err}"
    except Exception as e:
        logger.error(f"Unexpected error in check_gmail_invites: {e}")
        return f"Failed to search Gmail invites: {str(e)}"


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
        client_config = get_client_config()
        redirect_uri = get_redirect_uri(request)

        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=SCOPES,
            redirect_uri=redirect_uri,
            autogenerate_code_verifier=False,
        )

        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        request.session["oauth_state"] = state
        return RedirectResponse(url=authorization_url)

    except Exception as e:
        logger.error(f"Login initiation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate Google OAuth: {str(e)}",
        )


@app.get("/auth/add-account", response_class=RedirectResponse)
async def auth_add_account(request: Request):
    """Initiates OAuth consent flow to connect an additional Google account."""
    try:
        client_config = get_client_config()
        redirect_uri = get_redirect_uri(request)

        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=SCOPES,
            redirect_uri=redirect_uri,
            autogenerate_code_verifier=False,
        )

        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="select_account consent",
        )

        request.session["oauth_state"] = state
        return RedirectResponse(url=authorization_url)

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
    - Stores credentials inside the multi-account dictionary in session.
    - Flags if the account was already connected and sets it active.
    """
    state = request.session.get("oauth_state")
    code = request.query_params.get("code")

    if not code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing authorization code from Google.")

    try:
        client_config = get_client_config()
        redirect_uri = get_redirect_uri(request)

        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=SCOPES,
            state=state,
            redirect_uri=redirect_uri,
            autogenerate_code_verifier=False,
        )

        auth_response_url = str(request.url)
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

        accounts[email] = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
            "name": name,
            "picture": picture,
        }

        # Explicitly assign dictionary back to session
        request.session["accounts"] = accounts
        request.session["active_account"] = email

        if already_connected:
            request.session["account_notice"] = f"ℹ️ Account '{email}' was already connected and is now set as active."
        else:
            request.session["account_notice"] = f"✅ Successfully connected Google account: '{email}'."

        # Top-level sync
        request.session["user_email"] = email
        request.session["user_name"] = name
        request.session["user_picture"] = picture
        request.session["user_creds"] = accounts[email]
        request.session.pop("oauth_state", None)

        logger.info(f"Google Account '{email}' successfully merged into session (total accounts: {len(accounts)}) & set as active.")
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
    request.session["user_creds"] = acc_data
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
        now_dt = datetime.now(timezone.utc).astimezone()
        current_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
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
        ) -> str:
            """Create a new Google Calendar event with conflict checking asynchronously."""
            return await create_event_tool(
                user_creds,
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                description=description,
                location=location,
                attendees=attendees,
            )

        async def delete_calendar_event_wrapper(
            event_id: Optional[str] = None,
            summary: Optional[str] = None,
            time_min: Optional[str] = None,
        ) -> str:
            """Delete an event from user's primary Google Calendar by event_id or title."""
            return await delete_calendar_event_tool(user_creds, event_id=event_id, summary=summary, time_min=time_min)

        async def check_gmail_invites_wrapper(max_results: int = 10) -> str:
            """Scan unread Gmail messages for meeting, coffee, or tea invites in parallel."""
            return await check_gmail_invites_tool(user_creds, max_results=max_results)

        tools = [
            StructuredTool.from_function(
                coroutine=list_events_wrapper,
                name="list_events",
                description="List upcoming events from the user's primary Google Calendar. time_min is optional ISO string.",
            ),
            StructuredTool.from_function(
                coroutine=create_event_wrapper,
                name="create_event",
                description="Create a new event on Google Calendar with conflict checking. start_time and end_time must be ISO 8601 strings (e.g. 'YYYY-MM-DDTHH:MM:SS').",
            ),
            StructuredTool.from_function(
                coroutine=delete_calendar_event_wrapper,
                name="delete_calendar_event",
                description="Delete an existing event from Google Calendar using event_id, or search by summary/title and approximate time.",
            ),
            StructuredTool.from_function(
                coroutine=check_gmail_invites_wrapper,
                name="check_gmail_invites",
                description="Scan unread emails for tea, coffee, meetup, or meeting invitations in Gmail.",
            ),
        ]

        tool_map = {t.name: t for t in tools}

        # 3. System Instructions
        system_prompt = (
            "You are 'AI Calendar Agent', an expert, helpful, and friendly scheduling assistant.\n"
            f"Active Account: {active_email}\n"
            f"Current DateTime: {current_time_str} (ISO: {current_iso_str})\n"
            f"Default Timezone: {CALENDAR_TIMEZONE}\n\n"
            "Instructions:\n"
            "1. You have dynamic access to the active user's Google Calendar and Gmail tools.\n"
            "2. When the user asks about their schedule, events, or availability, call `list_events`.\n"
            "3. When the user asks to create/schedule an event (e.g., 'meeting with friends tomorrow at 3 PM'):\n"
            "   - Calculate start_time and end_time (defaulting to 1 hour duration if unspecified) relative to Current DateTime.\n"
            "   - Format start_time and end_time as ISO 8601 strings (e.g. 'YYYY-MM-DDTHH:MM:SS').\n"
            "   - Call `create_event` with summary, start_time, end_time, description, location, and attendees.\n"
            "   - Inform the user of successful creation with title, start/end time, location, clickable Google Calendar link, and include the event ID.\n"
            "4. When the user asks to delete, cancel, or remove an event (e.g., 'Delete my 3 PM meeting tomorrow' or 'Cancel event Team Sync'):\n"
            "   - Call `delete_calendar_event` with the event_id if known, or summary/title and approximate time.\n"
            "   - Confirm the deletion clearly with the event title.\n"
            "5. When listing or creating events, include a clean delete button format next to each event: `<button class=\"btn-delete-event\" data-event-id=\"EVENT_ID\"><i class=\"fa-solid fa-trash-can\"></i> Delete Event</button>` so the user can delete it with a single click.\n"
            "6. When the user asks about emails, tea, coffee, or meeting invitations, call `check_gmail_invites` and summarize relevant findings with clear options to schedule them.\n"
            "7. Format output using clean Markdown, bullet points, and nice styling. Include clickable links if available."
        )

        # 4. Initialize Gemini LLM with active Google AI models
        configured_model = (os.getenv("GEMINI_MODEL") or "gemini-3.6-flash").strip()
        raw_candidates = [
            configured_model,
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3.6-pro",
        ]
        legacy_deprecated = {"gemini-1.5-flash-latest", "models/gemini-1.5-flash-latest", "gemini-pro", "models/gemini-pro"}
        unique_candidates = []
        for cand in raw_candidates:
            cand_clean = cand.removeprefix("models/").strip()
            if cand_clean and cand_clean not in legacy_deprecated and cand_clean not in unique_candidates:
                unique_candidates.append(cand_clean)

        # 5. Multi-turn Agent Execution Loop with Model Fallback
        max_iterations = 6
        final_text = ""
        last_error = None

        for model_name in unique_candidates:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=clean_api_key,
                    temperature=0.2,
                )
                llm_with_tools = llm.bind_tools(tools)

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]

                for _ in range(max_iterations):
                    response = await llm_with_tools.ainvoke(messages)
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

            except Exception as candidate_err:
                logger.warning(f"Model '{model_name}' invocation failed: {candidate_err}. Attempting fallback...")
                last_error = candidate_err
                continue

        if not final_text:
            if last_error:
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
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"An error occurred while processing your request: {str(e)}"},
        )


# ---------------------------------------------------------------------------
# 10. Frontend UI Route (Gated Jinja2 Template Rendering)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """
    Renders the ChatGPT-style modern web interface for authenticated users.
    Redirects unauthenticated visitors to /login.
    """
    active_email = get_active_account_email(request)
    if not active_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

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
    logger.info(f"Starting AI Calendar Agent on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)