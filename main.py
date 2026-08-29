import os
import json
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

# Jinja2 Templates setup (points reliably to templates directory)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR) if TEMPLATES_DIR.exists() else "templates")

# Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", "ai-calendar-agent-secret-key-production-ready-2026")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
CALENDAR_TIMEZONE = os.getenv("CALENDAR_TIMEZONE", "UTC")

# Google OAuth 2.0 Scopes required for Calendar & Gmail
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]

# ---------------------------------------------------------------------------
# 2. FastAPI Application Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Calendar Agent",
    description="Multi-user AI Calendar & Gmail Assistant powered by Gemini & FastAPI",
    version="2.0.0",
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
# 3. Google OAuth 2.0 Helper Functions
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


def get_user_credentials(request: Request) -> Optional[Credentials]:
    """
    Dynamically deserialize and refresh the active user's Google OAuth Credentials
    from the current session.
    """
    creds_data = request.session.get("user_creds")
    if not creds_data:
        return None

    try:
        creds = Credentials(
            token=creds_data.get("token"),
            refresh_token=creds_data.get("refresh_token"),
            token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=creds_data.get("client_id"),
            client_secret=creds_data.get("client_secret"),
            scopes=creds_data.get("scopes", SCOPES),
        )

        # Refresh token automatically if expired
        if creds.expired and creds.refresh_token:
            logger.info("Access token expired. Refreshing token via Google OAuth...")
            creds.refresh(GoogleRequest())
            request.session["user_creds"] = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }

        return creds
    except Exception as e:
        logger.error(f"Error loading user credentials: {e}")
        return None


# ---------------------------------------------------------------------------
# 4. Multi-User Google Tool Functions
# ---------------------------------------------------------------------------
def list_events_tool(creds: Credentials, time_min: Optional[str] = None, max_results: int = 15) -> str:
    """Lists upcoming events from the user's primary Google Calendar."""
    try:
        service = build("calendar", "v3", credentials=creds)
        if not time_min:
            time_min = datetime.now(timezone.utc).isoformat()

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
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


def create_event_tool(
    creds: Credentials,
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    attendees: Optional[List[str]] = None,
) -> str:
    """Creates a new event on user's primary Google Calendar after conflict checking."""
    try:
        service = build("calendar", "v3", credentials=creds)

        # Conflict checking in the target time window
        conflict_msg = ""
        try:
            overlap_check = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_time,
                    timeMax=end_time,
                    singleEvents=True,
                )
                .execute()
            )
            conflicts = overlap_check.get("items", [])
            if conflicts:
                conflict_names = [f"'{c.get('summary', 'Untitled')}'" for c in conflicts]
                conflict_msg = f"⚠️ Conflict Notice: You already have event(s) scheduled at this time: {', '.join(conflict_names)}."
        except Exception as check_err:
            logger.warning(f"Conflict check warning: {check_err}")

        event_body: Dict[str, Any] = {
            "summary": summary,
            "description": description or "Scheduled via AI Calendar Agent",
            "start": {"dateTime": start_time},
            "end": {"dateTime": end_time},
        }

        if location:
            event_body["location"] = location

        if attendees:
            event_body["attendees"] = [{"email": email.strip()} for email in attendees if email.strip()]

        created_event = service.events().insert(calendarId="primary", body=event_body).execute()

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


def check_gmail_invites_tool(creds: Credentials, max_results: int = 10) -> str:
    """Searches unread emails for meeting, tea, coffee, or sync invitations."""
    try:
        service = build("gmail", "v1", credentials=creds)
        query = "is:unread (tea OR coffee OR meetup OR meeting OR sync OR invite)"

        response = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        messages = response.get("messages", [])

        if not messages:
            return "No unread invitation emails (tea, coffee, meetup, or meetings) found in your Gmail inbox."

        invitations = []
        for msg in messages:
            msg_id = msg["id"]
            msg_data = service.users().messages().get(
                userId="me", id=msg_id, format="metadata",
                metadataHeaders=["Subject", "From", "Date"]
            ).execute()

            headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "(No Subject)")
            sender = headers.get("From", "Unknown Sender")
            date_str = headers.get("Date", "")
            snippet = msg_data.get("snippet", "")

            invitations.append({
                "message_id": msg_id,
                "subject": subject,
                "sender": sender,
                "date": date_str,
                "snippet": snippet,
            })

        return json.dumps(invitations, indent=2)

    except HttpError as err:
        logger.error(f"Gmail API Error: {err}")
        return f"Error querying Gmail: {err}"
    except Exception as e:
        logger.error(f"Unexpected error in check_gmail_invites: {e}")
        return f"Failed to search Gmail invites: {str(e)}"


# ---------------------------------------------------------------------------
# 5. Google OAuth 2.0 Web Endpoints
# ---------------------------------------------------------------------------
@app.get("/login", response_class=RedirectResponse)
async def login(request: Request):
    """Initiates Google OAuth 2.0 consent flow."""
    try:
        client_config = get_client_config()
        redirect_uri = get_redirect_uri(request)

        # Explicitly disable PKCE verification for standard server-side OAuth
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


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    """Handles OAuth callback, exchanges code for credentials, and stores in session."""
    state = request.session.get("oauth_state")
    code = request.query_params.get("code")

    if not code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing authorization code from Google.")

    try:
        client_config = get_client_config()
        redirect_uri = get_redirect_uri(request)

        # Explicitly disable PKCE verification
        flow = Flow.from_client_config(
            client_config=client_config,
            scopes=SCOPES,
            state=state,
            redirect_uri=redirect_uri,
            autogenerate_code_verifier=False,
        )

        # Fix scheme mismatch when behind HTTPS reverse proxies (Render / Cloudflare)
        auth_response_url = str(request.url)
        if redirect_uri.startswith("https://") and auth_response_url.startswith("http://"):
            auth_response_url = auth_response_url.replace("http://", "https://", 1)

        # Exchange authorization code for tokens
        flow.fetch_token(authorization_response=auth_response_url)
        credentials = flow.credentials

        # Fetch authenticated user profile details
        userinfo_service = build("oauth2", "v2", credentials=credentials)
        user_info = userinfo_service.userinfo().get().execute()

        email = user_info.get("email", "")
        name = user_info.get("name", email)
        picture = user_info.get("picture", "")

        # Store credentials and user profile in session
        request.session["user_email"] = email
        request.session["user_name"] = name
        request.session["user_picture"] = picture
        request.session["user_creds"] = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
        }

        # Clear transient OAuth state
        request.session.pop("oauth_state", None)

        logger.info(f"User '{email}' successfully signed in via Google OAuth.")
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        logger.error(f"OAuth token exchange failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to authenticate with Google: {str(e)}",
        )


@app.get("/logout")
async def logout(request: Request):
    """Clears user session and logs out."""
    user_email = request.session.get("user_email")
    request.session.clear()
    if user_email:
        logger.info(f"User '{user_email}' logged out.")
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/api/me")
async def get_current_user(request: Request):
    """Returns current user profile and authentication status."""
    user_creds = request.session.get("user_creds")
    if not user_creds:
        return JSONResponse({"authenticated": False})

    return JSONResponse({
        "authenticated": True,
        "email": request.session.get("user_email", ""),
        "name": request.session.get("user_name", "User"),
        "picture": request.session.get("user_picture", ""),
    })


# ---------------------------------------------------------------------------
# 6. AI Agent Runner & LangChain Integration
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    AI Chatbot endpoint:
    - Validates user authentication.
    - Dynamically binds user Google Calendar and Gmail tools.
    - Runs Gemini 2.5 Flash with tool calling support.
    """
    user_creds = get_user_credentials(request)
    if not user_creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized. Please sign in with your Google Account first.",
        )

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    try:
        user_email = request.session.get("user_email", "User")
        now_dt = datetime.now(timezone.utc).astimezone()
        current_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        current_iso_str = now_dt.isoformat()

        # 1. Dynamically wrap tools with active user credentials
        def list_events_wrapper(time_min: Optional[str] = None, max_results: int = 10) -> str:
            """Query upcoming Google Calendar events."""
            return list_events_tool(user_creds, time_min=time_min, max_results=max_results)

        def create_event_wrapper(
            summary: str,
            start_time: str,
            end_time: str,
            description: str = "",
            location: str = "",
            attendees: Optional[List[str]] = None,
        ) -> str:
            """Create a new Google Calendar event with conflict checking."""
            return create_event_tool(
                user_creds,
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                description=description,
                location=location,
                attendees=attendees,
            )

        def check_gmail_invites_wrapper(max_results: int = 10) -> str:
            """Scan unread Gmail messages for meeting, coffee, or tea invites."""
            return check_gmail_invites_tool(user_creds, max_results=max_results)

        tools = [
            StructuredTool.from_function(
                func=list_events_wrapper,
                name="list_events",
                description="List upcoming events from the user's primary Google Calendar. time_min is optional ISO string.",
            ),
            StructuredTool.from_function(
                func=create_event_wrapper,
                name="create_event",
                description="Create a new event on Google Calendar with conflict checking. start_time and end_time must be ISO 8601 format.",
            ),
            StructuredTool.from_function(
                func=check_gmail_invites_wrapper,
                name="check_gmail_invites",
                description="Scan unread emails for tea, coffee, meetup, or meeting invitations in Gmail.",
            ),
        ]

        tool_map = {t.name: t for t in tools}

        # 2. System Instructions
        system_prompt = (
            "You are 'AI Calendar Agent', an expert, helpful, and friendly scheduling assistant.\n"
            f"Active User: {user_email}\n"
            f"Current DateTime: {current_time_str} (ISO: {current_iso_str})\n"
            f"Default Timezone: {CALENDAR_TIMEZONE}\n\n"
            "Instructions:\n"
            "1. You have dynamic access to the user's Google Calendar and Gmail tools.\n"
            "2. When the user asks about their schedule, events, or availability, call `list_events`.\n"
            "3. When creating an event, parse dates and times relative to Current DateTime, calculate proper ISO start and end times, and call `create_event`. Always inform the user if there are any conflicting events.\n"
            "4. When the user asks about emails, tea, coffee, or meeting invitations, call `check_gmail_invites` and summarize relevant findings with clear options to schedule them.\n"
            "5. Format output using clean Markdown, bullet points, and nice styling. Include clickable links if available."
        )

        # 3. Initialize Gemini LLM with bound tools
        configured_model = (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()
        if configured_model in {"gemini-2.5-flash", "models/gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.6-flash"}:
            configured_model = "gemini-1.5-flash"

        model_candidates = [
            configured_model,
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ]
        seen = set()
        unique_candidates = [m for m in model_candidates if m and not (m in seen or seen.add(m))]

        # 4. Multi-turn Agent Execution Loop with Model Fallback
        max_iterations = 6
        final_text = ""
        last_error = None

        for model_name in unique_candidates:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=GEMINI_API_KEY,
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
                        final_text = response.content
                        break

                    # Execute tool calls
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call.get("id", tool_name)

                        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

                        if tool_name in tool_map:
                            try:
                                tool_func = tool_map[tool_name]
                                tool_result = tool_func.invoke(tool_args)
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

        return JSONResponse({"response": final_text})

    except Exception as e:
        logger.error(f"Chat execution failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"An error occurred while processing your request: {str(e)}"},
        )


# ---------------------------------------------------------------------------
# 7. Frontend UI Route (Jinja2 Template Rendering)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Renders the ChatGPT-style modern web interface using Jinja2Templates."""
    user_email = request.session.get("user_email", "")
    user_name = request.session.get("user_name", "")
    user_picture = request.session.get("user_picture", "")
    authenticated = bool(request.session.get("user_creds"))

    context = {
        "request": request,
        "user_email": user_email,
        "user_name": user_name,
        "user_picture": user_picture,
        "authenticated": authenticated,
    }

    try:
        return templates.TemplateResponse(request=request, name="index.html", context=context)
    except Exception:
        return templates.TemplateResponse("index.html", context)


# ---------------------------------------------------------------------------
# 8. Server Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting AI Calendar Agent on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)