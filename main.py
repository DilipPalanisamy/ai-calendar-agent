import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from parser import parse_schedule_message
from calendar_service import (
    create_google_calendar_event,
    check_calendar_conflict,
    find_event_by_title,
    delete_google_calendar_event,
    reschedule_google_calendar_event,
    list_google_calendar_events,
)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize Telegram Application
bot_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build() if TELEGRAM_TOKEN else None


# --- Telegram Command Handlers ---
async def start_command(update: Update, context):
    await update.message.reply_text(
        "👋 Hi! I'm your AI Calendar Assistant via FastAPI.\n\n"
        "I can help you with:\n"
        "👉 **Create:** 'Team sync tomorrow at 3pm'\n"
        "👉 **List:** 'What is my schedule for tomorrow?'\n"
        "👉 **Reschedule:** 'Move team sync tomorrow to 5pm'\n"
        "👉 **Delete:** 'Cancel team sync tomorrow'\n"
    )


async def telegram_message_handler(update: Update, context):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    await update.message.reply_text("⏳ Processing calendar request...")

    try:
        parsed_data = parse_schedule_message(user_text)
        reply_lines = []

        for event in parsed_data.events:
            action = getattr(event, 'action', 'CREATE').upper()

            # --- LIST ACTION ---
            if action == "LIST":
                events = list_google_calendar_events(event.start_time, event.end_time)
                if events:
                    reply_lines.append("📅 **Your Schedule:**\n")
                    for ev in events:
                        reply_lines.append(f"• **{ev['summary']}** ({ev['start']} to {ev['end']})")
                else:
                    reply_lines.append("📋 No events found for this time period.")

            # --- DELETE ACTION ---
            elif action == "DELETE":
                target = find_event_by_title(event.event_name, time_min_iso=event.start_time)
                if target and delete_google_calendar_event(target['id']):
                    reply_lines.append(f"🗑️ **Deleted:** '{target.get('summary')}'")
                else:
                    reply_lines.append(f"❌ Could not find event matching '{event.event_name}' to delete.")

            # --- RESCHEDULE ACTION ---
            elif action == "RESCHEDULE":
                target = find_event_by_title(event.event_name, time_min_iso=event.start_time)
                if target:
                    link = reschedule_google_calendar_event(target['id'], event.start_time, event.end_time)
                    reply_lines.append(
                        f"🔄 **Rescheduled:** '{target.get('summary')}'\n"
                        f"⏰ {event.start_time} - {event.end_time}\n"
                        f"🔗 [View in Google Calendar]({link})\n"
                    )
                else:
                    reply_lines.append(f"❌ Could not find event matching '{event.event_name}' to reschedule.")

            # --- CREATE ACTION ---
            else:
                conflicts = check_calendar_conflict(event.start_time, event.end_time)
                if conflicts:
                    reply_lines.append(f"⚠️ **Conflict Detected:** Overlaps with {', '.join(conflicts)}.\n")

                link = create_google_calendar_event(event)
                reply_lines.append(
                    f"✅ **Created:** {event.event_name}\n"
                    f"⏰ {event.start_time} - {event.end_time}\n"
                    f"🔗 [View in Google Calendar]({link})\n"
                )

        await update.message.reply_markdown('\n'.join(reply_lines))

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")


# --- Register Telegram Handlers ---
if bot_app:
    bot_app.add_handler(CommandHandler("start", start_command))
    bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_message_handler))


# --- FastAPI Lifespan (Starts & Stops Telegram Bot) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if bot_app:
        await bot_app.initialize()
        await bot_app.start()
        await bot_app.updater.start_polling()  # <--- Added start_polling
        print("🤖 Telegram Bot Polling Started!")
    yield
    if bot_app:
        await bot_app.updater.stop()           # <--- Added updater stop
        await bot_app.stop()
        await bot_app.shutdown()


app = FastAPI(lifespan=lifespan)


# --- Endpoint 1: Standard API Webhook (for PowerShell, Slack, or web apps) ---
class WebhookPayload(BaseModel):
    source: str
    sender_id: str
    message: str
    timezone: Optional[str] = "Asia/Kolkata"


@app.post("/webhook/json")
async def handle_json_webhook(payload: WebhookPayload):
    try:
        parsed_data = parse_schedule_message(payload.message, payload.timezone)
        results = []

        for event in parsed_data.events:
            action = getattr(event, "action", "CREATE").upper()

            if action == "LIST":
                events = list_google_calendar_events(event.start_time, event.end_time)
                results.append({
                    "action": "LIST",
                    "total_events": len(events),
                    "events": events
                })

            elif action == "DELETE":
                target = find_event_by_title(event.event_name, time_min_iso=event.start_time)
                deleted = delete_google_calendar_event(target['id']) if target else False
                results.append({
                    "action": "DELETE",
                    "event_name": event.event_name,
                    "status": "success" if deleted else "not_found"
                })

            elif action == "RESCHEDULE":
                target = find_event_by_title(event.event_name, time_min_iso=event.start_time)
                if target:
                    link = reschedule_google_calendar_event(target['id'], event.start_time, event.end_time)
                    results.append({
                        "action": "RESCHEDULE",
                        "event_name": target.get('summary'),
                        "start_time": event.start_time,
                        "end_time": event.end_time,
                        "google_calendar_link": link
                    })
                else:
                    results.append({
                        "action": "RESCHEDULE",
                        "event_name": event.event_name,
                        "status": "not_found"
                    })

            else:
                link = create_google_calendar_event(event)
                results.append({
                    "action": "CREATE",
                    "event_name": event.event_name,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "google_calendar_link": link,
                })

        return {
            "status": "success",
            "raw_message": payload.message,
            "total_operations": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Endpoint 2: Telegram Webhook (Processes updates forwarded from Telegram) ---
@app.post("/webhook/telegram")
async def handle_telegram_webhook(request: Request):
    if not bot_app:
        raise HTTPException(status_code=500, detail="Telegram bot token not configured")

    req_json = await request.json()
    update = Update.de_json(req_json, bot_app.bot)
    await bot_app.process_update(update)
    return {"status": "ok"}