import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from parser import parse_schedule_message
from calendar_service import (
    create_google_calendar_event,
    check_calendar_conflict,
    find_event_by_title,
    delete_google_calendar_event,
    reschedule_google_calendar_event,
    get_primary_calendar_timezone,
    get_calendar_service,
    sync_calendar_timezone,
    find_gmail_drive_or_internship_messages,
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

pending_gmail_events = {}


def format_display_time(start_iso: str, end_iso: str, timezone_name: str) -> str:
    """Format start and end ISO strings into user-friendly localized date and time."""
    try:
        tz = ZoneInfo(timezone_name)
        start_clean = str(start_iso).strip().replace("Z", "+00:00")
        end_clean = str(end_iso).strip().replace("Z", "+00:00") if end_iso else ""

        start_dt = datetime.fromisoformat(start_clean)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=tz)
        else:
            start_dt = start_dt.astimezone(tz)

        date_str = start_dt.strftime("%A, %b %d, %Y")
        start_time_str = start_dt.strftime("%I:%M %p").lstrip("0")

        if end_clean:
            end_dt = datetime.fromisoformat(end_clean)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=tz)
            else:
                end_dt = end_dt.astimezone(tz)
            end_time_str = end_dt.strftime("%I:%M %p").lstrip("0")
            return f"📅 {date_str}\n⏰ {start_time_str} – {end_time_str} ({timezone_name})"

        return f"📅 {date_str}\n⏰ {start_time_str} ({timezone_name})"
    except Exception:
        return f"⏰ {start_iso} to {end_iso} ({timezone_name})"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your AI Calendar Assistant.\n\n"
        "I can help you with:\n"
        "👉 **Creating:** 'Meeting with Dilip at 2pm'\n"
        "👉 **Rescheduling:** 'Move meeting with Dilip to 4pm'\n"
        "👉 **Deleting:** 'Cancel meeting with Dilip'\n"
        "👉 **Listing:** 'What is on my schedule today?'\n"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.reply_text("⏳ Processing your request...")

    try:
        chat_id = update.effective_chat.id
        calendar_service = get_calendar_service()
        calendar_timezone = get_primary_calendar_timezone(calendar_service)
        
        normalized_text = user_text.strip().lower()
        if normalized_text in {"approve", "approved", "yes", "add it", "add to calendar"} or normalized_text.startswith("approve "):
            pending_events = pending_gmail_events.pop(chat_id, None)
            if not pending_events:
                messages = find_gmail_drive_or_internship_messages()
                pending_events = []
                for message in messages:
                    email_text = f"{message['subject']}\n{message['snippet']}"
                    pending_events.extend(parse_schedule_message(email_text, calendar_timezone).events)
            if not pending_events:
                await update.message.reply_text("There is no unread Gmail event waiting for approval.")
                return

            links = [create_google_calendar_event(event) for event in pending_events]
            await update.message.reply_text(
                "✅ Added the approved Gmail event(s) to Google Calendar.\n"
                + "\n".join(link for link in links if link)
            )
            return

        if any(word in normalized_text for word in ["gmail", "email", "drive", "internship"]):
            messages = find_gmail_drive_or_internship_messages()
            if not messages:
                await update.message.reply_text("No unread Drive or internship messages found in Gmail.")
                return

            pending_gmail_events[chat_id] = []
            lines = ["📧 Gmail messages found (Drive/internship):"]
            for message in messages:
                email_text = f"{message['subject']}\n{message['snippet']}"
                parsed_email = parse_schedule_message(email_text, calendar_timezone)
                pending_gmail_events[chat_id].extend(parsed_email.events)
                lines.append(
                    f"\nSubject: {message['subject']}\n"
                    f"From: {message['from']}\n"
                    f"Message: {message['display_snippet']}"
                )
            lines.append("\nReply 'approve' to add these detected event(s) to your calendar.")
            await update.message.reply_text("\n".join(lines))
            return

        parsed_data = parse_schedule_message(user_text, calendar_timezone)
        reply_lines = []

        for event in parsed_data.events:
            action = getattr(event, 'action', 'CREATE').upper()

            # --- DELETE ACTION ---
            if action == "DELETE":
                target_event = find_event_by_title(event.event_name)
                if target_event and delete_google_calendar_event(target_event['id']):
                    reply_lines.append(f"🗑️ **Deleted Event:** '{target_event.get('summary', event.event_name)}'")
                else:
                    reply_lines.append(f"❌ Could not find event matching '{event.event_name}' to delete.")

            # --- RESCHEDULE ACTION ---
            elif action == "RESCHEDULE":
                target_event = find_event_by_title(event.event_name)
                if target_event:
                    link = reschedule_google_calendar_event(target_event['id'], event.start_time, event.end_time)
                    time_display = format_display_time(event.start_time, event.end_time, calendar_timezone)
                    reply_lines.append(
                        f"🔄 **Rescheduled Event:** '{target_event.get('summary')}'\n"
                        f"{time_display}\n"
                        f"🔗 [View in Google Calendar]({link})\n"
                    )
                else:
                    reply_lines.append(f"❌ Could not find event matching '{event.event_name}' to reschedule.")

            # --- CREATE ACTION ---
            else:
                conflicts = check_calendar_conflict(event.start_time, event.end_time)
                if conflicts:
                    conflict_names = ", ".join([f"'{c}'" for c in conflicts])
                    reply_lines.append(f"⚠️ **Conflict Detected:** Overlaps with {conflict_names}.\n")

                link = create_google_calendar_event(event)
                time_display = format_display_time(event.start_time, event.end_time, calendar_timezone)
                reply_lines.append(
                    f"✅ **Created Event:** {event.event_name}\n"
                    f"{time_display}\n"
                    f"🔗 [View in Google Calendar]({link})\n"
                )

        await update.message.reply_markdown('\n'.join(reply_lines))

    except Exception as e:
        await update.message.reply_text(f"❌ Failed to process request: {str(e)}")


if __name__ == '__main__':
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env file!")
        exit(1)

    try:
        svc = get_calendar_service()
        sync_calendar_timezone(svc)
    except Exception as exc:
        print(f"Warning: Could not sync calendar timezone: {exc}")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("🤖 Telegram Bot is running! Press Ctrl+C to stop.")
    app.run_polling()